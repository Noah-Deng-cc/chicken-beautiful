"""同济智能体适配器：输入为 AgentSettings/查询和可注入传输，输出为 AgentReply；依赖 requests、标准库和核心契约。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import time
from typing import cast

from src.core import AgentSettings
from src.domain import AgentReply
from .base import (AgentClient, AgentConfigurationError, AgentResponseError,
                   AgentTransportError, HttpResponse, HttpTransport, RequestsTransport)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TongjiContract:
    """T27 核验前可替换的暂定 HTTP 字段映射。"""

    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    query_field: str = "query"
    context_field: str = "context"
    conversation_id_field: str = "conversation_id"
    agent_id_field: str | None = "agent_id"
    stream_field: str | None = "stream"
    response_text_path: tuple[str, ...] = ("data", "answer")
    response_conversation_id_path: tuple[str, ...] | None = ("data", "conversation_id")


def _utc_now() -> datetime:
    """生成带 UTC 时区的当前时间。

    Args: 无。
    Returns: 带时区的当前时间。
    Raises: 无。
    """
    return datetime.now(timezone.utc)


def _extract(body: Mapping[str, object], path: tuple[str, ...], label: str) -> object:
    """按字段路径提取响应值且不回显响应正文。

    Args: body: JSON 对象。path: 字段路径。label: 安全错误标签。
    Returns: 路径对应值。
    Raises: AgentResponseError: 字段缺失或中间节点不是对象。
    """
    current: object = body
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            raise AgentResponseError(f"agent response is missing field '{label}'")
        current = current[part]
    return current


class TongjiAgentClient(AgentClient):
    """可配置、可重试且不记录敏感请求内容的智能体客户端。"""

    def __init__(self, settings: AgentSettings, *, transport: HttpTransport | None = None,
                 contract: TongjiContract | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], datetime] = _utc_now, max_backoff_seconds: float = 30.0) -> None:
        """保存配置并延迟初始化真实 HTTP 会话。

        Args: settings: 已注入秘密的配置。transport: 可选 mock/真实传输。contract: 字段映射。
            sleep: 可注入等待函数。clock: 可注入时钟。max_backoff_seconds: 单次退避上限。
        Returns: 无。
        Raises: AgentConfigurationError: 密钥、地址、超时、重试或字段映射无效。
        """
        selected = contract if contract is not None else TongjiContract()
        required = (selected.auth_header, selected.query_field, selected.context_field,
                    selected.conversation_id_field, *selected.response_text_path)
        request_fields = tuple(value for value in (selected.query_field, selected.context_field,
                               selected.conversation_id_field, selected.agent_id_field,
                               selected.stream_field) if value is not None)
        if (not settings.api_key or not settings.base_url.strip() or not settings.endpoint.strip()
                or settings.connect_timeout_seconds <= 0 or settings.read_timeout_seconds <= 0
                or settings.max_retries < 0 or settings.backoff_seconds < 0 or max_backoff_seconds < 0
                or not selected.response_text_path or any(not value.strip() for value in required)
                or len(request_fields) != len(set(request_fields))):
            raise AgentConfigurationError("invalid agent client configuration")
        if settings.stream:
            raise AgentConfigurationError("streaming response contract has not been verified")
        if selected.agent_id_field is not None and not settings.agent_id:
            raise AgentConfigurationError("agent identifier is required by the request contract")
        self._settings = settings
        self._transport = transport if transport is not None else RequestsTransport()
        self._contract, self._sleep, self._clock = selected, sleep, clock
        self._max_backoff_seconds = max_backoff_seconds
        self._url = f"{settings.base_url.rstrip('/')}/{settings.endpoint.lstrip('/')}"

    def reply(self, query: str, context: Mapping[str, object],
              conversation_id: str | None) -> AgentReply:
        """调用非流式会话接口并解析回复。

        Args: query: 非空用户文本。context: JSON 兼容结构化上下文。conversation_id: 可选已有会话。
        Returns: 含 UTC 时间和服务端会话标识的回复。
        Raises: ValueError: 文本为空。TypeError: 文本或上下文类型错误。AgentError: 传输、状态或响应错误。
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        if conversation_id is not None and not isinstance(conversation_id, str):
            raise TypeError("conversation_id must be a string or None")
        if isinstance(conversation_id, str) and not conversation_id.strip():
            raise ValueError("conversation_id must be a non-empty string or None")
        if not isinstance(context, Mapping) or not all(isinstance(key, str) for key in context):
            raise TypeError("context must be a mapping with string keys")
        payload = self._payload(query, context, conversation_id)
        response = self._send(payload)
        return self._parse(response, conversation_id)

    def _payload(self, query: str, context: Mapping[str, object],
                 conversation_id: str | None) -> dict[str, object]:
        """按可配置字段构建并校验 JSON 请求体。

        Args: query: 用户文本。context: 结构化上下文。conversation_id: 可选会话标识。
        Returns: JSON 兼容请求字典。
        Raises: AgentConfigurationError: 请求内容不能安全序列化为 JSON。
        """
        contract = self._contract
        payload: dict[str, object] = {contract.query_field: query, contract.context_field: dict(context)}
        if conversation_id is not None:
            payload[contract.conversation_id_field] = conversation_id
        if contract.agent_id_field is not None:
            payload[contract.agent_id_field] = cast(str, self._settings.agent_id)
        if contract.stream_field is not None:
            payload[contract.stream_field] = False
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError):
            raise AgentConfigurationError("agent request payload is not JSON serializable") from None
        return payload

    def _send(self, payload: Mapping[str, object]) -> HttpResponse:
        """发送请求并对超时、429 和 5xx 执行有界指数退避。

        Args: payload: 已校验 JSON 请求体。
        Returns: 成功的 2xx 或无需重试的响应。
        Raises: AgentTransportError: 传输失败或可重试状态耗尽。
        """
        settings, contract = self._settings, self._contract
        credential = f"{contract.auth_scheme} {settings.api_key}".strip()
        headers = {"Content-Type": "application/json", contract.auth_header: credential}
        for attempt in range(settings.max_retries + 1):
            try:
                response = self._transport.post(
                    self._url, headers=headers, json_body=payload,
                    timeout=(settings.connect_timeout_seconds, settings.read_timeout_seconds))
            except Exception:
                if attempt >= settings.max_retries:
                    raise AgentTransportError("agent transport failed after retries") from None
                self._backoff(attempt, "transport")
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= settings.max_retries:
                    raise AgentTransportError(f"agent service unavailable after HTTP {response.status_code}")
                self._backoff(attempt, f"HTTP {response.status_code}")
                continue
            return response
        raise AgentTransportError("agent request retry loop ended unexpectedly")

    def _backoff(self, attempt: int, reason: str) -> None:
        """记录安全状态并执行指数退避。

        Args: attempt: 从零开始的失败次数。reason: 不含请求内容的安全原因。
        Returns: 无。
        Raises: 无。
        """
        delay = min(self._settings.backoff_seconds * (2 ** attempt), self._max_backoff_seconds)
        LOGGER.warning("agent request retry reason=%s attempt=%d", reason, attempt + 1)
        self._sleep(delay)

    def _parse(self, response: HttpResponse, previous_id: str | None) -> AgentReply:
        """校验状态与 JSON 字段并构造领域回复。

        Args: response: HTTP 响应。previous_id: 请求携带的会话标识。
        Returns: 已校验 AgentReply。
        Raises: AgentResponseError: 状态、JSON 或字段类型无效。
        """
        if not 200 <= response.status_code < 300:
            raise AgentResponseError(f"agent request rejected with HTTP {response.status_code}")
        if not isinstance(response.body, Mapping):
            raise AgentResponseError("agent response is not a JSON object")
        text = _extract(response.body, self._contract.response_text_path, "reply text")
        if not isinstance(text, str) or not text.strip():
            raise AgentResponseError("agent response field 'reply text' must be a non-empty string")
        path = self._contract.response_conversation_id_path
        value = previous_id if path is None else _extract(response.body, path, "conversation id")
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise AgentResponseError("agent response field 'conversation id' must be a string or null")
        return AgentReply(text.strip(), self._clock(), cast(str | None, value))
