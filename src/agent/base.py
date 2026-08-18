"""智能体基础契约：输入为查询、上下文和 HTTP 请求，输出为 AgentReply/HttpResponse；依赖 requests 和领域模型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import requests

from src.domain import AgentReply


class AgentError(RuntimeError):
    """智能体客户端错误的安全基类。"""


class AgentConfigurationError(AgentError):
    """表示本地智能体配置无效。"""


class AgentTransportError(AgentError):
    """表示网络超时、连接失败或远端暂时不可用。"""


class AgentResponseError(AgentError):
    """表示 HTTP 状态或响应字段不符合约定。"""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """与 HTTP 库无关的最小响应。"""

    status_code: int
    body: object = field(repr=False)
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)


class HttpTransport(Protocol):
    """可由真实 requests 或内存 mock 实现的传输协议。"""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, object],
        timeout: tuple[float, float],
    ) -> HttpResponse:
        """发送 JSON POST 请求。

        Args:
            url: 完整请求地址。
            headers: 请求头，可能包含秘密且不得记录。
            json_body: JSON 兼容请求体。
            timeout: 连接和读取超时秒数。
        Returns:
            与具体 HTTP 库解耦的响应。
        Raises:
            AgentTransportError: 请求超时或传输失败。
        """
        ...


class RequestsTransport:
    """首次请求时才创建 requests.Session 的真实传输。"""

    def __init__(self) -> None:
        """初始化未连接的传输。

        Args: 无。
        Returns: 无。
        Raises: 无。
        """
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        """延迟创建并返回 HTTP 会话。

        Args: 无。
        Returns: 可复用的 requests 会话。
        Raises: 无。
        """
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def post(self, url: str, *, headers: Mapping[str, str],
             json_body: Mapping[str, object], timeout: tuple[float, float]) -> HttpResponse:
        """发送请求并转换为安全的最小响应。

        Args: url: 完整地址。headers: 请求头。json_body: JSON 请求体。timeout: 连接/读取超时。
        Returns: 状态码、可选 JSON 与响应头。
        Raises: AgentTransportError: 请求超时或网络失败，错误文本不包含请求秘密。
        """
        try:
            response = self._get_session().post(
                url, headers=dict(headers), json=dict(json_body), timeout=timeout)
        except requests.Timeout:
            raise AgentTransportError("agent request timed out") from None
        except requests.RequestException:
            raise AgentTransportError("agent transport failed") from None
        try:
            body: object = response.json()
        except requests.exceptions.JSONDecodeError:
            body = None
        return HttpResponse(response.status_code, body, dict(response.headers))


class AgentClient(ABC):
    """会话式外部智能体抽象接口。"""

    @abstractmethod
    def reply(
        self,
        query: str,
        context: Mapping[str, object],
        conversation_id: str | None,
    ) -> AgentReply:
        """发送一次带上下文的会话查询。

        Args:
            query: 用户查询文本。
            context: 情绪、传感器和日程等结构化上下文。
            conversation_id: 已有会话标识，首次调用为 None。
        Returns:
            带服务端会话标识的智能体回复。
        Raises:
            AgentError: 配置、传输、HTTP 状态或响应契约错误。
            ValueError: 查询或会话标识为空。
            TypeError: 上下文键不是字符串。
        """
        raise NotImplementedError
