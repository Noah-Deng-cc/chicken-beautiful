"""同济 MCP stream-only agent client."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import json
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from src.core import AgentSettings
from src.domain import AgentReply
from .base import AgentClient, AgentConfigurationError, AgentResponseError, AgentTransportError


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TongjiMcpAgentClient(AgentClient):
    """调用同济 MCP ``chicken-beauty`` 工具并提取文本回复。"""

    def __init__(self, settings: AgentSettings, *, session: Any | None = None) -> None:
        if (not settings.api_key or not settings.base_url.strip() or not settings.endpoint.strip()
                or settings.connect_timeout_seconds <= 0 or settings.read_timeout_seconds <= 0
                or settings.max_retries < 0):
            raise AgentConfigurationError("invalid MCP agent client configuration")
        self._settings = settings
        self._session = session or requests.Session()
        endpoint = settings.endpoint.strip()
        raw_url = settings.base_url.rstrip("/") if endpoint in ("", "/") else \
            f"{settings.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        self._url = self._with_api_key(raw_url)
        self._session_id: str | None = None
        self._next_id = 1
        self._tool_checked = False

    def _with_api_key(self, url: str) -> str:
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["api_key"] = self._settings.api_key or ""
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

    def reply(self, query: str, context: Mapping[str, object], conversation_id: str | None) -> AgentReply:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(context, Mapping) or not all(isinstance(k, str) for k in context):
            raise TypeError("context must be a mapping with string keys")
        if self._session_id is None:
            self._initialize()
        if not self._tool_checked:
            self._check_tool()
        arguments: dict[str, object] = {"Query": self._query_with_context(query, context)}
        files = context.get("files")
        if isinstance(files, list) and all(isinstance(item, Mapping) for item in files):
            arguments["Files"] = [dict(item) for item in files]
        response = self._rpc("tools/call", {"name": "chicken-beauty", "arguments": arguments})
        text = self._text_from_response(response)
        return AgentReply(text, _now(), self._session_id)

    @staticmethod
    def _query_with_context(query: str, context: Mapping[str, object]) -> str:
        emotion = context.get("emotion")
        if isinstance(emotion, Mapping):
            emotion = emotion.get("dominant", "unknown")
        elif emotion is None:
            emotion = context.get("face_emotion", "unknown")
        return f"face_emotion={emotion}\nuser_speech={query}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream, application/json"}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _initialize(self) -> None:
        result = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "emo-reco", "version": "0.1"},
        }, capture_session=True)
        if not isinstance(result, Mapping):
            raise AgentResponseError("MCP initialize response is invalid")
        self._rpc("notifications/initialized", {}, notification=True)

    def _check_tool(self) -> None:
        result = self._rpc("tools/list", {})
        tools = result.get("tools") if isinstance(result, Mapping) else None
        if not isinstance(tools, list) or not any(isinstance(t, Mapping) and t.get("name") == "chicken-beauty" for t in tools):
            raise AgentResponseError("MCP tool 'chicken-beauty' is unavailable")
        self._tool_checked = True

    def _rpc(self, method: str, params: Mapping[str, object], *, capture_session: bool = False,
             notification: bool = False) -> object:
        request_id = None if notification else self._next_id
        if request_id is not None:
            self._next_id += 1
        payload: dict[str, object] = {"jsonrpc": "2.0", "method": method, "params": dict(params)}
        if request_id is not None:
            payload["id"] = request_id
        try:
            response = self._session.post(self._url, headers=self._headers(), json=payload,
                                          timeout=(self._settings.connect_timeout_seconds, self._settings.read_timeout_seconds),
                                          stream=True)
        except requests.Timeout:
            raise AgentTransportError("MCP request timed out") from None
        except requests.RequestException:
            raise AgentTransportError("MCP transport failed") from None
        if not 200 <= response.status_code < 300:
            raise AgentResponseError(f"MCP request rejected with HTTP {response.status_code}")
        if capture_session:
            self._session_id = response.headers.get("Mcp-Session-Id") or response.headers.get("mcp-session-id")
            if not self._session_id:
                raise AgentResponseError("MCP initialize response missing session id")
        if notification:
            return None
        return self._parse_stream(response)

    @staticmethod
    def _parse_stream(response: Any) -> object:
        events: list[object] = []
        block: list[str] = []
        for raw in response.iter_lines(decode_unicode=True):
            line = raw.decode() if isinstance(raw, bytes) else str(raw)
            if not line:
                if block:
                    events.append(TongjiMcpAgentClient._event(block)); block = []
            elif line.startswith("data:"):
                block.append(line[5:].lstrip())
            elif block and not line.startswith("event:"):
                # Some proxies omit the ``data:`` prefix on continuation chunks.
                block.append(line)
        if block:
            events.append(TongjiMcpAgentClient._event(block))
        if not events:
            try:
                return response.json()
            except Exception:
                raise AgentResponseError("MCP response contained no message") from None
        for item in reversed(events):
            if isinstance(item, Mapping) and "error" in item:
                raise AgentResponseError("MCP returned a JSON-RPC error")
            if isinstance(item, Mapping) and "result" in item:
                return item["result"]
        raise AgentResponseError("MCP response contained no result")

    @staticmethod
    def _event(lines: Iterable[str]) -> object:
        # HiAgent may split one JSON data field across several transport lines.
        data = "".join(lines)
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _text_from_response(result: object) -> str:
        if isinstance(result, Mapping) and isinstance(result.get("content"), list):
            texts = [item.get("text") for item in result["content"] if isinstance(item, Mapping)
                     and isinstance(item.get("text"), str)]
            text = "".join(texts).strip()
            if text:
                return text
        raise AgentResponseError("MCP tool response contained no text")
