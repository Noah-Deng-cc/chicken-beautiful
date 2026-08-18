from __future__ import annotations

from collections.abc import Mapping

from src.agent import AgentResponseError, TongjiMcpAgentClient
from src.core import AgentSettings


def settings() -> AgentSettings:
    return AgentSettings(True, "tongji_mcp", "https://mcp.invalid", "/proxy", "KEY", "ID",
                         1.0, 2.0, 0, 0.0, True, "secret-key", None)


class Response:
    def __init__(self, data: list[str], headers: Mapping[str, str] | None = None, status_code: int = 200):
        self.status_code = status_code
        self.headers = dict(headers or {})
        self._data = data

    def iter_lines(self, decode_unicode: bool = False):
        return iter(self._data)

    def json(self):
        return {}


class Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        method = kwargs["json"]["method"]
        if method == "initialize":
            return Response(["data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}", ""], {"Mcp-Session-Id": "session-1"})
        if method == "notifications/initialized":
            return Response([])
        if method == "tools/list":
            return Response(["data: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"chicken-beauty\"}]}}", ""])
        return Response([
            "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{\"data\":{\"content\":[{\"text\":\"你好\"}]}}}",
            "",
            "data: {\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"收到啦\"}]}}",
            "",
        ])


def test_mcp_session_tool_call_and_text() -> None:
    session = Session()
    reply = TongjiMcpAgentClient(settings(), session=session).reply(
        "我今天有点累", {"emotion": {"dominant": "sad"}, "files": [{"Name": "x", "Type": "image", "Url": "https://x"}]}, None)
    assert reply.text == "收到啦"
    assert reply.conversation_id == "session-1"
    assert len(session.calls) == 4
    url, call = session.calls[-1]
    assert "api_key=secret-key" in url
    assert call["headers"]["Mcp-Session-Id"] == "session-1"
    assert call["json"]["params"]["name"] == "chicken-beauty"
    assert call["json"]["params"]["arguments"]["Files"][0]["Type"] == "image"


def test_mcp_error_event_is_safe() -> None:
    class ErrorSession(Session):
        def post(self, url, **kwargs):
            return Response(["data: {\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"message\":\"secret-key\"}}", ""] , {"Mcp-Session-Id": "s"})

    try:
        TongjiMcpAgentClient(settings(), session=ErrorSession()).reply("hi", {}, None)
    except AgentResponseError as exc:
        assert "secret-key" not in str(exc)
    else:
        raise AssertionError("expected MCP error")
