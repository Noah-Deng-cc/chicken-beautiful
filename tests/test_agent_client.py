"""T17 external agent client acceptance tests with no real network access.

Inputs: injected settings, in-memory transports, and fake requests sessions.
Outputs: assertions for request mapping, retries, parsing, and secret safety.
Dependencies: pytest, requests, and the Python standard library only.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import logging
from pathlib import Path
import re
from typing import Any

import pytest
import requests

import src.agent as agent
from src.agent import (
    AgentConfigurationError,
    AgentResponseError,
    AgentTransportError,
    HttpResponse,
    RequestsTransport,
    TongjiAgentClient,
    TongjiContract,
    TongjiMcpAgentClient,
)
from src.core import AgentSettings


FIXED_TIME = datetime(2026, 8, 18, 10, 30, tzinfo=timezone.utc)
CANARY_KEY = "test-canary-key-never-disclose-83f7"


def make_settings(**overrides: object) -> AgentSettings:
    """Create valid deterministic settings without using a real credential."""
    values: dict[str, object] = {
        "enabled": True,
        "driver": "tongji",
        "base_url": "https://agent.invalid/api/proxy/api/v1/",
        "endpoint": "/chat",
        "api_key_env": "TEST_AGENT_API_KEY",
        "agent_id_env": "TEST_AGENT_ID",
        "connect_timeout_seconds": 1.25,
        "read_timeout_seconds": 4.5,
        "max_retries": 2,
        "backoff_seconds": 0.5,
        "stream": False,
        "api_key": CANARY_KEY,
        "agent_id": "agent-test-id",
    }
    values.update(overrides)
    return AgentSettings(**values)  # type: ignore[arg-type]


class FakeTransport:
    """Record calls and yield configured responses or exceptions in order."""

    def __init__(self, outcomes: list[HttpResponse | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, object],
        timeout: tuple[float, float],
    ) -> HttpResponse:
        """Record a request and return or raise the next configured outcome."""
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "json_body": dict(json_body),
                "timeout": timeout,
            }
        )
        if not self.outcomes:
            raise AssertionError("unexpected transport call")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def success(text: str = "收到", conversation_id: str | None = "conversation-2") -> HttpResponse:
    """Build a successful response matching the default provisional contract."""
    return HttpResponse(
        200,
        {"data": {"answer": text, "conversation_id": conversation_id}},
        {"X-Test": "safe"},
    )


def make_client(
    transport: FakeTransport,
    *,
    settings: AgentSettings | None = None,
    contract: TongjiContract | None = None,
    sleeps: list[float] | None = None,
    max_backoff_seconds: float = 30.0,
) -> TongjiAgentClient:
    """Create a client with injected transport, clock, and non-blocking sleep."""
    recorded_sleeps = sleeps if sleeps is not None else []
    return TongjiAgentClient(
        settings or make_settings(),
        transport=transport,
        contract=contract,
        sleep=recorded_sleeps.append,
        clock=lambda: FIXED_TIME,
        max_backoff_seconds=max_backoff_seconds,
    )


def test_public_exports_and_default_request_response_mapping() -> None:
    """Default URL, headers, payload, timeout, and response paths match the contract."""
    expected_exports = {
        "AgentClient",
        "AgentConfigurationError",
        "AgentError",
        "AgentResponseError",
        "AgentTransportError",
        "HttpResponse",
        "HttpTransport",
        "RequestsTransport",
        "TongjiAgentClient",
        "TongjiContract",
        "TongjiMcpAgentClient",
    }
    assert set(agent.__all__) == expected_exports
    transport = FakeTransport([success("  你好，世界  ", "next-session")])

    reply = make_client(transport).reply("我的状态如何？", {"情绪": "平静", "co2": 650}, None)

    assert reply.text == "你好，世界"
    assert reply.timestamp == FIXED_TIME
    assert reply.conversation_id == "next-session"
    assert transport.calls == [
        {
            "url": "https://agent.invalid/api/proxy/api/v1/chat",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {CANARY_KEY}",
            },
            "json_body": {
                "query": "我的状态如何？",
                "context": {"情绪": "平静", "co2": 650},
                "agent_id": "agent-test-id",
                "stream": False,
            },
            "timeout": (1.25, 4.5),
        }
    ]


def test_custom_url_endpoint_auth_and_field_mapping() -> None:
    """Every provisional request and response field can be remapped."""
    contract = TongjiContract(
        auth_header="X-Agent-Key",
        auth_scheme="Token",
        query_field="prompt",
        context_field="metadata",
        conversation_id_field="thread",
        agent_id_field="bot",
        stream_field="is_streaming",
        response_text_path=("result", "message", "content"),
        response_conversation_id_path=("result", "thread_id"),
    )
    transport = FakeTransport(
        [HttpResponse(201, {"result": {"message": {"content": "ok"}, "thread_id": "t-2"}})]
    )
    settings = make_settings(base_url="https://custom.invalid/root///", endpoint="///invoke")

    reply = make_client(transport, settings=settings, contract=contract).reply(
        "continue", {"nested": {"value": True}}, "t-1"
    )

    call = transport.calls[0]
    assert call["url"] == "https://custom.invalid/root/invoke"
    assert call["headers"] == {
        "Content-Type": "application/json",
        "X-Agent-Key": f"Token {CANARY_KEY}",
    }
    assert call["json_body"] == {
        "prompt": "continue",
        "metadata": {"nested": {"value": True}},
        "thread": "t-1",
        "bot": "agent-test-id",
        "is_streaming": False,
    }
    assert reply.conversation_id == "t-2"


def test_optional_contract_fields_and_previous_conversation_id_fallback() -> None:
    """Optional agent/stream fields may be omitted and an existing session can be retained."""
    contract = TongjiContract(
        agent_id_field=None,
        stream_field=None,
        response_conversation_id_path=None,
    )
    settings = make_settings(agent_id=None)
    transport = FakeTransport([HttpResponse(200, {"data": {"answer": "continued"}})])
    reply = make_client(transport, settings=settings, contract=contract).reply("next", {}, "old-id")
    payload = transport.calls[0]["json_body"]
    assert payload == {"query": "next", "context": {}, "conversation_id": "old-id"}
    assert reply.conversation_id == "old-id"


def test_unicode_empty_and_very_long_query_boundaries() -> None:
    """Unicode and long text are preserved while empty text is rejected before transport."""
    long_query = "情绪🙂" * 100_000
    transport = FakeTransport([success("unicode-ok"), success("long-ok")])
    client = make_client(transport)
    assert client.reply("你好，Dormitory！", {"备注": "温度 36.5℃"}, None).text == "unicode-ok"
    assert client.reply(long_query, {}, None).text == "long-ok"
    assert transport.calls[1]["json_body"]["query"] == long_query  # type: ignore[index]

    for empty in ("", "   ", "\n\t"):
        with pytest.raises(ValueError, match="non-empty"):
            client.reply(empty, {}, None)
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    ("query", "context", "conversation_id", "error", "message"),
    [
        (123, {}, None, TypeError, "query must be a string"),
        ("ok", [], None, TypeError, "context must be a mapping"),
        ("ok", {1: "bad-key"}, None, TypeError, "string keys"),
        ("ok", {}, 123, TypeError, "conversation_id must be"),
        ("ok", {}, "  ", ValueError, "conversation_id must be a non-empty"),
    ],
)
def test_invalid_reply_inputs_fail_before_transport(
    query: object,
    context: object,
    conversation_id: object,
    error: type[Exception],
    message: str,
) -> None:
    """Wrong direct input types and blank session IDs fail deterministically."""
    transport = FakeTransport([])
    with pytest.raises(error, match=message):
        make_client(transport).reply(query, context, conversation_id)  # type: ignore[arg-type]
    assert transport.calls == []


@pytest.mark.parametrize(
    "context",
    [
        {"value": object()},
        {"not_finite": float("nan")},
        {"nested": {"infinite": float("inf")}},
    ],
)
def test_non_json_context_is_rejected_without_transport(context: Mapping[str, object]) -> None:
    """Non-serializable and non-standard numeric context values never reach HTTP."""
    transport = FakeTransport([])
    with pytest.raises(AgentConfigurationError, match="not JSON serializable"):
        make_client(transport).reply("query", context, None)
    assert transport.calls == []


def test_requests_transport_creates_session_lazily_and_reuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requests Session is absent at construction, created on first post, and reused."""
    created: list[FakeSession] = []

    class FakeRawResponse:
        status_code = 200
        headers = {"X-Fake": "yes"}

        def json(self) -> object:
            return {"data": {"answer": "ok", "conversation_id": None}}

    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            created.append(self)

        def post(self, url: str, **kwargs: object) -> FakeRawResponse:
            self.calls.append({"url": url, **kwargs})
            return FakeRawResponse()

    monkeypatch.setattr(requests, "Session", FakeSession)
    transport = RequestsTransport()
    assert transport._session is None
    assert created == []
    first = transport.post("https://fake.invalid", headers={}, json_body={}, timeout=(1, 2))
    second = transport.post("https://fake.invalid", headers={}, json_body={}, timeout=(1, 2))
    assert first.status_code == second.status_code == 200
    assert len(created) == 1
    assert len(created[0].calls) == 2


@pytest.mark.parametrize("error", [requests.Timeout(CANARY_KEY), requests.ConnectionError(CANARY_KEY)])
def test_requests_transport_sanitizes_timeout_and_connection_errors(
    monkeypatch: pytest.MonkeyPatch, error: requests.RequestException
) -> None:
    """Raw requests exception text containing a credential is never propagated."""
    class FailingSession:
        def post(self, *args: object, **kwargs: object) -> Any:
            raise error

    monkeypatch.setattr(requests, "Session", FailingSession)
    with pytest.raises(AgentTransportError) as captured:
        RequestsTransport().post(
            "https://fake.invalid",
            headers={"Authorization": f"Bearer {CANARY_KEY}"},
            json_body={"query": CANARY_KEY},
            timeout=(1, 1),
        )
    assert CANARY_KEY not in str(captured.value)
    assert CANARY_KEY not in repr(captured.value)


def test_requests_transport_maps_non_json_and_empty_body_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid or empty response bodies become opaque None rather than leaking decoder text."""
    class NonJsonResponse:
        status_code = 200
        headers = {"X-Secret": CANARY_KEY}

        def json(self) -> object:
            raise requests.exceptions.JSONDecodeError(CANARY_KEY, "", 0)

    class FakeSession:
        def post(self, *args: object, **kwargs: object) -> NonJsonResponse:
            return NonJsonResponse()

    monkeypatch.setattr(requests, "Session", FakeSession)
    response = RequestsTransport().post(
        "https://fake.invalid", headers={}, json_body={}, timeout=(1, 1)
    )
    assert response.body is None
    assert CANARY_KEY not in repr(response)
    with pytest.raises(AgentResponseError, match="not a JSON object") as captured:
        make_client(FakeTransport([response])).reply("query", {}, None)
    assert CANARY_KEY not in str(captured.value)


@pytest.mark.parametrize(
    ("status", "expected_calls"),
    [(429, 3), (500, 3), (503, 3)],
)
def test_retryable_statuses_honor_retry_count_and_backoff_cap(
    status: int, expected_calls: int, caplog: pytest.LogCaptureFixture
) -> None:
    """429 and representative 5xx statuses retry exactly and cap exponential delays."""
    sleeps: list[float] = []
    transport = FakeTransport([HttpResponse(status, {"secret": CANARY_KEY})] * expected_calls)
    client = make_client(
        transport,
        settings=make_settings(max_retries=2, backoff_seconds=10.0),
        sleeps=sleeps,
        max_backoff_seconds=12.0,
    )
    with caplog.at_level(logging.WARNING, logger="src.agent.tongji"):
        with pytest.raises(AgentTransportError) as captured:
            client.reply("query", {}, None)
    assert len(transport.calls) == expected_calls
    assert sleeps == [10.0, 12.0]
    assert CANARY_KEY not in str(captured.value)
    assert CANARY_KEY not in caplog.text


def test_timeout_and_connection_failures_retry_then_recover() -> None:
    """Injected timeout and connection failures share bounded transport retry behavior."""
    sleeps: list[float] = []
    transport = FakeTransport(
        [requests.Timeout(CANARY_KEY), requests.ConnectionError(CANARY_KEY), success("recovered")]
    )
    reply = make_client(transport, sleeps=sleeps).reply("query", {}, None)
    assert reply.text == "recovered"
    assert len(transport.calls) == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize("status", [400, 401])
def test_non_retryable_client_errors_are_not_retried(status: int) -> None:
    """400 and 401 are parsed as rejections immediately with no sleep or retry."""
    sleeps: list[float] = []
    transport = FakeTransport([HttpResponse(status, {"detail": CANARY_KEY})])
    with pytest.raises(AgentResponseError, match=rf"HTTP {status}") as captured:
        make_client(transport, sleeps=sleeps).reply("query", {}, None)
    assert len(transport.calls) == 1
    assert sleeps == []
    assert CANARY_KEY not in str(captured.value)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (None, "not a JSON object"),
        ("", "not a JSON object"),
        ({}, "missing field 'reply text'"),
        ({"data": {}}, "missing field 'reply text'"),
        ({"data": {"answer": [CANARY_KEY], "conversation_id": "c"}}, "non-empty string"),
        ({"data": {"answer": "   ", "conversation_id": "c"}}, "non-empty string"),
        ({"data": {"answer": "ok"}}, "missing field 'conversation id'"),
        ({"data": {"answer": "ok", "conversation_id": 42}}, "string or null"),
    ],
)
def test_malformed_response_bodies_are_rejected_without_leaking(
    body: object, message: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-JSON, empty, missing, and wrongly typed fields yield safe errors."""
    response = HttpResponse(200, body, {"X-Attacker": CANARY_KEY})
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(AgentResponseError, match=re.escape(message)) as captured:
            make_client(FakeTransport([response])).reply("query", {}, None)
    combined = f"{captured.value!s}\n{captured.value!r}\n{response!r}\n{caplog.text}"
    assert CANARY_KEY not in combined


def test_malicious_transport_exception_is_sanitized_in_error_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Transport-controlled exception text and request secrets never escape diagnostics."""
    transport = FakeTransport([RuntimeError(CANARY_KEY)] * 3)
    with caplog.at_level(logging.WARNING, logger="src.agent.tongji"):
        with pytest.raises(AgentTransportError) as captured:
            make_client(transport).reply(CANARY_KEY, {"secret": CANARY_KEY}, None)
    combined = f"{captured.value!s}\n{captured.value!r}\n{caplog.text}"
    assert CANARY_KEY not in combined


@pytest.mark.parametrize(
    "overrides",
    [
        {"api_key": None},
        {"api_key": ""},
        {"base_url": " "},
        {"endpoint": ""},
        {"connect_timeout_seconds": 0},
        {"read_timeout_seconds": -1},
        {"max_retries": -1},
        {"backoff_seconds": -0.1},
    ],
)
def test_invalid_settings_are_rejected_without_secret_in_repr(overrides: dict[str, object]) -> None:
    """Invalid construction fails safely and settings repr always hides the API key."""
    settings = make_settings(**overrides)
    assert CANARY_KEY not in repr(settings)
    with pytest.raises(AgentConfigurationError, match="invalid agent client configuration"):
        TongjiAgentClient(settings, transport=FakeTransport([]))


def test_streaming_mode_is_explicitly_rejected() -> None:
    """Unverified streaming is refused during construction before any request."""
    transport = FakeTransport([])
    with pytest.raises(AgentConfigurationError, match="streaming response contract"):
        make_client(transport, settings=make_settings(stream=True))
    assert transport.calls == []


def test_duplicate_request_fields_and_missing_agent_id_are_rejected() -> None:
    """Ambiguous request mappings and required missing identifiers cannot initialize."""
    duplicate = TongjiContract(query_field="same", context_field="same")
    with pytest.raises(AgentConfigurationError, match="invalid agent client configuration"):
        make_client(FakeTransport([]), contract=duplicate)
    with pytest.raises(AgentConfigurationError, match="agent identifier is required"):
        make_client(FakeTransport([]), settings=make_settings(agent_id=None))


def test_relevant_source_has_no_hardcoded_request_credentials_or_key_literals() -> None:
    """Agent source contains neither request credentials nor key-shaped string literals."""
    project_root = Path(__file__).resolve().parents[1]
    request_fixture = project_root / "request.md"
    if not request_fixture.is_file():
        pytest.skip("local request.md fixture is unavailable")
    request_text = request_fixture.read_text(encoding="utf-8")
    source_files = sorted((project_root / "src" / "agent").glob("*.py"))
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in source_files)

    credential_candidates = {
        line.strip()
        for line in request_text.splitlines()
        if re.fullmatch(r"[A-Za-z0-9_-]{16,}", line.strip())
    }
    assert credential_candidates, "request fixture must contain a detectable credential candidate"
    assert all(candidate not in source_text for candidate in credential_candidates)
    assert not re.search(
        r"(?i)(?:api[_-]?key|token|authorization)\s*=\s*['\"][^'\"]{8,}['\"]",
        source_text,
    )
