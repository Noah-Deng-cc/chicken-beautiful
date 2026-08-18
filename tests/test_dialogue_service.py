"""T19 dialogue coordination acceptance tests using isolated in-memory collaborators.

Inputs: fake audio, agent, fusion, and state collaborators.
Outputs: observable dialogue turns, calls, state updates, and safe log assertions.
Dependencies: pytest and project domain objects only.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from threading import Event
from typing import Any

import pytest

from src.dialogue import DialogueService
from src.domain import AgentReply


FIXED_TIME = datetime(2026, 8, 18, 10, 30, tzinfo=timezone.utc)
PRIVATE_TEXT = "私密对话内容-不得进入日志"
CANARY_SECRET = "dialogue-canary-secret-never-log"


class FakeInput:
    """Return configured recognition outcomes and record requested timeouts."""

    def __init__(self, outcomes: list[str | None | BaseException]) -> None:
        self._outcomes = list(outcomes)
        self.timeouts: list[float] = []

    def listen(self, timeout: float) -> str | None:
        """Return the next outcome or raise its configured exception."""
        self.timeouts.append(timeout)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeOutput:
    """Record attempted speech and optionally fail individual calls."""

    def __init__(self, outcomes: list[bool | BaseException] | None = None) -> None:
        self._outcomes = list(outcomes or [])
        self.texts: list[str] = []

    def speak(self, text: str) -> bool:
        """Record text, then return or raise the next configured result."""
        self.texts.append(text)
        outcome = self._outcomes.pop(0) if self._outcomes else True
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeAgent:
    """Record context and conversation IDs, then yield configured replies."""

    def __init__(self, outcomes: list[AgentReply | BaseException]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, dict[str, object], str | None]] = []

    def reply(self, query: str, context: dict[str, object], conversation_id: str | None) -> AgentReply:
        """Record an agent request and return its configured result."""
        self.calls.append((query, context, conversation_id))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeFusion:
    """Return a fixed context and retain the exact supplied snapshot."""

    def __init__(self, context: dict[str, object]) -> None:
        self.context = context
        self.snapshots: list[object] = []

    def build_context(self, snapshot: object) -> dict[str, object]:
        """Record snapshot and return the configured structured context."""
        self.snapshots.append(snapshot)
        return self.context


class FakeState:
    """Supply a stable snapshot and record successful dialogue turn updates."""

    def __init__(self) -> None:
        self.snapshot = {"snapshot": "current"}
        self.updated: list[object] = []

    def get_snapshot(self) -> object:
        """Return the opaque current snapshot."""
        return self.snapshot

    def update(self, value: object) -> object:
        """Record a state update."""
        self.updated.append(value)
        return value


def reply(text: str = "请打开窗户", conversation_id: str | None = "conversation-1") -> AgentReply:
    """Create a valid deterministic agent reply."""
    return AgentReply(text=text, timestamp=FIXED_TIME, conversation_id=conversation_id)


def make_service(
    inputs: list[str | None | BaseException], outcomes: list[AgentReply | BaseException],
    *, output_outcomes: list[bool | BaseException] | None = None, context: dict[str, object] | None = None,
) -> tuple[DialogueService, FakeInput, FakeOutput, FakeAgent, FakeFusion, FakeState]:
    """Construct one dialogue service with fully observable fake collaborators."""
    input_device, output = FakeInput(inputs), FakeOutput(output_outcomes)
    agent, fusion, state = FakeAgent(outcomes), FakeFusion(context or {"co2": {"ppm": 1800}}), FakeState()
    return DialogueService(input_device, output, agent, fusion, state, 2.5), input_device, output, agent, fusion, state


@pytest.mark.parametrize("recognized", [None, "", " \t\n "])
def test_empty_recognition_skips_fusion_agent_and_playback(recognized: str | None) -> None:
    """Silence and whitespace must not invoke any downstream component."""
    service, _, output, agent, fusion, state = make_service([recognized], [reply()])

    assert service.run_once() is None
    assert agent.calls == []
    assert fusion.snapshots == []
    assert output.texts == []
    assert state.updated == []


def test_unicode_long_text_context_and_conversation_id_continuation() -> None:
    """Unicode long input preserves text, context, and server conversation succession."""
    long_text = ("你好，今天有点焦虑。" * 1_000) + "请帮我安排一下。"
    context = {"co2": {"ppm": 1800}, "reminders": [{"message": "通风"}]}
    service, input_device, output, agent, fusion, state = make_service(
        [long_text, "第二轮"], [reply("第一轮回复", "cid-1"), reply("第二轮回复", "cid-2")], context=context,
    )

    first, second = service.run_once(), service.run_once()

    assert first is not None and second is not None
    assert first.user_text == long_text
    assert second.user_text == "第二轮"
    assert agent.calls == [(long_text, context, None), ("第二轮", context, "cid-1")]
    assert input_device.timeouts == [2.5, 2.5]
    assert fusion.snapshots == [state.snapshot, state.snapshot]
    assert output.texts == ["第一轮回复", "第二轮回复"]
    assert state.updated == [first, second]
    assert service.conversation_id == "cid-2"


def test_none_conversation_id_does_not_discard_existing_session() -> None:
    """A reply without a conversation ID retains a previously established session."""
    service, _, _, agent, _, _ = make_service(
        ["第一轮", "第二轮", "第三轮"], [reply("一", "cid-1"), reply("二", None), reply("三", "cid-3")],
    )

    assert service.run_once() is not None
    assert service.run_once() is not None
    assert service.run_once() is not None
    assert [call[2] for call in agent.calls] == [None, "cid-1", "cid-1"]
    assert service.conversation_id == "cid-3"


def test_agent_failure_falls_back_without_state_update_and_keeps_prior_session(caplog: pytest.LogCaptureFixture) -> None:
    """Agent errors speak a local fallback, remain non-fatal, and never expose input or secrets."""
    failure = RuntimeError(f"remote failed for {PRIVATE_TEXT}; token={CANARY_SECRET}")
    service, _, output, agent, _, state = make_service(["ok", PRIVATE_TEXT], [reply("ok", "cid-1"), failure])

    assert service.run_once() is not None
    with caplog.at_level(logging.WARNING, logger="src.dialogue.service"):
        assert service.run_once() is None

    assert [call[2] for call in agent.calls] == [None, "cid-1"]
    assert service.conversation_id == "cid-1"
    assert output.texts == ["ok", "当前网络服务暂不可用，请稍后再试。"]
    assert len(state.updated) == 1
    assert PRIVATE_TEXT not in caplog.text
    assert CANARY_SECRET not in caplog.text


def test_reply_and_fallback_playback_failures_are_isolated(caplog: pytest.LogCaptureFixture) -> None:
    """Both normal and fallback TTS failures cannot prevent recoverable dialogue behavior."""
    service, _, output, _, _, state = make_service(
        ["正常播报失败", "降级播报失败"], [reply("reply"), RuntimeError("agent down")],
        output_outcomes=[RuntimeError("speaker down"), RuntimeError("speaker down")],
    )

    with caplog.at_level(logging.WARNING, logger="src.dialogue.service"):
        turn = service.run_once()
        fallback = service.run_once()

    assert turn is not None and fallback is None
    assert len(state.updated) == 1
    assert output.texts == ["reply", "当前网络服务暂不可用，请稍后再试。"]
    assert "dialogue reply playback failed" in caplog.text
    assert "dialogue fallback playback failed" in caplog.text


def test_run_honors_stop_event_after_one_iteration() -> None:
    """The loop exits once a collaborator sets the supplied stop event."""
    stop = Event()

    class StopInput(FakeInput):
        def listen(self, timeout: float) -> str | None:
            result = super().listen(timeout)
            stop.set()
            return result

    input_device = StopInput([None])
    service = DialogueService(input_device, FakeOutput(), FakeAgent([reply()]), FakeFusion({}), FakeState(), 1.0)
    service.run(stop)
    assert input_device.timeouts == [1.0]


def test_run_continues_after_iteration_exception_until_stopped(caplog: pytest.LogCaptureFixture) -> None:
    """An unexpected input exception is logged safely and does not crash the run loop."""
    stop = Event()

    class RecoveringInput(FakeInput):
        def listen(self, timeout: float) -> str | None:
            result = super().listen(timeout)
            if len(self.timeouts) == 2:
                stop.set()
            return result

    input_device = RecoveringInput([RuntimeError("microphone unavailable"), None])
    service = DialogueService(input_device, FakeOutput(), FakeAgent([reply()]), FakeFusion({}), FakeState(), 1.0)
    with caplog.at_level(logging.ERROR, logger="src.dialogue.service"):
        service.run(stop)
    assert input_device.timeouts == [1.0, 1.0]
    assert "dialogue loop iteration failed" in caplog.text
