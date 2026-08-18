"""T16 system TTS acceptance tests using only fake child processes.

Inputs: SystemSpeechOutput and injected Popen-compatible fakes.
Outputs: safety, error handling, lifecycle, and process cleanup assertions.
Dependencies: pytest and the Python standard library; never starts real TTS.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import subprocess
from threading import Event, Lock
from typing import Sequence

import pytest

from src.audio.system_tts import SystemSpeechOutput


class FakeProcess:
    """Deterministic Popen handle with configurable wait and cleanup behavior."""

    def __init__(self, *, return_code: int = 0, timeout_once: bool = False,
                 block: bool = False, terminate_times_out: bool = False) -> None:
        """Create a controllable fake subprocess.

        Args: return_code: Normal process result. timeout_once: First wait raises timeout.
            block: Initial wait blocks until terminate or kill. terminate_times_out: Graceful
            cleanup wait times out, requiring kill.
        Returns: None.
        Raises: None.
        """
        self.return_code = return_code
        self.timeout_once = timeout_once
        self.block = block
        self.terminate_times_out = terminate_times_out
        self.entered = Event()
        self.finished = Event()
        self.terminated = 0
        self.killed = 0
        self.wait_timeouts: list[float | None] = []
        self._running = True
        self._waits = 0
        self._lock = Lock()

    def poll(self) -> int | None:
        """Return None while running, otherwise the configured result."""
        with self._lock:
            return None if self._running else self.return_code

    def wait(self, timeout: float | None = None) -> int:
        """Simulate normal, timeout, or interruptible child-process waiting."""
        self.wait_timeouts.append(timeout)
        self._waits += 1
        if self.timeout_once and self._waits == 1:
            raise subprocess.TimeoutExpired("fake-tts", timeout)
        if self.block and self._waits == 1:
            self.entered.set()
            if not self.finished.wait(1.0):
                raise AssertionError("fake process was not cleaned up")
        if self.terminate_times_out and self.terminated and not self.killed:
            raise subprocess.TimeoutExpired("fake-tts", timeout)
        with self._lock:
            self._running = False
        return self.return_code

    def terminate(self) -> None:
        """Record graceful termination and release blocking waits."""
        self.terminated += 1
        self.finished.set()

    def kill(self) -> None:
        """Record forced termination and release blocking waits."""
        self.killed += 1
        self.finished.set()


class FakePopen:
    """Records safe process construction and returns preselected results."""

    def __init__(self, outcomes: Sequence[FakeProcess | BaseException]) -> None:
        """Store child outcomes for successive calls.

        Args: outcomes: Process handles or launch exceptions.
        Returns: None.
        Raises: None.
        """
        self._outcomes = list(outcomes)
        self.calls: list[tuple[tuple[str, ...], bool, int, int, int]] = []

    def __call__(self, args: Sequence[str], *, shell: bool, stdin: int,
                 stdout: int, stderr: int) -> FakeProcess:
        """Record invocation and return/raise the next configured outcome."""
        self.calls.append((tuple(args), shell, stdin, stdout, stderr))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_success_uses_literal_argv_without_shell_parsing() -> None:
    """Valid Chinese text is passed as one literal argv element with shell disabled."""
    factory = FakePopen([FakeProcess()])
    output = SystemSpeechOutput(("espeak-ng", "-v", "cmn", "{text}"), popen_factory=factory)

    assert output.speak("  你好; $(touch should-not-run)  ") is True
    assert factory.calls == [
        (("espeak-ng", "-v", "cmn", "你好; $(touch should-not-run)"), False,
         subprocess.DEVNULL, subprocess.DEVNULL, subprocess.DEVNULL)
    ]


@pytest.mark.parametrize("text", ["", " \t\n "])
def test_blank_text_is_rejected_without_launch(text: str) -> None:
    """Blank speech returns False and never creates a process."""
    factory = FakePopen([FakeProcess()])
    output = SystemSpeechOutput(popen_factory=factory)

    assert output.speak(text) is False
    assert factory.calls == []


@pytest.mark.parametrize("text", [None, 1, b"text"])
def test_non_string_text_is_rejected(text: object) -> None:
    """The public speech interface rejects wrong text types before launch."""
    output = SystemSpeechOutput(popen_factory=FakePopen([FakeProcess()]))

    with pytest.raises(TypeError, match="text must be a string"):
        output.speak(text)  # type: ignore[arg-type]


def test_long_and_nul_text_are_rejected_without_launch() -> None:
    """Length limits and NUL prevention are enforced before subprocess creation."""
    factory = FakePopen([FakeProcess(), FakeProcess()])
    output = SystemSpeechOutput(max_text_chars=3, popen_factory=factory)

    with pytest.raises(ValueError, match="too long or contains NUL"):
        output.speak("four")
    with pytest.raises(ValueError, match="too long or contains NUL"):
        output.speak("a\0b")
    assert factory.calls == []


def test_missing_command_nonzero_and_wait_error_return_false() -> None:
    """Recoverable launch, exit-code, and wait failures return False safely."""
    wait_failure = FakeProcess()
    def failed_wait(timeout: float | None = None) -> int:
        raise OSError("injected wait error")
    wait_failure.wait = failed_wait  # type: ignore[method-assign]
    factory = FakePopen([FileNotFoundError("missing"), FakeProcess(return_code=7), wait_failure])
    output = SystemSpeechOutput(popen_factory=factory)

    assert output.speak("missing") is False
    assert output.speak("nonzero") is False
    assert output.speak("wait failure") is False


def test_timeout_terminates_process_and_clears_reference() -> None:
    """A playback timeout performs graceful cleanup and leaves no active handle."""
    process = FakeProcess(timeout_once=True)
    output = SystemSpeechOutput(timeout_seconds=0.1, terminate_timeout_seconds=0.2,
                                popen_factory=FakePopen([process]))

    assert output.speak("timeout") is False
    assert process.terminated == 1
    assert process.killed == 0
    assert output._process is None


def test_timeout_kills_when_graceful_stop_expires() -> None:
    """Cleanup escalates to kill when terminate does not finish inside its grace period."""
    process = FakeProcess(timeout_once=True, terminate_times_out=True)
    output = SystemSpeechOutput(popen_factory=FakePopen([process]))

    assert output.speak("stuck") is False
    assert process.terminated == 1
    assert process.killed == 1
    assert output._process is None


def test_cancel_stops_active_speech_and_next_speech_can_run() -> None:
    """Cancel wakes an active call, cleans it up, and does not poison a later call."""
    blocked, recovered = FakeProcess(block=True), FakeProcess()
    output = SystemSpeechOutput(popen_factory=FakePopen([blocked, recovered]))

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(output.speak, "blocking")
        assert blocked.entered.wait(1.0), "speech did not reach fake wait"
        output.cancel()
        assert pending.result(timeout=1.0) is False

    assert blocked.terminated == 1
    assert output._process is None
    assert output.speak("recovered") is True


def test_close_is_concurrent_safe_idempotent_and_prevents_reuse() -> None:
    """Close cancels an in-flight process and permanently rejects new speech."""
    process = FakeProcess(block=True)
    output = SystemSpeechOutput(popen_factory=FakePopen([process]))

    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = executor.submit(output.speak, "blocking")
        assert process.entered.wait(1.0), "speech did not reach fake wait"
        closes = [executor.submit(output.close) for _ in range(2)]
        for close in closes:
            close.result(timeout=1.0)
        assert pending.result(timeout=1.0) is False

    assert output.closed is True
    assert process.terminated == 1
    assert output._process is None
    with pytest.raises(RuntimeError, match="closed"):
        output.speak("again")


@pytest.mark.parametrize(
    "template",
    [("", "{text}"), ("espeak-ng",), ("{text}", "-v", "cmn"),
     ("espeak-ng", "{text}", "{text}")],
)
def test_invalid_command_templates_are_rejected(template: tuple[str, ...]) -> None:
    """Templates require a fixed executable and exactly one text placeholder."""
    with pytest.raises(ValueError):
        SystemSpeechOutput(template)
