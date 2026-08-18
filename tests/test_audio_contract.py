"""T14 audio abstraction and deterministic mock acceptance tests.

Inputs: public audio interfaces and injected recognition/playback outcomes.
Outputs: assertions for validation, lifecycle, cancellation, concurrency, and dependencies.
Dependencies: pytest and the Python standard library only; no real audio devices.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import inspect
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock
from typing import get_type_hints

import pytest

import src.audio as audio
from src.audio import AudioComponent, MockSpeechInput, MockSpeechOutput, SpeechInput, SpeechOutput


FORBIDDEN_AUDIO_DEPENDENCIES = {
    "alsa",
    "pyaudio",
    "pygame",
    "pyttsx3",
    "sounddevice",
    "speech_recognition",
    "vosk",
}


class TrackingEvent:
    """Expose when the mock enters its real interruptible event wait."""

    def __init__(self) -> None:
        self.entered = Event()
        self._event = Event()

    def clear(self) -> None:
        """Clear the wrapped cancellation state."""
        self.entered.clear()
        self._event.clear()

    def is_set(self) -> bool:
        """Return the wrapped cancellation state."""
        return self._event.is_set()

    def set(self) -> None:
        """Set cancellation and wake any waiter."""
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Signal wait entry, then perform the real interruptible wait."""
        self.entered.set()
        return self._event.wait(timeout)


def test_public_api_and_abstract_contracts() -> None:
    """The package exposes the intended abstractions and none is instantiable."""
    assert set(audio.__all__) == {
        "AudioComponent",
        "InputItem",
        "MockSpeechInput",
        "MockSpeechOutput",
        "OutputOutcome",
        "SpeechInput",
        "SpeechOutput",
        "WaitFunction",
    }
    assert inspect.isabstract(AudioComponent)
    assert inspect.isabstract(SpeechInput)
    assert inspect.isabstract(SpeechOutput)
    assert AudioComponent.__abstractmethods__ == {"cancel", "close", "closed"}
    assert SpeechInput.__abstractmethods__ == {"cancel", "close", "closed", "listen"}
    assert SpeechOutput.__abstractmethods__ == {"cancel", "close", "closed", "speak"}
    for abstract_type in (AudioComponent, SpeechInput, SpeechOutput):
        with pytest.raises(TypeError, match="abstract"):
            abstract_type()  # type: ignore[abstract]


def test_required_method_signatures_and_return_types() -> None:
    """Input and output interfaces match the task's exact public signatures."""
    assert list(inspect.signature(SpeechInput.listen).parameters) == ["self", "timeout"]
    assert list(inspect.signature(SpeechOutput.speak).parameters) == ["self", "text"]
    assert list(inspect.signature(AudioComponent.cancel).parameters) == ["self"]
    assert list(inspect.signature(AudioComponent.close).parameters) == ["self"]
    assert get_type_hints(SpeechInput.listen)["timeout"] is float
    assert get_type_hints(SpeechInput.listen)["return"] == str | None
    assert get_type_hints(SpeechOutput.speak)["text"] is str
    assert get_type_hints(SpeechOutput.speak)["return"] is bool


@pytest.mark.parametrize(
    ("response", "expected"),
    [("  hello  ", "hello"), (None, None), (" \t\n ", None)],
)
def test_fixed_input_is_repeatable_and_normalized(
    response: str | None,
    expected: str | None,
) -> None:
    """Fixed text, fixed None, and fixed blank values repeat deterministically."""
    source = MockSpeechInput(response)
    assert [source.listen(1.0) for _ in range(3)] == [expected, expected, expected]


def test_input_sequence_preserves_order_and_exhausts_to_none() -> None:
    """Finite input values are consumed once and exhaustion remains None."""
    source = MockSpeechInput([" first ", None, "second", "   "])
    assert [source.listen(1.0) for _ in range(6)] == [
        "first",
        None,
        "second",
        None,
        None,
        None,
    ]


def test_injected_input_exception_is_consumed_then_next_item_remains_available() -> None:
    """Recognition failures propagate exactly once without corrupting the cursor."""
    failure = LookupError("injected ASR failure")
    source = MockSpeechInput([failure, "recovered"])
    with pytest.raises(LookupError, match="injected ASR failure") as caught:
        source.listen(1.0)
    assert caught.value is failure
    assert source.listen(1.0) == "recovered"
    assert source.listen(1.0) is None


def test_timeout_does_not_consume_the_pending_input() -> None:
    """A timeout returns promptly through injection and preserves the sequence head."""
    waited: list[float] = []

    def wait_without_sleep(seconds: float) -> bool:
        waited.append(seconds)
        return False

    source = MockSpeechInput(["after timeout"], delay_seconds=2.0, wait=wait_without_sleep)
    assert source.listen(0.25) is None
    assert waited == [0.25]
    assert source.listen(2.0) == "after timeout"
    assert waited == [0.25, 2.0]


def test_cancel_wakes_listener_without_consuming_and_next_call_succeeds() -> None:
    """Cancellation interrupts one call only; the next listen consumes the same item."""
    source_holder: dict[str, MockSpeechInput] = {}
    entered = Event()
    call_lock = Lock()
    calls = 0

    def controlled_wait(seconds: float) -> bool:
        nonlocal calls
        with call_lock:
            calls += 1
            current = calls
        if current == 1:
            entered.set()
            return source_holder["source"]._cancelled.wait(seconds)
        return source_holder["source"]._cancelled.is_set()

    source = MockSpeechInput(["still pending"], delay_seconds=30.0, wait=controlled_wait)
    source_holder["source"] = source
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(source.listen, 30.0)
        assert entered.wait(1.0), "listener did not enter its injected wait"
        source.cancel()
        assert pending.result(timeout=1.0) is None

    assert source.listen(30.0) == "still pending"
    assert calls == 2


def test_close_wakes_blocked_listener_and_is_idempotent() -> None:
    """Close interrupts the real event wait, can repeat, and leaves closed state."""
    source = MockSpeechInput("never consumed", delay_seconds=30.0)
    tracked = TrackingEvent()
    source._cancelled = tracked  # type: ignore[assignment]

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(source.listen, 30.0)
        assert tracked.entered.wait(1.0), "listener did not enter its event wait"
        source.close()
        assert pending.result(timeout=1.0) is None

    source.close()
    assert source.closed is True
    with pytest.raises(RuntimeError, match="speech input is closed"):
        source.listen(1.0)


@pytest.mark.parametrize(
    ("timeout", "error", "message"),
    [
        (True, TypeError, "timeout must be a number"),
        ("1", TypeError, "timeout must be a number"),
        (0, ValueError, "timeout must be finite and positive"),
        (-1, ValueError, "timeout must be finite and positive"),
        (float("inf"), ValueError, "timeout must be finite and positive"),
        (float("nan"), ValueError, "timeout must be finite and positive"),
    ],
)
def test_invalid_listen_timeouts_are_rejected(
    timeout: object,
    error: type[Exception],
    message: str,
) -> None:
    """Wrong, non-positive, and non-finite timeout values fail explicitly."""
    source = MockSpeechInput("unused")
    with pytest.raises(error, match=message):
        source.listen(timeout)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"responses": 7}, TypeError, "responses must be"),
        ({"responses": ["valid", object()]}, TypeError, "response items must be"),
        ({"delay_seconds": True}, TypeError, "delay_seconds must be a number"),
        ({"delay_seconds": -0.1}, ValueError, "delay_seconds must be finite"),
        ({"max_text_chars": True}, TypeError, "max_text_chars must be an integer"),
        ({"max_text_chars": 0}, ValueError, "max_text_chars must be positive"),
    ],
)
def test_invalid_input_constructor_values_are_rejected(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    """Invalid sources, delay values, and text limits fail during construction."""
    with pytest.raises(error, match=message):
        MockSpeechInput(**kwargs)  # type: ignore[arg-type]


def test_input_unicode_exact_limit_and_overlong_boundaries() -> None:
    """Unicode is preserved, the exact character limit passes, and limit+1 fails."""
    source = MockSpeechInput([" 你好ab ", "你好abc"], max_text_chars=4)
    assert source.listen(1.0) == "你好ab"
    with pytest.raises(ValueError, match="recognized text exceeds max_text_chars"):
        source.listen(1.0)


@pytest.mark.parametrize("outcome", [True, False])
def test_fixed_output_outcome_repeats_and_records_correctly(outcome: bool) -> None:
    """Fixed success/failure applies repeatedly; only successes enter spoken history."""
    sink = MockSpeechOutput(outcome)
    assert [sink.speak(" one "), sink.speak("two")] == [outcome, outcome]
    assert sink.attempted_texts == ("one", "two")
    assert sink.spoken_texts == (("one", "two") if outcome else ())


def test_output_sequence_success_failure_exception_and_exhaustion_fallback() -> None:
    """Outcome sequences consume in order and become successful after exhaustion."""
    failure = OSError("injected speaker failure")
    sink = MockSpeechOutput([True, False, failure])
    assert sink.speak("first") is True
    assert sink.speak("second") is False
    with pytest.raises(OSError, match="injected speaker failure") as caught:
        sink.speak("third")
    assert caught.value is failure
    assert sink.speak("fourth") is True
    assert sink.attempted_texts == ("first", "second", "third", "fourth")
    assert sink.spoken_texts == ("first", "fourth")


def test_blank_output_is_rejected_without_consuming_or_recording_outcome() -> None:
    """Blank text is a side-effect-free False and leaves the first outcome pending."""
    sink = MockSpeechOutput([False, True])
    assert sink.speak(" \t\n ") is False
    assert sink.attempted_texts == ()
    assert sink.spoken_texts == ()
    assert sink.speak("valid") is False
    assert sink.speak("next") is True


def test_output_unicode_exact_limit_overlong_and_wrong_types() -> None:
    """Output validates normalized Unicode character length and rejects non-strings."""
    sink = MockSpeechOutput(max_text_chars=4)
    assert sink.speak(" 你好ab ") is True
    with pytest.raises(ValueError, match="speech text exceeds max_text_chars"):
        sink.speak("你好abc")
    for invalid in (None, 1, True, b"bytes"):
        with pytest.raises(TypeError, match="text must be a string"):
            sink.speak(invalid)  # type: ignore[arg-type]
    assert sink.attempted_texts == ("你好ab",)
    assert sink.spoken_texts == ("你好ab",)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"outcomes": 1}, TypeError, "outcomes must be"),
        ({"outcomes": [True, 1]}, TypeError, "outcome items must be"),
        ({"max_text_chars": False}, TypeError, "max_text_chars must be an integer"),
        ({"max_text_chars": -1}, ValueError, "max_text_chars must be positive"),
    ],
)
def test_invalid_output_constructor_values_are_rejected(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    """Invalid outcome values and maximum lengths fail during construction."""
    with pytest.raises(error, match=message):
        MockSpeechOutput(**kwargs)  # type: ignore[arg-type]


def test_attempted_and_spoken_properties_are_immutable_snapshots() -> None:
    """History access returns tuples that cannot change with later calls."""
    sink = MockSpeechOutput()
    assert sink.speak("first") is True
    attempted_snapshot = sink.attempted_texts
    spoken_snapshot = sink.spoken_texts
    assert isinstance(attempted_snapshot, tuple)
    assert isinstance(spoken_snapshot, tuple)
    assert sink.speak("second") is True
    assert attempted_snapshot == ("first",)
    assert spoken_snapshot == ("first",)
    assert sink.attempted_texts == ("first", "second")
    assert sink.spoken_texts == ("first", "second")


def test_concurrent_output_recording_is_complete_and_consistent() -> None:
    """Concurrent calls lose, duplicate, or partially record no successful text."""
    messages = [f"message-{index}" for index in range(300)]
    sink = MockSpeechOutput(True)
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(sink.speak, messages))

    attempted = sink.attempted_texts
    spoken = sink.spoken_texts
    assert results == [True] * len(messages)
    assert len(attempted) == len(messages)
    assert len(spoken) == len(messages)
    assert len(set(attempted)) == len(messages)
    assert set(attempted) == set(messages)
    assert spoken == attempted


def test_output_cancel_close_idempotency_and_closed_calls() -> None:
    """Cancel is harmless, close repeats safely, and valid calls after close fail."""
    sink = MockSpeechOutput()
    sink.cancel()
    assert sink.speak("before close") is True
    sink.close()
    sink.close()
    assert sink.closed is True
    with pytest.raises(RuntimeError, match="speech output is closed"):
        sink.speak("after close")
    with pytest.raises(RuntimeError, match="audio component is closed"):
        sink.__enter__()


@pytest.mark.parametrize("component", [MockSpeechInput("text"), MockSpeechOutput()])
def test_context_manager_closes_and_propagates_body_exception(
    component: AudioComponent,
) -> None:
    """Exceptional context exit closes either component without suppression."""
    with pytest.raises(KeyError, match="context failure"):
        with component as active:
            assert active is component
            raise KeyError("context failure")
    assert component.closed is True


def test_audio_modules_statically_import_only_standard_library_and_local_code() -> None:
    """No concrete audio, ASR, or TTS dependency appears in the three task modules."""
    project_root = Path(__file__).resolve().parents[1]
    imported_roots: set[str] = set()
    for relative_path in ("src/audio/__init__.py", "src/audio/base.py", "src/audio/mock.py"):
        tree = ast.parse((project_root / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(FORBIDDEN_AUDIO_DEPENDENCIES)


def test_import_in_fresh_interpreter_does_not_request_audio_dependencies() -> None:
    """Importing src.audio succeeds when every concrete audio package is blocked."""
    project_root = Path(__file__).resolve().parents[1]
    blocker = f"""
import importlib.abc
import sys

blocked = {FORBIDDEN_AUDIO_DEPENDENCIES!r}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in blocked:
            raise RuntimeError(f'forbidden dependency requested: {{fullname}}')
        return None

sys.meta_path.insert(0, Blocker())
import src.audio
print('audio-import-ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", blocker],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "audio-import-ok"
