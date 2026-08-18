"""T15 microphone, VAD, and Vosk acceptance tests.

Inputs: synthetic little-endian int16 PCM, fake streams, sources, recognizers, and clocks.
Outputs: assertions for bounded capture, lazy dependencies, failures, and lifecycle behavior.
Dependencies: pytest and the Python standard library only; no device, model, sleep, or network.
"""

from __future__ import annotations

import ast
from array import array
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from src.audio.microphone import SoundDeviceMicrophone, _pcm_level
from src.audio.vosk_asr import (
    LazyVoskFactory,
    VoskSpeechInput,
    _VoskRecognizer,
    _result_text,
)


def pcm(*samples: int) -> bytes:
    """Encode signed samples using the host-independent little-endian PCM format."""
    values = array("h", samples)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


class SequenceClock:
    """Return deterministic monotonic values and then repeat the last value."""

    def __init__(self, *values: float) -> None:
        self.values = list(values)
        self.last = values[-1] if values else 0.0

    def __call__(self) -> float:
        if self.values:
            self.last = self.values.pop(0)
        return self.last


class FakeSource:
    """Provide scripted PCM reads without touching a microphone."""

    def __init__(self, items: list[bytes | None | Exception]) -> None:
        self.items = list(items)
        self.read_timeouts: list[float] = []
        self.starts = 0
        self.cancels = 0
        self.closes = 0

    def start(self) -> None:
        self.starts += 1

    def read(self, timeout: float) -> bytes | None:
        self.read_timeouts.append(timeout)
        if not self.items:
            return None
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def cancel(self) -> None:
        self.cancels += 1

    def close(self) -> None:
        self.closes += 1


class FakeRecognizer:
    """Record audio and return scripted accepted and final JSON payloads."""

    def __init__(
        self,
        accepted: list[str | None | Exception] | None = None,
        final: str | Exception = '{"text": ""}',
    ) -> None:
        self.accepted = list(accepted or [])
        self.final = final
        self.blocks: list[bytes] = []
        self.final_calls = 0

    def accept_waveform(self, block: bytes) -> str | None:
        self.blocks.append(block)
        item: str | None | Exception = self.accepted.pop(0) if self.accepted else None
        if isinstance(item, Exception):
            raise item
        return item

    def final_result(self) -> str:
        self.final_calls += 1
        if isinstance(self.final, Exception):
            raise self.final
        return self.final


class FakeFactory:
    """Return a fixed recognizer while recording resolved model configuration."""

    def __init__(self, recognizer: FakeRecognizer | Exception) -> None:
        self.recognizer = recognizer
        self.calls: list[tuple[Path, int]] = []

    def __call__(self, model_path: Path, sample_rate: int) -> FakeRecognizer:
        self.calls.append((model_path, sample_rate))
        if isinstance(self.recognizer, Exception):
            raise self.recognizer
        return self.recognizer


def speech_input(
    model_path: Path,
    source: Any,
    recognizer: FakeRecognizer | Exception,
    **kwargs: Any,
) -> tuple[VoskSpeechInput, FakeFactory]:
    """Build a Vosk input with all external effects replaced by fakes."""
    factory = FakeFactory(recognizer)
    instance = VoskSpeechInput(
        model_path,
        source=source,
        recognizer_factory=factory,
        **kwargs,
    )
    return instance, factory


def test_pcm_level_handles_little_endian_boundaries_and_invalid_lengths() -> None:
    """PCM level is the mean absolute int16 value and malformed lengths are ignored."""
    assert _pcm_level(pcm(-32768, -2, 0, 2, 32767)) == pytest.approx(13107.8)
    assert _pcm_level(pcm()) is None
    assert _pcm_level(b"\x01") is None
    assert _pcm_level(pcm(-300, 300)) == 300.0


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"sample_rate": True}, TypeError, "sample_rate must be an integer"),
        ({"sample_rate": 0}, ValueError, "sample_rate must be positive"),
        ({"block_size": 1.5}, TypeError, "block_size must be an integer"),
        ({"block_size": -1}, ValueError, "block_size must be positive"),
        ({"queue_blocks": False}, TypeError, "queue_blocks must be an integer"),
        ({"queue_blocks": 0}, ValueError, "queue_blocks must be positive"),
        ({"device": object()}, TypeError, "device must be an integer, string, or None"),
    ],
)
def test_microphone_rejects_invalid_pcm_configuration(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    """Wrong parameter types and non-positive PCM dimensions fail before import."""
    with pytest.raises(error, match=message):
        SoundDeviceMicrophone(**kwargs)  # type: ignore[arg-type]


def test_microphone_lazily_opens_with_exact_mono_int16_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction is inert and first start passes the exact PCM contract to sounddevice."""
    events: list[tuple[str, object]] = []

    class Stream:
        def __init__(self, **kwargs: object) -> None:
            events.append(("construct", kwargs))

        def start(self) -> None:
            events.append(("start", None))

        def stop(self) -> None:
            events.append(("stop", None))

        def close(self) -> None:
            events.append(("close", None))

    original = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        if name == "sounddevice":
            events.append(("import", name))
            return SimpleNamespace(RawInputStream=Stream)
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    microphone = SoundDeviceMicrophone(
        device="USB microphone", sample_rate=16000, block_size=640, queue_blocks=2
    )
    assert events == []
    microphone.start()
    microphone.start()
    assert events[:2] == [("import", "sounddevice"), ("construct", {
        "samplerate": 16000,
        "blocksize": 640,
        "device": "USB microphone",
        "channels": 1,
        "dtype": "int16",
        "callback": microphone._callback,
    })]
    assert [name for name, _ in events].count("start") == 1
    microphone.close()
    assert [name for name, _ in events][-2:] == ["stop", "close"]


def test_bounded_microphone_queue_drops_oldest_and_callback_copies_data() -> None:
    """A full queue retains the two newest immutable callback blocks."""
    microphone = SoundDeviceMicrophone(queue_blocks=2)
    mutable = bytearray(pcm(1))
    microphone._callback(mutable, 1, None, None)
    mutable[:] = pcm(9)
    microphone._callback(pcm(2), 1, None, None)
    microphone._callback(pcm(3), 1, None, None)
    assert microphone.read(0.01) == pcm(2)
    assert microphone.read(0.01) == pcm(3)
    assert microphone.read(0.001) is None


def test_callback_logs_status_rejects_bad_data_and_ignores_cancelled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Callback status and invalid buffers are recoverable and cancelled input is discarded."""
    microphone = SoundDeviceMicrophone(queue_blocks=3)
    microphone._callback(object(), 1, None, "overflow")
    microphone.cancel()
    microphone._callback(pcm(12), 1, None, None)
    assert microphone.read(0.01) is None
    assert "device status" in caplog.text
    assert "invalid PCM block" in caplog.text


@pytest.mark.parametrize("failure_stage", ["construct", "start"])
def test_microphone_open_failures_are_wrapped_and_partially_created_stream_is_closed(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    """Device occupation at construction/start becomes RuntimeError with cleanup when possible."""
    events: list[str] = []

    class Stream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("construct")
            if failure_stage == "construct":
                raise OSError("device occupied")

        def start(self) -> None:
            events.append("start")
            raise OSError("device occupied")

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(RawInputStream=Stream),
    )
    microphone = SoundDeviceMicrophone()
    with pytest.raises(RuntimeError, match="microphone could not be opened"):
        microphone.start()
    assert microphone._stream is None
    assert microphone._started is False
    assert events == (["construct"] if failure_stage == "construct" else ["construct", "start", "close"])


def test_microphone_missing_dependency_and_close_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing driver is wrapped; close wakes reads, is idempotent, and forbids reuse."""
    def missing(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    microphone = SoundDeviceMicrophone()
    with pytest.raises(RuntimeError, match="microphone could not be opened"):
        microphone.start()
    microphone.close()
    microphone.close()
    assert microphone.closed is True
    with pytest.raises(RuntimeError, match="microphone source is closed"):
        microphone.start()
    with pytest.raises(RuntimeError, match="microphone source is closed"):
        microphone.read(0.01)


@pytest.mark.parametrize(
    ("timeout", "error"),
    [(True, TypeError), ("1", TypeError), (0, ValueError), (-1, ValueError), (float("inf"), ValueError)],
)
def test_microphone_read_validates_timeout(timeout: object, error: type[Exception]) -> None:
    """Read accepts only positive finite numeric timeout values."""
    with pytest.raises(error):
        SoundDeviceMicrophone().read(timeout)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('{"text": "  你好 世界  "}', "你好 世界"),
        ('{"text": "\\t\\n"}', None),
        ('{"partial": "你好"}', None),
        ('{"text": 7}', None),
        ('["not", "an", "object"]', None),
        ("not-json", None),
        (7, None),
    ],
)
def test_result_text_handles_final_partial_blank_bad_json_and_wrong_types(
    payload: Any, expected: str | None
) -> None:
    """Only a nonblank string `text` field is accepted from Vosk JSON."""
    assert _result_text(payload) == expected  # type: ignore[arg-type]


def test_result_text_preserves_unicode_and_long_text() -> None:
    """Valid UTF-8 JSON has no arbitrary transcript truncation or character loss."""
    transcript = "普通话识别" * 10_000
    assert _result_text(json.dumps({"text": transcript}, ensure_ascii=False)) == transcript


def test_vosk_adapter_emits_only_accepted_results_and_final_result() -> None:
    """The adapter maps Vosk's uppercase streaming API without exposing it upstream."""
    class RawRecognizer:
        def __init__(self) -> None:
            self.accepted = [False, True]
            self.blocks: list[bytes] = []

        def AcceptWaveform(self, block: bytes) -> bool:
            self.blocks.append(block)
            return self.accepted.pop(0)

        def Result(self) -> str:
            return '{"text": "片段"}'

        def FinalResult(self) -> str:
            return '{"text": "最终"}'

    raw = RawRecognizer()
    adapted = _VoskRecognizer(raw)
    assert adapted.accept_waveform(pcm(1)) is None
    assert adapted.accept_waveform(pcm(2)) == '{"text": "片段"}'
    assert adapted.final_result() == '{"text": "最终"}'
    assert raw.blocks == [pcm(1), pcm(2)]


def test_lazy_vosk_factory_imports_on_call_caches_model_and_uses_requested_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Vosk import/model creation is deferred, cached, and configured for 16 kHz Mandarin."""
    events: list[tuple[str, object]] = []

    class Model:
        def __init__(self, path: str) -> None:
            events.append(("model", path))

    class Recognizer:
        def __init__(self, model: object, rate: int) -> None:
            events.append(("recognizer", rate))

    module = SimpleNamespace(Model=Model, KaldiRecognizer=Recognizer)
    original = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        if name == "vosk":
            events.append(("import", name))
            return module
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    factory = LazyVoskFactory()
    assert events == []
    model_path = tmp_path / "中文 普通话模型"
    factory(model_path, 16000)
    factory(model_path, 8000)
    assert [event for event in events if event[0] == "model"] == [("model", str(model_path))]
    assert [event for event in events if event[0] == "recognizer"] == [
        ("recognizer", 16000), ("recognizer", 8000)
    ]


def test_lazy_vosk_factory_wraps_import_failure_and_rejects_model_switch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dependency failures are stable RuntimeErrors and a cached factory cannot switch models."""
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )
    with pytest.raises(RuntimeError, match="vosk model or recognizer could not be loaded"):
        LazyVoskFactory()(tmp_path / "missing", 16000)

    module = SimpleNamespace(
        Model=lambda path: object(),
        KaldiRecognizer=lambda model, rate: object(),
    )
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    factory = LazyVoskFactory()
    factory(tmp_path / "first", 16000)
    with pytest.raises(RuntimeError, match="cannot switch model paths"):
        factory(tmp_path / "second", 16000)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"model_path": "model"}, TypeError),
        ({"sample_rate": True}, TypeError),
        ({"sample_rate": 0}, ValueError),
        ({"block_size": 0}, ValueError),
        ({"queue_blocks": 1.5}, TypeError),
        ({"silence_seconds": 0}, ValueError),
        ({"max_utterance_seconds": float("nan")}, ValueError),
        ({"vad_threshold": -0.1}, ValueError),
    ],
)
def test_vosk_input_rejects_invalid_configuration(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    """Path, PCM, VAD, and duration contracts are checked without external imports."""
    model_path = kwargs.pop("model_path", Path("unused"))
    with pytest.raises(error):
        VoskSpeechInput(model_path, **kwargs)  # type: ignore[arg-type]


def test_missing_model_returns_none_without_starting_source_or_factory(tmp_path: Path) -> None:
    """A missing Mandarin model is a recoverable result with no hardware side effect."""
    source = FakeSource([])
    instance, factory = speech_input(tmp_path / "不存在的普通话模型", source, FakeRecognizer())
    assert instance.listen(1.0) is None
    assert source.starts == 0
    assert factory.calls == []


def test_mandarin_path_is_resolved_and_model_and_source_are_lazy(tmp_path: Path) -> None:
    """First listen, not construction, receives the resolved Unicode model directory."""
    model = tmp_path / "模型" / "vosk-cn"
    model.mkdir(parents=True)
    source = FakeSource([pcm(400), pcm(0) * 10])
    recognizer = FakeRecognizer(final='{"text": "你好"}')
    instance, factory = speech_input(
        model,
        source,
        recognizer,
        sample_rate=10,
        silence_seconds=0.5,
        max_utterance_seconds=2,
        vad_threshold=300,
        clock=SequenceClock(0, 0, 0, 0),
    )
    assert source.starts == 0 and factory.calls == []
    assert instance.listen(1.0) == "你好"
    assert factory.calls == [(model.resolve(), 10)]
    assert source.starts == 1


def test_leading_silence_is_not_sent_and_vad_threshold_is_inclusive(tmp_path: Path) -> None:
    """Below-threshold leading blocks are ignored; equality starts speech; later silence terminates."""
    model = tmp_path / "model"
    model.mkdir()
    below = pcm(299, -299)
    equal = pcm(300, -300)
    above = pcm(301, -301)
    silence = pcm(0) * 10
    source = FakeSource([below, equal, above, silence])
    recognizer = FakeRecognizer(final='{"text":"阈值测试"}')
    instance, _ = speech_input(
        model,
        source,
        recognizer,
        sample_rate=10,
        silence_seconds=0.5,
        max_utterance_seconds=3,
        vad_threshold=300,
        clock=SequenceClock(0, 0, 0, 0, 0, 0),
    )
    assert instance.listen(2) == "阈值测试"
    assert recognizer.blocks == [equal, above, silence]


def test_streaming_final_bad_json_blank_and_silence_termination(tmp_path: Path) -> None:
    """Accepted segments join final text; malformed/blank results do not poison capture."""
    model = tmp_path / "model"
    model.mkdir()
    voiced = pcm(500) * 2
    quiet = pcm(0) * 2
    source = FakeSource([voiced, voiced, voiced, quiet, quiet])
    recognizer = FakeRecognizer(
        accepted=['{"text":"第一段"}', "bad-json", '{"text":"   "}', None, None],
        final='{"text":"收尾"}',
    )
    instance, _ = speech_input(
        model,
        source,
        recognizer,
        sample_rate=10,
        silence_seconds=0.4,
        max_utterance_seconds=5,
        vad_threshold=300,
        clock=SequenceClock(0, 0, 0, 0, 0, 0, 0),
    )
    assert instance.listen(2) == "第一段 收尾"
    assert len(recognizer.blocks) == 5
    assert recognizer.final_calls == 1


def test_max_utterance_stops_on_pcm_duration_without_wall_clock_sleep(tmp_path: Path) -> None:
    """Max utterance is enforced by audio duration even when every block remains voiced."""
    model = tmp_path / "model"
    model.mkdir()
    block = pcm(500) * 4  # 0.4 seconds at 10 samples/second.
    source = FakeSource([block, block, block, block])
    recognizer = FakeRecognizer(final='{"text":"达到上限"}')
    instance, _ = speech_input(
        model,
        source,
        recognizer,
        sample_rate=10,
        silence_seconds=1,
        max_utterance_seconds=0.8,
        vad_threshold=300,
        clock=SequenceClock(0, 0, 0, 0),
    )
    assert instance.listen(2) == "达到上限"
    assert recognizer.blocks == [block, block]
    assert len(source.items) == 2


def test_timeout_returns_none_and_bounds_each_source_read(tmp_path: Path) -> None:
    """Wall-clock timeout ends leading silence and each blocking read is capped at 250 ms."""
    model = tmp_path / "model"
    model.mkdir()
    source = FakeSource([None, None])
    recognizer = FakeRecognizer(final='{"text":"must not finalize"}')
    instance, _ = speech_input(
        model,
        source,
        recognizer,
        clock=SequenceClock(10.0, 10.0, 10.2, 10.6),
    )
    assert instance.listen(0.5) is None
    assert source.read_timeouts == [0.25, pytest.approx(0.25)]
    assert recognizer.final_calls == 0


class BlockingSource(FakeSource):
    """Block one read until cancel while preserving its queued PCM for a later listen."""

    def __init__(self, block: bytes) -> None:
        super().__init__([block])
        self.entered = Event()
        self.wake = Event()
        self.block_once = True

    def start(self) -> None:
        super().start()
        if self.starts > 1:
            self.wake.clear()

    def read(self, timeout: float) -> bytes | None:
        self.read_timeouts.append(timeout)
        if self.block_once:
            self.block_once = False
            self.entered.set()
            self.wake.wait(1.0)
            return None
        return super().read(timeout)

    def cancel(self) -> None:
        super().cancel()
        self.wake.set()


def test_cancel_wakes_current_listen_without_consuming_next_audio(tmp_path: Path) -> None:
    """Cancel affects one capture; a subsequent listen can consume the untouched PCM."""
    model = tmp_path / "model"
    model.mkdir()
    source = BlockingSource(pcm(500) * 2)
    recognizer = FakeRecognizer(final='{"text":"保留的语音"}')
    instance, _ = speech_input(
        model,
        source,
        recognizer,
        sample_rate=10,
        max_utterance_seconds=0.2,
        clock=lambda: 0.0,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(instance.listen, 5.0)
        assert source.entered.wait(1.0), "fake source did not enter blocking read"
        instance.cancel()
        assert pending.result(timeout=1.0) is None
    assert source.items == [pcm(500) * 2]
    assert instance.listen(5.0) == "保留的语音"


def test_close_wakes_listener_is_idempotent_and_forbids_future_listen(tmp_path: Path) -> None:
    """Close interrupts capture, closes the source once, and permanently rejects listen."""
    model = tmp_path / "model"
    model.mkdir()
    source = BlockingSource(pcm(500))
    instance, _ = speech_input(model, source, FakeRecognizer(), clock=lambda: 0.0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(instance.listen, 5.0)
        assert source.entered.wait(1.0), "fake source did not enter blocking read"
        instance.close()
        assert pending.result(timeout=1.0) is None
    instance.close()
    assert instance.closed is True
    assert source.cancels == 1
    assert source.closes == 1
    with pytest.raises(RuntimeError, match="vosk speech input is closed"):
        instance.listen(1.0)


@pytest.mark.parametrize(
    ("source_items", "factory_value"),
    [
        ([OSError("device occupied")], FakeRecognizer()),
        ([pcm(500)], RuntimeError("recognizer unavailable")),
        ([pcm(500)], FakeRecognizer(accepted=[RuntimeError("accept failed")])),
        ([pcm(500)], FakeRecognizer(final=RuntimeError("final failed"))),
        ([object()], FakeRecognizer()),
    ],
)
def test_source_recognizer_and_bad_pcm_failures_return_none(
    tmp_path: Path,
    source_items: list[Any],
    factory_value: FakeRecognizer | Exception,
) -> None:
    """Device, factory, recognizer, and malformed-source failures remain recoverable."""
    model = tmp_path / "model"
    model.mkdir()
    source = FakeSource(source_items)  # type: ignore[arg-type]
    instance, _ = speech_input(
        model,
        source,
        factory_value,
        sample_rate=10,
        max_utterance_seconds=0.1,
        clock=lambda: 0.0,
    )
    assert instance.listen(1.0) is None


def test_task_modules_have_no_static_sounddevice_or_vosk_imports() -> None:
    """The implementation only names concrete dependencies through dynamic import calls."""
    project_root = Path(__file__).resolve().parents[1]
    imported_roots: set[str] = set()
    for relative_path in ("src/audio/microphone.py", "src/audio/vosk_asr.py"):
        tree = ast.parse((project_root / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint({"sounddevice", "vosk"})


def test_fresh_interpreter_imports_task_modules_without_audio_dependencies() -> None:
    """Module import succeeds when sounddevice and vosk imports are actively rejected."""
    project_root = Path(__file__).resolve().parents[1]
    blocker = """
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'sounddevice', 'vosk'}:
            raise RuntimeError(f'forbidden dependency requested: {fullname}')
        return None

sys.meta_path.insert(0, Blocker())
import src.audio.microphone
import src.audio.vosk_asr
print('t15-import-ok')
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
    assert result.stdout.strip() == "t15-import-ok"
