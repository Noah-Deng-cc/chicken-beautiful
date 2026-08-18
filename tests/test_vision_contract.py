"""T06 vision abstraction and deterministic mock acceptance tests.

Inputs: public vision interfaces, domain emotion readings, and injected failures.
Outputs: assertions for type, lifecycle, concurrency, and dependency contracts.
Dependencies: pytest and the Python standard library only.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import inspect
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints

import pytest

import src.vision as vision
from src.domain import Emotion, EmotionReading
from src.vision import MockResult, MockVisionPipeline, VisionPipeline


BASE_TIME = datetime(2026, 8, 18, 9, 30, tzinfo=timezone.utc)
FORBIDDEN_DEPENDENCIES = {"cv2", "numpy", "torch", "ultralytics"}


def make_reading(index: int = 0) -> EmotionReading:
    """Build a unique valid emotion reading for deterministic assertions."""
    return EmotionReading(
        timestamp=BASE_TIME,
        dominant=Emotion.HAPPY,
        confidence=0.8,
        valence=0.6,
        arousal=0.2,
        person_id=f"resident-{index}",
    )


def test_public_api_and_abstract_contract() -> None:
    """The package exports the specified API and the base remains abstract."""
    assert set(vision.__all__) == {"MockResult", "MockVisionPipeline", "VisionPipeline"}
    assert vision.MockResult is MockResult
    assert inspect.isabstract(VisionPipeline)
    assert VisionPipeline.__abstractmethods__ == {"start", "read", "close"}
    with pytest.raises(TypeError, match="abstract"):
        VisionPipeline()  # type: ignore[abstract]


def test_method_signatures_and_emotion_reading_type_contract() -> None:
    """Required methods have no extra public arguments and return domain values."""
    assert list(inspect.signature(VisionPipeline.start).parameters) == ["self"]
    assert list(inspect.signature(VisionPipeline.read).parameters) == ["self"]
    assert list(inspect.signature(VisionPipeline.close).parameters) == ["self"]
    assert get_type_hints(VisionPipeline.start)["return"] is type(None)
    assert get_type_hints(VisionPipeline.read)["return"] == EmotionReading | None
    assert get_type_hints(VisionPipeline.close)["return"] is type(None)

    reading = make_reading()
    pipeline = MockVisionPipeline(reading)
    pipeline.start()
    actual = pipeline.read()
    assert actual is reading
    assert isinstance(actual, EmotionReading)


@pytest.mark.parametrize("fixed", [make_reading(), None])
def test_fixed_reading_and_none_are_repeatable(fixed: EmotionReading | None) -> None:
    """A fixed result is returned identically on every active read."""
    pipeline = MockVisionPipeline(fixed)
    pipeline.start()
    assert [pipeline.read(), pipeline.read(), pipeline.read()] == [fixed, fixed, fixed]
    assert pipeline.read_count == 3


def test_finite_sequence_preserves_order_then_exhausts() -> None:
    """A finite sequence emits each value once and returns None after exhaustion."""
    first, second = make_reading(1), make_reading(2)
    pipeline = MockVisionPipeline([first, None, second])
    pipeline.start()
    assert [pipeline.read() for _ in range(5)] == [first, None, second, None, None]
    assert pipeline.read_count == 5


def test_repeating_sequence_wraps_without_reordering() -> None:
    """A repeating sequence loops at its exact boundary."""
    first, second = make_reading(1), make_reading(2)
    pipeline = MockVisionPipeline((first, second), repeat=True)
    pipeline.start()
    assert [pipeline.read() for _ in range(5)] == [first, second, first, second, first]


def test_injected_exception_becomes_none_and_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """A recoverable injected exception never escapes read and records its cause."""
    pipeline = MockVisionPipeline([RuntimeError("camera disconnected"), make_reading(1)])
    pipeline.start()
    with caplog.at_level("ERROR", logger="src.vision.mock"):
        assert pipeline.read() is None
    assert "camera disconnected" in caplog.text
    assert pipeline.read() == make_reading(1)


@pytest.mark.parametrize(
    ("readings", "repeat", "error", "message"),
    [
        (123, False, TypeError, "readings must be"),
        ("happy", False, TypeError, "readings must be"),
        ([make_reading(), object()], False, TypeError, "each reading must be"),
        ([False], False, TypeError, "each reading must be"),
        ([], True, ValueError, "empty sequence cannot repeat"),
        ([make_reading()], 1, TypeError, "repeat must be a boolean"),
    ],
)
def test_invalid_constructor_inputs_are_rejected(
    readings: object,
    repeat: object,
    error: type[Exception],
    message: str,
) -> None:
    """Wrong direct values, sequence elements, and repeat flags fail explicitly."""
    with pytest.raises(error, match=message):
        MockVisionPipeline(readings, repeat=repeat)  # type: ignore[arg-type]


def test_read_before_start_is_side_effect_free() -> None:
    """Reading before start yields None without consuming the configured sequence."""
    reading = make_reading()
    pipeline = MockVisionPipeline([reading])
    assert pipeline.started is False
    assert pipeline.closed is False
    assert pipeline.read() is None
    assert pipeline.read_count == 0
    pipeline.start()
    assert pipeline.read() is reading


def test_repeated_start_is_idempotent_and_does_not_reset_sequence() -> None:
    """Calling start twice leaves lifecycle and sequence position intact."""
    first, second = make_reading(1), make_reading(2)
    pipeline = MockVisionPipeline([first, second])
    pipeline.start()
    assert pipeline.read() is first
    pipeline.start()
    assert pipeline.started is True
    assert pipeline.read() is second
    assert pipeline.read_count == 2


def test_close_is_idempotent_and_permanently_disables_reads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Repeated close is safe and a closed pipeline cannot restart or consume."""
    pipeline = MockVisionPipeline(make_reading())
    pipeline.start()
    pipeline.close()
    pipeline.close()
    assert pipeline.closed is True
    assert pipeline.started is False
    assert pipeline.read() is None
    assert pipeline.read_count == 0
    with caplog.at_level("WARNING", logger="src.vision.mock"):
        pipeline.start()
    assert pipeline.started is False
    assert "不能重新启动" in caplog.text


def test_context_manager_closes_and_does_not_suppress_body_exception() -> None:
    """Exceptional context exit closes resources and propagates the original error."""
    pipeline = MockVisionPipeline(make_reading())
    with pytest.raises(LookupError, match="body failure"):
        with pipeline as active:
            assert active is pipeline
            assert active.started is True
            raise LookupError("body failure")
    assert pipeline.closed is True
    assert pipeline.started is False


def test_concurrent_consumers_have_no_duplicates_or_omissions() -> None:
    """The sequence cursor is atomic across concurrent readers."""
    readings = [make_reading(index) for index in range(200)]
    pipeline = MockVisionPipeline(readings)
    pipeline.start()
    with ThreadPoolExecutor(max_workers=16) as executor:
        actual = list(executor.map(lambda _: pipeline.read(), range(len(readings))))

    assert None not in actual
    assert len(actual) == len(readings)
    assert len({id(item) for item in actual}) == len(readings)
    assert {item.person_id for item in actual if item is not None} == {
        item.person_id for item in readings
    }
    assert pipeline.read_count == len(readings)
    assert pipeline.read() is None


def test_import_in_fresh_interpreter_does_not_request_heavy_dependencies() -> None:
    """Importing the vision package never reaches Pi-inappropriate ML modules."""
    project_root = Path(__file__).resolve().parents[1]
    blocker = """
import importlib.abc
import sys

blocked = {"cv2", "numpy", "torch", "ultralytics"}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise RuntimeError(f"forbidden dependency requested: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import src.vision
print("vision-import-ok")
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
    assert result.stdout.strip() == "vision-import-ok"
