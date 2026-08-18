"""T08 thermal abstraction and deterministic mock acceptance tests.

Inputs: thermal frames, domain readings, and injected mock outcomes.
Outputs: assertions for frame validity, lifecycle, concurrency, and dependency isolation.
Dependencies: pytest and the Python standard library only.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timezone
import inspect
from pathlib import Path
import subprocess
import sys
from threading import Event
from typing import get_type_hints

import pytest

import src.thermal as thermal
from src.domain import TemperatureReading
from src.thermal import MockThermalSensor, ThermalSensor, summarize_temperature_frame, temperature_from_frame


BASE_TIME = datetime(2026, 8, 18, 10, 0, tzinfo=timezone.utc)
FORBIDDEN_DEPENDENCIES = {"adafruit_mlx90640", "board", "busio", "numpy", "smbus", "smbus2"}


def make_reading(index: int = 0) -> TemperatureReading:
    """Build a unique valid temperature reading for deterministic assertions."""
    return TemperatureReading(BASE_TIME, 36.5 + index / 1000, 36.0 + index / 1000, "good")


def test_public_api_abstract_contract_and_signatures() -> None:
    """The package exposes only the hardware-neutral T08 API with exact types."""
    assert set(thermal.__all__) == {
        "Clock", "MockThermalItem", "MockThermalSensor", "ThermalSensor",
        "summarize_temperature_frame", "temperature_from_frame",
    }
    assert inspect.isabstract(ThermalSensor)
    assert ThermalSensor.__abstractmethods__ == {"read", "close"}
    with pytest.raises(TypeError, match="abstract"):
        ThermalSensor()  # type: ignore[abstract]
    assert list(inspect.signature(ThermalSensor.read).parameters) == ["self"]
    assert list(inspect.signature(ThermalSensor.close).parameters) == ["self"]
    assert get_type_hints(ThermalSensor.read)["return"] == TemperatureReading | None
    assert get_type_hints(ThermalSensor.close)["return"] is type(None)


def test_flat_and_nested_frames_produce_maximum_average_timestamp_and_quality() -> None:
    """Normal and recursively nested arrays preserve all pixels in max/mean output."""
    assert summarize_temperature_frame([36.0, 37.5, 38.0]) == (38.0, 37.166666666666664)
    reading = temperature_from_frame([[36.0, (37.5,)], [38.0]], BASE_TIME)
    assert reading == TemperatureReading(BASE_TIME, 38.0, 37.166666666666664, "good")
    assert reading is not None
    assert reading.timestamp is BASE_TIME
    assert reading.quality == "good"


@pytest.mark.parametrize(
    "frame",
    [[], [36.0, float("nan")], [36.0, float("inf")], [36.0, "bad"], [36.0, True], [19.9], [45.1]],
)
def test_empty_nonfinite_nonnumeric_and_out_of_range_pixels_are_invalid(frame: object) -> None:
    """Every bad pixel invalidates the entire frame rather than leaking a partial reading."""
    assert summarize_temperature_frame(frame) is None  # type: ignore[arg-type]
    assert temperature_from_frame(frame, BASE_TIME) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("lower", "upper", "error"),
    [(True, 45.0, TypeError), (20.0, float("nan"), ValueError), (46.0, 45.0, ValueError), (-41.0, 45.0, ValueError)],
)
def test_invalid_temperature_bounds_are_rejected_before_frame_processing(
    lower: object, upper: object, error: type[Exception],
) -> None:
    """Invalid calibration bounds are explicit configuration errors."""
    with pytest.raises(error):
        summarize_temperature_frame([36.0], lower, upper)  # type: ignore[arg-type]


def test_fixed_frame_fixed_reading_and_sequence_outcomes_are_deterministic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fixed values repeat while sequence frames, failures, and exhaustion consume in order."""
    fixed = MockThermalSensor([36.0, 37.0], clock=lambda: BASE_TIME)
    assert fixed.read() == TemperatureReading(BASE_TIME, 37.0, 36.5, "good")
    assert fixed.read() == TemperatureReading(BASE_TIME, 37.0, 36.5, "good")
    original = make_reading()
    sensor = MockThermalSensor([original, [38.0, 39.0], RuntimeError("i2c disconnected"), None], clock=lambda: BASE_TIME)
    assert sensor.read() is original
    assert sensor.read() == TemperatureReading(BASE_TIME, 39.0, 38.5, "good")
    with caplog.at_level("ERROR", logger="src.thermal.mock"):
        assert sensor.read() is None
    assert "i2c disconnected" in caplog.text
    assert sensor.read() is None
    assert sensor.read() is None


def test_repeat_lifecycle_context_management_and_closed_reads() -> None:
    """Repeating sequences loop; close is idempotent and context managers always release."""
    sensor = MockThermalSensor([[30.0], [31.0]], repeat=True, clock=lambda: BASE_TIME)
    values = [sensor.read(), sensor.read(), sensor.read()]
    assert [item.maximum_celsius for item in values if item is not None] == [30.0, 31.0, 30.0]
    sensor.close()
    sensor.close()
    assert sensor.closed is True
    assert sensor.read() is None
    with pytest.raises(RuntimeError, match="closed"):
        sensor.__enter__()
    with pytest.raises(LookupError, match="body failure"):
        with MockThermalSensor([36.0]) as active:
            assert active.read() is not None
            raise LookupError("body failure")
    assert active.closed is True


def test_concurrent_sequence_consumers_do_not_duplicate_or_omit_items() -> None:
    """The mock cursor remains atomic under concurrent polling clients."""
    frames = [[30.0 + index / 100, 31.0 + index / 100] for index in range(200)]
    sensor = MockThermalSensor(frames, clock=lambda: BASE_TIME)
    with ThreadPoolExecutor(max_workers=16) as executor:
        actual = list(executor.map(lambda _: sensor.read(), range(len(frames))))
    assert None not in actual
    actual_maxima = [item.maximum_celsius for item in actual if item is not None]
    expected_maxima = [31.0 + index / 100 for index in range(200)]
    assert sorted(actual_maxima) == sorted(expected_maxima)
    assert Counter(actual_maxima) == Counter(expected_maxima)
    assert sensor.read_count == len(frames)
    assert sensor.read() is None


def test_concurrent_map_positions_can_swap_while_each_reading_is_consumed_once() -> None:
    """Event barriers prove map position does not define the sensor-lock acquisition order."""
    original_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(25):
            first_worker_waiting = Event()
            second_worker_read = Event()
            sensor = MockThermalSensor([[30.0], [31.0]], clock=lambda: BASE_TIME)

            def consume(worker_index: int) -> TemperatureReading | None:
                """Force worker one to claim the first item before worker zero continues."""
                if worker_index == 0:
                    first_worker_waiting.set()
                    assert second_worker_read.wait(1.0), "second worker did not consume first item"
                    return sensor.read()
                assert first_worker_waiting.wait(1.0), "first worker did not reach barrier"
                result = sensor.read()
                second_worker_read.set()
                return result

            with ThreadPoolExecutor(max_workers=2) as executor:
                actual = list(executor.map(consume, range(2)))

            assert [item.maximum_celsius for item in actual if item is not None] == [31.0, 30.0]
            assert sorted(item.maximum_celsius for item in actual if item is not None) == [30.0, 31.0]
            assert sensor.read_count == 2
            assert sensor.read() is None
    finally:
        sys.setswitchinterval(original_interval)


def test_thermal_package_imports_without_vendor_or_heavy_dependencies() -> None:
    """The T08 interface stays importable on Zero 2 W without a selected sensor SDK."""
    project_root = Path(__file__).resolve().parents[1]
    blocker = f"""
import importlib.abc
import sys

blocked = {FORBIDDEN_DEPENDENCIES!r}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in blocked:
            raise RuntimeError(f'forbidden dependency requested: {{fullname}}')
        return None
sys.meta_path.insert(0, Blocker())
import src.thermal
print('thermal-import-ok')
"""
    result = subprocess.run([sys.executable, "-c", blocker], cwd=project_root, text=True, capture_output=True, check=False, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "thermal-import-ok"
