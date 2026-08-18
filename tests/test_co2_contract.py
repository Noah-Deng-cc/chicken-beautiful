"""T10 CO2 abstraction and deterministic mock acceptance tests.

Inputs: ppm values, configured thresholds, mock source outcomes, and clocks.
Outputs: assertions for CO2 classification, lifecycle, concurrency, and Pi-safe imports.
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

import src.co2 as co2
from src.co2 import Co2Sensor, Co2Thresholds, MockCo2Sensor, classify_co2
from src.domain import Co2Level, Co2Reading


BASE_TIME = datetime(2026, 8, 18, 10, 30, tzinfo=timezone.utc)
FORBIDDEN_DEPENDENCIES = {"adafruit", "board", "busio", "serial", "smbus", "smbus2"}


def thresholds() -> Co2Thresholds:
    """Create stable valid classification boundaries for each test."""
    return Co2Thresholds(elevated=800, poor=1_500)


def test_public_api_abstract_contract_and_signatures() -> None:
    """The T10 package remains hardware-neutral and exposes its documented contract."""
    assert set(co2.__all__) == {"Co2Sensor", "Co2Thresholds", "MockCo2Sensor", "classify_co2"}
    assert inspect.isabstract(Co2Sensor)
    assert Co2Sensor.__abstractmethods__ == {"closed", "read", "close"}
    with pytest.raises(TypeError, match="abstract"):
        Co2Sensor()  # type: ignore[abstract]
    assert list(inspect.signature(Co2Sensor.read).parameters) == ["self"]
    assert list(inspect.signature(Co2Sensor.close).parameters) == ["self"]
    assert get_type_hints(Co2Sensor.read)["return"] == Co2Reading | None
    assert get_type_hints(Co2Sensor.close)["return"] is type(None)
    assert get_type_hints(classify_co2)["return"] is Co2Level


@pytest.mark.parametrize(
    ("ppm", "expected"),
    [
        (0, Co2Level.GOOD),
        (799, Co2Level.GOOD),
        (800, Co2Level.ELEVATED),
        (1_499, Co2Level.ELEVATED),
        (1_500, Co2Level.POOR),
        (100_000, Co2Level.POOR),
    ],
)
def test_classification_uses_configured_strict_boundaries(ppm: int, expected: Co2Level) -> None:
    """Every configured boundary deterministically maps to good, elevated, or poor."""
    assert classify_co2(ppm, thresholds()) is expected


@pytest.mark.parametrize("ppm", [-1, 100_001, True, False, 800.0, "800", None])
def test_invalid_ppm_never_becomes_a_valid_level(ppm: object) -> None:
    """Negative, too-large, boolean, and non-integer values always become invalid."""
    assert classify_co2(ppm, thresholds()) is Co2Level.INVALID  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("elevated", "poor", "error"),
    [
        (True, 1_500, TypeError),
        (800.0, 1_500, TypeError),
        ("800", 1_500, TypeError),
        (-1, 1_500, ValueError),
        (800, 800, ValueError),
        (1_500, 800, ValueError),
        (800, 100_001, ValueError),
    ],
)
def test_invalid_threshold_configurations_are_rejected(
    elevated: object, poor: object, error: type[Exception],
) -> None:
    """Configuration errors fail before a sensor is constructed or read."""
    with pytest.raises(error):
        Co2Thresholds(elevated=elevated, poor=poor)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="thresholds"):
        classify_co2(800, object())  # type: ignore[arg-type]


def test_fixed_sequence_exhaustion_exception_and_disconnect_sources(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mocks repeat fixed input, consume sequences, and safely model failures."""
    fixed = MockCo2Sensor(800, thresholds=thresholds(), clock=lambda: BASE_TIME)
    assert fixed.read() == Co2Reading(BASE_TIME, 800, Co2Level.ELEVATED)
    assert fixed.read() == Co2Reading(BASE_TIME, 800, Co2Level.ELEVATED)

    sensor = MockCo2Sensor(
        [799, 1_500, RuntimeError("sensor disconnected"), None, 100_001],
        thresholds=thresholds(),
        clock=lambda: BASE_TIME,
    )
    assert sensor.read() == Co2Reading(BASE_TIME, 799, Co2Level.GOOD)
    assert sensor.read() == Co2Reading(BASE_TIME, 1_500, Co2Level.POOR)
    with caplog.at_level("WARNING", logger="src.co2.mock"):
        assert sensor.read() is None
        assert sensor.read() is None
        invalid = sensor.read()
    assert invalid == Co2Reading(BASE_TIME, None, Co2Level.INVALID)
    assert "injected error" in caplog.text
    assert "disconnected" in caplog.text
    assert sensor.read() is None


@pytest.mark.parametrize("value", [-1, 100_001, True, 800.0, "800"])
def test_mock_preserves_invalid_values_as_invalid_readings(value: object) -> None:
    """Malformed non-disconnect source values produce a timestamped invalid reading."""
    sensor = MockCo2Sensor(value, thresholds=thresholds(), clock=lambda: BASE_TIME)
    assert sensor.read() == Co2Reading(BASE_TIME, None, Co2Level.INVALID)


def test_clock_must_be_callable_and_bad_clock_or_timezone_returns_none() -> None:
    """Clock misuse cannot leak a non-domain reading through the hardware-neutral mock."""
    with pytest.raises(TypeError, match="clock"):
        MockCo2Sensor(800, thresholds=thresholds(), clock=BASE_TIME)  # type: ignore[arg-type]
    naive = MockCo2Sensor(800, thresholds=thresholds(), clock=lambda: BASE_TIME.replace(tzinfo=None))
    failing = MockCo2Sensor(800, thresholds=thresholds(), clock=lambda: (_ for _ in ()).throw(OSError("clock failed")))
    assert naive.read() is None
    assert failing.read() is None


def test_close_context_management_and_concurrent_sequence_consumption() -> None:
    """Closing is idempotent, contexts release resources, and sequence polling is atomic."""
    sensor = MockCo2Sensor(range(200), thresholds=thresholds(), clock=lambda: BASE_TIME)
    with ThreadPoolExecutor(max_workers=16) as executor:
        actual = list(executor.map(lambda _: sensor.read(), range(200)))
    ppm_values = [reading.ppm for reading in actual if reading is not None]
    assert len(ppm_values) == 200
    assert set(ppm_values) == set(range(200))
    assert sensor.read() is None

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda _: sensor.close(), range(64)))
    assert sensor.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        sensor.read()
    with pytest.raises(RuntimeError, match="closed"):
        sensor.__enter__()

    managed = MockCo2Sensor(800, thresholds=thresholds(), clock=lambda: BASE_TIME)
    with pytest.raises(LookupError, match="body failure"):
        with managed as active:
            assert active is managed
            assert active.read() is not None
            raise LookupError("body failure")
    assert managed.closed is True


def test_co2_import_has_no_uart_or_i2c_dependency() -> None:
    """T10 imports on Zero 2 W before any UART/I2C driver package is selected."""
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
import src.co2
print('co2-import-ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", blocker], cwd=project_root, text=True,
        capture_output=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "co2-import-ok"
