"""T09 MLX90640 adapter acceptance tests without physical I2C hardware."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import pytest

from src.domain import TemperatureReading
from src.thermal import ThermalSensor
from src.thermal import mlx90640
from src.thermal.mlx90640 import Mlx90640Sensor


BASE_TIME = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def test_adapter_implements_thermal_contract_and_import_is_hardware_free() -> None:
    """Loading the adapter neither needs nor attempts vendor packages."""
    assert isinstance(Mlx90640Sensor(frame_reader=lambda: [36.0]), ThermalSensor)
    root = Path(__file__).resolve().parents[1]
    blocker = """
import importlib.abc
import sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'board', 'busio', 'adafruit_mlx90640'}:
            raise RuntimeError('vendor import at module load: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())
import src.thermal.mlx90640
print('mlx-import-ok')
"""
    result = subprocess.run([sys.executable, "-c", blocker], cwd=root, text=True, capture_output=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "mlx-import-ok"


def test_reader_calibrates_emissivity_and_offset_before_summary() -> None:
    """Emissivity compensation and operator offset affect max and mean consistently."""
    sensor = Mlx90640Sensor(
        emissivity=0.8, offset_celsius=0.5, frame_reader=lambda: [28.0, 29.6],
        clock=lambda: BASE_TIME, min_valid_celsius=20.0, max_valid_celsius=45.0,
    )
    assert sensor.read() == TemperatureReading(BASE_TIME, 37.5, 36.5, "good")


def test_i2c_reader_retries_then_recovers_with_injected_delay() -> None:
    """A transient reader exception consumes one retry and later returns a genuine reading."""
    results: deque[object] = deque([OSError("i2c busy"), [36.0, 37.0]])
    sleeps: list[float] = []

    def reader() -> object:
        item = results.popleft()
        if isinstance(item, BaseException):
            raise item
        return item

    sensor = Mlx90640Sensor(frame_reader=reader, retries=2, retry_delay_seconds=0.25, sleep=sleeps.append, clock=lambda: BASE_TIME)
    assert sensor.read() == TemperatureReading(
        BASE_TIME, 37.0 / 0.95, (36.0 / 0.95 + 37.0 / 0.95) / 2, "good",
    )
    assert sleeps == [0.25]


def test_i2c_reader_exhaustion_returns_none_and_logs_diagnostic(caplog: pytest.LogCaptureFixture) -> None:
    """Persistent bus errors are bounded and never escape into the service loop."""
    calls = 0

    def reader() -> list[float]:
        nonlocal calls
        calls += 1
        raise OSError("i2c disconnected")

    with caplog.at_level("WARNING", logger="src.thermal.mlx90640"):
        sensor = Mlx90640Sensor(frame_reader=reader, retries=2, retry_delay_seconds=0, sleep=lambda _: None)
        assert sensor.read() is None
    assert calls == 3
    assert "MLX90640 I2C read failed after retries" in caplog.text


def test_missing_vendor_dependency_degrades_only_read_and_is_diagnostic(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """An uninstalled optional SDK is contained at first read rather than module import."""
    requested: list[str] = []

    def missing(name: str) -> object:
        requested.append(name)
        raise ModuleNotFoundError("No module named " + name)

    monkeypatch.setattr(mlx90640.importlib, "import_module", missing)
    sensor = Mlx90640Sensor()
    with caplog.at_level("WARNING", logger="src.thermal.mlx90640"):
        assert sensor.read() is None
    assert requested == ["board"]
    assert "MLX90640 driver requires board, busio, and adafruit_mlx90640" in caplog.text


def test_fake_vendor_i2c_is_initialized_lazily_and_read_afterwards_uses_cached_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The physical path can be exercised through fake board, busio and MLX modules."""
    imports: list[str] = []
    i2c_calls: list[tuple[object, object]] = []
    created: list[object] = []

    class Board:
        SCL = "SCL"
        SDA = "SDA"

    class Busio:
        @staticmethod
        def I2C(scl: object, sda: object) -> object:
            i2c_calls.append((scl, sda))
            return "i2c"

    class Device:
        def __init__(self, i2c: object, *, address: int) -> None:
            assert i2c == "i2c"
            assert address == 0x33
            self.refresh_rate: object | None = None
            created.append(self)

        def getFrame(self, frame: list[float]) -> None:
            frame[:] = [36.0] * 768

    class Module:
        MLX90640 = Device

        class RefreshRate:
            REFRESH_2_HZ = "2hz"

    modules = {"board": Board, "busio": Busio, "adafruit_mlx90640": Module}

    def importer(name: str) -> object:
        imports.append(name)
        return modules[name]

    monkeypatch.setattr(mlx90640.importlib, "import_module", importer)
    sensor = Mlx90640Sensor(clock=lambda: BASE_TIME)
    assert sensor.read() == TemperatureReading(BASE_TIME, 36.0 / 0.95, 36.0 / 0.95, "good")
    assert sensor.read() == TemperatureReading(BASE_TIME, 36.0 / 0.95, 36.0 / 0.95, "good")
    assert imports == ["board", "busio", "adafruit_mlx90640"]
    assert i2c_calls == [("SCL", "SDA")]
    assert created[0].refresh_rate == "2hz"


def test_close_is_idempotent_and_prevents_reader_access() -> None:
    """Shutting down releases the polling path; repeated close remains harmless."""
    calls = 0

    def reader() -> list[float]:
        nonlocal calls
        calls += 1
        return [36.0]

    sensor = Mlx90640Sensor(frame_reader=reader)
    sensor.close()
    sensor.close()
    assert sensor.closed is True
    assert sensor.read() is None
    assert calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        sensor.__enter__()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"bus": False}, TypeError), ({"address": 0x02}, ValueError), ({"retries": -1}, ValueError),
        ({"emissivity": 0.09}, ValueError), ({"emissivity": float("nan")}, ValueError),
        ({"offset_celsius": 20.1}, ValueError), ({"retry_delay_seconds": -0.01}, ValueError),
        ({"min_valid_celsius": 46.0, "max_valid_celsius": 45.0}, ValueError),
    ],
)
def test_constructor_rejects_invalid_transport_calibration_and_temperature_bounds(
    kwargs: dict[str, object], error: type[Exception],
) -> None:
    """Invalid wiring and calibration settings fail early with configuration errors."""
    with pytest.raises(error):
        Mlx90640Sensor(**kwargs)  # type: ignore[arg-type]


def test_calibration_template_documents_zero_2_w_i2c_defaults_and_safe_bounds() -> None:
    """The shipped template exposes driver, I2C, retry and calibration controls without secrets."""
    config = (Path(__file__).resolve().parents[1] / "config/thermal_calibration.example.yaml").read_text(encoding="utf-8")
    assert 'driver: "mlx90640"' in config
    assert "bus: 1" in config
    assert 'address: "0x33"' in config
    assert "retries: 3" in config
    assert "emissivity: 0.95" in config
    assert "offset_celsius: 0.0" in config
    assert "min_valid_celsius: 20.0" in config
    assert "max_valid_celsius: 45.0" in config
    assert "password" not in config.lower()
