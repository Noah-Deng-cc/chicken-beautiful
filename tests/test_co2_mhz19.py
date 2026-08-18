"""T11 MH-Z19 UART acceptance tests using only in-memory serial doubles."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.co2 import Co2Thresholds
from src.domain import Co2Level, Co2Reading
from src.co2 import mhz19
from src.co2.mhz19 import Mhz19Sensor, parse_mhz19_frame


BASE_TIME = datetime(2026, 8, 18, 10, 30, tzinfo=timezone.utc)
COMMAND = b"\xff\x01\x86\x00\x00\x00\x00\x00\x79"


def response(ppm: int) -> bytes:
    """Build a checksummed nine-byte MH-Z19 response frame."""
    frame = bytearray((0xFF, 0x86, ppm >> 8, ppm & 0xFF, 0, 0, 0, 0, 0))
    frame[8] = (0xFF - (sum(frame[1:8]) % 256) + 1) & 0xFF
    return bytes(frame)


class FakeSerial:
    """Configurable pyserial-compatible double that never opens a real port."""

    def __init__(self, reads: list[object], *, write_error: BaseException | None = None) -> None:
        self._reads = deque(reads)
        self.write_error = write_error
        self.read_sizes: list[int] = []
        self.writes: list[bytes] = []
        self.reset_calls = 0
        self.flush_calls = 0
        self.close_calls = 0

    def reset_input_buffer(self) -> None:
        self.reset_calls += 1

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        if self.write_error is not None:
            raise self.write_error
        return len(data)

    def flush(self) -> None:
        self.flush_calls += 1

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        item = self._reads.popleft() if self._reads else b""
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, bytes)
        return item

    def close(self) -> None:
        self.close_calls += 1


def sensor(factory: object, **kwargs: object) -> Mhz19Sensor:
    """Construct a deterministic sensor with standard test thresholds."""
    return Mhz19Sensor(
        "/dev/serial0",
        thresholds=Co2Thresholds(elevated=800, poor=1_500),
        serial_factory=factory,  # type: ignore[arg-type]
        clock=lambda: BASE_TIME,
        **kwargs,
    )


def test_valid_frame_produces_timestamped_classified_reading_and_writes_command() -> None:
    """A valid 1000 ppm response is read from fake UART and classified from thresholds."""
    serial = FakeSerial([response(1_000)])
    created: list[dict[str, object]] = []

    def factory(**kwargs: object) -> FakeSerial:
        created.append(kwargs)
        return serial

    actual = sensor(factory).read()

    assert actual == Co2Reading(BASE_TIME, 1_000, Co2Level.ELEVATED)
    assert created == [{"port": "/dev/serial0", "baudrate": 9600, "timeout": 1.0}]
    assert serial.writes == [COMMAND]
    assert serial.reset_calls == serial.flush_calls == 1
    assert serial.read_sizes == [9]


def test_port_baud_rate_and_timeout_are_passed_to_lazy_serial_factory() -> None:
    """Pi UART path and all configured transport parameters reach pyserial unchanged."""
    captured: list[dict[str, object]] = []
    serial = FakeSerial([response(799)])
    instance = Mhz19Sensor(
        "/dev/ttyAMA0", baud_rate=19_200, timeout_seconds=0.25, retries=0,
        thresholds=Co2Thresholds(elevated=800, poor=1_500),
        serial_factory=lambda **kwargs: captured.append(kwargs) or serial,
        clock=lambda: BASE_TIME,
    )

    assert instance.read() == Co2Reading(BASE_TIME, 799, Co2Level.GOOD)
    assert captured == [{"port": "/dev/ttyAMA0", "baudrate": 19_200, "timeout": 0.25}]


@pytest.mark.parametrize(
    "frame",
    [
        b"\xff\x86\x03\xe8\x00\x00\x00\x00\x00",  # invalid checksum
        b"\xff\x85\x03\xe8\x00\x00\x00\x00\x00",  # wrong response header
        response(1000)[:4],  # short read followed by timeout
        b"",  # read timeout
    ],
)
def test_bad_frame_short_read_or_timeout_never_produces_a_reading(frame: bytes) -> None:
    """Every malformed or incomplete UART response safely returns None, never a fake ppm."""
    serial = FakeSerial([frame])
    instance = sensor(lambda **_: serial, retries=0)

    assert instance.read() is None
    assert serial.writes == [COMMAND]
    assert serial.close_calls == 0


def test_read_exact_accepts_fragmented_valid_frame() -> None:
    """UART chunking still collects all nine bytes before checksum validation."""
    frame = response(1_500)
    serial = FakeSerial([frame[:2], frame[2:6], frame[6:]])

    assert sensor(lambda **_: serial, retries=0).read() == Co2Reading(BASE_TIME, 1_500, Co2Level.POOR)
    assert serial.read_sizes == [9, 7, 3]


def test_retry_after_invalid_frame_keeps_connection_and_can_recover() -> None:
    """A bad response consumes a retry and a subsequent valid frame returns a real reading."""
    invalid = b"\xff\x86\x03\xe8\x00\x00\x00\x00\x00"
    serial = FakeSerial([invalid, response(650)])
    instance = sensor(lambda **_: serial, retries=1)

    assert instance.read() == Co2Reading(BASE_TIME, 650, Co2Level.GOOD)
    assert serial.writes == [COMMAND, COMMAND]
    assert serial.close_calls == 0


def test_disconnect_releases_failed_handle_and_reconnects_for_retry() -> None:
    """A transport error drops the old UART object and retries through a newly opened one."""
    disconnected = FakeSerial([], write_error=OSError("UART disconnected"))
    recovered = FakeSerial([response(1_600)])
    serials = iter([disconnected, recovered])
    factory_calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> FakeSerial:
        factory_calls.append(kwargs)
        return next(serials)

    instance = sensor(factory, retries=1)
    assert instance.read() == Co2Reading(BASE_TIME, 1_600, Co2Level.POOR)
    assert len(factory_calls) == 2
    assert disconnected.close_calls == 1
    assert recovered.close_calls == 0


def test_missing_pyserial_is_contained_and_module_import_remains_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent optional pyserial affects a read only; importing this driver needs no serial module."""
    calls: list[str] = []

    def missing(name: str) -> object:
        calls.append(name)
        raise ModuleNotFoundError("No module named serial")

    monkeypatch.setattr(mhz19.importlib, "import_module", missing)
    instance = Mhz19Sensor(
        "/dev/serial0", retries=0, thresholds=Co2Thresholds(elevated=800, poor=1_500),
        clock=lambda: BASE_TIME,
    )

    assert instance.read() is None
    assert calls == ["serial"]


def test_parse_rejects_non_bytes_and_close_is_idempotent_and_terminal() -> None:
    """Frame API rejects wrong types; close releases once and prevents later hardware access."""
    with pytest.raises(TypeError, match="frame must be bytes"):
        parse_mhz19_frame(bytearray(response(800)))  # type: ignore[arg-type]

    serial = FakeSerial([response(800)])
    instance = sensor(lambda **_: serial)
    instance.close()
    instance.close()

    assert instance.closed is True
    assert serial.close_calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        instance.read()

    opened = sensor(lambda **_: serial)
    assert opened.read() is not None
    opened.close()
    opened.close()
    assert serial.close_calls == 1


def test_calibration_template_matches_zero_2_w_mhz19_uart_defaults() -> None:
    """The shipped template documents usable serial defaults without credentials or fake offsets."""
    config = (Path(__file__).resolve().parents[1] / "config/co2_calibration.example.yaml").read_text(encoding="utf-8")
    assert 'port: "/dev/serial0"' in config
    assert "baud_rate: 9600" in config
    assert "timeout_seconds: 1.0" in config
    assert "retries: 3" in config
    assert "elevated: 1000" in config
    assert "poor: 1500" in config
