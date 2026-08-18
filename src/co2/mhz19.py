"""MH-Z19 UART 驱动：输入串口响应帧，输出 CO2 读数；serial 仅在首次读取时导入。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import importlib
import logging
from math import isfinite
from threading import RLock

from src.domain import Co2Reading

from .base import Co2Sensor, Co2Thresholds, classify_co2


SerialFactory = Callable[..., object]
Clock = Callable[[], datetime]
_COMMAND = b"\xff\x01\x86\x00\x00\x00\x00\x00\x79"


def _utc_now() -> datetime:
    """返回当前带 UTC 时区时间。\n\n    Args: 无。\n    Returns: 当前 UTC datetime。\n    Raises: 无。"""
    return datetime.now(timezone.utc)


def parse_mhz19_frame(frame: bytes) -> int | None:
    """校验 MH-Z19 九字节响应并解析 ppm。\n\n    Args: frame: 传感器返回的原始九字节帧。\n    Returns: 校验通过时的 ppm；长度、帧头或校验错误时为 None。\n    Raises: TypeError: frame 不是 bytes。"""
    if not isinstance(frame, bytes):
        raise TypeError("frame must be bytes")
    if len(frame) != 9 or frame[:2] != b"\xff\x86":
        return None
    checksum = (0xFF - (sum(frame[1:8]) % 256) + 1) & 0xFF
    if frame[8] != checksum:
        return None
    return frame[2] * 256 + frame[3]


class Mhz19Sensor(Co2Sensor):
    """带延迟串口导入、校验和断连恢复的 MH-Z19 适配器。"""

    def __init__(
        self, port: str, *, baud_rate: int = 9600, timeout_seconds: float = 1.0,
        retries: int = 3, thresholds: Co2Thresholds,
        serial_factory: SerialFactory | None = None, clock: Clock = _utc_now,
        logger: logging.Logger | None = None,
    ) -> None:
        """保存串口配置但不导入或打开设备。\n\n        Args: port: UART 设备路径。baud_rate: 波特率。timeout_seconds: 单次读取超时。retries: 首次失败后的额外重试次数。thresholds: CO2 分级阈值。serial_factory: 可注入串口构造器。clock: 可注入时钟。logger: 可选日志器。\n        Returns: 无。\n        Raises: TypeError: 参数类型错误。ValueError: 端口或数值配置无效。"""
        if not isinstance(port, str) or not port.strip():
            raise ValueError("port must be a non-empty string")
        if isinstance(baud_rate, bool) or not isinstance(baud_rate, int):
            raise TypeError("baud_rate must be an integer")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise TypeError("timeout_seconds must be numeric")
        if isinstance(retries, bool) or not isinstance(retries, int):
            raise TypeError("retries must be an integer")
        if baud_rate < 1 or not isfinite(float(timeout_seconds)) or timeout_seconds < 0 or retries < 0:
            raise ValueError("serial numeric settings are invalid")
        if not isinstance(thresholds, Co2Thresholds):
            raise TypeError("thresholds must be Co2Thresholds")
        if serial_factory is not None and not callable(serial_factory) or not callable(clock):
            raise TypeError("serial_factory and clock must be callable")
        self._port, self._baud_rate = port, baud_rate
        self._timeout, self._retries, self._thresholds = float(timeout_seconds), retries, thresholds
        self._factory, self._clock = serial_factory, clock
        self._logger, self._serial, self._closed = logger or logging.getLogger(__name__), None, False
        self._lock = RLock()

    @property
    def closed(self) -> bool:
        """查询线程安全关闭状态。\n\n        Args: 无。\n        Returns: 已关闭时为 True。\n        Raises: 无。"""
        with self._lock:
            return self._closed

    def read(self) -> Co2Reading | None:
        """查询传感器并在断连、超时或坏帧时安全降级。\n\n        Args: 无。\n        Returns: 校验和分级通过的读数；不可用时为 None。\n        Raises: RuntimeError: 传感器已关闭。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("CO2 sensor is closed")
            for _ in range(self._retries + 1):
                try:
                    serial = self._ensure_serial()
                    reset = getattr(serial, "reset_input_buffer", None)
                    if callable(reset):
                        reset()
                    serial.write(_COMMAND)
                    flush = getattr(serial, "flush", None)
                    if callable(flush):
                        flush()
                    ppm = parse_mhz19_frame(self._read_exact(serial))
                    if ppm is None:
                        self._logger.warning("MH-Z19 返回无效响应帧")
                        continue
                    timestamp = self._clock()
                    level = classify_co2(ppm, self._thresholds)
                    return Co2Reading(timestamp, ppm, level)
                except Exception as exc:
                    self._logger.warning("MH-Z19 读取失败: %s", exc)
                    self._release_serial()
            return None

    def close(self) -> None:
        """幂等关闭 UART 句柄。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无；关闭错误仅记录。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._release_serial()

    def _ensure_serial(self) -> object:
        """延迟导入 PySerial 并建立串口连接。\n\n        Args: 无。\n        Returns: 已打开的串口对象。\n        Raises: ImportError: PySerial 不可用。OSError: 串口无法打开。"""
        if self._serial is None:
            factory = self._factory
            if factory is None:
                factory = importlib.import_module("serial").Serial
            self._serial = factory(port=self._port, baudrate=self._baud_rate, timeout=self._timeout)
        return self._serial

    @staticmethod
    def _read_exact(serial: object) -> bytes:
        """尽可能收集完整响应帧。\n\n        Args: serial: 支持 read 方法的串口对象。\n        Returns: 收到的至多九字节数据。\n        Raises: TypeError: read 返回非字节对象。"""
        data = bytearray()
        while len(data) < 9:
            chunk = serial.read(9 - len(data))
            if not isinstance(chunk, bytes):
                raise TypeError("serial read must return bytes")
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)

    def _release_serial(self) -> None:
        """关闭并丢弃当前串口以允许下一次重连。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无；关闭失败仅记录。"""
        serial, self._serial = self._serial, None
        if serial is None:
            return
        try:
            close = getattr(serial, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            self._logger.warning("MH-Z19 串口关闭失败: %s", exc)
