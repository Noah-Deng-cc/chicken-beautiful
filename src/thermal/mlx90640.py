"""MLX90640 适配器：输入为 I2C 热阵列，输出温度读数；硬件库只在首次读取时延迟导入。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone
import importlib
import logging
from math import isfinite
from threading import RLock
import time

from src.domain import TemperatureReading

from .base import ThermalSensor, temperature_from_frame


LOGGER = logging.getLogger(__name__)
FrameReader = Callable[[], Iterable[object]]
Clock = Callable[[], datetime]
Sleep = Callable[[float], None]


def _utc_now() -> datetime:
    """返回带 UTC 时区的当前时间。\n\nArgs: 无。\nReturns: 当前时间。\nRaises: 无。"""
    return datetime.now(timezone.utc)


class Mlx90640Sensor(ThermalSensor):
    """通过可选 Adafruit I2C 驱动读取 MLX90640 的 32x24 热阵列。"""

    def __init__(self, *, bus: int = 1, address: int = 0x33, emissivity: float = 0.95,
                 offset_celsius: float = 0.0, retries: int = 3, retry_delay_seconds: float = 0.1,
                 min_valid_celsius: float = 20.0, max_valid_celsius: float = 45.0,
                 frame_reader: FrameReader | None = None, clock: Clock = _utc_now,
                 sleep: Sleep = time.sleep) -> None:
        """保存校准和 I2C 参数但不连接硬件。

        Args: bus: I2C 总线编号。address: 7 位设备地址。emissivity: 发射率校正系数。
            offset_celsius: 温度偏移。retries: 初次读取后的重试次数。retry_delay_seconds: 重试间隔。
            min_valid_celsius: 有效下界。max_valid_celsius: 有效上界。frame_reader: fake 或自定义读取器。
            clock: 带时区时钟。sleep: 可注入等待函数。
        Returns: 无。
        Raises: TypeError: 参数类型错误。ValueError: 配置范围错误。
        """
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (bus, address, retries)):
            raise TypeError("bus, address, and retries must be integers")
        if not 0 <= bus <= 255 or not 0x03 <= address <= 0x77 or retries < 0:
            raise ValueError("bus, address, or retries is out of range")
        values = (emissivity, offset_celsius, retry_delay_seconds, min_valid_celsius, max_valid_celsius)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise TypeError("calibration values must be numeric")
        if (not isfinite(float(emissivity)) or not 0.1 <= float(emissivity) <= 1.0
                or not isfinite(float(offset_celsius)) or not -20.0 <= float(offset_celsius) <= 20.0
                or not isfinite(float(retry_delay_seconds)) or float(retry_delay_seconds) < 0):
            raise ValueError("invalid calibration or retry delay")
        if not callable(clock) or not callable(sleep) or (frame_reader is not None and not callable(frame_reader)):
            raise TypeError("frame_reader, clock, and sleep must be callable")
        temperature_from_frame((), _utc_now(), min_valid_celsius, max_valid_celsius)
        self._bus, self._address, self._emissivity = bus, address, float(emissivity)
        self._offset, self._retries, self._delay = float(offset_celsius), retries, float(retry_delay_seconds)
        self._minimum, self._maximum, self._reader = min_valid_celsius, max_valid_celsius, frame_reader
        self._clock, self._sleep, self._closed, self._lock = clock, sleep, False, RLock()

    @property
    def closed(self) -> bool:
        """返回关闭状态。\n\nArgs: 无。\nReturns: 已关闭时为 True。\nRaises: 无。"""
        with self._lock:
            return self._closed

    def read(self) -> TemperatureReading | None:
        """读取、校准并汇总一帧温度，I2C 故障会有界重试。

        Args: 无。
        Returns: 校准后的读数；缺库、断连、坏帧或重试耗尽时为 None。
        Raises: 无。
        """
        with self._lock:
            if self._closed:
                return None
            reader = self._reader if self._reader is not None else self._hardware_reader()
            if self._reader is None and reader is not None:
                self._reader = reader
        if reader is None:
            return None
        for attempt in range(self._retries + 1):
            try:
                frame = tuple((float(value) / self._emissivity) + self._offset for value in reader())
                return temperature_from_frame(frame, self._clock(), self._minimum, self._maximum)
            except Exception:
                if attempt >= self._retries:
                    LOGGER.warning("MLX90640 I2C read failed after retries")
                    return None
                self._sleep(self._delay)
        return None

    def close(self) -> None:
        """关闭传感器；实际 I2C 对象由运行时与平台管理。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        with self._lock:
            self._closed = True

    def _hardware_reader(self) -> FrameReader | None:
        """延迟创建真实 Adafruit 帧读取器。

        Args: 无。
        Returns: 可读取一帧的函数；缺少库或初始化失败时为 None。
        Raises: 无。
        """
        try:
            board = importlib.import_module("board")
            busio = importlib.import_module("busio")
            module = importlib.import_module("adafruit_mlx90640")
            i2c = busio.I2C(board.SCL, board.SDA)
            device = module.MLX90640(i2c, address=self._address)
            device.refresh_rate = module.RefreshRate.REFRESH_2_HZ
        except ModuleNotFoundError:
            LOGGER.warning("MLX90640 driver requires board, busio, and adafruit_mlx90640")
            return None
        except Exception:
            LOGGER.warning("MLX90640 I2C initialization failed on bus %d", self._bus)
            return None

        def read_frame() -> Iterable[object]:
            """读取单帧并返回 768 个原始温度值。\n\nArgs: 无。\nReturns: 原始温度序列。\nRaises: Exception: I2C 读取失败。"""
            frame = [0.0] * 768
            device.getFrame(frame)
            return frame

        return read_frame
