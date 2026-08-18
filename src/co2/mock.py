"""CO2 模拟器：输入为固定/序列浓度、断连或异常，输出为带时区的 Co2Reading；依赖标准库和基础契约。"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
import logging
from threading import RLock

from src.domain import Co2Level, Co2Reading
from .base import Co2Sensor, Co2Thresholds, classify_co2


LOGGER = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """生成带 UTC 时区的当前时间。\n\nArgs: 无。\nReturns: 带时区时间。\nRaises: 无。"""
    return datetime.now(timezone.utc)


def _source(readings: object | Iterable[object]) -> tuple[bool, tuple[object, ...]]:
    """标准化固定或序列模拟输入。\n\nArgs: readings: 单个值或按次消费序列。\nReturns: 是否固定及输入元组。\nRaises: 无。"""
    if readings is None or isinstance(readings, (int, str, bytes, Exception)):
        return True, (readings,)
    try:
        return False, tuple(readings)
    except TypeError:
        return True, (readings,)


class MockCo2Sensor(Co2Sensor):
    """线程安全的 CO2 测试替身，支持断连和非法浓度。"""

    def __init__(self, readings: object | Iterable[object] = None, *, thresholds: Co2Thresholds,
                 clock: Callable[[], datetime] = _utc_now) -> None:
        """创建不连接硬件的传感器。\n\nArgs: readings: 固定值或序列，None 为断连。thresholds: 配置阈值。clock: 可注入时钟。\nReturns: 无。\nRaises: TypeError: thresholds/clock 类型错误。"""
        if not isinstance(thresholds, Co2Thresholds):
            raise TypeError("thresholds must be Co2Thresholds")
        if not callable(clock):
            raise TypeError("clock must be callable")
        fixed, values = _source(readings)
        self._fixed, self._readings = fixed, deque(values)
        self._thresholds, self._clock, self._lock = thresholds, clock, RLock()
        self._closed = False

    @property
    def closed(self) -> bool:
        """返回线程安全关闭状态。\n\nArgs: 无。\nReturns: 已关闭为 True。\nRaises: 无。"""
        with self._lock:
            return self._closed

    def read(self) -> Co2Reading | None:
        """读取并按配置生成模拟读数。\n\nArgs: 无。\nReturns: 正常/invalid 读数，断连或注入异常为 None。\nRaises: RuntimeError: 已关闭。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("CO2 sensor is closed")
            value = self._readings[0] if self._fixed and self._readings else (
                self._readings.popleft() if self._readings else None)
        if isinstance(value, Exception):
            LOGGER.warning("mock CO2 sensor raised an injected error")
            return None
        if value is None:
            LOGGER.warning("mock CO2 sensor is disconnected")
            return None
        level = classify_co2(value, self._thresholds) if isinstance(value, int) and not isinstance(value, bool) else Co2Level.INVALID
        try:
            timestamp = self._clock()
            if level is Co2Level.INVALID:
                LOGGER.warning("mock CO2 sensor returned an invalid ppm value")
                return Co2Reading(timestamp, None, Co2Level.INVALID)
            return Co2Reading(timestamp, value, level)
        except Exception:
            LOGGER.warning("mock CO2 sensor could not create a reading")
            return None

    def close(self) -> None:
        """幂等关闭模拟传感器。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        with self._lock:
            self._closed = True
