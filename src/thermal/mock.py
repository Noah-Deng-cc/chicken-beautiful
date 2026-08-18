"""热成像模拟器：输入固定/序列帧或故障，输出确定性温度读数；仅依赖标准库和热成像契约。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone
import logging
from numbers import Real
from threading import RLock

from src.domain.models import TemperatureReading

from .base import ThermalSensor, _temperature_bounds, temperature_from_frame


MockThermalItem = TemperatureReading | tuple[object, ...] | None | Exception
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    """返回带 UTC 时区的当前时间。\n\n    Args: 无。\n    Returns: 当前 UTC datetime。\n    Raises: 无。"""
    return datetime.now(timezone.utc)


def _normalize_source(
    source: TemperatureReading | Exception | Iterable[object] | None,
) -> tuple[bool, tuple[MockThermalItem, ...]]:
    """区分固定读数/帧与模拟结果序列。\n\n    Args: source: 固定读数、数值帧、结果序列、异常或 None。\n    Returns: 是否固定及不可变结果元组。\n    Raises: TypeError: 来源或序列项类型错误。"""
    if source is None or isinstance(source, (TemperatureReading, Exception)):
        return True, (source,)
    try:
        items = tuple(source)
    except TypeError:
        raise TypeError("source must be a reading, frame, exception, or iterable") from None
    if items and all(isinstance(item, Real) and not isinstance(item, bool) for item in items):
        return True, (items,)
    normalized: list[MockThermalItem] = []
    for item in items:
        if item is None or isinstance(item, (TemperatureReading, Exception)):
            normalized.append(item)
            continue
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes, bytearray)):
            normalized.append(tuple(item))
            continue
        raise TypeError("sequence items must be readings, frames, exceptions, or None")
    return False, tuple(normalized)


class MockThermalSensor(ThermalSensor):
    """支持固定/序列热阵列和异常注入的线程安全模拟器。"""

    def __init__(
        self, source: TemperatureReading | Exception | Iterable[object] | None = None, *,
        min_valid_celsius: float = 20.0, max_valid_celsius: float = 45.0,
        repeat: bool = False, clock: Clock = _utc_now,
        logger: logging.Logger | None = None,
    ) -> None:
        """创建无需硬件的热成像模拟器。\n\n        Args: source: 固定读数/数值帧，或按次消费的结果序列。min_valid_celsius: 帧下界。max_valid_celsius: 帧上界。repeat: 序列是否循环。clock: 带时区时钟。logger: 可选日志器。\n        Returns: 无。\n        Raises: TypeError: 参数类型错误。ValueError: 阈值无效或要求循环空序列。"""
        lower, upper = _temperature_bounds(min_valid_celsius, max_valid_celsius)
        if not isinstance(repeat, bool):
            raise TypeError("repeat must be a boolean")
        if not callable(clock):
            raise TypeError("clock must be callable")
        fixed, items = _normalize_source(source)
        if repeat and not fixed and not items:
            raise ValueError("an empty sequence cannot repeat")
        self._fixed, self._items = fixed, items
        self._lower, self._upper, self._repeat = lower, upper, repeat
        self._clock, self._logger = clock, logger or logging.getLogger(__name__)
        self._index, self._read_count = 0, 0
        self._closed, self._lock = False, RLock()

    @property
    def closed(self) -> bool:
        """线程安全查询关闭状态。\n\n        Args: 无。\n        Returns: 已关闭时为 True。\n        Raises: 无。"""
        with self._lock:
            return self._closed

    @property
    def read_count(self) -> int:
        """返回关闭前的读取调用次数。\n\n        Args: 无。\n        Returns: 累计读取次数。\n        Raises: 无。"""
        with self._lock:
            return self._read_count

    def read(self) -> TemperatureReading | None:
        """读取并转换下一个模拟项。\n\n        Args: 无。\n        Returns: 有效 TemperatureReading；关闭、耗尽、坏帧或故障时为 None。\n        Raises: 无；注入异常和转换异常均记录后吸收。"""
        with self._lock:
            if self._closed:
                return None
            self._read_count += 1
            item = self._next_item()
            if isinstance(item, TemperatureReading):
                return item
            if item is None:
                return None
            if isinstance(item, Exception):
                self._logger.error("热成像模拟读取失败: %s", item)
                return None
            try:
                return temperature_from_frame(item, self._clock(), self._lower, self._upper)
            except (TypeError, ValueError) as exc:
                self._logger.error("热成像模拟帧无效: %s", exc)
                return None

    def close(self) -> None:
        """幂等关闭模拟器。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无。"""
        with self._lock:
            self._closed = True

    def _next_item(self) -> MockThermalItem:
        """在持锁状态下读取固定值或下一序列项。\n\n        Args: 无。\n        Returns: 当前模拟项，耗尽时为 None。\n        Raises: 无。"""
        if self._fixed:
            return self._items[0]
        if self._index >= len(self._items):
            if not self._repeat:
                return None
            self._index = 0
        item = self._items[self._index]
        self._index += 1
        return item
