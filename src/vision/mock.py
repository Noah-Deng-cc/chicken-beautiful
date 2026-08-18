"""视觉模拟器：输入固定值、序列或异常，输出确定性情绪读数；仅依赖标准库和视觉契约。"""

from __future__ import annotations

from collections.abc import Iterable
import logging
from threading import RLock

from src.domain.models import EmotionReading

from .base import VisionPipeline


MockResult = EmotionReading | None | Exception


class MockVisionPipeline(VisionPipeline):
    """线程安全、无需摄像头的确定性视觉管道。"""

    def __init__(
        self,
        readings: MockResult | Iterable[MockResult] = None,
        *,
        repeat: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        """配置固定结果或结果序列。\n\n        Args: readings: 固定结果，或按次读取的结果序列；异常项会转换为 None。repeat: 序列耗尽后是否循环。logger: 可选日志器。\n        Returns: 无。\n        Raises: TypeError: readings 或序列项类型错误。ValueError: 空序列要求循环。"""
        if not isinstance(repeat, bool):
            raise TypeError("repeat must be a boolean")
        self._fixed: MockResult | None = None
        self._sequence: tuple[MockResult, ...] = ()
        if isinstance(readings, (EmotionReading, Exception)) or readings is None:
            self._fixed = readings
        elif isinstance(readings, Iterable) and not isinstance(readings, (str, bytes, bytearray)):
            sequence = tuple(readings)
            if not all(isinstance(item, (EmotionReading, Exception)) or item is None for item in sequence):
                raise TypeError("each reading must be EmotionReading, Exception, or None")
            if repeat and not sequence:
                raise ValueError("an empty sequence cannot repeat")
            self._sequence = sequence
        else:
            raise TypeError("readings must be a result or iterable of results")
        self._repeat = repeat
        self._logger = logger or logging.getLogger(__name__)
        self._lock = RLock()
        self._index = 0
        self._read_count = 0
        self._started = False
        self._closed = False

    @property
    def started(self) -> bool:
        """查询管道是否处于可读取状态。\n\n        Args: 无。\n        Returns: 已启动且未关闭时为 True。\n        Raises: 无。"""
        with self._lock:
            return self._started and not self._closed

    @property
    def closed(self) -> bool:
        """查询管道是否已永久关闭。\n\n        Args: 无。\n        Returns: 已关闭时为 True。\n        Raises: 无。"""
        with self._lock:
            return self._closed

    @property
    def read_count(self) -> int:
        """返回有效生命周期内的读取调用次数。\n\n        Args: 无。\n        Returns: start 后、close 前的 read 调用次数。\n        Raises: 无。"""
        with self._lock:
            return self._read_count

    def start(self) -> None:
        """幂等启动模拟管道。\n\n        Args: 无。\n        Returns: 无；已关闭时保持关闭并记录警告。\n        Raises: 无。"""
        with self._lock:
            if self._closed:
                self._logger.warning("已关闭的视觉模拟器不能重新启动")
                return
            self._started = True

    def read(self) -> EmotionReading | None:
        """返回下一个模拟结果并吸收注入异常。\n\n        Args: 无。\n        Returns: 固定/序列中的 EmotionReading；未启动、已关闭、耗尽、空结果或异常时为 None。\n        Raises: 无。"""
        with self._lock:
            if not self._started or self._closed:
                return None
            self._read_count += 1
            result = self._next_result()
            if isinstance(result, Exception):
                self._logger.error("视觉模拟读取失败: %s", result)
                return None
            return result

    def close(self) -> None:
        """幂等关闭模拟管道。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无。"""
        with self._lock:
            self._started = False
            self._closed = True

    def _next_result(self) -> MockResult:
        """在持锁状态下取得固定值或下一序列项。\n\n        Args: 无。\n        Returns: 下一个模拟结果，序列耗尽时为 None。\n        Raises: 无。"""
        if not self._sequence:
            return self._fixed
        if self._index >= len(self._sequence):
            if not self._repeat:
                return None
            self._index = 0
        result = self._sequence[self._index]
        self._index += 1
        return result
