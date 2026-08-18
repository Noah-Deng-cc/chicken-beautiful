"""运行状态存储：输入为领域读数，输出不可变系统快照；依赖标准库线程锁和领域模型。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from threading import RLock

from src.domain.models import (
    Co2Reading,
    DialogueTurn,
    EmotionReading,
    Reminder,
    SystemSnapshot,
    TemperatureReading,
)


Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    """返回当前带 UTC 时区的时间。

    Args: 无。
    Returns: 当前 aware UTC datetime。
    Raises: 无。
    """
    return datetime.now(timezone.utc)


class StateStore:
    """线程安全地保存各组件的最新读数。"""

    def __init__(self, initial: SystemSnapshot | None = None, *, clock: Clock | None = None) -> None:
        """创建状态存储。

        Args:
            initial: 可选的初始不可变快照。
            clock: 返回 aware datetime 的时间函数；省略时使用 UTC 当前时间。
        Returns: 无。
        Raises: TypeError: initial 不是快照或 clock 不可调用/返回 datetime。
            ValueError: clock 返回 naive datetime。
        """
        if initial is not None and not isinstance(initial, SystemSnapshot):
            raise TypeError("initial must be a SystemSnapshot or None")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._lock = RLock()
        self._clock = clock or _utc_now
        self._snapshot = initial or SystemSnapshot(timestamp=self._now())

    def _now(self) -> datetime:
        """读取并严格校验注入时钟的返回值。

        Args: 无。
        Returns: aware datetime。
        Raises: TypeError: 时钟返回非 datetime。
            ValueError: 时钟返回 naive datetime。
        """
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("clock must return a datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return an aware datetime")
        return value

    @property
    def snapshot(self) -> SystemSnapshot:
        """读取当前不可变快照。\n\n        Args: 无。\n        Returns: 当前 SystemSnapshot；调用者可安全跨线程持有。\n        Raises: 无。"""
        return self.get_snapshot()

    def get_snapshot(self) -> SystemSnapshot:
        """在线程锁保护下读取当前快照引用。\n\n        Args: 无。\n        Returns: 当前不可变 SystemSnapshot。\n        Raises: 无。"""
        with self._lock:
            return self._snapshot

    def update(self, reading: object) -> SystemSnapshot:
        """原子更新读数对应的快照字段。\n\n        Args: reading: 情绪、温度、CO2、提醒或对话领域对象。\n        Returns: 更新后的不可变 SystemSnapshot。\n        Raises: TypeError: reading 类型不受支持。"""
        with self._lock:
            if isinstance(reading, EmotionReading):
                updated = replace(self._snapshot, timestamp=reading.timestamp, emotion=reading)
            elif isinstance(reading, TemperatureReading):
                updated = replace(self._snapshot, timestamp=reading.timestamp, temperature=reading)
            elif isinstance(reading, Co2Reading):
                updated = replace(self._snapshot, timestamp=reading.timestamp, co2=reading)
            elif isinstance(reading, DialogueTurn):
                updated = replace(self._snapshot, timestamp=reading.timestamp, dialogue=reading)
            elif isinstance(reading, Reminder):
                reminders = self._upsert_reminder(self._snapshot.reminders, reading)
                updated = replace(
                    self._snapshot,
                    timestamp=self._now(),
                    reminders=reminders,
                )
            else:
                raise TypeError(f"unsupported reading type: {type(reading).__name__}")
            self._snapshot = updated
            return updated

    @staticmethod
    def _upsert_reminder(current: tuple[Reminder, ...], reading: Reminder) -> tuple[Reminder, ...]:
        """按提醒 ID 替换或追加提醒。\n\n        Args: current: 当前提醒元组。reading: 新提醒。\n        Returns: 保持原顺序的新提醒元组。\n        Raises: 无。"""
        for index, reminder in enumerate(current):
            if reminder.reminder_id == reading.reminder_id:
                return current[:index] + (reading,) + current[index + 1:]
        return current + (reading,)
