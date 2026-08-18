"""日程调度：输入提醒存储和停止信号，输出到期事件；依赖标准库、事件总线和日程存储。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import logging
from math import isfinite
from threading import Event, RLock

from src.core.events import EventBus
from src.domain.models import Reminder

from .store import ReminderStore


REMINDER_DUE_TOPIC = "reminder.due"
Clock = Callable[[], datetime]
WaitFunction = Callable[[Event, float], bool]


def _utc_now() -> datetime:
    """返回当前带 UTC 时区的时间。\n\n    Args: 无。\n    Returns: 当前 UTC 时间。\n    Raises: 无。"""
    return datetime.now(timezone.utc)


def _wait(stop_event: Event, seconds: float) -> bool:
    """在可中断等待中轮询停止信号。\n\n    Args: stop_event: 线程停止事件。seconds: 最长等待秒数。\n    Returns: 停止事件已设置时为 True。\n    Raises: 无。"""
    return stop_event.wait(seconds)


def _aware(value: datetime) -> None:
    """校验时钟返回带时区时间。\n\n    Args: value: 待校验时间。\n    Returns: 无。\n    Raises: TypeError: 值不是 datetime。ValueError: 时间缺少时区。"""
    if not isinstance(value, datetime):
        raise TypeError("clock must return a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")


class ReminderService:
    """轮询到期提醒并将每个提醒最多发布一次的服务。"""

    def __init__(
        self, store: ReminderStore, event_bus: EventBus, *, poll_interval_seconds: float = 1.0,
        topic: str = REMINDER_DUE_TOPIC, clock: Clock = _utc_now,
        wait: WaitFunction = _wait, logger: logging.Logger | None = None,
    ) -> None:
        """创建可测试且无后台线程的调度服务。\n\n        Args: store: 提醒持久化存储。event_bus: 进程内事件总线。poll_interval_seconds: 轮询周期。topic: 到期事件主题。clock: 可注入时钟。wait: 可注入可中断等待函数。logger: 可选日志器。\n        Returns: 无。\n        Raises: TypeError: 依赖或参数类型错误。ValueError: 轮询周期或主题非法。"""
        if not all(callable(getattr(store, name, None)) for name in ("due", "acknowledge")):
            raise TypeError("store must provide due and acknowledge")
        if not callable(getattr(event_bus, "publish", None)):
            raise TypeError("event_bus must provide publish")
        if isinstance(poll_interval_seconds, bool) or not isinstance(poll_interval_seconds, (int, float)):
            raise TypeError("poll_interval_seconds must be numeric")
        if not isfinite(float(poll_interval_seconds)) or poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive and finite")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be a non-empty string")
        if not callable(clock) or not callable(wait):
            raise TypeError("clock and wait must be callable")
        self._store, self._event_bus = store, event_bus
        self._interval, self._topic = float(poll_interval_seconds), topic
        self._clock, self._wait = clock, wait
        self._logger, self._lock = logger or logging.getLogger(__name__), RLock()
        self._published: dict[str, Reminder] = {}

    def run(self, stop_event: Event) -> None:
        """持续调度直到停止事件被设置。\n\n        Args: stop_event: 由应用关闭流程设置的线程事件。\n        Returns: 无。\n        Raises: TypeError: stop_event 不是 Event。"""
        if not isinstance(stop_event, Event):
            raise TypeError("stop_event must be a threading.Event")
        while not stop_event.is_set():
            self.run_once()
            if self._wait(stop_event, self._interval):
                return

    def run_once(self) -> int:
        """执行一次隔离故障的到期提醒轮询。\n\n        Args: 无。\n        Returns: 本次成功发布并确认的提醒数量。\n        Raises: 无；时钟、存储、发布及单个提醒异常仅写日志。"""
        try:
            now = self._clock()
            _aware(now)
            reminders = self._store.due(now)
        except Exception as exc:
            self._logger.error("日程轮询失败: %s", exc)
            return 0
        published = 0
        for reminder in reminders:
            try:
                if self._already_published(reminder):
                    self._acknowledge(reminder)
                    continue
                if bool(getattr(self._event_bus, "closed", False)):
                    raise RuntimeError("event bus is closed")
                self._event_bus.publish(self._topic, reminder)
                with self._lock:
                    self._published[reminder.reminder_id] = reminder
                self._acknowledge(reminder)
                published += 1
            except Exception as exc:
                self._logger.error("提醒 %s 发布失败: %s", reminder.reminder_id, exc)
        return published

    def _already_published(self, reminder: Reminder) -> bool:
        """判断该版本提醒是否已发布。\n\n        Args: reminder: 当前到期提醒。\n        Returns: 已发布相同提醒时为 True。\n        Raises: 无。"""
        with self._lock:
            return self._published.get(reminder.reminder_id) == reminder

    def _acknowledge(self, reminder: Reminder) -> None:
        """尽力确认已发布提醒，失败留待下次轮询重试。\n\n        Args: reminder: 已成功发布的提醒。\n        Returns: 无。\n        Raises: 无；存储错误仅记录。"""
        try:
            self._store.acknowledge(reminder.reminder_id)
        except Exception as exc:
            self._logger.error("提醒 %s 确认失败: %s", reminder.reminder_id, exc)
