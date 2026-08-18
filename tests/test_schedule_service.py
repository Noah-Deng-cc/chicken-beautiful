"""T13 日程调度服务验收：验证到期发布、幂等确认、故障隔离和可中断线程。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from threading import Event, Thread
from typing import NoReturn

import pytest

from src.domain.models import Reminder
from src.schedule.service import REMINDER_DUE_TOPIC, ReminderService


UTC = timezone.utc
BASE_TIME = datetime(2030, 1, 1, 9, 0, tzinfo=UTC)


def make_reminder(identifier: str = "reminder-1", *, message: str = "开窗通风") -> Reminder:
    """构造已到期的有效提醒。

    Args: identifier: 稳定提醒标识。message: 提醒正文。
    Returns: 已到期且未确认的 Reminder。
    Raises: 无。
    """
    return Reminder(identifier, message, BASE_TIME - timedelta(seconds=1))


class FakeStore:
    """记录 due 和 acknowledge 调用的最小提醒存储替身。"""

    def __init__(self, due_items: list[Reminder] | None = None) -> None:
        """初始化返回项和调用记录。

        Args: due_items: 每次到期查询返回的提醒。
        Returns: 无。
        Raises: 无。
        """
        self.due_items = due_items or []
        self.due_calls: list[datetime] = []
        self.acknowledged: list[str] = []

    def due(self, now: datetime) -> list[Reminder]:
        """记录查询并返回当前到期项。

        Args: now: 服务提供的当前时间。
        Returns: 配置的到期提醒副本。
        Raises: 无。
        """
        self.due_calls.append(now)
        return list(self.due_items)

    def acknowledge(self, reminder_id: str) -> None:
        """记录确认操作。

        Args: reminder_id: 已发布提醒标识。
        Returns: 无。
        Raises: 无。
        """
        self.acknowledged.append(reminder_id)


class FakeBus:
    """记录发布事件的最小事件总线替身。"""

    closed = False

    def __init__(self) -> None:
        """初始化发布记录。

        Args: 无。
        Returns: 无。
        Raises: 无。
        """
        self.events: list[tuple[str, Reminder]] = []

    def publish(self, topic: str, event: object) -> None:
        """记录提醒发布。

        Args: topic: 发布主题。event: 业务事件。
        Returns: 无。
        Raises: 无。
        """
        assert isinstance(event, Reminder)
        self.events.append((topic, event))


def service(store: FakeStore, bus: FakeBus, **kwargs: object) -> ReminderService:
    """使用确定性 UTC 时钟构造被测服务。

    Args: store: 测试存储。bus: 测试事件总线。kwargs: 对服务的覆写依赖。
    Returns: 已配置 ReminderService。
    Raises: 无。
    """
    options: dict[str, object] = {"clock": lambda: BASE_TIME}
    options.update(kwargs)
    return ReminderService(store, bus, **options)  # type: ignore[arg-type]


def test_run_once_publishes_due_reminder_and_acknowledges_it() -> None:
    """到期提醒应以稳定主题发布一次并被确认。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    reminder = make_reminder()
    store, bus = FakeStore([reminder]), FakeBus()

    assert service(store, bus).run_once() == 1
    assert bus.events == [(REMINDER_DUE_TOPIC, reminder)]
    assert store.acknowledged == [reminder.reminder_id]
    assert store.due_calls == [BASE_TIME]


def test_run_once_deduplicates_same_reminder_but_retries_confirmation() -> None:
    """已发布的同版本提醒不应二次发布，确认仍应在后续轮询重试。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    reminder = make_reminder()
    store, bus = FakeStore([reminder]), FakeBus()
    scheduler = service(store, bus)

    assert scheduler.run_once() == 1
    assert scheduler.run_once() == 0
    assert bus.events == [(REMINDER_DUE_TOPIC, reminder)]
    assert store.acknowledged == [reminder.reminder_id, reminder.reminder_id]


def test_run_once_republishes_replacement_with_same_identifier() -> None:
    """相同 ID 的替换提醒因版本不同应重新发布和确认。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    original = make_reminder("same", message="第一次")
    replacement = make_reminder("same", message="替换后")
    store, bus = FakeStore([original]), FakeBus()
    scheduler = service(store, bus)

    assert scheduler.run_once() == 1
    store.due_items = [replacement]
    assert scheduler.run_once() == 1
    assert bus.events == [(REMINDER_DUE_TOPIC, original), (REMINDER_DUE_TOPIC, replacement)]
    assert store.acknowledged == ["same", "same"]


def test_run_once_keeps_published_event_when_acknowledge_fails(caplog: pytest.LogCaptureFixture) -> None:
    """确认存储失败不应使已成功发布的提醒再次发布。

    Args: caplog: pytest 日志捕获。
    Returns: 无。
    Raises: 无。
    """
    reminder = make_reminder()
    store, bus = FakeStore([reminder]), FakeBus()

    def failing_acknowledge(reminder_id: str) -> NoReturn:
        """模拟存储写入故障。

        Args: reminder_id: 待确认标识。
        Returns: 无。
        Raises: OSError: 始终失败。
        """
        raise OSError(f"cannot acknowledge {reminder_id}")

    store.acknowledge = failing_acknowledge  # type: ignore[method-assign]
    scheduler = service(store, bus)
    with caplog.at_level(logging.ERROR):
        assert scheduler.run_once() == 1
        assert scheduler.run_once() == 0

    assert bus.events == [(REMINDER_DUE_TOPIC, reminder)]
    assert "确认失败" in caplog.text


@pytest.mark.parametrize("clock", [lambda: datetime(2030, 1, 1), lambda: "not-a-time"])
def test_run_once_isolates_invalid_clock_values(clock: object, caplog: pytest.LogCaptureFixture) -> None:
    """无时区或错误类型的时钟值必须只记录错误并结束本轮。

    Args: clock: 返回非法值的时钟。caplog: pytest 日志捕获。
    Returns: 无。
    Raises: 无。
    """
    store, bus = FakeStore([make_reminder()]), FakeBus()
    with caplog.at_level(logging.ERROR):
        assert service(store, bus, clock=clock).run_once() == 0

    assert store.due_calls == [] and bus.events == []
    assert "日程轮询失败" in caplog.text


def test_run_once_isolates_clock_and_due_storage_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    """时钟和到期存储异常都不得从调度循环泄漏。

    Args: caplog: pytest 日志捕获。
    Returns: 无。
    Raises: 无。
    """
    store, bus = FakeStore(), FakeBus()

    def failing_due(_: datetime) -> NoReturn:
        """模拟存储读取故障。

        Args: _: 当前时间。
        Returns: 无。
        Raises: OSError: 始终失败。
        """
        raise OSError("storage unavailable")

    store.due = failing_due  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR):
        assert service(store, bus).run_once() == 0
        assert service(store, bus, clock=lambda: (_ for _ in ()).throw(RuntimeError("clock failed"))).run_once() == 0

    assert bus.events == [] and caplog.text.count("日程轮询失败") == 2


def test_run_once_isolates_event_bus_failure_without_acknowledging(caplog: pytest.LogCaptureFixture) -> None:
    """发布失败时不得确认提醒，且其他提醒仍继续处理。

    Args: caplog: pytest 日志捕获。
    Returns: 无。
    Raises: 无。
    """
    first, second = make_reminder("first"), make_reminder("second")
    store, bus = FakeStore([first, second]), FakeBus()
    calls = 0

    def sometimes_failing_publish(topic: str, event: object) -> None:
        """让第一个发布失败、第二个正常记录。

        Args: topic: 发布主题。event: 业务事件。
        Returns: 无。
        Raises: RuntimeError: 第一次调用失败。
        """
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("bus unavailable")
        assert isinstance(event, Reminder)
        bus.events.append((topic, event))

    bus.publish = sometimes_failing_publish  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR):
        assert service(store, bus).run_once() == 1

    assert bus.events == [(REMINDER_DUE_TOPIC, second)]
    assert store.acknowledged == ["second"]
    assert "提醒 first 发布失败" in caplog.text


def test_run_returns_immediately_when_wait_observes_stop() -> None:
    """等待函数观察到停止信号后，run 只执行一个轮询周期。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    store, bus = FakeStore(), FakeBus()
    waits: list[float] = []

    def stop_on_wait(stop_event: Event, seconds: float) -> bool:
        """记录周期并模拟等待期间收到停止。

        Args: stop_event: 服务停止事件。seconds: 请求等待时长。
        Returns: True，代表停止。
        Raises: 无。
        """
        waits.append(seconds)
        stop_event.set()
        return True

    service(store, bus, poll_interval_seconds=0.25, wait=stop_on_wait).run(Event())
    assert len(store.due_calls) == 1 and waits == [0.25]


def test_run_thread_stops_during_real_wait_within_one_poll_cycle() -> None:
    """真实线程在 Event.set 后应中断等待，不需等待完整轮询周期。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    store, bus, stop_event = FakeStore(), FakeBus(), Event()
    scheduler = service(store, bus, poll_interval_seconds=2.0)
    thread = Thread(target=scheduler.run, args=(stop_event,), daemon=True)

    thread.start()
    assert len(store.due_calls) == 1
    stop_event.set()
    thread.join(timeout=0.5)

    assert not thread.is_alive(), "stop_event did not interrupt the polling wait"


def test_run_rejects_invalid_stop_event() -> None:
    """公共运行入口必须拒绝非 threading.Event 的停止对象。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    with pytest.raises(TypeError, match="stop_event must be a threading.Event"):
        service(FakeStore(), FakeBus()).run(object())  # type: ignore[arg-type]
