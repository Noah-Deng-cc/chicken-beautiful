"""T04 状态存储与事件总线验收：覆盖正常、边界、异常及并发行为。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from queue import Empty, Queue
from threading import Thread

import pytest

from src.core.events import EVENT_BUS_CLOSED, EventBus
from src.core.state import StateStore
from src.domain.models import (
    AgentReply,
    Co2Level,
    Co2Reading,
    DialogueTurn,
    Emotion,
    EmotionReading,
    Reminder,
    SystemSnapshot,
    TemperatureReading,
)


BASE_TIME = datetime(2026, 8, 18, 9, 30, tzinfo=timezone.utc)


def make_emotion(offset: int = 0) -> EmotionReading:
    """创建有效且可区分的情绪读数。"""
    return EmotionReading(
        timestamp=BASE_TIME + timedelta(seconds=offset),
        dominant=Emotion.HAPPY,
        confidence=0.8,
        valence=0.6,
        arousal=0.2,
        person_id=f"resident-{offset}",
    )


def make_temperature(offset: int = 0) -> TemperatureReading:
    """创建有效且可区分的温度读数。"""
    return TemperatureReading(
        timestamp=BASE_TIME + timedelta(seconds=offset),
        maximum_celsius=37.2,
        average_celsius=36.5,
    )


def make_co2(offset: int = 0) -> Co2Reading:
    """创建有效且可区分的 CO2 读数。"""
    return Co2Reading(
        timestamp=BASE_TIME + timedelta(seconds=offset),
        ppm=800 + offset,
        level=Co2Level.ELEVATED,
    )


def make_dialogue(offset: int = 0) -> DialogueTurn:
    """创建有效且可区分的对话轮次。"""
    timestamp = BASE_TIME + timedelta(seconds=offset)
    return DialogueTurn(
        timestamp=timestamp,
        user_text=f"空气怎么样 {offset}",
        reply=AgentReply(text="请开窗通风", timestamp=timestamp),
    )


def make_reminder(identifier: str = "reminder-1", message: str = "开窗") -> Reminder:
    """创建有效提醒。"""
    return Reminder(
        reminder_id=identifier,
        message=message,
        due_at=BASE_TIME + timedelta(hours=1),
    )


@pytest.mark.parametrize(
    ("reading", "field"),
    [
        (make_emotion(), "emotion"),
        (make_temperature(), "temperature"),
        (make_co2(), "co2"),
        (make_dialogue(), "dialogue"),
    ],
    ids=["emotion", "temperature", "co2", "dialogue"],
)
def test_state_store_updates_each_timestamped_reading(reading: object, field: str) -> None:
    """四类带时间读数应更新对应字段和快照时间。"""
    store = StateStore()

    actual = store.update(reading)

    assert getattr(actual, field) is reading
    assert actual.timestamp == reading.timestamp  # type: ignore[attr-defined]
    assert store.snapshot is actual
    assert store.get_snapshot() is actual


def test_state_store_snapshot_and_previous_versions_are_immutable() -> None:
    """更新必须创建新快照，旧版本和当前版本均不可被调用者修改。"""
    initial = SystemSnapshot(timestamp=BASE_TIME)
    store = StateStore(initial)
    previous = store.snapshot

    current = store.update(make_emotion())

    assert previous is initial
    assert previous.emotion is None
    assert current is not previous
    with pytest.raises(FrozenInstanceError):
        current.emotion = None  # type: ignore[misc]


def test_state_store_reminder_upsert_appends_then_replaces_in_place() -> None:
    """提醒按 ID 追加或原位替换，不能重复或改变其他提醒顺序。"""
    store = StateStore()
    first = make_reminder("first", "喝水")
    second = make_reminder("second", "开窗")
    replacement = make_reminder("first", "已经喝水")

    before = datetime.now(timezone.utc)
    store.update(first)
    store.update(second)
    actual = store.update(replacement)
    after = datetime.now(timezone.utc)

    assert actual.reminders == (replacement, second)
    assert before <= actual.timestamp <= after
    assert actual.timestamp.tzinfo is not None


def test_state_store_default_clock_initializes_an_aware_utc_snapshot() -> None:
    """默认时钟应创建接近当前 UTC 时间的 aware 初始快照。"""
    before = datetime.now(timezone.utc)

    snapshot = StateStore().snapshot

    after = datetime.now(timezone.utc)
    assert snapshot.timestamp.tzinfo is timezone.utc
    assert before <= snapshot.timestamp <= after


def test_state_store_injected_clock_controls_initial_and_reminder_timestamps() -> None:
    """注入时钟须同时决定初始快照和提醒更新的时间。"""
    fixed = BASE_TIME + timedelta(minutes=5)
    store = StateStore(clock=lambda: fixed)

    updated = store.update(make_reminder())

    assert store.snapshot.timestamp == fixed
    assert updated.timestamp == fixed


@pytest.mark.parametrize(
    ("clock", "error", "message"),
    [
        (object(), TypeError, "clock must be callable"),
        (lambda: "not-a-datetime", TypeError, "clock must return a datetime"),
        (lambda: datetime(2026, 8, 18, 9, 30), ValueError, "clock must return an aware datetime"),
    ],
    ids=["non-callable", "non-datetime-result", "naive-result"],
)
def test_state_store_rejects_invalid_clock_contract(
    clock: object, error: type[Exception], message: str
) -> None:
    """时钟本身、返回类型和时区契约均应在创建时明确失败。"""
    with pytest.raises(error, match=message):
        StateStore(clock=clock)  # type: ignore[arg-type]


@pytest.mark.parametrize("initial", [object(), {}, BASE_TIME])
def test_state_store_rejects_invalid_initial_snapshot(initial: object) -> None:
    """初始值仅接受 SystemSnapshot 或 None。"""
    with pytest.raises(TypeError, match="initial must be a SystemSnapshot or None"):
        StateStore(initial)  # type: ignore[arg-type]


@pytest.mark.parametrize("reading", [None, object(), "emotion", 42])
def test_state_store_rejects_unknown_reading_types(reading: object) -> None:
    """空值和未注册读数必须明确失败且不改变状态。"""
    store = StateStore(SystemSnapshot(timestamp=BASE_TIME))
    before = store.snapshot

    with pytest.raises(TypeError, match="unsupported reading type"):
        store.update(reading)

    assert store.snapshot is before


def test_state_store_concurrent_updates_do_not_lose_or_corrupt_state() -> None:
    """多线程更新不同模态和唯一提醒时，最终快照应保持完整一致。"""
    store = StateStore(SystemSnapshot(timestamp=BASE_TIME))
    reminders = [make_reminder(f"r-{index}", f"提醒 {index}") for index in range(100)]
    readings: list[object] = []
    for index in range(100):
        readings.extend(
            [make_emotion(index), make_temperature(index), make_co2(index), make_dialogue(index)]
        )
    readings.extend(reminders)

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(store.update, readings))

    actual = store.snapshot
    assert len(results) == 500
    assert isinstance(actual, SystemSnapshot)
    assert isinstance(actual.emotion, EmotionReading)
    assert isinstance(actual.temperature, TemperatureReading)
    assert isinstance(actual.co2, Co2Reading)
    assert isinstance(actual.dialogue, DialogueTurn)
    assert len(actual.reminders) == 100
    assert {item.reminder_id for item in actual.reminders} == {
        item.reminder_id for item in reminders
    }


def test_event_bus_subscribers_and_topics_are_independent() -> None:
    """同主题订阅者各自收到事件，其他主题不会串流。"""
    bus = EventBus(queue_size=2)
    first = bus.subscribe("sensor")
    second = bus.subscribe("sensor")
    other = bus.subscribe("dialogue")
    payload = {"ppm": 900}

    bus.publish("sensor", payload)

    assert first is not second
    assert first.get_nowait() is payload
    assert second.get_nowait() is payload
    with pytest.raises(Empty):
        other.get_nowait()


def test_event_bus_full_queue_drops_oldest_per_subscriber() -> None:
    """队列满时应保留最新事件，并按每个慢订阅者累计丢弃数。"""
    bus = EventBus(queue_size=2)
    first = bus.subscribe("sensor")
    second = bus.subscribe("sensor")

    for event in ("oldest", "middle", "latest"):
        bus.publish("sensor", event)

    assert [first.get_nowait(), first.get_nowait()] == ["middle", "latest"]
    assert [second.get_nowait(), second.get_nowait()] == ["middle", "latest"]
    assert bus.dropped_events == 2


def test_event_bus_publish_to_full_queue_completes_without_consumer() -> None:
    """慢消费者不读取时，发布线程仍须结束而不能等待队列空间。"""
    bus = EventBus(queue_size=1)
    subscriber = bus.subscribe("sensor")
    bus.publish("sensor", "first")
    publisher = Thread(target=bus.publish, args=("sensor", "second"), daemon=True)

    publisher.start()
    publisher.join(timeout=1.0)

    assert not publisher.is_alive(), "publish blocked on a full subscriber queue"
    assert subscriber.get_nowait() == "second"
    assert bus.dropped_events == 1


def test_event_bus_concurrent_publish_is_bounded_and_accounted() -> None:
    """并发发布到无人消费的队列时，容量和丢弃计数应保持准确。"""
    bus = EventBus(queue_size=8)
    subscriber = bus.subscribe("sensor")
    events = list(range(200))

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda event: bus.publish("sensor", event), events))

    remaining = [subscriber.get_nowait() for _ in range(8)]
    assert subscriber.empty()
    assert len(set(remaining)) == 8
    assert set(remaining).issubset(set(events))
    assert bus.dropped_events == len(events) - 8


def test_event_bus_unsubscribe_signals_and_stops_future_delivery() -> None:
    """退订应丢弃积压、发送终止标记，且重复退订安全。"""
    bus = EventBus(queue_size=2)
    subscriber = bus.subscribe("sensor")
    bus.publish("sensor", "backlog")

    bus.unsubscribe("sensor", subscriber)
    bus.unsubscribe("sensor", subscriber)
    bus.publish("sensor", "after-unsubscribe")

    assert subscriber.get_nowait() is EVENT_BUS_CLOSED
    with pytest.raises(Empty):
        subscriber.get_nowait()


def test_event_bus_close_clears_backlog_signals_all_and_is_idempotent() -> None:
    """关闭时所有积压队列只留下终止标记，重复关闭不追加标记。"""
    bus = EventBus(queue_size=3)
    first = bus.subscribe("sensor")
    second = bus.subscribe("dialogue")
    bus.publish("sensor", "stale-sensor")
    bus.publish("dialogue", "stale-dialogue")

    bus.close()
    bus.close()

    assert bus.closed is True
    assert first.get_nowait() is EVENT_BUS_CLOSED
    assert second.get_nowait() is EVENT_BUS_CLOSED
    assert first.empty()
    assert second.empty()


def test_event_bus_after_close_subscribe_is_terminated_and_publish_is_noop() -> None:
    """关闭后新订阅者应立即终止，合法发布不再投递或计为丢弃。"""
    bus = EventBus(queue_size=1)
    existing = bus.subscribe("sensor")
    bus.close()

    late = bus.subscribe("sensor")
    bus.publish("sensor", None)

    assert existing.get_nowait() is EVENT_BUS_CLOSED
    assert late.get_nowait() is EVENT_BUS_CLOSED
    assert bus.dropped_events == 0


@pytest.mark.parametrize(
    ("queue_size", "error"),
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError), ("2", TypeError)],
)
def test_event_bus_rejects_invalid_queue_sizes(
    queue_size: object, error: type[Exception]
) -> None:
    """队列容量必须是大于零的非布尔整数。"""
    with pytest.raises(error):
        EventBus(queue_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("topic", [None, 1, object()])
@pytest.mark.parametrize("operation", ["subscribe", "publish", "unsubscribe"])
def test_event_bus_rejects_non_string_topics(topic: object, operation: str) -> None:
    """所有公开主题入口都应拒绝非字符串主题。"""
    bus = EventBus()
    if operation == "subscribe":
        call = lambda: bus.subscribe(topic)  # type: ignore[arg-type]
    elif operation == "publish":
        call = lambda: bus.publish(topic, "event")  # type: ignore[arg-type]
    else:
        call = lambda: bus.unsubscribe(topic, Queue())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="topic must be a string"):
        call()


@pytest.mark.parametrize("topic", ["", " ", "\t\n"])
@pytest.mark.parametrize("operation", ["subscribe", "publish", "unsubscribe"])
def test_event_bus_rejects_blank_topics(topic: str, operation: str) -> None:
    """所有公开主题入口都应拒绝空白主题。"""
    bus = EventBus()
    if operation == "subscribe":
        call = lambda: bus.subscribe(topic)
    elif operation == "publish":
        call = lambda: bus.publish(topic, "event")
    else:
        call = lambda: bus.unsubscribe(topic, Queue())

    with pytest.raises(ValueError, match="topic must not be empty"):
        call()


def test_event_bus_rejects_invalid_subscriber_and_close_sentinel_event() -> None:
    """退订对象须为 Queue，内部终止标记不能作为业务事件发布。"""
    bus = EventBus()

    with pytest.raises(TypeError, match="subscriber must be a Queue"):
        bus.unsubscribe("sensor", object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="close sentinel cannot be published"):
        bus.publish("sensor", EVENT_BUS_CLOSED)
