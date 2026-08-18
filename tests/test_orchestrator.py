"""T22 主编排器验收：在无摄像头、I2C、串口或网络的环境验证常驻服务生命周期。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import logging
from pathlib import Path
import signal
from threading import Event, Thread
from typing import Callable

import pytest

import src.app as app
from src.app import Orchestrator
from src.core import SectionSettings, Settings, load_settings
from src.core.events import EventBus
from src.core.factory import Components
from src.core.state import StateStore
from src.domain import Co2Level, Co2Reading, Emotion, EmotionReading


ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc
NOW = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


class Probe:
    """无硬件采集器，记录生命周期并按注入函数产生读数。"""

    def __init__(self, read: Callable[[], object], *, start_error: bool = False) -> None:
        self._read, self._start_error = read, start_error
        self.starts = 0
        self.reads = 0
        self.closes = 0

    def start(self) -> None:
        self.starts += 1
        if self._start_error:
            raise OSError("camera unavailable")

    def read(self) -> object:
        self.reads += 1
        return self._read()

    def close(self) -> None:
        self.closes += 1


class AudioProbe:
    """记录音频取消与关闭，模拟阻塞麦克风/TTS 所需的最小接口。"""

    def __init__(self) -> None:
        self.cancels = 0
        self.closes = 0

    def cancel(self) -> None:
        self.cancels += 1

    def close(self) -> None:
        self.closes += 1


class RecorderProbe:
    """内存记录器，避免在树莓派验收测试中产生文件。"""

    def __init__(self) -> None:
        self.records: list[object] = []
        self.closes = 0

    def write(self, record: object) -> bool:
        self.records.append(record)
        return True

    def close(self) -> None:
        self.closes += 1


class LoopProbe:
    """供日程和对话线程使用的最小可停止服务。"""

    def __init__(self) -> None:
        self.runs = 0

    def run(self, stop_event: Event) -> None:
        self.runs += 1
        stop_event.wait(0.01)


def _settings(*, mode: str = "mock") -> Settings:
    """从公开示例构造短周期、禁用日程的确定性运行配置。"""
    base = load_settings(ROOT / "config" / "settings.example.yaml")

    def section(value: SectionSettings, **updates: object) -> SectionSettings:
        return SectionSettings(value.values | updates)

    return replace(
        base,
        runtime=section(base.runtime, mode=mode, shutdown_timeout_seconds=0.2),
        vision=section(base.vision, sample_interval_seconds=0.01),
        thermal=section(base.thermal, sample_interval_seconds=0.01),
        co2=section(base.co2, sample_interval_seconds=0.01),
        schedule=section(base.schedule, enabled=False),
        storage=section(base.storage, jsonl_enabled=False),
    )


def _components(*, vision: object = None, thermal: object = None, co2: object = None,
                input_device: object = None, output_device: object = None) -> Components:
    """组装仅含本测试替身的组件集合。"""
    return Components(vision, thermal, co2, input_device, output_device, None)  # type: ignore[arg-type]


def test_public_api_and_configurable_fixed_sensor_targets() -> None:
    """公共入口可导入；启用的三个采集器各对应一个固定名称和配置周期。"""
    vision = Probe(lambda: None)
    thermal = Probe(lambda: None)
    co2 = Probe(lambda: None)
    orchestrator = Orchestrator(_settings(), components=_components(vision=vision, thermal=thermal, co2=co2))

    targets = orchestrator._targets()

    assert app.Orchestrator is Orchestrator
    assert [name for name, _ in targets] == ["vision", "thermal", "co2"]
    assert [orchestrator._thread(name, target).name for name, target in targets] == [
        "dorm-vision", "dorm-thermal", "dorm-co2",
    ]
    assert all(not orchestrator._thread(name, target).daemon for name, target in targets)


def test_sensor_failure_is_logged_without_stopping_other_sensor_state_events_or_records(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """摄像头异常不应阻断 CO2 状态、事件发布、告警和 JSONL 记录。"""
    failed = Event()
    stop_after_co2 = Event()

    def vision_read() -> object:
        failed.set()
        raise OSError("camera disconnected")

    def co2_read() -> object:
        assert failed.wait(0.5), "vision loop did not run"
        stop_after_co2.set()
        return Co2Reading(NOW, 1800, Co2Level.POOR)

    vision, co2 = Probe(vision_read), Probe(co2_read)
    recorder, events, state = RecorderProbe(), EventBus(), StateStore()
    published: list[tuple[str, object]] = []
    original_publish = events.publish

    def capture_publish(topic: str, event: object) -> None:
        published.append((topic, event))
        original_publish(topic, event)

    events.publish = capture_publish  # type: ignore[method-assign]
    orchestrator = Orchestrator(
        _settings(), components=_components(vision=vision, co2=co2), state=state,
        event_bus=events, recorder=recorder,
    )

    runner = Thread(target=orchestrator.run, name="t22-runner")
    with caplog.at_level(logging.ERROR, logger="src.app.orchestrator"):
        runner.start()
        assert stop_after_co2.wait(0.75)
        orchestrator.stop()
        runner.join(1.0)

    assert not runner.is_alive()
    assert vision.starts == 1 and vision.reads >= 1
    assert co2.starts == 1 and state.snapshot.co2 == Co2Reading(NOW, 1800, Co2Level.POOR)
    assert published == [("co2.reading", Co2Reading(NOW, 1800, Co2Level.POOR))]
    assert any(getattr(item, "co2", None) == state.snapshot.co2 for item in recorder.records)
    assert any(isinstance(item, dict) and item.get("code") == "co2_concentration_high" for item in recorder.records)
    assert "component sampling failed topic=vision.reading" in caplog.text


def test_start_failure_is_isolated_and_sampling_wait_is_interruptible(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """树莓派摄像头初始化失败后仍会安全采样；stop 不等待完整采样周期。"""
    stop = Event()

    def read() -> object:
        stop.set()
        return EmotionReading(NOW, Emotion.NEUTRAL, 0.8, 0.0, 0.0)

    vision = Probe(read, start_error=True)
    orchestrator = Orchestrator(_settings(), components=_components(vision=vision))
    orchestrator._stop = stop
    with caplog.at_level(logging.ERROR, logger="src.app.orchestrator"):
        orchestrator._collect(vision, 60.0, "vision.reading")

    assert vision.starts == 1 and vision.reads == 1
    assert "component startup failed topic=vision.reading" in caplog.text

    read_once = Event()
    sleeper = Probe(lambda: read_once.set() or None)
    waiting = Orchestrator(_settings(), components=_components(vision=sleeper))
    thread = Thread(target=waiting._collect, args=(sleeper, 60.0, "vision.reading"))
    thread.start()
    assert read_once.wait(0.25)
    waiting.stop()
    thread.join(0.25)
    assert not thread.is_alive(), "stop did not interrupt the configured sampling wait"


def test_reminder_and_dialogue_loops_have_fixed_targets_and_reminders_are_recorded() -> None:
    """日程、提醒消费者和对话均纳入固定线程集合，到期事件会更新快照并记录。"""
    reminders, dialogue, recorder, events = LoopProbe(), LoopProbe(), RecorderProbe(), EventBus()
    orchestrator = Orchestrator(
        _settings(), components=_components(), event_bus=events, recorder=recorder,
        reminder_service=reminders, dialogue=dialogue,
    )

    assert [name for name, _ in orchestrator._targets()] == ["schedule", "reminder-events", "dialogue"]

    completed = Event()
    original_write = recorder.write

    def observe(record: object) -> bool:
        result = original_write(record)
        completed.set()
        return result

    recorder.write = observe  # type: ignore[method-assign]
    consumer = Thread(target=orchestrator._consume_reminders)
    consumer.start()
    from src.domain import Reminder

    reminder = Reminder("ventilate", "开窗通风", NOW)
    events.publish("reminder.due", reminder)
    assert completed.wait(0.5)
    orchestrator.stop()
    events.close()
    consumer.join(0.5)

    assert not consumer.is_alive()
    assert orchestrator._state.snapshot.reminders == (reminder,)
    assert recorder.records == [orchestrator._state.snapshot]


def test_stop_signal_and_shutdown_are_repeatable_and_close_all_local_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGINT/SIGTERM、重复 stop 和关闭异常均可完成有序本地资源释放。"""
    input_device, output_device = AudioProbe(), AudioProbe()
    vision, thermal, co2 = Probe(lambda: None), Probe(lambda: None), Probe(lambda: None)
    recorder, events = RecorderProbe(), EventBus()
    orchestrator = Orchestrator(
        _settings(mode="pi"), components=_components(
            vision=vision, thermal=thermal, co2=co2, input_device=input_device, output_device=output_device,
        ), event_bus=events, recorder=recorder,
    )
    installed: dict[int, object] = {}
    monkeypatch.setattr(signal, "signal", lambda signum, handler: installed.__setitem__(signum, handler))

    orchestrator._install_signals()
    assert set(installed) == {signal.SIGINT, signal.SIGTERM}
    handler = installed[signal.SIGTERM]
    assert callable(handler)
    handler(signal.SIGTERM, None)  # type: ignore[misc]
    orchestrator.stop()
    orchestrator._shutdown()

    assert orchestrator._stop.is_set() and events.closed
    assert input_device.cancels >= 3 and output_device.cancels >= 3
    assert recorder.closes == 1
    assert [item.closes for item in (vision, thermal, co2, input_device, output_device)] == [1, 1, 1, 1, 1]


def test_pi_mode_with_prebuilt_components_never_calls_factory_or_network() -> None:
    """Zero 2 W 无线环境可传入已装配硬件，编排器构造时不访问网络或重新建厂。"""
    called = False

    def forbidden_factory(_: Settings) -> Components:
        nonlocal called
        called = True
        raise AssertionError("factory must not run for injected Pi hardware")

    orchestrator = Orchestrator(
        _settings(mode="pi"), components=_components(vision=Probe(lambda: None)), factory=forbidden_factory,
    )

    assert called is False
    assert [name for name, _ in orchestrator._targets()] == ["vision"]
