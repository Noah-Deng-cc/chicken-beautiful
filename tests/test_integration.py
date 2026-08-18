"""T29 集成验收：输入本地模拟组件，输出完整快照、提醒、对话、播报与 JSONL 故障恢复证据。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from threading import Event, Thread

from src.agent import AgentClient, AgentTransportError
from src.app import Orchestrator
from src.audio import MockSpeechInput, MockSpeechOutput
from src.core import SectionSettings, Settings, load_settings
from src.core.events import EventBus
from src.core.factory import Components
from src.core.state import StateStore
from src.dialogue import DialogueService
from src.domain import AgentReply, Co2Level, Co2Reading, Emotion, EmotionReading, Reminder, TemperatureReading
from src.fusion import FusionRules, FusionService
from src.schedule.service import ReminderService
from src.storage import JsonlRecorder


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = json.loads((ROOT / "tests" / "fixtures" / "integration_scenario.json").read_text(encoding="utf-8"))
UTC = timezone.utc
NOW = datetime(2034, 1, 2, 3, 4, 5, tzinfo=UTC)


class SensorProbe:
    """按序返回读数或故障的无硬件传感器替身。"""

    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes, self.starts, self.closes = list(outcomes), 0, 0

    def start(self) -> None:
        """记录启动且不访问硬件。"""
        self.starts += 1

    def read(self) -> object:
        """返回下一个受控读数或抛出受控故障。"""
        item = self._outcomes.pop(0) if self._outcomes else None
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        """记录幂等关闭。"""
        self.closes += 1


class LocalAgent(AgentClient):
    """记录上下文的本地智能体，绝不建立网络连接。"""

    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes, self.calls = list(outcomes), []

    def reply(self, query: str, context: dict[str, object], conversation_id: str | None) -> AgentReply:
        """返回受控答复或模拟传输中断。"""
        self.calls.append((query, context, conversation_id))
        item = self._outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, AgentReply)
        return item


def _settings(tmp_path: Path) -> Settings:
    """生成短周期、只使用临时本地路径的 mock 配置。"""
    base = load_settings(ROOT / "config" / "settings.example.yaml")

    def section(value: SectionSettings, **updates: object) -> SectionSettings:
        return SectionSettings(value.values | updates)

    return replace(
        base,
        runtime=section(base.runtime, mode="mock", shutdown_timeout_seconds=0.1),
        vision=section(base.vision, sample_interval_seconds=0.005),
        thermal=section(base.thermal, sample_interval_seconds=0.005),
        co2=section(base.co2, sample_interval_seconds=0.005),
        schedule=section(base.schedule, enabled=True, store_path=tmp_path / "reminders.json", poll_interval_seconds=0.005),
        storage=section(base.storage, directory=tmp_path / "jsonl", jsonl_enabled=True, persist_dialogue_text=True),
    )


def _collect_until_reading(orchestrator: Orchestrator, probe: SensorProbe, topic: str, predicate: object) -> None:
    """在后台运行采样器，直到指定快照谓词成立后停止。"""
    thread = Thread(target=orchestrator._collect, args=(probe, 0.005, topic))
    thread.start()
    assert isinstance(predicate, Event) and predicate.wait(0.5), f"{topic} did not recover"
    orchestrator.stop()
    thread.join(0.5)
    assert not thread.is_alive()


def test_mock_end_to_end_records_complete_snapshot_reminder_agent_reply_and_playback(tmp_path: Path) -> None:
    """正常模拟场景产生三模态快照、日程、智能体答复、播报和可解析 JSONL。"""
    settings, state, events = _settings(tmp_path), StateStore(clock=lambda: NOW), EventBus()
    emotion = EmotionReading(NOW, Emotion(SCENARIO["vision"]["emotion"]), SCENARIO["vision"]["confidence"], 0.0, 0.0)
    thermal = TemperatureReading(NOW, SCENARIO["thermal"]["maximum_celsius"], SCENARIO["thermal"]["average_celsius"])
    co2 = Co2Reading(NOW, SCENARIO["co2"]["ppm"], Co2Level(SCENARIO["co2"]["level"]))
    vision, heat, air = SensorProbe([emotion]), SensorProbe([thermal]), SensorProbe([co2])
    microphone, speaker = MockSpeechInput([SCENARIO["dialogue"]["transcript"]]), MockSpeechOutput()
    agent = LocalAgent([AgentReply(SCENARIO["dialogue"]["reply"], NOW, "local-1")])
    recorder = JsonlRecorder(tmp_path / "jsonl", persist_dialogue_text=True, clock=lambda: NOW)
    components = Components(vision, heat, air, microphone, speaker, agent)  # type: ignore[arg-type]
    orchestrator = Orchestrator(settings, components=components, state=state, event_bus=events, recorder=recorder)

    for probe, topic, attribute in ((vision, "vision.reading", "emotion"), (heat, "thermal.reading", "temperature"), (air, "co2.reading", "co2")):
        observed = Event()
        original = probe.read
        def read_and_observe(original: object = original, attribute: str = attribute) -> object:
            result = original()  # type: ignore[operator]
            if result is not None:
                observed.set()
            return result
        probe.read = read_and_observe  # type: ignore[method-assign]
        _collect_until_reading(orchestrator, probe, topic, observed)
        orchestrator._stop.clear()

    due_at = datetime.now(UTC) - timedelta(seconds=1)
    reminder = Reminder(SCENARIO["reminder"]["id"], SCENARIO["reminder"]["message"], due_at)
    reminder_service = ReminderService(type("Store", (), {"due": lambda _, __: [reminder], "acknowledge": lambda _, __: None})(), events, clock=lambda: due_at + timedelta(seconds=1))
    consumer = Thread(target=orchestrator._consume_reminders)
    consumer.start()
    assert reminder_service.run_once() == 1
    for _ in range(50):
        if state.snapshot.reminders == (reminder,):
            break
        Event().wait(0.005)
    assert state.snapshot.reminders == (reminder,)

    dialogue = DialogueService(microphone, speaker, agent, FusionService(FusionRules(38.0, 1500, 60.0, "good")), state, 0.1)
    turn = dialogue.run_once()
    assert turn is not None and turn.reply.text == SCENARIO["dialogue"]["reply"]
    assert speaker.spoken_texts == (SCENARIO["dialogue"]["reply"],)
    assert agent.calls[0][1]["co2"] == {"timestamp": NOW.isoformat(), "age_seconds": 0.0, "fresh": True, "quality": "valid", "ppm": 1800, "level": "poor"}
    assert agent.calls[0][1]["reminders"][0]["due"] is True
    orchestrator._record(state.snapshot)
    orchestrator.stop(); events.close(); consumer.join(0.5); recorder.close()
    records = [json.loads(line) for line in (tmp_path / "jsonl" / "2034-01-02.jsonl").read_text(encoding="utf-8").splitlines()]
    complete = [item for item in records if item.get("dialogue")]
    assert complete and complete[-1]["emotion"]["dominant"] == "neutral"
    assert complete[-1]["temperature"]["maximum_celsius"] == 38.2 and complete[-1]["co2"]["ppm"] == 1800
    assert complete[-1]["reminders"][0]["message"] == "开窗通风"


def test_state_clock_makes_co2_freshness_and_reminder_due_status_reproducible() -> None:
    """固定时钟下 CO2 新鲜度和到期提醒应不依赖运行机器时间。"""
    reading = Co2Reading(NOW, 900, Co2Level.GOOD)
    due_reminder = Reminder("due", "开窗", NOW, False)
    fixed = StateStore(clock=lambda: NOW)
    fixed.update(reading)
    fresh_context = FusionService(FusionRules(38.0, 1500, 60.0, "good")).build_context(
        fixed.update(due_reminder)
    )

    later = StateStore(clock=lambda: NOW + timedelta(seconds=61))
    later.update(reading)
    stale_context = FusionService(FusionRules(38.0, 1500, 60.0, "good")).build_context(
        later.update(due_reminder)
    )

    assert fresh_context["co2"] is not None
    assert fresh_context["co2"]["age_seconds"] == 0.0
    assert fresh_context["co2"]["fresh"] is True
    assert fresh_context["reminders"] == [
        {"reminder_id": "due", "message": "开窗", "due_at": NOW.isoformat(), "acknowledged": False, "due": True}
    ]
    assert stale_context["co2"] is not None
    assert stale_context["co2"]["age_seconds"] == 61.0
    assert stale_context["co2"]["fresh"] is False
    assert stale_context["reminders"][0]["due"] is True


def test_faults_in_vision_thermal_co2_network_and_audio_degrade_then_recover(tmp_path: Path) -> None:
    """五类本地故障均被隔离，恢复读数、后续智能体答复和播报仍可完成。"""
    settings, state, events = _settings(tmp_path), StateStore(), EventBus()
    readings = (
        ("vision.reading", "emotion", SensorProbe([OSError("camera"), EmotionReading(NOW, Emotion.HAPPY, 0.9, 0.5, 0.2)])),
        ("thermal.reading", "temperature", SensorProbe([OSError("i2c"), TemperatureReading(NOW, 36.8, 36.2)])),
        ("co2.reading", "co2", SensorProbe([OSError("serial"), Co2Reading(NOW, 900, Co2Level.GOOD)])),
    )
    components = Components(readings[0][2], readings[1][2], readings[2][2], None, None, None)  # type: ignore[arg-type]
    orchestrator = Orchestrator(settings, components=components, state=state, event_bus=events, recorder=JsonlRecorder(tmp_path / "fault-jsonl", clock=lambda: NOW))
    for topic, attribute, probe in readings:
        observed = Event()
        original = probe.read
        def read_and_observe(original: object = original, attribute: str = attribute) -> object:
            result = original()  # type: ignore[operator]
            if result is not None:
                observed.set()
            return result
        probe.read = read_and_observe  # type: ignore[method-assign]
        _collect_until_reading(orchestrator, probe, topic, observed)
        orchestrator._stop.clear()
    assert state.snapshot.emotion is not None and state.snapshot.temperature is not None and state.snapshot.co2 is not None

    microphone, speaker = MockSpeechInput(["网络故障", "恢复对话"]), MockSpeechOutput([OSError("speaker"), True])
    agent = LocalAgent([AgentTransportError("offline"), AgentReply("已恢复本地服务。", NOW, "local-2")])
    service = DialogueService(microphone, speaker, agent, FusionService(FusionRules(38.0, 1500, 60.0, "good")), state, 0.1)
    assert service.run_once() is None
    recovered = service.run_once()
    assert recovered is not None and recovered.reply.text == "已恢复本地服务。"
    assert speaker.attempted_texts == ("当前网络服务暂不可用，请稍后再试。", "已恢复本地服务。")
    assert speaker.spoken_texts == ("已恢复本地服务。",)
    orchestrator._shutdown()
