"""主编排器：输入为组件读数和提醒，输出快照、告警与记录；使用固定线程且可有序停止。"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import logging
from queue import Empty
import signal
from threading import Event, Thread, current_thread, main_thread

from src.core import Settings
from src.core.events import EVENT_BUS_CLOSED, EventBus
from src.core.factory import ComponentFactory, Components
from src.core.state import StateStore
from src.dialogue import DialogueService
from src.domain import Reminder, SystemSnapshot
from src.fusion import FusionRules, FusionService
from src.schedule.service import REMINDER_DUE_TOPIC, ReminderService
from src.schedule.store import ReminderStore
from src.storage import JsonlRecorder

LOGGER = logging.getLogger(__name__)
Factory = Callable[[Settings], Components]
class Orchestrator:
    """以独立、可恢复循环执行边缘服务，避免单组件故障扩散。"""

    def __init__(self, settings: Settings, *, components: Components | None = None,
                 factory: Factory = ComponentFactory.build, state: StateStore | None = None,
                 event_bus: EventBus | None = None, fusion: FusionService | None = None,
                 recorder: JsonlRecorder | None = None, reminder_service: ReminderService | None = None,
                 dialogue: DialogueService | None = None) -> None:
        """装配服务，但不启动线程或打开硬件。

        Args: settings: 已校验全局配置。components: 可选预构造组件。factory: 组件工厂。
            state: 可选状态存储。event_bus: 可选总线。fusion: 可选融合服务。recorder: 可选记录器。
            reminder_service: 可选日程服务。dialogue: 可选对话服务。
        Returns: 无。
        Raises: TypeError: settings 或工厂类型错误。ValueError: 关闭超时非法。
        """
        if not isinstance(settings, Settings) or not callable(factory):
            raise TypeError("settings must be Settings and factory must be callable")
        timeout = settings.runtime.shutdown_timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")
        self._settings, self._components = settings, components if components is not None else factory(settings)
        self._state, self._events = state or StateStore(), event_bus or EventBus()
        self._fusion = fusion or FusionService(FusionRules(38.0, 1500, 60.0, "good"))
        storage = settings.storage
        self._recorder = recorder if recorder is not None else (JsonlRecorder(
            storage.directory, rotate_daily=storage.rotate_daily,
            persist_dialogue_text=storage.persist_dialogue_text,
            persist_raw_audio=storage.persist_raw_audio,
            persist_raw_images=storage.persist_raw_images) if storage.jsonl_enabled else None)
        self._stop, self._threads = Event(), ()
        self._reminders = reminder_service or self._make_reminders()
        self._reminder_queue = self._events.subscribe(REMINDER_DUE_TOPIC) if self._reminders is not None else None
        self._dialogue = dialogue or self._make_dialogue()

    def run(self) -> int:
        """启动固定线程并等待停止信号。

        Args: 无。
        Returns: 正常关闭时为零。
        Raises: 无；线程故障被隔离并记录。
        """
        self._install_signals()
        self._threads = tuple(self._thread(name, target) for name, target in self._targets())
        for thread in self._threads:
            thread.start()
        self._stop.wait()
        self._shutdown()
        return 0

    def stop(self) -> None:
        """请求所有循环停止并尽力取消阻塞音频。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        self._stop.set()
        for item in (self._components.speech_input, self._components.speech_output):
            if item is not None:
                try:
                    item.cancel()
                except Exception:
                    LOGGER.warning("audio component cancellation failed")

    def _make_reminders(self) -> ReminderService | None:
        """按配置创建日程服务。\n\nArgs: 无。\nReturns: 日程服务或 None。\nRaises: 无；构造失败会降级。"""
        if not self._settings.schedule.enabled:
            return None
        try:
            return ReminderService(ReminderStore(self._settings.schedule.store_path), self._events,
                                   poll_interval_seconds=self._settings.schedule.poll_interval_seconds)
        except Exception:
            LOGGER.exception("reminder service setup failed")
            return None

    def _make_dialogue(self) -> DialogueService | None:
        """仅在音频和智能体均可用时创建对话服务。\n\nArgs: 无。\nReturns: 对话服务或 None。\nRaises: 无；构造失败会降级。"""
        parts = self._components
        if parts.speech_input is None or parts.speech_output is None or parts.agent is None:
            return None
        try:
            return DialogueService(parts.speech_input, parts.speech_output, parts.agent, self._fusion,
                                   self._state, self._settings.audio.listen_timeout_seconds)
        except Exception:
            LOGGER.exception("dialogue service setup failed")
            return None

    def _targets(self) -> tuple[tuple[str, Callable[[], None]], ...]:
        """生成常量上界的运行线程集合。\n\nArgs: 无。\nReturns: 线程名和循环函数元组。\nRaises: 无。"""
        targets: list[tuple[str, Callable[[], None]]] = []
        for name, component, interval, topic in (
            ("vision", self._components.vision, self._settings.vision.sample_interval_seconds, "vision.reading"),
            ("thermal", self._components.thermal, self._settings.thermal.sample_interval_seconds, "thermal.reading"),
            ("co2", self._components.co2, self._settings.co2.sample_interval_seconds, "co2.reading"),
        ):
            if component is not None:
                targets.append((name, partial(self._collect, component, interval, topic)))
        if self._reminders is not None:
            targets.extend((("schedule", partial(self._reminders.run, self._stop)),
                            ("reminder-events", self._consume_reminders)))
        if self._dialogue is not None:
            targets.append(("dialogue", partial(self._dialogue.run, self._stop)))
        return tuple(targets)

    def _thread(self, name: str, target: Callable[[], None]) -> Thread:
        """创建固定名称的非守护恢复线程。\n\nArgs: name: 线程名称。target: 循环函数。\nReturns: 未启动线程。\nRaises: 无。"""
        return Thread(name=f"dorm-{name}", target=partial(self._guard, name, target), daemon=False)

    def _guard(self, name: str, target: Callable[[], None]) -> None:
        """隔离顶层循环崩溃。\n\nArgs: name: 组件名。target: 循环函数。\nReturns: 无。\nRaises: 无。"""
        try:
            target()
        except Exception:
            LOGGER.exception("component loop stopped component=%s", name)

    def _collect(self, component: object, interval: float, topic: str) -> None:
        """以可中断间隔采集并发布一个传感器组件。\n\nArgs: component: 具备 read 的传感器。interval: 采样秒数。topic: 事件主题。\nReturns: 无。\nRaises: 无。"""
        try:
            starter = getattr(component, "start", None)
            if callable(starter):
                starter()
        except Exception:
            LOGGER.exception("component startup failed topic=%s", topic)
        while not self._stop.is_set():
            try:
                reading = component.read()  # type: ignore[attr-defined]
                if reading is not None:
                    snapshot = self._state.update(reading)
                    self._events.publish(topic, reading)
                    self._record(snapshot)
            except Exception:
                LOGGER.exception("component sampling failed topic=%s", topic)
            self._stop.wait(max(0.01, float(interval)))

    def _consume_reminders(self) -> None:
        """将总线提醒同步至快照与记录器。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        queue = self._reminder_queue
        if queue is None: return
        while not self._stop.is_set():
            try:
                event = queue.get(timeout=0.25)
            except Empty:
                continue
            if event is EVENT_BUS_CLOSED:
                return
            if isinstance(event, Reminder):
                self._record(self._state.update(event))
            queue.task_done()
    def _record(self, snapshot: SystemSnapshot) -> None:
        """尽力记录快照和本地告警。\n\nArgs: snapshot: 最新系统快照。\nReturns: 无。\nRaises: 无。"""
        if self._recorder is None:
            return
        self._recorder.write(snapshot)
        for code in self._fusion.local_alerts(snapshot):
            self._recorder.write({"timestamp": self._state.snapshot.timestamp, "type": "local_alert", "code": code})
    def _install_signals(self) -> None:
        """仅在主线程安装 SIGINT/SIGTERM 的有序停止处理器。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        if current_thread() is not main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self._handle_signal)
    def _handle_signal(self, signum: int, frame: object) -> None:
        """接收系统停止信号。\n\nArgs: signum: 信号编号。frame: 当前栈帧。\nReturns: 无。\nRaises: 无。"""
        self.stop()
    def _shutdown(self) -> None:
        """等待固定线程并按依赖反向释放资源。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        self.stop()
        for thread in self._threads:
            thread.join(self._settings.runtime.shutdown_timeout_seconds)
        self._events.close()
        if self._recorder is not None:
            self._recorder.close()
        for item in (self._components.vision, self._components.thermal, self._components.co2,
                     self._components.speech_input, self._components.speech_output):
            if item is not None:
                try:
                    item.close()
                except Exception:
                    LOGGER.warning("component close failed")
