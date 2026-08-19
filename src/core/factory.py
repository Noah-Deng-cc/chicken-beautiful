"""组件工厂：输入为 Settings，输出可替换的运行组件；模拟模式不导入硬件实现或秘密。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.agent import AgentClient
from src.audio import SpeechInput, SpeechOutput
from src.co2 import Co2Sensor, Co2Thresholds
from src.domain import AgentReply, Emotion
from src.thermal import ThermalSensor
from src.vision import VisionPipeline

from . import SectionSettings, Settings


class ComponentFactoryError(RuntimeError):
    """表示已知组件无法按当前配置创建。"""


@dataclass(frozen=True, slots=True)
class Components:
    """编排器使用的可选组件集合。"""

    vision: VisionPipeline | None
    thermal: ThermalSensor | None
    co2: Co2Sensor | None
    speech_input: SpeechInput | None
    speech_output: SpeechOutput | None
    agent: AgentClient | None


class _MockAgent(AgentClient):
    """不访问网络且不回显查询的本地智能体替身。"""

    def reply(self, query: str, context: dict[str, object], conversation_id: str | None) -> AgentReply:
        """返回固定安全答复。\n\nArgs: query: 用户文本。context: 上下文。conversation_id: 会话标识。\nReturns: 模拟回复。\nRaises: 无。"""
        return AgentReply("当前为本地模拟模式。", datetime.now(timezone.utc), conversation_id)


class ComponentFactory:
    """依据白名单配置装配实现，不使用 eval 或任意模块加载。"""

    @staticmethod
    def build(settings: Settings) -> Components:
        """创建启用组件。\n\nArgs: settings: 已解析配置。\nReturns: 不可变组件集合。\nRaises: TypeError: 类型错误。ComponentFactoryError: 驱动或构造失败。"""
        if not isinstance(settings, Settings):
            raise TypeError("settings must be Settings")
        return Components(
            ComponentFactory._vision(settings) if settings.vision.enabled else None,
            ComponentFactory._thermal(settings) if settings.thermal.enabled else None,
            ComponentFactory._co2(settings) if settings.co2.enabled else None,
            ComponentFactory._input(settings) if settings.audio.enabled else None,
            ComponentFactory._output(settings) if settings.audio.enabled else None,
            ComponentFactory._agent(settings) if settings.agent.enabled else None,
        )

    @staticmethod
    def _driver(settings: Settings, section: SectionSettings, key: str) -> str:
        """读取驱动名。\n\nArgs: settings: 全局配置。section: 配置节。key: 字段名。\nReturns: 规范化驱动名。\nRaises: ComponentFactoryError: 值无效。"""
        if settings.runtime.mode == "mock":
            return "mock"
        value = getattr(section, key)
        if not isinstance(value, str) or not value.strip():
            raise ComponentFactoryError(f"{key} driver must be a non-empty string")
        return value.strip().lower()

    @staticmethod
    def _unknown(component: str, driver: str, available: tuple[str, ...]) -> ComponentFactoryError:
        """生成安全的驱动错误。\n\nArgs: component: 组件名。driver: 实现名。available: 白名单。\nReturns: 诊断错误。\nRaises: 无。"""
        return ComponentFactoryError(
            f"{component} component driver '{driver}' is unavailable; available: {', '.join(available)}")

    @staticmethod
    def _vision(settings: Settings) -> VisionPipeline:
        """创建视觉管道。\n\nArgs: settings: 全局配置。\nReturns: 视觉管道。\nRaises: ComponentFactoryError: 创建失败。"""
        driver = ComponentFactory._driver(settings, settings.vision, "driver")
        if driver == "mock":
            from src.domain import EmotionReading
            from src.vision import MockVisionPipeline

            try:
                reading = EmotionReading(datetime.now(timezone.utc), Emotion(settings.vision.mock.emotion),
                                         float(settings.vision.mock.confidence), 0.0, 0.0)
                return MockVisionPipeline(reading)
            except Exception as exc:
                raise ComponentFactoryError("vision component configuration is invalid") from exc
        if driver != "yolo":
            raise ComponentFactory._unknown("vision", driver, ("mock", "yolo"))
        try:
            from src.vision.camera import CameraSource
            from src.vision.yolo import HaarFaceDetector, YoloEmotionPipeline

            camera = settings.vision.camera
            face = settings.vision.face_detection
            detector = HaarFaceDetector(face.cascade_path, scale_factor=float(face.scale_factor),
                                        min_neighbors=int(face.min_neighbors),
                                        min_size=(int(face.min_width), int(face.min_height))) \
                if face.enabled else None
            return YoloEmotionPipeline(CameraSource(camera.source, backend=camera.backend,
                width=camera.width, height=camera.height, fps=camera.fps), settings.vision.model_path,
                backend=settings.vision.model_format, confidence_threshold=settings.vision.confidence_threshold,
                sample_interval_seconds=settings.vision.sample_interval_seconds, device=settings.vision.device,
                face_detector=detector)
        except Exception as exc:
            raise ComponentFactoryError("vision component could not be created") from exc

    @staticmethod
    def _thermal(settings: Settings) -> ThermalSensor:
        """创建热成像传感器。\n\nArgs: settings: 全局配置。\nReturns: 热成像传感器。\nRaises: ComponentFactoryError: 创建失败。"""
        driver = ComponentFactory._driver(settings, settings.thermal, "driver")
        if driver == "mock":
            from src.thermal import MockThermalSensor
            try:
                return MockThermalSensor((settings.thermal.mock.maximum_celsius,
                                          settings.thermal.mock.average_celsius),
                                         min_valid_celsius=settings.thermal.min_valid_celsius,
                                         max_valid_celsius=settings.thermal.max_valid_celsius)
            except Exception as exc:
                raise ComponentFactoryError("thermal component configuration is invalid") from exc
        if driver != "mlx90640":
            raise ComponentFactory._unknown("thermal", driver, ("mock", "mlx90640"))
        try:
            from src.thermal.mlx90640 import Mlx90640Sensor
            connection, calibration = settings.thermal.connection, settings.thermal.calibration
            return Mlx90640Sensor(bus=connection.bus, address=int(connection.address, 0),
                emissivity=calibration.emissivity, offset_celsius=calibration.offset_celsius,
                retries=connection.retries, min_valid_celsius=settings.thermal.min_valid_celsius,
                max_valid_celsius=settings.thermal.max_valid_celsius)
        except Exception as exc:
            raise ComponentFactoryError("thermal component could not be created") from exc

    @staticmethod
    def _co2(settings: Settings) -> Co2Sensor:
        """创建 CO2 传感器。\n\nArgs: settings: 全局配置。\nReturns: CO2 传感器。\nRaises: ComponentFactoryError: 创建失败。"""
        driver = ComponentFactory._driver(settings, settings.co2, "driver")
        levels = settings.co2.thresholds_ppm
        if driver == "mock":
            from src.co2 import MockCo2Sensor
            try:
                return MockCo2Sensor(settings.co2.mock.ppm,
                    thresholds=Co2Thresholds(levels.elevated, levels.poor))
            except Exception as exc:
                raise ComponentFactoryError("co2 component configuration is invalid") from exc
        if driver != "mhz19":
            raise ComponentFactory._unknown("co2", driver, ("mock", "mhz19"))
        try:
            from src.co2.mhz19 import Mhz19Sensor
            connection = settings.co2.connection
            return Mhz19Sensor(connection.port, baud_rate=connection.baud_rate,
                timeout_seconds=connection.timeout_seconds, retries=connection.retries,
                thresholds=Co2Thresholds(levels.elevated, levels.poor))
        except Exception as exc:
            raise ComponentFactoryError("co2 component could not be created") from exc

    @staticmethod
    def _input(settings: Settings) -> SpeechInput:
        """创建语音输入。\n\nArgs: settings: 全局配置。\nReturns: 语音输入。\nRaises: ComponentFactoryError: 创建失败。"""
        driver = ComponentFactory._driver(settings, settings.audio, "input_driver")
        if driver == "mock":
            from src.audio import MockSpeechInput
            return MockSpeechInput()
        if driver != "vosk":
            raise ComponentFactory._unknown("audio input", driver, ("mock", "vosk"))
        try:
            from src.audio.vosk_asr import VoskSpeechInput
            return VoskSpeechInput(settings.audio.input.vosk_model_path,
                                   device=settings.audio.input.device,
                                   sample_rate=settings.audio.input.sample_rate_hz)
        except Exception as exc:
            raise ComponentFactoryError("audio input component could not be created") from exc

    @staticmethod
    def _output(settings: Settings) -> SpeechOutput:
        """创建语音输出。\n\nArgs: settings: 全局配置。\nReturns: 语音输出。\nRaises: ComponentFactoryError: 创建失败。"""
        driver = ComponentFactory._driver(settings, settings.audio, "output_driver")
        if driver == "mock":
            from src.audio import MockSpeechOutput
            return MockSpeechOutput()
        if driver != "system_tts":
            raise ComponentFactory._unknown("audio output", driver, ("mock", "system_tts"))
        try:
            from src.audio.system_tts import SystemSpeechOutput
            return SystemSpeechOutput(settings.audio.output.command_argv,
                                      timeout_seconds=settings.audio.output.timeout_seconds)
        except Exception as exc:
            raise ComponentFactoryError("audio output component could not be created") from exc

    @staticmethod
    def _agent(settings: Settings) -> AgentClient:
        """创建智能体。\n\nArgs: settings: 全局配置。\nReturns: 智能体客户端。\nRaises: ComponentFactoryError: 创建失败。"""
        driver = ComponentFactory._driver(settings, settings.agent, "driver")
        if driver == "mock":
            return _MockAgent()
        if driver not in ("tongji", "tongji_mcp"):
            raise ComponentFactory._unknown("agent", driver, ("mock", "tongji", "tongji_mcp"))
        try:
            if driver == "tongji_mcp":
                from src.agent import TongjiMcpAgentClient
                return TongjiMcpAgentClient(settings.agent)
            from src.agent import TongjiAgentClient
            return TongjiAgentClient(settings.agent)
        except Exception as exc:
            raise ComponentFactoryError("agent component could not be created") from exc
