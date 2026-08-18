"""T21 组件工厂与显式注册表的验收测试。"""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import yaml

from src.core import load_settings
from src.core.factory import ComponentFactory, ComponentFactoryError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "config" / "settings.example.yaml"
REGISTRY = PROJECT_ROOT / "config" / "component_registry.yaml"
FACTORY = PROJECT_ROOT / "src" / "core" / "factory.py"


def _settings(tmp_path: Path, *, mutate: object | None = None,
              monkeypatch: pytest.MonkeyPatch | None = None):
    """写入独立配置并加载为真实 Settings。"""
    data = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    copied = deepcopy(data)
    if mutate is not None:
        mutate(copied)
    path = tmp_path / "project" / "config" / "settings.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(copied, allow_unicode=True), encoding="utf-8")
    return load_settings(path)


def _all_enabled_mock(data: dict[str, object]) -> None:
    """开启所有可替换组件并保留模拟运行时。"""
    for name in ("vision", "thermal", "co2", "audio", "agent"):
        section = data[name]
        assert isinstance(section, dict)
        section["enabled"] = True
    data["vision"]["driver"] = "yolo"  # type: ignore[index]
    data["audio"]["input_driver"] = "vosk"  # type: ignore[index]
    data["audio"]["output_driver"] = "system_tts"  # type: ignore[index]
    # 配置加载阶段必须先满足智能体真实驱动的凭据契约；其他真实驱动
    # 仍用于验证工厂的 mock 强制覆盖。
    data["agent"]["driver"] = "mock"  # type: ignore[index]


def _real_vision(data: dict[str, object]) -> None:
    """切换为不会打开摄像头的真实 YOLO 构造配置。"""
    data["runtime"]["mode"] = "pi"  # type: ignore[index]
    data["vision"]["enabled"] = True  # type: ignore[index]
    data["vision"]["driver"] = "yolo"  # type: ignore[index]


def test_mock_mode_forces_all_drivers_and_avoids_real_implementations(tmp_path: Path) -> None:
    """mock 模式不因配置中的真实驱动加载相机、Vosk 或 TTS 实现。"""
    for name in ("src.vision.camera", "src.vision.yolo", "src.audio.vosk_asr", "src.audio.system_tts"):
        sys.modules.pop(name, None)
    components = ComponentFactory.build(_settings(tmp_path, mutate=_all_enabled_mock))
    assert all(getattr(components, name) is not None for name in
               ("vision", "thermal", "co2", "speech_input", "speech_output", "agent"))
    assert not any(name in sys.modules for name in (
        "src.vision.camera", "src.vision.yolo", "src.audio.vosk_asr", "src.audio.system_tts"))
    assert components.agent.reply("private query", {}, None).text == "当前为本地模拟模式。"


def test_settings_integration_builds_default_enabled_component(tmp_path: Path) -> None:
    """模板 Settings 可直接交给工厂，禁用组件保持 None。"""
    components = ComponentFactory.build(_settings(tmp_path))
    assert components.vision is not None
    assert components.thermal is components.co2 is components.speech_input is None
    assert components.speech_output is components.agent is None


def test_real_yolo_construction_is_lazy_and_does_not_open_camera(tmp_path: Path) -> None:
    """Zero 2 W 的真实摄像头配置仅保存参数，不加载 OpenCV 或设备。"""
    sys.modules.pop("cv2", None)
    components = ComponentFactory.build(_settings(tmp_path, mutate=_real_vision))
    assert type(components.vision).__name__ == "YoloEmotionPipeline"
    assert "cv2" not in sys.modules


@pytest.mark.parametrize(
    ("section", "field", "component", "available"),
    [
        ("vision", "driver", "vision", "mock, yolo"),
        ("audio", "input_driver", "audio input", "mock, vosk"),
        ("agent", "driver", "agent", "mock, tongji, tongji_mcp"),
    ],
)
def test_unknown_drivers_are_rejected_with_whitelist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, section: str, field: str,
    component: str, available: str,
) -> None:
    """任意未登记的驱动不会触发任意导入，且给出可用选项。"""
    def mutate(data: dict[str, object]) -> None:
        data["runtime"]["mode"] = "pi"  # type: ignore[index]
        data[section]["enabled"] = True  # type: ignore[index]
        data[section][field] = "not_registered"  # type: ignore[index]

    monkeypatch.setenv("DORM_ASSISTANT_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("DORM_ASSISTANT_AGENT_ID", "test-agent")
    with pytest.raises(ComponentFactoryError) as caught:
        ComponentFactory.build(_settings(tmp_path, mutate=mutate))
    assert component in str(caught.value)
    assert available in str(caught.value)
    assert "not_registered" in str(caught.value)


def test_blank_real_driver_is_rejected(tmp_path: Path) -> None:
    """空白驱动是配置错误，不能退化为任意实现。"""
    def mutate(data: dict[str, object]) -> None:
        data["runtime"]["mode"] = "pi"  # type: ignore[index]
        data["vision"]["driver"] = "  "  # type: ignore[index]

    with pytest.raises(ComponentFactoryError, match="driver must be a non-empty string"):
        ComponentFactory.build(_settings(tmp_path, mutate=mutate))


def test_constructor_failure_has_component_context_without_secret(tmp_path: Path) -> None:
    """错误只公开组件层诊断，配置中的秘密不进入对外消息。"""
    secret = "factory-secret-72931"

    def mutate(data: dict[str, object]) -> None:
        data["runtime"]["mode"] = "pi"  # type: ignore[index]
        data["vision"]["driver"] = "yolo"  # type: ignore[index]
        data["vision"]["camera"]["backend"] = secret  # type: ignore[index]

    with pytest.raises(ComponentFactoryError) as caught:
        ComponentFactory.build(_settings(tmp_path, mutate=mutate))
    assert str(caught.value) == "vision component could not be created"
    assert secret not in str(caught.value)


def test_registry_is_parseable_explicit_and_matches_factory_whitelists() -> None:
    """注册表使用固定名称列表，不提供模块路径或可执行表达式。"""
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    assert registry == {
        "vision": ["mock", "yolo"], "thermal": ["mock", "mlx90640"],
        "co2": ["mock", "mhz19"],
        "audio_input": ["mock", "vosk"], "audio_output": ["mock", "system_tts"],
        "agent": ["mock", "tongji", "tongji_mcp"],
    }
    assert all(all(isinstance(item, str) and "." not in item for item in values)
               for values in registry.values())


def test_factory_has_no_eval_exec_or_arbitrary_dynamic_import() -> None:
    """AST 审查禁止 eval/exec/importlib 与非白名单的运行时模块加载。"""
    tree = ast.parse(FACTORY.read_text(encoding="utf-8"))
    banned = {"eval", "exec", "__import__"}
    calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)]
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not (banned & set(calls))
    assert "importlib" not in FACTORY.read_text(encoding="utf-8")
    assert set(imports) <= {"__future__", "dataclasses", "datetime", "src.agent", "src.audio",
                            "src.co2", "src.domain", "src.thermal", "src.vision", "src.vision.camera",
                            "src.vision.yolo", "src.thermal.mlx90640", "src.co2.mhz19",
                            "src.audio.vosk_asr", "src.audio.system_tts", ".", None}


def test_invalid_settings_argument_is_rejected() -> None:
    """边界输入不能绕过 Settings 加载与冻结流程。"""
    with pytest.raises(TypeError, match="settings must be Settings"):
        ComponentFactory.build(object())  # type: ignore[arg-type]
