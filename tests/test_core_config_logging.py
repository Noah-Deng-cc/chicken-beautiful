"""T03 配置加载与日志的正常、边界和异常验收测试。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import logging
from pathlib import Path
from typing import Iterator

import pytest
import yaml

import src.core as core
from src.core import ConfigError, LoggingSettings, configure_logging, load_settings
from src.core.config import _REQUIRED


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "config" / "settings.example.yaml"


@pytest.fixture
def config_data() -> dict[str, object]:
    """返回完整模板配置的独立可修改副本。"""
    loaded = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture
def write_config(tmp_path: Path):
    """创建按项目根布局写入 YAML 的测试辅助函数。"""
    def _write(data: object, *, name: str = "config/settings.yaml") -> Path:
        path = tmp_path / "project" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def restore_root_logging() -> Iterator[None]:
    """在日志用例后恢复根记录器并关闭 T03 创建的处理器。"""
    root = logging.getLogger()
    original_handlers = tuple(root.handlers)
    original_level = root.level
    yield
    for handler in tuple(root.handlers):
        if handler not in original_handlers:
            root.removeHandler(handler)
            handler.close()
    root.handlers[:] = list(original_handlers)
    root.setLevel(original_level)


def _delete_path(data: dict[str, object], dotted_path: str) -> None:
    """从嵌套字典删除点分隔字段。"""
    parts = dotted_path.split(".")
    current = data
    for part in parts[:-1]:
        child = current[part]
        assert isinstance(child, dict)
        current = child
    del current[parts[-1]]


def _logging_settings(directory: Path, **overrides: object) -> LoggingSettings:
    """构造隔离的有效日志配置。"""
    values: dict[str, object] = {
        "level": "INFO",
        "console_enabled": False,
        "file_enabled": True,
        "directory": directory,
        "filename": "app.log",
        "max_bytes": 1024,
        "backup_count": 2,
        "redact_values": (),
    }
    values.update(overrides)
    return LoggingSettings(**values)  # type: ignore[arg-type]


def test_public_api_and_complete_template_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """完整模板可加载，且 T03 公共接口可从 src.core 导入。"""
    monkeypatch.delenv("DORM_ASSISTANT_AGENT_API_KEY", raising=False)
    monkeypatch.delenv("DORM_ASSISTANT_AGENT_ID", raising=False)
    settings = load_settings(TEMPLATE)
    assert core.load_settings is load_settings
    assert core.configure_logging is configure_logging
    assert settings.source_path == TEMPLATE.resolve()
    assert settings.project_root == PROJECT_ROOT
    assert settings.runtime.mode == "mock"
    assert settings.vision.camera.width == 640
    assert settings.audio.output.command_argv == ("espeak-ng", "-v", "cmn", "{text}")
    assert settings.agent.api_key is None


ALL_REQUIRED_PATHS = tuple(
    [section for section in _REQUIRED]
    + [f"{section}.{field}" for section, fields in _REQUIRED.items() for field in fields]
)


@pytest.mark.parametrize("missing_path", ALL_REQUIRED_PATHS)
def test_every_required_field_reports_full_path(
    missing_path: str,
    config_data: dict[str, object],
    write_config,
) -> None:
    """每个必填节和字段缺失时都明确报告完整字段路径。"""
    _delete_path(config_data, missing_path)
    path = write_config(config_data)
    with pytest.raises(ConfigError, match="missing required configuration field") as caught:
        load_settings(path)
    assert f"'{missing_path}'" in str(caught.value)


@pytest.mark.parametrize("root", [None, [], "text", 3, True])
def test_yaml_root_must_be_mapping(root: object, write_config) -> None:
    """空值和所有常见非映射 YAML 根类型均被拒绝。"""
    path = write_config(root)
    with pytest.raises(ConfigError, match=r"'<root>' must be a mapping"):
        load_settings(path)


def test_invalid_yaml_and_missing_file_are_wrapped(tmp_path: Path) -> None:
    """YAML 语法错误和文件读取错误均统一成为 ConfigError 并保留原因链。"""
    malformed = tmp_path / "bad.yaml"
    malformed.write_text("runtime: [unterminated", encoding="utf-8")
    with pytest.raises(ConfigError, match="unable to load configuration") as yaml_error:
        load_settings(malformed)
    assert isinstance(yaml_error.value.__cause__, yaml.YAMLError)
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError, match="unable to load configuration") as io_error:
        load_settings(missing)
    assert isinstance(io_error.value.__cause__, OSError)


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_type"),
    [
        ("vision.model_path", None, "str"),
        ("agent.enabled", 1, "bool"),
        ("agent.connect_timeout_seconds", True, "float"),
        ("agent.max_retries", 2.5, "int"),
        ("logging.console_enabled", "yes", "bool"),
        ("logging.max_bytes", True, "int"),
    ],
)
def test_typed_fields_reject_wrong_yaml_types(
    field: str,
    bad_value: object,
    expected_type: str,
    config_data: dict[str, object],
    write_config,
) -> None:
    """显式类型契约拒绝 null、字符串、布尔伪整数和错误数值类型。"""
    parts = field.split(".")
    current = config_data
    for part in parts[:-1]:
        child = current[part]
        assert isinstance(child, dict)
        current = child
    current[parts[-1]] = bad_value
    with pytest.raises(ConfigError, match=rf"'{field}' must be {expected_type}"):
        load_settings(write_config(config_data))


def test_relative_paths_resolve_from_project_root(config_data: dict[str, object], write_config) -> None:
    """五个相对路径字段均以 config 目录的父目录为项目根解析。"""
    path = write_config(config_data)
    project = path.parents[1].resolve()
    settings = load_settings(path)
    assert settings.vision.model_path == project / "data/models/emotion.onnx"
    assert settings.audio.input.vosk_model_path == project / "data/models/vosk-cn"
    assert settings.schedule.store_path == project / "data/schedule/reminders.json"
    assert settings.storage.directory == project / "data/runtime"
    assert settings.logging.directory == project / "logs"


def test_non_config_location_uses_source_parent_as_root(config_data: dict[str, object], write_config) -> None:
    """配置不在名为 config 的目录时，以配置文件所在目录为项目根。"""
    path = write_config(config_data, name="custom/settings.yaml")
    settings = load_settings(path)
    assert settings.project_root == path.parent.resolve()
    assert settings.storage.directory == path.parent.resolve() / "data/runtime"


def test_absolute_and_parent_traversal_paths_are_normalized(
    tmp_path: Path,
    config_data: dict[str, object],
    write_config,
) -> None:
    """绝对路径保持其目标，父目录穿越按当前策略规范化而不静默改写。"""
    absolute = (tmp_path / "external" / "models" / "emotion.onnx").resolve()
    vision = config_data["vision"]
    storage = config_data["storage"]
    assert isinstance(vision, dict) and isinstance(storage, dict)
    vision["model_path"] = str(absolute)
    storage["directory"] = "../outside-runtime"
    path = write_config(config_data)
    settings = load_settings(path)
    assert settings.vision.model_path == absolute
    assert settings.storage.directory == path.parents[2].resolve() / "outside-runtime"


def test_environment_credentials_are_injected_and_hidden_from_repr(
    monkeypatch: pytest.MonkeyPatch,
    config_data: dict[str, object],
    write_config,
) -> None:
    """环境变量覆盖注入两项凭据，且对象表示不泄漏任一值。"""
    api_key = "api-secret-837461"
    agent_id = "agent-secret-194752"
    agent = config_data["agent"]
    assert isinstance(agent, dict)
    agent.update({"enabled": True, "driver": "tongji"})
    monkeypatch.setenv("DORM_ASSISTANT_AGENT_API_KEY", api_key)
    monkeypatch.setenv("DORM_ASSISTANT_AGENT_ID", agent_id)
    settings = load_settings(write_config(config_data))
    assert settings.agent.api_key == api_key
    assert settings.agent.agent_id == agent_id
    assert api_key not in repr(settings)
    assert agent_id not in repr(settings)


@pytest.mark.parametrize(
    ("missing_name", "expected_path"),
    [
        ("DORM_ASSISTANT_AGENT_API_KEY", "agent.api_key"),
        ("DORM_ASSISTANT_AGENT_ID", "agent.agent_id"),
    ],
)
def test_enabled_real_agent_requires_each_environment_credential(
    missing_name: str,
    expected_path: str,
    monkeypatch: pytest.MonkeyPatch,
    config_data: dict[str, object],
    write_config,
) -> None:
    """真实智能体启用时，任一缺失凭据都报告稳定字段路径。"""
    agent = config_data["agent"]
    assert isinstance(agent, dict)
    agent.update({"enabled": True, "driver": "tongji"})
    monkeypatch.setenv("DORM_ASSISTANT_AGENT_API_KEY", "present-api-key")
    monkeypatch.setenv("DORM_ASSISTANT_AGENT_ID", "present-agent-id")
    monkeypatch.delenv(missing_name)
    with pytest.raises(ConfigError) as caught:
        load_settings(write_config(config_data))
    assert expected_path in str(caught.value)


def test_nested_settings_are_deeply_immutable(config_data: dict[str, object], write_config) -> None:
    """顶层、嵌套映射和 YAML 列表均不可在加载后修改。"""
    settings = load_settings(write_config(config_data))
    with pytest.raises(FrozenInstanceError):
        settings.runtime = settings.runtime  # type: ignore[misc]
    with pytest.raises(TypeError):
        settings.vision.values["driver"] = "yolo"  # type: ignore[index]
    with pytest.raises(TypeError):
        settings.vision.camera.values["width"] = 1  # type: ignore[index]
    assert isinstance(settings.audio.output.command_argv, tuple)
    with pytest.raises(TypeError):
        settings.audio.output.command_argv[0] = "sh"  # type: ignore[index]


def test_repeated_configuration_does_not_stack_owned_handlers(
    tmp_path: Path,
    restore_root_logging: None,
) -> None:
    """重复初始化会替换自有 handler，同时保留外部 handler。"""
    root = logging.getLogger()
    external = logging.NullHandler()
    root.addHandler(external)
    settings = _logging_settings(tmp_path, console_enabled=True)
    configure_logging(settings)
    configure_logging(settings)
    owned = [item for item in root.handlers if getattr(item, "_dorm_assistant_handler", False)]
    assert len(owned) == 2
    assert external in root.handlers


def test_file_log_rotates_at_size_boundary(tmp_path: Path, restore_root_logging: None) -> None:
    """超过 max_bytes 后保留当前文件和至少一个轮转备份。"""
    settings = _logging_settings(tmp_path, max_bytes=120, backup_count=2)
    configure_logging(settings)
    logger = logging.getLogger("t03.rotation")
    for index in range(12):
        logger.info("record-%02d-%s", index, "x" * 50)
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert (tmp_path / "app.log").is_file()
    assert (tmp_path / "app.log.1").is_file()
    assert len(list(tmp_path.glob("app.log*"))) <= 3


def test_secrets_are_redacted_in_all_logging_forms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_root_logging: None,
) -> None:
    """普通消息、格式化参数和异常栈中的配置/环境秘密均被脱敏。"""
    configured_secret = "configured-secret-923748"
    environment_secret = "environment-token-492731"
    monkeypatch.setenv("T03_ACCESS_TOKEN", environment_secret)
    settings = _logging_settings(tmp_path, redact_values=(configured_secret,))
    configure_logging(settings)
    logger = logging.getLogger("t03.redaction")
    logger.info("api_key=%s", configured_secret)
    logger.warning("ordinary value %s", environment_secret)
    try:
        raise RuntimeError(f"password={configured_secret}; token={environment_secret}")
    except RuntimeError:
        logger.exception("request failed for %s", configured_secret)
    for handler in logging.getLogger().handlers:
        handler.flush()
    content = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert configured_secret not in content
    assert environment_secret not in content
    assert "<redacted>" in content
    assert "Traceback (most recent call last)" in content


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"level": "NO_SUCH_LEVEL"}, "invalid logging level"),
        ({"max_bytes": 0}, "rotation values"),
        ({"backup_count": -1}, "rotation values"),
        ({"filename": "nested/app.log"}, "must not contain path components"),
        ({"filename": "../app.log"}, "must not contain path components"),
    ],
)
def test_invalid_logging_values_are_rejected(
    overrides: dict[str, object],
    message: str,
    tmp_path: Path,
    restore_root_logging: None,
) -> None:
    """非法级别、轮转边界和含路径成分的文件名被明确拒绝。"""
    with pytest.raises(ValueError, match=message):
        configure_logging(_logging_settings(tmp_path, **overrides))


def test_logging_directory_cannot_be_existing_file(
    tmp_path: Path,
    restore_root_logging: None,
) -> None:
    """日志目录指向普通文件时传播可诊断的文件系统异常。"""
    invalid_directory = tmp_path / "not-a-directory"
    invalid_directory.write_text("occupied", encoding="utf-8")
    with pytest.raises(OSError):
        configure_logging(_logging_settings(invalid_directory))
