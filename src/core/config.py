"""配置加载：输入为 YAML 路径和环境变量，输出为不可变 Settings；依赖 pathlib、标准库和 PyYAML。"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

from . import AgentSettings, ConfigError, LoggingSettings, SectionSettings, Settings


_REQUIRED: Mapping[str, tuple[str, ...]] = {
    "runtime": ("mode", "timezone", "shutdown_timeout_seconds"),
    "vision": ("enabled", "driver", "model_path", "model_format", "device", "confidence_threshold", "sample_interval_seconds", "camera.source", "camera.backend", "camera.width", "camera.height", "camera.fps", "face_detection.enabled", "face_detection.cascade_path", "face_detection.scale_factor", "face_detection.min_neighbors", "face_detection.min_width", "face_detection.min_height", "mock.emotion", "mock.confidence"),
    "thermal": ("enabled", "driver", "sample_interval_seconds", "min_valid_celsius", "max_valid_celsius", "calibration.emissivity", "calibration.offset_celsius", "connection.bus", "connection.address", "connection.retries", "mock.maximum_celsius", "mock.average_celsius"),
    "co2": ("enabled", "driver", "sample_interval_seconds", "thresholds_ppm.elevated", "thresholds_ppm.poor", "connection.port", "connection.baud_rate", "connection.timeout_seconds", "connection.retries", "mock.ppm"),
    "audio": ("enabled", "input_driver", "output_driver", "language", "listen_timeout_seconds", "input.device", "input.sample_rate_hz", "input.vosk_model_path", "output.device", "output.command_argv", "output.timeout_seconds"),
    "agent": ("enabled", "driver", "base_url", "endpoint", "api_key_env", "agent_id_env", "connect_timeout_seconds", "read_timeout_seconds", "max_retries", "backoff_seconds", "stream"),
    "schedule": ("enabled", "store_path", "poll_interval_seconds", "default_timezone"),
    "storage": ("directory", "jsonl_enabled", "rotate_daily", "persist_dialogue_text", "persist_raw_audio", "persist_raw_images"),
    "logging": ("level", "console_enabled", "file_enabled", "directory", "filename", "max_bytes", "backup_count"),
}
_PATH_FIELDS = ("vision.model_path", "vision.face_detection.cascade_path", "audio.input.vosk_model_path", "schedule.store_path", "storage.directory", "logging.directory")


def _mapping(value: object, path: str) -> dict[str, object]:
    """校验并复制字符串键映射。

    Args:
        value: 待校验值。
        path: 用于错误信息的字段路径。
    Returns:
        可安全修改的字典副本。
    Raises:
        ConfigError: 值不是字符串键映射。
    """
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"configuration field '{path}' must be a mapping")
    return {str(key): item for key, item in value.items()}


def _value(root: Mapping[str, object], path: str) -> object:
    """读取必填的点分隔字段。

    Args:
        root: 配置根映射。
        path: 点分隔字段路径。
    Returns:
        字段值，允许显式 null。
    Raises:
        ConfigError: 任一级字段缺失或父节点不是映射。
    """
    current: object = root
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"missing required configuration field '{path}'")
        current = current[part]
    return current


def _typed(root: Mapping[str, object], path: str, expected: type[object]) -> object:
    """读取字段并校验基础类型。

    Args:
        root: 配置根映射。
        path: 点分隔字段路径。
        expected: 预期基础类型。
    Returns:
        类型匹配的字段值。
    Raises:
        ConfigError: 字段类型不正确。
    """
    value = _value(root, path)
    valid = isinstance(value, expected) and not (expected in (int, float) and isinstance(value, bool))
    if expected is float:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not valid:
        raise ConfigError(f"configuration field '{path}' must be {expected.__name__}")
    return value


def _freeze(value: object) -> object:
    """递归冻结配置容器。

    Args:
        value: YAML 产生的任意值。
    Returns:
        映射转为只读配置节、列表转为元组后的值。
    Raises:
        ConfigError: 映射键不是字符串。
    """
    if isinstance(value, Mapping):
        copied = _mapping(value, "<nested>")
        frozen = {key: _freeze(item) for key, item in copied.items()}
        return SectionSettings(MappingProxyType(frozen))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _resolve_path(root: dict[str, object], field_path: str, project_root: Path) -> None:
    """将配置中的文件路径解析到项目根目录。

    Args:
        root: 可修改的配置根字典。
        field_path: 点分隔路径字段。
        project_root: 项目根目录。
    Returns:
        无。
    Raises:
        ConfigError: 路径值不是非空字符串。
    """
    parts = field_path.split(".")
    raw = _typed(root, field_path, str)
    candidate = Path(cast(str, raw)).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
    container = root
    for part in parts[:-1]:
        nested = container.get(part)
        if not isinstance(nested, dict):
            raise ConfigError(f"configuration field '{field_path}' has a non-mapping parent")
        container = nested
    container[parts[-1]] = resolved


def _agent(root: Mapping[str, object]) -> AgentSettings:
    """构建并校验智能体配置。

    Args:
        root: 完整配置映射。
    Returns:
        不在表示中泄露秘密的智能体配置。
    Raises:
        ConfigError: 字段类型错误或真实驱动缺少环境变量。
    """
    api_env = cast(str, _typed(root, "agent.api_key_env", str))
    agent_env = cast(str, _typed(root, "agent.agent_id_env", str))
    api_key, agent_id = os.environ.get(api_env), os.environ.get(agent_env)
    enabled = cast(bool, _typed(root, "agent.enabled", bool))
    driver = cast(str, _typed(root, "agent.driver", str))
    requires_id = driver not in ("mock", "tongji_mcp")
    if enabled and driver != "mock" and (not api_key or (requires_id and not agent_id)):
        missing = "agent.api_key" if not api_key else "agent.agent_id"
        raise ConfigError(f"missing required configuration field '{missing}' (environment variable)")
    base_url = cast(str, _typed(root, "agent.base_url", str))
    endpoint = cast(str, _typed(root, "agent.endpoint", str))
    return AgentSettings(enabled, driver, base_url, endpoint, api_env, agent_env,
                         float(cast(float, _typed(root, "agent.connect_timeout_seconds", float))),
                         float(cast(float, _typed(root, "agent.read_timeout_seconds", float))),
                         cast(int, _typed(root, "agent.max_retries", int)),
                         float(cast(float, _typed(root, "agent.backoff_seconds", float))),
                         cast(bool, _typed(root, "agent.stream", bool)), api_key, agent_id)


def load_settings(path: Path) -> Settings:
    """加载 YAML、解析路径并注入环境变量中的智能体秘密。

    Args:
        path: YAML 配置文件路径。
    Returns:
        完全不可变的应用配置。
    Raises:
        ConfigError: 文件不可读、YAML 无效、必填字段缺失或类型错误。
    """
    source = path.expanduser().resolve()
    try:
        raw = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "<root>")
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"unable to load configuration '{source}': {exc}") from exc
    for section, fields in _REQUIRED.items():
        _mapping(_value(raw, section), section)
        for name in fields:
            _value(raw, f"{section}.{name}")
    project_root = source.parent.parent if source.parent.name == "config" else source.parent
    for field_path in _PATH_FIELDS:
        _resolve_path(raw, field_path, project_root)
    agent = _agent(raw)
    redact_values = tuple(value for value in (agent.api_key,) if value)
    logging = LoggingSettings(
        cast(str, _typed(raw, "logging.level", str)), cast(bool, _typed(raw, "logging.console_enabled", bool)),
        cast(bool, _typed(raw, "logging.file_enabled", bool)), cast(Path, _value(raw, "logging.directory")),
        cast(str, _typed(raw, "logging.filename", str)), cast(int, _typed(raw, "logging.max_bytes", int)),
        cast(int, _typed(raw, "logging.backup_count", int)), redact_values)
    sections = {name: cast(SectionSettings, _freeze(_value(raw, name)))
                for name in ("runtime", "vision", "thermal", "co2", "audio", "schedule", "storage")}
    return Settings(source, project_root, sections["runtime"], sections["vision"], sections["thermal"],
                    sections["co2"], sections["audio"], agent, sections["schedule"],
                    sections["storage"], logging)
