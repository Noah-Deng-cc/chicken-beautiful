"""核心设施入口：输入为配置值，输出为不可变配置契约及加载、日志接口；依赖 pathlib 和标准库。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    """表示配置缺失、类型错误或无法读取。"""


@dataclass(frozen=True, slots=True)
class SectionSettings:
    """递归冻结的通用配置节。"""

    values: Mapping[str, object] = field(repr=False)

    def __getattr__(self, name: str) -> object:
        """按属性名读取配置值。

        Args:
            name: 配置键名。
        Returns:
            对应的只读配置值。
        Raises:
            AttributeError: 配置键不存在。
        """
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True, slots=True)
class AgentSettings:
    """外部智能体配置；敏感值不参与对象表示。"""

    enabled: bool
    driver: str
    base_url: str
    endpoint: str
    api_key_env: str
    agent_id_env: str
    connect_timeout_seconds: float
    read_timeout_seconds: float
    max_retries: int
    backoff_seconds: float
    stream: bool
    api_key: str | None = field(default=None, repr=False)
    agent_id: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    """控制台和轮转文件日志配置。"""

    level: str
    console_enabled: bool
    file_enabled: bool
    directory: Path
    filename: str
    max_bytes: int
    backup_count: int
    redact_values: tuple[str, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class Settings:
    """应用全部配置及解析后的项目位置。"""

    source_path: Path
    project_root: Path
    runtime: SectionSettings
    vision: SectionSettings
    thermal: SectionSettings
    co2: SectionSettings
    audio: SectionSettings
    agent: AgentSettings
    schedule: SectionSettings
    storage: SectionSettings
    logging: LoggingSettings


from .config import load_settings
from .logging import configure_logging

__all__ = [
    "AgentSettings",
    "ConfigError",
    "LoggingSettings",
    "SectionSettings",
    "Settings",
    "configure_logging",
    "load_settings",
]
