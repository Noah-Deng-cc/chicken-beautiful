"""日志初始化：输入为 LoggingSettings，输出为脱敏的控制台和轮转文件日志；仅依赖标准库。"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re

from . import LoggingSettings


_SENSITIVE_NAME = re.compile(r"(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)
_SENSITIVE_TEXT = re.compile(
    r"(?i)((?:authorization\s*[:=]\s*bearer|api[_-]?key|token|password)\s*[:=]?\s*)[^\s,;]+"
)


def _redact(message: str, secrets: tuple[str, ...]) -> str:
    """清除消息中的已知秘密和常见鉴权字段。

    Args:
        message: 原始日志文本。
        secrets: 需要逐字替换的秘密。
    Returns:
        已脱敏文本。
    Raises:
        无。
    """
    for secret in secrets:
        message = message.replace(secret, "<redacted>")
    return _SENSITIVE_TEXT.sub(r"\1<redacted>", message)


class RedactingFormatter(logging.Formatter):
    """对含异常堆栈的最终日志文本再次脱敏。"""

    def __init__(self, secrets: tuple[str, ...]) -> None:
        """初始化格式和秘密集合。

        Args:
            secrets: 需要逐字替换的秘密。
        Returns:
            无。
        Raises:
            无。
        """
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        """格式化并脱敏完整日志记录。

        Args:
            record: 待格式化日志记录。
        Returns:
            已脱敏的完整文本。
        Raises:
            无。
        """
        return _redact(super().format(record), self._secrets)


class RedactingFilter(logging.Filter):
    """清除日志消息中的环境秘密和常见鉴权字段。"""

    def __init__(self, secrets: tuple[str, ...]) -> None:
        """保存需要逐字替换的秘密。

        Args:
            secrets: 非空秘密值集合。
        Returns:
            无。
        Raises:
            无。
        """
        super().__init__()
        self._secrets = tuple(sorted(set(secrets), key=len, reverse=True))

    def filter(self, record: logging.LogRecord) -> bool:
        """原地脱敏一条日志记录。

        Args:
            record: 待输出的日志记录。
        Returns:
            始终为 True，允许记录继续输出。
        Raises:
            无。
        """
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            message = str(record.msg)
        record.msg = _redact(message, self._secrets)
        record.args = ()
        return True


def _environment_secrets() -> tuple[str, ...]:
    """发现当前进程中应从日志删除的秘密。

    Args:
        无。
    Returns:
        长度至少为四的敏感环境变量值。
    Raises:
        无。
    """
    return tuple(value for name, value in os.environ.items() if _SENSITIVE_NAME.search(name) and len(value) >= 4)


def _formatter(secrets: tuple[str, ...]) -> logging.Formatter:
    """创建统一日志格式器。

    Args:
        secrets: 需要从最终输出移除的秘密。
    Returns:
        含时间、级别、记录器和消息的格式器。
    Raises:
        无。
    """
    return RedactingFormatter(secrets)


def _attach(handler: logging.Handler, formatter: logging.Formatter, redactor: RedactingFilter) -> None:
    """标记并配置应用拥有的处理器。

    Args:
        handler: 待配置处理器。
        formatter: 日志格式器。
        redactor: 脱敏过滤器。
    Returns:
        无。
    Raises:
        无。
    """
    handler.setFormatter(formatter)
    handler.addFilter(redactor)
    setattr(handler, "_dorm_assistant_handler", True)
    logging.getLogger().addHandler(handler)


def configure_logging(settings: LoggingSettings) -> None:
    """幂等配置根记录器的控制台与按大小轮转文件输出。

    Args:
        settings: 已解析的日志配置。
    Returns:
        无。
    Raises:
        ValueError: 日志级别无效或文件参数不合法。
        OSError: 日志目录或文件无法创建。
    """
    level = logging.getLevelName(settings.level.upper())
    if not isinstance(level, int):
        raise ValueError(f"invalid logging level: {settings.level}")
    if settings.max_bytes <= 0 or settings.backup_count < 0:
        raise ValueError("logging rotation values must be non-negative and max_bytes must be positive")
    if Path(settings.filename).name != settings.filename:
        raise ValueError("logging filename must not contain path components")
    root = logging.getLogger()
    root.setLevel(level)
    for handler in tuple(root.handlers):
        if getattr(handler, "_dorm_assistant_handler", False):
            root.removeHandler(handler)
            handler.close()
    secrets = tuple(set(_environment_secrets() + settings.redact_values))
    formatter, redactor = _formatter(secrets), RedactingFilter(secrets)
    if settings.console_enabled:
        _attach(logging.StreamHandler(), formatter, redactor)
    if settings.file_enabled:
        directory = Path(settings.directory)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            directory / settings.filename, maxBytes=settings.max_bytes,
            backupCount=settings.backup_count, encoding="utf-8")
        _attach(file_handler, formatter, redactor)
