"""日程持久化：输入不可变提醒与带时区时间，输出本地 JSON 状态；依赖标准库和领域模型。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from src.domain.models import Reminder


def _aware(value: datetime, name: str) -> None:
    """校验带有效时区的时间。\n\n    Args: value: 待校验时间。name: 参数名。\n    Returns: 无。\n    Raises: TypeError: 值不是 datetime。ValueError: 时间缺少时区。"""
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _reminder_id(value: str) -> None:
    """校验非空提醒标识。\n\n    Args: value: 提醒标识。\n    Returns: 无。\n    Raises: TypeError: 值不是字符串。ValueError: 值为空白。"""
    if not isinstance(value, str):
        raise TypeError("reminder_id must be a string")
    if not value.strip():
        raise ValueError("reminder_id must not be empty")


def _encode(reminder: Reminder) -> dict[str, object]:
    """将提醒转换为稳定 JSON 字段。\n\n    Args: reminder: 要保存的领域提醒。\n    Returns: JSON 兼容字典。\n    Raises: TypeError: reminder 类型错误。"""
    if not isinstance(reminder, Reminder):
        raise TypeError("reminder must be a Reminder")
    return {"reminder_id": reminder.reminder_id, "message": reminder.message,
            "due_at": reminder.due_at.isoformat(), "acknowledged": reminder.acknowledged}


def _decode(value: object) -> Reminder:
    """将单个 JSON 对象解析为提醒。\n\n    Args: value: JSON 解码后的提醒对象。\n    Returns: 已校验的 Reminder。\n    Raises: ValueError: 字段缺失、格式或类型错误。"""
    if not isinstance(value, Mapping):
        raise ValueError("reminder entry must be an object")
    try:
        due_at = datetime.fromisoformat(value["due_at"])
        reminder = Reminder(value["reminder_id"], value["message"], due_at, value["acknowledged"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid reminder entry: {exc}") from exc
    return reminder


class ReminderStore:
    """线程安全、原子写入的本地提醒存储。"""

    def __init__(self, path: Path, logger: logging.Logger | None = None) -> None:
        """打开提醒文件并在损坏时恢复为空集合。\n\n        Args: path: JSON 存储文件。logger: 可选日志器。\n        Returns: 无。\n        Raises: TypeError: path 不是 Path。ValueError: path 指向非普通文件。"""
        if not isinstance(path, Path):
            raise TypeError("path must be a Path")
        self._path = path.expanduser().resolve()
        if self._path.exists() and not self._path.is_file():
            raise ValueError("path must name a regular file")
        self._logger, self._lock = logger or logging.getLogger(__name__), RLock()
        self._items: dict[str, Reminder] = {}
        self.reload()

    @property
    def path(self) -> Path:
        """返回解析后的存储路径。\n\n        Args: 无。\n        Returns: 绝对 JSON 文件路径。\n        Raises: 无。"""
        return self._path

    def reload(self) -> None:
        """从磁盘重新加载，绝不累计重复提醒。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无；读取或解析失败会备份损坏文件并恢复为空集合。"""
        with self._lock:
            try:
                self._items = self._read()
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                self._logger.error("日程文件损坏，已恢复为空集合: %s", exc)
                self._backup_corrupt()
                self._items = {}

    def add(self, reminder: Reminder) -> None:
        """按标识新增或替换提醒并原子保存。\n\n        Args: reminder: 要保存的不可变提醒。\n        Returns: 无。\n        Raises: TypeError: reminder 类型错误。OSError: 原子写入失败。"""
        _encode(reminder)
        with self._lock:
            candidate = dict(self._items)
            candidate[reminder.reminder_id] = reminder
            self._write(candidate)
            self._items = candidate

    def list(self) -> list[Reminder]:
        """列出当前唯一提醒。\n\n        Args: 无。\n        Returns: 按持久化顺序的提醒副本列表。\n        Raises: 无。"""
        with self._lock:
            return list(self._items.values())

    def due(self, now: datetime) -> list[Reminder]:
        """列出指定时间已到期且未确认的提醒。\n\n        Args: now: 带时区的比较时间。\n        Returns: 按到期时间和标识排序的到期提醒。\n        Raises: TypeError: now 类型错误。ValueError: now 缺少时区。"""
        _aware(now, "now")
        with self._lock:
            return sorted((item for item in self._items.values()
                           if not item.acknowledged and item.due_at <= now),
                          key=lambda item: (item.due_at, item.reminder_id))

    def acknowledge(self, reminder_id: str) -> None:
        """确认指定提醒，重复确认安全。\n\n        Args: reminder_id: 要确认的提醒标识。\n        Returns: 无；未知标识不产生变更。\n        Raises: TypeError: 标识类型错误。ValueError: 标识为空白。OSError: 写入失败。"""
        _reminder_id(reminder_id)
        with self._lock:
            current = self._items.get(reminder_id)
            if current is None or current.acknowledged:
                return
            candidate = dict(self._items)
            candidate[reminder_id] = replace(current, acknowledged=True)
            self._write(candidate)
            self._items = candidate

    def remove(self, reminder_id: str) -> bool:
        """删除指定提醒。\n\n        Args: reminder_id: 要删除的提醒标识。\n        Returns: 删除发生时为 True，未知标识时为 False。\n        Raises: TypeError: 标识类型错误。ValueError: 标识为空白。OSError: 写入失败。"""
        _reminder_id(reminder_id)
        with self._lock:
            if reminder_id not in self._items:
                return False
            candidate = dict(self._items)
            del candidate[reminder_id]
            self._write(candidate)
            self._items = candidate
            return True

    def delete(self, reminder_id: str) -> bool:
        """作为 remove 的语义别名删除提醒。\n\n        Args: reminder_id: 要删除的提醒标识。\n        Returns: 删除发生时为 True。\n        Raises: TypeError: 标识类型错误。ValueError: 标识为空白。OSError: 写入失败。"""
        return self.remove(reminder_id)

    def _read(self) -> dict[str, Reminder]:
        """读取并去重持久化内容。\n\n        Args: 无。\n        Returns: 以标识为键的唯一提醒字典。\n        Raises: OSError: 文件无法读取。ValueError: JSON 根或字段无效。"""
        if not self._path.exists():
            return {}
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            if raw.get("version") != 1 or not isinstance(raw.get("reminders"), list):
                raise ValueError("unsupported reminder store format")
            entries = raw["reminders"]
        elif isinstance(raw, list):
            entries = raw
        else:
            raise ValueError("reminder store root must be an object or list")
        items: dict[str, Reminder] = {}
        for entry in entries:
            reminder = _decode(entry)
            items[reminder.reminder_id] = reminder
        return items

    def _write(self, items: Mapping[str, Reminder]) -> None:
        """通过同目录临时文件原子替换存储。\n\n        Args: items: 即将持久化的唯一提醒映射。\n        Returns: 无。\n        Raises: OSError: 目录创建、写入或替换失败。"""
        payload = {"version": 1, "reminders": [_encode(item) for item in items.values()]}
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(data, encoding="utf-8")
            os.replace(temporary, self._path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _backup_corrupt(self) -> None:
        """将损坏文件移动为可审计备份。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无；备份失败仅记录日志。"""
        if not self._path.exists():
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = self._path.with_name(f"{self._path.name}.corrupt-{stamp}-{uuid4().hex}.bak")
        try:
            os.replace(self._path, backup)
        except OSError as exc:
            self._logger.error("无法备份损坏日程文件: %s", exc)
