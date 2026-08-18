"""JSONL 记录：输入领域事件，输出按日期滚动的隐私过滤 JSONL；依赖标准库和领域序列化。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timezone
import logging
from pathlib import Path
from threading import RLock

from src.domain.serialization import to_data, to_json


Clock = Callable[[], datetime]
_AUDIO_KEYS = frozenset({"audio", "raw_audio", "audio_data"})
_IMAGE_KEYS = frozenset({"image", "raw_image", "frame", "frames", "image_data"})


def _utc_now() -> datetime:
    """返回当前带 UTC 时区时间。\n\n    Args: 无。\n    Returns: 当前 UTC datetime。\n    Raises: 无。"""
    return datetime.now(timezone.utc)


def _as_date(value: object, fallback: datetime) -> date:
    """从事件时间字段解析写入日期。\n\n    Args: value: 事件中的 timestamp 值。fallback: 缺失或无效时间的后备时钟。\n    Returns: 带时区事件时间或后备时间对应的日期。\n    Raises: 无。"""
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                return parsed.date()
        except ValueError:
            pass
    return fallback.date()


def _filter(value: object, keep_dialogue: bool, keep_audio: bool,
            keep_images: bool, in_dialogue: bool = False) -> object:
    """递归移除隐私关闭的文本和原始媒体字段。\n\n    Args: value: JSON 兼容事件数据。keep_dialogue: 是否保留对话文本。keep_audio: 是否保留原始音频。keep_images: 是否保留原始图像。in_dialogue: 当前节点是否属于对话。\n    Returns: 已过滤的 JSON 兼容数据。\n    Raises: 无。"""
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        dialogue = (in_dialogue or "dialogue" in value or ("user_text" in value and "reply" in value)
                    or ("conversation_id" in value and "text" in value))
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            if key in _AUDIO_KEYS and not keep_audio:
                continue
            if key in _IMAGE_KEYS and not keep_images:
                continue
            if not keep_dialogue and key in {"user_text", "transcript"}:
                continue
            if not keep_dialogue and dialogue and key == "text":
                continue
            result[key] = _filter(item, keep_dialogue, keep_audio, keep_images,
                                  dialogue or key in {"dialogue", "reply"})
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_filter(item, keep_dialogue, keep_audio, keep_images, in_dialogue) for item in value]
    return value


class JsonlRecorder:
    """线程安全的按日期事件 JSONL 记录器。"""

    def __init__(
        self, directory: Path, *, rotate_daily: bool = True, persist_dialogue_text: bool = False,
        persist_raw_audio: bool = False, persist_raw_images: bool = False,
        clock: Clock = _utc_now, logger: logging.Logger | None = None,
    ) -> None:
        """保存配置但不创建目录或文件。\n\n        Args: directory: JSONL 输出目录。rotate_daily: 是否按事件日期分文件。persist_dialogue_text: 是否保留对话文字。persist_raw_audio: 是否保留原始音频字段。persist_raw_images: 是否保留原始图像字段。clock: 后备时钟。logger: 可选日志器。\n        Returns: 无。\n        Raises: TypeError: 路径、开关或时钟类型错误。"""
        if not isinstance(directory, Path):
            raise TypeError("directory must be a Path")
        if not all(isinstance(value, bool) for value in
                   (rotate_daily, persist_dialogue_text, persist_raw_audio, persist_raw_images)):
            raise TypeError("privacy and rotation options must be booleans")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._directory = directory.expanduser().resolve()
        self._rotate, self._dialogue = rotate_daily, persist_dialogue_text
        self._audio, self._images, self._clock = persist_raw_audio, persist_raw_images, clock
        self._logger, self._lock, self._closed = logger or logging.getLogger(__name__), RLock(), False

    @property
    def closed(self) -> bool:
        """查询记录器关闭状态。\n\n        Args: 无。\n        Returns: 已关闭时为 True。\n        Raises: 无。"""
        with self._lock:
            return self._closed

    def path_for(self, event_date: date) -> Path:
        """返回指定日期的目标 JSONL 路径。\n\n        Args: event_date: 事件所属日期。\n        Returns: 日期轮转或固定的 JSONL 路径。\n        Raises: TypeError: event_date 不是 date。"""
        if not isinstance(event_date, date):
            raise TypeError("event_date must be a date")
        name = f"{event_date.isoformat()}.jsonl" if self._rotate else "events.jsonl"
        return self._directory / name

    def write(self, event: object) -> bool:
        """过滤隐私字段并追加一条有效 JSON。\n\n        Args: event: 领域事件或 JSON 兼容值。\n        Returns: 成功写入为 True；关闭、序列化或 I/O 失败为 False。\n        Raises: 无；采集循环不因记录失败中断。"""
        with self._lock:
            if self._closed:
                return False
            try:
                fallback = self._clock()
                if not isinstance(fallback, datetime) or fallback.tzinfo is None or fallback.utcoffset() is None:
                    raise ValueError("clock must return a timezone-aware datetime")
                data = _filter(to_data(event), self._dialogue, self._audio, self._images)
                timestamp = data.get("timestamp") if isinstance(data, Mapping) else None
                target = self.path_for(_as_date(timestamp, fallback))
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(to_json(data) + "\n")
                return True
            except (OSError, TypeError, ValueError) as exc:
                self._logger.error("JSONL 记录失败: %s", exc)
                return False

    def close(self) -> None:
        """幂等关闭记录器。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无。"""
        with self._lock:
            self._closed = True
