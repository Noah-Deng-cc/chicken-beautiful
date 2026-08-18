"""领域数据模型：输入为各采集/服务结果，输出为不可变契约；仅依赖标准库。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite
from numbers import Real


class Emotion(str, Enum):
    """七类情绪的稳定标签。"""
    ANGRY = "angry"
    DISGUSTED = "disgusted"
    FEARFUL = "fearful"
    HAPPY = "happy"
    NEUTRAL = "neutral"
    SAD = "sad"
    SURPRISED = "surprised"


class Co2Level(str, Enum):
    """二氧化碳浓度等级。"""
    GOOD = "good"
    ELEVATED = "elevated"
    POOR = "poor"
    INVALID = "invalid"


def _require_aware(value: datetime, name: str) -> None:
    """校验时间包含有效时区。\n\n    Args: value: 待校验时间。name: 字段名。\n    Returns: 无。\n    Raises: TypeError: 不是 datetime。ValueError: 缺少时区。"""
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _require_number(value: Real, name: str, lower: float, upper: float) -> None:
    """校验有限数值及闭区间范围。\n\n    Args: value: 待校验值。name: 字段名。lower: 下界。upper: 上界。\n    Returns: 无。\n    Raises: TypeError: 值不是实数。ValueError: 值非有限或越界。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not isfinite(float(value)) or not lower <= float(value) <= upper:
        raise ValueError(f"{name} must be finite and within [{lower}, {upper}]")


def _require_text(value: str, name: str, allow_empty: bool = False) -> None:
    """校验字符串字段。\n\n    Args: value: 待校验值。name: 字段名。allow_empty: 是否允许空串。\n    Returns: 无。\n    Raises: TypeError: 值不是字符串。ValueError: 不允许的空串。"""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class EmotionReading:
    """单次七类情绪推理结果。"""
    timestamp: datetime
    dominant: Emotion
    confidence: float
    valence: float
    arousal: float
    person_id: str | None = None

    def __post_init__(self) -> None:
        """校验情绪读数。\n\n        Args: 无。\n        Returns: 无。\n        Raises: TypeError: 字段类型错误。ValueError: 标签未知、数值越界或时间无时区。"""
        _require_aware(self.timestamp, "timestamp")
        if not isinstance(self.dominant, Emotion):
            raise ValueError("dominant must be an Emotion")
        _require_number(self.confidence, "confidence", 0.0, 1.0)
        _require_number(self.valence, "valence", -1.0, 1.0)
        _require_number(self.arousal, "arousal", -1.0, 1.0)
        if self.person_id is not None:
            _require_text(self.person_id, "person_id")

    def to_dict(self) -> dict[str, object]:
        """转换为 JSON 兼容字典。\n\n        Args: 无。\n        Returns: 含 ISO8601 时间和情绪值的字典。\n        Raises: 无。"""
        return {"timestamp": self.timestamp.isoformat(), "dominant": self.dominant.value,
                "confidence": self.confidence, "valence": self.valence,
                "arousal": self.arousal, "person_id": self.person_id}


@dataclass(frozen=True, slots=True)
class TemperatureReading:
    """热阵列提取的温度读数，单位为摄氏度。"""
    timestamp: datetime
    maximum_celsius: float
    average_celsius: float
    quality: str = "good"

    def __post_init__(self) -> None:
        """校验温度读数。\n\n        Args: 无。\n        Returns: 无。\n        Raises: TypeError: 字段类型错误。ValueError: 温度越界、均值大于最大值或时间无时区。"""
        _require_aware(self.timestamp, "timestamp")
        _require_number(self.maximum_celsius, "maximum_celsius", -40.0, 300.0)
        _require_number(self.average_celsius, "average_celsius", -40.0, 300.0)
        if self.average_celsius > self.maximum_celsius:
            raise ValueError("average_celsius must not exceed maximum_celsius")
        _require_text(self.quality, "quality")


@dataclass(frozen=True, slots=True)
class Co2Reading:
    """单次二氧化碳浓度与分级结果。"""
    timestamp: datetime
    ppm: int | None
    level: Co2Level

    def __post_init__(self) -> None:
        """校验 CO2 读数。\n\n        Args: 无。\n        Returns: 无。\n        Raises: TypeError: ppm 不是整数。ValueError: ppm 越界、等级不匹配或时间无时区。"""
        _require_aware(self.timestamp, "timestamp")
        if not isinstance(self.level, Co2Level):
            raise ValueError("level must be a Co2Level")
        if self.ppm is None:
            if self.level is not Co2Level.INVALID:
                raise ValueError("missing ppm requires the invalid level")
        else:
            if isinstance(self.ppm, bool) or not isinstance(self.ppm, int):
                raise TypeError("ppm must be an integer or None")
            _require_number(self.ppm, "ppm", 0.0, 100_000.0)
            if self.level is Co2Level.INVALID:
                raise ValueError("a valid ppm cannot use the invalid level")


@dataclass(frozen=True, slots=True)
class Reminder:
    """可持久化的日程提醒。"""
    reminder_id: str
    message: str
    due_at: datetime
    acknowledged: bool = False

    def __post_init__(self) -> None:
        """校验提醒。\n\n        Args: 无。\n        Returns: 无。\n        Raises: TypeError: 字段类型错误。ValueError: 文本为空或时间无时区。"""
        _require_text(self.reminder_id, "reminder_id")
        _require_text(self.message, "message")
        _require_aware(self.due_at, "due_at")
        if not isinstance(self.acknowledged, bool):
            raise TypeError("acknowledged must be a boolean")


@dataclass(frozen=True, slots=True)
class AgentReply:
    """外部智能体返回的文本与会话标识。"""
    text: str
    timestamp: datetime
    conversation_id: str | None = None

    def __post_init__(self) -> None:
        """校验智能体回复。\n\n        Args: 无。\n        Returns: 无。\n        Raises: TypeError: 字段类型错误。ValueError: 文本为空或时间无时区。"""
        _require_text(self.text, "text")
        _require_aware(self.timestamp, "timestamp")
        if self.conversation_id is not None:
            _require_text(self.conversation_id, "conversation_id")


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    """一次用户输入与智能体回复。"""
    timestamp: datetime
    user_text: str
    reply: AgentReply

    def __post_init__(self) -> None:
        """校验对话轮次。\n\n        Args: 无。\n        Returns: 无。\n        Raises: TypeError: 回复类型错误。ValueError: 用户文本为空或时间无时区。"""
        _require_aware(self.timestamp, "timestamp")
        _require_text(self.user_text, "user_text")
        if not isinstance(self.reply, AgentReply):
            raise TypeError("reply must be an AgentReply")


@dataclass(frozen=True, slots=True)
class SystemSnapshot:
    """各模态最新状态组成的不可变系统快照。"""
    timestamp: datetime
    emotion: EmotionReading | None = None
    temperature: TemperatureReading | None = None
    co2: Co2Reading | None = None
    reminders: tuple[Reminder, ...] = ()
    dialogue: DialogueTurn | None = None

    def __post_init__(self) -> None:
        """校验快照成员。\n\n        Args: 无。\n        Returns: 无。\n        Raises: TypeError: 成员类型错误。ValueError: 时间无时区。"""
        _require_aware(self.timestamp, "timestamp")
        expected = ((self.emotion, EmotionReading, "emotion"),
                    (self.temperature, TemperatureReading, "temperature"),
                    (self.co2, Co2Reading, "co2"), (self.dialogue, DialogueTurn, "dialogue"))
        for value, kind, name in expected:
            if value is not None and not isinstance(value, kind):
                raise TypeError(f"{name} has an invalid type")
        if not isinstance(self.reminders, tuple) or not all(isinstance(item, Reminder) for item in self.reminders):
            raise TypeError("reminders must be a tuple of Reminder values")

    def to_dict(self) -> dict[str, object]:
        """递归转换为 JSON 兼容字典。\n\n        Args: 无。\n        Returns: 含全部模态和 ISO8601 时间的字典。\n        Raises: TypeError: 遇到不支持的嵌套值。"""
        from .serialization import to_data

        value = to_data(self)
        if not isinstance(value, dict):
            raise TypeError("snapshot serialization did not produce a dictionary")
        return value
