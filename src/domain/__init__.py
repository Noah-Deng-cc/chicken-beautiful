"""领域包入口：输入为无；输出公共数据契约与序列化函数；仅依赖标准库。"""

from .models import (
    AgentReply,
    Co2Level,
    Co2Reading,
    DialogueTurn,
    Emotion,
    EmotionReading,
    Reminder,
    SystemSnapshot,
    TemperatureReading,
)
from .serialization import Serializable, to_json

__all__ = [
    "AgentReply",
    "Co2Level",
    "Co2Reading",
    "DialogueTurn",
    "Emotion",
    "EmotionReading",
    "Reminder",
    "Serializable",
    "SystemSnapshot",
    "TemperatureReading",
    "to_json",
]
