"""音频组件入口：输入为语音超时或播报文本，输出为硬件无关接口和模拟实现；仅依赖标准库。"""

from .base import AudioComponent, InputItem, OutputOutcome, SpeechInput, SpeechOutput, WaitFunction
from .mock import MockSpeechInput, MockSpeechOutput

__all__ = [
    "AudioComponent",
    "InputItem",
    "MockSpeechInput",
    "MockSpeechOutput",
    "OutputOutcome",
    "SpeechInput",
    "SpeechOutput",
    "WaitFunction",
]
