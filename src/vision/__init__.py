"""视觉包入口：输入为无，输出视觉抽象和模拟实现；仅依赖标准库及领域模型。"""

from .base import VisionPipeline
from .mock import MockResult, MockVisionPipeline

__all__ = ["MockResult", "MockVisionPipeline", "VisionPipeline"]
