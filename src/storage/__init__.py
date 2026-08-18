"""存储包入口：输入领域事件，输出按日期轮转的 JSONL 记录器；仅依赖标准库和领域模型。"""

from .jsonl import JsonlRecorder

__all__ = ["JsonlRecorder"]
