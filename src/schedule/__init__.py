"""日程包入口：输入本地提醒 JSON 与时间，输出提醒持久化接口；仅依赖标准库和领域模型。"""

from .store import ReminderStore

__all__ = ["ReminderStore"]
