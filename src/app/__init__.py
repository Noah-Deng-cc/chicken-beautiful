"""应用编排入口：输入为配置和可替换组件，输出为可停止的树莓派常驻服务。"""

from .orchestrator import Orchestrator

__all__ = ["Orchestrator"]
