"""融合组件入口：输入为系统快照和本地规则，输出为结构化上下文及本地告警；不依赖云端。"""

from .service import FusionRules, FusionService

__all__ = ["FusionRules", "FusionService"]
