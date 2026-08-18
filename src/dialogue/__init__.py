"""对话组件入口：输入为语音、状态与智能体，输出为已保存的对话轮次；不记录敏感全文。"""

from .service import DialogueService

__all__ = ["DialogueService"]
