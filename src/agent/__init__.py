"""智能体组件入口：输入为配置和结构化会话，输出为 AgentReply；依赖基础契约与同济适配器。"""

from .base import (AgentClient, AgentConfigurationError, AgentError,
                   AgentResponseError, AgentTransportError, HttpResponse, HttpTransport,
                   RequestsTransport)
from .tongji import TongjiAgentClient, TongjiContract
from .mcp import TongjiMcpAgentClient

__all__ = [
    "AgentClient",
    "AgentConfigurationError",
    "AgentError",
    "AgentResponseError",
    "AgentTransportError",
    "HttpResponse",
    "HttpTransport",
    "RequestsTransport",
    "TongjiAgentClient",
    "TongjiContract",
    "TongjiMcpAgentClient",
]
