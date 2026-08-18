"""视觉管道抽象：输入为生命周期调用，输出情绪读数；依赖标准库 ABC 和领域模型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

from src.domain.models import EmotionReading


class VisionPipeline(ABC):
    """不泄漏摄像头或推理框架类型的视觉管道契约。

    实现应将可恢复的采集、断连和推理异常记录后转换为 ``read()`` 的
    ``None``；配置错误可由 ``start()`` 抛出。``close()`` 必须幂等。
    """

    @abstractmethod
    def start(self) -> None:
        """启动所需资源，重复调用应安全。\n\n        Args: 无。\n        Returns: 无。\n        Raises: RuntimeError: 配置错误或资源无法初始化。"""
        raise NotImplementedError

    @abstractmethod
    def read(self) -> EmotionReading | None:
        """读取一次最新情绪结果。\n\n        Args: 无。\n        Returns: 成功时返回 EmotionReading；无结果或可恢复故障时返回 None。\n        Raises: 无；实现应吸收并记录设备和推理异常。"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """幂等释放全部资源。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无；实现应记录而不是传播清理异常。"""
        raise NotImplementedError

    def __enter__(self) -> VisionPipeline:
        """启动管道并进入上下文。\n\n        Args: 无。\n        Returns: 已调用 start 的当前管道。\n        Raises: RuntimeError: start 无法初始化资源。"""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """离开上下文并关闭管道。\n\n        Args: exc_type: 异常类型。exc_value: 异常实例。traceback: 回溯对象。\n        Returns: 无，不抑制上下文异常。\n        Raises: 无；close 遵循异常安全契约。"""
        self.close()
