"""音频基础契约：输入为超时或播报文本，输出为识别文本/播报状态；仅依赖标准库，不泄漏硬件库类型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from math import isfinite
from types import TracebackType


InputItem = str | None | Exception
OutputOutcome = bool | Exception
WaitFunction = Callable[[float], bool]


def _duration(value: float, name: str, *, allow_zero: bool) -> float:
    """校验有限时长。

    Args: value: 待校验秒数。name: 参数名。allow_zero: 是否允许零。
    Returns: 浮点秒数。
    Raises: TypeError: 不是数值。ValueError: 非有限、负数或不允许的零。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return result


def _input_source(source: InputItem | Iterable[InputItem]) -> tuple[bool, tuple[InputItem, ...]]:
    """标准化固定或序列输入。

    Args: source: 单个固定结果或结果序列。
    Returns: 是否固定及不可变结果元组。
    Raises: TypeError: 来源或序列元素类型错误。
    """
    if source is None or isinstance(source, (str, Exception)):
        return True, (source,)
    try:
        items = tuple(source)
    except TypeError:
        raise TypeError("responses must be an input item or iterable") from None
    if not all(item is None or isinstance(item, (str, Exception)) for item in items):
        raise TypeError("response items must be strings, None, or exceptions")
    return False, items


def _output_source(source: OutputOutcome | Iterable[OutputOutcome]) -> tuple[bool, tuple[OutputOutcome, ...]]:
    """标准化固定或序列播报结果。

    Args: source: 单个固定结果或结果序列。
    Returns: 是否固定及不可变结果元组。
    Raises: TypeError: 来源或序列元素不是 bool/Exception。
    """
    if isinstance(source, (bool, Exception)):
        return True, (source,)
    try:
        items = tuple(source)
    except TypeError:
        raise TypeError("outcomes must be a bool, exception, or iterable") from None
    if not all(isinstance(item, (bool, Exception)) for item in items):
        raise TypeError("outcome items must be booleans or exceptions")
    return False, items


class AudioComponent(ABC):
    """具有取消、关闭和上下文管理能力的音频组件。"""

    @property
    @abstractmethod
    def closed(self) -> bool:
        """返回组件是否已关闭。

        Args: 无。
        Returns: 已关闭时为 True。
        Raises: 无。
        """
        raise NotImplementedError

    @abstractmethod
    def cancel(self) -> None:
        """尽力中断当前阻塞操作且不影响下一次调用。

        Args: 无。
        Returns: 无。
        Raises: 无。
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """幂等关闭资源并取消当前操作。

        Args: 无。
        Returns: 无。
        Raises: 无。
        """
        raise NotImplementedError

    def __enter__(self) -> AudioComponent:
        """进入上下文并拒绝复用已关闭组件。

        Args: 无。
        Returns: 当前音频组件。
        Raises: RuntimeError: 组件已关闭。
        """
        if self.closed:
            raise RuntimeError("audio component is closed")
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        """退出上下文并关闭组件。

        Args: exc_type: 异常类型。exc_value: 异常对象。traceback: 回溯对象。
        Returns: 无。
        Raises: 无。
        """
        self.close()


class SpeechInput(AudioComponent):
    """麦克风/VAD/ASR 的硬件无关输入接口。"""

    @abstractmethod
    def listen(self, timeout: float) -> str | None:
        """在超时内返回非空识别文本。

        Args: timeout: 正有限等待秒数。
        Returns: 识别文本；静音、空白、超时或取消时为 None。
        Raises: TypeError: 超时类型错误。ValueError: 超时或实现定义的文本长度无效。
            RuntimeError: 组件已关闭或底层识别失败。
        """
        raise NotImplementedError


class SpeechOutput(AudioComponent):
    """TTS 与音响播放的硬件无关输出接口。"""

    @abstractmethod
    def speak(self, text: str) -> bool:
        """同步播报非空文本。

        Args: text: 待播报文本。
        Returns: 播报成功时为 True；空白或可恢复失败时为 False。
        Raises: TypeError: 文本类型错误。ValueError: 文本超过实现定义的安全上限。
            RuntimeError: 组件已关闭或底层播放异常。
        """
        raise NotImplementedError
