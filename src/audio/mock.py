"""音频模拟器：输入为固定/序列结果、延迟与故障，输出为确定性识别和线程安全播报记录；仅依赖标准库。"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from threading import Event, RLock

from .base import (InputItem, OutputOutcome, SpeechInput, SpeechOutput, WaitFunction,
                   _duration, _input_source, _output_source)


class MockSpeechInput(SpeechInput):
    """支持固定/序列/空值/异常和可控延迟的语音输入。"""

    def __init__(self, responses: InputItem | Iterable[InputItem] = None, *, delay_seconds: float = 0.0,
                 max_text_chars: int = 4096, wait: WaitFunction | None = None) -> None:
        """创建不打开设备的输入模拟器。

        Args: responses: 固定结果或按次消费且耗尽返回 None 的序列。delay_seconds: 每次结果延迟。
            max_text_chars: 文本上限。wait: 返回 True 表示取消的可注入等待函数。
        Returns: 无。
        Raises: TypeError: 参数类型错误。ValueError: 延迟或上限无效。
        """
        fixed, items = _input_source(responses)
        if isinstance(max_text_chars, bool) or not isinstance(max_text_chars, int):
            raise TypeError("max_text_chars must be an integer")
        if max_text_chars <= 0:
            raise ValueError("max_text_chars must be positive")
        self._fixed, self._items = fixed, deque(items)
        self._delay = _duration(delay_seconds, "delay_seconds", allow_zero=True)
        self._max_text_chars, self._wait = max_text_chars, wait
        self._cancelled, self._lock, self._listen_lock = Event(), RLock(), RLock()
        self._closed = False

    @property
    def closed(self) -> bool:
        """线程安全返回关闭状态。

        Args: 无。
        Returns: 已关闭时为 True。
        Raises: 无。
        """
        with self._lock:
            return self._closed

    def listen(self, timeout: float) -> str | None:
        """等待模拟结果，超时/取消不消费序列项。

        Args: timeout: 正有限等待秒数。
        Returns: 去除首尾空白的文本；空白、耗尽、超时或取消为 None。
        Raises: TypeError: 超时类型错误。ValueError: 超时或文本过长。RuntimeError: 已关闭。
            Exception: 原样抛出注入异常。
        """
        limit = _duration(timeout, "timeout", allow_zero=False)
        with self._listen_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("speech input is closed")
                self._cancelled.clear()
            wait_for = min(self._delay, limit)
            cancelled = self._cancelled.wait(wait_for) if self._wait is None else bool(self._wait(wait_for))
            with self._lock:
                if cancelled or self._cancelled.is_set() or self._closed or self._delay > limit:
                    return None
                item = self._items[0] if self._fixed and self._items else (
                    self._items.popleft() if self._items else None)
            if isinstance(item, Exception):
                raise item
            if item is None or not item.strip():
                return None
            text = item.strip()
            if len(text) > self._max_text_chars:
                raise ValueError("recognized text exceeds max_text_chars")
            return text

    def cancel(self) -> None:
        """中断当前等待且不消费结果。

        Args: 无。
        Returns: 无。
        Raises: 无。
        """
        self._cancelled.set()

    def close(self) -> None:
        """幂等关闭并中断当前等待。

        Args: 无。
        Returns: 无。
        Raises: 无。
        """
        with self._lock:
            self._closed = True
            self._cancelled.set()


class MockSpeechOutput(SpeechOutput):
    """线程安全记录成功播报并支持固定/序列失败或异常。"""

    def __init__(self, outcomes: OutputOutcome | Iterable[OutputOutcome] = True,
                 *, max_text_chars: int = 4096) -> None:
        """创建无设备输出模拟器。

        Args: outcomes: 固定或按次消费且耗尽回退成功的结果。max_text_chars: 文本上限。
        Returns: 无。
        Raises: TypeError: 参数类型错误。ValueError: 上限非正。
        """
        fixed, items = _output_source(outcomes)
        if isinstance(max_text_chars, bool) or not isinstance(max_text_chars, int):
            raise TypeError("max_text_chars must be an integer")
        if max_text_chars <= 0:
            raise ValueError("max_text_chars must be positive")
        self._fixed, self._outcomes = fixed, deque(items)
        self._max_text_chars, self._closed = max_text_chars, False
        self._spoken: list[str] = []
        self._attempted: list[str] = []
        self._lock = RLock()

    @property
    def closed(self) -> bool:
        """线程安全返回关闭状态。

        Args: 无。
        Returns: 已关闭时为 True。
        Raises: 无。
        """
        with self._lock:
            return self._closed

    @property
    def spoken_texts(self) -> tuple[str, ...]:
        """返回成功播报文本的不可变快照。

        Args: 无。
        Returns: 按成功顺序排列的文本元组。
        Raises: 无。
        """
        with self._lock:
            return tuple(self._spoken)

    @property
    def attempted_texts(self) -> tuple[str, ...]:
        """返回全部有效播报尝试的不可变快照。

        Args: 无。
        Returns: 按调用顺序排列的文本元组。
        Raises: 无。
        """
        with self._lock:
            return tuple(self._attempted)

    def speak(self, text: str) -> bool:
        """记录一次播报并应用注入结果。

        Args: text: 待播报文本。
        Returns: 成功为 True，空白或注入失败为 False。
        Raises: TypeError: 文本类型错误。ValueError: 文本过长。RuntimeError: 已关闭。
            Exception: 原样抛出注入异常。
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = text.strip()
        if not normalized:
            return False
        if len(normalized) > self._max_text_chars:
            raise ValueError("speech text exceeds max_text_chars")
        with self._lock:
            if self._closed:
                raise RuntimeError("speech output is closed")
            self._attempted.append(normalized)
            outcome = self._outcomes[0] if self._fixed and self._outcomes else (
                self._outcomes.popleft() if self._outcomes else True)
            if isinstance(outcome, Exception):
                raise outcome
            if outcome:
                self._spoken.append(normalized)
            return outcome

    def cancel(self) -> None:
        """模拟器无阻塞播放，因此取消为空操作。

        Args: 无。
        Returns: 无。
        Raises: 无。
        """

    def close(self) -> None:
        """幂等关闭输出模拟器。

        Args: 无。
        Returns: 无。
        Raises: 无。
        """
        with self._lock:
            self._closed = True
