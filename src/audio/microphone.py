"""麦克风 PCM 采集：输入为设备/采样参数，输出为有限队列中的单声道 int16 字节块；依赖标准库并延迟导入 sounddevice。"""

from __future__ import annotations

from array import array
import importlib
import logging
from queue import Empty, Full, Queue
import sys
from threading import Event, RLock
from typing import Protocol

from .base import _duration


LOGGER = logging.getLogger(__name__)


def _pcm_level(pcm: bytes) -> float | None:
    """计算 PCM 振幅。\n\nArgs: pcm: little-endian int16 PCM。\nReturns: 平均振幅/None。\nRaises: 无。"""
    if not pcm or len(pcm) % 2:
        return None
    samples = array("h")
    try:
        samples.frombytes(pcm)
    except (BufferError, ValueError):
        return None
    if sys.byteorder != "little":
        samples.byteswap()
    return sum(abs(value) for value in samples) / len(samples) if samples else None


class PcmSource(Protocol):
    """与音频库无关的 PCM 数据源。"""

    def start(self) -> None:
        """启动或重置采集。\n\nArgs: 无。\nReturns: 无。\nRaises: RuntimeError: 数据源关闭、设备占用或初始化失败。"""
        ...

    def read(self, timeout: float) -> bytes | None:
        """读取 PCM 块。\n\nArgs: timeout: 正有限等待秒数。\nReturns: PCM 字节；超时/取消为 None。\nRaises: TypeError: 类型错误。ValueError: 非正。RuntimeError: 已关闭。"""
        ...

    def cancel(self) -> None:
        """唤醒读取。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        ...

    def close(self) -> None:
        """幂等关闭。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        ...


class SoundDeviceMicrophone:
    """使用 sounddevice.RawInputStream 的有界 PCM 数据源。"""

    def __init__(self, *, device: int | str | None = None, sample_rate: int = 16000,
                 block_size: int = 1600, queue_blocks: int = 8) -> None:
        """保存参数且不打开设备。\n\nArgs: device: 设备。sample_rate: 采样率。block_size: 帧数。queue_blocks: 队列块数。\nReturns: 无。\nRaises: TypeError: 类型错误。ValueError: 参数非正。"""
        for name, value in (("sample_rate", sample_rate), ("block_size", block_size),
                            ("queue_blocks", queue_blocks)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if device is not None and (isinstance(device, bool) or not isinstance(device, (int, str))):
            raise TypeError("device must be an integer, string, or None")
        self._device, self._sample_rate, self._block_size = device, sample_rate, block_size
        self._queue: Queue[bytes | None] = Queue(maxsize=queue_blocks)
        self._cancelled, self._lock = Event(), RLock()
        self._stream: object | None = None
        self._started = False
        self._closed = False

    @property
    def closed(self) -> bool:
        """返回关闭状态。\n\nArgs: 无。\nReturns: 已关闭为 True。\nRaises: 无。"""
        with self._lock:
            return self._closed

    def _put_latest(self, item: bytes | None) -> None:
        """写入最新块。\n\nArgs: item: PCM/哨兵。\nReturns: 无。\nRaises: 无。"""
        try:
            self._queue.put_nowait(item)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except Full:
                pass

    def _callback(self, input_data: object, frames: int,
                  time_info: object, status: object) -> None:
        """复制回调数据。\n\nArgs: input_data: 缓冲区。frames: 帧数。time_info: 时间。status: 状态。\nReturns: 无。\nRaises: 无。"""
        del frames, time_info
        if status:
            LOGGER.warning("microphone callback reported a device status")
        if self._cancelled.is_set() or self._closed:
            return
        try:
            self._put_latest(bytes(input_data))
        except (TypeError, ValueError):
            LOGGER.warning("microphone callback received an invalid PCM block")

    def _drain(self) -> None:
        """清空遗留块。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return

    def start(self) -> None:
        """延迟打开/重置流。\n\nArgs: 无。\nReturns: 无。\nRaises: RuntimeError: 已关闭、依赖缺失或设备失败。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("microphone source is closed")
            self._cancelled.clear()
            self._drain()
            if self._started:
                return
            try:
                module = importlib.import_module("sounddevice")
                stream_type = getattr(module, "RawInputStream")
                self._stream = stream_type(samplerate=self._sample_rate, blocksize=self._block_size,
                                           device=self._device, channels=1, dtype="int16",
                                           callback=self._callback)
                getattr(self._stream, "start")()
                self._started = True
            except Exception:
                stream, self._stream = self._stream, None
                if stream is not None:
                    try:
                        getattr(stream, "close")()
                    except Exception:
                        pass
                LOGGER.error("microphone could not be opened")
                raise RuntimeError("microphone could not be opened") from None

    def read(self, timeout: float) -> bytes | None:
        """读取有限队列。\n\nArgs: timeout: 正等待秒数。\nReturns: PCM 或 None。\nRaises: TypeError: 类型错误。ValueError: 非正。RuntimeError: 已关闭。"""
        limit = _duration(timeout, "timeout", allow_zero=False)
        if self.closed:
            raise RuntimeError("microphone source is closed")
        if self._cancelled.is_set():
            return None
        try:
            item = self._queue.get(timeout=limit)
        except Empty:
            return None
        return None if self._cancelled.is_set() else item

    def cancel(self) -> None:
        """取消并唤醒。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        self._cancelled.set()
        self._put_latest(None)

    def close(self) -> None:
        """幂等关闭流。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.cancel()
            stream, self._stream, self._started = self._stream, None, False
        if stream is not None:
            for method_name in ("stop", "close"):
                try:
                    getattr(stream, method_name)()
                except Exception:
                    LOGGER.warning("microphone stream %s failed", method_name)
