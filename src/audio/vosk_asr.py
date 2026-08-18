"""Vosk 普通话输入：输入为 PCM 数据源和模型目录，输出为静音截断后的文本；依赖标准库并延迟导入 vosk。"""

from __future__ import annotations

from collections.abc import Callable
import importlib
import json
import logging
from pathlib import Path
from threading import Event, RLock
import time
from typing import Protocol, cast

from .base import SpeechInput, _duration
from .microphone import PcmSource, SoundDeviceMicrophone, _pcm_level


LOGGER = logging.getLogger(__name__)

class PcmRecognizer(Protocol):
    """不泄漏 Vosk 对象的流式识别器契约。"""

    def accept_waveform(self, pcm: bytes) -> str | None:
        """接收 PCM。\n\nArgs: pcm: int16 PCM。\nReturns: 片段 JSON/None。\nRaises: RuntimeError: 失败。"""
        ...

    def final_result(self) -> str:
        """返回最终结果。\n\nArgs: 无。\nReturns: JSON。\nRaises: RuntimeError: 失败。"""
        ...

class RecognizerFactory(Protocol):
    """可注入且可缓存模型的识别器工厂。"""

    def __call__(self, model_path: Path, sample_rate: int) -> PcmRecognizer:
        """创建识别器。\n\nArgs: model_path: 模型目录。sample_rate: 采样率。\nReturns: 识别器。\nRaises: RuntimeError: 加载失败。"""
        ...

class _VoskRecognizer:
    """把 Vosk 大写方法适配为内部契约。"""

    def __init__(self, recognizer: object) -> None:
        """保存识别器。\n\nArgs: recognizer: Vosk 实例。\nReturns: 无。\nRaises: 无。"""
        self._recognizer = recognizer

    def accept_waveform(self, pcm: bytes) -> str | None:
        """输入 PCM。\n\nArgs: pcm: PCM。\nReturns: JSON/None。\nRaises: RuntimeError: 失败。"""
        accepted = bool(getattr(self._recognizer, "AcceptWaveform")(pcm))
        return cast(str, getattr(self._recognizer, "Result")()) if accepted else None

    def final_result(self) -> str:
        """返回最终 JSON。\n\nArgs: 无。\nReturns: JSON。\nRaises: RuntimeError: 失败。"""
        return cast(str, getattr(self._recognizer, "FinalResult")())


class LazyVoskFactory:
    """首次创建识别器时才导入 Vosk 并加载一个共享模型。"""

    def __init__(self) -> None:
        """初始化缓存。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        self._model: object | None = None
        self._model_path: Path | None = None
        self._lock = RLock()

    def __call__(self, model_path: Path, sample_rate: int) -> PcmRecognizer:
        """延迟创建识别器。\n\nArgs: model_path: 模型目录。sample_rate: 采样率。\nReturns: 识别器。\nRaises: RuntimeError: 依赖/模型失败。"""
        with self._lock:
            if self._model is not None and self._model_path != model_path:
                raise RuntimeError("vosk factory cannot switch model paths")
            try:
                module = importlib.import_module("vosk")
                if self._model is None:
                    self._model = getattr(module, "Model")(str(model_path))
                    self._model_path = model_path
                raw = getattr(module, "KaldiRecognizer")(self._model, sample_rate)
            except Exception:
                raise RuntimeError("vosk model or recognizer could not be loaded") from None
        return _VoskRecognizer(raw)


def _result_text(payload: str) -> str | None:
    """解析结果 JSON。\n\nArgs: payload: JSON。\nReturns: 文本/None。\nRaises: 无。"""
    try:
        value = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        LOGGER.warning("vosk returned invalid JSON")
        return None
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        LOGGER.warning("vosk result did not contain a text field")
        return None
    return cast(str, value["text"]).strip() or None


class VoskSpeechInput(SpeechInput):
    """使用简单能量 VAD 和静音终止的本地普通话输入。"""

    def __init__(self, model_path: Path, *, device: int | str | None = None, sample_rate: int = 16000,
                 block_size: int = 1600, silence_seconds: float = 1.0,
                 max_utterance_seconds: float = 15.0, vad_threshold: float = 300.0,
                 queue_blocks: int = 8, source: PcmSource | None = None,
                 recognizer_factory: RecognizerFactory | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """保存参数且不加载设备/模型。\n\nArgs: model_path: 模型。device: 设备。sample_rate: 采样率。block_size: 帧数。silence_seconds: 静音。max_utterance_seconds: 最长语句。vad_threshold: 阈值。queue_blocks: 队列。source: PCM 源。recognizer_factory: 工厂。clock: 时钟。\nReturns: 无。\nRaises: TypeError: 类型错误。ValueError: 数值无效。"""
        if not isinstance(model_path, Path):
            raise TypeError("model_path must be a Path")
        for name, value in (("sample_rate", sample_rate), ("block_size", block_size), ("queue_blocks", queue_blocks)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self._silence = _duration(silence_seconds, "silence_seconds", allow_zero=False)
        self._maximum = _duration(max_utterance_seconds, "max_utterance_seconds", allow_zero=False)
        self._threshold = _duration(vad_threshold, "vad_threshold", allow_zero=True)
        self._model_path, self._sample_rate = model_path.expanduser().resolve(), sample_rate
        self._source = source if source is not None else SoundDeviceMicrophone(
            device=device, sample_rate=sample_rate, block_size=block_size, queue_blocks=queue_blocks)
        self._factory = recognizer_factory if recognizer_factory is not None else LazyVoskFactory()
        self._clock, self._cancelled, self._lock = clock, Event(), RLock()
        self._listen_lock, self._closed = RLock(), False

    @property
    def closed(self) -> bool:
        """返回关闭状态。\n\nArgs: 无。\nReturns: 已关闭为 True。\nRaises: 无。"""
        with self._lock:
            return self._closed

    def listen(self, timeout: float) -> str | None:
        """采集并识别。\n\nArgs: timeout: 正总等待秒数。\nReturns: 文本或 None。\nRaises: TypeError: 类型错误。ValueError: 非正。RuntimeError: 已关闭。"""
        limit = _duration(timeout, "timeout", allow_zero=False)
        with self._listen_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("vosk speech input is closed")
                self._cancelled.clear()
            if not self._model_path.is_dir():
                LOGGER.error("vosk model directory is missing")
                return None
            try:
                recognizer = self._factory(self._model_path, self._sample_rate)
                self._source.start()
                return self._capture(recognizer, limit)
            except Exception:
                LOGGER.error("local speech recognition failed")
                return None

    def _capture(self, recognizer: PcmRecognizer, timeout: float) -> str | None:
        """执行有界识别。\n\nArgs: recognizer: 识别器。timeout: 总秒数。\nReturns: 文本/None。\nRaises: RuntimeError: 数据源/识别失败。"""
        deadline, speech, silence, elapsed = self._clock() + timeout, False, 0.0, 0.0
        texts: list[str] = []
        while not self._cancelled.is_set():
            remaining = deadline - self._clock()
            if remaining <= 0:
                break
            pcm = self._source.read(min(0.25, remaining))
            if pcm is None:
                continue
            level = _pcm_level(pcm)
            if level is None:
                LOGGER.warning("microphone returned an invalid PCM block")
                continue
            duration = len(pcm) / (self._sample_rate * 2)
            voiced = level >= self._threshold
            if voiced:
                speech, silence = True, 0.0
            elif speech:
                silence += duration
            if not speech:
                continue
            elapsed += duration
            result = recognizer.accept_waveform(pcm)
            text = _result_text(result) if result is not None else None
            if text:
                texts.append(text)
            if silence >= self._silence or elapsed >= self._maximum:
                break
        if not speech or self._cancelled.is_set():
            return None
        final = _result_text(recognizer.final_result())
        if final:
            texts.append(final)
        return " ".join(texts).strip() or None

    def cancel(self) -> None:
        """中断读取。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        self._cancelled.set()
        try:
            self._source.cancel()
        except Exception:
            LOGGER.warning("PCM source cancellation failed")

    def close(self) -> None:
        """关闭并唤醒。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.cancel()
        try:
            self._source.close()
        except Exception:
            LOGGER.warning("PCM source close failed")
