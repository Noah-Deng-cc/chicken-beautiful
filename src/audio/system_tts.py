"""系统 TTS 播报：输入为文本和安全 argv 模板，输出为进程成功状态；依赖标准库并默认调用 Zero 2 W 的 espeak-ng。"""

from __future__ import annotations

from collections.abc import Sequence
import logging
import subprocess
from threading import Event, Lock, RLock
from typing import Protocol, cast

from .base import SpeechOutput, _duration


LOGGER = logging.getLogger(__name__)


class ProcessHandle(Protocol):
    """不泄漏具体 subprocess 泛型的进程契约。"""

    def poll(self) -> int | None:
        """查询退出码。\n\nArgs: 无。\nReturns: 退出码或 None。\nRaises: OSError: 查询失败。"""
        ...

    def wait(self, timeout: float | None = None) -> int:
        """等待进程。\n\nArgs: timeout: 最长秒数。\nReturns: 退出码。\nRaises: TimeoutExpired: 超时。"""
        ...

    def terminate(self) -> None:
        """请求终止。\n\nArgs: 无。\nReturns: 无。\nRaises: OSError: 终止失败。"""
        ...

    def kill(self) -> None:
        """强制终止。\n\nArgs: 无。\nReturns: 无。\nRaises: OSError: 强制终止失败。"""
        ...


class PopenFactory(Protocol):
    """可注入的无 shell 进程工厂。"""

    def __call__(self, args: Sequence[str], *, shell: bool, stdin: int,
                 stdout: int, stderr: int) -> ProcessHandle:
        """创建子进程。\n\nArgs: args: argv。shell: 必须为 False。stdin: 标准输入。stdout: 标准输出。stderr: 标准错误。\nReturns: 进程句柄。\nRaises: OSError: 程序缺失或创建失败。"""
        ...


class SystemSpeechOutput(SpeechOutput):
    """通过安全 argv 调用系统 TTS 的并发安全输出。"""

    def __init__(self, command_argv: Sequence[str] = ("espeak-ng", "-v", "cmn", "{text}"), *,
                 timeout_seconds: float = 20.0, terminate_timeout_seconds: float = 2.0,
                 max_text_chars: int = 4096, popen_factory: PopenFactory | None = None) -> None:
        """保存命令模板但不启动进程。\n\nArgs: command_argv: 含唯一文本占位符的 argv。timeout_seconds: 播报超时。terminate_timeout_seconds: 终止宽限。max_text_chars: 文本上限。popen_factory: fake/真实工厂。\nReturns: 无。\nRaises: TypeError: 参数类型错误。ValueError: 模板、时长或上限无效。"""
        if isinstance(command_argv, (str, bytes)) or not isinstance(command_argv, Sequence):
            raise TypeError("command_argv must be a sequence of strings")
        template = tuple(command_argv)
        if not template or not all(isinstance(item, str) for item in template):
            raise TypeError("command_argv must contain strings")
        if not template[0].strip() or "{text}" in template[0]:
            raise ValueError("the executable must be fixed and non-empty")
        if sum(item.count("{text}") for item in template) != 1:
            raise ValueError("command_argv must contain exactly one {text} placeholder")
        if isinstance(max_text_chars, bool) or not isinstance(max_text_chars, int):
            raise TypeError("max_text_chars must be an integer")
        if max_text_chars <= 0:
            raise ValueError("max_text_chars must be positive")
        self._template, self._maximum = template, max_text_chars
        self._timeout = _duration(timeout_seconds, "timeout_seconds", allow_zero=False)
        self._terminate_timeout = _duration(
            terminate_timeout_seconds, "terminate_timeout_seconds", allow_zero=False)
        self._popen = popen_factory if popen_factory is not None else cast(PopenFactory, subprocess.Popen)
        self._cancelled, self._lock, self._speak_lock = Event(), RLock(), Lock()
        self._process: ProcessHandle | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        """返回关闭状态。\n\nArgs: 无。\nReturns: 已关闭为 True。\nRaises: 无。"""
        with self._lock:
            return self._closed

    def _argv(self, text: str) -> tuple[str, ...]:
        """替换文本占位符且不进行 shell 解析。\n\nArgs: text: 已校验文本。\nReturns: 独立 argv 元组。\nRaises: 无。"""
        return tuple(item.replace("{text}", text) for item in self._template)

    def _stop_process(self, process: ProcessHandle) -> None:
        """先温和终止，超时后强制杀死。\n\nArgs: process: 待停止进程。\nReturns: 无。\nRaises: 无，清理错误仅记录。"""
        try:
            if process.poll() is not None:
                return
        except Exception:
            pass
        try:
            process.terminate()
            try:
                process.wait(timeout=self._terminate_timeout)
                return
            except subprocess.TimeoutExpired:
                pass
        except Exception:
            pass
        try:
            process.kill()
            process.wait(timeout=self._terminate_timeout)
        except Exception:
            LOGGER.warning("TTS process cleanup failed")

    def speak(self, text: str) -> bool:
        """同步启动并等待一次系统播报。\n\nArgs: text: 待播报文本。\nReturns: 成功退出为 True；空白、缺程序、超时、非零退出或异常为 False。\nRaises: TypeError: 文本类型错误。ValueError: 文本过长或含 NUL。RuntimeError: 已关闭。"""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = text.strip()
        if not normalized:
            return False
        if len(normalized) > self._maximum or "\0" in normalized:
            raise ValueError("speech text is too long or contains NUL")
        with self._speak_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("system speech output is closed")
                self._cancelled.clear()
            try:
                process = self._popen(self._argv(normalized), shell=False, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                LOGGER.error("TTS executable is not installed")
                return False
            except Exception:
                LOGGER.error("TTS process could not be started")
                return False
            with self._lock:
                aborted = self._closed or self._cancelled.is_set()
                if not aborted:
                    self._process = process
            if aborted:
                self._stop_process(process)
                return False
            try:
                return_code = process.wait(timeout=self._timeout)
                if self._cancelled.is_set():
                    return False
                if return_code != 0:
                    LOGGER.warning("TTS process exited with code %d", return_code)
                    return False
                return True
            except subprocess.TimeoutExpired:
                LOGGER.warning("TTS process timed out")
                self._stop_process(process)
                return False
            except Exception:
                LOGGER.warning("TTS process wait failed")
                self._stop_process(process)
                return False
            finally:
                with self._lock:
                    if self._process is process:
                        self._process = None

    def cancel(self) -> None:
        """取消当前播报并唤醒等待。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        self._cancelled.set()
        with self._lock:
            process = self._process
        if process is not None:
            self._stop_process(process)

    def close(self) -> None:
        """幂等关闭并取消当前播报。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.cancel()
