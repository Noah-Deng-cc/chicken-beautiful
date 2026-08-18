"""对话协调服务：输入为语音和系统快照，输出为播报及对话轮次；依赖音频、融合、智能体和状态契约。"""

from __future__ import annotations

import logging
from math import isfinite
from threading import Event, RLock

from src.agent import AgentClient
from src.audio import SpeechInput, SpeechOutput
from src.core.state import StateStore
from src.domain import DialogueTurn
from src.fusion import FusionService


LOGGER = logging.getLogger(__name__)


class DialogueService:
    """串行协调一次语音对话，且不会将完整对话写入日志。"""

    def __init__(
        self,
        speech_input: SpeechInput,
        speech_output: SpeechOutput,
        agent: AgentClient,
        fusion: FusionService,
        state: StateStore,
        listen_timeout_seconds: float,
        fallback_text: str = "当前网络服务暂不可用，请稍后再试。",
    ) -> None:
        """保存已装配组件及对话会话状态。

        Args: speech_input: 麦克风与语音识别接口。speech_output: 本地播报接口。agent: 外部智能体。
            fusion: 快照上下文构建器。state: 线程安全状态存储。listen_timeout_seconds: 单次监听时限。
            fallback_text: 智能体失败时的短本地提示。
        Returns: 无。
        Raises: TypeError: 监听时限或降级提示类型错误。ValueError: 时限或提示为空。
        """
        if isinstance(listen_timeout_seconds, bool) or not isinstance(listen_timeout_seconds, (int, float)):
            raise TypeError("listen_timeout_seconds must be a number")
        if not isfinite(float(listen_timeout_seconds)) or listen_timeout_seconds <= 0:
            raise ValueError("listen_timeout_seconds must be positive")
        if not isinstance(fallback_text, str):
            raise TypeError("fallback_text must be a string")
        if not fallback_text.strip():
            raise ValueError("fallback_text must be non-empty")
        self._input = speech_input
        self._output = speech_output
        self._agent = agent
        self._fusion = fusion
        self._state = state
        self._timeout = float(listen_timeout_seconds)
        self._fallback_text = fallback_text.strip()
        self._conversation_id: str | None = None
        self._lock = RLock()

    @property
    def conversation_id(self) -> str | None:
        """读取最近一次成功回复提供的会话标识。

        Args: 无。
        Returns: 当前会话标识；首次成功调用前为 None。
        Raises: 无。
        """
        with self._lock:
            return self._conversation_id

    def run_once(self) -> DialogueTurn | None:
        """执行一次监听、融合、智能体调用和播报。

        Args: 无。
        Returns: 成功智能体回复对应的对话轮次；静音或智能体失败时为 None。
        Raises: Exception: 本地输入、融合或状态组件的不可恢复错误。
        """
        with self._lock:
            transcript = self._input.listen(self._timeout)
            if transcript is None or not transcript.strip():
                return None
            text = transcript.strip()
            try:
                context = self._fusion.build_context(self._state.get_snapshot())
                reply = self._agent.reply(text, context, self._conversation_id)
            except Exception:
                LOGGER.warning("dialogue agent request failed; using local fallback")
                self._speak_fallback()
                return None
            if reply.conversation_id is not None:
                self._conversation_id = reply.conversation_id
            try:
                spoken = self._output.speak(reply.text)
                if not spoken:
                    LOGGER.warning("dialogue reply playback was unavailable")
            except Exception:
                LOGGER.warning("dialogue reply playback failed")
            turn = DialogueTurn(timestamp=reply.timestamp, user_text=text, reply=reply)
            self._state.update(turn)
            return turn

    def run(self, stop_event: Event) -> None:
        """循环执行对话，直到停止事件被设置。

        Args: stop_event: 由编排器设置的线程停止事件。
        Returns: 无。
        Raises: 无。
        """
        while not stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                LOGGER.exception("dialogue loop iteration failed")

    def _speak_fallback(self) -> None:
        """尽力播报本地降级提示，且不让播报故障终止循环。

        Args: 无。
        Returns: 无。
        Raises: 无。
        """
        try:
            self._output.speak(self._fallback_text)
        except Exception:
            LOGGER.warning("dialogue fallback playback failed")
