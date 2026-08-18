"""轻量 YOLO 情绪推理：输入原始帧，输出领域读数；依赖标准库，OpenCV 仅运行时导入。"""
from __future__ import annotations
from collections.abc import Sequence
from datetime import datetime, timezone
import importlib
import logging
from math import isfinite
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Protocol
from src.domain.models import Emotion, EmotionReading
from .base import VisionPipeline
from .camera import CameraSource
EMOTION_LABELS: tuple[Emotion, ...] = (
    Emotion.ANGRY, Emotion.DISGUSTED, Emotion.FEARFUL, Emotion.HAPPY,
    Emotion.NEUTRAL, Emotion.SAD, Emotion.SURPRISED,
)
_VA = {Emotion.ANGRY: (-0.7, 0.8), Emotion.DISGUSTED: (-0.6, 0.4),
       Emotion.FEARFUL: (-0.8, 0.8), Emotion.HAPPY: (0.8, 0.6),
       Emotion.NEUTRAL: (0.0, 0.0), Emotion.SAD: (-0.7, -0.4),
       Emotion.SURPRISED: (0.2, 0.8)}
class InferenceBackend(Protocol):
    """可替换的轻量推理后端契约。"""
    def load(self, model_path: Path, input_size: int) -> None:
        """加载模型。\n\n        Args: model_path: 模型路径。input_size: 方形输入边长。\n        Returns: 无。\n        Raises: RuntimeError: 模型不可用。"""
        ...
    def infer(self, frame: object) -> object:
        """执行一次推理。\n\n        Args: frame: 后端原生帧。\n        Returns: 原始模型输出。\n        Raises: RuntimeError: 推理失败。"""
        ...
    def close(self) -> None:
        """释放后端资源。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无。"""
        ...
class OpenCvOnnxBackend:
    """使用 OpenCV DNN CPU 执行 ONNX 的延迟加载后端。"""
    def __init__(self) -> None:
        """创建未加载后端。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无。"""
        self._cv2: object | None = None
        self._net: object | None = None
        self._input_size = 320
    def load(self, model_path: Path, input_size: int) -> None:
        """延迟导入 cv2 并加载 ONNX。\n\n        Args: model_path: ONNX 文件。input_size: 输入边长。\n        Returns: 无。\n        Raises: FileNotFoundError: 模型不存在。RuntimeError: OpenCV 或模型加载失败。"""
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        cv2 = importlib.import_module("cv2")
        cv2.setNumThreads(1)
        net = cv2.dnn.readNetFromONNX(str(model_path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._cv2, self._net, self._input_size = cv2, net, input_size
    def infer(self, frame: object) -> object:
        """预处理帧并返回未解析输出。\n\n        Args: frame: OpenCV/Picamera2 帧。\n        Returns: ONNX 原始输出。\n        Raises: RuntimeError: 尚未加载或推理失败。"""
        if self._cv2 is None or self._net is None:
            raise RuntimeError("ONNX backend is not loaded")
        blob = self._cv2.dnn.blobFromImage(  # type: ignore[union-attr]
            frame, 1.0 / 255.0, (self._input_size, self._input_size), swapRB=True, crop=False)
        self._net.setInput(blob)  # type: ignore[union-attr]
        return self._net.forward()  # type: ignore[union-attr,no-any-return]
    def close(self) -> None:
        """释放网络引用。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无。"""
        self._net, self._cv2 = None, None
def parse_emotion_output(raw: object, threshold: float) -> tuple[Emotion, float] | None:
    """解析分类或未做 NMS 的 YOLO 检测输出。\n\n    Args: raw: 支持 tolist 的数组或嵌套序列。threshold: 最低置信度。\n    Returns: 最高置信度的情绪与分数；坏输出或无检测时为 None。\n    Raises: 无。"""
    try:
        value = raw.tolist() if hasattr(raw, "tolist") else raw
        while isinstance(value, Sequence) and len(value) == 1:
            value = value[0]
            value = value.tolist() if hasattr(value, "tolist") else value
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            return None
        if all(isinstance(item, (int, float)) for item in value):
            rows = [value]
        elif all(isinstance(item, Sequence) for item in value):
            rows = list(value)
            if len(rows) in (11, 12) and len(rows[0]) not in (6, 7, 11, 12):
                rows = [list(column) for column in zip(*rows)]
        else:
            return None
        best: tuple[Emotion, float] | None = None
        for row in rows:
            candidate = _parse_row(row)
            if candidate and candidate[1] >= threshold and (best is None or candidate[1] > best[1]):
                best = candidate
        return best
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return None
def _parse_row(row: Sequence[object]) -> tuple[Emotion, float] | None:
    """解析单个分类或检测行。\n\n    Args: row: 7、6、11 或 12 列数值。\n    Returns: 情绪与置信度，格式或数值非法时为 None。\n    Raises: 无。"""
    if len(row) == 6:
        score, class_id = row[4], row[5]
        if (isinstance(score, bool) or isinstance(class_id, bool) or not isinstance(score, (int, float))
                or not isinstance(class_id, (int, float)) or int(class_id) != class_id):
            return None
        index, confidence = int(class_id), float(score)  # type: ignore[arg-type]
    else:
        offset = 0 if len(row) == 7 else 4 if len(row) == 11 else 5 if len(row) == 12 else -1
        if (offset < 0 or not all(isinstance(item, (int, float)) and not isinstance(item, bool)
                                  for item in row[offset:offset + 7])
                or len(row) == 12 and (isinstance(row[4], bool) or not isinstance(row[4], (int, float)))):
            return None
        scores = [float(item) for item in row[offset:offset + 7]]
        index = max(range(7), key=scores.__getitem__)
        confidence = scores[index] * (float(row[4]) if len(row) == 12 else 1.0)
    if not 0 <= index < 7 or not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return EMOTION_LABELS[index], confidence


class YoloEmotionPipeline(VisionPipeline):
    """面向低内存树莓派的单帧、CPU 情绪推理管道。"""

    def __init__(self, source: CameraSource, model_path: Path, *,
                 backend: str | InferenceBackend = "onnx", confidence_threshold: float = 0.5,
                 input_size: int = 320, device: str = "cpu", sample_interval_seconds: float = 0.2,
                 logger: logging.Logger | None = None) -> None:
        """保存推理配置且不加载模型。\n\n        Args: source: 可注入帧源。model_path: 导出模型路径。backend: onnx、ncnn 或后端对象。confidence_threshold: 最低置信度。input_size: 输入边长，最大 320。device: 设备端固定为 cpu。sample_interval_seconds: 最短采样间隔。logger: 可选日志器。\n        Returns: 无。\n        Raises: TypeError: 参数类型错误。ValueError: 数值、设备或后端名称非法。"""
        if not isinstance(model_path, Path):
            raise TypeError("model_path must be a Path")
        if isinstance(input_size, bool) or not isinstance(input_size, int) or not isinstance(device, str):
            raise TypeError("input_size must be an integer and device must be a string")
        if not isinstance(confidence_threshold, (int, float)) or isinstance(confidence_threshold, bool):
            raise TypeError("confidence_threshold must be numeric")
        if not isinstance(sample_interval_seconds, (int, float)) or isinstance(sample_interval_seconds, bool):
            raise TypeError("sample_interval_seconds must be numeric")
        if (not 32 <= input_size <= 320 or not isfinite(float(confidence_threshold))
                or not 0.0 <= confidence_threshold <= 1.0 or not isfinite(float(sample_interval_seconds))
                or sample_interval_seconds < 0):
            raise ValueError("invalid input size, threshold, or sample interval")
        if (isinstance(backend, str) and backend.lower() not in {"onnx", "ncnn"}) or device.lower().strip() != "cpu":
            raise ValueError("backend must be onnx/ncnn and Raspberry Pi device must be cpu")
        if not isinstance(backend, str) and not all(callable(getattr(backend, name, None)) for name in ("load", "infer", "close")):
            raise TypeError("injected backend must implement load, infer, and close")
        self._source, self._model_path, self._backend_spec = source, model_path, backend
        self._threshold, self._input_size = float(confidence_threshold), input_size
        self._interval, self._logger = float(sample_interval_seconds), logger or logging.getLogger(__name__)
        self._backend: InferenceBackend | None = None
        self._loaded = self._started = self._closed = False
        self._last_sample = float("-inf")
        self._lock = RLock()

    def start(self) -> None:
        """启动帧源但保持模型未加载。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无；帧源启动错误被记录。"""
        with self._lock:
            if self._closed or self._started:
                return
            try:
                self._source.start()
                self._started = True
            except Exception as exc:
                self._logger.error("视觉帧源启动失败: %s", exc)

    def read(self) -> EmotionReading | None:
        """按采样间隔执行一次情绪推理。\n\n        Args: 无。\n        Returns: 最高置信度 EmotionReading；节流、无帧、无检测或故障时为 None。\n        Raises: 无。"""
        with self._lock:
            now = monotonic()
            if not self._started or self._closed or now - self._last_sample < self._interval:
                return None
            self._last_sample = now
            try:
                frame = self._source.read()
                if frame is None:
                    return None
                backend = self._ensure_backend()
                detected = parse_emotion_output(backend.infer(frame), self._threshold)
                if detected is None:
                    return None
                emotion, confidence = detected
                valence, arousal = _VA[emotion]
                return EmotionReading(datetime.now(timezone.utc), emotion, confidence, valence, arousal)
            except Exception as exc:
                self._logger.error("YOLO 情绪推理失败: %s", exc)
                return None

    def close(self) -> None:
        """幂等关闭帧源与推理后端。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无；清理错误被记录。"""
        with self._lock:
            if self._closed:
                return
            self._closed, self._started = True, False
            for resource in (self._backend, self._source):
                try:
                    if resource is not None:
                        resource.close()
                except Exception as exc:
                    self._logger.warning("视觉资源清理失败: %s", exc)

    def _ensure_backend(self) -> InferenceBackend:
        """首次读取时创建并加载后端。\n\n        Args: 无。\n        Returns: 已加载的后端。\n        Raises: RuntimeError: NCNN 未注入适配器或后端契约不完整。"""
        if self._backend is None:
            if isinstance(self._backend_spec, str):
                if self._backend_spec.lower() == "ncnn":
                    raise RuntimeError("NCNN adapter unavailable; inject an InferenceBackend implementation")
                self._backend = OpenCvOnnxBackend()
            else:
                self._backend = self._backend_spec
        if not self._loaded:
            self._backend.load(self._model_path, self._input_size)
            self._loaded = True
        return self._backend
