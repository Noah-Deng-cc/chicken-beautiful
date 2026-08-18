"""树莓派摄像头源：输入设备配置，输出原始帧对象；依赖均通过标准库延迟导入。"""

from __future__ import annotations

import importlib
import logging
from threading import RLock


class CameraSource:
    """兼容 USB OpenCV 与 CSI Picamera2/libcamera 的懒加载帧源。"""

    def __init__(
        self,
        source: int | str = 0,
        *,
        backend: str = "opencv",
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        logger: logging.Logger | None = None,
    ) -> None:
        """保存摄像头配置但不导入库或打开设备。\n\n        Args: source: OpenCV 设备序号或路径。backend: opencv、libcamera 或 picamera2。width: 帧宽。height: 帧高。fps: 目标帧率。logger: 可选日志器。\n        Returns: 无。\n        Raises: TypeError: 配置类型错误。ValueError: 后端未知或数值越界。"""
        if isinstance(source, bool) or not isinstance(source, (int, str)):
            raise TypeError("source must be an integer or string")
        if not all(isinstance(value, int) and not isinstance(value, bool)
                   for value in (width, height, fps)):
            raise TypeError("width, height, and fps must be integers")
        normalized = backend.lower().strip() if isinstance(backend, str) else ""
        if normalized not in {"opencv", "libcamera", "picamera2"}:
            raise ValueError("backend must be opencv, libcamera, or picamera2")
        if not 1 <= width <= 4096 or not 1 <= height <= 4096 or not 1 <= fps <= 120:
            raise ValueError("camera dimensions or fps are out of range")
        self._source, self._backend = source, normalized
        self._width, self._height, self._fps = width, height, fps
        self._logger = logger or logging.getLogger(__name__)
        self._handle: object | None = None
        self._started = False
        self._closed = False
        self._lock = RLock()

    def start(self) -> None:
        """启用懒打开状态，重复调用安全。\n\n        Args: 无。\n        Returns: 无；设备直到 read 时才打开。\n        Raises: 无。"""
        with self._lock:
            if not self._closed:
                self._started = True

    def read(self) -> object | None:
        """读取一帧且不额外复制。\n\n        Args: 无。\n        Returns: 后端原生帧对象；未启动、关闭或故障时为 None。\n        Raises: 无；导入、打开和采集错误均记录后转换为 None。"""
        with self._lock:
            if not self._started or self._closed:
                return None
            try:
                if self._handle is None:
                    self._open()
                if self._backend == "opencv":
                    ok, frame = self._handle.read()  # type: ignore[union-attr]
                    if not ok or frame is None:
                        raise RuntimeError("摄像头未返回有效帧")
                    return frame
                frame = self._handle.capture_array("main")  # type: ignore[union-attr]
                if frame is None:
                    raise RuntimeError("CSI 摄像头未返回有效帧")
                return frame
            except Exception as exc:
                self._logger.error("摄像头读取失败: %s", exc)
                self._release_handle()
                return None

    def close(self) -> None:
        """幂等释放摄像头并进入终止状态。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无；后端清理错误仅写日志。"""
        with self._lock:
            if self._closed:
                return
            self._started = False
            self._closed = True
            self._release_handle()

    def _open(self) -> None:
        """延迟导入选定后端并打开摄像头。\n\n        Args: 无。\n        Returns: 无。\n        Raises: ImportError: 后端库不可用。RuntimeError: 设备无法打开。"""
        if self._backend == "opencv":
            cv2 = importlib.import_module("cv2")
            handle = cv2.VideoCapture(self._source)
            if not handle.isOpened():
                handle.release()
                raise RuntimeError(f"无法打开 OpenCV 摄像头 {self._source!r}")
            # set 通常原地生效，避免为了调整尺寸再复制帧。
            handle.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            handle.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            handle.set(cv2.CAP_PROP_FPS, self._fps)
            self._handle = handle
            return
        module = importlib.import_module("picamera2")
        handle = module.Picamera2()
        config = handle.create_video_configuration(
            main={"size": (self._width, self._height), "format": "BGR888"},
            controls={"FrameRate": self._fps},
            buffer_count=2,
        )
        handle.configure(config)
        handle.start()
        self._handle = handle

    def _release_handle(self) -> None:
        """释放当前后端句柄并允许下次读取重连。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无。"""
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if self._backend == "opencv":
                handle.release()  # type: ignore[attr-defined]
            else:
                handle.stop()  # type: ignore[attr-defined]
                handle.close()  # type: ignore[attr-defined]
        except Exception as exc:
            self._logger.warning("摄像头清理失败: %s", exc)
