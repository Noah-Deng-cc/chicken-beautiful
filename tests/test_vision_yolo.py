"""T07 Raspberry Pi YOLO and camera acceptance tests.

Inputs: synthetic model tensors and fully injected camera/inference doubles.
Outputs: assertions for parsing, lazy loading, CPU limits, recovery, and cleanup.
Dependencies: pytest and the Python standard library only; no real hardware/model.
"""

from __future__ import annotations

import importlib.abc
import json
import logging
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from src.domain.models import Emotion, EmotionReading
from src.vision.camera import CameraSource
import src.vision.camera as camera_module
import src.vision.yolo as yolo_module
from src.vision.yolo import OpenCvOnnxBackend, YoloEmotionPipeline, parse_emotion_output


LABELS = (
    Emotion.ANGRY,
    Emotion.DISGUSTED,
    Emotion.FEARFUL,
    Emotion.HAPPY,
    Emotion.NEUTRAL,
    Emotion.SAD,
    Emotion.SURPRISED,
)


class FakeSource:
    """Injectable frame source with call counters and optional failures."""

    def __init__(self, frames: list[object | BaseException] | None = None) -> None:
        self.frames = list(frames or [])
        self.start_calls = 0
        self.read_calls = 0
        self.close_calls = 0
        self.start_error: BaseException | None = None
        self.close_error: BaseException | None = None

    def start(self) -> None:
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error

    def read(self) -> object | None:
        self.read_calls += 1
        value = self.frames.pop(0) if self.frames else None
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeBackend:
    """Injectable inference backend that records model lifecycle."""

    def __init__(self, outputs: list[object | BaseException] | None = None) -> None:
        self.outputs = list(outputs or [])
        self.load_calls: list[tuple[Path, int]] = []
        self.infer_frames: list[object] = []
        self.close_calls = 0
        self.load_error: BaseException | None = None
        self.close_error: BaseException | None = None

    def load(self, model_path: Path, input_size: int) -> None:
        self.load_calls.append((model_path, input_size))
        if self.load_error is not None:
            raise self.load_error

    def infer(self, frame: object) -> object:
        self.infer_frames.append(frame)
        value = self.outputs.pop(0) if self.outputs else []
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeFaceDetector:
    def __init__(self, result: object | None) -> None:
        self.result = result
        self.frames: list[object] = []
        self.close_calls = 0

    def detect(self, frame: object) -> object | None:
        self.frames.append(frame)
        return self.result

    def close(self) -> None:
        self.close_calls += 1


def classification(index: int, score: float = 0.9) -> list[float]:
    """Build one seven-class classification row."""
    values = [0.01] * 7
    values[index] = score
    return values


@pytest.mark.parametrize("index", range(7))
def test_seven_column_classification_maps_every_emotion(index: int) -> None:
    """The fixed training label order maps all seven output positions exactly."""
    assert parse_emotion_output(classification(index), 0.5) == (LABELS[index], 0.9)


def test_seven_column_logits_are_softmax_normalized() -> None:
    """Raw seven-class ONNX logits become a valid confidence and label."""
    assert parse_emotion_output([-1.0, -2.0, 0.1, -0.5, 0.4, -0.2, 1.2], 0.4) == (
        Emotion.SURPRISED, pytest.approx(0.4232, rel=1e-3)
    )


def test_pi_capture_fixture_maps_to_expected_emotion() -> None:
    """A captured Pi logits fixture exercises the deployed model data contract."""
    fixture = Path(__file__).parent / "fixtures" / "emotion_logits.json"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    parsed = parse_emotion_output(data["logits"], float(data["minimum_confidence"]))
    assert parsed is not None
    assert parsed[0].value == data["expected_label"]
    assert parsed[1] >= float(data["minimum_confidence"])


def test_eleven_and_twelve_column_detection_outputs() -> None:
    """YOLO rows with and without objectness produce the correct confidence."""
    row11 = [1.0, 2.0, 3.0, 4.0, *classification(5, 0.8)]
    row12 = [1.0, 2.0, 3.0, 4.0, 0.5, *classification(3, 0.8)]
    assert parse_emotion_output([row11], 0.5) == (Emotion.SAD, 0.8)
    assert parse_emotion_output([row12], 0.4) == (Emotion.HAPPY, 0.4)


def test_six_column_nms_output_and_threshold_equality() -> None:
    """Post-NMS class ids are mapped and a score equal to threshold is accepted."""
    raw = [[10.0, 20.0, 30.0, 40.0, 0.5, 6.0]]
    assert parse_emotion_output(raw, 0.5) == (Emotion.SURPRISED, 0.5)
    assert parse_emotion_output(raw, 0.5000001) is None


def test_batched_nxc_and_cxn_outputs_select_highest_detection() -> None:
    """Both [1,N,C] and [1,C,N] layouts are normalized before selection."""
    first = [0.0, 0.0, 0.0, 0.0, *classification(0, 0.6)]
    second = [0.0, 0.0, 0.0, 0.0, *classification(4, 0.95)]
    nxc = [[first, second]]
    cxn = [[[row[column] for row in (first, second)] for column in range(11)]]
    assert parse_emotion_output(nxc, 0.5) == (Emotion.NEUTRAL, 0.95)
    assert parse_emotion_output(cxn, 0.5) == (Emotion.NEUTRAL, 0.95)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        [[]],
        "not-a-tensor",
        object(),
        [0.1] * 5,
        [0.1] * 8,
        [[0.1] * 11, [0.1] * 10],
        [0.1, 0.1, float("nan"), 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, float("inf"), 0.1, 0.1, 0.1, 0.1],
        [False, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [[0.0, 0.0, 0.0, 0.0, 0.7, True]],
        [[0.0, 0.0, 0.0, 0.0, True, 1.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.7, -1.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.7, 7.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.7, 1.5]],
        [[0.0, 0.0, 0.0, 0.0, float("nan"), 1.0]],
        [[0.0, 0.0, 0.0, 0.0, float("inf"), 1.0]],
        [[0.0, 0.0, 0.0, 0.0, 1.01, 1.0]],
    ],
)
def test_empty_bad_shapes_and_invalid_numbers_return_none(raw: object) -> None:
    """Empty, malformed, non-finite, boolean, and out-of-range outputs are rejected."""
    assert parse_emotion_output(raw, 0.5) is None


def test_tolist_array_like_is_supported_without_numpy() -> None:
    """An array-like result can expose tolist without importing NumPy."""
    raw = SimpleNamespace(tolist=lambda: [[classification(2, 0.75)]])
    assert parse_emotion_output(raw, 0.75) == (Emotion.FEARFUL, 0.75)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"input_size": 321}, ValueError),
        ({"input_size": True}, TypeError),
        ({"device": "cuda"}, ValueError),
        ({"backend": "ultralytics"}, ValueError),
        ({"confidence_threshold": float("nan")}, ValueError),
        ({"confidence_threshold": True}, TypeError),
        ({"sample_interval_seconds": -0.1}, ValueError),
        ({"sample_interval_seconds": float("inf")}, ValueError),
    ],
)
def test_pipeline_rejects_non_pi_or_invalid_configuration(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    """The constructor enforces CPU, bounded input, finite thresholds, and backend names."""
    with pytest.raises(error):
        YoloEmotionPipeline(FakeSource(), Path("model.onnx"), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("input_size", [320, 32])
def test_default_maximum_and_minimum_sizes_reach_injected_backend(input_size: int) -> None:
    """The default/maximum 320 and lower accepted boundary are passed to model load."""
    frame = object()
    source = FakeSource([frame])
    backend = FakeBackend([classification(3)])
    kwargs: dict[str, int] = {} if input_size == 320 else {"input_size": input_size}
    pipeline = YoloEmotionPipeline(source, Path("model.onnx"), backend=backend, **kwargs)
    assert backend.load_calls == []
    pipeline.start()
    result = pipeline.read()
    assert isinstance(result, EmotionReading)
    assert result.dominant is Emotion.HAPPY
    assert backend.load_calls == [(Path("model.onnx"), input_size)]
    assert backend.infer_frames == [frame]


def test_face_detector_crops_before_emotion_inference() -> None:
    """Configured face detection prevents full-frame classification and is closed."""
    full_frame, face = object(), object()
    detector = FakeFaceDetector(face)
    backend = FakeBackend([classification(3)])
    pipeline = YoloEmotionPipeline(FakeSource([full_frame]), Path("model.onnx"),
                                   backend=backend, face_detector=detector,
                                   sample_interval_seconds=0)
    pipeline.start()
    assert pipeline.read() is not None
    assert detector.frames == [full_frame]
    assert backend.infer_frames == [face]
    pipeline.close()
    assert detector.close_calls == 1


def test_model_is_not_loaded_before_first_valid_frame() -> None:
    """Construction, start, and an empty camera read leave the model unloaded."""
    source = FakeSource([None, object()])
    backend = FakeBackend([classification(4)])
    pipeline = YoloEmotionPipeline(
        source, Path("model.onnx"), backend=backend, sample_interval_seconds=0
    )
    assert backend.load_calls == []
    pipeline.start()
    assert backend.load_calls == []
    assert pipeline.read() is None
    assert backend.load_calls == []
    assert pipeline.read() is not None
    assert len(backend.load_calls) == 1


def test_sampling_throttle_skips_camera_and_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reads inside the interval return early without touching camera or backend."""
    times = iter([10.0, 10.5, 11.0])
    monkeypatch.setattr(yolo_module, "monotonic", lambda: next(times))
    source = FakeSource([object(), object()])
    backend = FakeBackend([classification(0), classification(1)])
    pipeline = YoloEmotionPipeline(
        source, Path("model.onnx"), backend=backend, sample_interval_seconds=1.0
    )
    pipeline.start()
    assert pipeline.read() is not None
    assert pipeline.read() is None
    assert pipeline.read() is not None
    assert source.read_calls == 2
    assert len(backend.infer_frames) == 2
    assert len(backend.load_calls) == 1


def test_source_start_and_read_failures_are_contained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Source startup/read exceptions become None and preserve diagnostic logging."""
    failed_start = FakeSource()
    failed_start.start_error = RuntimeError("start disconnected")
    first = YoloEmotionPipeline(failed_start, Path("model.onnx"), backend=FakeBackend())
    with caplog.at_level(logging.ERROR, logger="src.vision.yolo"):
        first.start()
    assert first.read() is None
    assert "start disconnected" in caplog.text

    failed_read = FakeSource([OSError("read disconnected")])
    second = YoloEmotionPipeline(
        failed_read, Path("model.onnx"), backend=FakeBackend(), sample_interval_seconds=0
    )
    second.start()
    with caplog.at_level(logging.ERROR, logger="src.vision.yolo"):
        assert second.read() is None
    assert "read disconnected" in caplog.text


@pytest.mark.parametrize("phase", ["load", "infer"])
def test_backend_failures_return_none_and_are_logged(
    phase: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Model load and inference errors never escape the pipeline read boundary."""
    backend = FakeBackend([RuntimeError("infer failed")])
    if phase == "load":
        backend.load_error = RuntimeError("load failed")
    pipeline = YoloEmotionPipeline(
        FakeSource([object()]), Path("model.onnx"), backend=backend, sample_interval_seconds=0
    )
    pipeline.start()
    with caplog.at_level(logging.ERROR, logger="src.vision.yolo"):
        assert pipeline.read() is None
    assert f"{phase} failed" in caplog.text


def test_pipeline_start_close_are_idempotent_and_release_both_resources(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Repeated lifecycle calls are safe and cleanup attempts every created resource once."""
    source = FakeSource([object()])
    backend = FakeBackend([classification(3)])
    pipeline = YoloEmotionPipeline(source, Path("model.onnx"), backend=backend)
    pipeline.start()
    pipeline.start()
    assert pipeline.read() is not None
    backend.close_error = RuntimeError("backend close failed")
    source.close_error = RuntimeError("source close failed")
    with caplog.at_level(logging.WARNING, logger="src.vision.yolo"):
        pipeline.close()
        pipeline.close()
    assert source.start_calls == 1
    assert backend.close_calls == 1
    assert source.close_calls == 1
    assert "backend close failed" in caplog.text
    assert "source close failed" in caplog.text
    assert pipeline.read() is None


def test_ncnn_without_adapter_is_explicit_and_injected_adapter_works(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """NCNN has a clear adapter requirement while protocol injection remains usable."""
    missing = YoloEmotionPipeline(
        FakeSource([object()]), Path("model.ncnn"), backend="ncnn", sample_interval_seconds=0
    )
    missing.start()
    with caplog.at_level(logging.ERROR, logger="src.vision.yolo"):
        assert missing.read() is None
    assert "NCNN adapter unavailable" in caplog.text

    adapter = FakeBackend([classification(6, 0.88)])
    injected = YoloEmotionPipeline(
        FakeSource([object()]), Path("model.ncnn"), backend=adapter
    )
    injected.start()
    result = injected.read()
    assert result is not None
    assert result.dominant is Emotion.SURPRISED
    assert result.confidence == pytest.approx(0.88)


class FakeNet:
    """Minimal OpenCV DNN network double."""

    def __init__(self, output: object) -> None:
        self.output = output
        self.backend: object | None = None
        self.target: object | None = None
        self.inputs: list[object] = []

    def setPreferableBackend(self, value: object) -> None:
        self.backend = value

    def setPreferableTarget(self, value: object) -> None:
        self.target = value

    def setInput(self, value: object) -> None:
        self.inputs.append(value)

    def forward(self) -> object:
        return self.output


def test_opencv_backend_lazy_import_cpu_thread_limit_and_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenCV imports only on load, uses one CPU thread, and builds a 320 RGB blob."""
    model = tmp_path / "emotion.onnx"
    model.touch()
    frame, blob = object(), object()
    net = FakeNet(classification(3))
    calls: dict[str, Any] = {"threads": [], "models": [], "blobs": []}
    fake_cv2 = SimpleNamespace(
        dnn=SimpleNamespace(
            DNN_BACKEND_OPENCV="opencv-cpu",
            DNN_TARGET_CPU="cpu",
            readNetFromONNX=lambda path: calls["models"].append(path) or net,
            blobFromImage=lambda *args, **kwargs: calls["blobs"].append((args, kwargs)) or blob,
        ),
        setNumThreads=lambda count: calls["threads"].append(count),
    )
    imports: list[str] = []
    monkeypatch.setattr(
        yolo_module.importlib,
        "import_module",
        lambda name: imports.append(name) or fake_cv2,
    )
    backend = OpenCvOnnxBackend()
    assert imports == []
    backend.load(model, 320)
    assert imports == ["cv2"]
    assert calls["threads"] == [1]
    assert calls["models"] == [str(model)]
    assert net.backend == "opencv-cpu"
    assert net.target == "cpu"
    assert backend.infer(frame) == classification(3)
    args, kwargs = calls["blobs"][0]
    assert args == (frame, 1.0 / 255.0, (320, 320))
    assert kwargs == {"swapRB": True, "crop": False}
    assert net.inputs == [blob]
    backend.close()
    with pytest.raises(RuntimeError, match="not loaded"):
        backend.infer(frame)


def test_missing_onnx_is_reported_before_cv2_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonexistent model fails clearly without importing or initializing OpenCV."""
    imports: list[str] = []
    monkeypatch.setattr(
        yolo_module.importlib,
        "import_module",
        lambda name: imports.append(name) or object(),
    )
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        OpenCvOnnxBackend().load(tmp_path / "missing.onnx", 320)
    assert imports == []


class FakeCapture:
    """OpenCV camera handle double with configurable reads."""

    def __init__(self, reads: list[tuple[bool, object | None]], opened: bool = True) -> None:
        self.reads = list(reads)
        self.opened = opened
        self.release_calls = 0
        self.settings: list[tuple[object, object]] = []

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop: object, value: object) -> bool:
        self.settings.append((prop, value))
        return True

    def read(self) -> tuple[bool, object | None]:
        return self.reads.pop(0)

    def release(self) -> None:
        self.release_calls += 1


def test_usb_camera_lazy_open_read_reconnect_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """USB opens on read, releases a failed handle, reconnects, and returns the native frame."""
    frame = object()
    failed = FakeCapture([(False, None)])
    recovered = FakeCapture([(True, frame)])
    handles = iter([failed, recovered])
    opens: list[object] = []
    fake_cv2 = SimpleNamespace(
        CAP_PROP_FRAME_WIDTH="width",
        CAP_PROP_FRAME_HEIGHT="height",
        CAP_PROP_FPS="fps",
        VideoCapture=lambda source: opens.append(source) or next(handles),
    )
    imports: list[str] = []
    monkeypatch.setattr(
        camera_module.importlib,
        "import_module",
        lambda name: imports.append(name) or fake_cv2,
    )
    camera = CameraSource("/dev/video0", width=320, height=240, fps=10)
    assert imports == []
    assert camera.read() is None
    assert imports == []
    camera.start()
    assert imports == []
    assert camera.read() is None
    assert failed.release_calls == 1
    assert camera.read() is frame
    assert opens == ["/dev/video0", "/dev/video0"]
    assert recovered.settings == [("width", 320), ("height", 240), ("fps", 10)]
    camera.close()
    camera.close()
    assert recovered.release_calls == 1
    assert camera.read() is None


class FakePicamera:
    """Picamera2 handle double with native-frame and cleanup tracking."""

    def __init__(self, frames: list[object | None]) -> None:
        self.frames = list(frames)
        self.configs: list[object] = []
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0
        self.capture_streams: list[str] = []

    def create_video_configuration(self, **kwargs: object) -> object:
        self.configs.append(kwargs)
        return {"configured": kwargs}

    def configure(self, config: object) -> None:
        self.configs.append(config)

    def start(self) -> None:
        self.start_calls += 1

    def capture_array(self, stream: str) -> object | None:
        self.capture_streams.append(stream)
        return self.frames.pop(0)

    def stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize("backend", ["picamera2", "libcamera"])
def test_csi_camera_lazy_open_read_reconnect_and_identity(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CSI aliases open Picamera2 lazily, recover after no frame, and avoid copying."""
    frame = object()
    failed, recovered = FakePicamera([None]), FakePicamera([frame])
    handles = iter([failed, recovered])
    imports: list[str] = []
    fake_module = SimpleNamespace(Picamera2=lambda: next(handles))
    monkeypatch.setattr(
        camera_module.importlib,
        "import_module",
        lambda name: imports.append(name) or fake_module,
    )
    camera = CameraSource(0, backend=backend, width=320, height=240, fps=12)
    camera.start()
    assert imports == []
    assert camera.read() is None
    assert (failed.stop_calls, failed.close_calls) == (1, 1)
    assert camera.read() is frame
    assert imports == ["picamera2", "picamera2"]
    assert recovered.capture_streams == ["main"]
    config_args = recovered.configs[0]
    assert isinstance(config_args, dict)
    assert config_args["main"] == {"size": (320, 240), "format": "BGR888"}
    assert config_args["controls"] == {"FrameRate": 12}
    assert config_args["buffer_count"] == 2
    camera.close()
    camera.close()
    assert (recovered.stop_calls, recovered.close_calls) == (1, 1)


def test_camera_open_failures_return_none_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable USB handles are released and a later read retries opening."""
    unavailable, frame = FakeCapture([], opened=False), object()
    available = FakeCapture([(True, frame)])
    handles = iter([unavailable, available])
    fake_cv2 = SimpleNamespace(
        CAP_PROP_FRAME_WIDTH=1,
        CAP_PROP_FRAME_HEIGHT=2,
        CAP_PROP_FPS=3,
        VideoCapture=lambda source: next(handles),
    )
    monkeypatch.setattr(camera_module.importlib, "import_module", lambda name: fake_cv2)
    camera = CameraSource(0)
    camera.start()
    assert camera.read() is None
    assert unavailable.release_calls == 1
    assert camera.read() is frame


@pytest.mark.parametrize(
    ("args", "kwargs", "error"),
    [
        ((True,), {}, TypeError),
        ((object(),), {}, TypeError),
        ((0,), {"backend": "unknown"}, ValueError),
        ((0,), {"width": 0}, ValueError),
        ((0,), {"height": 4097}, ValueError),
        ((0,), {"fps": 121}, ValueError),
        ((0,), {"width": True}, TypeError),
    ],
)
def test_camera_rejects_invalid_source_backend_and_dimensions(
    args: tuple[object, ...], kwargs: dict[str, object], error: type[Exception]
) -> None:
    """Bad device identifiers, backends, booleans, and bounds fail before hardware access."""
    with pytest.raises(error):
        CameraSource(*args, **kwargs)  # type: ignore[arg-type]


def test_modules_import_without_torch_ultralytics_cv2_or_picamera2() -> None:
    """Fresh imports do not request ML runtimes or camera libraries."""
    project_root = Path(__file__).resolve().parents[1]
    blocker = """
import importlib.abc
import sys

blocked = {"cv2", "picamera2", "torch", "ultralytics"}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise RuntimeError(f"forbidden dependency requested: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import src.vision.camera
import src.vision.yolo
print("t07-import-ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", blocker],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "t07-import-ok"


def test_source_files_do_not_import_torch_or_ultralytics() -> None:
    """Static source text contains no direct forbidden runtime import statements."""
    project_root = Path(__file__).resolve().parents[1]
    for relative in (Path("src/vision/yolo.py"), Path("src/vision/camera.py")):
        source = (project_root / relative).read_text(encoding="utf-8")
        assert "import torch" not in source
        assert "from torch" not in source
        assert "import ultralytics" not in source
        assert "from ultralytics" not in source
