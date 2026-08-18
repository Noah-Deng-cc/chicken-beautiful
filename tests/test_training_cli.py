"""T05 training and export CLI acceptance tests using injected runtime doubles."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import training.export as export_module
import training.train as train_module


LABELS = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)


class FakeYOLO:
    """Record constructor, training, and export calls without model operations."""

    instances: list["FakeYOLO"] = []
    train_error: BaseException | None = None
    export_error: BaseException | None = None

    def __init__(self, model: str) -> None:
        self.model = model
        self.train_calls: list[dict[str, Any]] = []
        self.export_calls: list[dict[str, Any]] = []
        type(self).instances.append(self)

    def train(self, **kwargs: Any) -> object:
        """Record a fake training operation."""
        self.train_calls.append(kwargs)
        if type(self).train_error is not None:
            raise type(self).train_error
        return object()

    def export(self, **kwargs: Any) -> Path:
        """Record a fake export operation."""
        self.export_calls.append(kwargs)
        if type(self).export_error is not None:
            raise type(self).export_error
        return Path("fake-output")


@pytest.fixture(autouse=True)
def reset_fake_yolo() -> None:
    """Reset fake global state between tests."""
    FakeYOLO.instances.clear()
    FakeYOLO.train_error = None
    FakeYOLO.export_error = None


def write_dataset(
    path: Path,
    *,
    names: object = LABELS,
    nc: object = 7,
    root: object = "dataset",
    train: object = "images/train",
    val: object = "images/val",
    download: object | None = None,
) -> Path:
    """Write a small YAML configuration containing no image data."""
    payload: dict[str, object] = {
        "path": root,
        "train": train,
        "val": val,
        "nc": nc,
        "names": names,
    }
    if download is not None:
        payload["download"] = download
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def fake_runtime_imports(
    monkeypatch: pytest.MonkeyPatch, module: object
) -> list[str]:
    """Inject fake Ultralytics while retaining real PyYAML imports."""
    real_import = importlib.import_module
    imports: list[str] = []

    def fake_import(name: str, package: str | None = None) -> object:
        imports.append(name)
        if name == "ultralytics":
            return SimpleNamespace(YOLO=FakeYOLO)
        return real_import(name, package)

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    return imports


def test_repository_dataset_has_exact_fixed_order_and_local_paths() -> None:
    """The shipped dataset configuration preserves the contractual seven labels."""
    source = Path(train_module.__file__).with_name("emotions.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert raw["nc"] == 7
    assert tuple(raw["names"].values()) == LABELS
    assert list(raw["names"].keys()) == list(range(7))
    assert "download" not in raw
    assert all(
        isinstance(raw[field], str) and raw[field] and "://" not in raw[field]
        for field in ("path", "train", "val")
    )
    assert train_module.validate_dataset(source) == source.resolve()


@pytest.mark.parametrize("mapping_keys", [list(range(7)), [str(i) for i in range(7)]])
def test_dataset_accepts_integer_or_string_index_mapping(
    tmp_path: Path, mapping_keys: list[int] | list[str]
) -> None:
    """Both common YAML index representations normalize to the same fixed order."""
    names = dict(zip(mapping_keys, LABELS, strict=True))
    source = write_dataset(tmp_path / "dataset.yaml", names=names)
    assert train_module.validate_dataset(source) == source.resolve()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"root": "https://example.test/data"}, "field 'path'"),
        ({"train": "s3://bucket/train"}, "field 'train'"),
        ({"val": "http://example.test/val"}, "field 'val'"),
        ({"root": ""}, "field 'path'"),
        ({"train": "   "}, "field 'train'"),
        ({"val": None}, "field 'val'"),
        ({"download": "https://example.test/archive.zip"}, "automatic downloads"),
    ],
)
def test_dataset_rejects_url_download_empty_and_non_string_paths(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    """Training inputs cannot trigger remote access or omit local path fields."""
    source = write_dataset(tmp_path / "bad.yaml", **changes)
    with pytest.raises(ValueError, match=message):
        train_module.validate_dataset(source)


@pytest.mark.parametrize(
    ("names", "nc"),
    [
        (LABELS[::-1], 7),
        (LABELS[:-1], 7),
        ((*LABELS, "extra"), 7),
        ({index + 1: label for index, label in enumerate(LABELS)}, 7),
        ({index: label for index, label in enumerate(LABELS[:-1])}, 7),
        ("angry,disgusted,fearful,happy,neutral,sad,surprised", 7),
        (LABELS, 6),
        (LABELS, "7"),
    ],
)
def test_dataset_rejects_wrong_labels_indices_and_class_count(
    tmp_path: Path, names: object, nc: object
) -> None:
    """Wrong order, missing/extra labels, index gaps, and noninteger nc fail."""
    source = write_dataset(tmp_path / "bad-labels.yaml", names=names, nc=nc)
    with pytest.raises(ValueError, match="dataset (labels|names)"):
        train_module.validate_dataset(source)


@pytest.mark.parametrize("content", ["", "- list-root\n", "names: [broken", "null\n"])
def test_dataset_rejects_missing_malformed_and_non_mapping_yaml(
    tmp_path: Path, content: str
) -> None:
    """Empty, malformed, or structurally invalid YAML is rejected locally."""
    source = tmp_path / "invalid.yaml"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        train_module.validate_dataset(source)
    with pytest.raises(FileNotFoundError, match="dataset YAML not found"):
        train_module.validate_dataset(tmp_path / "missing.yaml")


def test_train_all_cli_parameters_are_forwarded_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every public training CLI option reaches the fake Ultralytics API."""
    dataset = write_dataset(tmp_path / "dataset.yaml")
    model = tmp_path / "base.pt"
    model.touch()
    project = tmp_path / "custom-runs"
    imports = fake_runtime_imports(monkeypatch, train_module)
    result = train_module.main(
        [
            "--dataset", str(dataset), "--model", str(model), "--epochs", "3",
            "--imgsz", "256", "--batch", "2", "--device", "cpu",
            "--workers", "0", "--project", str(project), "--name", "trial",
            "--seed", "9", "--resume", "--exist-ok",
        ]
    )
    assert result == 0
    assert imports == ["yaml", "ultralytics"]
    assert len(FakeYOLO.instances) == 1
    instance = FakeYOLO.instances[0]
    assert instance.model == str(model.resolve())
    assert instance.train_calls == [
        {
            "data": str(dataset.resolve()),
            "epochs": 3,
            "imgsz": 256,
            "batch": 2,
            "device": "cpu",
            "workers": 0,
            "project": str(project.resolve()),
            "name": "trial",
            "seed": 9,
            "resume": True,
            "exist_ok": True,
        }
    ]


def test_train_defaults_include_zero2w_friendly_320_and_no_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training defaults remain stable, especially the 320 pixel image size."""
    model = tmp_path / "base.pt"
    model.touch()
    fake_runtime_imports(monkeypatch, train_module)
    assert train_module.main(["--model", str(model)]) == 0
    options = FakeYOLO.instances[0].train_calls[0]
    assert options == {
        "data": str(Path(train_module.__file__).with_name("emotions.yaml").resolve()),
        "epochs": 100,
        "imgsz": 320,
        "batch": 16,
        "device": "0",
        "workers": 4,
        "project": str(Path("runs/train").resolve()),
        "name": "emotion-yolo",
        "seed": 42,
        "resume": False,
        "exist_ok": False,
    }


def test_train_accepts_numeric_lower_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Minimum accepted numeric values are passed without coercion or replacement."""
    model = tmp_path / "base.pt"
    model.touch()
    fake_runtime_imports(monkeypatch, train_module)
    assert train_module.main(
        ["--model", str(model), "--epochs", "1", "--imgsz", "32", "--batch", "1",
         "--workers", "0", "--seed", "0"]
    ) == 0
    options = FakeYOLO.instances[0].train_calls[0]
    assert (options["epochs"], options["imgsz"], options["batch"]) == (1, 32, 1)
    assert (options["workers"], options["seed"]) == (0, 0)


@pytest.mark.parametrize(
    "option",
    [
        ("--epochs", "0"), ("--epochs", "-1"), ("--imgsz", "31"),
        ("--batch", "0"), ("--workers", "-1"), ("--seed", "-1"),
        ("--device", " "), ("--name", " "),
    ],
)
def test_train_invalid_boundaries_return_input_error_before_runtime_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: tuple[str, str]
) -> None:
    """Invalid numeric/text boundaries return 2 without constructing a model."""
    model = tmp_path / "base.pt"
    model.touch()
    imports = fake_runtime_imports(monkeypatch, train_module)
    assert train_module.main(["--model", str(model), *option]) == 2
    assert "ultralytics" not in imports
    assert FakeYOLO.instances == []


def test_train_model_and_resume_checkpoint_must_exist_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both initial weights and an explicit resume checkpoint are local files."""
    imports = fake_runtime_imports(monkeypatch, train_module)
    assert train_module.main(["--model", str(tmp_path / "missing.pt")]) == 2
    model = tmp_path / "base.pt"
    model.touch()
    assert train_module.main(
        ["--model", str(model), "--resume", str(tmp_path / "missing-last.pt")]
    ) == 2
    assert "ultralytics" not in imports


def test_train_explicit_resume_uses_checkpoint_as_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit checkpoint initializes YOLO and enables resume=True."""
    model, checkpoint = tmp_path / "base.pt", tmp_path / "last.pt"
    model.touch()
    checkpoint.touch()
    fake_runtime_imports(monkeypatch, train_module)
    assert train_module.main(
        ["--model", str(model), "--resume", str(checkpoint)]
    ) == 0
    assert FakeYOLO.instances[0].model == str(checkpoint.resolve())
    assert FakeYOLO.instances[0].train_calls[0]["resume"] is True


def test_train_dependency_and_execution_failures_return_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Runtime import and training exceptions become diagnostic return code 1."""
    model = tmp_path / "base.pt"
    model.touch()
    real_import = importlib.import_module

    def missing_runtime(name: str, package: str | None = None) -> object:
        if name == "ultralytics":
            raise ModuleNotFoundError("runtime absent")
        return real_import(name, package)

    monkeypatch.setattr(train_module.importlib, "import_module", missing_runtime)
    assert train_module.main(["--model", str(model)]) == 1
    assert "runtime absent" in capsys.readouterr().err

    fake_runtime_imports(monkeypatch, train_module)
    FakeYOLO.train_error = RuntimeError("fake training failed")
    assert train_module.main(["--model", str(model)]) == 1
    assert "fake training failed" in capsys.readouterr().err


def test_export_onnx_default_and_optional_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ONNX defaults and dynamic/simplify flags are forwarded exactly."""
    weights = tmp_path / "best.pt"
    weights.touch()
    imports = fake_runtime_imports(monkeypatch, export_module)
    assert export_module.main(["--weights", str(weights)]) == 0
    assert imports == ["ultralytics"]
    assert FakeYOLO.instances[0].export_calls == [
        {
            "format": "onnx", "imgsz": 320, "device": "cpu", "half": False,
            "int8": False, "dynamic": False, "opset": 12, "simplify": False,
        }
    ]

    assert export_module.main(
        ["--weights", str(weights), "--format", "onnx", "--imgsz", "32",
         "--opset", "20", "--dynamic", "--simplify"]
    ) == 0
    assert FakeYOLO.instances[1].export_calls[0] == {
        "format": "onnx", "imgsz": 32, "device": "cpu", "half": False,
        "int8": False, "dynamic": True, "opset": 20, "simplify": True,
    }


@pytest.mark.parametrize("flag", ["--half", "--int8"])
def test_export_onnx_rejects_half_and_int8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    """ONNX is constrained to FP32 for the deployment contract."""
    weights = tmp_path / "best.pt"
    weights.touch()
    imports = fake_runtime_imports(monkeypatch, export_module)
    args = ["--weights", str(weights), flag]
    if flag == "--int8":
        args += ["--dataset", str(write_dataset(tmp_path / "calib.yaml"))]
    assert export_module.main(args) == 2
    assert "ultralytics" not in imports


def test_export_ncnn_fp32_and_half_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NCNN supports its plain FP32 and explicit FP16 forms."""
    weights = tmp_path / "best.pt"
    weights.touch()
    fake_runtime_imports(monkeypatch, export_module)
    assert export_module.main(["--weights", str(weights), "--format", "ncnn"]) == 0
    assert FakeYOLO.instances[0].export_calls[0] == {
        "format": "ncnn", "imgsz": 320, "device": "cpu", "half": False,
        "int8": False, "dynamic": False,
    }
    assert export_module.main(
        ["--weights", str(weights), "--format", "ncnn", "--half"]
    ) == 0
    assert FakeYOLO.instances[1].export_calls[0]["half"] is True


def test_export_ncnn_int8_validates_and_forwards_calibration_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NCNN INT8 receives a fully validated local seven-label calibration YAML."""
    weights = tmp_path / "best.pt"
    weights.touch()
    dataset = write_dataset(tmp_path / "calib.yaml", names=dict(enumerate(LABELS)))
    imports = fake_runtime_imports(monkeypatch, export_module)
    assert export_module.main(
        ["--weights", str(weights), "--format", "ncnn", "--int8",
         "--dataset", str(dataset)]
    ) == 0
    assert imports == ["yaml", "ultralytics"]
    assert FakeYOLO.instances[0].export_calls == [
        {
            "format": "ncnn", "imgsz": 320, "device": "cpu", "half": False,
            "int8": True, "dynamic": False, "data": str(dataset.resolve()),
        }
    ]


@pytest.mark.parametrize(
    "options",
    [
        ["--format", "ncnn", "--half", "--int8", "--dataset", "CALIB"],
        ["--format", "ncnn", "--int8"],
        ["--format", "ncnn", "--dynamic"],
        ["--format", "ncnn", "--simplify"],
    ],
)
def test_export_ncnn_rejects_incompatible_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str]
) -> None:
    """Mutually exclusive and format-specific NCNN options fail before runtime import."""
    weights = tmp_path / "best.pt"
    weights.touch()
    dataset = write_dataset(tmp_path / "calib.yaml")
    options = [str(dataset) if item == "CALIB" else item for item in options]
    imports = fake_runtime_imports(monkeypatch, export_module)
    assert export_module.main(["--weights", str(weights), *options]) == 2
    assert "ultralytics" not in imports


@pytest.mark.parametrize(
    "options",
    [
        ["--imgsz", "31"], ["--imgsz", "321"], ["--device", "0"],
        ["--device", "cuda"], ["--opset", "9"], ["--opset", "21"],
    ],
)
def test_export_enforces_zero2w_cpu_size_and_opset_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str]
) -> None:
    """Zero 2 W export remains CPU-only, at most 320, and uses supported opsets."""
    weights = tmp_path / "best.pt"
    weights.touch()
    imports = fake_runtime_imports(monkeypatch, export_module)
    assert export_module.main(["--weights", str(weights), *options]) == 2
    assert "ultralytics" not in imports


@pytest.mark.parametrize("imgsz", [32, 320])
def test_export_accepts_minimum_and_maximum_image_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, imgsz: int
) -> None:
    """Both documented image-size endpoints are accepted."""
    weights = tmp_path / "best.pt"
    weights.touch()
    fake_runtime_imports(monkeypatch, export_module)
    assert export_module.main(
        ["--weights", str(weights), "--imgsz", str(imgsz)]
    ) == 0
    assert FakeYOLO.instances[0].export_calls[0]["imgsz"] == imgsz


@pytest.mark.parametrize(
    ("changes", "mutator"),
    [
        ({"names": LABELS[::-1]}, None),
        ({"names": dict(enumerate((*LABELS, "extra")))}, None),
        ({"nc": 6}, None),
        ({"root": "https://example.test/data"}, None),
        ({"train": None}, None),
        ({"download": "https://example.test/data.zip"}, None),
        ({}, "non_mapping"),
        ({}, "empty_mapping"),
        ({}, "null_root"),
        ({}, "malformed"),
    ],
)
def test_export_int8_rejects_bad_calibration_yaml_with_input_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    mutator: str | None,
) -> None:
    """Every invalid calibration YAML form returns code 2 instead of escaping."""
    weights = tmp_path / "best.pt"
    weights.touch()
    dataset = write_dataset(tmp_path / "calib.yaml", **changes)
    if mutator == "non_mapping":
        dataset.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    elif mutator == "empty_mapping":
        dataset.write_text("{}\n", encoding="utf-8")
    elif mutator == "null_root":
        dataset.write_text("null\n", encoding="utf-8")
    elif mutator == "malformed":
        dataset.write_text("names: [broken", encoding="utf-8")
    imports = fake_runtime_imports(monkeypatch, export_module)
    result = export_module.main(
        ["--weights", str(weights), "--format", "ncnn", "--int8",
         "--dataset", str(dataset)]
    )
    assert result == 2
    assert "ultralytics" not in imports


@pytest.mark.parametrize(
    ("names_yaml", "case"),
    [
        (
            "  0: angry\n  1: disgusted\n  2: fearful\n  3: happy\n"
            "  4: neutral\n  5: sad\n",
            "missing index 6",
        ),
        (
            "  0: angry\n  '0': angry\n  1: disgusted\n  2: fearful\n"
            "  3: happy\n  4: neutral\n  5: sad\n  6: surprised\n",
            "integer/string duplicate index",
        ),
        (
            "  0: wrong\n  0: angry\n  1: disgusted\n  2: fearful\n"
            "  3: happy\n  4: neutral\n  5: sad\n  6: surprised\n",
            "duplicate YAML mapping key",
        ),
        (
            "  '00': angry\n  1: disgusted\n  2: fearful\n  3: happy\n"
            "  4: neutral\n  5: sad\n  6: surprised\n",
            "noncanonical numeric key",
        ),
    ],
)
def test_export_int8_rejects_missing_duplicate_and_noncanonical_label_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    names_yaml: str,
    case: str,
) -> None:
    """Calibration mappings must contain one canonical representation of each index."""
    weights = tmp_path / "best.pt"
    weights.touch()
    dataset = tmp_path / "calib.yaml"
    dataset.write_text(
        "path: dataset\ntrain: images/train\nval: images/val\nnc: 7\nnames:\n"
        + names_yaml,
        encoding="utf-8",
    )
    imports = fake_runtime_imports(monkeypatch, export_module)
    result = export_module.main(
        [
            "--weights",
            str(weights),
            "--format",
            "ncnn",
            "--int8",
            "--dataset",
            str(dataset),
        ]
    )
    assert result == 2, case
    assert "ultralytics" not in imports


@pytest.mark.parametrize(
    ("yaml_text", "case"),
    [
        (
            "path: dataset\ntrain: images/train\nval: images/val\nnc: 7\n"
            "metadata:\n  source: first\n  source: second\n"
            "names: [angry, disgusted, fearful, happy, neutral, sad, surprised]\n",
            "nested duplicate mapping key",
        ),
        (
            "? [unhashable, key]\n: rejected\npath: dataset\ntrain: images/train\n"
            "val: images/val\nnc: 7\n"
            "names: [angry, disgusted, fearful, happy, neutral, sad, surprised]\n",
            "unhashable mapping key",
        ),
        (
            "path: dataset\ntrain: images/train\nval: images/val\nnc: 7\n"
            "names: [angry, disgusted",
            "YAML syntax error",
        ),
    ],
)
def test_export_rejects_recursive_yaml_structure_errors_before_heavy_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    yaml_text: str,
    case: str,
) -> None:
    """Recursive duplicates, unhashable keys, and syntax errors stop before export."""
    weights = tmp_path / "best.pt"
    weights.touch()
    dataset = tmp_path / "calib.yaml"
    dataset.write_text(yaml_text, encoding="utf-8")
    imports = fake_runtime_imports(monkeypatch, export_module)
    result = export_module.main(
        [
            "--weights",
            str(weights),
            "--format",
            "ncnn",
            "--int8",
            "--dataset",
            str(dataset),
        ]
    )
    assert result == 2, case
    assert imports == ["yaml"]
    assert FakeYOLO.instances == []


def test_export_unique_loader_does_not_change_global_yaml_safe_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate-key enforcement remains local to calibration file loading."""
    weights = tmp_path / "best.pt"
    weights.touch()
    dataset = tmp_path / "duplicate.yaml"
    dataset.write_text(
        "path: dataset\ntrain: images/train\nval: images/val\nnc: 7\nnames:\n"
        "  0: wrong\n  0: angry\n  1: disgusted\n  2: fearful\n"
        "  3: happy\n  4: neutral\n  5: sad\n  6: surprised\n",
        encoding="utf-8",
    )
    original_safe_load = yaml.safe_load
    imports = fake_runtime_imports(monkeypatch, export_module)
    assert export_module.main(
        [
            "--weights",
            str(weights),
            "--format",
            "ncnn",
            "--int8",
            "--dataset",
            str(dataset),
        ]
    ) == 2
    assert imports == ["yaml"]
    assert FakeYOLO.instances == []
    assert yaml.safe_load is original_safe_load
    assert yaml.safe_load("ordinary: first\nordinary: second\n") == {
        "ordinary": "second"
    }


def test_export_weights_and_dataset_local_file_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing weights and missing calibration files fail before model loading."""
    imports = fake_runtime_imports(monkeypatch, export_module)
    assert export_module.main(["--weights", str(tmp_path / "missing.pt")]) == 2
    weights = tmp_path / "best.pt"
    weights.touch()
    assert export_module.main(
        ["--weights", str(weights), "--format", "ncnn", "--int8",
         "--dataset", str(tmp_path / "missing.yaml")]
    ) == 2
    assert "ultralytics" not in imports


def test_export_dependency_and_execution_failures_return_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exporter runtime failures provide code 1 and a diagnostic message."""
    weights = tmp_path / "best.pt"
    weights.touch()
    real_import = importlib.import_module

    def missing_runtime(name: str, package: str | None = None) -> object:
        if name == "ultralytics":
            raise ModuleNotFoundError("export runtime absent")
        return real_import(name, package)

    monkeypatch.setattr(export_module.importlib, "import_module", missing_runtime)
    assert export_module.main(["--weights", str(weights)]) == 1
    assert "export runtime absent" in capsys.readouterr().err

    fake_runtime_imports(monkeypatch, export_module)
    FakeYOLO.export_error = RuntimeError("fake export failed")
    assert export_module.main(["--weights", str(weights)]) == 1
    assert "fake export failed" in capsys.readouterr().err


@pytest.mark.parametrize("module", ["training.train", "training.export"])
def test_module_import_does_not_load_ml_runtime(module: str) -> None:
    """Fresh module imports do not request Torch or Ultralytics."""
    project_root = Path(__file__).resolve().parents[1]
    script = f'''
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {{"torch", "ultralytics"}}:
            raise RuntimeError(f"forbidden dependency requested: {{fullname}}")
        return None

sys.meta_path.insert(0, Blocker())
import {module}
print("lazy-import-ok")
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=project_root, text=True,
        capture_output=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lazy-import-ok"


@pytest.mark.parametrize("module", ["training.train", "training.export"])
def test_help_command_does_not_load_ml_or_yaml_runtime(module: str) -> None:
    """Help exits successfully without importing heavy ML or even YAML dependencies."""
    project_root = Path(__file__).resolve().parents[1]
    script = f'''
import importlib.abc
import runpy
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {{"torch", "ultralytics", "yaml"}}:
            raise RuntimeError(f"forbidden dependency requested: {{fullname}}")
        return None

sys.meta_path.insert(0, Blocker())
sys.argv = ["{module}", "--help"]
runpy.run_module("{module}", run_name="__main__")
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=project_root, text=True,
        capture_output=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "forbidden dependency" not in result.stderr


@pytest.mark.parametrize(
    ("module", "argv"),
    [
        (train_module, []),
        (train_module, ["--model", "model.pt", "--unknown"]),
        (export_module, []),
        (export_module, ["--weights", "best.pt", "--format", "torchscript"]),
    ],
)
def test_argparse_missing_unknown_and_invalid_choice_exit_two(
    module: object, argv: list[str]
) -> None:
    """CLI syntax errors retain argparse's standard exit status 2."""
    with pytest.raises(SystemExit) as exc_info:
        module.main(argv)
    assert exc_info.value.code == 2
