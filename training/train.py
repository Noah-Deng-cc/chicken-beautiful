"""情绪训练入口：输入本地模型/数据配置和 CLI 参数，输出训练产物；依赖 pathlib、PyYAML、懒导入 Ultralytics。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib
from pathlib import Path
import sys


LABELS = ("angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised")


def _parser() -> argparse.ArgumentParser:
    """构建训练参数解析器。\n\n    Args: 无。\n    Returns: 配置完成的 ArgumentParser。\n    Raises: 无。"""
    parser = argparse.ArgumentParser(description="使用本地 YOLO 模型训练七类情绪检测器")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("emotions.yaml"))
    parser.add_argument("--model", type=Path, required=True, help="本地 .pt 模型或检查点")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="开发机训练设备，如 0 或 cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", default="emotion-yolo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", nargs="?", const=True, default=False, metavar="CHECKPOINT")
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def _labels(raw: object) -> tuple[str, ...]:
    """规范化 YAML 中的标签映射。\n\n    Args: raw: names 字段值。\n    Returns: 按类别编号排序的标签元组。\n    Raises: ValueError: names 不是连续的 0 到 6 映射或列表。"""
    if isinstance(raw, Mapping):
        try:
            keys = {int(key) for key in raw}
            values = tuple(str(raw[index] if index in raw else raw[str(index)]) for index in range(7))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("dataset names must map indices 0..6") from exc
        if len(raw) != 7 or keys != set(range(7)):
            raise ValueError("dataset names must contain exactly indices 0..6")
        return values
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(str(item) for item in raw)
    raise ValueError("dataset names must be a mapping or sequence")


def validate_dataset(path: Path) -> Path:
    """校验本地数据 YAML 和固定标签顺序。\n\n    Args: path: 数据 YAML 路径。\n    Returns: 解析后的绝对路径。\n    Raises: FileNotFoundError: 文件不存在。ValueError: YAML、路径字段或标签不合规。ImportError: PyYAML 不可用。"""
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {source}")
    yaml = importlib.import_module("yaml")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"unable to parse dataset YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("dataset YAML root must be a mapping")
    if "download" in raw:
        raise ValueError("dataset YAML must not define automatic downloads")
    if raw.get("nc") != 7 or _labels(raw.get("names")) != LABELS:
        raise ValueError(f"dataset labels must be exactly: {', '.join(LABELS)}")
    for field in ("path", "train", "val"):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip() or "://" in value:
            raise ValueError(f"dataset field '{field}' must be a non-empty local path")
    return source


def _local_file(path: Path, description: str) -> Path:
    """要求输入为已有本地文件。\n\n    Args: path: 候选路径。description: 错误信息中的用途。\n    Returns: 解析后的绝对路径。\n    Raises: FileNotFoundError: 文件不存在。"""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"local {description} not found: {resolved}")
    return resolved


def _positive_args(args: argparse.Namespace) -> None:
    """校验训练数值与文本参数。\n\n    Args: args: 已解析命令参数。\n    Returns: 无。\n    Raises: ValueError: 参数超出允许范围。"""
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1 or args.workers < 0 or args.seed < 0:
        raise ValueError("epochs/batch must be positive, imgsz >= 32, workers/seed >= 0")
    if not str(args.device).strip() or not str(args.name).strip():
        raise ValueError("device and name must not be empty")


def main(argv: Sequence[str] | None = None) -> int:
    """校验输入并调用 Ultralytics 训练。\n\n    Args: argv: 可选 CLI 参数；None 时读取 sys.argv。\n    Returns: 成功为 0，输入错误为 2，依赖或训练失败为 1。\n    Raises: SystemExit: argparse 遇到未知或缺失参数。"""
    args = _parser().parse_args(argv)
    try:
        _positive_args(args)
        dataset = validate_dataset(args.dataset)
        model_path = _local_file(Path(args.model), "model")
        resume = args.resume
        if isinstance(resume, str):
            model_path, resume = _local_file(Path(resume), "resume checkpoint"), True
    except (ImportError, OSError, ValueError) as exc:
        print(f"训练输入错误: {exc}", file=sys.stderr)
        return 2
    try:
        ultralytics = importlib.import_module("ultralytics")
        model = ultralytics.YOLO(str(model_path))
        model.train(data=str(dataset), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, workers=args.workers, project=str(args.project.resolve()),
                    name=args.name, seed=args.seed, resume=bool(resume), exist_ok=args.exist_ok)
        return 0
    except Exception as exc:
        print(f"训练执行失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
