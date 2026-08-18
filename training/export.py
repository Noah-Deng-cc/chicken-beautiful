"""模型导出入口：输入本地权重和导出选项，输出 ONNX/NCNN；依赖 pathlib、PyYAML、懒导入 Ultralytics。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib
from pathlib import Path
import sys


LABELS = ("angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised")


def _parser() -> argparse.ArgumentParser:
    """构建带格式兼容说明的导出解析器。\n\n    Args: 无。\n    Returns: 配置完成的 ArgumentParser。\n    Raises: 无。"""
    parser = argparse.ArgumentParser(
        description="导出 Zero 2 W 模型；ONNX 支持 dynamic，NCNN 支持 half/int8")
    parser.add_argument("--weights", type=Path, required=True, help="本地训练权重")
    parser.add_argument("--format", choices=("onnx", "ncnn"), default="onnx")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true", help="仅 ONNX；Zero 2 W 通常使用固定形状更快")
    parser.add_argument("--half", action="store_true", help="仅 NCNN FP16")
    parser.add_argument("--int8", action="store_true", help="仅 NCNN，必须提供校准数据")
    parser.add_argument("--dataset", type=Path, help="INT8 校准所需七类数据 YAML")
    parser.add_argument("--simplify", action="store_true", help="仅 ONNX")
    return parser


def _local_file(path: Path, description: str) -> Path:
    """要求输入为已有本地文件。\n\n    Args: path: 候选路径。description: 文件用途。\n    Returns: 解析后的绝对路径。\n    Raises: FileNotFoundError: 文件不存在。"""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"local {description} not found: {resolved}")
    return resolved


def _mapping_labels(names: Mapping[object, object]) -> tuple[str, ...]:
    """严格规范化整数或数字字符串标签键。\n\n    Args: names: YAML names 映射。\n    Returns: 按 0 到 6 排序的标签。\n    Raises: ValueError: 键类型、集合、数量或唯一性不合规。"""
    normalized: dict[int, str] = {}
    for key, value in names.items():
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            raise ValueError("calibration label keys must be integers or canonical numeric strings")
        try:
            index = int(key)
        except ValueError as exc:
            raise ValueError("calibration label keys must be 0..6") from exc
        if isinstance(key, str) and key != str(index):
            raise ValueError("calibration string label keys must be canonical")
        if index in normalized:
            raise ValueError("calibration label keys must not duplicate an index")
        normalized[index] = str(value)
    if len(names) != 7 or set(normalized) != set(range(7)):
        raise ValueError("calibration labels must contain exactly indices 0..6")
    return tuple(normalized[index] for index in range(7))


def _load_unique_yaml(yaml_module: object, text: str) -> object:
    """使用局部加载器递归拒绝重复或不可哈希映射键。\n\n    Args: yaml_module: 延迟导入的 PyYAML 模块。text: YAML 原文。\n    Returns: YAML 构造出的对象。\n    Raises: ValueError: 映射键重复或不可哈希。yaml.YAMLError: YAML 语法错误。"""
    class UniqueKeyLoader(yaml_module.SafeLoader):  # type: ignore[name-defined,misc]
        """仅本次校准文件使用的 SafeLoader 子类。"""

    def construct_unique_mapping(loader: object, node: object,
                                 deep: bool = False) -> dict[object, object]:
        """构造映射并在覆盖发生前检测键。\n\n        Args: loader: 当前局部加载器。node: YAML 映射节点。deep: 是否深层构造。\n        Returns: 不含重复键的映射。\n        Raises: ValueError: 键重复或不可哈希。"""
        mapping: dict[object, object] = {}
        construct = getattr(loader, "construct_object")
        for key_node, value_node in getattr(node, "value"):
            key = construct(key_node, deep=deep)
            try:
                if key in mapping:
                    raise ValueError(f"duplicate YAML mapping key: {key!r}")
                mapping[key] = construct(value_node, deep=deep)
            except TypeError as exc:
                raise ValueError("YAML mapping keys must be hashable") from exc
        return mapping

    tag = yaml_module.resolver.BaseResolver.DEFAULT_MAPPING_TAG  # type: ignore[attr-defined]
    UniqueKeyLoader.add_constructor(tag, construct_unique_mapping)
    return yaml_module.load(text, Loader=UniqueKeyLoader)  # type: ignore[attr-defined]


def _calibration_dataset(path: Path) -> Path:
    """校验 INT8 校准 YAML 的七类标签。\n\n    Args: path: 校准数据 YAML。\n    Returns: 解析后的绝对路径。\n    Raises: FileNotFoundError: 文件不存在。ValueError: YAML 或标签不合规。ImportError: PyYAML 不可用。"""
    source = _local_file(path, "calibration dataset YAML")
    yaml = importlib.import_module("yaml")
    try:
        raw = _load_unique_yaml(yaml, source.read_text(encoding="utf-8"))
        names = raw.get("names") if isinstance(raw, Mapping) else None
        if isinstance(names, Mapping):
            labels = _mapping_labels(names)
        elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            labels = tuple(str(item) for item in names)
        else:
            labels = ()
    except Exception as exc:
        raise ValueError(f"unable to parse calibration dataset: {exc}") from exc
    local_paths = isinstance(raw, Mapping) and all(
        isinstance(raw.get(field), str) and raw[field].strip() and "://" not in raw[field]
        for field in ("path", "train", "val"))
    if not local_paths or "download" in raw or raw.get("nc") != 7 or labels != LABELS:
        raise ValueError("calibration dataset must use the fixed seven labels and no download")
    return source


def _validate(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """校验导出格式和 Zero 2 W 兼容约束。\n\n    Args: args: 已解析参数。\n    Returns: 本地权重与可选校准 YAML。\n    Raises: ValueError: 选项组合不兼容。FileNotFoundError: 输入文件不存在。"""
    if not 32 <= args.imgsz <= 320 or args.opset < 10 or args.opset > 20:
        raise ValueError("imgsz must be 32..320 and opset must be 10..20")
    if str(args.device).lower() != "cpu":
        raise ValueError("Zero 2 W export validation requires device=cpu")
    if args.format == "onnx" and (args.half or args.int8):
        raise ValueError("ONNX path supports FP32 only; use NCNN for half/int8")
    if args.format == "ncnn" and (args.dynamic or args.simplify):
        raise ValueError("NCNN does not support dynamic or ONNX simplify options")
    if args.half and args.int8:
        raise ValueError("half and int8 are mutually exclusive")
    if args.int8 and args.dataset is None:
        raise ValueError("NCNN int8 requires --dataset calibration YAML")
    if not args.int8 and args.dataset is not None:
        raise ValueError("--dataset is only valid with --int8")
    weights = _local_file(args.weights, "weights")
    dataset = _calibration_dataset(args.dataset) if args.dataset is not None else None
    return weights, dataset


def main(argv: Sequence[str] | None = None) -> int:
    """校验参数并调用 Ultralytics 导出。\n\n    Args: argv: 可选 CLI 参数；None 时读取 sys.argv。\n    Returns: 成功为 0，输入错误为 2，依赖或导出失败为 1。\n    Raises: SystemExit: argparse 遇到未知或缺失参数。"""
    args = _parser().parse_args(argv)
    try:
        weights, dataset = _validate(args)
    except (ImportError, OSError, ValueError) as exc:
        print(f"导出输入错误: {exc}", file=sys.stderr)
        return 2
    options: dict[str, object] = {"format": args.format, "imgsz": args.imgsz,
                                  "device": args.device, "half": args.half,
                                  "int8": args.int8, "dynamic": args.dynamic}
    if args.format == "onnx":
        options.update(opset=args.opset, simplify=args.simplify)
    if dataset is not None:
        options["data"] = str(dataset)
    try:
        ultralytics = importlib.import_module("ultralytics")
        ultralytics.YOLO(str(weights)).export(**options)
        return 0
    except Exception as exc:
        print(f"导出执行失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
