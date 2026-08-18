"""领域序列化：输入为领域对象/JSON 值，输出为稳定 JSON；仅依赖标准库和领域模型。"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
import json
from collections.abc import Mapping, Sequence
from typing import TypeAlias


Serializable: TypeAlias = object


def to_data(value: Serializable) -> object:
    """递归转换领域值为 JSON 兼容数据。\n\n    Args: value: 领域对象、枚举、带时区时间或 JSON 容器。\n    Returns: 仅含 JSON 原生类型的数据。\n    Raises: TypeError: 时间无时区、映射键非字符串或类型不受支持。ValueError: 数值不是有限值。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise TypeError("datetime values must be timezone-aware")
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_data(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("mapping keys must be strings")
        return {key: to_data(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_data(item) for item in value]
    raise TypeError(f"unsupported serializable type: {type(value).__name__}")


def to_json(value: Serializable) -> str:
    """生成键顺序稳定且保留 Unicode 的紧凑 JSON。\n\n    Args: value: 可由 ``to_data`` 转换的值。\n    Returns: UTF-8 友好的 JSON 字符串。\n    Raises: TypeError: 值不可序列化。ValueError: 包含非有限浮点数。"""
    return json.dumps(to_data(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
