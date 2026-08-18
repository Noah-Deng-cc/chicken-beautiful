"""热成像基础契约：输入热阵列数值，输出温度读数；仅依赖标准库与领域模型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import datetime
from math import fsum, isfinite
from numbers import Real
from types import TracebackType

from src.domain.models import TemperatureReading


def _temperature_bounds(minimum: float, maximum: float) -> tuple[float, float]:
    """校验温度闭区间配置。\n\n    Args: minimum: 最低有效摄氏度。maximum: 最高有效摄氏度。\n    Returns: 浮点化的下界与上界。\n    Raises: TypeError: 阈值不是实数。ValueError: 阈值非有限、越过领域范围或顺序错误。"""
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in (minimum, maximum)):
        raise TypeError("temperature bounds must be real numbers")
    lower, upper = float(minimum), float(maximum)
    if not isfinite(lower) or not isfinite(upper) or lower < -40.0 or upper > 300.0 or lower > upper:
        raise ValueError("temperature bounds must be finite, ordered, and within -40..300")
    return lower, upper


def _flatten_pixels(value: object) -> Iterable[object]:
    """递归展开热阵列，同时保留非法叶值供校验。\n\n    Args: value: 标量、嵌套可迭代对象或非法值。\n    Returns: 深度优先的像素叶值迭代器。\n    Raises: 无。"""
    if isinstance(value, (str, bytes, bytearray)) or isinstance(value, Real):
        yield value
        return
    try:
        iterator = iter(value)  # type: ignore[arg-type]
    except TypeError:
        yield value
        return
    for item in iterator:
        yield from _flatten_pixels(item)


def summarize_temperature_frame(
    frame: Iterable[object], min_valid_celsius: float = 20.0,
    max_valid_celsius: float = 45.0,
) -> tuple[float, float] | None:
    """按全帧有效规则计算最高和平均温度。\n\n    Args: frame: 一维可迭代热阵列；调用方负责展平。min_valid_celsius: 有效下界。max_valid_celsius: 有效上界。\n    Returns: ``(最高值, 平均值)``；空帧或任一像素非法时为 None。\n    Raises: TypeError: 阈值类型错误。ValueError: 阈值范围错误。"""
    lower, upper = _temperature_bounds(min_valid_celsius, max_valid_celsius)
    values = tuple(_flatten_pixels(frame))
    if not values:
        return None
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        number = float(value)
        if not isfinite(number) or not lower <= number <= upper:
            return None
        numbers.append(number)
    return max(numbers), fsum(numbers) / len(numbers)


def temperature_from_frame(
    frame: Iterable[object], timestamp: datetime, min_valid_celsius: float = 20.0,
    max_valid_celsius: float = 45.0,
) -> TemperatureReading | None:
    """将有效热阵列转换为带时间和质量标记的领域读数。\n\n    Args: frame: 一维热阵列。timestamp: 带时区采集时间。min_valid_celsius: 有效下界。max_valid_celsius: 有效上界。\n    Returns: 质量为 good 的 TemperatureReading；帧无效时为 None。\n    Raises: TypeError: 时间或阈值类型错误。ValueError: 时间无时区或阈值无效。"""
    summary = summarize_temperature_frame(frame, min_valid_celsius, max_valid_celsius)
    if summary is None:
        return None
    maximum, average = summary
    return TemperatureReading(timestamp, maximum, average, "good")


class ThermalSensor(ABC):
    """不绑定总线、厂商或阵列类型的热成像传感器契约。"""

    @property
    def closed(self) -> bool:
        """查询传感器关闭状态的默认值。\n\n        Args: 无。\n        Returns: 未提供生命周期状态时为 False。\n        Raises: 无。"""
        return False

    @abstractmethod
    def read(self) -> TemperatureReading | None:
        """读取一次温度。\n\n        Args: 无。\n        Returns: 有效读数；空帧、越界或可恢复硬件故障时为 None。\n        Raises: 无；实现应记录并吸收硬件异常。"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """幂等释放传感器资源。\n\n        Args: 无。\n        Returns: 无。\n        Raises: 无。"""
        raise NotImplementedError

    def __enter__(self) -> ThermalSensor:
        """进入传感器上下文。\n\n        Args: 无。\n        Returns: 未关闭的当前传感器。\n        Raises: RuntimeError: 传感器已关闭。"""
        if self.closed:
            raise RuntimeError("thermal sensor is closed")
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        """退出上下文并关闭传感器。\n\n        Args: exc_type: 异常类型。exc_value: 异常实例。traceback: 回溯对象。\n        Returns: 无，不抑制异常。\n        Raises: 无。"""
        self.close()
