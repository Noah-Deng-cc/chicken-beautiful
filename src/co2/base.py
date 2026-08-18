"""CO2 基础契约：输入为 ppm 与配置阈值，输出为 Co2Level/Co2Reading；依赖标准库和领域模型，不绑定传输总线。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import TracebackType

from src.domain import Co2Level, Co2Reading


@dataclass(frozen=True, slots=True)
class Co2Thresholds:
    """由配置提供的 CO2 浓度分级阈值。"""

    elevated: int
    poor: int

    def __post_init__(self) -> None:
        """校验阈值范围和严格递增关系。\n\nArgs: 无。\nReturns: 无。\nRaises: TypeError: 阈值不是整数。ValueError: 阈值越界或顺序无效。"""
        values = (self.elevated, self.poor)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("CO2 thresholds must be integers")
        if not 0 <= self.elevated < self.poor <= 100_000:
            raise ValueError("CO2 thresholds must satisfy 0 <= elevated < poor <= 100000")


def classify_co2(ppm: int, thresholds: Co2Thresholds) -> Co2Level:
    """按配置阈值分级 ppm。\n\nArgs: ppm: 浓度整数。thresholds: 已校验阈值。\nReturns: good、elevated、poor 或 invalid。\nRaises: TypeError: thresholds 类型错误。"""
    if not isinstance(thresholds, Co2Thresholds):
        raise TypeError("thresholds must be Co2Thresholds")
    if isinstance(ppm, bool) or not isinstance(ppm, int) or not 0 <= ppm <= 100_000:
        return Co2Level.INVALID
    if ppm < thresholds.elevated:
        return Co2Level.GOOD
    if ppm < thresholds.poor:
        return Co2Level.ELEVATED
    return Co2Level.POOR


class Co2Sensor(ABC):
    """可替换的 CO2 采集接口。"""

    @property
    @abstractmethod
    def closed(self) -> bool:
        """返回关闭状态。\n\nArgs: 无。\nReturns: 已关闭为 True。\nRaises: 无。"""
        raise NotImplementedError

    @abstractmethod
    def read(self) -> Co2Reading | None:
        """读取一次浓度。\n\nArgs: 无。\nReturns: 有效/无效读数，断连或暂不可用为 None。\nRaises: RuntimeError: 传感器已关闭。"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """幂等关闭传感器资源。\n\nArgs: 无。\nReturns: 无。\nRaises: 无。"""
        raise NotImplementedError

    def __enter__(self) -> Co2Sensor:
        """进入传感器上下文。\n\nArgs: 无。\nReturns: 当前传感器。\nRaises: RuntimeError: 已关闭。"""
        if self.closed:
            raise RuntimeError("CO2 sensor is closed")
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        """退出上下文并关闭资源。\n\nArgs: exc_type: 异常类型。exc_value: 异常对象。traceback: 回溯。\nReturns: 无。\nRaises: 无。"""
        self.close()
