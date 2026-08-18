"""CO2 组件入口：输入为浓度与阈值，输出为分级、传感器接口和模拟实现；依赖领域模型与标准库。"""

from .base import Co2Sensor, Co2Thresholds, classify_co2
from .mock import MockCo2Sensor

__all__ = ["Co2Sensor", "Co2Thresholds", "MockCo2Sensor", "classify_co2"]
