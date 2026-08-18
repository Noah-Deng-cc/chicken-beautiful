"""热成像包入口：输入热阵列或模拟结果，输出硬件无关接口与温度读数；仅依赖标准库和领域模型。"""

from .base import ThermalSensor, summarize_temperature_frame, temperature_from_frame
from .mock import Clock, MockThermalItem, MockThermalSensor

__all__ = [
    "Clock",
    "MockThermalItem",
    "MockThermalSensor",
    "ThermalSensor",
    "summarize_temperature_frame",
    "temperature_from_frame",
]
