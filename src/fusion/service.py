"""融合服务：输入为 SystemSnapshot 和本地规则，输出为智能体上下文与本地告警；依赖标准库和领域模型，不调用云端且不作医学诊断。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite

from src.domain import Co2Reading, EmotionReading, SystemSnapshot, TemperatureReading


@dataclass(frozen=True, slots=True)
class FusionRules:
    """由规则配置文件提供的新鲜度与本地告警阈值。"""

    high_temperature_celsius: float
    high_co2_ppm: int
    stale_after_seconds: float
    temperature_quality: str

    def __post_init__(self) -> None:
        """校验规则值。\n\nArgs: 无。\nReturns: 无。\nRaises: TypeError: 字段类型错误。ValueError: 范围或质量文本无效。"""
        if isinstance(self.high_temperature_celsius, bool) or not isinstance(self.high_temperature_celsius, (int, float)):
            raise TypeError("high_temperature_celsius must be a number")
        if not isfinite(float(self.high_temperature_celsius)) or not 0.0 <= float(self.high_temperature_celsius) <= 100.0:
            raise ValueError("high_temperature_celsius must be within [0, 100]")
        if isinstance(self.high_co2_ppm, bool) or not isinstance(self.high_co2_ppm, int):
            raise TypeError("high_co2_ppm must be an integer")
        if not 0 <= self.high_co2_ppm <= 100_000:
            raise ValueError("high_co2_ppm must be within [0, 100000]")
        if isinstance(self.stale_after_seconds, bool) or not isinstance(self.stale_after_seconds, (int, float)):
            raise TypeError("stale_after_seconds must be a number")
        if not isfinite(float(self.stale_after_seconds)) or self.stale_after_seconds < 0:
            raise ValueError("stale_after_seconds must be finite and non-negative")
        if not isinstance(self.temperature_quality, str) or not self.temperature_quality.strip():
            raise ValueError("temperature_quality must be a non-empty string")


def _metadata(timestamp: datetime, snapshot_time: datetime, quality: str,
              rules: FusionRules) -> dict[str, object]:
    """生成包含时间、质量与新鲜度的传感器元数据。\n\nArgs: timestamp: 读数时间。snapshot_time: 快照时间。quality: 质量标记。rules: 融合规则。\nReturns: JSON 兼容元数据。\nRaises: 无。"""
    age = max(0.0, (snapshot_time - timestamp).total_seconds())
    return {"timestamp": timestamp.isoformat(), "age_seconds": round(age, 3),
            "fresh": age <= rules.stale_after_seconds, "quality": quality}


class FusionService:
    """把独立模态转换为可安全传递给智能体的观察上下文。"""

    def __init__(self, rules: FusionRules) -> None:
        """保存不可变的本地规则。\n\nArgs: rules: 外部配置生成的规则。\nReturns: 无。\nRaises: TypeError: rules 类型错误。"""
        if not isinstance(rules, FusionRules):
            raise TypeError("rules must be FusionRules")
        self._rules = rules

    def build_context(self, snapshot: SystemSnapshot) -> dict[str, object]:
        """构建不含医学诊断的结构化上下文。\n\nArgs: snapshot: 已校验系统快照。\nReturns: 传感器时间、质量、新鲜度和到期提醒上下文。\nRaises: TypeError: snapshot 类型错误。"""
        if not isinstance(snapshot, SystemSnapshot):
            raise TypeError("snapshot must be SystemSnapshot")
        timestamp = snapshot.timestamp
        return {
            "snapshot_timestamp": timestamp.isoformat(),
            "disclaimer": "传感器观察结果不构成医学诊断。",
            "emotion": self._emotion(snapshot.emotion, timestamp),
            "temperature": self._temperature(snapshot.temperature, timestamp),
            "co2": self._co2(snapshot.co2, timestamp),
            "reminders": [{"reminder_id": item.reminder_id, "message": item.message,
                           "due_at": item.due_at.isoformat(), "acknowledged": item.acknowledged,
                           "due": not item.acknowledged and item.due_at <= timestamp}
                          for item in snapshot.reminders],
        }

    def _emotion(self, reading: EmotionReading | None, timestamp: datetime) -> dict[str, object] | None:
        """转换可选情绪观察。\n\nArgs: reading: 情绪读数。timestamp: 快照时间。\nReturns: 观察字典或 None。\nRaises: 无。"""
        if reading is None:
            return None
        return _metadata(reading.timestamp, timestamp, "observed", self._rules) | {
            "dominant": reading.dominant.value, "confidence": reading.confidence,
            "valence": reading.valence, "arousal": reading.arousal, "person_id": reading.person_id}

    def _temperature(self, reading: TemperatureReading | None,
                     timestamp: datetime) -> dict[str, object] | None:
        """转换可选温度观察。\n\nArgs: reading: 温度读数。timestamp: 快照时间。\nReturns: 观察字典或 None。\nRaises: 无。"""
        if reading is None:
            return None
        return _metadata(reading.timestamp, timestamp, reading.quality, self._rules) | {
            "maximum_celsius": reading.maximum_celsius, "average_celsius": reading.average_celsius}

    def _co2(self, reading: Co2Reading | None, timestamp: datetime) -> dict[str, object] | None:
        """转换可选 CO2 观察。\n\nArgs: reading: CO2 读数。timestamp: 快照时间。\nReturns: 观察字典或 None。\nRaises: 无。"""
        if reading is None:
            return None
        quality = "valid" if reading.ppm is not None else "invalid"
        return _metadata(reading.timestamp, timestamp, quality, self._rules) | {
            "ppm": reading.ppm, "level": reading.level.value}

    def local_alerts(self, snapshot: SystemSnapshot) -> list[str]:
        """仅由本地快照和规则生成高温/高 CO2 告警。\n\nArgs: snapshot: 已校验系统快照。\nReturns: 稳定告警代码列表。\nRaises: TypeError: snapshot 类型错误。"""
        if not isinstance(snapshot, SystemSnapshot):
            raise TypeError("snapshot must be SystemSnapshot")
        alerts: list[str] = []
        temperature, co2 = snapshot.temperature, snapshot.co2
        if temperature is not None and temperature.quality == self._rules.temperature_quality:
            fresh = _metadata(temperature.timestamp, snapshot.timestamp, temperature.quality, self._rules)["fresh"]
            if fresh and temperature.maximum_celsius >= self._rules.high_temperature_celsius:
                alerts.append("temperature_observation_high")
        if co2 is not None and co2.ppm is not None:
            fresh = _metadata(co2.timestamp, snapshot.timestamp, "valid", self._rules)["fresh"]
            if fresh and co2.ppm >= self._rules.high_co2_ppm:
                alerts.append("co2_concentration_high")
        return alerts
