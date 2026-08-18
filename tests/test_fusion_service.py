"""T18 融合服务验收测试：仅使用固定领域快照，验证离线规则和 JSON 上下文。"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest
import yaml

import src.fusion as fusion
from src.domain import Co2Level, Co2Reading, Emotion, EmotionReading, Reminder, SystemSnapshot, TemperatureReading
from src.fusion import FusionRules, FusionService


UTC = timezone.utc
SHANGHAI = timezone(timedelta(hours=8))
BASE_TIME = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)


def rules() -> FusionRules:
    """构造 T18 固定的本地融合规则。

    Args: 无。
    Returns: 可用于边界测试的融合规则。
    Raises: 无。
    """
    return FusionRules(38.0, 1_500, 60.0, "good")


def complete_snapshot(timestamp: datetime = BASE_TIME) -> SystemSnapshot:
    """创建包含所有模态和两类提醒的合法快照。

    Args: timestamp: 快照采样时刻。
    Returns: 完整系统快照。
    Raises: 无。
    """
    return SystemSnapshot(
        timestamp=timestamp,
        emotion=EmotionReading(timestamp - timedelta(seconds=5), Emotion.SAD, 0.8, -0.6, 0.2, "person-1"),
        temperature=TemperatureReading(timestamp - timedelta(seconds=10), 38.0, 37.4, "good"),
        co2=Co2Reading(timestamp - timedelta(seconds=60), 1_500, Co2Level.POOR),
        reminders=(
            Reminder("due", "开窗通风", timestamp, False),
            Reminder("done", "已完成", timestamp - timedelta(minutes=1), True),
        ),
    )


def test_public_exports_complete_context_json_and_two_local_alerts() -> None:
    """完整新鲜快照产生 JSON 兼容上下文，并在等于阈值时触发两类告警。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    service = FusionService(rules())
    context = service.build_context(complete_snapshot())
    assert fusion.__all__ == ["FusionRules", "FusionService"]
    assert context["snapshot_timestamp"] == BASE_TIME.isoformat()
    assert context["disclaimer"] == "传感器观察结果不构成医学诊断。"
    assert context["emotion"] == {
        "timestamp": (BASE_TIME - timedelta(seconds=5)).isoformat(), "age_seconds": 5.0,
        "fresh": True, "quality": "observed", "dominant": "sad", "confidence": 0.8,
        "valence": -0.6, "arousal": 0.2, "person_id": "person-1",
    }
    assert context["temperature"] is not None and context["temperature"]["fresh"] is True
    assert context["co2"] is not None and context["co2"]["age_seconds"] == 60.0
    assert context["reminders"] == [
        {"reminder_id": "due", "message": "开窗通风", "due_at": BASE_TIME.isoformat(), "acknowledged": False, "due": True},
        {"reminder_id": "done", "message": "已完成", "due_at": (BASE_TIME - timedelta(minutes=1)).isoformat(), "acknowledged": True, "due": False},
    ]
    assert json.loads(json.dumps(context, ensure_ascii=False)) == context
    assert service.local_alerts(complete_snapshot()) == ["temperature_observation_high", "co2_concentration_high"]


def test_missing_modalities_and_reminders_do_not_block_context() -> None:
    """缺失输入保留稳定字段，且没有告警或云端副作用。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    snapshot = SystemSnapshot(timestamp=BASE_TIME)
    service = FusionService(rules())
    assert service.build_context(snapshot) == {
        "snapshot_timestamp": BASE_TIME.isoformat(),
        "disclaimer": "传感器观察结果不构成医学诊断。",
        "emotion": None, "temperature": None, "co2": None, "reminders": [],
    }
    assert service.local_alerts(snapshot) == []


def test_stale_and_low_quality_temperature_and_stale_co2_never_alert() -> None:
    """陈旧或质量不合格的读数仍可观察，但不得生成本地告警。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    snapshot = SystemSnapshot(
        timestamp=BASE_TIME,
        temperature=TemperatureReading(BASE_TIME - timedelta(seconds=61), 39.0, 37.0, "good"),
        co2=Co2Reading(BASE_TIME - timedelta(seconds=61), 1_600, Co2Level.POOR),
    )
    service = FusionService(rules())
    context = service.build_context(snapshot)
    assert context["temperature"] is not None and context["temperature"]["fresh"] is False
    assert context["co2"] is not None and context["co2"]["fresh"] is False
    assert service.local_alerts(snapshot) == []

    low_quality = SystemSnapshot(
        timestamp=BASE_TIME,
        temperature=TemperatureReading(BASE_TIME, 39.0, 37.0, "estimated"),
        co2=Co2Reading(BASE_TIME, 1_499, Co2Level.ELEVATED),
    )
    assert service.local_alerts(low_quality) == []


def test_invalid_co2_is_contextualized_without_alert() -> None:
    """无效 CO2 快照明确标记 invalid，且不会被高浓度规则误报。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    snapshot = SystemSnapshot(timestamp=BASE_TIME, co2=Co2Reading(BASE_TIME, None, Co2Level.INVALID))
    context = FusionService(rules()).build_context(snapshot)
    assert context["co2"] == {
        "timestamp": BASE_TIME.isoformat(), "age_seconds": 0.0, "fresh": True,
        "quality": "invalid", "ppm": None, "level": "invalid",
    }
    assert FusionService(rules()).local_alerts(snapshot) == []


def test_timezone_offsets_and_future_readings_are_json_safe_and_fresh() -> None:
    """跨时区等价时刻的年龄为零，未来读数不会生成负年龄。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    local_time = datetime(2026, 8, 18, 20, 0, tzinfo=SHANGHAI)
    snapshot = SystemSnapshot(
        timestamp=local_time,
        emotion=EmotionReading(BASE_TIME, Emotion.HAPPY, 1.0, 1.0, 1.0),
        co2=Co2Reading(local_time + timedelta(seconds=5), 800, Co2Level.ELEVATED),
    )
    context = FusionService(rules()).build_context(snapshot)
    assert context["snapshot_timestamp"] == local_time.isoformat()
    assert context["emotion"] is not None and context["emotion"]["age_seconds"] == 0.0
    assert context["co2"] is not None and context["co2"]["age_seconds"] == 0.0
    assert json.dumps(context)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"high_temperature_celsius": True}, TypeError),
        ({"high_temperature_celsius": 100.1}, ValueError),
        ({"high_co2_ppm": 1_500.0}, TypeError),
        ({"high_co2_ppm": 100_001}, ValueError),
        ({"stale_after_seconds": float("inf")}, ValueError),
        ({"temperature_quality": "  "}, ValueError),
    ],
)
def test_rule_validation_rejects_invalid_boundary_configuration(kwargs: dict[str, object], error: type[Exception]) -> None:
    """规则构造器在服务创建前拒绝无效类型、范围及空质量标识。

    Args: kwargs: 覆盖字段。error: 预期异常类型。
    Returns: 无。
    Raises: 无。
    """
    values: dict[str, object] = {"high_temperature_celsius": 38.0, "high_co2_ppm": 1_500,
                                  "stale_after_seconds": 60.0, "temperature_quality": "good"}
    values.update(kwargs)
    with pytest.raises(error):
        FusionRules(**values)  # type: ignore[arg-type]


def test_invalid_service_and_snapshot_arguments_are_rejected() -> None:
    """服务拒绝非规则及非快照入参，避免不受控字典绕过领域校验。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    with pytest.raises(TypeError, match="rules must be FusionRules"):
        FusionService(object())  # type: ignore[arg-type]
    service = FusionService(rules())
    with pytest.raises(TypeError, match="snapshot must be SystemSnapshot"):
        service.build_context({})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="snapshot must be SystemSnapshot"):
        service.local_alerts(None)  # type: ignore[arg-type]


def test_rules_example_is_parseable_and_matches_deployment_defaults() -> None:
    """规则样例可由树莓派部署配置直接读取，并映射为完整本地规则。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    path = Path(__file__).resolve().parents[1] / "config" / "rules.example.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data == {
        "freshness": {"stale_after_seconds": 60.0},
        "alerts": {"high_temperature_celsius": 38.0, "high_co2_ppm": 1_500,
                   "required_temperature_quality": "good"},
    }
    assert FusionRules(data["alerts"]["high_temperature_celsius"], data["alerts"]["high_co2_ppm"],
                       data["freshness"]["stale_after_seconds"], data["alerts"]["required_temperature_quality"]) == rules()


def test_fusion_uses_no_network_import_or_embedded_credential() -> None:
    """融合源码仅使用本地领域模型，不导入联网库或内置可疑凭据字段。

    Args: 无。
    Returns: 无。
    Raises: 无。
    """
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "fusion" / "service.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported.isdisjoint({"requests", "http", "urllib", "socket", "websockets", "boto3"})
    assert not any(token in source.lower() for token in ("api_key", "apikey", "authorization", "bearer "))
