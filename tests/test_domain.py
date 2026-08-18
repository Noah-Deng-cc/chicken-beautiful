"""T02 domain contract acceptance tests.

Inputs: public domain models and JSON serialization helpers.
Outputs: assertions for validation, immutability, and stable serialization.
Dependencies: pytest and the Python standard library.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import json

import pytest

import src.domain as domain
from src.domain import (
    AgentReply,
    Co2Level,
    Co2Reading,
    DialogueTurn,
    Emotion,
    EmotionReading,
    Reminder,
    SystemSnapshot,
    TemperatureReading,
    to_json,
)


UTC_TIME = datetime(2026, 8, 18, 9, 30, 15, tzinfo=timezone.utc)
OFFSET_TIME = datetime(2026, 8, 18, 17, 30, 15, tzinfo=timezone(timedelta(hours=8)))
LONG_TEXT = "界" * 10_000


def make_reply(**overrides: object) -> AgentReply:
    """Build a valid agent reply with optional field overrides."""
    values: dict[str, object] = {"text": "请记得通风", "timestamp": UTC_TIME}
    values.update(overrides)
    return AgentReply(**values)  # type: ignore[arg-type]


def make_emotion(**overrides: object) -> EmotionReading:
    """Build a valid emotion reading with optional field overrides."""
    values: dict[str, object] = {
        "timestamp": UTC_TIME,
        "dominant": Emotion.HAPPY,
        "confidence": 0.8,
        "valence": 0.6,
        "arousal": 0.2,
        "person_id": "resident-1",
    }
    values.update(overrides)
    return EmotionReading(**values)  # type: ignore[arg-type]


def make_temperature(**overrides: object) -> TemperatureReading:
    """Build a valid temperature reading with optional field overrides."""
    values: dict[str, object] = {
        "timestamp": UTC_TIME,
        "maximum_celsius": 37.2,
        "average_celsius": 36.5,
        "quality": "good",
    }
    values.update(overrides)
    return TemperatureReading(**values)  # type: ignore[arg-type]


def make_co2(**overrides: object) -> Co2Reading:
    """Build a valid CO2 reading with optional field overrides."""
    values: dict[str, object] = {
        "timestamp": UTC_TIME,
        "ppm": 800,
        "level": Co2Level.GOOD,
    }
    values.update(overrides)
    return Co2Reading(**values)  # type: ignore[arg-type]


def make_reminder(**overrides: object) -> Reminder:
    """Build a valid reminder with optional field overrides."""
    values: dict[str, object] = {
        "reminder_id": "reminder-1",
        "message": "开窗通风",
        "due_at": OFFSET_TIME,
        "acknowledged": False,
    }
    values.update(overrides)
    return Reminder(**values)  # type: ignore[arg-type]


def make_dialogue(**overrides: object) -> DialogueTurn:
    """Build a valid dialogue turn with optional field overrides."""
    values: dict[str, object] = {
        "timestamp": UTC_TIME,
        "user_text": "现在空气怎么样？",
        "reply": make_reply(),
    }
    values.update(overrides)
    return DialogueTurn(**values)  # type: ignore[arg-type]


def test_public_api_exports_complete_contract() -> None:
    """The package entry point exports every documented T02 symbol."""
    expected = {
        "AgentReply",
        "Co2Level",
        "Co2Reading",
        "DialogueTurn",
        "Emotion",
        "EmotionReading",
        "Reminder",
        "Serializable",
        "SystemSnapshot",
        "TemperatureReading",
        "to_json",
    }
    assert set(domain.__all__) == expected
    assert all(hasattr(domain, name) for name in expected)


def test_emotion_enum_has_exactly_seven_stable_labels() -> None:
    """Emotion labels and their training order stay fixed."""
    assert [item.value for item in Emotion] == [
        "angry",
        "disgusted",
        "fearful",
        "happy",
        "neutral",
        "sad",
        "surprised",
    ]
    assert len(Emotion) == 7


def test_unknown_emotion_and_co2_level_are_rejected() -> None:
    """Enum constructors reject unknown wire values explicitly."""
    with pytest.raises(ValueError, match="not a valid Emotion"):
        Emotion("excited")
    with pytest.raises(ValueError, match="not a valid Co2Level"):
        Co2Level("dangerous")


def test_emotion_normal_to_dict_is_json_compatible() -> None:
    """A normal inference reading exposes the documented dictionary."""
    reading = make_emotion()
    assert reading.to_dict() == {
        "timestamp": "2026-08-18T09:30:15+00:00",
        "dominant": "happy",
        "confidence": 0.8,
        "valence": 0.6,
        "arousal": 0.2,
        "person_id": "resident-1",
    }
    json.dumps(reading.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence", 0.0),
        ("confidence", 1.0),
        ("valence", -1.0),
        ("valence", 1.0),
        ("arousal", -1.0),
        ("arousal", 1.0),
    ],
)
def test_emotion_numeric_boundaries_are_inclusive(field: str, value: float) -> None:
    """All documented emotion endpoints are accepted."""
    assert getattr(make_emotion(**{field: value}), field) == value


def test_emotion_optional_and_long_person_ids_are_preserved() -> None:
    """A missing identity and a long opaque identity are valid boundaries."""
    assert make_emotion(person_id=None).person_id is None
    assert make_emotion(person_id=LONG_TEXT).person_id == LONG_TEXT


@pytest.mark.parametrize("timestamp", [datetime(2026, 8, 18), "2026-08-18T00:00:00Z", None])
def test_emotion_rejects_invalid_timestamps(timestamp: object) -> None:
    """Emotion timestamps must be timezone-aware datetime objects."""
    error = ValueError if isinstance(timestamp, datetime) else TypeError
    with pytest.raises(error):
        make_emotion(timestamp=timestamp)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("confidence", -0.001, ValueError),
        ("confidence", 1.001, ValueError),
        ("confidence", float("nan"), ValueError),
        ("confidence", float("inf"), ValueError),
        ("confidence", True, TypeError),
        ("confidence", "0.8", TypeError),
        ("valence", -1.001, ValueError),
        ("valence", float("-inf"), ValueError),
        ("arousal", 1.001, ValueError),
        ("arousal", float("nan"), ValueError),
    ],
)
def test_emotion_rejects_invalid_numeric_values(
    field: str, value: object, error: type[Exception]
) -> None:
    """Emotion scores reject wrong types, infinities, NaN, and overflow."""
    with pytest.raises(error):
        make_emotion(**{field: value})


@pytest.mark.parametrize("dominant", ["happy", "unknown", None, 1])
def test_emotion_rejects_non_enum_dominant(dominant: object) -> None:
    """Callers must use the stable Emotion enum, including for known text."""
    with pytest.raises(ValueError, match="dominant must be an Emotion"):
        make_emotion(dominant=dominant)


@pytest.mark.parametrize("person_id", ["", "   ", 123, False])
def test_emotion_rejects_invalid_person_id(person_id: object) -> None:
    """A present identity must be a nonblank string."""
    error = TypeError if not isinstance(person_id, str) else ValueError
    with pytest.raises(error):
        make_emotion(person_id=person_id)


def test_temperature_normal_and_offset_timestamp() -> None:
    """A normal thermal reading preserves its measurement and timezone."""
    reading = make_temperature(timestamp=OFFSET_TIME)
    assert reading.maximum_celsius == 37.2
    assert reading.average_celsius == 36.5
    assert reading.timestamp.utcoffset() == timedelta(hours=8)


@pytest.mark.parametrize(
    ("maximum", "average"),
    [(-40.0, -40.0), (300.0, -40.0), (300.0, 300.0)],
)
def test_temperature_boundaries_are_inclusive(maximum: float, average: float) -> None:
    """Thermal endpoints and equal average/maximum are accepted."""
    reading = make_temperature(maximum_celsius=maximum, average_celsius=average)
    assert (reading.maximum_celsius, reading.average_celsius) == (maximum, average)


def test_temperature_long_quality_is_preserved() -> None:
    """The contract does not truncate an opaque quality marker."""
    assert make_temperature(quality=LONG_TEXT).quality == LONG_TEXT


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("maximum_celsius", -40.001, ValueError),
        ("maximum_celsius", 300.001, ValueError),
        ("maximum_celsius", float("nan"), ValueError),
        ("maximum_celsius", float("inf"), ValueError),
        ("maximum_celsius", True, TypeError),
        ("average_celsius", -40.001, ValueError),
        ("average_celsius", float("-inf"), ValueError),
        ("average_celsius", "36.5", TypeError),
    ],
)
def test_temperature_rejects_invalid_numeric_values(
    field: str, value: object, error: type[Exception]
) -> None:
    """Temperature rejects invalid range, type, NaN, and infinity."""
    with pytest.raises(error):
        make_temperature(**{field: value})


def test_temperature_rejects_average_above_maximum() -> None:
    """The aggregate thermal values remain internally consistent."""
    with pytest.raises(ValueError, match="average_celsius must not exceed"):
        make_temperature(maximum_celsius=36.0, average_celsius=36.1)


@pytest.mark.parametrize("quality", ["", "  ", None, 1])
def test_temperature_rejects_invalid_quality(quality: object) -> None:
    """A quality marker must be nonblank text."""
    error = TypeError if not isinstance(quality, str) else ValueError
    with pytest.raises(error):
        make_temperature(quality=quality)


@pytest.mark.parametrize("timestamp", [datetime(2026, 8, 18), 0])
def test_temperature_rejects_invalid_timestamps(timestamp: object) -> None:
    """Temperature timestamps must be aware datetime values."""
    error = ValueError if isinstance(timestamp, datetime) else TypeError
    with pytest.raises(error):
        make_temperature(timestamp=timestamp)


@pytest.mark.parametrize(
    ("ppm", "level"),
    [(450, Co2Level.GOOD), (1_200, Co2Level.ELEVATED), (3_000, Co2Level.POOR)],
)
def test_co2_normal_valid_levels(ppm: int, level: Co2Level) -> None:
    """Every non-invalid CO2 level accepts a valid measurement."""
    reading = make_co2(ppm=ppm, level=level)
    assert (reading.ppm, reading.level) == (ppm, level)


def test_co2_missing_measurement_requires_invalid_level() -> None:
    """A disconnected sensor is represented by the sole consistent pair."""
    reading = make_co2(ppm=None, level=Co2Level.INVALID)
    assert reading.ppm is None and reading.level is Co2Level.INVALID


@pytest.mark.parametrize("ppm", [0, 100_000])
def test_co2_ppm_boundaries_are_inclusive(ppm: int) -> None:
    """The physical contract endpoints are valid integers."""
    assert make_co2(ppm=ppm).ppm == ppm


@pytest.mark.parametrize(
    ("ppm", "level"),
    [(None, Co2Level.GOOD), (None, Co2Level.POOR), (800, Co2Level.INVALID)],
)
def test_co2_rejects_inconsistent_ppm_and_level(
    ppm: int | None, level: Co2Level
) -> None:
    """Missing and valid ppm values cannot be paired with conflicting levels."""
    with pytest.raises(ValueError):
        make_co2(ppm=ppm, level=level)


@pytest.mark.parametrize(
    ("ppm", "error"),
    [
        (-1, ValueError),
        (100_001, ValueError),
        (800.0, TypeError),
        (float("nan"), TypeError),
        (float("inf"), TypeError),
        (True, TypeError),
        ("800", TypeError),
    ],
)
def test_co2_rejects_invalid_ppm(ppm: object, error: type[Exception]) -> None:
    """PPM rejects wrong types, nonfinite floats, and range overflow."""
    with pytest.raises(error):
        make_co2(ppm=ppm)


@pytest.mark.parametrize("level", ["good", "invalid", None, 1])
def test_co2_rejects_non_enum_level(level: object) -> None:
    """Callers must use Co2Level rather than unvalidated wire strings."""
    with pytest.raises(ValueError, match="level must be a Co2Level"):
        make_co2(level=level)


@pytest.mark.parametrize("timestamp", [datetime(2026, 8, 18), []])
def test_co2_rejects_invalid_timestamps(timestamp: object) -> None:
    """CO2 timestamps must be aware datetime values."""
    error = ValueError if isinstance(timestamp, datetime) else TypeError
    with pytest.raises(error):
        make_co2(timestamp=timestamp)


def test_reminder_normal_boundary_and_long_text() -> None:
    """Reminder IDs/messages preserve Unicode, long text, and true acknowledgment."""
    reminder = make_reminder(reminder_id=LONG_TEXT, message="服药提醒：" + LONG_TEXT, acknowledged=True)
    assert reminder.reminder_id == LONG_TEXT
    assert reminder.message.endswith(LONG_TEXT)
    assert reminder.acknowledged is True


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("reminder_id", "", ValueError),
        ("reminder_id", "   ", ValueError),
        ("reminder_id", None, TypeError),
        ("message", "", ValueError),
        ("message", 12, TypeError),
        ("acknowledged", 1, TypeError),
        ("acknowledged", "false", TypeError),
    ],
)
def test_reminder_rejects_invalid_fields(
    field: str, value: object, error: type[Exception]
) -> None:
    """Reminder required text and boolean fields are strict."""
    with pytest.raises(error):
        make_reminder(**{field: value})


@pytest.mark.parametrize("due_at", [datetime(2026, 8, 18), None, "tomorrow"])
def test_reminder_rejects_invalid_due_time(due_at: object) -> None:
    """Reminder due times must be timezone-aware datetime values."""
    error = ValueError if isinstance(due_at, datetime) else TypeError
    with pytest.raises(error):
        make_reminder(due_at=due_at)


def test_agent_reply_normal_optional_and_long_values() -> None:
    """Agent replies preserve long Unicode and optional conversation IDs."""
    assert make_reply(text=LONG_TEXT, conversation_id=None).text == LONG_TEXT
    assert make_reply(conversation_id="会话-1").conversation_id == "会话-1"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("text", "", ValueError),
        ("text", "  ", ValueError),
        ("text", None, TypeError),
        ("conversation_id", "", ValueError),
        ("conversation_id", 1, TypeError),
    ],
)
def test_agent_reply_rejects_invalid_text_fields(
    field: str, value: object, error: type[Exception]
) -> None:
    """Reply text and present conversation IDs must be nonblank strings."""
    with pytest.raises(error):
        make_reply(**{field: value})


@pytest.mark.parametrize("timestamp", [datetime(2026, 8, 18), None])
def test_agent_reply_rejects_invalid_timestamps(timestamp: object) -> None:
    """Reply timestamps must be aware datetime values."""
    error = ValueError if isinstance(timestamp, datetime) else TypeError
    with pytest.raises(error):
        make_reply(timestamp=timestamp)


def test_dialogue_normal_and_long_user_text() -> None:
    """A normal dialogue preserves its nested reply and long user input."""
    dialogue = make_dialogue(user_text=LONG_TEXT)
    assert dialogue.user_text == LONG_TEXT
    assert dialogue.reply.text == "请记得通风"


@pytest.mark.parametrize("user_text", ["", " ", None, 3])
def test_dialogue_rejects_invalid_user_text(user_text: object) -> None:
    """Dialogue user text must be a nonblank string."""
    error = TypeError if not isinstance(user_text, str) else ValueError
    with pytest.raises(error):
        make_dialogue(user_text=user_text)


@pytest.mark.parametrize("reply", [None, "reply", {"text": "reply"}])
def test_dialogue_rejects_invalid_reply(reply: object) -> None:
    """Dialogue nesting only accepts validated AgentReply instances."""
    with pytest.raises(TypeError, match="reply must be an AgentReply"):
        make_dialogue(reply=reply)


@pytest.mark.parametrize("timestamp", [datetime(2026, 8, 18), "now"])
def test_dialogue_rejects_invalid_timestamps(timestamp: object) -> None:
    """Dialogue timestamps must be aware datetime values."""
    error = ValueError if isinstance(timestamp, datetime) else TypeError
    with pytest.raises(error):
        make_dialogue(timestamp=timestamp)


def test_system_snapshot_normal_nested_to_dict() -> None:
    """A full snapshot recursively becomes JSON-native structured data."""
    snapshot = SystemSnapshot(
        timestamp=OFFSET_TIME,
        emotion=make_emotion(),
        temperature=make_temperature(),
        co2=make_co2(ppm=None, level=Co2Level.INVALID),
        reminders=(make_reminder(),),
        dialogue=make_dialogue(),
    )
    actual = snapshot.to_dict()
    assert actual == {
        "timestamp": "2026-08-18T17:30:15+08:00",
        "emotion": {
            "timestamp": "2026-08-18T09:30:15+00:00",
            "dominant": "happy",
            "confidence": 0.8,
            "valence": 0.6,
            "arousal": 0.2,
            "person_id": "resident-1",
        },
        "temperature": {
            "timestamp": "2026-08-18T09:30:15+00:00",
            "maximum_celsius": 37.2,
            "average_celsius": 36.5,
            "quality": "good",
        },
        "co2": {
            "timestamp": "2026-08-18T09:30:15+00:00",
            "ppm": None,
            "level": "invalid",
        },
        "reminders": [
            {
                "reminder_id": "reminder-1",
                "message": "开窗通风",
                "due_at": "2026-08-18T17:30:15+08:00",
                "acknowledged": False,
            }
        ],
        "dialogue": {
            "timestamp": "2026-08-18T09:30:15+00:00",
            "user_text": "现在空气怎么样？",
            "reply": {
                "text": "请记得通风",
                "timestamp": "2026-08-18T09:30:15+00:00",
                "conversation_id": None,
            },
        },
    }
    assert json.loads(json.dumps(actual, ensure_ascii=False, allow_nan=False)) == actual


def test_system_snapshot_empty_boundary_serializes_nulls_and_empty_list() -> None:
    """An empty initial snapshot retains a complete predictable schema."""
    assert SystemSnapshot(timestamp=UTC_TIME).to_dict() == {
        "timestamp": "2026-08-18T09:30:15+00:00",
        "emotion": None,
        "temperature": None,
        "co2": None,
        "reminders": [],
        "dialogue": None,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("emotion", object()),
        ("temperature", {}),
        ("co2", 800),
        ("dialogue", make_reply()),
        ("reminders", []),
        ("reminders", (make_reminder(), "bad")),
    ],
)
def test_system_snapshot_rejects_invalid_members(field: str, value: object) -> None:
    """Snapshot members cannot bypass their validated domain types."""
    with pytest.raises(TypeError):
        SystemSnapshot(timestamp=UTC_TIME, **{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("timestamp", [datetime(2026, 8, 18), None])
def test_system_snapshot_rejects_invalid_timestamps(timestamp: object) -> None:
    """Snapshot timestamps must be aware datetime values."""
    error = ValueError if isinstance(timestamp, datetime) else TypeError
    with pytest.raises(error):
        SystemSnapshot(timestamp=timestamp)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "instance",
    [
        make_emotion(),
        make_temperature(),
        make_co2(),
        make_reminder(),
        make_reply(),
        make_dialogue(),
        SystemSnapshot(timestamp=UTC_TIME),
    ],
    ids=lambda item: type(item).__name__,
)
def test_all_models_are_frozen(instance: object) -> None:
    """All seven shared record types reject post-construction mutation."""
    with pytest.raises(FrozenInstanceError):
        setattr(instance, next(iter(instance.__dataclass_fields__)), "changed")


def test_to_json_has_stable_sorted_keys_and_preserves_unicode() -> None:
    """JSON output is compact, deterministic, sorted, and Unicode friendly."""
    value = {"z": 1, "中文": "情绪", "a": {"y": 2, "x": 1}}
    expected = '{"a":{"x":1,"y":2},"z":1,"中文":"情绪"}'
    assert to_json(value) == expected
    assert to_json(value) == to_json(value)
    assert "\\u" not in to_json(value)


def test_to_json_serializes_full_snapshot_and_enum_values() -> None:
    """Domain records, tuples, dates, and enums serialize recursively."""
    snapshot = SystemSnapshot(
        timestamp=UTC_TIME,
        emotion=make_emotion(dominant=Emotion.SURPRISED),
        reminders=(make_reminder(),),
    )
    encoded = to_json(snapshot)
    decoded = json.loads(encoded)
    assert decoded == snapshot.to_dict()
    assert decoded["emotion"]["dominant"] == "surprised"
    assert to_json(Co2Level.POOR) == '"poor"'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_to_json_rejects_nonfinite_numbers(value: float) -> None:
    """JSON never emits nonstandard NaN or Infinity tokens."""
    with pytest.raises(ValueError):
        to_json({"measurement": value})


@pytest.mark.parametrize(
    "value",
    [object(), {1, 2}, b"bytes", {1: "non-string key"}],
    ids=["object", "set", "bytes", "non-string-key"],
)
def test_to_json_rejects_unsupported_types(value: object) -> None:
    """Unsupported objects and non-string mapping keys fail explicitly."""
    with pytest.raises(TypeError):
        to_json(value)


def test_to_json_rejects_naive_datetime_outside_models() -> None:
    """The serializer independently enforces timezone-aware datetimes."""
    with pytest.raises(TypeError, match="timezone-aware"):
        to_json(datetime(2026, 8, 18, 9, 30, 15))
