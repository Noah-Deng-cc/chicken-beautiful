"""T20 JSONL 存储验收：输入领域事件与文件故障，输出轮转、隐私和恢复行为验证。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path

import pytest

from src.storage import JsonlRecorder


UTC = timezone.utc


def fixed_clock() -> datetime:
    """提供稳定的带时区后备时钟。

    Args: 无。
    Returns: 固定 UTC 时间。
    Raises: 无。
    """
    return datetime(2031, 2, 3, 4, 5, 6, tzinfo=UTC)


def read_records(path: Path) -> list[object]:
    """读取并逐行解析 JSONL 文件。

    Args: path: JSONL 路径。
    Returns: 按写入顺序解析的 JSON 值。
    Raises: JSONDecodeError: 存在无效 JSON 行时抛出。
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_public_api_rotation_and_stable_unicode_json(tmp_path: Path) -> None:
    """验证公共导入、按事件日期轮转及稳定 Unicode JSON 输出。

    Args: tmp_path: pytest 隔离目录。
    Returns: 无。
    Raises: 无。
    """
    recorder = JsonlRecorder(tmp_path, clock=fixed_clock)
    event = {"message": "中文摄氏度 38.0C", "timestamp": "2030-01-02T23:59:59+08:00", "z": 1, "a": 2}

    assert recorder.write(event) is True
    target = tmp_path / "2030-01-02.jsonl"
    assert recorder.path_for(date(2030, 1, 2)) == target
    assert target.read_text(encoding="utf-8") == '{"a":2,"message":"中文摄氏度 38.0C","timestamp":"2030-01-02T23:59:59+08:00","z":1}\n'
    assert read_records(target) == [event]


def test_fixed_file_and_invalid_or_naive_timestamp_fall_back_to_clock(tmp_path: Path) -> None:
    """验证关闭轮转和不合格时间戳均稳定使用后备日期。

    Args: tmp_path: pytest 隔离目录。
    Returns: 无。
    Raises: 无。
    """
    recorder = JsonlRecorder(tmp_path, rotate_daily=False, clock=fixed_clock)
    assert recorder.write({"timestamp": "not-a-date", "value": 1}) is True
    assert recorder.write({"timestamp": "2030-01-02T03:04:05", "value": 2}) is True
    assert read_records(tmp_path / "events.jsonl") == [
        {"timestamp": "not-a-date", "value": 1},
        {"timestamp": "2030-01-02T03:04:05", "value": 2},
    ]


def test_default_privacy_filters_nested_dialogue_and_raw_media(tmp_path: Path) -> None:
    """验证默认隐私配置递归排除对话、原始音频和图像。

    Args: tmp_path: pytest 隔离目录。
    Returns: 无。
    Raises: 无。
    """
    event = {
        "timestamp": "2030-01-02T00:00:00+00:00",
        "transcript": "must-not-persist",
        "audio": "raw-audio",
        "frame": "raw-frame",
        "details": [
            {"dialogue": {"text": "private dialogue", "reply": {"text": "private reply"}}},
            {"raw_audio": [1, 2], "image_data": "pixels", "safe": "keep"},
        ],
        "measurement": {"temperature": 36.5, "unit": "C"},
    }

    recorder = JsonlRecorder(tmp_path, clock=fixed_clock)
    assert recorder.write(event) is True
    stored = read_records(tmp_path / "2030-01-02.jsonl")[0]
    assert stored == {
        "timestamp": "2030-01-02T00:00:00+00:00",
        "details": [{"dialogue": {"reply": {}}}, {"safe": "keep"}],
        "measurement": {"temperature": 36.5, "unit": "C"},
    }


def test_explicit_privacy_configuration_keeps_dialogue_and_media(tmp_path: Path) -> None:
    """验证所有持久化开关开启后不会删除允许的内容。

    Args: tmp_path: pytest 隔离目录。
    Returns: 无。
    Raises: 无。
    """
    event = {
        "timestamp": "2030-01-02T00:00:00+00:00",
        "transcript": "你好",
        "raw_audio": [1, 2],
        "raw_image": "pixels",
        "dialogue": {"text": "继续对话", "reply": {"text": "好的"}},
    }
    recorder = JsonlRecorder(tmp_path, persist_dialogue_text=True, persist_raw_audio=True,
                             persist_raw_images=True, clock=fixed_clock)

    assert recorder.write(event) is True
    assert read_records(tmp_path / "2030-01-02.jsonl") == [event]


@pytest.mark.parametrize("event", [
    {"timestamp": datetime(2030, 1, 2)},
    {"unsupported": object()},
    {"bad-key": {1: "not-string"}},
])
def test_nonserializable_and_naive_datetime_fail_without_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, event: object,
) -> None:
    """验证不可序列化输入受控失败且记录诊断日志。

    Args: tmp_path: pytest 隔离目录。caplog: 日志捕获器。event: 不合法事件。
    Returns: 无。
    Raises: 无。
    """
    recorder = JsonlRecorder(tmp_path, clock=fixed_clock)
    with caplog.at_level(logging.ERROR):
        assert recorder.write(event) is False
    assert "JSONL 记录失败" in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_clock_and_io_failures_do_not_escape_or_create_partial_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """验证无时区时钟和目录 I/O 错误不会中断采集调用方。

    Args: tmp_path: pytest 隔离目录。monkeypatch: pytest 替换工具。caplog: 日志捕获器。
    Returns: 无。
    Raises: 无。
    """
    naive = JsonlRecorder(tmp_path, clock=lambda: datetime(2031, 2, 3))
    with caplog.at_level(logging.ERROR):
        assert naive.write({"value": 1}) is False

    recorder = JsonlRecorder(tmp_path / "io", clock=fixed_clock)
    def fail_mkdir(*_: object, **__: object) -> None:
        """模拟目录创建失败。

        Args: _: 任意位置参数。__: 任意关键字参数。
        Returns: 无。
        Raises: OSError: 始终抛出。
        """
        raise OSError("injected mkdir failure")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    with caplog.at_level(logging.ERROR):
        assert recorder.write({"value": 2}) is False
    assert caplog.text.count("JSONL 记录失败") >= 2
    assert not (tmp_path / "io").exists()


def test_concurrent_writes_are_complete_valid_jsonl_and_close_is_idempotent(tmp_path: Path) -> None:
    """验证并发追加不交错、不丢失，关闭后拒绝写入。

    Args: tmp_path: pytest 隔离目录。
    Returns: 无。
    Raises: 无。
    """
    recorder = JsonlRecorder(tmp_path, clock=fixed_clock)
    identifiers = list(range(160))
    with ThreadPoolExecutor(max_workers=16) as executor:
        outcomes = list(executor.map(lambda identifier: recorder.write({"id": identifier}), identifiers))

    assert all(outcomes)
    stored = read_records(tmp_path / "2031-02-03.jsonl")
    assert len(stored) == len(identifiers)
    assert {item["id"] for item in stored if isinstance(item, dict)} == set(identifiers)
    recorder.close()
    recorder.close()
    assert recorder.closed is True
    assert recorder.write({"id": "after-close"}) is False
    assert len(read_records(tmp_path / "2031-02-03.jsonl")) == len(identifiers)


@pytest.mark.parametrize("directory", ["text", 1, None])
def test_constructor_and_path_validation(directory: object, tmp_path: Path) -> None:
    """验证路径与布尔配置的类型边界。

    Args: directory: 错误目录参数。tmp_path: pytest 隔离目录。
    Returns: 无。
    Raises: 无。
    """
    with pytest.raises(TypeError, match="directory must be a Path"):
        JsonlRecorder(directory)  # type: ignore[arg-type]
    recorder = JsonlRecorder(tmp_path)
    with pytest.raises(TypeError, match="event_date must be a date"):
        recorder.path_for("2030-01-01")  # type: ignore[arg-type]
