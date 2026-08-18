"""日程存储验收：输入 Reminder 与文件故障，输出持久化、恢复及并发契约验证；依赖 pytest 和标准库。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from src.domain.models import Reminder
from src.schedule import ReminderStore
import src.schedule.store as schedule_store


UTC = timezone.utc


def reminder(identifier: str, offset_minutes: int = 0, *, acknowledged: bool = False) -> Reminder:
    """构造稳定的带时区测试提醒。

    Args: identifier: 提醒标识。offset_minutes: 相对基准到期分钟。acknowledged: 是否已确认。
    Returns: 有效 Reminder。
    Raises: 无。
    """
    return Reminder(identifier, f"message {identifier}", datetime(2030, 1, 1, tzinfo=UTC) + timedelta(minutes=offset_minutes), acknowledged)


def test_add_list_replaces_duplicate_and_persists(tmp_path: Path) -> None:
    """验证新增、列举、重复标识替换及重开持久化。

    Args: tmp_path: pytest 临时目录。
    Returns: 无。
    Raises: 无。
    """
    path = tmp_path / "reminders.json"
    store = ReminderStore(path)
    first = reminder("same", 1)
    replacement = Reminder("same", "replacement", first.due_at + timedelta(minutes=3))
    store.add(first)
    store.add(reminder("other", 2))
    store.add(replacement)

    assert store.list() == [replacement, reminder("other", 2)]
    assert ReminderStore(path).list() == [replacement, reminder("other", 2)]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1 and len(data["reminders"]) == 2


def test_due_timezone_acknowledge_remove_and_delete_alias(tmp_path: Path) -> None:
    """验证跨时区到期比较、确认和两个删除入口。

    Args: tmp_path: pytest 临时目录。
    Returns: 无。
    Raises: 无。
    """
    store = ReminderStore(tmp_path / "reminders.json")
    store.add(reminder("late", 90))
    store.add(reminder("early", -10))
    store.add(reminder("done", -20, acknowledged=True))
    utc_now = datetime(2030, 1, 1, 0, 0, tzinfo=UTC)
    plus_eight_now = utc_now.astimezone(timezone(timedelta(hours=8)))

    assert [item.reminder_id for item in store.due(plus_eight_now)] == ["early"]
    store.acknowledge("early")
    assert store.due(utc_now) == []
    store.acknowledge("early")
    assert store.remove("late") is True
    assert store.remove("missing") is False
    assert store.delete("done") is True
    assert store.delete("done") is False
    assert store.list() == [Reminder("early", "message early", reminder("early", -10).due_at, True)]


@pytest.mark.parametrize("now", [datetime(2030, 1, 1), "2030-01-01", None])
def test_due_rejects_naive_or_non_datetime(tmp_path: Path, now: object) -> None:
    """验证到期查询拒绝无时区和错误类型。

    Args: tmp_path: pytest 临时目录。now: 非法当前时间。
    Returns: 无。
    Raises: 无。
    """
    store = ReminderStore(tmp_path / "reminders.json")
    expected = ValueError if isinstance(now, datetime) else TypeError
    with pytest.raises(expected):
        store.due(now)  # type: ignore[arg-type]


def test_write_is_same_directory_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证提交经由同目录临时文件原子替换并清理临时文件。

    Args: tmp_path: pytest 临时目录。monkeypatch: pytest 替换工具。
    Returns: 无。
    Raises: 无。
    """
    path = tmp_path / "nested" / "reminders.json"
    calls: list[tuple[Path, Path]] = []
    real_replace = schedule_store.os.replace

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        """记录替换参数后调用真实原子替换。

        Args: source: 源临时文件。destination: 最终文件。
        Returns: 无。
        Raises: OSError: 原始替换失败。
        """
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(schedule_store.os, "replace", recording_replace)
    ReminderStore(path).add(reminder("atomic"))
    assert len(calls) == 1
    source, destination = calls[0]
    assert source.parent == path.parent and source.name.startswith(".reminders.json.") and source.suffix == ".tmp"
    assert destination == path.resolve() and not source.exists()
    assert ReminderStore(path).list() == [reminder("atomic")]


def test_concurrent_adds_are_unique_and_reopenable(tmp_path: Path) -> None:
    """验证多线程同时新增不会遗漏或覆盖不同标识。

    Args: tmp_path: pytest 临时目录。
    Returns: 无。
    Raises: 无。
    """
    store = ReminderStore(tmp_path / "reminders.json")
    identifiers = [f"id-{number:03d}" for number in range(120)]
    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(lambda identifier: store.add(reminder(identifier)), identifiers))
    stored = store.list()
    reopened = ReminderStore(store.path).list()
    assert len(stored) == len(identifiers)
    assert {item.reminder_id for item in stored} == set(identifiers)
    assert reopened == stored


def test_corrupt_file_is_backed_up_and_recovered_empty(tmp_path: Path) -> None:
    """验证损坏 JSON 被备份且内存恢复为空集合。

    Args: tmp_path: pytest 临时目录。
    Returns: 无。
    Raises: 无。
    """
    path = tmp_path / "reminders.json"
    damaged = "{not valid json"
    path.write_text(damaged, encoding="utf-8")
    store = ReminderStore(path)
    backups = list(tmp_path.glob("reminders.json.corrupt-*.bak"))
    assert store.list() == [] and not path.exists()
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == damaged
    store.add(reminder("after-recovery"))
    assert ReminderStore(path).list() == [reminder("after-recovery")]


@pytest.mark.parametrize("operation", ["add", "acknowledge", "remove"])
def test_write_failure_does_not_mutate_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    """验证原子写失败不会提交候选内存状态。

    Args: tmp_path: pytest 临时目录。monkeypatch: pytest 替换工具。operation: 被测写操作。
    Returns: 无。
    Raises: 无。
    """
    store = ReminderStore(tmp_path / "reminders.json")
    existing = reminder("existing", -1)
    store.add(existing)
    original = store.list()

    def failing_write(_: object) -> None:
        """模拟不可恢复的存储写入失败。

        Args: _: 待写映射。
        Returns: 无。
        Raises: OSError: 始终抛出。
        """
        raise OSError("injected write failure")

    monkeypatch.setattr(store, "_write", failing_write)
    with pytest.raises(OSError, match="injected write failure"):
        if operation == "add":
            store.add(reminder("new"))
        elif operation == "acknowledge":
            store.acknowledge("existing")
        else:
            store.remove("existing")
    assert store.list() == original
    assert ReminderStore(store.path).list() == original
