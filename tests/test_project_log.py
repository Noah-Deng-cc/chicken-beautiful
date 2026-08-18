"""T34 project-log audit; this test never contacts hardware or remote services."""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_LOG = Path(__file__).parents[1] / "docs" / "project-log.md"


def _content() -> str:
    return PROJECT_LOG.read_text(encoding="utf-8")


def test_project_log_records_current_full_suite_count() -> None:
    assert "`800 passed`" in _content()


def test_project_log_distinguishes_local_automation_from_pi_acceptance() -> None:
    content = _content()
    assert "不访问摄像头、声卡、I2C、UART、GPIO、树莓派或公网" in content
    assert "## 未完成的外部验收" in content
    assert "树莓派实机验收" in content
    assert "待验状态" in content


def test_project_log_lists_completed_capabilities_and_pending_work() -> None:
    content = _content()
    assert "## 已完成的软件能力" in content
    for item in ("七类情绪模型训练", "智能体真实 API 契约", "远程仓库同步"):
        assert item in content


def test_project_log_has_no_address_credentials_or_raw_personal_data() -> None:
    content = _content()
    forbidden = (
        r"(?<![\w.])(?:10|127)(?:\.\d{1,3}){3}(?![\w.])",
        r"(?<![\w.])(?:192\.168|172\.(?:1[6-9]|2\d|3[0-1]))(?:\.\d{1,3}){2}(?![\w.])",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"(?im)^\s*(?:password|passwd|api[_ -]?key|token|secret)\s*[:=]\s*\S+",
        r"(?i)(?:authorization:\s*bearer|sk-[A-Za-z0-9_-]{12,})",
    )
    assert all(re.search(pattern, content) is None for pattern in forbidden)
    assert "原始音频、图像或对话文字" in content
    assert "原始个人数据" in content


def test_project_log_includes_validation_date() -> None:
    assert re.search(r"20\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2}", _content()) is not None
