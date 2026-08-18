"""T35 Git 发布策略验收；仅在临时 Git 仓库验证忽略规则。"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GITIGNORE = ROOT / ".gitignore"
CHECKLIST = ROOT / "docs" / "git-publish-checklist.md"


def _git_ignore_output(repository: Path, path: str) -> subprocess.CompletedProcess[str]:
    """在临时仓库中查询单个路径是否被当前策略忽略。"""
    return subprocess.run(
        ["git", "check-ignore", "-v", "--", path],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )


def test_policy_contains_required_secret_runtime_and_model_rules() -> None:
    """策略明确忽略需求原文、秘密、模型、原始数据和运行日志。"""
    content = GITIGNORE.read_text(encoding="utf-8")
    for rule in (
        "request.md",
        ".env",
        ".env.*",
        "*.key",
        "*.onnx",
        "*.tflite",
        "data/raw/",
        "logs/",
        "*.log",
    ):
        assert rule in content


def test_project_log_is_explicitly_preserved_for_tracking() -> None:
    """项目验证摘要有明确反忽略规则，防止未来宽泛规则误伤。"""
    assert "!docs/project-log.md" in GITIGNORE.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for policy behavior verification")
def test_git_check_ignore_matches_sensitive_paths_in_temporary_repository(tmp_path: Path) -> None:
    """临时仓库的 Git 行为与策略文本一致，且项目日志可被跟踪。"""
    shutil.copy2(GITIGNORE, tmp_path / ".gitignore")
    initialized = subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=False, capture_output=True, text=True)
    assert initialized.returncode == 0, initialized.stderr

    ignored = (
        "request.md",
        ".env",
        "keys/device.key",
        "data/models/emotion.onnx",
        "data/raw/camera.csv",
        "logs/runtime.log",
        "runs/train.log",
    )
    for path in ignored:
        result = _git_ignore_output(tmp_path, path)
        assert result.returncode == 0, f"{path} should be ignored: {result.stderr}"

    retained = _git_ignore_output(tmp_path, "docs/project-log.md")
    assert retained.returncode == 0, retained.stderr
    assert "!docs/project-log.md" in retained.stdout
    project_log = tmp_path / "docs" / "project-log.md"
    project_log.parent.mkdir(parents=True)
    project_log.write_text("verification summary\n", encoding="utf-8")
    staged = subprocess.run(
        ["git", "add", "--", "docs/project-log.md"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert staged.returncode == 0, staged.stderr


def test_checklist_requires_status_staging_secret_scan_size_and_diff_checks() -> None:
    """清单覆盖工作区审核、受控暂存、敏感扫描、大文件与差异复查。"""
    content = CHECKLIST.read_text(encoding="utf-8")
    for command in (
        "git status --short",
        "git add <已审阅文件>",
        "git diff --cached --name-status",
        "git diff --cached --text",
        "Get-Item -LiteralPath",
        "Where-Object Length -gt 10MB",
        "git diff --cached --check",
        "git diff --cached",
    ):
        assert command in content


def test_checklist_requires_git_ignore_verification() -> None:
    """清单通过 Git 原生命令验证敏感路径被忽略、验证日志未被忽略。"""
    content = CHECKLIST.read_text(encoding="utf-8")
    assert "git check-ignore -v request.md .env logs/example.log data/raw/example.csv" in content
    assert "git check-ignore -v docs/project-log.md" in content


def test_policy_files_do_not_contain_credentials_or_private_network_addresses() -> None:
    """发布策略和清单不携带可用凭据、私网地址或私钥块。"""
    content = "\n".join((GITIGNORE.read_text(encoding="utf-8"), CHECKLIST.read_text(encoding="utf-8")))
    forbidden = (
        r"(?<![\w.])(?:10|127)(?:\.\d{1,3}){3}(?![\w.])",
        r"(?<![\w.])(?:192\.168|172\.(?:1[6-9]|2\d|3[0-1]))(?:\.\d{1,3}){2}(?![\w.])",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"(?im)^\s*(?:password|passwd|api[_ -]?key|token|secret)\s*[:=]\s*\S+",
        r"(?i)authorization:\s*bearer\s+\S+",
    )
    assert all(re.search(pattern, content) is None for pattern in forbidden)
