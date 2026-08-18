"""T24 部署资产验收：仅静态检查，不在宿主机执行安装或触碰真实硬件。"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"


def _read(name: str) -> str:
    """读取受测部署资产的 UTF-8 文本。"""
    return (DEPLOY / name).read_text(encoding="utf-8")


def test_install_script_is_strict_parseable_and_has_read_only_check_mode() -> None:
    """安装脚本使用严格模式、可解析 Bash，且检查模式在任何变更前退出。"""
    script = _read("install_pi.sh")

    assert script.startswith("#!/usr/bin/env bash")
    assert "set -eu" in script
    assert "--check-only" in script
    check_exit = "if test \"$failures\" -ne 0 || test \"$CHECK_ONLY\" = true; then exit \"$failures\"; fi"
    assert check_exit in script
    assert script.index(check_exit) < script.index('if test "$(id -u)" -ne 0; then')
    if bash := shutil.which("bash"):
        completed = subprocess.run([bash, "-n", str(DEPLOY / "install_pi.sh")], check=False, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr


def test_install_script_checks_pi_platform_resources_and_hardware_interfaces() -> None:
    """Zero 2 W 前置检查覆盖架构、系统、Python、内存和全部已接入接口。"""
    script = _read("install_pi.sh")

    required = [
        'test "$(uname -m)" = aarch64',
        "has_pios",
        "has_python",
        "has_memory",
        "has_camera",
        "has_i2c",
        "has_uart",
        "has_audio",
        "python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'",
        "-ge 450",
    ]
    for marker in required:
        assert marker in script


def test_install_script_is_idempotent_uses_non_root_service_user_and_keeps_secrets_out() -> None:
    """重复安装复用用户和本机配置，并且资产没有网络地址或凭据字面量。"""
    script = _read("install_pi.sh")

    assert 'APP_USER="${APP_USER:-dormassistant}"' in script
    assert 'if ! id "$APP_USER" >/dev/null 2>&1; then useradd --system' in script
    assert "for group in video audio i2c dialout" in script
    assert 'if test ! -f "$APP_DIR/config/settings.pi.yaml";' in script
    assert 'if test ! -f "$ENV_DIR/dorm-assistant.env";' in script
    assert "chown root:\"$APP_USER\" \"$ENV_DIR/dorm-assistant.env\"" in script
    assert "chmod 0640 \"$ENV_DIR/dorm-assistant.env\"" in script
    assert not re.search(r"(?<![A-Za-z0-9_])(?:\d{1,3}\.){3}\d{1,3}(?![A-Za-z0-9_])", script)
    assert not re.search(r"(?:password|api[_-]?key)\s*=\s*['\"]?(?!\$|REPLACE_)[^\s'\"]+", script, re.IGNORECASE)


def test_systemd_service_has_constrained_runtime_contract() -> None:
    """服务以专用低权限用户启动，并固定运行目录、环境文件、内存和重启策略。"""
    service = _read("dorm-assistant.service")

    for line in [
        "User=dormassistant",
        "Group=dormassistant",
        "WorkingDirectory=/opt/dorm-assistant",
        "EnvironmentFile=/etc/dorm-assistant/dorm-assistant.env",
        "MemoryMax=400M",
        "Restart=on-failure",
        "RestartSec=5",
        "NoNewPrivileges=yes",
        "ReadWritePaths=/opt/dorm-assistant /var/lib/dorm-assistant",
    ]:
        assert line in service
    assert "User=root" not in service
    assert re.search(r"^ExecStart=/opt/dorm-assistant/\.venv/bin/python /opt/dorm-assistant/main\.py run --config /opt/dorm-assistant/config/settings\.pi\.yaml$", service, re.MULTILINE)


def test_environment_template_is_placeholder_only_and_gitignore_excludes_real_files() -> None:
    """环境模板只保留待替换标记，忽略规则防止实际环境文件、日志和秘密进入版本库。"""
    env_example = _read("dorm-assistant.env.example")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "DORM_ASSISTANT_AGENT_API_KEY=REPLACE_WITH_ROTATED_API_KEY" in env_example
    assert "DORM_ASSISTANT_AGENT_ID=REPLACE_WITH_AGENT_ID" in env_example
    assert "REPLACE_WITH" in env_example
    assert not re.search(r"(?<!REPLACE_WITH_)[A-Za-z0-9]{20,}", env_example)
    for rule in [".env", ".env.*", "*.key", "credentials*.json", "secrets/", "logs/", "*.log"]:
        assert rule in gitignore
