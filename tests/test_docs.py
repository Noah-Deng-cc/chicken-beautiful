"""T25 文档验收：验证运行说明与当前 CLI、配置及部署资产保持一致。"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
ARCHITECTURE = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
HARDWARE = (ROOT / "docs" / "hardware-checklist.md").read_text(encoding="utf-8")
DOCUMENTS = (README, ARCHITECTURE, HARDWARE)


def test_readme_documents_reproducible_cli_and_mock_workflow() -> None:
    """README 所列命令与 CLI 的三个实际子命令及样例配置相符。"""
    cli = (ROOT / "src" / "app" / "cli.py").read_text(encoding="utf-8")
    settings = (ROOT / "config" / "settings.example.yaml").read_text(encoding="utf-8")

    for command in [
        "python main.py check-config --config config/settings.example.yaml",
        "python main.py self-test --config config/settings.example.yaml",
        "python main.py run --config config/settings.example.yaml --mock",
        "python -m pytest -q",
    ]:
        assert command in README
    for subcommand in ("check-config", "self-test", "run"):
        assert f'add_parser("{subcommand}"' in cli
    assert 'mode: "mock"' in settings
    assert "默认模拟模式，不需要摄像头、声卡、GPIO 或公网" in README
    assert not re.search(r"(?:[A-Za-z]:\\\\|/home/|/Users/)", README)


def test_docs_keep_training_off_pi_and_state_zero2w_limits() -> None:
    """训练/推理边界、Zero 2 W 资源限制和可复现训练入口均被明确记录。"""
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "开发机：七类情绪 YOLO 训练与 ONNX/NCNN 导出入口" in README
    assert "树莓派：仅加载导出的模型进行低分辨率 CPU 推理" in README
    assert "训练和推理严格分离" in ARCHITECTURE
    assert "树莓派不需要 PyTorch、CUDA 或训练数据集" in ARCHITECTURE
    assert "Zero 2 W" in ARCHITECTURE
    assert "最大 320 像素输入" in ARCHITECTURE
    assert "峰值 RSS 不高于 350 MB" in ARCHITECTURE
    assert "platform_machine != \"aarch64\"" in requirements
    assert "python training/train.py --model /path/to/base.pt" in README
    assert "python training/export.py --weights /path/to/best.pt" in README


def test_hardware_checklist_matches_deployment_and_marks_real_acceptance_open() -> None:
    """硬件、校准、部署检查与尚未完成的两小时实机验收均可追溯。"""
    install_script = (ROOT / "deploy" / "install_pi.sh").read_text(encoding="utf-8")

    assert "Raspberry Pi Zero 2 W" in HARDWARE
    assert "64 位 Raspberry Pi OS Lite" in HARDWARE
    for interface in ("摄像头", "I2C", "UART", "音频"):
        assert interface in HARDWARE
    assert "MLX90640" in HARDWARE and "MH-Z19" in HARDWARE
    assert "发射率和偏移" in HARDWARE
    assert "预热/校准" in HARDWARE
    assert "sudo deploy/install_pi.sh --check-only" in HARDWARE
    for probe in ("has_camera", "has_i2c", "has_uart", "has_audio"):
        assert probe in install_script
    assert "连续运行至少两小时" in HARDWARE
    assert "尚未在树莓派上完成连续两小时、内存、温度和端到端延迟验收" in README


def test_docs_describe_mock_isolation_and_actual_component_boundaries() -> None:
    """模拟模式与真实驱动的说明对应工厂的实际强制 mock 策略。"""
    factory = (ROOT / "src" / "core" / "factory.py").read_text(encoding="utf-8")
    registry = (ROOT / "config" / "component_registry.yaml").read_text(encoding="utf-8")

    assert "runtime.mode: mock" in ARCHITECTURE
    assert "强制使用 mock" in ARCHITECTURE
    assert "不导入相机、I2C、串口、Vosk 或真实网络组件" in ARCHITECTURE
    assert 'if settings.runtime.mode == "mock":\n            return "mock"' in factory
    for driver in ("yolo", "mlx90640", "mhz19", "vosk", "system_tts", "tongji"):
        assert driver in registry
    assert "外部智能体真实 API 契约尚待" in README


def test_security_guidance_uses_placeholders_and_contains_no_real_endpoint_or_credential() -> None:
    """文档要求密钥轮换和最小权限，且不泄漏 IPv4 地址或真实凭据赋值。"""
    combined = "\n".join(DOCUMENTS)

    assert "SSH 密钥认证" in README
    assert "轮换" in README
    assert "最小权限" in README
    assert "非 root" in ARCHITECTURE
    assert "0640" in README and "0640" in HARDWARE
    assert not re.search(r"(?<![A-Za-z0-9_])(?:\d{1,3}\.){3}\d{1,3}(?![A-Za-z0-9_])", combined)
    assert not re.search(r"(?im)^(?:api[_ -]?key|password|token)\s*[:=]\s*(?!<|REPLACE_|\$)[^\s]+", combined)
    assert "request.md" not in combined
