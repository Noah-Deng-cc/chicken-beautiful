"""T23 CLI 验收：输入为命令行参数，输出为可测试的退出码和无硬件副作用。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import main as program
from src.app import cli
from src.core import ConfigError, SectionSettings, Settings, load_settings
from src.core.factory import Components


ROOT = Path(__file__).resolve().parents[1]


class Probe:
    """记录自检顺序、失败和关闭调用的最小组件替身。"""

    def __init__(self, name: str, calls: list[str], *, reading: object = object(),
                 start_error: bool = False, close_error: bool = False) -> None:
        """保存受控生命周期行为。"""
        self.name, self.calls = name, calls
        self.reading, self.start_error, self.close_error = reading, start_error, close_error

    def start(self) -> None:
        """记录启动，按需产生硬件故障。"""
        self.calls.append(f"start:{self.name}")
        if self.start_error:
            raise OSError(f"{self.name} unavailable")

    def read(self) -> object:
        """记录读取并返回预设读数。"""
        self.calls.append(f"read:{self.name}")
        return self.reading

    def close(self) -> None:
        """记录关闭，按需产生关闭故障。"""
        self.calls.append(f"close:{self.name}")
        if self.close_error:
            raise OSError(f"{self.name} close unavailable")


def _settings() -> Settings:
    """加载无需实体硬件的公开配置样例。"""
    return load_settings(ROOT / "config" / "settings.example.yaml")


def _components(*items: object | None) -> Components:
    """按 CLI 固定组件字段顺序组装替身。"""
    return Components(*items)  # type: ignore[arg-type]


def test_main_is_import_safe_and_forwards_explicit_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """程序入口导入不启动服务，main 仅转发传入参数。"""
    received: list[object] = []
    monkeypatch.setattr(program, "_main", lambda argv: received.append(argv) or 17)

    assert program.main(("check-config",)) == 17
    assert received == [("check-config",)]


def test_help_is_available_without_loading_config_or_hardware(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """帮助文本在配置加载和组件装配之前返回成功。"""
    monkeypatch.setattr(cli, "load_settings", lambda _: pytest.fail("help loaded configuration"))
    monkeypatch.setattr(cli.ComponentFactory, "build", lambda _: pytest.fail("help built components"))

    with pytest.raises(SystemExit) as completed:
        cli.main(["--help"])
    assert completed.value.code == 0
    output = capsys.readouterr().out
    assert "check-config" in output and "self-test" in output and "run" in output


@pytest.mark.parametrize(
    "argv, expected_path",
    [
        (["--config", "before.yaml", "check-config"], Path("before.yaml")),
        (["check-config", "--config", "after.yaml"], Path("after.yaml")),
    ],
)
def test_check_config_accepts_config_before_or_after_command_without_building_hardware(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str], expected_path: Path,
) -> None:
    """两种惯用参数位置都只加载配置，不进入组件工厂。"""
    loaded: list[Path] = []
    settings = _settings()
    monkeypatch.setattr(cli, "load_settings", lambda path: loaded.append(path) or settings)
    monkeypatch.setattr(cli.ComponentFactory, "build", lambda _: pytest.fail("check-config built components"))

    assert cli.main(argv) == 0
    assert loaded == [expected_path]
    assert "配置有效:" in capsys.readouterr().out


def test_invalid_or_missing_config_returns_two_without_component_construction(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    """配置加载错误返回 2，工厂和硬件均不应接触。"""
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(cli, "load_settings", lambda _: (_ for _ in ()).throw(ConfigError("missing runtime")))
    monkeypatch.setattr(cli.ComponentFactory, "build", lambda _: pytest.fail("invalid config built components"))

    assert cli.main(["check-config", "--config", str(missing)]) == 2
    assert "配置无效或不可读取: missing runtime" in capsys.readouterr().err


def test_run_invalid_config_returns_two_without_constructing_orchestrator(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """run 在配置无效时返回 2，且不创建编排器。"""
    monkeypatch.setattr(cli, "load_settings", lambda _: (_ for _ in ()).throw(ConfigError("missing runtime")))
    monkeypatch.setattr(cli, "Orchestrator", lambda _: pytest.fail("invalid config created orchestrator"))

    assert cli.main(["run"]) == 2
    assert "配置无效或不可读取: missing runtime" in capsys.readouterr().err


def test_self_test_probes_in_fixed_order_reports_skips_and_closes_every_component(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """自检按注册顺序调用 start/read，None 为 skipped，最后均关闭。"""
    calls: list[str] = []
    vision, co2, input_device, output_device, agent = (
        Probe("vision", calls), Probe("co2", calls), Probe("speech_input", calls),
        Probe("speech_output", calls), Probe("agent", calls),
    )
    components = _components(vision, None, co2, input_device, output_device, agent)
    monkeypatch.setattr(cli, "load_settings", lambda _: _settings())
    monkeypatch.setattr(cli.ComponentFactory, "build", lambda _: components)

    assert cli.main(["self-test"]) == 0
    assert calls == [
        "start:vision", "read:vision", "start:co2", "read:co2", "start:speech_input",
        "read:speech_input", "start:speech_output", "read:speech_output", "start:agent", "read:agent",
        "close:vision", "close:co2", "close:speech_input", "close:speech_output", "close:agent",
    ]
    assert capsys.readouterr().out.splitlines() == [
        "vision: ok", "thermal: skipped", "co2: ok", "speech_input: ok", "speech_output: ok", "agent: ok",
    ]


def test_self_test_failure_and_close_failure_return_nonzero_but_release_everything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """读数为空或启动异常判失败，关闭异常不得阻止后续释放。"""
    calls: list[str] = []
    components = _components(
        Probe("vision", calls, reading=None), Probe("thermal", calls, start_error=True), None,
        Probe("speech_input", calls, close_error=True), None, Probe("agent", calls),
    )
    monkeypatch.setattr(cli, "load_settings", lambda _: _settings())
    monkeypatch.setattr(cli.ComponentFactory, "build", lambda _: components)

    assert cli.main(["self-test"]) == 1
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "co2: skipped", "speech_input: ok", "speech_output: skipped", "agent: ok",
    ]
    assert captured.err.splitlines() == ["vision: failed", "thermal: failed", "speech_input: close failed"]
    assert calls[-4:] == ["close:vision", "close:thermal", "close:speech_input", "close:agent"]


def test_self_test_component_factory_failure_returns_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """组件总装配失败有稳定的非零退出码与脱敏诊断。"""
    monkeypatch.setattr(cli, "load_settings", lambda _: _settings())
    monkeypatch.setattr(cli.ComponentFactory, "build", lambda _: (_ for _ in ()).throw(RuntimeError("secret")))

    assert cli.main(["self-test"]) == 1
    assert capsys.readouterr().err == "组件装配失败\n"


def test_run_preserves_configured_runtime_mode_and_mock_overrides_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认 run 保留配置模式，--mock 仅覆盖运行时模式。"""
    settings = replace(_settings(), runtime=SectionSettings({"mode": "pi"}))
    created: list[Settings] = []

    class FakeOrchestrator:
        """避免测试真正进入常驻事件循环。"""

        def __init__(self, current: Settings) -> None:
            """保存实际运行配置。"""
            created.append(current)

        def run(self) -> int:
            """返回可验证退出码。"""
            return 9

    monkeypatch.setattr(cli, "load_settings", lambda _: settings)
    monkeypatch.setattr(cli, "Orchestrator", FakeOrchestrator)

    assert cli.main(["run"]) == 9
    assert len(created) == 1 and created[0] is settings
    assert cli.main(["run", "--mock"]) == 9
    assert len(created) == 2 and created[1].runtime.mode == "mock"
    assert created[1].runtime.values == {"mode": "mock"}


def test_invalid_command_and_unreadable_path_have_cli_error_contract(
    capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    """语法无效由 argparse 终止；不可读取 YAML 返回可脚本处理的 2。"""
    with pytest.raises(SystemExit) as invalid:
        cli.main(["unsupported"])
    assert invalid.value.code == 2
    assert "invalid choice" in capsys.readouterr().err

    assert cli.main(["--config", str(tmp_path / "absent.yaml"), "check-config"]) == 2
    assert "配置无效或不可读取:" in capsys.readouterr().err
