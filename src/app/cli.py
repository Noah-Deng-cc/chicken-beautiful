"""应用 CLI：输入为配置和子命令，输出为状态摘要与退出码；导入时不启动硬件或服务。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
import sys

from src.core import ConfigError, SectionSettings, Settings, load_settings
from src.core.factory import ComponentFactory, Components

from .orchestrator import Orchestrator


_NAMES = ("vision", "thermal", "co2", "speech_input", "speech_output", "agent")


def _parser() -> argparse.ArgumentParser:
    """构造顶层命令解析器。

    Args: 无。
    Returns: 配置完成的参数解析器。
    Raises: 无。
    """
    parser = argparse.ArgumentParser(description="宿舍情绪推荐助手")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"),
                        help="YAML 配置路径，默认 config/settings.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-config", help="只校验并加载配置，不接触硬件")
    check.add_argument("--config", type=Path, default=argparse.SUPPRESS)
    self_test = commands.add_parser("self-test", help="构造并逐项检查已启用组件")
    self_test.add_argument("--config", type=Path, default=argparse.SUPPRESS)
    run = commands.add_parser("run", help="启动常驻服务")
    run.add_argument("--config", type=Path, default=argparse.SUPPRESS)
    run.add_argument("--mock", action="store_true", help="明确以模拟模式启动")
    return parser


def _settings(path: Path) -> Settings | None:
    """加载配置并把面向用户的错误写入标准错误。

    Args: path: 配置文件路径。
    Returns: 已加载配置；失败时为 None。
    Raises: 无。
    """
    try:
        return load_settings(path)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"配置无效或不可读取: {exc}", file=sys.stderr)
        return None


def _mock_settings(settings: Settings) -> Settings:
    """生成只替换运行模式的不可变模拟配置。

    Args: settings: 已校验原始配置。
    Returns: runtime.mode 为 mock 的新配置。
    Raises: 无。
    """
    return replace(settings, runtime=SectionSettings(settings.runtime.values | {"mode": "mock"}))


def _probe(name: str, component: object) -> bool:
    """对单个已创建组件执行最小且有界的健康探测。

    Args: name: 固定组件名。component: 工厂创建的组件。
    Returns: 探测成功时为 True。
    Raises: 无；组件故障转为状态行。
    """
    if component is None:
        print(f"{name}: skipped")
        return True
    try:
        starter = getattr(component, "start", None)
        if callable(starter):
            starter()
        reader = getattr(component, "read", None)
        if callable(reader) and reader() is None:
            raise RuntimeError("没有可用读数")
    except Exception:
        print(f"{name}: failed", file=sys.stderr)
        return False
    print(f"{name}: ok")
    return True


def _close(components: Components) -> None:
    """尽力释放自检创建的全部组件。

    Args: components: 待关闭组件集合。
    Returns: 无。
    Raises: 无；关闭故障不覆盖自检结果。
    """
    for name in _NAMES:
        item = getattr(components, name)
        closer = getattr(item, "close", None)
        try:
            if callable(closer):
                closer()
        except Exception:
            print(f"{name}: close failed", file=sys.stderr)


def _self_test(settings: Settings) -> int:
    """装配组件、逐项汇总状态并释放资源。

    Args: settings: 已校验配置。
    Returns: 所有已启用组件成功或跳过时为 0，否则为 1。
    Raises: 无。
    """
    try:
        components = ComponentFactory.build(settings)
    except Exception:
        print("组件装配失败", file=sys.stderr)
        return 1
    try:
        results = tuple(_probe(name, getattr(components, name)) for name in _NAMES)
        return 0 if all(results) else 1
    finally:
        _close(components)


def main(argv: Sequence[str] | None = None) -> int:
    """执行配置检查、硬件自检或模拟运行命令。

    Args: argv: 可选命令行参数；None 时由 argparse 读取进程参数。
    Returns: 成功为 0，自检或运行失败为 1，配置错误为 2。
    Raises: SystemExit: 参数语法或子命令无效。
    """
    args = _parser().parse_args(argv)
    settings = _settings(args.config)
    if settings is None:
        return 2
    if args.command == "check-config":
        print(f"配置有效: {settings.source_path}")
        return 0
    if args.command == "self-test":
        return _self_test(settings)
    try:
        runtime_settings = _mock_settings(settings) if args.mock else settings
        return Orchestrator(runtime_settings).run()
    except Exception:
        print("模拟服务启动失败", file=sys.stderr)
        return 1
