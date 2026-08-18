"""程序命令入口：输入为命令行参数，输出为明确的进程退出码；导入时不启动服务。"""

from __future__ import annotations

from collections.abc import Sequence

from src.app.cli import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    """转发命令行参数至应用 CLI。

    Args: argv: 可选命令行参数；None 时由 argparse 读取进程参数。
    Returns: CLI 的进程退出码。
    Raises: SystemExit: 命令格式无效时由 argparse 产生。
    """
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
