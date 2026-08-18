"""T31 真实 MLX90640 和 MH-Z19 组件工厂验收测试。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import yaml

from src.core import load_settings
from src.core.factory import ComponentFactory, ComponentFactoryError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "config" / "settings.example.yaml"


def _settings(tmp_path: Path, mutate: object) -> object:
    """写入独立 Pi 配置并返回经过真实加载的 Settings。

    Args: tmp_path: pytest 临时目录。mutate: 修改 YAML 根映射的函数。
    Returns: 已验证的不可变配置。
    Raises: 无。
    """
    data = deepcopy(yaml.safe_load(TEMPLATE.read_text(encoding="utf-8")))
    assert isinstance(data, dict) and callable(mutate)
    mutate(data)
    path = tmp_path / "project" / "config" / "settings.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_settings(path)


def _real_sensors(data: dict[str, object]) -> None:
    """启用两类真实传感器，其他设备保持禁用。

    Args: data: 可变 YAML 根映射。
    Returns: 无。
    Raises: 无。
    """
    data["runtime"]["mode"] = "pi"  # type: ignore[index]
    data["vision"]["enabled"] = False  # type: ignore[index]
    data["thermal"]["enabled"] = True  # type: ignore[index]
    data["thermal"]["driver"] = "mlx90640"  # type: ignore[index]
    data["thermal"]["connection"] = {"bus": 3, "address": "0x35", "retries": 4}  # type: ignore[index]
    data["thermal"]["calibration"] = {"emissivity": 0.91, "offset_celsius": 1.25}  # type: ignore[index]
    data["thermal"]["min_valid_celsius"] = 19.0  # type: ignore[index]
    data["thermal"]["max_valid_celsius"] = 47.0  # type: ignore[index]
    data["co2"]["enabled"] = True  # type: ignore[index]
    data["co2"]["driver"] = "mhz19"  # type: ignore[index]
    data["co2"]["connection"] = {"port": "/dev/ttyAMA3", "baud_rate": 19200, "timeout_seconds": 0.4, "retries": 5}  # type: ignore[index]
    data["co2"]["thresholds_ppm"] = {"elevated": 850, "poor": 1300}  # type: ignore[index]


def test_real_sensor_names_import_fixed_modules_and_forward_every_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实驱动由固定模块提供，所有 I2C/UART 配置精确进入构造器。

    Args: tmp_path: pytest 临时目录。monkeypatch: pytest 补丁工具。
    Returns: 无。
    Raises: 无。
    """
    from src.co2 import mhz19
    from src.thermal import mlx90640

    thermal_calls: list[dict[str, object]] = []
    co2_calls: list[tuple[object, dict[str, object]]] = []

    class ThermalDouble:
        """捕获 MLX90640 构造参数的替身。"""

        def __init__(self, **kwargs: object) -> None:
            """保存参数。

            Args: kwargs: 工厂传入的构造参数。
            Returns: 无。
            Raises: 无。
            """
            thermal_calls.append(kwargs)

    class Co2Double:
        """捕获 MH-Z19 构造参数的替身。"""

        def __init__(self, port: object, **kwargs: object) -> None:
            """保存参数。

            Args: port: UART 端口。kwargs: 工厂传入的构造参数。
            Returns: 无。
            Raises: 无。
            """
            co2_calls.append((port, kwargs))

    monkeypatch.setattr(mlx90640, "Mlx90640Sensor", ThermalDouble)
    monkeypatch.setattr(mhz19, "Mhz19Sensor", Co2Double)
    components = ComponentFactory.build(_settings(tmp_path, _real_sensors))

    assert type(components.thermal).__name__ == "ThermalDouble"
    assert type(components.co2).__name__ == "Co2Double"
    assert thermal_calls == [{"bus": 3, "address": 0x35, "emissivity": 0.91,
                              "offset_celsius": 1.25, "retries": 4,
                              "min_valid_celsius": 19.0, "max_valid_celsius": 47.0}]
    port, options = co2_calls[0]
    assert port == "/dev/ttyAMA3"
    assert options["baud_rate"] == 19200
    assert options["timeout_seconds"] == 0.4
    assert options["retries"] == 5
    assert options["thresholds"].elevated == 850
    assert options["thresholds"].poor == 1300


def test_mock_mode_never_imports_real_sensor_driver_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模拟模式强制替身，MLX90640/MH-Z19 模块都不被工厂导入。

    Args: tmp_path: pytest 临时目录。monkeypatch: pytest 补丁工具。
    Returns: 无。
    Raises: 无。
    """
    for module in ("src.thermal.mlx90640", "src.co2.mhz19"):
        monkeypatch.delitem(sys.modules, module, raising=False)

    def mocked(data: dict[str, object]) -> None:
        """把配置中的真实名留在原位但使用 mock 模式。

        Args: data: 可变 YAML 根映射。
        Returns: 无。
        Raises: 无。
        """
        data["runtime"]["mode"] = "mock"  # type: ignore[index]
        data["thermal"]["enabled"] = True  # type: ignore[index]
        data["thermal"]["driver"] = "mlx90640"  # type: ignore[index]
        data["co2"]["enabled"] = True  # type: ignore[index]
        data["co2"]["driver"] = "mhz19"  # type: ignore[index]

    components = ComponentFactory.build(_settings(tmp_path, mocked))
    assert components.thermal is not None and components.co2 is not None
    assert "src.thermal.mlx90640" not in sys.modules
    assert "src.co2.mhz19" not in sys.modules


@pytest.mark.parametrize("component", ("thermal", "co2"))
def test_real_sensor_constructor_failure_is_component_scoped_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str,
) -> None:
    """真实传感器构造异常不泄露连接细节或秘密。

    Args: tmp_path: pytest 临时目录。monkeypatch: pytest 补丁工具。component: 对外组件名。
    Returns: 无。
    Raises: 无。
    """
    secret = "sensor-secret-72931"
    def fails(*args: object, **kwargs: object) -> object:
        """模拟包含秘密的底层失败。

        Args: args: 位置参数。kwargs: 关键字参数。
        Returns: 从不返回。
        Raises: RuntimeError: 固定模拟错误。
        """
        raise RuntimeError(secret)

    if component == "thermal":
        from src.thermal import mlx90640
        monkeypatch.setattr(mlx90640, "Mlx90640Sensor", fails)
    else:
        from src.co2 import mhz19
        monkeypatch.setattr(mhz19, "Mhz19Sensor", fails)
    with pytest.raises(ComponentFactoryError) as caught:
        ComponentFactory.build(_settings(tmp_path, _real_sensors))
    assert str(caught.value) == f"{component} component could not be created"
    assert secret not in str(caught.value)
