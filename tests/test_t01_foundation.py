"""T01 repository foundation acceptance tests.

Inputs: requirements.txt, .gitignore, and config/settings.example.yaml.
Outputs: pytest assertions covering syntax, portability, and secret hygiene.
Dependencies: pytest, PyYAML, packaging, and Git.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path, PureWindowsPath

import yaml
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = ROOT / "requirements.txt"
GITIGNORE_PATH = ROOT / ".gitignore"
SETTINGS_PATH = ROOT / "config" / "settings.example.yaml"
REQUIRED_DOMAINS = {
    "runtime",
    "vision",
    "thermal",
    "co2",
    "audio",
    "agent",
    "schedule",
    "storage",
    "logging",
}


def _load_settings() -> dict[str, object]:
    """Load the example YAML as a mapping."""
    value = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict), "settings example must contain a YAML mapping"
    return value


def _requirements() -> list[Requirement]:
    """Parse every executable requirement line using the PEP 508 parser."""
    lines = REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
    return [
        Requirement(line.strip())
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _walk_strings(value: object, prefix: str = "") -> list[tuple[str, str]]:
    """Return dotted YAML paths and their string values recursively."""
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_walk_strings(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_strings(child, f"{prefix}[{index}]"))
    elif isinstance(value, str):
        found.append((prefix, value))
    return found


def test_normal_yaml_is_parseable_and_has_nine_domains() -> None:
    """The template exposes all nine accepted configuration domains."""
    settings = _load_settings()
    assert REQUIRED_DOMAINS <= settings.keys()
    assert settings["runtime"]["mode"] in {"mock", "development", "pi"}
    for domain in REQUIRED_DOMAINS:
        assert isinstance(settings[domain], dict), f"{domain} must be a mapping"


def test_normal_requirements_are_pep508_and_compatibility_bounded() -> None:
    """Runtime, training, and test dependencies parse and have bounds."""
    requirements = _requirements()
    names = {requirement.name.lower() for requirement in requirements}
    assert {
        "pyyaml",
        "requests",
        "numpy",
        "opencv-python-headless",
        "vosk",
        "sounddevice",
        "pyserial",
        "ultralytics",
        "pytest",
        "pytest-cov",
    } <= names
    for requirement in requirements:
        operators = {specifier.operator for specifier in requirement.specifier}
        assert operators & {">", ">=", "~=", "=="}, f"missing lower bound: {requirement}"
        assert operators & {"<", "<=", "~=", "=="}, f"missing upper bound: {requirement}"


def test_boundary_placeholder_and_empty_optional_values_are_safe() -> None:
    """Optional nulls and placeholders are allowed without embedding secrets."""
    settings = _load_settings()
    assert settings["audio"]["input"]["device"] is None
    assert settings["audio"]["output"]["device"] is None
    assert settings["co2"]["connection"]["port"] == "<serial-port>"
    agent = settings["agent"]
    assert "api_key" not in agent and "agent_id" not in agent
    assert agent["api_key_env"] == "DORM_ASSISTANT_AGENT_API_KEY"
    assert agent["agent_id_env"] == "DORM_ASSISTANT_AGENT_ID"


def test_boundary_all_configured_filesystem_paths_are_relative() -> None:
    """Portable filesystem paths must not be POSIX or Windows absolute paths."""
    path_suffixes = ("_path", "_directory")
    checked: list[str] = []
    for key, value in _walk_strings(_load_settings()):
        leaf = key.rsplit(".", 1)[-1]
        if leaf.endswith(path_suffixes) or leaf == "directory":
            checked.append(key)
            assert not Path(value).is_absolute(), f"absolute POSIX/current-OS path at {key}: {value}"
            assert not PureWindowsPath(value).is_absolute(), f"absolute Windows path at {key}: {value}"
    assert checked == [
        "vision.model_path",
        "audio.input.vosk_model_path",
        "schedule.store_path",
        "storage.directory",
        "logging.directory",
    ]


def test_boundary_ultralytics_marker_excludes_arm_only() -> None:
    """The training stack is disabled on both supported Pi ARM identifiers."""
    requirement = next(item for item in _requirements() if item.name.lower() == "ultralytics")
    assert requirement.marker is not None
    assert requirement.marker.evaluate({"platform_machine": "x86_64"})
    assert not requirement.marker.evaluate({"platform_machine": "aarch64"})
    assert not requirement.marker.evaluate({"platform_machine": "armv7l"})


def test_security_files_contain_no_real_credential_signatures() -> None:
    """Tracked foundation files must not contain inline credentials or private hosts."""
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REQUIREMENTS_PATH, GITIGNORE_PATH, SETTINGS_PATH)
    )
    forbidden = {
        "inline secret assignment": re.compile(
            r"(?im)^\s*(?:api[_-]?key|password|passwd|secret|access[_-]?token)\s*:\s*(?!null\s*$|[\"']?<)[^\s#]+"
        ),
        "private IPv4 address": re.compile(
            r"(?<!\d)(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?!\d)"
        ),
        "URL userinfo": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@", re.IGNORECASE),
        "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    }
    for label, pattern in forbidden.items():
        assert pattern.search(combined) is None, f"detected {label}"


def test_security_gitignore_blocks_secret_and_generated_artifacts(tmp_path: Path) -> None:
    """Git semantics ignore secrets, model/data bulk, logs, and caches."""
    (tmp_path / ".gitignore").write_text(
        GITIGNORE_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    subprocess.run(
        ["git", "init", "--quiet"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    ignored = [
        ".env",
        ".env.production",
        "config/settings.yaml",
        "device.pem",
        "credentials-prod.json",
        "secrets/token.txt",
        "emotion.pt",
        "emotion.onnx",
        "data/models/emotion.onnx",
        "datasets/train/image.jpg",
        "data/sessions/2026-08-18/session.json",
        "logs/service.log",
        "src/__pycache__/module.pyc",
        "runs/large-training-output.bin",
    ]
    for relative in ignored:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", relative],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"expected ignored path: {relative}"


def test_security_gitignore_keeps_safe_templates_trackable(tmp_path: Path) -> None:
    """Secret-safe examples remain committable despite broad settings patterns."""
    (tmp_path / ".gitignore").write_text(
        GITIGNORE_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    subprocess.run(
        ["git", "init", "--quiet"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    for relative in (".env.example", "config/settings.example.yaml"):
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", relative],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, f"safe template unexpectedly ignored: {relative}"


def test_security_template_defaults_to_mock_and_disables_external_io() -> None:
    """A fresh copy cannot contact hardware or external APIs unexpectedly."""
    settings = _load_settings()
    assert settings["runtime"]["mode"] == "mock"
    assert settings["vision"]["driver"] == "mock"
    for domain in ("thermal", "co2", "audio", "agent"):
        assert settings[domain]["enabled"] is False
    assert settings["agent"]["base_url"] == "https://example.invalid/api"
