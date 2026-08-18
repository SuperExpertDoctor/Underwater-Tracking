# tests/main/test_main.py
"""Tests for the root-level ``main.py`` one-command entry script.

The script lives outside the installed package, so it is loaded by path
rather than imported as a module.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def main_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("main_script", _ROOT / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["main_script"] = module
    spec.loader.exec_module(module)
    return module


def test_build_serve_argv_forwards_defaults(main_script: ModuleType) -> None:
    argv = main_script.build_serve_argv(
        Path("configs/scenario/default.yaml"),
        steps=0,
        seed=42,
        host="127.0.0.1",
        port=8000,
    )
    assert argv == [
        "serve",
        "--config",
        "configs/scenario/default.yaml",
        "--steps",
        "0",
        "--seed",
        "42",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]


def test_build_serve_argv_forwards_overrides(main_script: ModuleType) -> None:
    argv = main_script.build_serve_argv(
        Path("other.yaml"),
        steps=120,
        seed=7,
        host="0.0.0.0",
        port=9000,
    )
    assert argv[0] == "serve"
    assert argv[argv.index("--config") + 1] == "other.yaml"
    assert argv[argv.index("--steps") + 1] == "120"
    assert argv[argv.index("--seed") + 1] == "7"
    assert argv[argv.index("--host") + 1] == "0.0.0.0"
    assert argv[argv.index("--port") + 1] == "9000"


def test_check_frontend_prereqs_reports_missing_npm(
    main_script: ModuleType, tmp_path: Path
) -> None:
    error = main_script.check_frontend_prereqs(tmp_path, npm_cmd=None)
    assert error is not None
    assert "npm" in error


def test_check_frontend_prereqs_reports_missing_dependencies(
    main_script: ModuleType, tmp_path: Path
) -> None:
    error = main_script.check_frontend_prereqs(tmp_path, npm_cmd="npm")
    assert error is not None
    assert "install" in error


def test_check_frontend_prereqs_ready_when_npm_and_dependencies_present(
    main_script: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "node_modules").mkdir()
    error = main_script.check_frontend_prereqs(tmp_path, npm_cmd="npm")
    assert error is None


def test_vite_command_is_arg_list_on_posix(main_script: ModuleType) -> None:
    command = main_script.vite_command(Path("ui"), "npm", windows=False)
    assert command == [
        "npm",
        "--prefix",
        "ui",
        "run",
        "dev",
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        "5173",
        "--strictPort",
    ]


def test_vite_command_is_shell_string_on_windows(main_script: ModuleType) -> None:
    command = main_script.vite_command(Path("ui"), r"C:\npm.cmd", windows=True)
    assert isinstance(command, str)
    assert '"C:\\npm.cmd" --prefix "ui" run dev -- --host "127.0.0.1" --port 5173 --strictPort' in command


def test_find_available_port_skips_occupied_port(main_script: ModuleType) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        port = occupied.getsockname()[1]
        available = main_script.find_available_port(port, "127.0.0.1")
    assert available > port


def test_banner_names_web_ui_and_api_addresses(main_script: ModuleType) -> None:
    banner = main_script.banner_lines(host="127.0.0.1", api_port=8000, vite_port=5173)
    joined = "\n".join(banner)
    assert "http://127.0.0.1:5173" in joined
    assert "http://127.0.0.1:8000" in joined
    assert "Web UI" in joined
