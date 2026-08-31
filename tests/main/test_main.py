# tests/main/test_main.py
"""Tests for the root-level ``main.py`` one-command entry script.

The script lives outside the installed package, so it is loaded by path
rather than imported as a module.
"""

from __future__ import annotations

import importlib.util
import socket
import signal
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
        str(Path("configs/scenario/default.yaml")),
        "--steps",
        "0",
        "--seed",
        "42",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--require-real-provider",
    ]


def test_default_entrypoint_uses_explicit_uuv_only_scenario(main_script: ModuleType) -> None:
    assert main_script._DEFAULT_CONFIG == (
        _ROOT / "configs" / "scenario" / "uuv_only_single_target.yaml"
    )


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


def test_build_serve_argv_forwards_real_provider_requirement(
    main_script: ModuleType,
) -> None:
    argv = main_script.build_serve_argv(
        Path("other.yaml"),
        steps=0,
        seed=7,
        host="127.0.0.1",
        port=9000,
        require_real_provider=True,
    )

    assert "--require-real-provider" in argv


def test_build_serve_argv_can_bootstrap_before_finite_steps(
    main_script: ModuleType,
) -> None:
    argv = main_script.build_serve_argv(
        Path("other.yaml"),
        steps=120,
        seed=7,
        host="127.0.0.1",
        port=9000,
        bootstrap_planning=True,
    )

    assert "--bootstrap-planning" in argv


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


def test_resolve_runtime_ports_skips_occupied_and_duplicate_ports(
    main_script: ModuleType,
) -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied_api,
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied_ui,
    ):
        occupied_api.bind(("127.0.0.1", 0))
        occupied_ui.bind(("127.0.0.1", 0))
        api_start = occupied_api.getsockname()[1]
        ui_start = occupied_ui.getsockname()[1]

        api_port, ui_port = main_script.resolve_runtime_ports(
            host="127.0.0.1",
            api_start=api_start,
            ui_start=ui_start,
        )

    assert api_port > api_start
    assert ui_port > ui_start
    assert api_port != ui_port


def test_resolve_runtime_ports_never_returns_zero_for_ephemeral_request(
    main_script: ModuleType,
) -> None:
    api_port, ui_port = main_script.resolve_runtime_ports(
        host="127.0.0.1",
        api_start=0,
        ui_start=0,
    )

    assert api_port > 0
    assert ui_port > 0
    assert api_port != ui_port


def test_main_runs_one_formal_server_with_static_ui(
    main_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    monkeypatch.setattr(main_script, "check_frontend_dist", lambda *_args: None)

    def run_formal_server(argv: list[str]) -> int:
        observed.extend(argv)
        return 0

    monkeypatch.setattr(main_script, "run_formal_server", run_formal_server)

    result = main_script.main(
        [
            "--config",
            "scenario.yaml",
            "--steps",
            "1",
            "--seed",
            "7",
            "--port",
            "8123",
            "--ui-dist",
            "dist",
        ]
    )

    assert result == 0
    serve_argv = observed
    assert isinstance(serve_argv, list)
    assert serve_argv[serve_argv.index("--port") + 1] == "8123"
    assert serve_argv[serve_argv.index("--static-ui-dir") + 1] == "dist"
    assert "--web-ui-url" not in serve_argv
    assert "--ui-port" not in serve_argv


def test_main_maps_formal_server_keyboard_interrupt_to_exit_code(
    main_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_script, "check_frontend_dist", lambda *_args: None)

    def interrupt(_argv: list[str]) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(main_script, "run_formal_server", interrupt)

    assert main_script.main(["--steps", "1", "--ui-dist", "dist"]) == 130


def test_banner_uses_one_api_address_for_web_ui_and_api(main_script: ModuleType) -> None:
    banner = main_script.banner_lines(host="127.0.0.1", api_port=8000)
    joined = "\n".join(banner)
    assert "http://127.0.0.1:8000" in joined
    assert "Web UI" in joined


def test_stop_vite_cleans_process_group_after_parent_exits(
    main_script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FinishedProcess:
        pid = 4321

        def poll(self) -> int:
            return 0

    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        main_script.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
        raising=False,
    )
    monkeypatch.setattr(main_script.os, "name", "posix")

    main_script.stop_vite(FinishedProcess())

    assert signals == [(4321, signal.SIGTERM)]


def test_shutdown_signal_uses_same_cleanup_path_as_ctrl_c(
    main_script: ModuleType,
) -> None:
    with pytest.raises(KeyboardInterrupt):
        main_script.handle_shutdown_signal(signal.SIGTERM, None)


def test_stop_vite_uses_taskkill_tree_on_windows(
    main_script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RunningProcess:
        pid = 9876

        def wait(self, timeout: float) -> int:
            return 0

    commands: list[list[str]] = []
    monkeypatch.setattr(main_script.os, "name", "nt")
    monkeypatch.setattr(
        main_script.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )

    main_script.stop_vite(RunningProcess())

    assert commands == [["taskkill", "/PID", "9876", "/T", "/F"]]
