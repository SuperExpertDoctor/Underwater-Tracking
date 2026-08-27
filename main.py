# main.py
"""One-command entry point for the whole underwater-tracking algorithm.

``python main.py`` starts one agent-coupled pipeline and serves the built React
command center from the same FastAPI/WebSocket process.  The run is
interactive: it runs until Ctrl+C by default.  Vite remains a development
helper and is never started by this formal entry point.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _ROOT / "src"
_UI_DIR = _SRC_DIR / "underwater_tracking" / "ui"
_DEFAULT_CONFIG = _ROOT / "configs" / "scenario" / "uuv_only_single_target.yaml"
_DEFAULT_STEPS = 0  # run until Ctrl+C, matching the interactive serve mode
_DEFAULT_SEED = 42
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_API_PORT = 8000
_VITE_PORT = 5173


def ensure_src_on_path() -> None:
    """Make ``src`` importable without an editable install."""
    src = str(_SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)


def build_serve_argv(
    config: Path,
    steps: int,
    seed: int,
    host: str,
    port: int,
    *,
    web_ui_url: str | None = None,
    continuous: bool = False,
    verification_audit: bool = False,
    require_real_provider: bool = True,
    bootstrap_planning: bool = False,
    static_ui_dir: Path | None = None,
    output_root: Path | None = None,
) -> list[str]:
    """The ``serve`` argv forwarded to the installed CLI."""
    argv = [
        "serve",
        "--config",
        str(config),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if web_ui_url is not None:
        argv.extend(["--web-ui-url", web_ui_url])
    if continuous:
        argv.append("--continuous")
    if verification_audit:
        argv.append("--verification-audit")
    if require_real_provider:
        argv.append("--require-real-provider")
    if bootstrap_planning:
        argv.append("--bootstrap-planning")
    if static_ui_dir is not None:
        argv.extend(["--static-ui-dir", str(static_ui_dir)])
    if output_root is not None:
        argv.extend(["--output-root", str(output_root)])
    return argv


def check_frontend_dist(ui_dir: Path, static_ui_dir: Path | None = None) -> str | None:
    """Return a clear error when the formal static UI build is unavailable."""
    dist_dir = static_ui_dir or (ui_dir / "dist")
    if not (dist_dir / "index.html").is_file():
        return (
            f"built frontend is missing at {dist_dir}; run: "
            f"npm --prefix {ui_dir} run build"
        )
    return None


def run_formal_server(serve_argv: list[str]) -> int:
    """Run the one in-process CLI server used by the formal entry point."""
    ensure_src_on_path()
    from underwater_tracking.cli import main as cli_main

    return cli_main(serve_argv)


def check_frontend_prereqs(ui_dir: Path, npm_cmd: str | None) -> str | None:
    """A clear failure message, or ``None`` when the frontend can launch."""
    if npm_cmd is None:
        return "npm (Node.js) is required to serve the web UI and was not found on PATH"
    if not (ui_dir / "node_modules").is_dir():
        return f"frontend dependencies are missing; run: npm --prefix {ui_dir} install"
    return None


def vite_command(
    ui_dir: Path,
    npm_cmd: str,
    *,
    windows: bool,
    host: str = _DEFAULT_HOST,
    port: int = _VITE_PORT,
) -> list[str] | str:
    """The dev-server command; a shell string on Windows, an arg list on POSIX."""
    if windows:
        return (
            f'"{npm_cmd}" --prefix "{ui_dir}" run dev -- '
            f'--host "{host}" --port {port} --strictPort'
        )
    return [
        npm_cmd,
        "--prefix",
        str(ui_dir),
        "run",
        "dev",
        "--",
        "--host",
        host,
        "--port",
        str(port),
        "--strictPort",
    ]


def find_available_port(
    start: int,
    host: str,
    *,
    excluded: set[int] | frozenset[int] = frozenset(),
) -> int:
    """Return a bindable port, preferring ``start`` and skipping exclusions."""
    if not 0 <= start <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    probe_host = "0.0.0.0" if host in {"0.0.0.0", "::"} else host

    candidates = range(start, 65_536) if start else (0,)
    for port in candidates:
        if port in excluded:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((probe_host, port))
            except OSError:
                continue
            selected = int(probe.getsockname()[1])
            if selected not in excluded:
                return selected
    raise RuntimeError(f"no available port at or above {start}")


def resolve_runtime_ports(*, host: str, api_start: int, ui_start: int) -> tuple[int, int]:
    """Select distinct bindable ports for the API and Vite processes."""
    api_port = find_available_port(api_start, host)
    ui_port = find_available_port(ui_start, host, excluded={api_port})
    return api_port, ui_port


def spawn_vite(
    ui_dir: Path,
    npm_cmd: str,
    *,
    host: str = _DEFAULT_HOST,
    port: int = _VITE_PORT,
    api_port: int = _DEFAULT_API_PORT,
) -> subprocess.Popen[bytes]:
    """Start the Vite dev server in its own process group for clean shutdown."""
    command = vite_command(
        ui_dir,
        npm_cmd,
        windows=os.name == "nt",
        host=host,
        port=port,
    )
    vite_env = os.environ.copy()
    vite_env["UNDERWATER_TRACKING_API_PORT"] = str(api_port)
    if os.name == "nt":
        return subprocess.Popen(
            command,
            shell=True,
            env=vite_env,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    return subprocess.Popen(command, env=vite_env, start_new_session=True)


def spawn_backend(serve_argv: list[str]) -> subprocess.Popen[bytes]:
    """Start the API/simulation backend as a process owned by this entry point."""
    backend_env = os.environ.copy()
    inherited_path = backend_env.get("PYTHONPATH")
    backend_env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(_SRC_DIR), inherited_path) if part
    )
    command = [sys.executable, "-m", "underwater_tracking.cli", *serve_argv]
    if os.name == "nt":
        return subprocess.Popen(command, env=backend_env)
    return subprocess.Popen(command, env=backend_env, start_new_session=True)


def stop_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Stop an owned child tree, tolerating a child that has already exited."""
    if os.name == "nt":
        # npm is a Windows batch file. Terminating the shell that launched it
        # does not reliably terminate npm, sh, and the Vite node process.
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        return

    # ``start_new_session=True`` makes the Popen pid the process-group id.
    # Keep attempting group cleanup even when the wrapper already exited: its
    # descendants may still own the listening port.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if proc.poll() is None:
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=1.0)


def stop_vite(proc: subprocess.Popen[bytes]) -> None:
    """Stop the Vite child, killing its whole group so no orphan survives."""
    stop_process_tree(proc)


def stop_backend(proc: subprocess.Popen[bytes]) -> None:
    """Stop the API/simulation child tree before this supervisor exits."""
    stop_process_tree(proc)


def wait_for_api_ready(
    host: str,
    port: int,
    backend: subprocess.Popen[bytes],
    *,
    timeout_s: float = 30.0,
) -> bool:
    """Wait until the backend accepts connections before exposing the UI."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            if backend.poll() is not None:
                return False
            time.sleep(0.05)
    return False


def handle_shutdown_signal(signum: int, frame: object) -> None:
    """Route process shutdown signals through the same ``finally`` path."""
    del signum, frame
    raise KeyboardInterrupt


def banner_lines(host: str, api_port: int, vite_port: int | None = None) -> list[str]:
    """Print the single address shared by the web UI and API transport."""
    del vite_port
    return [
        "",
        "Underwater tracking command center:",
        f"  Web UI:  http://{host}:{api_port}",
        f"  API/WS:  http://{host}:{api_port}  (docs: http://{host}:{api_port}/docs)",
        "  (Ctrl+C to stop)",
    ]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Run the full underwater-tracking algorithm with the web command center."
        ),
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=_DEFAULT_STEPS)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="continue past the configured scenario duration",
    )
    parser.add_argument(
        "--verification-audit",
        action="store_true",
        help="enable the redacted in-process physics verification endpoint",
    )
    parser.add_argument(
        "--require-real-provider",
        action="store_true",
        default=True,
        help="refuse startup unless all three configured HTTP role providers are active",
    )
    parser.add_argument(
        "--bootstrap-planning",
        action="store_true",
        help="run the initial planning epoch before finite-step simulation",
    )
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--host", default=_DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=_DEFAULT_API_PORT)
    parser.add_argument(
        "--ui-dist",
        type=Path,
        default=_UI_DIR / "dist",
        help="directory containing the built React application",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="root directory for the single run-* output directory",
    )
    parser.add_argument("--ui-port", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_src_on_path()

    error = check_frontend_dist(_UI_DIR, args.ui_dist)
    if error is not None:
        print(f"main.py: {error}", file=sys.stderr)
        return 2
    serve_argv = build_serve_argv(
        args.config,
        args.steps,
        args.seed,
        args.host,
        args.port,
        continuous=bool(args.continuous),
        verification_audit=bool(args.verification_audit),
        require_real_provider=True,
        bootstrap_planning=bool(args.bootstrap_planning),
        static_ui_dir=args.ui_dist,
        output_root=args.output_root,
    )
    print("\n".join(banner_lines(args.host, args.port)), flush=True)
    previous_sigbreak = None
    if hasattr(signal, "SIGBREAK"):
        previous_sigbreak = signal.getsignal(signal.SIGBREAK)
        signal.signal(signal.SIGBREAK, handle_shutdown_signal)
    try:
        return run_formal_server(serve_argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 1
    except KeyboardInterrupt:
        return 130
    finally:
        if previous_sigbreak is not None:
            signal.signal(signal.SIGBREAK, previous_sigbreak)


if __name__ == "__main__":
    sys.exit(main())
