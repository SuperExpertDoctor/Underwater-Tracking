# main.py
"""One-command entry point for the whole underwater-tracking algorithm.

``python main.py`` starts the complete agent-coupled pipeline (delegated to
``underwater_tracking.cli serve``: background simulation thread, carrier
runtime, FastAPI/WebSocket transport), launches the Vite dev server that
serves the React command center, and prints both addresses.  The backend
run is interactive: it runs until Ctrl+C by default.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _ROOT / "src"
_UI_DIR = _SRC_DIR / "underwater_tracking" / "ui"
_DEFAULT_CONFIG = _ROOT / "configs" / "scenario" / "default.yaml"
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
    config: Path, steps: int, seed: int, host: str, port: int
) -> list[str]:
    """The ``serve`` argv forwarded to the installed CLI."""
    return [
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


def check_frontend_prereqs(ui_dir: Path, npm_cmd: str | None) -> str | None:
    """A clear failure message, or ``None`` when the frontend can launch."""
    if npm_cmd is None:
        return "npm (Node.js) is required to serve the web UI and was not found on PATH"
    if not (ui_dir / "node_modules").is_dir():
        return f"frontend dependencies are missing; run: npm --prefix {ui_dir} install"
    return None


def vite_command(ui_dir: Path, npm_cmd: str, *, windows: bool) -> list[str] | str:
    """The dev-server command; a shell string on Windows, an arg list on POSIX."""
    if windows:
        return f'"{npm_cmd}" --prefix "{ui_dir}" run dev'
    return [npm_cmd, "--prefix", str(ui_dir), "run", "dev"]


def spawn_vite(ui_dir: Path, npm_cmd: str) -> subprocess.Popen[bytes]:
    """Start the Vite dev server in its own process group for clean shutdown."""
    command = vite_command(ui_dir, npm_cmd, windows=os.name == "nt")
    if os.name == "nt":
        return subprocess.Popen(
            command,
            shell=True,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    return subprocess.Popen(command, start_new_session=True)


def stop_vite(proc: subprocess.Popen[bytes]) -> None:
    """Stop the Vite child, killing its whole group so no orphan survives."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        proc.terminate()
    else:
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def banner_lines(host: str, api_port: int, vite_port: int) -> list[str]:
    """The printed addresses: the web UI first, then the API transport."""
    return [
        "",
        "Underwater tracking command center:",
        f"  Web UI:  http://{host}:{vite_port}",
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
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--host", default=_DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=_DEFAULT_API_PORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_src_on_path()

    # Imported only after ``src`` is on ``sys.path`` so an editable install
    # is not required to run the entry point.
    from underwater_tracking import cli

    npm_cmd = shutil.which("npm")
    error = check_frontend_prereqs(_UI_DIR, npm_cmd)
    if error is not None:
        print(f"main.py: {error}", file=sys.stderr)
        return 2
    assert npm_cmd is not None  # narrowed by check_frontend_prereqs

    vite = spawn_vite(_UI_DIR, npm_cmd)
    print("\n".join(banner_lines(args.host, args.port, _VITE_PORT)), flush=True)
    try:
        return cli.main(
            build_serve_argv(args.config, args.steps, args.seed, args.host, args.port)
        )
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 1
    except KeyboardInterrupt:
        return 130
    finally:
        stop_vite(vite)


if __name__ == "__main__":
    sys.exit(main())
