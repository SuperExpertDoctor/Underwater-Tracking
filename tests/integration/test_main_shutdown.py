from __future__ import annotations

import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/scenario/uuv_only_single_target.yaml"


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        try:
            probe.connect(("127.0.0.1", port))
        except OSError:
            return False
        return True


@pytest.mark.skipif(
    not (ROOT / "src/underwater_tracking/ui/node_modules").is_dir(),
    reason="frontend dependencies are not installed",
)
def test_main_exits_cleanly_after_one_signal() -> None:
    with socket.socket() as api_probe, socket.socket() as ui_probe:
        api_probe.bind(("127.0.0.1", 0))
        ui_probe.bind(("127.0.0.1", 0))
        api_port = api_probe.getsockname()[1]
        ui_port = ui_probe.getsockname()[1]

    environment = {
        **os.environ,
        "UNDERWATER_TRACKING_API_KEY": "",
    }
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--config",
            str(CONFIG_PATH),
            "--steps",
            "0",
            "--seed",
            "7",
            "--port",
            str(api_port),
            "--ui-port",
            str(ui_port),
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(f"main exited early: {stdout}\n{stderr}")
            try:
                with urlopen(f"http://127.0.0.1:{api_port}/api/health", timeout=0.5):
                    break
            except (OSError, URLError):
                time.sleep(0.05)
        else:
            raise AssertionError("API did not become ready")

        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=10.0) == 130
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and (
            _port_is_open(api_port) or _port_is_open(ui_port)
        ):
            time.sleep(0.05)
        assert not _port_is_open(api_port)
        assert not _port_is_open(ui_port)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5.0)
