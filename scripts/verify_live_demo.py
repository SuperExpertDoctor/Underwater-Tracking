#!/usr/bin/env python3
"""Run the read-only live-demo gate against a real ``main.py`` process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from underwater_tracking.verification.live_demo import (  # noqa: E402
    LiveDemoAcceptanceResult,
    verify_live_demo,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the real live-demo acceptance gate")
    parser.add_argument("--main", type=Path, default=ROOT / "main.py")
    parser.add_argument("--scenario", type=Path, default=None)
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    parser.add_argument("--expected-duration-s", type=int, default=28_800)
    parser.add_argument("--require-real-provider", action="store_true")
    parser.add_argument("--output-report", type=Path, default=None)
    args = parser.parse_args(argv)

    api_port = _free_port()
    command = [
        sys.executable,
        str(args.main),
        "--port",
        str(api_port),
        "--verification-audit",
    ]
    if args.scenario is not None:
        command.extend(["--config", str(args.scenario)])
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=(sys.platform != "win32"),
    )
    base_url = f"http://127.0.0.1:{api_port}"
    started = time.monotonic()
    while time.monotonic() - started < min(args.timeout_s, 60.0):
        if process.poll() is not None:
            output = process.stdout.read()[-1000:] if process.stdout is not None else ""
            result = LiveDemoAcceptanceResult(
                violations=(f"main_process_exit:{process.returncode}", _redact(output)),
            )
            return _finish(result, args.output_report)
        try:
            with urlopen(base_url + "/api/health", timeout=1.0):
                break
        except (OSError, URLError):
            time.sleep(0.25)
    else:
        result = LiveDemoAcceptanceResult(violations=("api_boot_timeout",))
        _stop_process(process)
        return _finish(result, args.output_report)

    result = verify_live_demo(
        base_url=base_url,
        output_dir=ROOT / "outputs",
        require_real_provider=args.require_real_provider,
        wall_timeout_s=args.timeout_s,
        expected_duration_s=args.expected_duration_s,
    )
    shutdown_started = time.monotonic()
    _stop_process(process)
    shutdown_s = time.monotonic() - shutdown_started
    violations = list(result.violations)
    if process.poll() is None:
        violations.append("main_shutdown_timeout")
    result = result.model_copy(update={"shutdown_s": round(shutdown_s, 3), "violations": tuple(dict.fromkeys(violations))})
    return _finish(result, args.output_report)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


def _finish(result: LiveDemoAcceptanceResult, report: Path | None) -> int:
    payload = result.model_dump(mode="json")
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if not result.violations else 1


def _redact(value: str) -> str:
    return re.sub(r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]?\s*\S+", r"\1=<redacted>", value)[:1000]


if __name__ == "__main__":
    raise SystemExit(main())
