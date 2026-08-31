"""Offline ownership and polling tests for the live acceptance driver."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from tools import run_default_live_acceptance as driver


def _fixture_server(tmp_path: Path) -> Path:
    script = tmp_path / "fixture_server.py"
    script.write_text(
        """
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import signal
import sys
from urllib.parse import parse_qs, urlparse

parser = argparse.ArgumentParser()
parser.add_argument('--host', required=True)
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--unhealthy', action='store_true')
args = parser.parse_args()

frame = {
    'frame_id': 1,
    'sim_time_s': 1,
    'plan_version': 1,
    'events': [{'event_id': 'fixture-event', 'event_type': 'fixture_ready', 'sim_time_s': 1}],
}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/health':
            status = 503 if args.unhealthy else 200
            self.respond(status, {'status': 'starting' if args.unhealthy else 'ok', 'plan_version': 1})
            return
        if parsed.path == '/api/operational/snapshot':
            self.respond(200, frame)
            return
        if parsed.path == '/api/replay':
            query = parse_qs(parsed.query)
            limit = int(query.get('limit', ['250'])[0])
            assert limit <= 250
            self.respond(200, {'frames': [frame], 'total_count': 1, 'offset': 0, 'limit': limit})
            return
        self.respond(404, {'detail': 'not found'})

    def respond(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

server = HTTPServer((args.host, args.port), Handler)
signal.signal(signal.SIGINT, lambda *_args: sys.exit(130))
server.serve_forever()
""",
        encoding="utf-8",
    )
    return script


def _checkpoints() -> tuple[driver.AcceptanceCheckpoint, ...]:
    return (
        driver.AcceptanceCheckpoint(
            "health_ready",
            None,
            lambda state: state.get("health_http_status") == 200,
            3.0,
        ),
        driver.AcceptanceCheckpoint(
            "fixture_ready",
            "fixture_ready",
            lambda state: True,
            3.0,
        ),
    )


def test_driver_owns_process_paginates_replay_and_injects_browser_url(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(driver, "_GLOBAL_DEADLINE_S", 10.0)
    script = _fixture_server(tmp_path)
    output = tmp_path / "acceptance.json"
    playwright = (
        sys.executable,
        "-c",
        "import os, sys; sys.exit(0 if os.environ.get('PLAYWRIGHT_BASE_URL', '').startswith('http://127.0.0.1:') else 7)",
    )

    result = driver.run_acceptance(
        command=(sys.executable, str(script)),
        api_port=0,
        output_path=output,
        checkpoints=_checkpoints(),
        playwright_command=playwright,
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["shutdown"]["sigint_count"] == 1
    assert report["process"]["returncode"] in {130, 3221225786}
    assert report["playwright"]["returncode"] == 0
    assert report["playwright"]["contained_in_owned_process"] is True
    assert set(report["ports"]) == {"api"}
    assert report["playwright"]["base_url"].endswith(str(report["ports"]["api"]))
    assert all(
        item["name"] in {"health_ready", "fixture_ready"}
        for item in report["checkpoints"]
    )


def test_driver_reports_timeout_and_cleans_owned_group(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(driver, "_GLOBAL_DEADLINE_S", 5.0)
    script = _fixture_server(tmp_path)
    output = tmp_path / "timeout.json"
    checkpoints = (
        driver.AcceptanceCheckpoint(
            "health_ready",
            None,
            lambda state: state.get("health_http_status") == 200,
            0.5,
        ),
    )

    result = driver.run_acceptance(
        command=(sys.executable, str(script), "--unhealthy"),
        api_port=0,
        output_path=output,
        checkpoints=checkpoints,
    )

    assert result == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "health_ready" in report["failure"]
    assert report["shutdown"]["sigint_count"] == 1
    assert report["process"]["returncode"] in {130, 3221225786}
