from __future__ import annotations

import json
from pathlib import Path
import socket
from threading import Event, Thread

import pytest

from underwater_tracking.runtime.process_supervisor import ProcessSupervisor


class _FakeProcess:
    pid = 4321

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            raise TimeoutError("process did not stop")
        return self.returncode


def test_supervisor_closes_registered_resources_and_writes_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    supervisor = ProcessSupervisor(run_dir)
    stopped = Event()
    thread = Thread(target=stopped.wait, name="test-worker", daemon=True)
    thread.start()
    process = _FakeProcess()
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    handle = (run_dir / "owned.log").open("w", encoding="utf-8")

    supervisor.register_thread(thread, name="test-worker", stop=stopped.set)
    supervisor.register_process(process, name="test-process")
    supervisor.register_port(
        server.getsockname()[1],
        host="127.0.0.1",
        name="test-port",
        close=server.close,
    )
    supervisor.register_file_handle(handle, name="test-log")

    assert supervisor.shutdown(timeout_s=1.0, reason="test") is True

    report_path = run_dir / "process-shutdown.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["completed"] is True
    assert report["reason"] == "test"
    assert report["remaining_resources"] == []
    assert process.terminate_calls == 1
    assert not thread.is_alive()
    assert handle.closed
    assert report["threads"][0]["alive"] is False
    assert report["processes"][0]["returncode"] == 0
    assert report["ports"][0]["open"] is False


def test_supervisor_persists_timeout_report_and_can_retry(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-2"
    run_dir.mkdir()
    supervisor = ProcessSupervisor(run_dir)
    release = Event()
    thread = Thread(target=release.wait, name="blocked-worker", daemon=True)
    thread.start()
    supervisor.register_thread(thread, name="blocked-worker")

    assert supervisor.shutdown(timeout_s=0.01, reason="timeout") is False
    report = json.loads(
        (run_dir / "process-shutdown.json").read_text(encoding="utf-8")
    )
    assert report["completed"] is False
    assert "thread:blocked-worker" in report["remaining_resources"]

    release.set()
    assert supervisor.shutdown(timeout_s=1.0, reason="retry") is True
    final_report = json.loads(
        (run_dir / "process-shutdown.json").read_text(encoding="utf-8")
    )
    assert final_report["completed"] is True
    assert final_report["reason"] == "retry"


def test_supervisor_close_is_idempotent(tmp_path: Path) -> None:
    supervisor = ProcessSupervisor(tmp_path / "run-3")
    calls: list[str] = []
    supervisor.register_resource("marker", lambda: calls.append("closed"))

    assert supervisor.close() is True
    assert supervisor.close() is True
    assert calls == ["closed"]


def test_supervisor_rejects_registration_after_successful_close(tmp_path: Path) -> None:
    supervisor = ProcessSupervisor(tmp_path / "run-4")
    assert supervisor.close() is True

    with pytest.raises(RuntimeError, match="shut down"):
        supervisor.register_resource("late", lambda: None)
