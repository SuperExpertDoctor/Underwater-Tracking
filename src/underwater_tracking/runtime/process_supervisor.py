"""Bounded ownership and shutdown auditing for one live run."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import socket
from threading import Lock, Thread
import time
from typing import Any


CloseCallback = Callable[[], object]


@dataclass(slots=True)
class _ThreadEntry:
    name: str
    thread: Thread
    stop: CloseCallback | None = None


@dataclass(slots=True)
class _ProcessEntry:
    name: str
    process: Any
    stop: CloseCallback | None = None


@dataclass(slots=True)
class _PortEntry:
    name: str
    host: str
    port: int
    close: CloseCallback | None = None


@dataclass(slots=True)
class _ResourceEntry:
    name: str
    close: CloseCallback
    kind: str


class ProcessSupervisor:
    """Own bounded process resources and persist the final shutdown audit.

    The supervisor deliberately does not create resources.  The run owner
    registers every thread, child process, listening port, and file handle as
    soon as it is created.  A failed shutdown remains retryable so a provider
    timeout can be followed by a later cooperative cleanup attempt.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.run_dir / "process-shutdown.json"
        self._lock = Lock()
        self._threads: dict[str, _ThreadEntry] = {}
        self._processes: dict[str, _ProcessEntry] = {}
        self._ports: dict[str, _PortEntry] = {}
        self._resources: dict[str, _ResourceEntry] = {}
        self._closed_resources: set[str] = set()
        self._closed = False
        self._report: dict[str, object] | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    def register_thread(
        self,
        thread: Thread,
        *,
        name: str | None = None,
        stop: CloseCallback | None = None,
    ) -> None:
        """Register a thread and its cooperative stop callback."""
        self._ensure_open()
        entry_name = name or thread.name
        self._threads[entry_name] = _ThreadEntry(entry_name, thread, stop)

    def register_process(
        self,
        process: Any,
        *,
        name: str,
        stop: CloseCallback | None = None,
    ) -> None:
        """Register a Popen-compatible child process."""
        self._ensure_open()
        self._processes[name] = _ProcessEntry(name, process, stop)

    register_subprocess = register_process

    def register_port(
        self,
        port: int,
        *,
        host: str = "127.0.0.1",
        name: str | None = None,
        close: CloseCallback | None = None,
    ) -> None:
        """Register a listening port and an optional server close callback."""
        if not 0 <= port <= 65_535:
            raise ValueError("port must be between 0 and 65535")
        self._ensure_open()
        entry_name = name or f"{host}:{port}"
        self._ports[entry_name] = _PortEntry(entry_name, host, port, close)

    def register_file_handle(
        self,
        handle: Any,
        *,
        name: str,
        close: CloseCallback | None = None,
    ) -> None:
        """Register a file-like handle for final close and audit."""
        callback = close or getattr(handle, "close", None)
        if not callable(callback):
            raise TypeError(f"file handle {name!r} does not expose close()")
        self.register_resource(name, callback, kind="file")

    def register_resource(
        self,
        name: str,
        resource: object | CloseCallback,
        *,
        close: CloseCallback | None = None,
        kind: str = "resource",
    ) -> None:
        """Register an arbitrary idempotently closable run resource."""
        callback = close
        if callback is None and callable(resource):
            callback = resource
        if callback is None:
            callback = getattr(resource, "close", None)
        if not callable(callback):
            raise TypeError(f"resource {name!r} does not expose close()")
        self._ensure_open()
        self._resources[name] = _ResourceEntry(name, callback, kind)

    def unregister(self, name: str) -> None:
        """Forget a resource that was transferred to another owner."""
        self._threads.pop(name, None)
        self._processes.pop(name, None)
        self._ports.pop(name, None)
        self._resources.pop(name, None)

    def shutdown(self, *, timeout_s: float = 10.0, reason: str = "normal") -> bool:
        """Stop everything within ``timeout_s`` and always write the audit."""
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        with self._lock:
            if self._closed:
                return True
            started = time.monotonic()
            deadline = started + timeout_s
            remaining: list[str] = []

            for entry in self._threads.values():
                if entry.stop is not None:
                    self._invoke(entry.stop, f"thread:{entry.name}", remaining)
            for entry in self._processes.values():
                if entry.stop is not None:
                    self._invoke(entry.stop, f"process:{entry.name}", remaining)
            for entry in self._ports.values():
                if entry.close is not None:
                    self._invoke(entry.close, f"port:{entry.name}", remaining)

            for entry in self._threads.values():
                if entry.thread.is_alive():
                    entry.thread.join(timeout=max(0.0, deadline - time.monotonic()))
                if entry.thread.is_alive():
                    remaining.append(f"thread:{entry.name}")

            for entry in self._processes.values():
                self._stop_process(entry, deadline, remaining)

            for entry in self._resources.values():
                if entry.name in self._closed_resources:
                    continue
                try:
                    entry.close()
                except BaseException:  # noqa: BLE001 - shutdown must always be audited
                    remaining.append(f"{entry.kind}:{entry.name}")
                else:
                    self._closed_resources.add(entry.name)

            for entry in self._ports.values():
                if self._port_is_open(entry.host, entry.port):
                    remaining.append(f"port:{entry.name}")

            report = self._build_report(
                reason=reason,
                started=started,
                completed=not remaining,
                remaining=remaining,
            )
            self._write_report(report)
            if report["completed"] is True:
                self._closed = True
            return bool(report["completed"])

    close = shutdown

    def report(self) -> dict[str, object]:
        """Return the latest report without touching registered resources."""
        if self._report is None:
            return self._build_report(
                reason="not_shutdown",
                started=time.monotonic(),
                completed=False,
                remaining=self._resource_names(),
            )
        return deepcopy(self._report)

    shutdown_report = report

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("process supervisor has already shut down")

    @staticmethod
    def _invoke(callback: CloseCallback, resource_name: str, remaining: list[str]) -> None:
        try:
            callback()
        except BaseException:  # noqa: BLE001 - shutdown must always be audited
            remaining.append(resource_name)

    @staticmethod
    def _stop_process(
        entry: _ProcessEntry,
        deadline: float,
        remaining: list[str],
    ) -> None:
        process = entry.process
        try:
            running = process.poll() is None
        except BaseException:  # noqa: BLE001 - shutdown must always be audited
            running = True
        if running:
            try:
                process.terminate()
            except BaseException:  # noqa: BLE001 - shutdown must always be audited
                remaining.append(f"process:{entry.name}")
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except BaseException:  # noqa: BLE001 - shutdown must always be audited
                try:
                    process.kill()
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except BaseException:  # noqa: BLE001 - shutdown must always be audited
                    remaining.append(f"process:{entry.name}")
        try:
            if process.poll() is None:
                remaining.append(f"process:{entry.name}")
        except BaseException:  # noqa: BLE001 - shutdown must always be audited
            remaining.append(f"process:{entry.name}")

    def _resource_names(self) -> list[str]:
        return [
            *(f"thread:{name}" for name in self._threads),
            *(f"process:{name}" for name in self._processes),
            *(f"port:{name}" for name in self._ports),
            *(f"{entry.kind}:{name}" for name, entry in self._resources.items()),
        ]

    def _build_report(
        self,
        *,
        reason: str,
        started: float,
        completed: bool,
        remaining: list[str],
    ) -> dict[str, object]:
        unique_remaining = list(dict.fromkeys(remaining))
        return {
            "run_id": self.run_dir.name,
            "reason": reason,
            "completed": completed,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_s": round(max(0.0, time.monotonic() - started), 3),
            "remaining_resources": unique_remaining,
            "threads": [
                {"name": entry.name, "alive": entry.thread.is_alive()}
                for entry in self._threads.values()
            ],
            "processes": [
                {
                    "name": entry.name,
                    "pid": getattr(entry.process, "pid", None),
                    "returncode": self._process_returncode(entry.process),
                    "alive": self._process_alive(entry.process),
                }
                for entry in self._processes.values()
            ],
            "ports": [
                {
                    "name": entry.name,
                    "host": entry.host,
                    "port": entry.port,
                    "open": self._port_is_open(entry.host, entry.port),
                }
                for entry in self._ports.values()
            ],
            "file_handles": [
                {
                    "name": entry.name,
                    "closed": self._resource_closed(entry.name),
                }
                for entry in self._resources.values()
                if entry.kind == "file"
            ],
        }

    def _write_report(self, report: dict[str, object]) -> None:
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._report = deepcopy(report)

    def _resource_closed(self, name: str) -> bool:
        return name in self._closed_resources

    @staticmethod
    def _process_returncode(process: Any) -> int | None:
        try:
            value = process.poll()
        except BaseException:  # noqa: BLE001 - shutdown must always be audited
            return None
        return value if isinstance(value, int) else None

    @staticmethod
    def _process_alive(process: Any) -> bool:
        try:
            return process.poll() is None
        except BaseException:  # noqa: BLE001 - shutdown must always be audited
            return True

    @staticmethod
    def _port_is_open(host: str, port: int) -> bool:
        if port <= 0:
            return False
        try:
            with socket.create_connection((host, port), timeout=0.05):
                return True
        except OSError:
            return False


__all__ = ["ProcessSupervisor"]
