"""Run the owned-process semantic acceptance for the default entry point.

The driver deliberately speaks only to the public HTTP surface.  It owns the
``main.py`` process group, walks replay with a bounded cursor, and writes a
machine-readable report without prompts, provider responses, or credentials.
The expensive path is opt-in; the module is also imported by offline driver
tests with a small fixture server.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from typing import cast

import httpx


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_HTTP_TIMEOUT_S = 2.0
_HEALTH_INTERVAL_S = 1.0
_MIN_HEALTH_SAMPLES = 60
_GLOBAL_DEADLINE_S = 1_800.0
_PROCESS_SHUTDOWN_TIMEOUT_S = 10.0
_REPLAY_PAGE_SIZE = 250
_MAX_REPLAY_PAGES_PER_POLL = 64
_MAX_REPLAY_FRAMES = 50_000
_POLL_INTERVAL_S = 0.5


Predicate = Callable[[dict[str, object]], bool]
QueryScalar = str | int | float | bool | None
QueryParams = Mapping[str, QueryScalar | Sequence[QueryScalar]]


@dataclass(frozen=True, slots=True)
class AcceptanceCheckpoint:
    """One ordered semantic condition in the acceptance run."""

    name: str
    event_type: str | None
    predicate: Predicate
    timeout_s: float


class _AcceptanceFailure(RuntimeError):
    """A bounded acceptance failure with a user-facing diagnostic."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utc_now().isoformat()


def _as_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("HTTP response was not a JSON object")
    return cast(dict[str, object], value)


def _request_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    params: QueryParams | None = None,
    payload: Mapping[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    response = client.request(
        method,
        path,
        params=params,
        json=payload,
        timeout=_HTTP_TIMEOUT_S,
    )
    try:
        body = _as_object(response.json())
    except (ValueError, RuntimeError):
        body = {}
    return response.status_code, body


def _allocate_port(requested: int) -> int:
    if not 0 <= requested <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", requested))
        return int(probe.getsockname()[1])


def _spawn_owned_process(command: tuple[str, ...]) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    if os.name == "nt":
        return subprocess.Popen(
            list(command),
            cwd=str(_REPOSITORY_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    return subprocess.Popen(
        list(command),
        cwd=str(_REPOSITORY_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _owned_process_group_is_valid(process: subprocess.Popen[bytes]) -> bool:
    if process.poll() is not None:
        return False
    if os.name == "nt":
        return True
    try:
        return os.getpgid(process.pid) == process.pid
    except ProcessLookupError:
        return False


def _send_sigint_once(
    process: subprocess.Popen[bytes],
    shutdown: dict[str, object],
) -> None:
    if process.poll() is not None or bool(shutdown.get("sigint_sent")):
        return
    if os.name == "nt":
        control_break = cast(int, getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT))
        process.send_signal(control_break)
    else:
        if not _owned_process_group_is_valid(process):
            raise _AcceptanceFailure("refused SIGINT because the owned process group was invalid")
        os.killpg(process.pid, signal.SIGINT)
    shutdown["sigint_sent"] = True
    shutdown["sigint_sent_at"] = _timestamp()
    previous_count = shutdown.get("sigint_count", 0)
    shutdown["sigint_count"] = (int(previous_count) if isinstance(previous_count, int) else 0) + 1


def _terminate_validated_group(
    process: subprocess.Popen[bytes],
    shutdown: dict[str, object],
) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.kill()
        return
    if not _owned_process_group_is_valid(process):
        shutdown["forced_kill_refused"] = True
        raise _AcceptanceFailure("refused forced cleanup because the process group was invalid")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2.0)
        shutdown["forced_signal"] = "SIGTERM"
        return
    except subprocess.TimeoutExpired:
        pass
    if not _owned_process_group_is_valid(process):
        shutdown["forced_kill_refused"] = True
        raise _AcceptanceFailure("refused SIGKILL because the process group was invalid")
    os.killpg(process.pid, signal.SIGKILL)
    shutdown["forced_signal"] = "SIGKILL"
    process.wait(timeout=2.0)


def _shutdown_owned_process(
    process: subprocess.Popen[bytes],
    shutdown: dict[str, object],
) -> None:
    shutdown["started_at"] = shutdown.get("started_at", _timestamp())
    try:
        _send_sigint_once(process, shutdown)
        process.wait(timeout=_PROCESS_SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        shutdown["sigint_timeout"] = True
        _terminate_validated_group(process, shutdown)
    finally:
        shutdown["completed_at"] = _timestamp()
        shutdown["returncode"] = process.poll()


class _HealthSampler:
    """One bounded health request per wall-clock second for the whole run."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.samples: list[dict[str, object]] = []
        self._thread = threading.Thread(target=self._run, name="acceptance-health", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        next_due = time.monotonic()
        with httpx.Client(base_url=self._base_url) as client:
            while not self._stop.is_set():
                started = time.perf_counter()
                status: int | None = None
                payload: dict[str, object] = {}
                error: str | None = None
                try:
                    status, payload = _request_json(client, "GET", "/api/health")
                except Exception as exc:  # noqa: BLE001 - report the bounded probe failure
                    error = type(exc).__name__
                latency_ms = (time.perf_counter() - started) * 1000.0
                sample = {
                    "wall_time_utc": _timestamp(),
                    "latency_ms": round(latency_ms, 3),
                    "http_status": status,
                    "status": payload.get("status"),
                    "plan_version": payload.get("plan_version"),
                    "error": error,
                }
                with self._lock:
                    self.samples.append(sample)
                next_due += _HEALTH_INTERVAL_S
                wait_s = max(0.0, next_due - time.monotonic())
                self._stop.wait(wait_s)

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [dict(sample) for sample in self.samples]


class _ReplayScanner:
    """Read replay using the API cursor and a bounded page size."""

    def __init__(self) -> None:
        self.offset = 0
        self.frames: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []
        self._event_keys: set[str] = set()
        self._next_event_order = 0

    def poll(self, client: httpx.Client) -> None:
        pages = 0
        while pages < _MAX_REPLAY_PAGES_PER_POLL:
            status, payload = _request_json(
                client,
                "GET",
                "/api/replay",
                params={"start_s": 0, "offset": self.offset, "limit": _REPLAY_PAGE_SIZE},
            )
            if status == 503:
                return
            if status != 200:
                raise RuntimeError(f"replay returned HTTP {status}")
            raw_frames = payload.get("frames", [])
            if not isinstance(raw_frames, list):
                raise RuntimeError("replay frames were not a JSON array")
            frames = [
                cast(dict[str, object], frame)
                for frame in raw_frames
                if isinstance(frame, dict)
            ]
            if not frames:
                return
            self.frames.extend(frames)
            if len(self.frames) > _MAX_REPLAY_FRAMES:
                del self.frames[:-_MAX_REPLAY_FRAMES]
            for frame in frames:
                raw_events = frame.get("events", [])
                if not isinstance(raw_events, list):
                    continue
                for raw_event in raw_events:
                    if not isinstance(raw_event, dict):
                        continue
                    event = cast(dict[str, object], raw_event)
                    key = str(
                        event.get(
                            "event_id",
                            f"{event.get('event_type')}:{event.get('sim_time_s')}:{self._next_event_order}",
                        )
                    )
                    if key in self._event_keys:
                        continue
                    self._event_keys.add(key)
                    event_with_order = dict(event)
                    event_with_order["_acceptance_order"] = self._next_event_order
                    self.events.append(event_with_order)
                    self._next_event_order += 1
            page_count = len(frames)
            self.offset += page_count
            pages += 1
            total_count = payload.get("total_count")
            if page_count < _REPLAY_PAGE_SIZE:
                return
            if isinstance(total_count, int) and self.offset >= total_count:
                return

    def payload(self) -> dict[str, object]:
        return {
            "frames": list(self.frames),
            "count": len(self.frames),
            "offset": self.offset,
            "page_size": _REPLAY_PAGE_SIZE,
            "events_seen": len(self.events),
        }


def _read_state(
    client: httpx.Client,
    scanner: _ReplayScanner,
    operator_state: Mapping[str, object],
) -> dict[str, object]:
    health_status, health = _request_json(client, "GET", "/api/health")
    snapshot_status, snapshot = _request_json(client, "GET", "/api/operational/snapshot")
    if snapshot_status not in {200, 503}:
        raise RuntimeError(f"operational snapshot returned HTTP {snapshot_status}")
    scanner.poll(client)
    latest_sample = operator_state.get("latest_health_latency_ms")
    state: dict[str, object] = {
        "health": health,
        "health_http_status": health_status,
        "snapshot": snapshot if snapshot_status == 200 else {},
        "replay": scanner.payload(),
        "frames": list(scanner.frames),
        "event_records": list(scanner.events),
        "replay_cursor": scanner.offset,
        "operator": dict(operator_state),
    }
    if latest_sample is not None:
        state["health_latency_ms"] = latest_sample
    return state


def _event_records(
    state: Mapping[str, object],
    event_type: str,
    after_order: int,
) -> list[dict[str, object]]:
    raw_records = state.get("event_records", [])
    if not isinstance(raw_records, list):
        return []
    return [
        cast(dict[str, object], record)
        for record in raw_records
        if isinstance(record, dict)
        and record.get("event_type") == event_type
        and int(record.get("_acceptance_order", -1)) > after_order
    ]


def _event_predicate(event_type: str) -> Predicate:
    def predicate(state: dict[str, object]) -> bool:
        return bool(_event_records(state, event_type, -1))

    return predicate


def _snapshot_plan_version(state: Mapping[str, object]) -> int:
    snapshot = state.get("snapshot")
    if not isinstance(snapshot, dict):
        return -1
    value = snapshot.get("plan_version")
    return int(value) if isinstance(value, (int, float)) else -1


def _int_value(value: object, default: int = 0) -> int:
    return int(value) if isinstance(value, (int, float)) else default


def _health_ready(state: Mapping[str, object]) -> bool:
    health = state.get("health")
    return (
        state.get("health_http_status") == 200
        and isinstance(health, dict)
        and health.get("status") in {"ok", "paused"}
    )


def _has_passive_track(state: Mapping[str, object]) -> bool:
    snapshot = state.get("snapshot")
    if not isinstance(snapshot, dict):
        return False
    groups = snapshot.get("groups", [])
    return isinstance(groups, list) and any(
        isinstance(group, dict) and group.get("mode") == "passive_track" for group in groups
    )


def _operator_finished(state: Mapping[str, object]) -> bool:
    operator = state.get("operator")
    if not isinstance(operator, dict):
        return False
    return bool(operator.get("done")) and operator.get("error") is None


def default_acceptance_checkpoints() -> tuple[AcceptanceCheckpoint, ...]:
    """Return the ordered semantic gates used by the CLI wrapper."""

    return (
        AcceptanceCheckpoint(
            "health_ready",
            None,
            _health_ready,
            30.0,
        ),
        AcceptanceCheckpoint("plan_committed", None, lambda state: _snapshot_plan_version(state) > 0, 300.0),
        AcceptanceCheckpoint(
            "carrier_dispatched",
            "carrier_dispatch_completed",
            _event_predicate("carrier_dispatch_completed"),
            180.0,
        ),
        AcceptanceCheckpoint("uuv_deployed", "uuv_deployed", _event_predicate("uuv_deployed"), 180.0),
        AcceptanceCheckpoint("active_scan", "active_ping", _event_predicate("active_ping"), 180.0),
        AcceptanceCheckpoint(
            "target_detection_acquired",
            "target_detection_acquired",
            _event_predicate("target_detection_acquired"),
            180.0,
        ),
        AcceptanceCheckpoint(
            "adversary_decision",
            "target_mission_decision",
            _event_predicate("target_mission_decision"),
            300.0,
        ),
        AcceptanceCheckpoint("passive_track", None, _has_passive_track, 180.0),
        AcceptanceCheckpoint("handoff_completed", "handoff_completed", _event_predicate("handoff_completed"), 300.0),
        AcceptanceCheckpoint("uuv_recovered", "uuv_recovered", _event_predicate("uuv_recovered"), 300.0),
        AcceptanceCheckpoint(
            "carrier_returned_to_fleet",
            "carrier_returned_to_fleet",
            _event_predicate("carrier_returned_to_fleet"),
            300.0,
        ),
        AcceptanceCheckpoint(
            "memory_source_processed",
            "memory_version_created",
            _operator_finished,
            180.0,
        ),
    )


def _health_latency(state: Mapping[str, object]) -> float | None:
    value = state.get("health_latency_ms")
    return float(value) if isinstance(value, (int, float)) else None


def _checkpoint_record(
    checkpoint: AcceptanceCheckpoint,
    state: Mapping[str, object],
    event: Mapping[str, object] | None,
) -> dict[str, object]:
    snapshot = state.get("snapshot")
    frame = cast(dict[str, object], snapshot) if isinstance(snapshot, dict) else {}
    event_ids: list[str] = []
    evidence_ids: list[str] = []
    if event is not None:
        event_id = event.get("event_id")
        if isinstance(event_id, str):
            event_ids.append(event_id)
        for field in ("evidence_ids", "source_ids", "source_event_ids", "source_decision_ids"):
            values = event.get(field)
            if isinstance(values, list):
                evidence_ids.extend(str(value) for value in values)
    return {
        "name": checkpoint.name,
        "event_type": checkpoint.event_type,
        "sim_time_s": frame.get("sim_time_s", event.get("sim_time_s") if event else None),
        "wall_time_utc": _timestamp(),
        "wall_time_s": time.time(),
        "frame_id": frame.get("frame_id"),
        "plan_version": frame.get("plan_version"),
        "event_ids": sorted(set(event_ids)),
        "evidence_ids": sorted(set(evidence_ids)),
        "health_latency_ms": _health_latency(state),
    }


def _latest_snapshot(client: httpx.Client) -> dict[str, object]:
    status, snapshot = _request_json(client, "GET", "/api/operational/snapshot")
    if status != 200:
        raise _AcceptanceFailure(f"operational snapshot returned HTTP {status}")
    return snapshot


def _current_scope(snapshot: Mapping[str, object]) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    targets_raw = snapshot.get("target_estimates", [])
    target_items = targets_raw if isinstance(targets_raw, list) else []
    target_ids = tuple(
        str(item.get("target_id"))
        for item in target_items
        if isinstance(item, dict) and item.get("target_id")
    )
    regions_raw = snapshot.get("regions", [])
    region_items = regions_raw if isinstance(regions_raw, list) else []
    region_ids = tuple(
        str(item.get("region_id"))
        for item in region_items
        if isinstance(item, dict) and item.get("region_id")
    )
    scenario_id = str(snapshot.get("scenario_id") or "uuv-only-single-target")
    return target_ids, region_ids, scenario_id


def _memory_stream_event(
    client: httpx.Client,
    *,
    conversation_id: str,
    scenario_id: str,
    after_cursor: int,
    event_type: str,
    deadline: float,
) -> tuple[dict[str, object], int]:
    cursor = after_cursor
    while time.monotonic() < deadline:
        status, payload = _request_json(
            client,
            "GET",
            "/api/assistant/memory/stream",
            params={
                "user_id": "operator",
                "conversation_id": conversation_id,
                "scenario_id": scenario_id,
                "after_cursor": cursor,
                "limit": 128,
                "include_scenario_events": "true",
            },
        )
        if status != 200:
            raise _AcceptanceFailure(f"memory stream returned HTTP {status}")
        events_raw = payload.get("events", [])
        events = [cast(dict[str, object], item) for item in events_raw if isinstance(item, dict)] if isinstance(events_raw, list) else []
        for event in events:
            if event.get("type") == event_type:
                next_cursor = payload.get("next_cursor")
                return event, int(next_cursor) if isinstance(next_cursor, int) else cursor
        next_cursor = payload.get("next_cursor")
        if isinstance(next_cursor, int) and next_cursor > cursor:
            cursor = next_cursor
        time.sleep(_POLL_INTERVAL_S)
    raise _AcceptanceFailure(f"memory event {event_type} timed out")


def _operator_checkpoint(
    records: list[dict[str, object]],
    name: str,
    event_type: str | None,
    snapshot: Mapping[str, object],
    event: Mapping[str, object] | None = None,
) -> None:
    state = {"snapshot": dict(snapshot), "health_latency_ms": None}
    records.append(_checkpoint_record(AcceptanceCheckpoint(name, event_type, lambda _: True, 0), state, event))


def _assert_safe_feedback(
    proposal: Mapping[str, object],
    target_ids: tuple[str, ...],
    region_ids: tuple[str, ...],
) -> None:
    directive_raw = proposal.get("directive")
    directive = cast(dict[str, object], directive_raw) if isinstance(directive_raw, dict) else {}
    locked_members = directive.get("locked_members", {})
    if locked_members not in ({}, None):
        raise _AcceptanceFailure("assistant preview changed locked UUV membership")
    assignment_ids = directive.get("assignment_uuv_ids", [])
    if assignment_ids not in ([], (), None):
        raise _AcceptanceFailure("assistant preview introduced an assignment")
    disabled_ids = directive.get("disabled_uuv_ids", [])
    if disabled_ids not in ([], (), None):
        raise _AcceptanceFailure("assistant preview disabled a UUV")
    feedback_regions = directive.get("feedback_region_ids", [])
    if isinstance(feedback_regions, list) and not set(map(str, feedback_regions)) <= set(region_ids):
        raise _AcceptanceFailure("assistant preview changed the task region")
    preview_targets = directive.get("target_scope", [])
    if isinstance(preview_targets, list) and not set(map(str, preview_targets)) <= set(target_ids):
        raise _AcceptanceFailure("assistant preview changed the target scope")


def _run_operator_lane(
    base_url: str,
    operator_state: dict[str, object],
    global_deadline: float,
) -> None:
    records: list[dict[str, object]] = []
    try:
        with httpx.Client(base_url=base_url) as client:
            snapshot = _latest_snapshot(client)
            target_ids, region_ids, scenario_id = _current_scope(snapshot)
            conversation_id = "acceptance-memory"
            cursor = 0
            first_plan_value = snapshot.get("plan_version", 0)
            first_plan_version = int(first_plan_value) if isinstance(first_plan_value, (int, float)) else 0
            status, first_turn = _request_json(
                client,
                "POST",
                "/api/conversation/messages",
                payload={
                    "conversation_id": conversation_id,
                    "user_id": "operator",
                    "assistant_mode": "auto",
                    "text": "请记住：目标接触后优先维持被动协同跟踪，只有丢失接触时才启用主动扫描。",
                    "expected_plan_version": first_plan_version,
                    "target_scope": target_ids,
                    "region_scope": region_ids,
                },
            )
            if status != 200:
                raise _AcceptanceFailure(f"durable preference submission returned HTTP {status}")
            first_event, cursor = _memory_stream_event(
                client,
                conversation_id=conversation_id,
                scenario_id=scenario_id,
                after_cursor=cursor,
                event_type="memory_version_created",
                deadline=min(global_deadline, time.monotonic() + 180.0),
            )
            memory_one = str(first_event.get("memory_id") or "")
            family_id = str(first_event.get("memory_family_id") or "")
            if not memory_one or not family_id:
                raise _AcceptanceFailure("memory version 1 had no family or memory id")
            _operator_checkpoint(records, "memory_version_1_created", "memory_version_created", snapshot, first_event)

            snapshot = _latest_snapshot(client)
            second_plan_value = snapshot.get("plan_version", 0)
            second_plan_version = int(second_plan_value) if isinstance(second_plan_value, (int, float)) else 0
            status, _ = _request_json(
                client,
                "POST",
                "/api/conversation/messages",
                payload={
                    "conversation_id": conversation_id,
                    "user_id": "operator",
                    "assistant_mode": "auto",
                    "text": "请改为：目标接触后优先主动扫描，只有确认接触稳定后才转被动协同跟踪。",
                    "expected_plan_version": second_plan_version,
                    "target_scope": target_ids,
                    "region_scope": region_ids,
                },
            )
            if status != 200:
                raise _AcceptanceFailure(f"conflicting preference submission returned HTTP {status}")
            second_event, cursor = _memory_stream_event(
                client,
                conversation_id=conversation_id,
                scenario_id=scenario_id,
                after_cursor=cursor,
                event_type="memory_version_created",
                deadline=min(global_deadline, time.monotonic() + 180.0),
            )
            memory_two = str(second_event.get("memory_id") or "")
            if not memory_two or memory_two == memory_one:
                raise _AcceptanceFailure("memory version 2 did not create a new memory id")
            _operator_checkpoint(records, "memory_version_2_created", "memory_version_created", snapshot, second_event)
            status, versions = _request_json(
                client,
                "GET",
                f"/api/assistant/memory/{family_id}/versions",
                params={"user_id": "operator", "scenario_id": scenario_id},
            )
            if status != 200 or not isinstance(versions.get("versions"), list):
                raise _AcceptanceFailure("memory version chain was unavailable")

            snapshot = _latest_snapshot(client)
            evidence_plan_value = snapshot.get("plan_version", 0)
            evidence_plan_version = int(evidence_plan_value) if isinstance(evidence_plan_value, (int, float)) else 0
            question_status, question = _request_json(
                client,
                "POST",
                "/api/questions",
                payload={"text": "为什么跟踪策略发生变化？"},
            )
            if question_status not in {200, 422}:
                raise _AcceptanceFailure(f"evidence question returned HTTP {question_status}")
            evidence_status, _ = _request_json(
                client,
                "POST",
                "/api/conversation/messages",
                payload={
                    "conversation_id": "acceptance-evidence",
                    "user_id": "operator",
                    "assistant_mode": "evidence_query",
                    "text": "请给出跟踪策略变化的可验证证据。",
                    "expected_plan_version": evidence_plan_version,
                    "target_scope": target_ids,
                    "region_scope": region_ids,
                },
            )
            if evidence_status != 200:
                raise _AcceptanceFailure(f"evidence conversation returned HTTP {evidence_status}")
            evidence_event, cursor = _memory_stream_event(
                client,
                conversation_id="acceptance-evidence",
                scenario_id=scenario_id,
                after_cursor=0,
                event_type="evidence_trace_completed",
                deadline=min(global_deadline, time.monotonic() + 180.0),
            )
            if question_status == 200 and not question:
                raise _AcceptanceFailure("evidence question returned an empty payload")
            _operator_checkpoint(records, "evidence_trace_completed", "evidence_trace_completed", snapshot, evidence_event)

            snapshot = _latest_snapshot(client)
            feedback_plan_value = snapshot.get("plan_version", 0)
            feedback_plan_version = int(feedback_plan_value) if isinstance(feedback_plan_value, (int, float)) else 0
            feedback_status, feedback = _request_json(
                client,
                "POST",
                "/api/conversation/messages",
                payload={
                    "conversation_id": "acceptance-feedback",
                    "user_id": "operator",
                    "assistant_mode": "plan_revision",
                    "text": "保持当前任务区域与 UUV 分配，仅重新确认交接窗口",
                    "expected_plan_version": feedback_plan_version,
                    "target_scope": target_ids,
                    "region_scope": region_ids,
                },
            )
            if feedback_status != 200:
                raise _AcceptanceFailure(f"assistant preview returned HTTP {feedback_status}")
            proposal = feedback.get("proposal")
            if not isinstance(proposal, dict):
                raise _AcceptanceFailure("assistant feedback did not return a preview")
            _assert_safe_feedback(cast(dict[str, object], proposal), target_ids, region_ids)
            turn_id = feedback.get("turn_id")
            if not isinstance(turn_id, str):
                raise _AcceptanceFailure("assistant preview did not return a turn id")
            _operator_checkpoint(records, "assistant_preview_created", "conversation_preview_created", snapshot)
            applied_status, applied = _request_json(
                client,
                "POST",
                "/api/conversation/acceptance-feedback/apply",
                payload={
                    "user_id": "operator",
                    "turn_id": turn_id,
                    "expected_plan_version": feedback_plan_version,
                },
            )
            if applied_status != 200:
                raise _AcceptanceFailure(f"assistant plan apply returned HTTP {applied_status}")
            if applied.get("applied") is not True and applied.get("proposal") is None:
                raise _AcceptanceFailure("assistant apply returned no applied result")
            apply_deadline = min(global_deadline, time.monotonic() + 300.0)
            while time.monotonic() < apply_deadline:
                snapshot = _latest_snapshot(client)
                if _int_value(snapshot.get("plan_version")) > feedback_plan_version:
                    break
                time.sleep(_POLL_INTERVAL_S)
            else:
                raise _AcceptanceFailure("assistant apply did not commit a newer plan")
            _operator_checkpoint(records, "assistant_plan_applied", "plan_committed", snapshot)

            delete_status, _ = _request_json(
                client,
                "DELETE",
                f"/api/assistant/memory/{memory_two}",
                params={
                    "user_id": "operator",
                    "scenario_id": scenario_id,
                    "conversation_id": conversation_id,
                },
            )
            if delete_status != 200:
                raise _AcceptanceFailure(f"memory deletion returned HTTP {delete_status}")
            snapshot = _latest_snapshot(client)
            semantic_raw = snapshot.get("semantic", [])
            semantic_items = semantic_raw if isinstance(semantic_raw, list) else []
            active_ids = {
                str(item.get("memory_id")) for item in semantic_items if isinstance(item, dict)
            }
            if memory_two in active_ids:
                raise _AcceptanceFailure("deleted memory version remained active")
            deleted_event, _ = _memory_stream_event(
                client,
                conversation_id=conversation_id,
                scenario_id=scenario_id,
                after_cursor=cursor,
                event_type="memory_deleted",
                deadline=min(global_deadline, time.monotonic() + 180.0),
            )
            if not deleted_event.get("source_message_ids"):
                raise _AcceptanceFailure("memory deletion did not retain source message provenance")
            _operator_checkpoint(records, "memory_version_deleted", "memory_deleted", snapshot, deleted_event)
    except Exception as exc:  # noqa: BLE001 - lane failure belongs in the artifact
        operator_state["error"] = str(exc)[:500]
    finally:
        operator_state["checkpoints"] = records
        operator_state["done"] = True


def _health_p95(samples: Sequence[Mapping[str, object]]) -> float | None:
    values: list[float] = []
    for sample in samples:
        value = sample.get("latency_ms")
        if isinstance(value, (int, float)):
            values.append(float(value))
    values.sort()
    if len(values) < _MIN_HEALTH_SAMPLES:
        return None
    index = min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)
    return round(values[index], 3)


def _output_run_dirs() -> set[Path]:
    output_root = _REPOSITORY_ROOT / "outputs"
    if not output_root.is_dir():
        return set()
    return {
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "agent.db").is_file()
    }


def _final_database_checks(before: set[Path]) -> dict[str, object]:
    candidates = sorted(_output_run_dirs() - before, key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise _AcceptanceFailure("no owned run directory with agent.db was created")
    run_dir = candidates[-1]
    database_path = run_dir / "agent.db"
    checks: dict[str, object] = {"run_dir": str(run_dir), "database": str(database_path)}
    uri = f"file:{database_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        counts: dict[str, int] = {}
        for table in (
            "llm_calls",
            "plans",
            "long_term_memories",
            "memory_stream_events",
            "short_term_messages",
        ):
            if table in tables:
                counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        operations = (
            sorted(
                str(row[0])
                for row in connection.execute("SELECT DISTINCT operation FROM llm_calls")
            )
            if "llm_calls" in tables
            else []
        )
    replay_path = run_dir / "operational_frames.jsonl"
    frame_count = 0
    contains_usv = False
    if replay_path.is_file():
        with replay_path.open(encoding="utf-8") as replay_file:
            for line in replay_file:
                frame_count += 1
                contains_usv = contains_usv or "usv" in line.lower()
    checks.update(
        {
            "table_counts": counts,
            "llm_operations": operations,
            "replay_frame_count": frame_count,
            "contains_usv": contains_usv,
        }
    )
    if contains_usv:
        raise _AcceptanceFailure("persisted operational replay contains a USV entity")
    if counts.get("plans", 0) < 2:
        raise _AcceptanceFailure("persisted run contains fewer than two plan records")
    if counts.get("memory_stream_events", 0) == 0:
        raise _AcceptanceFailure("persisted run contains no memory stream events")
    return checks


def run_acceptance(
    *,
    command: tuple[str, ...],
    api_port: int,
    ui_port: int,
    output_path: Path,
    checkpoints: tuple[AcceptanceCheckpoint, ...],
    playwright_command: tuple[str, ...] | None = None,
) -> int:
    """Run an owned process and return zero only when every gate passes."""

    output_path = output_path if output_path.is_absolute() else _REPOSITORY_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "status": "failed",
        "started_at": _timestamp(),
        "command": list(command),
        "checkpoints": [],
        "operator_checkpoints": [],
        "failure": None,
    }
    process: subprocess.Popen[bytes] | None = None
    browser: subprocess.Popen[bytes] | None = None
    sampler: _HealthSampler | None = None
    operator_thread: threading.Thread | None = None
    operator_state: dict[str, object] = {"done": False, "error": None, "checkpoints": []}
    shutdown: dict[str, object] = {"sigint_sent": False, "sigint_count": 0}
    process_started = time.monotonic()
    run_dirs_before = _output_run_dirs()
    success = False
    try:
        allocated_api_port = _allocate_port(api_port)
        allocated_ui_port = _allocate_port(ui_port)
        if allocated_api_port == allocated_ui_port:
            allocated_ui_port = _allocate_port(0)
        report["ports"] = {"api": allocated_api_port, "ui": allocated_ui_port}
        owned_command = command + (
            "--host",
            "127.0.0.1",
            "--port",
            str(allocated_api_port),
            "--ui-port",
            str(allocated_ui_port),
        )
        report["owned_command"] = list(owned_command)
        process = _spawn_owned_process(owned_command)
        process_started = time.monotonic()
        shutdown["process_started_at"] = _timestamp()
        report["process"] = {"pid": process.pid, "start_new_session": os.name != "nt"}
        base_url = f"http://127.0.0.1:{allocated_api_port}"
        ui_url = f"http://127.0.0.1:{allocated_ui_port}"
        sampler = _HealthSampler(base_url)
        sampler.start()
        scanner = _ReplayScanner()
        global_deadline = process_started + _GLOBAL_DEADLINE_S
        last_event_order = -1
        operator_enabled = any(cp.name == "memory_source_processed" for cp in checkpoints)
        with httpx.Client(base_url=base_url) as client:
            for checkpoint in checkpoints:
                if checkpoint.timeout_s <= 0:
                    raise ValueError(f"checkpoint {checkpoint.name} has a non-positive timeout")
                checkpoint_deadline = min(global_deadline, time.monotonic() + checkpoint.timeout_s)
                latest_state: dict[str, object] = {}
                selected_event: dict[str, object] | None = None
                predicate_error: str | None = None
                while time.monotonic() < checkpoint_deadline:
                    if process.poll() is not None:
                        raise _AcceptanceFailure(
                            f"owned process exited with code {process.returncode} before {checkpoint.name}"
                        )
                    try:
                        samples = sampler.snapshot()
                        if samples:
                            operator_state["latest_health_latency_ms"] = samples[-1].get("latency_ms")
                        latest_state = _read_state(client, scanner, operator_state)
                        candidates = (
                            _event_records(latest_state, checkpoint.event_type, last_event_order)
                            if checkpoint.event_type is not None
                            else []
                        )
                        selected_event = candidates[0] if candidates else None
                        try:
                            predicate_ok = checkpoint.predicate(latest_state)
                        except Exception as exc:  # noqa: BLE001 - report malformed fixture state
                            predicate_ok = False
                            predicate_error = f"{type(exc).__name__}: {exc}"
                        if predicate_ok and (checkpoint.event_type is None or selected_event is not None):
                            record = _checkpoint_record(checkpoint, latest_state, selected_event)
                            cast(list[object], report["checkpoints"]).append(record)
                            if selected_event is not None:
                                last_event_order = _int_value(
                                    selected_event.get("_acceptance_order"), last_event_order
                                )
                            if checkpoint.name == "health_ready":
                                if playwright_command is not None and browser is None:
                                    browser_env = os.environ.copy()
                                    browser_env["PLAYWRIGHT_BASE_URL"] = ui_url
                                    browser = subprocess.Popen(
                                        list(playwright_command),
                                        cwd=str(_REPOSITORY_ROOT),
                                        env=browser_env,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        start_new_session=os.name != "nt",
                                    )
                                    report["playwright"] = {
                                        "command": list(playwright_command),
                                        "base_url": ui_url,
                                        "started_at": _timestamp(),
                                        "pid": browser.pid,
                                    }
                            if checkpoint.name == "plan_committed" and operator_enabled and operator_thread is None:
                                operator_thread = threading.Thread(
                                    target=_run_operator_lane,
                                    args=(base_url, operator_state, global_deadline),
                                    name="acceptance-operator-lane",
                                    daemon=True,
                                )
                                operator_thread.start()
                            break
                    except _AcceptanceFailure:
                        raise
                    except Exception as exc:  # noqa: BLE001 - bounded retry until deadline
                        predicate_error = f"{type(exc).__name__}: {exc}"
                    time.sleep(_POLL_INTERVAL_S)
                else:
                    cursor = latest_state.get("replay_cursor")
                    raise _AcceptanceFailure(
                        f"checkpoint {checkpoint.name} timed out; replay_cursor={cursor!r}; "
                        f"plan_version={_snapshot_plan_version(latest_state)}; "
                        f"predicate_error={predicate_error}"
                    )
                if time.monotonic() >= global_deadline:
                    raise _AcceptanceFailure("global acceptance deadline exceeded")

            if operator_thread is not None:
                remaining = max(0.0, global_deadline - time.monotonic())
                operator_thread.join(timeout=remaining)
                if operator_thread.is_alive():
                    raise _AcceptanceFailure("operator and memory lane did not finish before the global deadline")
                if operator_state.get("error") is not None:
                    raise _AcceptanceFailure(str(operator_state["error"]))
            report["operator_checkpoints"] = list(cast(list[object], operator_state.get("checkpoints", [])))

            if browser is not None:
                remaining = max(0.0, global_deadline - time.monotonic())
                try:
                    browser.wait(timeout=remaining)
                except subprocess.TimeoutExpired as exc:
                    raise _AcceptanceFailure("Playwright did not finish before the global deadline") from exc
                playwright_report = cast(dict[str, object], report["playwright"])
                playwright_report["ended_at"] = _timestamp()
                playwright_report["returncode"] = browser.returncode
                if browser.returncode != 0:
                    raise _AcceptanceFailure(f"Playwright exited with code {browser.returncode}")
                process_end = time.monotonic()
                if process_end < process_started:
                    raise _AcceptanceFailure("invalid process interval recorded for Playwright coupling")
                playwright_report["contained_in_owned_process"] = True

            if sampler is not None:
                samples = sampler.snapshot()
                if operator_enabled and len(samples) < _MIN_HEALTH_SAMPLES:
                    raise _AcceptanceFailure(
                        f"health sample count {len(samples)} is below {_MIN_HEALTH_SAMPLES}"
                    )
        success = True
    except Exception as exc:  # noqa: BLE001 - report and clean up all owned resources
        report["failure"] = str(exc)[:1000]
    finally:
        if sampler is not None:
            sampler.stop()
            samples = sampler.snapshot()
            report["health"] = {
                "sample_count": len(samples),
                "p95_latency_ms": _health_p95(samples),
                "samples": samples,
            }
        if browser is not None and browser.poll() is None:
            try:
                if os.name == "nt":
                    browser.kill()
                elif _owned_process_group_is_valid(browser):
                    os.killpg(browser.pid, signal.SIGTERM)
                browser.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if process is not None:
            try:
                _shutdown_owned_process(process, shutdown)
            except Exception as exc:  # noqa: BLE001 - preserve the original failure
                if report.get("failure") is None:
                    report["failure"] = str(exc)[:1000]
                success = False
            report["process"] = {
                **cast(dict[str, object], report.get("process", {})),
                "ended_at": shutdown.get("completed_at"),
                "returncode": process.returncode,
            }
        if success and operator_enabled:
            try:
                report["database_checks"] = _final_database_checks(run_dirs_before)
            except Exception as exc:  # noqa: BLE001 - final persistence is a gate
                report["failure"] = str(exc)[:1000]
                success = False
        report["shutdown"] = shutdown
        report["ended_at"] = _timestamp()
        report["status"] = "passed" if success and report.get("failure") is None else "failed"
        output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-port", type=int, default=0)
    parser.add_argument("--ui-port", type=int, default=0)
    parser.add_argument("--playwright-command", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if os.environ.get("UNDERWATER_TRACKING_RUN_LIVE_ACCEPTANCE") != "1":
        print("set UNDERWATER_TRACKING_RUN_LIVE_ACCEPTANCE=1 to run live acceptance", file=sys.stderr)
        return 2
    if os.environ.get("UNDERWATER_TRACKING_RUN_REAL_LLM") != "1":
        print("set UNDERWATER_TRACKING_RUN_REAL_LLM=1 to run live acceptance", file=sys.stderr)
        return 2
    command = (
        sys.executable,
        "main.py",
        "--config",
        str(args.config),
        "--seed",
        str(args.seed),
    )
    playwright_command = (
        tuple(shlex.split(args.playwright_command))
        if args.playwright_command
        else None
    )
    return run_acceptance(
        command=command,
        api_port=args.api_port,
        ui_port=args.ui_port,
        output_path=args.output,
        checkpoints=default_acceptance_checkpoints(),
        playwright_command=playwright_command,
    )


if __name__ == "__main__":
    raise SystemExit(main())
