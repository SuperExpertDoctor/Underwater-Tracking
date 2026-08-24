"""Read-only live-demo acceptance polling.

This module intentionally has no mutation client.  It observes the public
health, snapshot, replay and memory endpoints and records a failure when the
real provider cannot produce a terminal planning result.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import Field

from underwater_tracking.api.replay import ReplayIndexError, ReplayService
from underwater_tracking.domain.models import StrictModel


class LiveDemoAcceptanceResult(StrictModel):
    wall_clock_start_utc: str | None = None
    wall_clock_end_utc: str | None = None
    first_plan_wall_s: float | None = None
    final_sim_time_s: int = Field(default=0, ge=0)
    final_plan_version: int = Field(default=0, ge=0)
    observed_stage_ids: frozenset[str] = frozenset()
    stage_sim_times_s: dict[str, int] = Field(default_factory=dict)
    stage_plan_versions: dict[str, int] = Field(default_factory=dict)
    final_run_phase: str = "unknown"
    adversary_llm_decision_count: int = Field(default=0, ge=0)
    memory_event_count: int = Field(default=0, ge=0)
    api_p95_ms: float = Field(default=0.0, ge=0)
    failed_request_count: int = Field(default=0, ge=0)
    output_bytes: int = Field(default=0, ge=0)
    shutdown_s: float = Field(default=0.0, ge=0)
    violations: tuple[str, ...] = ()


_STAGE_MARKERS: dict[str, frozenset[str]] = {
    "carrier_dispatch": frozenset({"carrier_dispatch_completed"}),
    "uuv_deployed": frozenset({"uuv_deployed"}),
    "active_scan": frozenset({"active_ping"}),
    "passive_track": frozenset({"target_estimate_updated"}),
    "handoff": frozenset({"handoff_completed"}),
    "resource_threshold": frozenset(
        {"endurance_threshold_crossed", "battery_rotation", "uuv_range_exhausted"}
    ),
    "recovery": frozenset({"uuv_recovery_requested"}),
    "uuv_recovered": frozenset({"uuv_recovered", "recovery_completed"}),
    "carrier_returned": frozenset({"carrier_returned_to_fleet"}),
}

_REQUIRED_STAGE_ORDER = (
    "initial_plan_committed",
    "carrier_dispatch",
    "uuv_deployed",
    "active_scan",
    "passive_track",
    "handoff",
    "resource_threshold",
    "recovery",
    "uuv_recovered",
    "carrier_returned",
)


def required_stage_order_violations(
    stage_sim_times_s: Mapping[str, int],
) -> tuple[str, ...]:
    """Reject a stage set that cannot represent the required blue chain."""
    violations: list[str] = []
    for earlier, later in zip(
        _REQUIRED_STAGE_ORDER[:-1],
        _REQUIRED_STAGE_ORDER[1:],
        strict=True,
    ):
        earlier_time = stage_sim_times_s.get(earlier)
        later_time = stage_sim_times_s.get(later)
        if earlier_time is not None and later_time is not None and earlier_time > later_time:
            violations.append(f"stage_order:{earlier}_after_{later}")
    return tuple(violations)


def verify_live_demo(
    *,
    base_url: str,
    output_dir: Path,
    require_real_provider: bool,
    wall_timeout_s: float = 1200.0,
    expected_duration_s: int = 28_800,
    poll_interval_s: float = 1.0,
) -> LiveDemoAcceptanceResult:
    """Poll one running command center until completion or a hard failure."""
    if wall_timeout_s <= 0.0 or poll_interval_s <= 0.0:
        raise ValueError("timeouts and poll interval must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    wall_clock_start_utc = datetime.now(timezone.utc).isoformat()
    latencies_ms: list[float] = []
    stages: set[str] = set()
    stage_sim_times_s: dict[str, int] = {}
    stage_plan_versions: dict[str, int] = {}
    violations: list[str] = []
    first_plan_wall_s: float | None = None
    final_sim_time_s = 0
    final_plan_version = 0
    adversary_ids: set[str] = set()
    memory_event_count = 0
    last_frame: Mapping[str, Any] | None = None
    terminal_planning_failure_seen = False
    failed_request_count = 0
    pending_memory_source_checks = 0
    final_run_phase = "unknown"
    previous_frame_sim_time_s: int | None = None

    while time.monotonic() - started < wall_timeout_s:
        poll_started = time.monotonic()
        try:
            health, frame, view_latencies = _get_consistent_live_views(base_url)
            latencies_ms.extend(view_latencies)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            failed_request_count += 1
            violations.append(f"api_poll_failed:{type(exc).__name__}")
            break
        if not isinstance(health, Mapping) or not isinstance(frame, Mapping):
            violations.append("api_poll_returned_non_object")
            break
        last_frame = frame
        final_sim_time_s = _int_value(frame.get("sim_time_s"), final_sim_time_s)
        final_plan_version = _int_value(frame.get("plan_version"), final_plan_version)
        final_run_phase = str(frame.get("run_phase", final_run_phase))
        if (
            previous_frame_sim_time_s is not None
            and final_sim_time_s < previous_frame_sim_time_s
        ):
            violations.append("operational_frame_sim_time_regressed")
        previous_frame_sim_time_s = final_sim_time_s
        if final_plan_version > 0 and first_plan_wall_s is None:
            first_plan_wall_s = time.monotonic() - started
        stages_before = set(stages)
        stages.update(collect_stage_ids(frame))
        for stage_id in stages - stages_before:
            stage_sim_times_s[stage_id] = final_sim_time_s
            stage_plan_versions[stage_id] = final_plan_version
        for event in frame.get("events", ()):
            if isinstance(event, Mapping):
                event_type = str(event.get("event_type", ""))
                if event_type == "target_mission_decision":
                    adversary_ids.add(str(event.get("event_id", "")))
        planning = health.get("planning")
        if isinstance(planning, Mapping):
            status = str(planning.get("status", ""))
            error = str(planning.get("last_error") or "")
            if _planning_failure_is_terminal(planning):
                terminal_planning_failure_seen = True
                violations.append(f"planning_{status}:{_redact(error)}")
                break
        phase = str(frame.get("run_phase", "running"))
        if phase in {"failed", "awaiting_retry"}:
            violations.append(f"run_phase:{phase}")
            break
        # The full acceptance target is also the configured scenario duration.
        # Wait for RunController to publish its terminal phase so the physics
        # endpoint contains exactly the expected final frame, rather than a
        # one-second polling overshoot. Short diagnostic runs keep their old
        # bounded behavior because the application is intentionally longer.
        if phase == "completed" or (
            final_sim_time_s >= expected_duration_s and expected_duration_s < 28_800
        ):
            break
        try:
            memory, memory_latency = _get_json_with_retries(
                base_url,
                "/api/assistant/memory/stream",
                query={"user_id": "operator", "conversation_id": "verification", "limit": "128"},
                attempts=3,
                retry_delay_s=0.05,
            )
            latencies_ms.append(memory_latency)
            if isinstance(memory, Mapping):
                raw_events = memory.get("events", ())
                if isinstance(raw_events, list):
                    memory_event_count = max(memory_event_count, len(raw_events))
                consistency_violations = _operational_consistency_violations(
                    health, frame, memory
                )
            else:
                consistency_violations = _operational_consistency_violations(
                    health, frame, None
                )
            if "memory_source_missing_from_operational_views" in consistency_violations:
                # MemoryWorker is intentionally asynchronous. A queued item can
                # reference the next committed plan before the next frame has
                # published that plan. Require persistence across live polls;
                # unresolved references still fail the gate below.
                pending_memory_source_checks += 1
            else:
                pending_memory_source_checks = 0
            violations.extend(
                item
                for item in consistency_violations
                if item != "memory_source_missing_from_operational_views"
            )
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            # Memory is a separately reported release condition.  Keep polling
            # the authoritative run state so the final result is still useful.
            failed_request_count += 1
            violations.append(f"memory_request_failed:{type(exc).__name__}")
        remaining_poll_delay = poll_interval_s - (time.monotonic() - poll_started)
        if remaining_poll_delay > 0.0:
            time.sleep(remaining_poll_delay)
    if pending_memory_source_checks >= 3:
        violations.append("memory_source_missing_from_operational_views")

    if last_frame is None:
        violations.append("no_operational_frame")
    if time.monotonic() - started >= wall_timeout_s and final_sim_time_s < expected_duration_s:
        violations.append("wall_timeout")
    if terminal_planning_failure_seen and final_run_phase == "bootstrap_planning":
        try:
            terminal_frame, terminal_latency = _get_json(
                base_url, "/api/operational/snapshot"
            )
            latencies_ms.append(terminal_latency)
            if isinstance(terminal_frame, Mapping):
                last_frame = terminal_frame
                final_run_phase = str(terminal_frame.get("run_phase", final_run_phase))
                final_sim_time_s = _int_value(
                    terminal_frame.get("sim_time_s"), final_sim_time_s
                )
                final_plan_version = _int_value(
                    terminal_frame.get("plan_version"), final_plan_version
                )
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            failed_request_count += 1
            violations.append(f"terminal_snapshot_request_failed:{type(exc).__name__}")
    if first_plan_wall_s is None:
        violations.append("initial_plan_not_committed")
    if final_plan_version == 0 and final_sim_time_s >= expected_duration_s:
        violations.append("plan_version_zero_at_deadline")
    if final_sim_time_s > expected_duration_s + 5:
        violations.append("simulation_exceeded_duration")
    replay_violation = _collect_persisted_replay_stages(
        output_dir,
        stages,
        stage_sim_times_s,
        stage_plan_versions,
        expected_duration_s=expected_duration_s,
        expected_plan_version=final_plan_version,
    )
    if replay_violation is not None:
        violations.append(replay_violation)
    required_stages = frozenset(_STAGE_MARKERS)
    missing = sorted(required_stages - stages)
    if missing:
        violations.append("missing_stages:" + ",".join(missing))
    violations.extend(required_stage_order_violations(stage_sim_times_s))
    if memory_event_count <= 0:
        violations.append("memory_stream_empty")
    try:
        evidence, evidence_latency = _get_json_with_retries(
            base_url,
            "/api/verification/evidence",
            attempts=3,
            retry_delay_s=0.1,
        )
        latencies_ms.append(evidence_latency)
        if isinstance(evidence, Mapping):
            raw_decisions = evidence.get("adversary_decisions", ())
            if isinstance(raw_decisions, (list, tuple)):
                adversary_ids.update(
                    str(item.get("decision_id"))
                    for item in raw_decisions
                    if isinstance(item, Mapping) and item.get("decision_id")
                )
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        failed_request_count += 1
        violations.append(f"verification_evidence_request_failed:{type(exc).__name__}")
        if require_real_provider:
            violations.append("verification_evidence_unavailable")
    if len(adversary_ids) == 0 and require_real_provider:
        violations.append("adversary_llm_decision_not_observed")
    api_p95_ms = _percentile(latencies_ms, 0.95)
    if api_p95_ms > 200.0:
        violations.append("api_p95_exceeded_200ms")
    output_bytes = _output_bytes(output_dir)
    if output_bytes > 250 * 1024 * 1024:
        violations.append("output_exceeded_250MiB")
    return LiveDemoAcceptanceResult(
        wall_clock_start_utc=wall_clock_start_utc,
        wall_clock_end_utc=datetime.now(timezone.utc).isoformat(),
        first_plan_wall_s=first_plan_wall_s,
        final_sim_time_s=final_sim_time_s,
        final_plan_version=final_plan_version,
        final_run_phase=final_run_phase,
        observed_stage_ids=frozenset(stages),
        stage_sim_times_s=stage_sim_times_s,
        stage_plan_versions=stage_plan_versions,
        adversary_llm_decision_count=len(adversary_ids),
        memory_event_count=memory_event_count,
        api_p95_ms=api_p95_ms,
        failed_request_count=failed_request_count,
        output_bytes=output_bytes,
        violations=tuple(dict.fromkeys(violations)),
    )


def _get_json(
    base_url: str,
    path: str,
    *,
    query: Mapping[str, str] | None = None,
) -> tuple[object, float]:
    url = base_url.rstrip("/") + path
    if query:
        url += "?" + urlencode(query)
    started = time.perf_counter()
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=10.0) as response:
        payload = json.loads(response.read())
    return payload, (time.perf_counter() - started) * 1000.0


def _get_json_with_retries(
    base_url: str,
    path: str,
    *,
    query: Mapping[str, str] | None = None,
    attempts: int = 3,
    retry_delay_s: float = 0.05,
) -> tuple[object, float]:
    """Retry a transient read without hiding a persistent endpoint failure."""
    if attempts < 1 or retry_delay_s < 0.0:
        raise ValueError("attempts must be positive and retry_delay_s non-negative")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _get_json(base_url, path, query=query)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            if retry_delay_s:
                time.sleep(retry_delay_s)
    assert last_error is not None
    raise last_error


def _get_consistent_live_views(
    base_url: str,
    *,
    attempts: int = 3,
    retry_delay_s: float = 0.05,
) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[float, ...]]:
    """Read health and frame again when a planning transition crosses the pair.

    Health and the WebSocket-backed frame are served by separate snapshots. A
    planning commit can therefore land between the two HTTP reads without
    indicating a real contradiction. The final attempt remains strict: a
    persistent mismatch is returned to the normal consistency audit.
    """
    if attempts < 1 or retry_delay_s < 0.0:
        raise ValueError("attempts must be positive and retry_delay_s non-negative")
    latencies: list[float] = []
    latest: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
    for attempt in range(attempts):
        health, health_latency = _get_json(base_url, "/api/health")
        frame, frame_latency = _get_json(base_url, "/api/operational/snapshot")
        latencies.extend((health_latency, frame_latency))
        if not isinstance(health, Mapping) or not isinstance(frame, Mapping):
            return health, frame, tuple(latencies)  # type: ignore[return-value]
        latest = (health, frame)
        health_view = (_planning_status(health), _planning_epoch_id(health))
        frame_view = (_planning_status(frame), _planning_epoch_id(frame))
        if health_view == frame_view or attempt == attempts - 1:
            return health, frame, tuple(latencies)
        if retry_delay_s:
            time.sleep(retry_delay_s)
    assert latest is not None
    return latest[0], latest[1], tuple(latencies)


def _planning_status(payload: Mapping[str, Any]) -> object:
    planning = payload.get("planning")
    return planning.get("status") if isinstance(planning, Mapping) else None


def _planning_epoch_id(payload: Mapping[str, Any]) -> object:
    planning = payload.get("planning")
    return planning.get("epoch_id") if isinstance(planning, Mapping) else None


def _planning_failure_is_terminal(planning: Mapping[str, Any]) -> bool:
    """Stop only when a planning failure has no automatic retry path.

    A slow real provider can finish after the physical state has changed. The
    epoch coordinator records that semantic invalidation as ``degraded`` and
    keeps the trigger in its retry mailbox. Treating that intermediate health
    state as terminal would interrupt the real provider call and manufacture a
    false provider outage. Dead-lettered or otherwise unqueued failures remain
    terminal and are still reported by the acceptance monitor.
    """
    status = str(planning.get("status", ""))
    if status not in {"awaiting_retry", "failed", "rejected", "degraded"}:
        return False
    queued_event_count = planning.get("queued_event_count", 0)
    if isinstance(queued_event_count, bool):
        queued_event_count = 0
    retry_not_before = planning.get("retry_not_before_utc_ms")
    dead_letter_event_ids = planning.get("dead_letter_event_ids", ())
    has_retry_mailbox = (
        isinstance(queued_event_count, int)
        and queued_event_count > 0
    ) or retry_not_before is not None
    if has_retry_mailbox:
        return False
    if isinstance(dead_letter_event_ids, (list, tuple, set, frozenset)):
        return bool(dead_letter_event_ids) or status in {
            "awaiting_retry",
            "failed",
            "rejected",
            "degraded",
        }
    return True


def _collect_stages(frame: Mapping[str, Any], stages: set[str]) -> None:
    event_types: set[str] = set()
    for key in ("events", "mission_events"):
        raw_events = frame.get(key, ())
        if isinstance(raw_events, list):
            event_types.update(
                str(item.get("event_type", ""))
                for item in raw_events
                if isinstance(item, Mapping)
            )
    for stage_id, markers in _STAGE_MARKERS.items():
        if event_types.intersection(markers):
            stages.add(stage_id)
    estimates = frame.get("target_estimates", ())
    rays = frame.get("bearing_rays", ())
    groups = frame.get("execution_groups", ())
    has_active_group = isinstance(groups, list) and any(
        isinstance(group, Mapping) and group.get("mode") == "active_scan"
        for group in groups
    )
    has_passive_group = isinstance(groups, list) and any(
        isinstance(group, Mapping) and group.get("mode") == "passive_track"
        for group in groups
    )
    if has_active_group:
        # The physical active-search stage begins when an active execution
        # group is installed. The first ping can arrive one physics tick
        # later than a passive estimate from the same group; using the group
        # lifecycle avoids reporting a false passive-before-active ordering.
        stages.add("active_scan")
    if (
        has_passive_group
        and isinstance(estimates, list)
        and estimates
        and isinstance(rays, list)
        and rays
    ):
        stages.add("passive_track")
    planning = frame.get("planning")
    if isinstance(planning, Mapping) and planning.get("status") == "committed":
        stages.add("initial_plan_committed")


def collect_stage_ids(frame: Mapping[str, Any]) -> frozenset[str]:
    """Extract acceptance stages from one public operational frame."""
    stages: set[str] = set()
    _collect_stages(frame, stages)
    return frozenset(stages)


def _collect_persisted_replay_stages(
    output_dir: Path,
    stages: set[str],
    stage_sim_times_s: dict[str, int],
    stage_plan_versions: dict[str, int],
    *,
    expected_duration_s: int | None = None,
    expected_plan_version: int | None = None,
) -> str | None:
    """Recover transient stage events from the authoritative local replay."""
    replay_path = output_dir / "operational_frames.jsonl"
    if not replay_path.exists():
        return "persisted_replay_unavailable"
    try:
        replay = ReplayService(replay_path)
        total = replay.count()
        if total <= 0:
            return "persisted_replay_empty"
        last_frame = None
        for offset in range(0, total, 1_000):
            for frame in replay.range(offset=offset, limit=min(1_000, total - offset)):
                last_frame = frame
                event_types = {
                    event.event_type
                    for event in (*frame.events, *frame.mission_events)
                }
                for stage_id, markers in _STAGE_MARKERS.items():
                    if not event_types.intersection(markers):
                        continue
                    current_time = stage_sim_times_s.get(stage_id)
                    if current_time is None or frame.sim_time_s < current_time:
                        stages.add(stage_id)
                        stage_sim_times_s[stage_id] = frame.sim_time_s
                        stage_plan_versions[stage_id] = frame.plan_version
                if frame.planning is not None and frame.planning.status == "committed":
                    stages.add("initial_plan_committed")
                    current_time = stage_sim_times_s.get("initial_plan_committed")
                    if current_time is None or frame.sim_time_s < current_time:
                        stage_sim_times_s["initial_plan_committed"] = frame.sim_time_s
                        stage_plan_versions["initial_plan_committed"] = frame.plan_version
        if last_frame is None:
            return "persisted_replay_empty"
        terminal_phase = getattr(last_frame.run_phase, "value", last_frame.run_phase)
        if (
            expected_duration_s is not None
            and last_frame.sim_time_s < expected_duration_s
        ) or (
            expected_plan_version is not None
            and last_frame.plan_version != expected_plan_version
        ) or (
            expected_duration_s is not None
            and expected_duration_s >= 28_800
            and terminal_phase != "completed"
        ):
            return "persisted_replay_terminal_mismatch"
    except (ReplayIndexError, OSError, ValueError):
        return "persisted_replay_invalid"
    return None


def _operational_consistency_violations(
    health: Mapping[str, Any],
    frame: Mapping[str, Any],
    memory: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Check that the public frame does not contradict its backing ledgers."""
    violations: list[str] = []
    health_planning = health.get("planning")
    frame_planning = frame.get("planning")
    if isinstance(health_planning, Mapping) and isinstance(frame_planning, Mapping):
        health_status = health_planning.get("status")
        frame_status = frame_planning.get("status")
        if (
            health_status is not None
            and frame_status is not None
            and health_status != frame_status
            and not _expected_planning_view_transition(
                health_planning, frame_planning
            )
            and not _health_view_is_newer_than_frame(health_planning, frame)
        ):
            violations.append("planning_health_frame_mismatch")
    frame_sim_time = _int_value(frame.get("sim_time_s"), -1)
    frame_plan_version = _int_value(frame.get("plan_version"), -1)
    if frame_sim_time < 0 or frame_plan_version < 0:
        violations.append("operational_frame_version_invalid")
    frame_phase = str(frame.get("run_phase", ""))
    content_gate_active = frame_plan_version > 0 and frame_phase in {
        "running",
        "completed",
    }
    event_ids: set[str] = set()
    raw_events = frame.get("events", ())
    if isinstance(raw_events, (list, tuple)):
        for event in raw_events:
            if not isinstance(event, Mapping):
                violations.append("operational_event_not_object")
                continue
            event_id = event.get("event_id")
            event_time = event.get("sim_time_s")
            if not isinstance(event_id, str) or not event_id:
                violations.append("operational_event_id_invalid")
            elif event_id in event_ids:
                violations.append("operational_event_id_duplicate")
            else:
                event_ids.add(event_id)
            if isinstance(event_time, (int, float)) and event_time > frame_sim_time:
                violations.append("operational_event_time_ahead_of_frame")
    ledger_ids: set[str] = set()
    raw_ledger = frame.get("ledger", ())
    if isinstance(raw_ledger, (list, tuple)):
        for row in raw_ledger:
            if not isinstance(row, Mapping):
                violations.append("ledger_row_not_object")
                continue
            decision_id = row.get("decision_id")
            if not isinstance(decision_id, str) or not decision_id:
                violations.append("ledger_decision_id_invalid")
            elif decision_id in ledger_ids:
                violations.append("ledger_decision_id_duplicate")
            else:
                ledger_ids.add(decision_id)
            ledger_time = row.get("sim_time_s")
            if isinstance(ledger_time, (int, float)) and ledger_time > frame_sim_time:
                violations.append("ledger_time_ahead_of_frame")
            final_version = row.get("final_plan_version")
            if isinstance(final_version, (int, float)) and final_version > frame_plan_version:
                violations.append("ledger_plan_version_ahead_of_frame")
            trigger_ids = row.get("trigger_event_ids", ())
            if isinstance(trigger_ids, (list, tuple)) and any(
                isinstance(item, str) and item not in event_ids for item in trigger_ids
            ):
                violations.append("ledger_trigger_missing_from_events")
    raw_timeline = frame.get("plan_timeline", ())
    timeline_plan_ids: set[str] = set()
    timeline_versions: set[int] = set()
    if isinstance(raw_timeline, (list, tuple)):
        if content_gate_active and not raw_timeline:
            violations.append("plan_timeline_empty")
        for item in raw_timeline:
            if not isinstance(item, Mapping):
                violations.append("plan_timeline_row_not_object")
                continue
            timeline_time = item.get("sim_time_s")
            if isinstance(timeline_time, (int, float)) and timeline_time > frame_sim_time:
                violations.append("plan_timeline_time_ahead_of_frame")
            plan = item.get("plan")
            if isinstance(plan, Mapping):
                version = plan.get("version")
                if isinstance(version, (int, float)) and version > frame_plan_version:
                    violations.append("plan_timeline_version_ahead_of_frame")
                if isinstance(version, int) and not isinstance(version, bool):
                    timeline_versions.add(version)
                plan_id = plan.get("plan_id")
                if isinstance(plan_id, str) and plan_id:
                    timeline_plan_ids.add(plan_id)
            factors = item.get("factors", ())
            if isinstance(factors, list):
                for factor in factors:
                    if (
                        isinstance(factor, Mapping)
                        and factor.get("kind") == "event"
                        and isinstance(factor.get("ref_id"), str)
                        and factor["ref_id"] not in event_ids
                    ):
                        violations.append("plan_timeline_event_missing_from_events")
    source_ids = frame.get("llm_thinking_source_event_ids", ())
    if isinstance(source_ids, (list, tuple)) and any(
        isinstance(item, str) and item not in event_ids for item in source_ids
    ):
        violations.append("llm_thinking_source_missing_from_events")
    if content_gate_active:
        thinking = frame.get("llm_thinking")
        if not isinstance(thinking, str) or not thinking.strip():
            violations.append("llm_thinking_empty")
        if not isinstance(source_ids, (list, tuple)) or not source_ids:
            violations.append("llm_thinking_sources_empty")
        if not isinstance(frame.get("llm_thinking_epoch_id"), str) or not str(
            frame.get("llm_thinking_epoch_id")
        ).strip():
            violations.append("llm_thinking_epoch_empty")
        if frame_plan_version not in timeline_versions:
            violations.append("plan_timeline_current_plan_missing")
    if isinstance(memory, Mapping):
        if memory.get("user_id") != "operator":
            violations.append("memory_user_scope_mismatch")
        if memory.get("conversation_id") != "verification":
            violations.append("memory_conversation_scope_mismatch")
        after_cursor = memory.get("after_cursor")
        next_cursor = memory.get("next_cursor")
        raw_memory_events = memory.get("events", ())
        if not isinstance(after_cursor, int) or not isinstance(next_cursor, int):
            violations.append("memory_cursor_invalid")
        elif next_cursor < after_cursor:
            violations.append("memory_cursor_regressed")
        if isinstance(raw_memory_events, (list, tuple)):
            cursors: list[int] = []
            source_reference_count = 0
            known_reference_ids = event_ids | ledger_ids | timeline_plan_ids
            raw_audit_ids = frame.get("operator_audit_event_ids", ())
            if isinstance(raw_audit_ids, (list, tuple)):
                known_reference_ids.update(
                    item for item in raw_audit_ids if isinstance(item, str) and item
                )
            if isinstance(source_ids, (list, tuple)):
                known_reference_ids.update(
                    item for item in source_ids if isinstance(item, str) and item
                )
            for event in raw_memory_events:
                if not isinstance(event, Mapping):
                    violations.append("memory_event_not_object")
                    continue
                cursor = event.get("cursor")
                if not isinstance(cursor, int):
                    violations.append("memory_event_cursor_invalid")
                else:
                    cursors.append(cursor)
                    if isinstance(after_cursor, int) and cursor <= after_cursor:
                        violations.append("memory_event_cursor_not_after_query")
                if event.get("user_id") != "operator":
                    violations.append("memory_event_user_scope_mismatch")
                payload = event.get("payload")
                if isinstance(payload, Mapping):
                    references: set[str] = set()
                    for field in (
                        "source_ids",
                        "source_event_ids",
                        "source_decision_ids",
                        "source_plan_ids",
                    ):
                        raw_references = payload.get(field, ())
                        if isinstance(raw_references, (list, tuple)):
                            references.update(
                                item
                                for item in raw_references
                                if isinstance(item, str) and item
                            )
                    source_reference_count += len(references)
                    if references and not references.intersection(known_reference_ids):
                        violations.append("memory_source_missing_from_operational_views")
            if cursors != sorted(set(cursors)):
                violations.append("memory_event_cursor_not_strictly_increasing")
            if content_gate_active and raw_memory_events and source_reference_count == 0:
                violations.append("memory_sources_empty")
    return tuple(dict.fromkeys(violations))


def _expected_planning_view_transition(
    health: Mapping[str, Any], frame: Mapping[str, Any]
) -> bool:
    """Allow a live frame to lag one planning lifecycle transition."""
    health_epoch = health.get("epoch_id")
    frame_epoch = frame.get("epoch_id")
    if health_epoch != frame_epoch:
        return False
    lifecycle = {health.get("status"), frame.get("status")}
    return lifecycle <= {"idle", "queued", "running", "committed"}


def _health_view_is_newer_than_frame(
    health_planning: Mapping[str, Any], frame: Mapping[str, Any]
) -> bool:
    """Recognize a health read that crossed a newly queued planning epoch."""
    frame_revision = _int_value(frame.get("planning_snapshot_revision"), -1)
    health_base_revision = _int_value(
        health_planning.get("base_physics_revision"), -1
    )
    health_latest_revision = _int_value(
        health_planning.get("latest_physics_revision"), -1
    )
    if frame_revision < 0:
        return health_base_revision >= 0 or health_latest_revision >= 0
    return (
        health_base_revision >= frame_revision >= 0
        or health_latest_revision > frame_revision
    )


def _int_value(value: object, default: int) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return round(ordered[index], 3)


def _output_bytes(output_dir: Path) -> int:
    total = 0
    if not output_dir.exists():
        return 0
    for path in output_dir.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def _redact(value: str) -> str:
    return value.replace("Bearer ", "Bearer <redacted>")[:200]


__all__ = ["LiveDemoAcceptanceResult", "verify_live_demo"]
