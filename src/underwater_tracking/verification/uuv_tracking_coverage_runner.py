"""Deterministic no-network runner for the multi-UUV audit."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from itertools import combinations, pairwise
from math import ceil, hypot, isfinite
from pathlib import Path
from typing import TypeVar, cast

from underwater_tracking.cli import _AgentLoop, _mission_controller_for
from underwater_tracking.agent.llm import LLMError
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.execution_models import OperationalExecutionSnapshot
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.runtime.execution_evidence import (
    ExecutionEvidenceResolver,
    answer_execution_question,
)
from underwater_tracking.persistence.events import EventRepository, StoredEvent
from underwater_tracking.memory.source_reader import tracking_episode_kind
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.verification.uuv_tracking_coverage_audit import (
    audit_runtime_execution_trace,
    command_motion_counts,
    deterministic_trace_digest,
    minimum_pairwise_separation_m,
    percentile_summary,
    sampled_footprint_fraction,
    target_position_errors_m,
    waypoint_visit_fraction,
)

Point = tuple[float, float]
ResponseT = TypeVar("ResponseT")
_PHYSICS_AUDIT_SCOPE = "post_deterministic_baseline"
_MIN_UUV_SEPARATION_M = 300.0
REMEDIATION_METRIC_NAMES = (
    "execution_refresh_attempt_count",
    "execution_refresh_commit_count",
    "execution_refresh_rejection_count",
    "execution_expired_frame_count",
    "execution_recovery_count",
    "max_execution_data_age_s",
    "stale_llm_result_count",
    "llm_failure_physics_gap_count",
    "assistant_answer_revision_mismatch_count",
    "memory_episode_source_gap_count",
)

_REFRESH_ATTEMPT_EVENT = "execution_refresh_attempted"
_REFRESH_COMMIT_EVENT = "execution_refresh_committed"
_REFRESH_REJECTION_EVENTS = frozenset(
    {"execution_refresh_rejected", "execution_snapshot_rejected"}
)
_REFRESH_RECOVERY_EVENT = "execution_snapshot_recovered"
_STALE_RESULT_REASONS = frozenset(
    {
        "stale_execution_revision",
        "stale_execution_base",
        "base_revision_mismatch",
        "cas_rejected",
        "execution_revision_not_forward",
    }
)
_ASSISTANT_CASE_QUESTIONS = {
    "expired": "Why is the execution snapshot expired?",
    "active_scan": "Why is the target still ACTIVE_SCAN 1/2?",
    "owner_none": "Why is the tracking owner missing?",
    "handoff_pending": "Why is handoff waiting at 2/3?",
}


class NoNetworkLLMError(LLMError, RuntimeError):
    """Provider failure that preserves the audit's legacy RuntimeError API."""


class NoNetworkLLM:
    """Fail closed if the deterministic audit reaches an LLM call surface."""

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[ResponseT],
        *,
        prompt_version: str = "",
    ) -> ResponseT:
        del payload, response_model, prompt_version
        raise NoNetworkLLMError(
            f"network LLM is disabled for audit operation {operation!r}"
        )

    def cancel(self) -> None:
        return None

    def close(self) -> None:
        return None


def project_audit_frame(
    operational: Mapping[str, object],
    truth: Mapping[str, object],
    *,
    mission_modes: Mapping[str, str],
    region_lifecycles: Mapping[str, str],
    region_assignments: Mapping[str, object] | None = None,
    published_frame: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Pair one operational frame with same-time evaluation-only truth."""
    if operational.get("sim_time_s") != truth.get("sim_time_s"):
        raise ValueError("operational and truth frames must share sim_time_s")
    projected = {
        "sim_time_s": operational.get("sim_time_s"),
        "uuvs": operational.get("uuvs", []),
        "tracks": operational.get("tracks", []),
        "quality": operational.get("quality", []),
        "events": _canonicalize_audit_json(operational.get("events", [])),
        "waypoint_commands": operational.get("waypoint_commands", {}),
        "target_truth": truth.get("targets", []),
        "mission_modes": dict(sorted(mission_modes.items())),
        "region_lifecycles": dict(sorted(region_lifecycles.items())),
        "region_assignments": dict(sorted((region_assignments or {}).items())),
    }
    if published_frame is not None:
        canonical_frame = _canonicalize_audit_json(
            json.loads(
                json.dumps(
                    published_frame,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        )
        if not isinstance(canonical_frame, Mapping):
            raise TypeError("published frame must serialize to a mapping")
        projected["operational_frame"] = canonical_frame
        projected["execution"] = canonical_frame.get("execution")
        projected["runtime_uuvs"] = canonical_frame.get("uuvs", [])
        projected["transport_hash"] = sha256(
            json.dumps(
                canonical_frame,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
    return projected


def _route_projection(
    engine: SimulationEngine,
) -> tuple[dict[str, object], dict[str, object]]:
    snapshot = engine.mission_snapshot()
    if snapshot is None:
        return {}, {}
    routes: dict[str, object] = {
        region.region_id: {
            uuv_id: [list(point) for point in route]
            for uuv_id, route in sorted(region.scan_waypoints_by_uuv.items())
        }
        for region in snapshot.regions
    }
    regions: dict[str, object] = {
        region.region_id: {
            "target_id": region.target_id,
            "polygon": [list(point) for point in region.region_polygon],
            "active_scan_uuv_ids": list(region.active_scan_uuv_ids),
            "passive_track_uuv_ids": list(region.passive_track_uuv_ids),
        }
        for region in snapshot.regions
    }
    return routes, regions


def _stored_event_projection(event: StoredEvent) -> dict[str, object]:
    """Keep the deterministic event evidence needed by the remediation audit."""

    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "scenario_id": event.scenario_id,
        "target_id": event.target_id,
        "sim_time_s": event.sim_time_s,
        "severity": event.severity,
        "payload": dict(event.payload),
    }


def _canonicalize_audit_json(value: object, *, field_name: str | None = None) -> object:
    """Normalize unordered public collections before persisting an audit trace."""

    if isinstance(value, Mapping):
        normalized = {
            key: _canonicalize_audit_json(child, field_name=str(key))
            for key, child in value.items()
        }
        if (
            field_name == "planning"
            and normalized.get("last_result_status") == "failed"
            and normalized.get("status") in {"queued", "degraded"}
        ):
            normalized["status"] = "degraded"
            normalized["deadline_utc_ms"] = None
            normalized["attempt"] = None
        return normalized
    if isinstance(value, list):
        normalized = [
            _canonicalize_audit_json(child, field_name=field_name)
            for child in value
        ]
        if field_name == "audiences":
            return sorted(normalized, key=str)
        return normalized
    return value


def _repository_event_sequence(
    repository: EventRepository,
    scenario_id: str,
) -> list[dict[str, object]]:
    """Read a stable, timestamp-free event sequence from the durable store."""

    events = repository.list_events(
        scenario_id=scenario_id,
        limit=1_000_000,
    )
    return [
        _stored_event_projection(event)
        for event in sorted(
            events,
            key=lambda item: (item.sim_time_s, item.event_type, item.event_id),
        )
    ]


def _deterministic_trace_digest(trace: Mapping[str, object]) -> str:
    """Hash an audit trace without wall-clock-only planning presentation data."""

    def normalize(value: object, *, field_name: str | None = None) -> object:
        if isinstance(value, Mapping):
            normalized = {
                key: (
                    None
                    if key in {"deadline_utc_ms", "transport_hash"}
                    else normalize(child, field_name=str(key))
                )
                for key, child in value.items()
            }
            if (
                field_name == "planning"
                and normalized.get("last_result_status") == "failed"
                and normalized.get("status") in {"queued", "degraded"}
            ):
                normalized["status"] = "degraded"
                normalized["deadline_utc_ms"] = None
                normalized["attempt"] = None
            return normalized
        if isinstance(value, list):
            normalized = [
                normalize(child, field_name=field_name) for child in value
            ]
            if field_name == "audiences":
                return sorted(normalized, key=str)
            return normalized
        if isinstance(value, tuple):
            normalized = [
                normalize(child, field_name=field_name) for child in value
            ]
            if field_name == "audiences":
                return sorted(normalized, key=str)
            return tuple(normalized)
        return value

    normalized = normalize(trace)
    if not isinstance(normalized, Mapping):
        raise TypeError("audit trace digest input must remain a mapping")
    return deterministic_trace_digest(normalized)


def _assistant_case_matches(
    case: str,
    snapshot: OperationalExecutionSnapshot,
    situation: SituationSnapshot,
) -> bool:
    def _status_text(value: object) -> str | None:
        raw = getattr(value, "value", value)
        return raw if isinstance(raw, str) else None

    sim_time_s = float(situation.sim_time_s)
    if case == "expired":
        return sim_time_s > float(snapshot.valid_until_s)
    if case == "active_scan":
        return any(
            _status_text(getattr(group, "lifecycle", None)) == "active_scan"
            for group in snapshot.task_groups
        )
    if case == "owner_none":
        return snapshot.tracking_control.tracking_owner_group_id is None
    return (
        snapshot.tracking_control.pending_successor_group_id is not None
        or any(
            _status_text(getattr(region, "status", None)) == "handoff_pending"
            for region in snapshot.regions
        )
        or any(
            _status_text(getattr(group, "lifecycle", None))
            in {"exiting", "dedicated_release_pending"}
            for group in snapshot.task_groups
        )
    )


def _assistant_case_audit(
    state_history: Sequence[Mapping[str, object]],
    loop: _AgentLoop,
) -> list[dict[str, object]]:
    """Answer four deterministic questions against captured real public states."""

    usable = tuple(
        entry
        for entry in state_history
        if isinstance(entry.get("snapshot"), OperationalExecutionSnapshot)
        and isinstance(entry.get("situation"), SituationSnapshot)
    )
    results: list[dict[str, object]] = []
    for case, question in _ASSISTANT_CASE_QUESTIONS.items():
        selected = next(
            (
                entry
                for entry in usable
                if _assistant_case_matches(
                    case,
                    cast(OperationalExecutionSnapshot, entry["snapshot"]),
                    cast(SituationSnapshot, entry["situation"]),
                )
            ),
            usable[-1] if usable else None,
        )
        if selected is None:
            results.append(
                {
                    "case": case,
                    "available": False,
                    "request": {
                        "question": question,
                        "frame_id": None,
                        "execution_revision": None,
                    },
                    "response": None,
                    "error": "execution_snapshot_unavailable",
                }
            )
            continue
        snapshot = cast(OperationalExecutionSnapshot, selected["snapshot"])
        situation = cast(SituationSnapshot, selected["situation"])
        frame_id = selected.get("frame_id")
        if not isinstance(frame_id, int) or isinstance(frame_id, bool):
            frame_id = snapshot.frame_id
        request = {
            "case": case,
            "question": question,
            "frame_id": frame_id,
            "execution_revision": snapshot.execution_revision,
            "evidence_ids": list(snapshot.evidence_ids[:16]),
        }
        available = _assistant_case_matches(case, snapshot, situation)
        try:
            response = answer_execution_question(
                snapshot,
                question,
                resolver=ExecutionEvidenceResolver(
                    snapshot,
                    events=loop.events,
                    ledger=loop.ledger,
                    plans=loop.plans,
                    frame_id=frame_id,
                ),
                frame_id=frame_id,
                situation=situation,
                current_sim_time_s=situation.sim_time_s,
            )
            results.append(
                {
                    "case": case,
                    "available": available,
                    "request": request,
                    "response": response,
                }
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            results.append(
                {
                    "case": case,
                    "available": available,
                    "request": request,
                    "response": None,
                    "error": type(exc).__name__,
                }
            )
    return results


_MEMORY_EPISODE_AUDIT_FIELDS = (
    "episode_id",
    "user_id",
    "scenario_id",
    "kind",
    "entity_id",
    "opening_execution_revision",
    "status",
    "summary",
    "source_event_ids",
    "source_message_ids",
    "source_decision_ids",
    "source_plan_ids",
    "started_sim_time_s",
    "ended_sim_time_s",
    "closing_execution_revision",
    "frame_id",
    "episode_key",
)


def _memory_episode_projection(episode: object) -> dict[str, object]:
    payload = (
        episode.model_dump(mode="json")
        if hasattr(episode, "model_dump")
        else {}
    )
    return {
        field: payload.get(field)
        for field in _MEMORY_EPISODE_AUDIT_FIELDS
        if field in payload
    }


def _memory_episode_audit(
    loop: _AgentLoop,
    event_sequence: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Rebuild source-backed episodes before closing the audit repositories."""

    service = getattr(loop, "_memory_service", None)
    if service is None:
        return []
    try:
        ingest = getattr(service, "ingest_tracking_events", None)
        if callable(ingest):
            ingest("operator", loop.scenario_id, event_sequence)
        reader = getattr(service, "tracking_episodes", None)
        episodes = (
            reader("operator", loop.scenario_id, limit=10_000)
            if callable(reader)
            else ()
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return []
    return [
        _memory_episode_projection(episode)
        for episode in sorted(
            episodes,
            key=lambda item: (
                str(getattr(item, "kind", "")),
                str(getattr(item, "entity_id", "")),
                int(getattr(item, "opening_execution_revision", 0)),
                str(getattr(item, "episode_id", "")),
            ),
        )
    ]


def run_once(
    *,
    config_path: Path,
    seed: int,
    steps: int,
    work_dir: Path,
) -> dict[str, object]:
    """Run one repository-native deterministic trace without an LLM provider."""
    if steps < 1:
        raise ValueError("steps must be positive")
    work_dir.mkdir(parents=True, exist_ok=False)
    config = load_app_config(config_path)
    controller = _mission_controller_for(config)
    if controller is None:
        raise RuntimeError("audit scenario requires a mission controller")
    truth_frames: list[dict[str, object]] = []
    loop = _AgentLoop(
        config,
        database_path=work_dir / "agent.db",
        llm={"master": NoNetworkLLM()},
        run_id=f"audit-seed-{seed}-steps-{steps}",
        steps=steps,
        seed=seed,
    )
    engine: SimulationEngine | None = None
    frames: list[dict[str, object]] = []
    routes: dict[str, object] = {}
    regions: dict[str, object] = {}
    active_ranges_m: dict[str, float] = {}
    physics_initial_conditions: dict[str, object] = {}
    physics: dict[str, object] = {}
    evidence: dict[str, object] = {}
    state_history: list[dict[str, object]] = []
    event_sequence: list[dict[str, object]] = []
    assistant_cases: list[dict[str, object]] = []
    memory_episodes: list[dict[str, object]] = []
    failure: BaseException | None = None
    try:
        engine = SimulationEngine(
            config,
            seed=seed,
            output_dir=work_dir / "frames",
            evaluation_sink=truth_frames.append,
            mission_controller=controller,
            carrier=loop.on_situation,
            transition_coordinator=loop._transition_coordinator,
            event_repository=loop.events,
            verification_audit=True,
        )
        loop.attach(engine)
        baseline = loop.install_deterministic_baseline(
            engine.publication_situation()
        )
        if baseline is None:
            raise RuntimeError("deterministic baseline was not installed")
        # Baseline installation is pre-run setup and may place assigned UUVs
        # at their deployment boundary. Start motion evidence from that state,
        # so setup placement is not misclassified as commanded teleportation.
        engine._verification_monitor = engine._build_verification_monitor()
        initial_motion_frame = engine.verification_motion_snapshot()
        engine._verification_monitor.observe(initial_motion_frame)
        raw_entities = initial_motion_frame.get("entities")
        entities = (
            tuple(raw_entities)
            if isinstance(raw_entities, Sequence)
            and not isinstance(raw_entities, (str, bytes))
            else ()
        )
        deployed_uuv_ids = tuple(
            sorted(
                str(entity["entity_id"])
                for entity in entities
                if isinstance(entity, Mapping)
                and entity.get("entity_kind") == "uuv"
                and entity.get("lifecycle_state") == "deployed"
                and isinstance(entity.get("entity_id"), str)
            )
        )
        physics_initial_conditions = {
            "frame_id": initial_motion_frame.get("frame_id"),
            "sim_time_s": initial_motion_frame.get("sim_time_s"),
            "deployed_uuv_ids": deployed_uuv_ids,
        }
        routes, regions = _route_projection(engine)
        active_ranges_m = {
            uuv_id: float(uuv.capability.active_range_m)
            for uuv_id, uuv in sorted(engine._uuvs.items())
        }
        for _ in range(steps):
            truth_count = len(truth_frames)
            operational = engine.step()
            if len(truth_frames) != truth_count + 1:
                raise RuntimeError(
                    "evaluation sink must produce exactly one truth frame per step"
                )
            snapshot = engine.mission_snapshot()
            modes = (
                {
                    key: value.value
                    for key, value in snapshot.uuv_modes.items()
                }
                if snapshot is not None
                else {}
            )
            lifecycles = (
                {
                    region.region_id: region.lifecycle.value
                    for region in snapshot.regions
                }
                if snapshot is not None
                else {}
            )
            assignments = (
                {
                    region.region_id: {
                        "active_scan_uuv_ids": list(region.active_scan_uuv_ids),
                        "passive_track_uuv_ids": list(region.passive_track_uuv_ids),
                    }
                    for region in snapshot.regions
                }
                if snapshot is not None
                else {}
            )
            loop.publish_latest()
            published = loop.hub.snapshot()
            if published is None:
                raise RuntimeError("operational publisher did not produce a frame")
            projected = project_audit_frame(
                operational,
                truth_frames[truth_count],
                mission_modes=modes,
                region_lifecycles=lifecycles,
                region_assignments=assignments,
                published_frame=published.model_dump(mode="json"),
            )
            projected["agent_telemetry"] = {
                "llm_failure_count": int(getattr(loop, "_llm_failure_count", 0)),
                "carrier_error_count": int(getattr(loop, "carrier_error_count", 0)),
                "physics_time_advanced": True,
            }
            frames.append(projected)
            runtime = getattr(loop, "_runtime", None)
            current_execution = (
                runtime.current_execution_snapshot()
                if runtime is not None
                and callable(getattr(runtime, "current_execution_snapshot", None))
                else None
            )
            if isinstance(current_execution, OperationalExecutionSnapshot):
                state_history.append(
                    {
                        "frame_id": published.frame_id,
                        "snapshot": current_execution.model_copy(deep=True),
                        "situation": (
                            loop.situation.model_copy(deep=True)
                            if isinstance(loop.situation, SituationSnapshot)
                            else None
                        ),
                    }
                )
        physics = engine.verification_audit()
        evidence = engine.verification_evidence()
        worker = getattr(loop, "_memory_worker", None)
        if worker is not None:
            worker.stop(timeout=5.0)
        event_sequence = _repository_event_sequence(loop.events, loop.scenario_id)
        assistant_cases = _assistant_case_audit(state_history, loop)
        memory_episodes = _memory_episode_audit(loop, event_sequence)
    except BaseException as error:
        failure = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        if engine is not None:
            try:
                engine.logger.close()
            except BaseException as error:  # noqa: BLE001
                # Preserve an active simulation failure across cleanup.
                cleanup_error = error
        try:
            close_ok = loop.close()
        except BaseException as error:  # noqa: BLE001
            # The primary failure remains the most useful diagnostic.
            if cleanup_error is None:
                cleanup_error = error
            close_ok = False
        if failure is None:
            if cleanup_error is not None:
                raise cleanup_error
            if not close_ok:
                raise RuntimeError("agent loop did not close cleanly")
    return {
        "schema_version": 1,
        "scenario": config.scenario.scenario_id,
        "seed": seed,
        "steps": steps,
        "physics_step_s": config.timing.physics_step_s,
        "routes": routes,
        "regions": regions,
        "active_ranges_m": active_ranges_m,
        "frames": frames,
        "event_sequence": event_sequence,
        "assistant_cases": assistant_cases,
        "memory_episodes": memory_episodes,
        "physics_audit_scope": _PHYSICS_AUDIT_SCOPE,
        "physics_audit_initial_conditions": physics_initial_conditions,
        "physics_audit": physics,
        "verification_evidence": evidence,
    }


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return cast(Mapping[str, object], value)


def _as_items(value: object) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(value)


def _point(value: object) -> Point | None:
    values = _as_items(value)
    if len(values) < 2:
        return None
    x, y = values[:2]
    if (
        not isinstance(x, (int, float))
        or isinstance(x, bool)
        or not isinstance(y, (int, float))
        or isinstance(y, bool)
    ):
        return None
    point = (float(x), float(y))
    return point if all(isfinite(component) for component in point) else None


def _frames(trace: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw_frames = _as_items(trace.get("frames"))
    if not raw_frames:
        raise ValueError("audit trace contains no frames")
    frames: list[Mapping[str, object]] = []
    for raw in raw_frames:
        if not isinstance(raw, Mapping):
            raise TypeError("every audit frame must be a mapping")
        frames.append(cast(Mapping[str, object], raw))
    return tuple(frames)


def _uuv_positions(frame: Mapping[str, object]) -> dict[str, Point]:
    positions: dict[str, Point] = {}
    for raw in _as_items(frame.get("uuvs")):
        item = _as_mapping(raw)
        uuv_id = item.get("platform_id")
        point = _point(item.get("position_xy"))
        if (
            isinstance(uuv_id, str)
            and point is not None
            and item.get("deployment_state") == "deployed"
        ):
            positions[uuv_id] = point
    return positions


def _uuv_trajectories(
    frames: Sequence[Mapping[str, object]],
) -> dict[str, tuple[Point, ...]]:
    trajectories: dict[str, list[Point]] = {}
    for frame in frames:
        for uuv_id, point in _uuv_positions(frame).items():
            trajectories.setdefault(uuv_id, []).append(point)
    return {
        uuv_id: tuple(points)
        for uuv_id, points in sorted(trajectories.items())
    }


def _assigned_group_separation_metrics(
    trace: Mapping[str, object],
    frames: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], bool]:
    """Measure each assigned group's separation after initial deployment spread.

    The 300 m value is a task-group waypoint constraint, not a global fleet
    exclusion radius.  Assigned UUVs may begin at a shared deployment boundary;
    once a pair first establishes the required spacing, every later observed
    state must preserve it.
    """

    static_regions = _as_mapping(trace.get("regions"))
    samples_by_pair: dict[
        tuple[str, str, str], list[tuple[float | None, float]]
    ] = {}
    for frame in frames:
        raw_assignments = frame.get("region_assignments")
        regions = (
            cast(Mapping[str, object], raw_assignments)
            if isinstance(raw_assignments, Mapping)
            else static_regions
        )
        positions = _uuv_positions(frame)
        raw_time = frame.get("sim_time_s")
        sim_time_s = (
            float(raw_time)
            if isinstance(raw_time, (int, float))
            and not isinstance(raw_time, bool)
            and isfinite(float(raw_time))
            else None
        )
        for region_id, raw_region in sorted(regions.items()):
            region = _as_mapping(raw_region)
            member_ids = tuple(
                dict.fromkeys(
                    member
                    for field in (
                        "active_scan_uuv_ids",
                        "passive_track_uuv_ids",
                    )
                    for member in _as_items(region.get(field))
                    if isinstance(member, str) and member
                )
            )
            for first_id, second_id in combinations(member_ids, 2):
                samples = samples_by_pair.setdefault(
                    (str(region_id), first_id, second_id), []
                )
                if first_id not in positions or second_id not in positions:
                    continue
                first = positions[first_id]
                second = positions[second_id]
                samples.append(
                    (sim_time_s, hypot(first[0] - second[0], first[1] - second[1]))
                )

    metrics: dict[str, object] = {}
    all_safe = True
    for (region_id, first_id, second_id), samples in sorted(
        samples_by_pair.items()
    ):
        established_index = next(
            (
                index
                for index, (_, distance) in enumerate(samples)
                if distance >= _MIN_UUV_SEPARATION_M - 1.0e-6
            ),
            None,
        )
        established = established_index is not None
        minimum_after_established = (
            min(distance for _, distance in samples[established_index:])
            if established_index is not None
            else None
        )
        pair_safe = (
            established
            and minimum_after_established is not None
            and minimum_after_established >= _MIN_UUV_SEPARATION_M - 1.0e-6
        )
        key = f"{region_id}:{first_id}|{second_id}"
        metrics[key] = {
            "region_id": region_id,
            "member_uuv_ids": [first_id, second_id],
            "sample_count": len(samples),
            "required_minimum_separation_m": _MIN_UUV_SEPARATION_M,
            "minimum_observed_separation_m": (
                min(distance for _, distance in samples) if samples else None
            ),
            "separation_established_at_s": (
                samples[established_index][0]
                if established_index is not None
                else None
            ),
            "minimum_after_establishment_m": minimum_after_established,
            "safe_after_establishment": pair_safe,
        }
        all_safe = all_safe and pair_safe
    return metrics, all_safe


def _polyline_length_m(points: Sequence[Point]) -> float:
    return sum(
        hypot(right[0] - left[0], right[1] - left[1])
        for left, right in pairwise(points)
    )


def _polygon_area_twice(polygon: Sequence[Point]) -> float:
    return sum(
        start[0] * end[1] - end[0] * start[1]
        for start, end in zip(polygon, (*polygon[1:], polygon[0]))
    )


def _point_in_or_on_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        tolerance = 1.0e-9 * max(1.0, abs(dx), abs(dy))
        cross = (x - x1) * dy - (y - y1) * dx
        on_segment = (
            abs(cross) <= tolerance
            and min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
            and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
        )
        if on_segment:
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * dx / dy
            if x < crossing_x:
                inside = not inside
    return inside


def _active_emissions_by_target(
    frames: Sequence[Mapping[str, object]],
    active_ranges_m: Mapping[str, object],
) -> dict[str, tuple[tuple[Point, float], ...]]:
    emissions: dict[str, list[tuple[Point, float]]] = {}
    for frame in frames:
        positions = _uuv_positions(frame)
        for raw_event in _as_items(frame.get("events")):
            event = _as_mapping(raw_event)
            if event.get("event_type") != "active_ping":
                continue
            payload = _as_mapping(event.get("payload"))
            emitter_id = payload.get("emitter_id")
            target_id = event.get("entity_id")
            radius = (
                active_ranges_m.get(emitter_id)
                if isinstance(emitter_id, str)
                else None
            )
            if (
                isinstance(emitter_id, str)
                and isinstance(target_id, str)
                and emitter_id in positions
                and isinstance(radius, (int, float))
                and not isinstance(radius, bool)
                and isfinite(float(radius))
                and float(radius) > 0.0
            ):
                emissions.setdefault(target_id, []).append(
                    (positions[emitter_id], float(radius))
                )
    return {
        target_id: tuple(values)
        for target_id, values in sorted(emissions.items())
    }


def _project_points(value: object) -> tuple[tuple[Point, ...], bool]:
    raw_points = _as_items(value)
    points = tuple(
        point
        for raw_point in raw_points
        if (point := _point(raw_point)) is not None
    )
    return points, len(points) == len(raw_points)


def _coverage_metrics(
    trace: Mapping[str, object],
    frames: Sequence[Mapping[str, object]],
    trajectories: Mapping[str, Sequence[Point]],
) -> tuple[dict[str, object], bool, int]:
    routes = _as_mapping(trace.get("routes"))
    regions = _as_mapping(trace.get("regions"))
    active_ranges = _as_mapping(trace.get("active_ranges_m"))
    emissions = _active_emissions_by_target(frames, active_ranges)
    coverage: dict[str, object] = {}
    geometry_valid = True
    route_count = 0
    for region_id, raw_region in sorted(regions.items()):
        region = _as_mapping(raw_region)
        polygon, polygon_values_valid = _project_points(region.get("polygon"))
        polygon_valid = (
            polygon_values_valid
            and len(polygon) >= 3
            and _polygon_area_twice(polygon) != 0.0
        )
        geometry_valid = geometry_valid and polygon_valid
        by_uuv = _as_mapping(routes.get(region_id))
        route_visitation: dict[str, float | None] = {}
        lengths: dict[str, float] = {}
        for uuv_id, raw_route in sorted(by_uuv.items()):
            route, route_values_valid = _project_points(raw_route)
            if not isinstance(uuv_id, str) or not route or not route_values_valid:
                geometry_valid = False
                continue
            route_count += 1
            geometry_valid = geometry_valid and polygon_valid and all(
                _point_in_or_on_polygon(point, polygon) for point in route
            )
            lengths[uuv_id] = _polyline_length_m(route)
            route_visitation[uuv_id] = waypoint_visit_fraction(
                trajectories.get(uuv_id, ()),
                route,
            )
        target_id = region.get("target_id")
        target_emissions = (
            emissions.get(target_id, ())
            if isinstance(target_id, str)
            else ()
        )
        footprint = (
            sampled_footprint_fraction(polygon, target_emissions)
            if polygon_valid
            else None
        )
        length_values = tuple(lengths.values())
        coverage[region_id] = {
            "target_id": target_id,
            "assigned_route_count": len(lengths),
            "planned_route_length_m_by_uuv": lengths,
            "planned_route_load_span_m": (
                max(length_values) - min(length_values)
                if length_values
                else None
            ),
            "waypoint_visit_fraction_by_uuv": route_visitation,
            "active_emission_count": len(target_emissions),
            "sampled_active_sonar_footprint_fraction": footprint,
            "sampled_active_sonar_footprint_unavailable_reason": (
                None
                if footprint is not None
                else "invalid region polygon"
                if not polygon_valid
                else "no emitted active-sonar footprint was available"
            ),
        }
    if not regions:
        geometry_valid = False
    return coverage, geometry_valid, route_count


def _target_ids(
    frames: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    identifiers = {
        target_id
        for frame in frames
        for raw in _as_items(frame.get("target_truth"))
        if isinstance((target_id := _as_mapping(raw).get("target_id")), str)
    }
    return tuple(sorted(identifiers))


def _strict_sequence(value: object) -> tuple[object, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return tuple(value)


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _entity_ids(value: object) -> tuple[str, ...] | None:
    items = _strict_sequence(value)
    if items is None or any(not isinstance(item, str) or not item for item in items):
        return None
    identifiers = cast(tuple[str, ...], items)
    return identifiers if len(identifiers) == len(set(identifiers)) else None


def _physics_audit_result(
    physics: Mapping[str, object],
    *,
    steps: int,
) -> tuple[int, bool]:
    expected_monitor_frames = steps + 1
    entity_count = _nonnegative_int(physics.get("entity_count"))
    raw_audits = _strict_sequence(physics.get("audits"))
    valid = steps > 0 and entity_count is not None and entity_count > 0
    if raw_audits is None:
        raw_audits = ()
        valid = False
    if entity_count is None or len(raw_audits) != entity_count:
        valid = False

    violation_count = 0
    audit_entity_ids: list[str] = []
    for raw in raw_audits:
        if not isinstance(raw, Mapping):
            valid = False
            continue
        audit = cast(Mapping[str, object], raw)
        entity_id = audit.get("entity_id")
        if (
            not isinstance(entity_id, str)
            or not entity_id
            or entity_id in audit_entity_ids
        ):
            valid = False
        else:
            audit_entity_ids.append(entity_id)
        limit_count = _nonnegative_int(audit.get("limit_violation_count"))
        if limit_count is None:
            valid = False
        else:
            # Teleport and boundary counters are categories already included
            # in limit_violation_count, not independent violations to add again.
            violation_count += limit_count

    raw_coverage = physics.get("coverage")
    if not isinstance(raw_coverage, Mapping):
        return violation_count, False
    coverage = cast(Mapping[str, object], raw_coverage)
    expected_count = _nonnegative_int(coverage.get("expected_entity_count"))
    observed_count = _nonnegative_int(coverage.get("observed_entity_count"))
    expected_ids = _entity_ids(coverage.get("expected_entity_ids"))
    observed_ids = _entity_ids(coverage.get("observed_entity_ids"))
    if (
        entity_count is None
        or expected_count != entity_count
        or observed_count != entity_count
        or expected_ids is None
        or observed_ids is None
        or len(expected_ids) != entity_count
        or len(observed_ids) != entity_count
        or set(expected_ids) != set(observed_ids)
        or set(expected_ids) != set(audit_entity_ids)
    ):
        valid = False

    for field in (
        "observed_frame_count",
        "observed_frame_observation_count",
        "sequence_expected_frame_count",
    ):
        if _nonnegative_int(coverage.get(field)) != expected_monitor_frames:
            valid = False
    first_frame_id = _nonnegative_int(coverage.get("first_frame_id"))
    last_frame_id = _nonnegative_int(coverage.get("last_frame_id"))
    if (
        first_frame_id is None
        or last_frame_id is None
        or first_frame_id != 0
        or last_frame_id != steps
        or last_frame_id - first_frame_id + 1 != expected_monitor_frames
    ):
        valid = False

    for field in (
        "duplicate_frame_ids",
        "duplicate_entity_frame_ids",
        "frame_id_gaps",
        "nonmonotonic_frame_ids",
        "nonmonotonic_sim_time_frame_ids",
        "inconsistent_sample_frame_ids",
    ):
        items = _strict_sequence(coverage.get(field))
        if items is None or items:
            valid = False
    missing_entity_frames = coverage.get("missing_entity_frame_ids")
    if not isinstance(missing_entity_frames, Mapping) or missing_entity_frames:
        valid = False
    return violation_count, valid


def _all_finite(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_finite(child) for child in value)
    return False


def _trace_event_sequence(trace: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """Return unique persisted-or-frame events for metric reduction."""

    raw_sequence = trace.get("event_sequence")
    candidates: list[object] = []
    if (
        isinstance(raw_sequence, Sequence)
        and not isinstance(raw_sequence, (str, bytes))
        and raw_sequence
    ):
        candidates.extend(raw_sequence)
    else:
        raw_events = trace.get("events")
        if isinstance(raw_events, Sequence) and not isinstance(raw_events, (str, bytes)):
            candidates.extend(raw_events)
        for raw_frame in _as_items(trace.get("frames")):
            if not isinstance(raw_frame, Mapping):
                continue
            frame_events = raw_frame.get("events")
            if isinstance(frame_events, Sequence) and not isinstance(frame_events, (str, bytes)):
                candidates.extend(frame_events)
            operational = raw_frame.get("operational_frame")
            if isinstance(operational, Mapping):
                public_events = operational.get("events")
                if isinstance(public_events, Sequence) and not isinstance(
                    public_events, (str, bytes)
                ):
                    candidates.extend(public_events)

    unique: dict[str, Mapping[str, object]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        event_type = candidate.get("event_type")
        if not isinstance(event_type, str) or not event_type:
            continue
        event_id = candidate.get("event_id")
        if isinstance(event_id, str) and event_id:
            key = f"id:{event_id}"
        else:
            key = "value:" + json.dumps(
                dict(candidate),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        unique.setdefault(key, cast(Mapping[str, object], candidate))
    return tuple(
        sorted(
            unique.values(),
            key=lambda event: (
                event.get("sim_time_s")
                if isinstance(event.get("sim_time_s"), (int, float))
                and not isinstance(event.get("sim_time_s"), bool)
                else 0,
                str(event.get("event_type", "")),
                str(event.get("event_id", "")),
            ),
        )
    )


def _runtime_execution_mapping(frame: Mapping[str, object]) -> Mapping[str, object] | None:
    operational = frame.get("operational_frame")
    if isinstance(operational, Mapping):
        execution = operational.get("execution")
        if isinstance(execution, Mapping):
            return execution
    execution = frame.get("execution")
    return execution if isinstance(execution, Mapping) else None


def _remediation_metrics(trace: Mapping[str, object]) -> dict[str, int | float]:
    events = _trace_event_sequence(trace)
    event_types = [event.get("event_type") for event in events]
    refresh_rejections = sum(
        event_type in _REFRESH_REJECTION_EVENTS for event_type in event_types
    )
    refresh_attempts = sum(event_type == _REFRESH_ATTEMPT_EVENT for event_type in event_types)
    refresh_commits = sum(event_type == _REFRESH_COMMIT_EVENT for event_type in event_types)
    recoveries = sum(event_type == _REFRESH_RECOVERY_EVENT for event_type in event_types)

    frames = _frames(trace)
    expired_frame_count = 0
    ages: list[float] = []
    for frame in frames:
        execution = _runtime_execution_mapping(frame)
        if execution is None:
            continue
        if execution.get("health_status") == "expired":
            expired_frame_count += 1
        age = execution.get("data_age_s")
        if isinstance(age, (int, float)) and not isinstance(age, bool) and isfinite(float(age)):
            ages.append(max(0.0, float(age)))

    stale_result_count = 0
    for event in events:
        payload = _as_mapping(event.get("payload"))
        reason = payload.get("reason_code", payload.get("reason"))
        stale = (
            isinstance(reason, str) and reason in _STALE_RESULT_REASONS
        ) or payload.get("failure_category") == "stale"
        status = payload.get("commit_status", payload.get("status"))
        if stale or (
            status == "stale"
            and str(event.get("event_type", "")).startswith("planning_")
        ):
            stale_result_count += 1

    llm_failure_physics_gap_count = 0
    for frame in frames:
        telemetry = _as_mapping(frame.get("agent_telemetry"))
        failure_count = telemetry.get("llm_failure_count", 0)
        failure_active = (
            isinstance(failure_count, int)
            and not isinstance(failure_count, bool)
            and failure_count > 0
        ) or telemetry.get("llm_failure") is True
        if failure_active and telemetry.get("physics_time_advanced") is False:
            llm_failure_physics_gap_count += 1

    assistant_mismatch_count = 0
    raw_assistant = trace.get("assistant_cases")
    if not isinstance(raw_assistant, Sequence) or isinstance(raw_assistant, (str, bytes)):
        raw_assistant = trace.get("assistant_answers")
    if isinstance(raw_assistant, Sequence) and not isinstance(raw_assistant, (str, bytes)):
        for raw_case in raw_assistant:
            if not isinstance(raw_case, Mapping):
                continue
            request = _as_mapping(raw_case.get("request"))
            response = _as_mapping(raw_case.get("response"))
            expected_frame = request.get("frame_id", raw_case.get("expected_frame_id"))
            expected_revision = request.get(
                "execution_revision",
                raw_case.get("expected_execution_revision"),
            )
            if (
                response.get("frame_id") != expected_frame
                or response.get("execution_revision") != expected_revision
            ):
                assistant_mismatch_count += 1

    raw_episodes = trace.get("memory_episodes")
    if not isinstance(raw_episodes, Sequence) or isinstance(raw_episodes, (str, bytes)):
        raw_audit = _as_mapping(trace.get("memory_episode_audit"))
        raw_episodes = raw_audit.get("episodes", ())
    event_ids = {
        event_id
        for event in events
        if isinstance((event_id := event.get("event_id")), str) and event_id
    }
    invalid_episode_count = 0
    if isinstance(raw_episodes, Sequence) and not isinstance(raw_episodes, (str, bytes)):
        for raw_episode in raw_episodes:
            if not isinstance(raw_episode, Mapping):
                invalid_episode_count += 1
                continue
            source_ids = raw_episode.get("source_event_ids")
            if not isinstance(source_ids, Sequence) or isinstance(source_ids, (str, bytes)):
                invalid_episode_count += 1
                continue
            valid_source_ids = {
                source_id
                for source_id in source_ids
                if isinstance(source_id, str) and source_id in event_ids
            }
            if not source_ids or len(valid_source_ids) != len(source_ids):
                invalid_episode_count += 1
    expected_episode_source_ids = {
        event_id
        for event in events
        if isinstance((event_id := event.get("event_id")), str)
        and event_id
        and tracking_episode_kind(event.get("event_type", "")) is not None
    }
    episode_records_present = (
        isinstance(raw_episodes, Sequence)
        and not isinstance(raw_episodes, (str, bytes))
        and bool(raw_episodes)
    )
    memory_gap_count = (
        invalid_episode_count
        if episode_records_present
        else len(expected_episode_source_ids)
    )

    return {
        "execution_refresh_attempt_count": refresh_attempts,
        "execution_refresh_commit_count": refresh_commits,
        "execution_refresh_rejection_count": refresh_rejections,
        "execution_expired_frame_count": expired_frame_count,
        "execution_recovery_count": recoveries,
        "max_execution_data_age_s": max(ages, default=0.0),
        "stale_llm_result_count": stale_result_count,
        "llm_failure_physics_gap_count": llm_failure_physics_gap_count,
        "assistant_answer_revision_mismatch_count": assistant_mismatch_count,
        "memory_episode_source_gap_count": memory_gap_count,
    }


def summarize_trace(trace: Mapping[str, object]) -> dict[str, object]:
    """Aggregate hard gates and descriptive metrics from one saved trace."""
    frames = _frames(trace)
    target_ids = _target_ids(frames)
    tracking: dict[str, object] = {}
    estimate_available = bool(target_ids)
    for target_id in target_ids:
        errors = target_position_errors_m(frames, target_id)
        summary = percentile_summary(errors)
        estimate_available = estimate_available and summary is not None
        tracking[target_id] = {
            "sample_count": len(errors),
            "position_error_m": summary,
            "unavailable_reason": (
                None
                if summary is not None
                else "the shortened run produced no fused estimate for this target"
            ),
        }
    trajectories = _uuv_trajectories(frames)
    control = command_motion_counts(frames)
    coverage, geometry_valid, route_count = _coverage_metrics(
        trace,
        frames,
        trajectories,
    )
    physics = _as_mapping(trace.get("physics_audit"))
    physics_scope = trace.get("physics_audit_scope")
    physics_initial_conditions = _as_mapping(
        trace.get("physics_audit_initial_conditions")
    )
    initial_deployed_uuv_ids = _entity_ids(
        physics_initial_conditions.get("deployed_uuv_ids")
    )
    physics_scope_valid = (
        physics_scope == _PHYSICS_AUDIT_SCOPE
        and _nonnegative_int(physics_initial_conditions.get("frame_id")) == 0
        and _nonnegative_int(physics_initial_conditions.get("sim_time_s")) == 0
        and initial_deployed_uuv_ids is not None
        and bool(initial_deployed_uuv_ids)
    )
    trace_steps = _nonnegative_int(trace.get("steps"))
    trace_steps_valid = (
        trace_steps is not None
        and trace_steps > 0
        and trace_steps == len(frames)
    )
    violation_count, physics_complete = _physics_audit_result(
        physics,
        steps=trace_steps if trace_steps_valid and trace_steps is not None else -1,
    )
    verification = _as_mapping(trace.get("verification_evidence"))
    public_observation_count = len(
        _as_items(verification.get("public_observation_ids"))
    )
    minimum_separation = minimum_pairwise_separation_m(frames)
    assigned_group_separation, assigned_groups_safe = (
        _assigned_group_separation_metrics(trace, frames)
    )
    runtime_execution = audit_runtime_execution_trace(trace)
    remediation_metrics = _remediation_metrics(trace)
    descriptive: dict[str, object] = {
        "tracking": tracking,
        "control_and_motion": {
            **control,
            "minimum_pairwise_separation_m": minimum_separation,
            "minimum_pairwise_separation_scope": "descriptive_global_fleet",
            "assigned_group_separation": assigned_group_separation,
            "trajectory_sample_count_by_uuv": {
                uuv_id: len(points)
                for uuv_id, points in trajectories.items()
            },
            "distance_travelled_m_by_uuv": {
                uuv_id: _polyline_length_m(points)
                for uuv_id, points in trajectories.items()
            },
        },
        "coverage": coverage,
        "physics_audit_scope": physics_scope,
        "physics_audit_initial_conditions": dict(physics_initial_conditions),
        "physics_audit": dict(physics),
        "evidence": {
            "public_observation_count": public_observation_count,
        },
        "runtime_execution": runtime_execution,
        **remediation_metrics,
    }
    for metric_name in (
        "region_side_m",
        "target_detection_radius_m",
        "uuv_detection_radius_m",
        "task_group_size",
        "max_coverage_gap_area_m2",
        "active_ping_count_during_passive",
        "tracking_owner_gap_frames",
        "max_visible_uuv_count",
    ):
        descriptive[metric_name] = runtime_execution.get(metric_name)
    hard_checks = {
        "truth_targets_present": bool(target_ids),
        "fused_tracking_estimate_available": estimate_available,
        "assigned_routes_present": route_count > 0,
        "assigned_route_geometry_valid": geometry_valid,
        "commands_emitted": control["commanded_intervals"] > 0,
        "commanded_uuv_motion_observed": control["moved_intervals"] > 0,
        "assigned_group_separation_after_establishment": assigned_groups_safe,
        "configured_physics_invariants": (
            trace_steps_valid
            and physics_scope_valid
            and physics_complete
            and violation_count == 0
        ),
        "metrics_finite": _all_finite(trace) and _all_finite(descriptive),
    }
    if runtime_execution.get("available") is True:
        hard_checks["runtime_execution_contract"] = (
            runtime_execution.get("valid") is True
        )
    return {
        "schema_version": 1,
        "scenario": trace.get("scenario"),
        "seed": trace.get("seed"),
        "steps": trace.get("steps"),
        **descriptive,
        "physics_violation_count": violation_count,
        "hard_checks": hard_checks,
        "status": "PASS" if all(hard_checks.values()) else "FAIL",
    }


def _write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    pretty: bool,
) -> None:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)


def run_duration_audit(
    *,
    config_path: Path,
    seed: int,
    duration_s: float,
    output_dir: Path,
) -> dict[str, object]:
    """Run one duration-based audit and persist its trace and summary."""
    if isinstance(duration_s, bool) or not isfinite(float(duration_s)):
        raise ValueError("duration_s must be finite and positive")
    if float(duration_s) <= 0:
        raise ValueError("duration_s must be finite and positive")
    config = load_app_config(config_path)
    physics_step_s = float(config.timing.physics_step_s)
    steps = max(1, ceil(float(duration_s) / physics_step_s))
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    trace = run_once(
        config_path=config_path,
        seed=seed,
        steps=steps,
        work_dir=output_dir / "work",
    )
    metrics = summarize_trace(trace)
    metrics["requested_duration_s"] = float(duration_s)
    metrics["actual_duration_s"] = steps * physics_step_s
    metrics["trace_digest"] = _deterministic_trace_digest(trace)
    _write_json(output_dir / "trajectory.json", trace, pretty=False)
    _write_json(output_dir / "metrics.json", metrics, pretty=True)
    return metrics


def run_audit(
    *,
    config_path: Path,
    seed: int,
    steps: int,
    repeat: int,
    work_dir: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    """Run two identical traces and persist non-overwriting audit evidence."""
    if repeat != 2:
        raise ValueError("final audit requires exactly two repeated runs")
    run_dirs = (work_dir / "run-a", work_dir / "run-b")
    for path in (*run_dirs, evidence_dir):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing path: {path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    first = run_once(
        config_path=config_path,
        seed=seed,
        steps=steps,
        work_dir=run_dirs[0],
    )
    second = run_once(
        config_path=config_path,
        seed=seed,
        steps=steps,
        work_dir=run_dirs[1],
    )
    first_digest = _deterministic_trace_digest(first)
    second_digest = _deterministic_trace_digest(second)
    metrics = summarize_trace(first)
    raw_checks = metrics.get("hard_checks")
    if not isinstance(raw_checks, dict):
        raise TypeError("summary hard_checks must be a mutable dictionary")
    hard_checks = cast(dict[str, bool], raw_checks)
    hard_checks["deterministic_repeat"] = first_digest == second_digest
    metrics["trace_digests"] = {
        "run-a": first_digest,
        "run-b": second_digest,
    }
    metrics["status"] = "PASS" if all(hard_checks.values()) else "FAIL"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    _write_json(evidence_dir / "trajectory.json", first, pretty=False)
    _write_json(evidence_dir / "metrics.json", metrics, pretty=True)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Run the deterministic multi-UUV tracking/coverage audit."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    duration_mode = args.duration_s is not None or args.output_dir is not None
    if duration_mode:
        if args.duration_s is None or args.output_dir is None:
            parser.error("--duration-s and --output-dir must be provided together")
        if any(
            value is not None
            for value in (args.steps, args.repeat, args.work_dir, args.evidence_dir)
        ):
            parser.error(
                "duration mode cannot be combined with --steps, --repeat, "
                "--work-dir, or --evidence-dir"
            )
        metrics = run_duration_audit(
            config_path=args.config,
            seed=args.seed,
            duration_s=args.duration_s,
            output_dir=args.output_dir,
        )
        output_key = "output_dir"
        output_path = args.output_dir
    else:
        if any(
            value is None
            for value in (args.steps, args.repeat, args.work_dir, args.evidence_dir)
        ):
            parser.error(
                "legacy mode requires --steps, --repeat, --work-dir, and "
                "--evidence-dir"
            )
        metrics = run_audit(
            config_path=args.config,
            seed=args.seed,
            steps=args.steps,
            repeat=args.repeat,
            work_dir=args.work_dir,
            evidence_dir=args.evidence_dir,
        )
        output_key = "evidence_dir"
        output_path = args.evidence_dir
    status = metrics.get("status")
    output = {
        "status": status,
        "trace_digest": metrics.get("trace_digest"),
        "trace_digests": metrics.get("trace_digests"),
        output_key: str(output_path),
    }
    print(
        json.dumps(
            output,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
