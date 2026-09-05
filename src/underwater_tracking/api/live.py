"""Bridge the checkpointed carrier runtime to live operational frames.

The simulation/agent loop owns LangGraph and SQLite state.  This adapter is
the narrow publication seam: it resolves the latest checkpointed semantic
outputs, combines them with the current estimator snapshot, builds a
truth-safe ``OperationalFrame``, persists it for replay, and publishes the
same object to the WebSocket hub.  It never reads the evaluation sink.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from typing import Any, Literal, Protocol, cast

from underwater_tracking.api.frame_builder import (
    build_operational_frame,
    operational_frame_json,
)
from underwater_tracking.api.frame_logger import FrameLogger
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    IntentHypothesis,
    PlanAdjustmentSuggestion,
    TrajectoryDiffGateState,
    TrajectoryDiffResult,
    TrackingPlan,
)
from underwater_tracking.domain.event_registry import is_blue_public
from underwater_tracking.domain.models import (
    DEFAULT_EVENT_AUDIENCES,
    EventAudience,
    EventLevel,
    RuntimeEvent,
    SituationSnapshot,
)
from underwater_tracking.domain.ui_models import (
    BrainActivityRecord,
    MetricView,
    OperationalFrame,
    OperationalStage,
    OperationalThinkingSummary,
    PlanningHealthView,
)
from underwater_tracking.domain.execution_models import OperationalExecutionSnapshot
from underwater_tracking.domain.prediction_models import AcceptedPrediction
from underwater_tracking.runtime.execution_health import ExecutionHealth
from underwater_tracking.runtime.mission_controller import MissionSnapshot
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.world_model.models import WorldModelForecast


# The live hub keeps the complete in-process frame.  Replay persistence only
# needs a bounded recent context because each accepted event is already written
# to the durable event/ledger stores and stage markers are emitted at their own
# frame boundaries.
_REPLAY_EVENT_HISTORY_LIMIT = 64
_REPLAY_MISSION_EVENT_HISTORY_LIMIT = 16
_REPLAY_LEDGER_HISTORY_LIMIT = 32
_REPLAY_OPERATOR_AUDIT_ID_LIMIT = 512
_REPLAY_PLAN_TIMELINE_LIMIT = 32


class FramePersistencePolicy:
    """Decide which live frames are durable replay boundaries."""

    def __init__(self, sample_interval_s: int | None) -> None:
        self._sample_interval_s = sample_interval_s
        self._last: OperationalFrame | None = None

    def should_persist(
        self,
        frame: OperationalFrame,
        previous: OperationalFrame | None = None,
    ) -> bool:
        prior = previous if previous is not None else self._last
        if prior is None:
            self._last = frame
            return True
        boundary = (
            frame.plan_version != prior.plan_version
            or _has_new_items(frame.events, prior.events)
            or _has_new_items(frame.mission_events, prior.mission_events)
            or frame.run_phase != prior.run_phase
            or frame.sim_time_s >= prior.sim_time_s + (self._sample_interval_s or 0)
        )
        if self._sample_interval_s is None:
            boundary = True
        if boundary:
            self._last = frame
        return boundary


def _has_new_items(current: Sequence[object], previous: Sequence[object]) -> bool:
    """Detect appended event history without treating the retained tail as new."""
    previous_ids = {
        str(getattr(item, "event_id", ""))
        for item in previous
        if getattr(item, "event_id", None) is not None
    }
    current_ids = {
        str(getattr(item, "event_id", ""))
        for item in current
        if getattr(item, "event_id", None) is not None
    }
    if current_ids or previous_ids:
        return bool(current_ids - previous_ids)
    return tuple(current) != tuple(previous)


def compact_operational_frame(frame: OperationalFrame) -> OperationalFrame:
    """Bound replay-only geometry while retaining current operational evidence.

    The projection is used for public Hub/HTTP snapshots and JSONL persistence.
    Internal verification and durable event/ledger stores retain their full
    state, while public consumers receive a bounded recent context.
    """
    compact_uuvs = tuple(
        uuv.model_copy(
            update={
                "breadcrumb": uuv.breadcrumb[-24:],
                "connected_peer_ids": uuv.connected_peer_ids[:4],
            }
        )
        for uuv in frame.uuvs
    )
    if len(frame.region_timeline) > 16:
        priority = {
            "active": 0,
            "handed_off": 1,
            "degraded": 2,
            "planned": 3,
            "uncovered": 4,
        }
        indexed_rows = sorted(
            enumerate(frame.region_timeline),
            key=lambda item: (priority[item[1].status], item[0]),
        )[:16]
        compact_timeline = tuple(row for _, row in indexed_rows)
    else:
        compact_timeline = frame.region_timeline
    event_ids = {event.event_id for event in frame.events}
    referenced_event_ids = {
        event_id
        for row in frame.ledger
        for event_id in getattr(row, "trigger_event_ids", ())
    }
    referenced_event_ids.update(
        factor.ref_id
        for row in frame.plan_timeline
        for factor in getattr(row, "factors", ())
        if getattr(factor, "kind", None) == "event"
    )
    referenced_event_ids.update(frame.llm_thinking_source_event_ids)
    selected_event_ids = referenced_event_ids & event_ids
    recent_events = frame.events[-_REPLAY_EVENT_HISTORY_LIMIT:]
    compact_events = tuple(
        event
        for event in frame.events
        if event.event_id in selected_event_ids
        or event in recent_events
    )
    compact_ledger = frame.ledger[-_REPLAY_LEDGER_HISTORY_LIMIT:]
    return frame.model_copy(
        update={
            "uuvs": compact_uuvs,
            "region_timeline": compact_timeline,
            "events": compact_events,
            "mission_events": frame.mission_events[-_REPLAY_MISSION_EVENT_HISTORY_LIMIT:],
            "ledger": compact_ledger,
            "operator_audit_event_ids": frame.operator_audit_event_ids[
                -_REPLAY_OPERATOR_AUDIT_ID_LIMIT:
            ],
            "plan_timeline": frame.plan_timeline[-_REPLAY_PLAN_TIMELINE_LIMIT:],
        }
    )


class RuntimeFramePort(Protocol):
    def active_plan(self) -> TrackingPlan | None: ...

    def get_state(self) -> Mapping[str, object]: ...


class EventPort(Protocol):
    def list_events(
        self, *, scenario_id: str | None = None, limit: int = 1000
    ) -> Sequence[Any]: ...

    def get(self, event_id: str) -> Any | None: ...


class LedgerPort(Protocol):
    def list_decisions(
        self, scenario_id: str | None = None, limit: int = 100
    ) -> Sequence[Any]: ...

    def list_directives(
        self, scenario_id: str | None = None, status: str | None = None
    ) -> Sequence[ExpertDirective]: ...


class OperationalFramePublisher:
    """Publish one complete frame after every carrier observation cycle."""

    def __init__(
        self,
        *,
        runtime: RuntimeFramePort,
        ledger: LedgerPort | DecisionLedger,
        events: EventPort | EventRepository,
        hub: OperationalHub,
        logger: FrameLogger | None = None,
        mission_snapshot_provider: Callable[[], MissionSnapshot | None] | None = None,
        candidate_regions_provider: Callable[[], Mapping[str, object]] | None = None,
        history_limit: int = 300,
        event_history_limit: int | None = None,
        physics_step_s: int = 5,
        mission_event_history_limit: int = 2048,
        configured_roles: Sequence[Literal["master", "slave", "adversary"]] = (
            "master",
            "slave",
            "adversary",
        ),
        planning_health_provider: Callable[[], PlanningHealthView] | None = None,
        run_phase_provider: Callable[[], str] | None = None,
        persistence_policy: FramePersistencePolicy | None = None,
        persistence_projection: Callable[[OperationalFrame], OperationalFrame] | None = None,
    ) -> None:
        self._runtime = runtime
        self._ledger = ledger
        self._events = events
        self._hub = hub
        self._logger = logger
        self._mission_snapshot_provider = mission_snapshot_provider
        self._candidate_regions_provider = candidate_regions_provider
        self._history_limit = max(1, history_limit)
        self._event_history_limit = max(
            1, event_history_limit if event_history_limit is not None else history_limit
        )
        self._physics_step_s = max(1, physics_step_s)
        self._mission_event_history_limit = max(1, mission_event_history_limit)
        self._configured_roles = tuple(dict.fromkeys(configured_roles))
        self._planning_health_provider = planning_health_provider
        self._run_phase_provider = run_phase_provider
        self._persistence_policy = persistence_policy
        self._persistence_projection = persistence_projection
        self._breadcrumbs: dict[str, list[tuple[float, float]]] = {}
        self._operator_audit_event_ids: set[str] = set()
        self._last_frame_id = -1
        self._event_reference_cache: dict[str, RuntimeEvent | None] = {}

    def publish(self, snapshot: SituationSnapshot) -> OperationalFrame:
        self._record_breadcrumbs(snapshot)
        state = self._runtime.get_state()
        hypotheses = _mapping_of(state.get("intent_hypotheses"), IntentHypothesis)
        accepted_predictions = _mapping_of(
            state.get("accepted_predictions"), AcceptedPrediction
        )
        prediction_diffs = _mapping_of(
            state.get("prediction_diffs"), TrajectoryDiffResult
        )
        prediction_gates = _mapping_of(
            state.get("prediction_diff_gates"), TrajectoryDiffGateState
        )
        world_model_forecasts = _mapping_of(
            state.get("world_model_forecasts"), WorldModelForecast
        )
        raw_suggestions = state.get("plan_adjustment_suggestions")
        suggestions = tuple(
            item
            for item in cast(Sequence[object], raw_suggestions or ())
            if isinstance(item, PlanAdjustmentSuggestion)
        )
        stored_events = self._stored_events(snapshot)
        mission_snapshot = (
            self._mission_snapshot_provider()
            if self._mission_snapshot_provider is not None
            else None
        )
        decisions = self._ledger.list_decisions(snapshot.scenario_id, limit=50)
        applied = self._ledger.list_directives(snapshot.scenario_id, status="applied")
        planning_revision = _nonnegative_int(state.get("snapshot_revision"))
        planning_sim_time_s = _nonnegative_int(state.get("snapshot_sim_time_s"))
        if planning_sim_time_s is None:
            planning_age_s = None
            planning_status: Literal["current", "stale", "unavailable"] = "unavailable"
        else:
            planning_age_s = max(0, snapshot.sim_time_s - planning_sim_time_s)
            planning_status = "current" if planning_age_s == 0 else "stale"
        mission_event_tail = (
            tuple(mission_snapshot.events[-self._mission_event_history_limit :])
            if mission_snapshot is not None
            else ()
        )
        active_plan = self._runtime.active_plan()
        execution_snapshot = _current_execution_snapshot(self._runtime)
        world_model_reader = getattr(self._runtime, "world_model_forecasts_for_publication", None)
        if callable(world_model_reader):
            world_model_forecasts = world_model_reader(snapshot, execution_snapshot)
        authoritative_execution_health = _authoritative_execution_health(
            self._runtime,
            sim_time_s=snapshot.sim_time_s,
        )
        if execution_snapshot is not None:
            accepted = accepted_predictions.get(execution_snapshot.target_id)
            if not _accepted_prediction_matches_execution(accepted, execution_snapshot):
                accepted_predictions = dict(accepted_predictions)
                accepted_predictions.pop(execution_snapshot.target_id, None)
        public_plan = None if execution_snapshot is not None else active_plan
        stored_events = self._include_referenced_events(
            snapshot,
            stored_events,
            decisions,
            active_plan,
        )
        role_activity = self._role_activity(snapshot.scenario_id)
        planning_health = (
            self._planning_health_provider()
            if self._planning_health_provider is not None
            else None
        )
        if planning_health is not None:
            role_status: Literal[
                "ready", "running", "succeeded", "degraded", "failed"
            ]
            if planning_health.status == "running":
                role_status = "running"
            elif planning_health.status in {"rejected", "failed"}:
                role_status = "failed"
            elif planning_health.status in {
                "awaiting_retry",
                "invalidated",
                "degraded",
            }:
                role_status = "degraded"
            elif planning_health.status == "committed":
                role_status = "succeeded"
            else:
                role_status = "ready"
            role_activity["master"] = BrainActivityRecord(
                brain_id="carrier-master",
                role="master",
                status=role_status,
                operation="planning_epoch",
                sim_time_s=snapshot.sim_time_s,
                message=(
                    planning_health.last_error
                    or f"planning epoch: {planning_health.status}"
                )[:2000],
            )
        run_phase = _run_phase_value(
            self._run_phase_provider()
            if self._run_phase_provider is not None
            else None
        )
        stage_flags = _operational_stage_flags(
            snapshot=snapshot,
            state=state,
            events=stored_events,
            active_plan=active_plan,
            mission_snapshot=mission_snapshot,
            physics_step_s=self._physics_step_s,
        )
        thinking_summary = _operator_thinking(
            snapshot=snapshot,
            state=state,
            events=stored_events,
            active_plan=active_plan,
            stage_flags=stage_flags,
            physics_step_s=self._physics_step_s,
        )
        deterministic_candidates = (
            dict(self._candidate_regions_provider())
            if self._candidate_regions_provider is not None
            else {}
        )
        runtime_candidates = _candidate_regions(state.get("regional_candidates"))
        frame = build_operational_frame(
            snapshot,
            public_plan,
            decisions,
            stored_events,
            _metrics(snapshot, stored_events),
            intent_hypotheses=hypotheses,
            # Raw predictions are kept in runtime state for diff/audit data,
            # but only accepted predictions may create a live corridor.
            predictions={},
            accepted_predictions=accepted_predictions,
            live_authoritative=True,
            prediction_diffs=prediction_diffs,
            prediction_gates=prediction_gates,
            world_model_forecasts=world_model_forecasts,
            applied_directives=applied,
            breadcrumbs={key: tuple(value) for key, value in self._breadcrumbs.items()},
            frame_id=max(snapshot.snapshot_revision, self._last_frame_id + 1),
            physics_step_s=self._physics_step_s,
            llm_paused=bool(getattr(self._runtime, "llm_paused", False)),
            plan_adjustment_suggestions=suggestions,
            mission_snapshot=mission_snapshot,
            execution_snapshot=execution_snapshot,
            authoritative_execution_health=authoritative_execution_health,
            candidate_regions={**deterministic_candidates, **runtime_candidates},
            uuv_only=mission_snapshot is not None or execution_snapshot is not None,
            run_phase=run_phase,
            planning=planning_health,
            operator_audit_event_ids=tuple(sorted(self._operator_audit_event_ids)),
            planning_snapshot_revision=planning_revision,
            planning_sim_time_s=planning_sim_time_s,
            planning_data_age_s=planning_age_s,
            planning_data_status=planning_status,
            mission_event_tail=mission_event_tail,
            operational_stage_flags=stage_flags,
            thinking_summary=thinking_summary,
            role_activity=role_activity,
            configured_roles=self._configured_roles,
        )
        self._last_frame_id = frame.frame_id
        projected = (
            self._persistence_projection(frame)
            if self._persistence_projection is not None
            else frame
        )
        public_frame = OperationalFrame.model_validate(
            projected.model_dump(mode="python")
        )
        serialized = operational_frame_json(public_frame).encode("utf-8")
        if self._logger is not None and (
            self._persistence_policy is None
            or self._persistence_policy.should_persist(public_frame)
        ):
            self._logger.append_serialized(serialized)
        self._hub.publish(public_frame, serialized)
        return frame

    def _role_activity(
        self, scenario_id: str
    ) -> dict[str, BrainActivityRecord]:
        reader = getattr(self._ledger, "latest_role_activity", None)
        if not callable(reader):
            return {}
        try:
            return dict(reader(scenario_id))
        except Exception:  # noqa: BLE001 - status projection must not stop publishing
            return {}

    def close(self) -> None:
        if self._logger is not None:
            self._logger.close()

    @property
    def frame_count(self) -> int:
        return self._logger.count if self._logger is not None else 0

    def _record_breadcrumbs(self, snapshot: SituationSnapshot) -> None:
        for uuv in snapshot.uuvs:
            trail = self._breadcrumbs.setdefault(uuv.uuv_id, [])
            point = (float(uuv.position_xy[0]), float(uuv.position_xy[1]))
            if not trail or trail[-1] != point:
                trail.append(point)
            del trail[:-self._history_limit]

    def _stored_events(self, snapshot: SituationSnapshot) -> tuple[RuntimeEvent, ...]:
        events: dict[str, RuntimeEvent] = {}
        stored_rows = self._events.list_events(
            scenario_id=snapshot.scenario_id, limit=self._event_history_limit
        )
        for row in stored_rows:
            audiences = getattr(row, "audiences", DEFAULT_EVENT_AUDIENCES)
            event_id = str(row.event_id)
            if EventAudience.OPERATOR_AUDIT in audiences:
                self._operator_audit_event_ids.add(event_id)
            if not _is_blue_public_event(str(row.event_type), audiences):
                continue
            events[event_id] = RuntimeEvent(
                event_id=event_id,
                scenario_id=str(row.scenario_id),
                sim_time_s=int(row.sim_time_s),
                event_type=str(row.event_type),
                entity_id=row.target_id,
                level=_event_level(str(row.severity), str(row.event_type)),
                audiences=audiences,
                payload=dict(row.payload),
            )
        for event in snapshot.pending_events:
            if _is_blue_public_event(event.event_type, event.audiences):
                events.setdefault(event.event_id, event)
        return tuple(sorted(events.values(), key=lambda event: (event.sim_time_s, event.event_id)))

    def _include_referenced_events(
        self,
        snapshot: SituationSnapshot,
        events: Sequence[RuntimeEvent],
        decisions: Sequence[Any],
        active_plan: TrackingPlan | None,
    ) -> tuple[RuntimeEvent, ...]:
        """Retain durable trigger events referenced by visible ledgers."""
        referenced_ids = {
            str(event_id)
            for decision in decisions
            for event_id in getattr(decision, "trigger_event_ids", ())
            if isinstance(event_id, str) and event_id
        }
        if active_plan is not None:
            referenced_ids.update(
                event_id
                for event_id in active_plan.trigger_event_ids
                if isinstance(event_id, str) and event_id
            )
        known = {event.event_id for event in events}
        getter = getattr(self._events, "get", None)
        if callable(getter):
            for event_id in sorted(referenced_ids - known):
                if event_id not in self._event_reference_cache:
                    try:
                        row = getter(event_id)
                    except Exception:  # noqa: BLE001 - stale references are handled as absent
                        row = None
                    self._event_reference_cache[event_id] = (
                        _runtime_event_from_stored(row, snapshot.scenario_id)
                        if row is not None
                        else None
                    )
                cached = self._event_reference_cache[event_id]
                if cached is not None:
                    events = (*events, cached)
        return tuple(
            sorted(
                {event.event_id: event for event in events}.values(),
                key=lambda event: (event.sim_time_s, event.event_id),
            )
        )


def _is_blue_public_event(
    event_type: str, audiences: frozenset[EventAudience]
) -> bool:
    try:
        return is_blue_public(event_type, audiences)
    except ValueError:
        return EventAudience.BLUE_PLANNING in audiences


def _runtime_event_from_stored(row: Any, scenario_id: str) -> RuntimeEvent:
    """Convert one repository row without exposing persistence internals."""
    audiences = getattr(row, "audiences", DEFAULT_EVENT_AUDIENCES)
    event_type = str(row.event_type)
    return RuntimeEvent(
        event_id=str(row.event_id),
        scenario_id=str(getattr(row, "scenario_id", scenario_id)),
        sim_time_s=int(row.sim_time_s),
        event_type=event_type,
        entity_id=getattr(row, "target_id", None),
        level=_event_level(str(getattr(row, "severity", "informational")), event_type),
        audiences=audiences,
        payload=dict(getattr(row, "payload", {}) or {}),
    )


def _current_execution_snapshot(runtime: object) -> OperationalExecutionSnapshot | None:
    """Read the authoritative execution snapshot without assuming property shape."""
    reader = getattr(runtime, "current_execution_snapshot", None)
    value = reader() if callable(reader) else reader
    if isinstance(value, OperationalExecutionSnapshot):
        return value
    reader = getattr(runtime, "execution_snapshot", None)
    value = reader() if callable(reader) else reader
    return value if isinstance(value, OperationalExecutionSnapshot) else None


def _authoritative_execution_health(
    runtime: object,
    *,
    sim_time_s: int,
) -> ExecutionHealth | None:
    """Read terminal execution health from the same coordinator as the runtime."""
    coordinator = getattr(runtime, "execution_coordinator", None)
    reader = getattr(coordinator, "execution_health", None)
    if not callable(reader):
        return None
    dependencies = getattr(runtime, "_dependencies", None)
    hard_stale_s = float(
        getattr(dependencies, "execution_hard_stale_s", 900.0)
    )
    health = reader(
        sim_time_s=float(sim_time_s),
        hard_stale_s=hard_stale_s,
    )
    return health if isinstance(health, ExecutionHealth) else None


def _accepted_prediction_matches_execution(
    accepted: AcceptedPrediction | None,
    execution_snapshot: OperationalExecutionSnapshot,
) -> bool:
    if accepted is None or accepted.health.status == "unavailable":
        return False
    prediction = accepted.prediction
    if prediction is None:
        return False
    return (
        prediction.target_id == execution_snapshot.target_id
        and prediction.prediction_id == execution_snapshot.prediction_id
        and float(prediction.sim_time_s)
        == float(execution_snapshot.prediction.origin_sim_time_s)
        and (
            accepted.health.raw_prediction_id is None
            or accepted.health.raw_prediction_id == prediction.prediction_id
        )
    )


def _mapping_of(value: object, expected_type: type[Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, expected_type)
    }


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


_RUN_PHASES = frozenset(
    {
        "created",
        "bootstrap_planning",
        "awaiting_retry",
        "running",
        "completed",
        "stopping",
        "stopped",
        "failed",
    }
)


def _run_phase_value(value: object) -> Literal[
    "created",
    "bootstrap_planning",
    "awaiting_retry",
    "running",
    "completed",
    "stopping",
    "stopped",
    "failed",
]:
    normalized = str(value or "running")
    return cast(Any, normalized if normalized in _RUN_PHASES else "running")


def _candidate_regions(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


_STAGE_ORDER: tuple[OperationalStage, ...] = (
    "task_execution",
    "event_trigger",
    "human_feedback",
    "dynamic_adjustment",
)
_HUMAN_EVENT_MARKERS = (
    "directive",
    "assignment",
    "conversation",
    "question",
    "sensor_mode",
    "manual",
    "feedback",
    "expert",
)
_ADJUSTMENT_EVENT_MARKERS = (
    "plan",
    "replan",
    "handoff",
    "rotation",
    "resource",
    "quality_guard",
    "communication_link",
    "intent_change",
    "target_maneuver",
    "target_detection",
)


def _current_cycle_events(
    snapshot: SituationSnapshot,
    events: Sequence[RuntimeEvent],
    physics_step_s: int,
) -> tuple[RuntimeEvent, ...]:
    window_start = max(0, snapshot.sim_time_s - max(1, physics_step_s))
    current = {
        event.event_id: event
        for event in events
        if window_start <= event.sim_time_s <= snapshot.sim_time_s
    }
    for event in snapshot.pending_events:
        if _is_blue_public_event(event.event_type, event.audiences):
            current[event.event_id] = event
    return tuple(sorted(current.values(), key=lambda item: (item.sim_time_s, item.event_id)))


def _event_has_marker(event: RuntimeEvent, markers: Sequence[str]) -> bool:
    event_type = event.event_type.lower()
    return any(marker in event_type for marker in markers)


def _operational_stage_flags(
    *,
    snapshot: SituationSnapshot,
    state: Mapping[str, object],
    events: Sequence[RuntimeEvent],
    active_plan: TrackingPlan | None,
    mission_snapshot: MissionSnapshot | None,
    physics_step_s: int,
) -> tuple[OperationalStage, ...]:
    """Derive UI phase indicators from current-cycle, auditable state."""
    current_events = _current_cycle_events(snapshot, events, physics_step_s)
    has_plan = active_plan is not None or state.get("selected_plan_ref") is not None
    has_execution = bool(snapshot.uuvs or mission_snapshot is not None or has_plan)
    has_human_feedback = any(
        _event_has_marker(event, _HUMAN_EVENT_MARKERS) for event in current_events
    )
    has_adjustment = any(
        _event_has_marker(event, _ADJUSTMENT_EVENT_MARKERS) for event in current_events
    )
    enabled = {
        "task_execution": has_execution,
        "event_trigger": bool(current_events),
        "human_feedback": has_human_feedback,
        "dynamic_adjustment": has_adjustment,
    }
    return tuple(stage for stage in _STAGE_ORDER if enabled[stage])


def _operator_thinking(
    *,
    snapshot: SituationSnapshot,
    state: Mapping[str, object],
    events: Sequence[RuntimeEvent],
    active_plan: TrackingPlan | None,
    stage_flags: Sequence[OperationalStage],
    physics_step_s: int,
) -> OperationalThinkingSummary:
    """Create a bounded, operator-safe explanation for one planning epoch."""
    current_events = _current_cycle_events(snapshot, events, physics_step_s)
    latest_event = current_events[-1] if current_events else None
    has_current_human_feedback = any(
        _event_has_marker(event, _HUMAN_EVENT_MARKERS) for event in current_events
    )
    has_current_adjustment = any(
        _event_has_marker(event, _ADJUSTMENT_EVENT_MARKERS) for event in current_events
    )
    directive = state.get("latest_directive") if has_current_human_feedback else None
    directive_text = getattr(directive, "raw_text", None)
    raw_reasons = state.get("strategic_replan_reasons") if has_current_adjustment else None
    reasons = (
        tuple(str(item) for item in raw_reasons)
        if isinstance(raw_reasons, (tuple, list))
        else ()
    )

    source_events = current_events
    if not source_events:
        previous_events = tuple(
            event for event in events if event.sim_time_s <= snapshot.sim_time_s
        )
        if previous_events:
            source_events = (
                max(previous_events, key=lambda event: (event.sim_time_s, event.event_id)),
            )
    source_event_ids = tuple(event.event_id for event in source_events[-32:])
    plan_version = active_plan.revision if active_plan is not None else 0
    epoch = state.get("planning_epoch")
    epoch_id = getattr(epoch, "epoch_id", None)
    if not isinstance(epoch_id, str) or not epoch_id:
        epoch_id = f"epoch:{snapshot.scenario_id}:{plan_version}"

    if directive_text:
        trigger = "expert_feedback"
        detail = str(directive_text).strip().replace("\n", " ")[:100]
        thinking = f"已纳入操作员反馈“{detail}”，正在按当前方案版本校验编组、声纳模式与接力资源。"
    elif reasons:
        trigger = "critical_event"
        thinking = "检测到影响跟踪连续性的态势变化，已重新核验区域任务、UUV 资源余量和交接窗口。"
    elif latest_event is not None:
        trigger = "critical_event"
        thinking = (
            f"已处理 {latest_event.event_type} 事件，结合当前观测与通信状态继续执行"
            f"方案 #{plan_version}。"
        )
    elif active_plan is not None:
        trigger = "initialization" if plan_version <= 1 else "critical_event"
        thinking = (
            f"当前无新的人工指令，持续执行方案 #{active_plan.revision}，"
            "监视目标机动、编组质量和UUV剩余航程。"
        )
    else:
        trigger = "initialization"
        thinking = "正在等待首轮有效观测和资源状态，暂不生成超出证据范围的跟踪调整。"

    if "human_feedback" in stage_flags and not directive_text:
        trigger = "expert_feedback"
        thinking = "已保留最近的人工反馈约束，并在当前证据范围内继续校验任务分配。"
    return OperationalThinkingSummary(
        epoch_id=epoch_id,
        plan_version=plan_version,
        trigger=trigger,
        summary=thinking[:240],
        source_event_ids=source_event_ids,
    )


def _event_level(severity: str, event_type: str) -> EventLevel:
    normalized = severity.lower()
    if normalized in {level.value for level in EventLevel}:
        return cast(EventLevel, normalized)
    if any(token in event_type for token in ("plan", "directive", "target_added", "target_lost")):
        return EventLevel.STRATEGIC
    if any(token in event_type for token in ("quality", "ping", "route", "group")):
        return EventLevel.TACTICAL
    return EventLevel.INFORMATIONAL


def _metrics(
    snapshot: SituationSnapshot,
    events: Sequence[RuntimeEvent] = (),
) -> tuple[MetricView, ...]:
    metrics: list[MetricView] = []
    for report in snapshot.group_reports:
        metrics.append(
            MetricView(
                metric_id=f"quality:{report.target_id}",
                label=f"{report.target_id} 编组质量",
                value=report.quality.window_mean,
                unit="score",
                threshold=None,
                window_s=300,
                series=(report.quality.instant, report.quality.window_mean, report.quality.ewma),
            )
        )
        metrics.append(
            MetricView(
                metric_id=f"fim:{report.target_id}",
                label=f"{report.target_id} FIM 最小特征值",
                value=report.belief.fim_min_eigenvalue,
                unit="m⁻²",
                threshold=0.0,
                window_s=300,
                series=(report.belief.fim_min_eigenvalue,),
            )
        )
    if snapshot.uuvs:
        metrics.append(
            MetricView(
                metric_id="fleet:energy",
                label="编队平均剩余能量",
                value=sum(uuv.energy_fraction for uuv in snapshot.uuvs) / len(snapshot.uuvs),
                unit="fraction",
                threshold=0.1,
                window_s=0,
                series=tuple(uuv.energy_fraction for uuv in snapshot.uuvs),
            )
        )
    observability_series: dict[str, list[tuple[int, dict[str, object]]]] = {}
    for event in sorted(events, key=lambda item: (item.sim_time_s, item.event_id)):
        if not event.event_type.startswith("observability_"):
            continue
        tracks = event.payload.get("tracks")
        if not isinstance(tracks, list):
            continue
        for track in tracks:
            if not isinstance(track, dict):
                continue
            target_id = track.get("track_id")
            metric_payload = track.get("metrics")
            if not isinstance(target_id, str) or not isinstance(metric_payload, dict):
                continue
            for metric_id, raw_metric in metric_payload.items():
                if isinstance(metric_id, str) and isinstance(raw_metric, dict):
                    key = f"observability:{target_id}:{metric_id}"
                    observability_series.setdefault(key, []).append((event.sim_time_s, raw_metric))
    for metric_id, samples in sorted(observability_series.items()):
        _, latest = samples[-1]
        numeric_values = [
            instant
            for _, raw in samples
            if (instant := _finite_or_none(raw.get("instant"))) is not None
        ]
        value = numeric_values[-1] if numeric_values else 0.0
        trend = latest.get("trend_per_sec")
        metrics.append(
            MetricView(
                metric_id=metric_id,
                label=f"{metric_id.split(':', 2)[1]} {metric_id.rsplit(':', 1)[-1]}",
                value=value,
                unit=str(latest.get("unit", "")),
                threshold=None,
                window_s=300,
                series=tuple(numeric_values[-30:]),
                status=str(latest.get("status", "UNKNOWN")),
                mean_window=_finite_or_none(latest.get("mean_window")),
                worst_window=_finite_or_none(latest.get("worst_window")),
                trend_per_sec=_finite_or_none(trend),
                valid_fraction=_finite_or_none(latest.get("valid_fraction")),
                reason=str(latest.get("reason", "")),
            )
        )
    return tuple(metrics)


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None
