"""Bridge the checkpointed carrier runtime to live operational frames.

The simulation/agent loop owns LangGraph and SQLite state.  This adapter is
the narrow publication seam: it resolves the latest checkpointed semantic
outputs, combines them with the current estimator snapshot, builds a
truth-safe ``OperationalFrame``, persists it for replay, and publishes the
same object to the WebSocket hub.  It never reads the evaluation sink.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Callable, Literal, Protocol, cast

from underwater_tracking.api.frame_builder import build_operational_frame
from underwater_tracking.api.frame_logger import FrameLogger
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    IntentHypothesis,
    PlanAdjustmentSuggestion,
    PredictedTrackRef,
    TrackingPlan,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent, SituationSnapshot
from underwater_tracking.domain.ui_models import (
    BrainActivityRecord,
    MetricView,
    OperationalFrame,
    OperationalStage,
    PlanningHealthView,
)
from underwater_tracking.runtime.mission_controller import MissionSnapshot
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger


class RuntimeFramePort(Protocol):
    def active_plan(self) -> TrackingPlan | None: ...

    def get_state(self) -> Mapping[str, object]: ...


class EventPort(Protocol):
    def list_events(
        self, *, scenario_id: str | None = None, limit: int = 1000
    ) -> Sequence[Any]: ...


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
        history_limit: int = 300,
        physics_step_s: int = 5,
        mission_event_history_limit: int = 2048,
        configured_roles: Sequence[Literal["master", "slave", "adversary"]] = (
            "master",
            "slave",
            "adversary",
        ),
        planning_health_provider: Callable[[], PlanningHealthView] | None = None,
    ) -> None:
        self._runtime = runtime
        self._ledger = ledger
        self._events = events
        self._hub = hub
        self._logger = logger
        self._mission_snapshot_provider = mission_snapshot_provider
        self._history_limit = max(1, history_limit)
        self._physics_step_s = max(1, physics_step_s)
        self._mission_event_history_limit = max(1, mission_event_history_limit)
        self._configured_roles = tuple(dict.fromkeys(configured_roles))
        self._planning_health_provider = planning_health_provider
        self._breadcrumbs: dict[str, list[tuple[float, float]]] = {}
        self._last_frame_id = -1

    def publish(self, snapshot: SituationSnapshot) -> OperationalFrame:
        self._record_breadcrumbs(snapshot)
        state = self._runtime.get_state()
        hypotheses = _mapping_of(state.get("intent_hypotheses"), IntentHypothesis)
        predictions = _mapping_of(state.get("predictions"), PredictedTrackRef)
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
        role_activity = self._role_activity(snapshot.scenario_id)
        planning_health = (
            self._planning_health_provider()
            if self._planning_health_provider is not None
            else None
        )
        if planning_health is not None and planning_health.status == "running":
            role_activity["master"] = BrainActivityRecord(
                brain_id="carrier-master",
                role="master",
                status="running",
                operation="planning_epoch",
                sim_time_s=snapshot.sim_time_s,
                message="规划纪元执行中",
            )
        stage_flags = _operational_stage_flags(
            snapshot=snapshot,
            state=state,
            events=stored_events,
            active_plan=active_plan,
            mission_snapshot=mission_snapshot,
            physics_step_s=self._physics_step_s,
        )
        thinking, thinking_trigger = _operator_thinking(
            snapshot=snapshot,
            state=state,
            events=stored_events,
            active_plan=active_plan,
            stage_flags=stage_flags,
            physics_step_s=self._physics_step_s,
        )
        frame = build_operational_frame(
            snapshot,
            active_plan,
            decisions,
            stored_events,
            _metrics(snapshot, stored_events),
            intent_hypotheses=hypotheses,
            predictions=predictions,
            applied_directives=applied,
            breadcrumbs={key: tuple(value) for key, value in self._breadcrumbs.items()},
            frame_id=max(snapshot.snapshot_revision, self._last_frame_id + 1),
            physics_step_s=self._physics_step_s,
            llm_paused=bool(getattr(self._runtime, "llm_paused", False)),
            plan_adjustment_suggestions=suggestions,
            mission_snapshot=mission_snapshot,
            candidate_regions=_candidate_regions(state.get("regional_candidates")),
            uuv_only=mission_snapshot is not None,
            planning_snapshot_revision=planning_revision,
            planning_sim_time_s=planning_sim_time_s,
            planning_data_age_s=planning_age_s,
            planning_data_status=planning_status,
            mission_event_tail=mission_event_tail,
            operational_stage_flags=stage_flags,
            llm_thinking=thinking,
            llm_thinking_trigger=thinking_trigger,
            role_activity=role_activity,
            configured_roles=self._configured_roles,
        )
        self._last_frame_id = frame.frame_id
        if self._logger is not None:
            self._logger.append(frame)
        self._hub.publish(frame)
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
        for row in self._events.list_events(
            scenario_id=snapshot.scenario_id, limit=self._history_limit
        ):
            event_id = str(row.event_id)
            events[event_id] = RuntimeEvent(
                event_id=event_id,
                scenario_id=str(row.scenario_id),
                sim_time_s=int(row.sim_time_s),
                event_type=str(row.event_type),
                entity_id=row.target_id,
                level=_event_level(str(row.severity), str(row.event_type)),
                payload=dict(row.payload),
            )
        for event in snapshot.pending_events:
            events.setdefault(event.event_id, event)
        return tuple(sorted(events.values(), key=lambda event: (event.sim_time_s, event.event_id)))


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
) -> tuple[str, str]:
    """Create a bounded, operator-safe explanation for the current frame."""
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

    if directive_text:
        trigger = "人工反馈"
        detail = str(directive_text).strip().replace("\n", " ")[:100]
        thinking = f"已纳入操作员反馈“{detail}”，正在按当前方案版本校验编组、声纳模式与接力资源。"
    elif reasons:
        trigger = "动态调整：" + "、".join(reasons[:3])
        thinking = "检测到影响跟踪连续性的态势变化，已重新核验区域任务、UUV 资源余量和交接窗口。"
    elif latest_event is not None:
        trigger = latest_event.event_type
        thinking = (
            f"已处理 {latest_event.event_type} 事件，结合当前观测与通信状态继续执行"
            f"方案 #{active_plan.revision if active_plan is not None else 0}。"
        )
    elif active_plan is not None:
        trigger = "周期性态势评估"
        thinking = (
            f"当前无新的人工指令，持续执行方案 #{active_plan.revision}，"
            "监视目标机动、编组质量和UUV剩余航程。"
        )
    else:
        trigger = "等待首轮态势输入"
        thinking = "正在等待首轮有效观测和资源状态，暂不生成超出证据范围的跟踪调整。"

    if "human_feedback" in stage_flags and not directive_text:
        thinking = "已保留最近的人工反馈约束，并在当前证据范围内继续校验任务分配。"
    return thinking[:240], trigger[:120]


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
