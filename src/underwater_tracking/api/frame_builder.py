# src/underwater_tracking/api/frame_builder.py
"""Pure adapter from estimator-visible runtime state to ``OperationalFrame``.

``build_operational_frame`` maps only the state the estimator and planner
produce for the operator: carrier and UUV states, public communication links, target
beliefs (covariance rendered as ellipse axes plus rotation), bearing
observations carried by sonar contacts, group reports, runtime events, the
committed plan, the decision ledger tail, and metrics. Geometry uses the
explicit ``map_bounds_xy`` when supplied (or when a future snapshot exposes
that field), otherwise it uses ``DEFAULT_MAP_BOUNDS``. Every entity list is
sorted by stable ID, so a given run state always produces byte-identical
frames.

Intent hypotheses, predicted corridors, applied reservations, and breadcrumbs
are optional inputs so the adapter remains useful for a cold-start frame but
the live publisher can carry the full estimator state once it is available.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast, get_args

from underwater_tracking.domain import (
    BearingRayView,
    AdversaryView,
    BrainActivityRecord,
    CarrierView,
    CovarianceEllipse,
    EstimateQualityView,
    EventView,
    GroupQualityView,
    GroupView,
    IntentView,
    LedgerView,
    MapBounds,
    MetricView,
    MissionEventView,
    OperationalFrame,
    OperationalSchemeView,
    PlanTimelineView,
    PlanView,
    Point2D,
    PredictionCorridorView,
    PredictionDiffView,
    PredictionHealthView,
    PredictionGridCellView,
    PredictionGridView,
    WorldModelEvidenceView,
    WorldModelEventView,
    WorldModelForecastView,
    WorldModelHorizonView,
    RegionalPlanView,
    RegionalMissionView,
    RegionTaskView,
    TargetEstimateView,
    TimelineFactorView,
    TimelinePlanView,
    TrackingEffectView,
    IntelligenceView,
    UUVView,
    UUVResourceView,
    CarrierMissionView,
    ExecutionGroupView,
    ExecutionRegionView,
    ExecutionView,
    FrameConsistencyReport,
    PlannedAssignmentView,
    TaskGroupView,
)
from underwater_tracking.domain.platforms import (
    PlatformSnapshot,
    UUVPlatformState,
)
from underwater_tracking.domain.ui_models import (
    BrainView,
    CommunicationLinkView,
    OperationalStage,
    OperationalThinkingSummary,
    PlanningHealthView,
    RegionAssignmentView,
    RegionTimelineView,
)
from underwater_tracking.domain.agent_models import (
    DecisionRecord,
    ExpertDirective,
    IntentHypothesis,
    IntentLabel,
    PlanAdjustmentSuggestion,
    PredictedTrackRef,
    TrajectoryDiffGateState,
    TrajectoryDiffResult,
    TrackingPlan,
)
from underwater_tracking.domain.mission_models import (
    ExecutableMissionPlan,
    MissionCandidate,
    PredictionGrid,
)
from underwater_tracking.domain.execution_models import OperationalExecutionSnapshot
from underwater_tracking.domain.prediction_models import AcceptedPrediction
from underwater_tracking.domain.regional_models import (
    RegionalMissionCandidate,
    RegionCell,
    RegionTask,
    TargetRegionPlan,
)
from underwater_tracking.domain.models import (
    BearingObservation,
    CarrierState,
    ContactClassification,
    GroupReport,
    IntelligenceReport,
    OperationalScheme,
    RuntimeEvent,
    SituationSnapshot,
    TargetBelief,
    UUVState,
)
from underwater_tracking.domain.adversary_models import AdversaryOperationalSummary
from underwater_tracking.runtime.mission_controller import MissionSnapshot
from underwater_tracking.world_model.models import WorldModelForecast

# Fallback for legacy/cold-start snapshots without environment metadata. This
# mirrors configs/environment.yaml and the live engine's map_bounds_xy contract.
DEFAULT_MAP_BOUNDS = MapBounds(
    min_x=-12000.0,
    min_y=-12000.0,
    max_x=12000.0,
    max_y=12000.0,
)


def operational_frame_payload(frame: OperationalFrame) -> dict[str, object]:
    """Return the canonical JSON-compatible operational payload."""
    return cast(dict[str, object], frame.model_dump(mode="json"))


def operational_frame_json(frame: OperationalFrame) -> str:
    """Serialize one operational frame through the legacy-field boundary."""
    return frame.model_dump_json()

# Floor for the semiminor axis of a degenerate covariance (meters); the
# frame contract requires strictly positive axes.
_MIN_AXIS_M = 1.0e-3

# Non-finite FIM condition numbers (the belief default when no FIM estimate
# exists yet) render as this finite cap: JSONL cannot carry non-finite
# floats through a pydantic round trip.
_FIM_CONDITION_FINITE_CAP = 1.0e6

_INTENT_LABELS: frozenset[str] = frozenset(get_args(IntentLabel))
_MAX_SCHEME_CONSTRAINTS = 16
_MAX_INTELLIGENCE_VIEWS = 16
_MAX_INTELLIGENCE_SUMMARY_LENGTH = 160


def build_operational_frame(
    snapshot: SituationSnapshot,
    plan: TrackingPlan | None,
    ledger_tail: Sequence[DecisionRecord],
    events: Sequence[RuntimeEvent],
    metrics: Sequence[MetricView],
    *,
    intent_hypotheses: Mapping[str, IntentHypothesis] | None = None,
    predictions: Mapping[str, PredictedTrackRef] | None = None,
    accepted_predictions: Mapping[str, AcceptedPrediction] | None = None,
    live_authoritative: bool = False,
    prediction_diffs: Mapping[str, TrajectoryDiffResult] | None = None,
    prediction_gates: Mapping[str, TrajectoryDiffGateState] | None = None,
    world_model_forecasts: Mapping[str, WorldModelForecast] | None = None,
    applied_directives: Sequence[ExpertDirective] = (),
    breadcrumbs: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    map_bounds_xy: Sequence[float] | None = None,
    frame_id: int | None = None,
    physics_step_s: int = 5,
    llm_paused: bool = False,
    plan_adjustment_suggestions: Sequence[PlanAdjustmentSuggestion] = (),
    mission_snapshot: MissionSnapshot | None = None,
    mission: ExecutableMissionPlan | None = None,
    execution_snapshot: OperationalExecutionSnapshot | None = None,
    prediction_grids: Sequence[PredictionGrid] = (),
    candidate_regions: Mapping[str, object] | None = None,
    uuv_only: bool | None = None,
    run_phase: Literal[
        "created",
        "bootstrap_planning",
        "awaiting_retry",
        "running",
        "completed",
        "stopping",
        "stopped",
        "failed",
    ] = "running",
    planning: PlanningHealthView | None = None,
    operator_audit_event_ids: Sequence[str] = (),
    planning_snapshot_revision: int | None = None,
    planning_sim_time_s: int | None = None,
    planning_data_age_s: int | None = None,
    planning_data_status: Literal["current", "stale", "unavailable"] = "unavailable",
    mission_event_tail: Sequence[RuntimeEvent] | None = None,
    operational_stage_flags: Sequence[OperationalStage] = (),
    llm_thinking: str | None = None,
    llm_thinking_trigger: str | None = None,
    thinking_summary: OperationalThinkingSummary | None = None,
    role_activity: Mapping[str, BrainActivityRecord] | None = None,
    configured_roles: Sequence[Literal["master", "slave", "adversary"]] = (
        "master",
        "slave",
        "adversary",
    ),
    ) -> OperationalFrame:
    """Build one validated operational frame from estimator-visible state.

    ``frame_id`` defaults to the snapshot revision but live publishers may
    supply a monotonically increasing publication sequence. ``plan_version``
    mirrors the committed plan's revision (``0`` with no plan), and the rendered
    active plan carries the same version so the frame contract's
    consistency validator passes. ``events`` and ``ledger_tail`` are mapped
    from the caller's arguments, not from the snapshot's pending-events
    field. ``map_bounds_xy`` follows the environment contract order
    ``(min_x, max_x, min_y, max_y)`` and is injectable until the snapshot
    schema carries the environment bounds itself.
    """
    snapshot_map_bounds = getattr(snapshot, "map_bounds_xy", None)
    map_bounds = _map_bounds(
        map_bounds_xy if map_bounds_xy is not None else snapshot_map_bounds
    )
    by_uuv = {state.uuv_id: state for state in snapshot.uuvs}
    link_views, peers_by_platform = _build_platform_views(
        snapshot.platform_snapshot, map_bounds
    )
    platform_uuvs = {
        state.platform_id: state for state in snapshot.platform_snapshot.roster.uuvs
    } if snapshot.platform_snapshot is not None else {}
    reports = sorted(snapshot.group_reports, key=lambda report: report.target_id)
    adversary_by_target = {
        summary.target_id: summary for summary in snapshot.adversary_summaries
    }
    active_pingers = {
        uuv.uuv_id for uuv in snapshot.uuvs if uuv.sensor_mode == "active"
    }
    active_pingers.update(
        uuv_id
        for event in events
        if event.event_type == "active_ping"
        for uuv_id in event.payload.get("uuv_ids", ())
        if isinstance(uuv_id, str)
    )
    reserved_uuvs = {
        uuv_id
        for directive in applied_directives
        if directive.directive_type == "assignment"
        for uuv_id in directive.assignment_uuv_ids
    }
    uuv_views = tuple(
        _build_uuv_view(
            state,
            plan,
            breadcrumbs=breadcrumbs,
            active_pingers=active_pingers,
            reserved_uuvs=reserved_uuvs,
            platform_state=platform_uuvs.get(state.uuv_id),
            connected_peer_ids=peers_by_platform.get(state.uuv_id, ()),
            map_bounds=map_bounds,
        )
        for state in sorted(snapshot.uuvs, key=lambda state: state.uuv_id)
    )
    latest_ping_by_target = {
        event.entity_id: event.sim_time_s
        for event in sorted(events, key=lambda event: (event.sim_time_s, event.event_id))
        if event.event_type == "active_ping" and event.entity_id is not None
    }
    classification_by_target = {
        contact.contact_id: contact.classification.value for contact in snapshot.contacts
    }
    estimates = tuple(
        _build_estimate(
            report,
            plan,
            intent_hypotheses=intent_hypotheses,
            predictions=predictions,
            accepted_predictions=accepted_predictions,
            live_authoritative=live_authoritative,
            execution_snapshot=(
                execution_snapshot
                if execution_snapshot is not None
                and execution_snapshot.target_id == report.target_id
                else None
            ),
            prediction_diffs=prediction_diffs,
            prediction_gates=prediction_gates,
            world_model_forecasts=world_model_forecasts,
            events=events,
            classification=classification_by_target.get(report.target_id, "unknown"),
            last_ping_s=latest_ping_by_target.get(report.target_id),
            adversary_summary=adversary_by_target.get(report.target_id),
            map_bounds=map_bounds,
        )
        for report in reports
    )
    reported_target_ids = {estimate.target_id for estimate in estimates}
    known_submarines = tuple(
        _build_known_submarine_estimate(
            contact_id=contact.contact_id,
            position_xy=contact.estimated_position_xy,
            plan=plan,
            intent_hypotheses=intent_hypotheses,
            predictions=predictions,
            accepted_predictions=accepted_predictions,
            live_authoritative=live_authoritative,
            execution_snapshot=(
                execution_snapshot
                if execution_snapshot is not None
                and execution_snapshot.target_id == contact.contact_id
                else None
            ),
            adversary_summary=adversary_by_target.get(contact.contact_id),
            map_bounds=map_bounds,
        )
        for contact in snapshot.contacts
        if contact.contact_id not in reported_target_ids
        and contact.classification is ContactClassification.SUBMARINE
        and contact.estimated_position_xy is not None
    )
    estimates = tuple(sorted((*estimates, *known_submarines), key=lambda item: item.target_id))
    groups = tuple(_build_group(report) for report in reports)
    rays = tuple(
        _build_ray(observation, by_uuv, map_bounds)
        for contact in sorted(snapshot.contacts, key=lambda c: c.contact_id)
        for observation in sorted(contact.bearing_rays, key=lambda o: o.observation_id)
        if observation.uuv_id in by_uuv
    )
    plan_views = (_build_plan_view(plan),) if plan is not None else ()
    ledger_views = tuple(
        _build_ledger_view(decision)
        for decision in sorted(ledger_tail, key=lambda d: d.decision_id)
    )
    event_views = tuple(
        _build_event_view(event) for event in sorted(events, key=lambda e: e.event_id)
    )
    metric_views = tuple(sorted(metrics, key=lambda m: m.metric_id))
    mission_is_uuv_only = mission_snapshot is not None if uuv_only is None else uuv_only
    mission_events = (
        tuple(mission_event_tail)
        if mission_event_tail is not None
        else (mission_snapshot.events if mission_snapshot is not None else ())
    )
    plan_version = max(
        plan.revision if plan is not None else 0,
        mission_snapshot.plan_revision if mission_snapshot is not None else 0,
    )
    execution_view = _build_execution_view(
        execution_snapshot,
        current_sim_time_s=snapshot.sim_time_s,
    )
    return OperationalFrame(
        scenario_id=snapshot.scenario_id,
        frame_id=snapshot.snapshot_revision if frame_id is None else frame_id,
        sim_time_s=snapshot.sim_time_s,
        physics_step_s=physics_step_s,
        plan_version=plan_version,
        run_phase=run_phase,
        planning_snapshot_revision=planning_snapshot_revision,
        planning_sim_time_s=planning_sim_time_s,
        planning_data_age_s=planning_data_age_s,
        planning_data_status=planning_data_status,
        operational_stage_flags=tuple(operational_stage_flags),
        llm_thinking=thinking_summary.summary if thinking_summary is not None else llm_thinking,
        llm_thinking_trigger=(
            thinking_summary.trigger if thinking_summary is not None else llm_thinking_trigger
        ),
        llm_thinking_epoch_id=(
            thinking_summary.epoch_id if thinking_summary is not None else None
        ),
        llm_thinking_source_event_ids=(
            thinking_summary.source_event_ids if thinking_summary is not None else ()
        ),
        uuv_only=mission_is_uuv_only,
        map_bounds=map_bounds,
        planning=planning,
        execution=execution_view,
        execution_consistency=(
            FrameConsistencyReport(
                valid=True,
                execution_revision=execution_view.execution_revision,
                source_snapshot_revision=execution_view.source_snapshot_revision,
            )
            if execution_view is not None
            else None
        ),
        operator_audit_event_ids=tuple(
            sorted({item for item in operator_audit_event_ids if item})
        ),
        carrier=(
            None
            if mission_is_uuv_only
            else _build_carrier_view(
                snapshot.carrier,
                snapshot.platform_snapshot.carrier.support_radius_m
                if snapshot.platform_snapshot is not None
                else None,
                map_bounds,
            )
        ),
        carriers=(
            () if mission_is_uuv_only else _build_carrier_views(snapshot, map_bounds)
        ),
        uuvs=uuv_views,
        communication_links=link_views,
        brains=_build_brain_views(
            snapshot,
            events,
            plan,
            llm_paused=llm_paused,
            role_activity=role_activity,
            configured_roles=configured_roles,
        ),
        planned_assignments=(
            ()
            if mission_is_uuv_only
            else _build_planned_assignment_views(mission_snapshot)
        ),
        execution_groups=(
            _execution_group_views(execution_snapshot)
            if execution_snapshot is not None
            else tuple(
                ExecutionGroupView(
                    group_id=group.group_id,
                    target_id=group.target_id,
                    region_id=group.region_id,
                    member_ids=group.member_ids,
                    mode=group.mode,
                )
                for group in sorted(snapshot.execution_groups, key=lambda item: item.group_id)
            )
        ),
        adversaries=tuple(
            _build_adversary_view(summary)
            for summary in sorted(
                snapshot.adversary_summaries, key=lambda item: item.target_id
            )
        ),
        target_estimates=estimates,
        bearing_rays=rays,
        groups=groups,
        regional_plans=_build_regional_plan_views(
            plan, reports, snapshot.sim_time_s, events
        ),
        events=event_views,
        plans=plan_views,
        ledger=ledger_views,
        metrics=metric_views,
        scheme=_build_scheme_view(snapshot.operational_scheme, snapshot.sim_time_s),
        intelligence=_build_intelligence_views(
            snapshot.intelligence_reports, snapshot.sim_time_s
        ),
        plan_timeline=_build_plan_timeline(
            ledger_tail,
            events,
            active_plan=plan,
            current_sim_time_s=snapshot.sim_time_s,
        ),
        region_timeline=build_region_timeline(plan, snapshot.sim_time_s, link_views),
        plan_adjustment_suggestions=tuple(plan_adjustment_suggestions),
        prediction_grids=_build_prediction_grid_views(prediction_grids),
        regional_missions=(
            _execution_regional_mission_views(execution_snapshot)
            if execution_snapshot is not None
            else tuple(
                view.model_copy(update={"carrier_task_id": None})
                for view in _build_regional_mission_views(
                    mission_snapshot,
                    mission,
                    candidate_regions or {},
                )
            )
        ),
        carrier_missions=(
            ()
            if mission_is_uuv_only
            else _build_carrier_mission_views(mission_snapshot)
        ),
        mission_events=_build_mission_event_views(mission_events),
        uuv_mission_modes=(
            {
                uuv_id: mode.value
                for uuv_id, mode in sorted(mission_snapshot.uuv_modes.items())
            }
            if mission_snapshot is not None
            else {}
        ),
        uuv_resources=_build_uuv_resource_views(mission_snapshot),
    )


def _build_execution_view(
    execution: OperationalExecutionSnapshot | None,
    *,
    current_sim_time_s: int,
) -> ExecutionView | None:
    if execution is None:
        return None
    from underwater_tracking.runtime.execution_health import classify_execution_health

    health = classify_execution_health(
        execution,
        sim_time_s=float(current_sim_time_s),
        hard_stale_s=900.0,
    )
    health_status = health.status
    health_reasons = list(health.reason_codes)
    if execution.degradation.degraded and health_status == "current":
        health_status = "degraded"
    health_reasons.extend(execution.degradation.reasons)
    regions = tuple(
        ExecutionRegionView(
            region_id=region.region_id,
            target_id=region.target_id,
            slot_index=region.slot_index,
            execution_revision=region.execution_revision,
            prediction_id=region.prediction_id,
            geometry=tuple(Point2D(x=x, y=y) for x, y in region.geometry),
            start_s=region.start_s,
            end_s=region.end_s,
            geometry_revision=region.geometry_revision,
            predecessor_region_id=region.predecessor_region_id,
            successor_region_id=region.successor_region_id,
            handoff_start_s=region.handoff_start_s,
            handoff_end_s=region.handoff_end_s,
            status=region.status,
            task_group_id=region.task_group_id or "unassigned",
            evidence_ids=region.evidence_ids,
        )
        for region in execution.regions
    )
    task_groups = tuple(
        TaskGroupView(
            task_group_id=group.task_group_id,
            target_id=group.target_id,
            region_id=group.region_id,
            execution_revision=group.execution_revision,
            member_uuv_ids=group.member_uuv_ids,
            active_verifier_uuv_id=group.active_verifier_uuv_id,
            passive_tracker_uuv_id=group.passive_tracker_uuv_id,
            status=group.status,
            evidence_ids=group.evidence_ids,
        )
        for group in execution.task_groups
    )
    reasons = tuple(execution.degradation.reasons)
    return ExecutionView(
        target_id=execution.target_id,
        execution_revision=execution.execution_revision,
        source_snapshot_revision=execution.source_snapshot_revision,
        prediction_revision=execution.prediction_revision,
        intent_revision=execution.intent_revision,
        data_age_s=max(0.0, health.age_s),
        valid_from_s=execution.valid_from_s,
        valid_until_s=execution.valid_until_s,
        health_status=health_status,
        health_reasons=tuple(dict.fromkeys(health_reasons)),
        region_generation_mode=_execution_region_generation_mode(execution),
        plan_source=execution.plan_source,
        current_region_id=execution.current_region_id,
        next_region_id=execution.next_region_id,
        evidence_ids=execution.evidence_ids,
        regions=regions,
        task_groups=task_groups,
        reserve_uuv_ids=tuple(
            sorted(reserve.uuv_id for reserve in execution.reserve_uuvs)
        ),
        degraded=execution.degradation.degraded,
        degradation_reasons=reasons,
        active_plan_preserved=execution.degradation.active_plan_preserved,
    )


def _execution_region_generation_mode(
    execution: OperationalExecutionSnapshot,
) -> Literal[
    "imm",
    "degraded_prediction",
    "boundary_recovery",
    "reprojected_previous",
]:
    marker = "region_generation_mode:"
    explicit = next(
        (
            reason.removeprefix(marker)
            for reason in execution.degradation.reasons
            if reason.startswith(marker)
        ),
        None,
    )
    if explicit in {
        "imm",
        "degraded_prediction",
        "boundary_recovery",
        "reprojected_previous",
    }:
        return cast(
            Literal[
                "imm",
                "degraded_prediction",
                "boundary_recovery",
                "reprojected_previous",
            ],
            explicit,
        )
    if execution.prediction.prediction_regime == "boundary_recovery":
        return "boundary_recovery"
    if execution.prediction.prediction_regime == "imm":
        return "imm"
    return "degraded_prediction"


def _execution_group_views(
    execution: OperationalExecutionSnapshot | None,
) -> tuple[ExecutionGroupView, ...]:
    if execution is None:
        return ()
    return tuple(
        ExecutionGroupView(
            group_id=group.task_group_id,
            target_id=group.target_id,
            region_id=group.region_id,
            member_ids=group.member_uuv_ids,
            mode=(
                "active_scan"
                if group.status in {"active", "handoff_pending"}
                else "passive_track"
            ),
        )
        for group in execution.task_groups
    )


def _execution_regional_mission_views(
    execution: OperationalExecutionSnapshot | None,
) -> tuple[RegionalMissionView, ...]:
    if execution is None:
        return ()
    groups_by_region = {group.region_id: group for group in execution.task_groups}
    lifecycle_by_status = {
        "planned": "PLANNED",
        "prepositioning": "CARRIER_DEPLOYING",
        "active": "ACTIVE_SCAN",
        "passive": "PASSIVE_TRACK",
        "handoff_pending": "HANDOFF_PENDING",
        "handoff_completed": "TRACKING_COMPLETED",
        "monitoring_complete": "TRACKING_COMPLETED",
        "degraded": "DEGRADED",
        "uncovered": "UNCOVERED",
    }
    return tuple(
        RegionalMissionView(
            region_id=region.region_id,
            target_id=region.target_id,
            geometry=tuple(Point2D(x=x, y=y) for x, y in region.geometry),
            entry_s=int(region.start_s),
            exit_s=max(int(region.start_s) + 1, int(region.end_s)),
            lifecycle=lifecycle_by_status[region.status],
            active_scan_uuv_ids=(groups_by_region[region.region_id].active_verifier_uuv_id,),
            passive_track_uuv_ids=(groups_by_region[region.region_id].passive_tracker_uuv_id,),
            coverage=0.0 if region.status in {"degraded", "uncovered"} else 1.0,
            tracking_quality=0.0,
            handoff_from=region.predecessor_region_id,
            handoff_to=region.successor_region_id,
            degraded_reasons=execution.degradation.reasons
            if execution.degradation.degraded
            else (),
            plan_revision=execution.execution_revision,
        )
        for region in execution.regions
    )


def build_uuv_only_frame(
    *,
    snapshot: MissionSnapshot,
    mission: ExecutableMissionPlan | None = None,
    events: Sequence[RuntimeEvent] = (),
    prediction_grids: Sequence[PredictionGrid] = (),
    candidate_regions: Mapping[str, object] | None = None,
    situation: SituationSnapshot | None = None,
    map_bounds_xy: Sequence[float] | None = None,
    frame_id: int | None = None,
    physics_step_s: int = 5,
    run_phase: Literal[
        "created",
        "bootstrap_planning",
        "awaiting_retry",
        "running",
        "completed",
        "stopping",
        "stopped",
        "failed",
    ] = "running",
    planning: PlanningHealthView | None = None,
    operational_stage_flags: Sequence[OperationalStage] = (),
    llm_thinking: str | None = None,
    llm_thinking_trigger: str | None = None,
    thinking_summary: OperationalThinkingSummary | None = None,
    execution_snapshot: OperationalExecutionSnapshot | None = None,
) -> OperationalFrame:
    """Build the strict UUV-only projection from an immutable mission snapshot."""
    if situation is not None:
        return build_operational_frame(
            situation,
            plan=None,
            ledger_tail=(),
            events=events,
            metrics=(),
            map_bounds_xy=map_bounds_xy,
            frame_id=frame_id,
            physics_step_s=physics_step_s,
            mission_snapshot=snapshot,
            mission=mission,
            execution_snapshot=execution_snapshot,
            prediction_grids=prediction_grids,
            candidate_regions=candidate_regions,
            uuv_only=True,
            run_phase=run_phase,
            planning=planning,
            operational_stage_flags=operational_stage_flags,
            llm_thinking=llm_thinking,
            llm_thinking_trigger=llm_thinking_trigger,
        )
    bounds = _map_bounds(map_bounds_xy)
    return OperationalFrame(
        scenario_id=snapshot.scenario_id,
        frame_id=snapshot.sim_time_s if frame_id is None else frame_id,
        sim_time_s=snapshot.sim_time_s,
        physics_step_s=physics_step_s,
        plan_version=snapshot.plan_revision,
        run_phase=run_phase,
        operational_stage_flags=tuple(operational_stage_flags),
        llm_thinking=thinking_summary.summary if thinking_summary is not None else llm_thinking,
        llm_thinking_trigger=(
            thinking_summary.trigger if thinking_summary is not None else llm_thinking_trigger
        ),
        llm_thinking_epoch_id=(
            thinking_summary.epoch_id if thinking_summary is not None else None
        ),
        llm_thinking_source_event_ids=(
            thinking_summary.source_event_ids if thinking_summary is not None else ()
        ),
        uuv_only=True,
        map_bounds=bounds,
        planning=planning,
        execution=_build_execution_view(
            execution_snapshot,
            current_sim_time_s=int(snapshot.sim_time_s),
        ),
        execution_consistency=(
            FrameConsistencyReport(
                valid=True,
                execution_revision=execution_snapshot.execution_revision,
                source_snapshot_revision=execution_snapshot.source_snapshot_revision,
            )
            if execution_snapshot is not None
            else None
        ),
        events=tuple(_build_event_view(event) for event in events),
        prediction_grids=_build_prediction_grid_views(prediction_grids),
        regional_missions=(
            _execution_regional_mission_views(execution_snapshot)
            if execution_snapshot is not None
            else tuple(
                view.model_copy(update={"carrier_task_id": None})
                for view in _build_regional_mission_views(
                    snapshot,
                    mission,
                    candidate_regions or {},
                )
            )
        ),
        carrier_missions=(),
        mission_events=_build_mission_event_views(events),
        uuv_mission_modes={
            uuv_id: mode.value for uuv_id, mode in sorted(snapshot.uuv_modes.items())
        },
        uuv_resources=_build_uuv_resource_views(snapshot),
    )


def _build_uuv_resource_views(
    snapshot: MissionSnapshot | None,
) -> tuple[UUVResourceView, ...]:
    if snapshot is None:
        return ()
    return tuple(
        UUVResourceView(
            uuv_id=resource.uuv_id,
            carrier_id=resource.carrier_id,
            mileage_m=resource.mileage_m,
            energy_fraction=resource.energy_fraction,
            healthy=resource.healthy,
            capability_active=resource.capability_active,
            deployment_state=resource.deployment_state,
            resource_episode=resource.resource_episode,
        )
        for resource in (
            snapshot.uuv_resources[uuv_id]
            for uuv_id in sorted(snapshot.uuv_resources)
        )
    )


def _build_prediction_grid_views(
    grids: Sequence[PredictionGrid],
) -> tuple[PredictionGridView, ...]:
    return tuple(
        PredictionGridView(
            target_id=grid.target_id,
            revision=grid.revision,
            origin=Point2D(x=grid.origin[0], y=grid.origin[1]),
            cell_size_m=grid.cell_size_m,
            centerline_region_ids=grid.centerline_region_ids,
            cells=tuple(
                PredictionGridCellView(
                    region_id=cell.region_id,
                    target_id=cell.target_id,
                    revision=cell.revision,
                    grid_x=cell.grid_x,
                    grid_y=cell.grid_y,
                    bounds=MapBounds(
                        min_x=cell.min_x,
                        min_y=cell.min_y,
                        max_x=cell.max_x,
                        max_y=cell.max_y,
                    ),
                    probability=cell.probability,
                    first_entry_s=cell.first_entry_s,
                    last_exit_s=cell.last_exit_s,
                    imm_model_probabilities=dict(
                        sorted(cell.imm_model_probabilities.items())
                    ),
                    covariance_summary=cell.covariance_summary,
                    intent_label=cell.intent_label,
                    intent_confidence=cell.intent_confidence,
                )
                for cell in sorted(grid.cells, key=lambda item: item.region_id)
            ),
        )
        for grid in sorted(grids, key=lambda item: (item.target_id, item.revision))
    )


def _build_regional_mission_views(
    snapshot: MissionSnapshot | None,
    mission: ExecutableMissionPlan | None,
    candidate_regions: Mapping[str, object],
) -> tuple[RegionalMissionView, ...]:
    if snapshot is None:
        return ()
    candidates = _flatten_candidate_regions(candidate_regions)
    batches = {
        batch.candidate_id: batch
        for batch in (mission.batches if mission is not None else ())
    }
    views: list[RegionalMissionView] = []
    for region in sorted(snapshot.regions, key=lambda item: item.region_id):
        if "region_cap_not_selected" in region.degraded_reasons:
            continue
        candidate = candidates.get(region.region_id)
        batch = batches.get(region.region_id)
        geometry: tuple[Point2D, ...] = ()
        cell_ids: tuple[str, ...] = ()
        entry_s = batch.entry_s if batch is not None else 0
        exit_s = batch.exit_s if batch is not None else 1
        if isinstance(candidate, RegionalMissionCandidate):
            geometry = tuple(Point2D(x=x, y=y) for x, y in candidate.perimeter_points)
            cell_ids = candidate.cell_ids
            entry_s = candidate.time_window.start_s
            exit_s = candidate.time_window.end_s
        elif isinstance(candidate, MissionCandidate):
            geometry = tuple(Point2D(x=x, y=y) for x, y in candidate.perimeter_points)
            entry_s = candidate.entry_s
            exit_s = candidate.exit_s
        elif region.region_polygon:
            geometry = tuple(
                Point2D(x=x, y=y) for x, y in region.region_polygon
            )
        elif batch is not None:
            points = tuple(
                point
                for point in (batch.deployment_point, batch.recovery_point)
                if point is not None
            )
            geometry = tuple(Point2D(x=x, y=y) for x, y in points)
        views.append(
            RegionalMissionView(
                region_id=region.region_id,
                target_id=region.target_id,
                cell_ids=cell_ids,
                geometry=geometry,
                entry_s=entry_s,
                exit_s=max(entry_s + 1, exit_s),
                lifecycle=region.lifecycle.value,
                active_scan_uuv_ids=region.active_scan_uuv_ids,
                passive_track_uuv_ids=region.passive_track_uuv_ids,
                reserve_uuv_ids=region.reserve_uuv_ids,
                coverage=region.coverage,
                tracking_quality=region.tracking_quality,
                handoff_from=region.handoff_from,
                handoff_to=region.handoff_to,
                carrier_task_id=region.carrier_task_id,
                carrier_id=batch.carrier_id if batch is not None else None,
                degraded_reasons=region.degraded_reasons,
                plan_revision=region.plan_revision,
            )
        )
    return tuple(views)


def _flatten_candidate_regions(candidate_regions: Mapping[str, object]) -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in candidate_regions.items():
        if isinstance(value, (RegionalMissionCandidate, MissionCandidate)):
            flattened[str(key)] = value
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for candidate in value:
                if isinstance(candidate, (RegionalMissionCandidate, MissionCandidate)):
                    flattened[candidate.candidate_id] = candidate
    return flattened


def _build_carrier_mission_views(
    snapshot: MissionSnapshot | None,
) -> tuple[CarrierMissionView, ...]:
    if snapshot is None:
        return ()
    return tuple(
        CarrierMissionView(
            carrier_id=carrier.carrier_id,
            role=carrier.role,
            home_battle_group_id=carrier.home_battle_group_id,
            mission_type=carrier.mission_type,
            route_status=carrier.route_status.value,
            route=tuple(Point2D(x=x, y=y) for x, y in carrier.route_xy),
            stop_ids=carrier.stop_ids,
            onboard_uuv_ids=carrier.onboard_uuv_ids,
            ready_uuv_ids=carrier.ready_uuv_ids,
            reserved_uuv_ids=carrier.reserved_uuv_ids,
            recoverable_uuv_ids=carrier.recoverable_uuv_ids,
        )
        for carrier in (
            snapshot.carrier_missions[carrier_id]
            for carrier_id in sorted(snapshot.carrier_missions)
        )
    )


def _build_mission_event_views(
    events: Sequence[RuntimeEvent],
) -> tuple[MissionEventView, ...]:
    return tuple(
        MissionEventView(
            event_id=event.event_id,
            sim_time_s=event.sim_time_s,
            event_type=event.event_type,
            level=event.level,
            entity_id=event.entity_id,
            payload=cast(dict[str, Any], event.model_dump(mode="json")["payload"]),
        )
        for event in sorted(events, key=lambda item: (item.sim_time_s, item.event_id))
    )


def build_region_timeline(
    plan: TrackingPlan | None,
    sim_time_s: int,
    communication_links: Sequence[CommunicationLinkView] = (),
) -> tuple[RegionTimelineView, ...]:
    """Derive the operator-safe regional Gantt rows from a tracking plan."""
    if plan is None:
        return ()
    rows: list[RegionTimelineView] = []
    for regional_plan in plan.regional_plans.values():
        cells = {cell.region_id: cell for cell in regional_plan.cells}
        for task in regional_plan.tasks:
            cell = cells.get(task.region_id)
            if cell is None:
                continue
            rows.append(
                _region_timeline_row(
                    regional_plan,
                    task,
                    cell,
                    sim_time_s,
                    communication_links,
                )
            )
    return tuple(sorted(rows, key=lambda row: (row.start_offset_s, row.region_id)))


def _region_timeline_row(
    regional_plan: TargetRegionPlan,
    task: RegionTask,
    cell: RegionCell,
    sim_time_s: int,
    communication_links: Sequence[CommunicationLinkView],
) -> RegionTimelineView:
    start_offset = float(task.active_window.start_s - sim_time_s)
    end_offset = float(task.active_window.end_s - sim_time_s)
    uuv_ids = tuple(sorted(task.assigned_uuv_ids))
    uuv_assignments = tuple(
        RegionAssignmentView(
            platform_id=platform_id,
            platform_kind="uuv",
            role=(
                task.uuv_roles[index]
                if index < len(task.uuv_roles)
                else "passive_tracker"
            ),
            start_offset_s=start_offset,
            end_offset_s=end_offset,
            sonar_mode=task.current_sonar_mode,
        )
        for index, platform_id in enumerate(uuv_ids)
    )
    link_refs = {
        tuple(reference.split("->", maxsplit=1))
        for reference in task.communication_links
        if "->" in reference
    }
    links = tuple(
        link
        for link in communication_links
        if (link.source_id, link.target_id) in link_refs
    )
    evidence_ids = tuple(
        sorted(set(regional_plan.evidence_ids) | set(cell.evidence_ids) | set(task.evidence_ids))
    )
    return RegionTimelineView(
        region_id=task.region_id,
        target_id=regional_plan.target_id,
        center=Point2D(x=cell.center_xy[0], y=cell.center_xy[1]),
        bounds=MapBounds(
            min_x=cell.min_x,
            min_y=cell.min_y,
            max_x=cell.max_x,
            max_y=cell.max_y,
        ),
        start_offset_s=start_offset,
        end_offset_s=end_offset,
        status=task.assignment_status,
        coverage_mode=task.coverage_mode,
        priority=task.priority,
        occupancy_likelihood=cell.occupancy_likelihood,
        uuv_assignments=uuv_assignments,
        communication_links=links,
        handoff_from=task.predecessor_region_id,
        handoff_to=task.successor_region_id,
        evidence_ids=evidence_ids,
        degraded_reasons=tuple(sorted(task.degraded_reasons)),
        plan_revision=max(regional_plan.plan_revision, task.plan_revision),
    )


def _build_scheme_view(
    scheme: OperationalScheme | None, sim_time_s: int
) -> OperationalSchemeView | None:
    if scheme is None or not scheme.valid_from_s <= sim_time_s < scheme.valid_until_s:
        return None
    return OperationalSchemeView(
        scheme_id=scheme.scheme_id,
        version=scheme.version,
        valid_from_s=scheme.valid_from_s,
        valid_until_s=scheme.valid_until_s,
        target_priorities=dict(sorted(scheme.target_priorities.items())),
        minimum_quality=dict(sorted(scheme.minimum_quality.items())),
        constraints=tuple(sorted(scheme.constraints)[:_MAX_SCHEME_CONSTRAINTS]),
    )


def _build_intelligence_views(
    reports: Sequence[IntelligenceReport], sim_time_s: int
) -> tuple[IntelligenceView, ...]:
    current = sorted(
        (
            report
            for report in reports
            if report.issued_at_s <= sim_time_s < report.valid_until_s
        ),
        key=lambda report: (report.issued_at_s, report.report_id),
        reverse=True,
    )[:_MAX_INTELLIGENCE_VIEWS]
    return tuple(
        IntelligenceView(
            report_id=report.report_id,
            source=report.source,
            target_id=report.target_id,
            confidence=report.confidence,
            issued_at_s=report.issued_at_s,
            valid_until_s=report.valid_until_s,
            content_summary=(
                report.content_summary[:_MAX_INTELLIGENCE_SUMMARY_LENGTH]
                if report.content_summary is not None
                else None
            ),
        )
        for report in current
    )


def _build_uuv_view(
    state: UUVState,
    plan: TrackingPlan | None,
    *,
    breadcrumbs: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    active_pingers: set[str] | None = None,
    reserved_uuvs: set[str] | None = None,
    platform_state: UUVPlatformState | None = None,
    connected_peer_ids: Sequence[str] = (),
    map_bounds: MapBounds = DEFAULT_MAP_BOUNDS,
) -> UUVView:
    waypoints = plan.waypoints_by_member.get(state.uuv_id, ()) if plan is not None else ()
    current_waypoint = _clip_point(waypoints[0].x, waypoints[0].y, map_bounds) if waypoints else None
    trail = breadcrumbs.get(state.uuv_id, ()) if breadcrumbs is not None else ()
    return UUVView(
        uuv_id=state.uuv_id,
        status=state.status,
        deployment_state=state.deployment_state,
        physically_exposed=state.physically_exposed,
        display_opacity=state.display_opacity,
        position=_clip_point(state.position_xy[0], state.position_xy[1], map_bounds),
        heading_rad=state.heading_rad,
        sensor_heading_rad=state.sensor_heading_rad,
        speed_mps=state.speed_mps,
        energy_fraction=state.energy_fraction,
        remaining_range_m=state.remaining_range_m,
        group_id=state.group_id,
        current_waypoint=current_waypoint,
        breadcrumb=tuple(_clip_point(x, y, map_bounds) for x, y in trail),
        sensor_mode=(
            "active"
            if state.uuv_id in (active_pingers or set())
            else state.sensor_mode
        ),
        reserved=state.reserved or state.uuv_id in (reserved_uuvs or set()),
        passive_range_m=(
            platform_state.capability.sonar.passive_range_m
            if platform_state is not None
            else None
        ),
        active_range_m=(
            min(
                platform_state.capability.sonar.active_source_range_m,
                platform_state.capability.sonar.active_receive_range_m,
            )
            if platform_state is not None
            else None
        ),
        active_capable=(
            platform_state.capability.sonar.active_capable
            if platform_state is not None
            else False
        ),
        is_group_leader=(platform_state.is_group_leader if platform_state is not None else False),
        master_connected=(platform_state.master_connected if platform_state is not None else False),
        connected_peer_ids=tuple(sorted(connected_peer_ids)),
        communication_status=(
            "carrier"
            if platform_state is not None and platform_state.master_connected
            else "relay"
            if connected_peer_ids
            else "disconnected"
        ),
        tracked_target_id=state.group_id,
    )


def _build_carrier_view(
    carrier: CarrierState | None,
    support_radius_m: float | None = None,
    map_bounds: MapBounds = DEFAULT_MAP_BOUNDS,
) -> CarrierView | None:
    if carrier is None:
        return None
    return CarrierView(
        carrier_id=carrier.carrier_id,
        role=carrier.role,
        position=_clip_point(carrier.position_xy[0], carrier.position_xy[1], map_bounds),
        heading_rad=carrier.heading_rad,
        speed_mps=carrier.speed_mps,
        status=carrier.status,
        onboard_uuv_ids=tuple(sorted(carrier.onboard_uuv_ids)),
        deployed_uuv_ids=tuple(sorted(carrier.deployed_uuv_ids)),
        returning_uuv_ids=tuple(sorted(carrier.returning_uuv_ids)),
        support_radius_m=support_radius_m,
    )


def _build_carrier_views(
    snapshot: SituationSnapshot,
    map_bounds: MapBounds,
) -> tuple[CarrierView, ...]:
    """Project every carrier while retaining the primary compatibility field."""
    carrier_states = snapshot.carriers or (
        (snapshot.carrier,) if snapshot.carrier is not None else ()
    )
    support_radii = {
        carrier.carrier_id: carrier.support_radius_m
        for carrier in (
            snapshot.platform_snapshot.carriers
            if snapshot.platform_snapshot is not None
            and snapshot.platform_snapshot.carriers
            else (snapshot.platform_snapshot.carrier,)
            if snapshot.platform_snapshot is not None
            else ()
        )
    }
    views: list[CarrierView] = []
    for carrier in sorted(carrier_states, key=lambda item: item.carrier_id):
        view = _build_carrier_view(
            carrier,
            support_radii.get(carrier.carrier_id),
            map_bounds,
        )
        if view is not None:
            views.append(view)
    return tuple(views)


def _build_platform_views(
    snapshot: PlatformSnapshot | None,
    map_bounds: MapBounds = DEFAULT_MAP_BOUNDS,
) -> tuple[
    tuple[CommunicationLinkView, ...],
    dict[str, tuple[str, ...]],
]:
    """Project carrier and UUV connectivity into public link views."""
    if snapshot is None:
        return (), {}

    carrier_states = snapshot.carriers or (snapshot.carrier,)
    carrier_ids = {carrier.carrier_id for carrier in carrier_states}
    platform_positions: dict[str, tuple[float, float]] = {
        carrier.carrier_id: carrier.position_xy for carrier in carrier_states
    }
    platform_positions.update(
        {state.platform_id: state.position_xy for state in snapshot.roster.uuvs}
    )
    connected_raw = {
        frozenset((link.source_id, link.target_id)): link
        for link in snapshot.communication_links
    }
    candidates: list[
        tuple[str, str, Literal["surface", "acoustic"], float, float]
    ] = []
    uuv_states = tuple(snapshot.roster.uuvs)
    for carrier in carrier_states:
        for state in uuv_states:
            candidates.append(
                (
                    carrier.carrier_id,
                    state.platform_id,
                    "surface",
                    min(
                        carrier.support_radius_m,
                        state.capability.communications.surface_range_m,
                    ),
                    _distance(carrier.position_xy, state.position_xy),
                )
            )
    for index, left in enumerate(carrier_states):
        for right in carrier_states[index + 1 :]:
            candidates.append(
                (
                    left.carrier_id,
                    right.carrier_id,
                    "surface",
                    min(left.support_radius_m, right.support_radius_m),
                    _distance(left.position_xy, right.position_xy),
                )
            )
    for index, left in enumerate(uuv_states):
        for right in uuv_states[index + 1 :]:
            candidates.append(
                (
                    left.platform_id,
                    right.platform_id,
                    "acoustic",
                    min(
                        left.capability.communications.acoustic_range_m,
                        right.capability.communications.acoustic_range_m,
                    ),
                    _distance(left.position_xy, right.position_xy),
                )
            )

    link_views: list[CommunicationLinkView] = []
    peers: dict[str, set[str]] = {}
    for source_id, target_id, medium, limit, distance_m in candidates:
        raw = connected_raw.get(frozenset((source_id, target_id)))
        connected = raw is not None or distance_m <= limit
        status: Literal["connected", "disconnected"] = (
            "connected" if connected else "disconnected"
        )
        if connected:
            peers.setdefault(source_id, set()).add(target_id)
            peers.setdefault(target_id, set()).add(source_id)
        link_views.append(
            CommunicationLinkView(
                source_id=source_id,
                target_id=target_id,
                medium=medium,
                distance_m=distance_m,
                limit_m=limit,
                status=status,
                relay=connected and (source_id in carrier_ids or target_id in carrier_ids),
            )
        )

    return (
        tuple(sorted(link_views, key=lambda link: (link.source_id, link.target_id))),
        {platform_id: tuple(sorted(peer_ids)) for platform_id, peer_ids in peers.items()},
    )


def _build_known_submarine_estimate(
    *,
    contact_id: str,
    position_xy: tuple[float, float] | None,
    plan: TrackingPlan | None,
    intent_hypotheses: Mapping[str, IntentHypothesis] | None,
    predictions: Mapping[str, PredictedTrackRef] | None,
    accepted_predictions: Mapping[str, AcceptedPrediction] | None,
    live_authoritative: bool,
    execution_snapshot: OperationalExecutionSnapshot | None,
    adversary_summary: AdversaryOperationalSummary | None,
    map_bounds: MapBounds,
) -> TargetEstimateView:
    """Project an already identified submarine before tracking reports exist."""
    assert position_xy is not None
    heading = adversary_summary.heading if adversary_summary is not None else 0.0
    return TargetEstimateView(
        target_id=contact_id,
        mean=_clip_point(position_xy[0], position_xy[1], map_bounds),
        covariance_ellipse=CovarianceEllipse(
            semimajor_m=25.0,
            semiminor_m=12.0,
            rotation_rad=heading or 0.0,
        ),
        intent=_build_intent(plan, contact_id, intent_hypotheses),
        prediction=_build_prediction(
            predictions.get(contact_id) if predictions else None,
            accepted=(accepted_predictions or {}).get(contact_id),
            execution_snapshot=execution_snapshot,
            live_authoritative=live_authoritative,
        ),
        quality=EstimateQualityView(
            quality_score=1.0,
            estimated_rmse_m=0.0,
            fim_min_eigenvalue=1.0,
            fim_condition=1.0,
        ),
        classification="submarine",
        detection_range_m=(
            adversary_summary.detection_range_m
            if adversary_summary is not None
            else 1.0
        ),
        detected_platform_ids=(
            adversary_summary.detected_platform_ids
            if adversary_summary is not None
            else ()
        ),
    )


def _build_planned_assignment_views(
    snapshot: MissionSnapshot | None,
) -> tuple[PlannedAssignmentView, ...]:
    """Project planned regional ownership independently of exposed groups."""
    if snapshot is None:
        return ()
    owners = {
        uuv_id: resource.carrier_id
        for uuv_id, resource in snapshot.uuv_resources.items()
        if resource.carrier_id is not None
    }
    views: list[PlannedAssignmentView] = []
    for region in sorted(snapshot.regions, key=lambda item: item.region_id):
        uuv_ids = tuple(
            sorted(
                {
                    *region.active_scan_uuv_ids,
                    *region.passive_track_uuv_ids,
                    *region.reserve_uuv_ids,
                }
            )
        )
        if not uuv_ids:
            continue
        carrier_ids = tuple(
            sorted({owners[uuv_id] for uuv_id in uuv_ids if uuv_id in owners})
        )
        if len(carrier_ids) != 1:
            continue
        lifecycle = region.lifecycle.value
        status: Literal["planned", "transporting", "ready_to_deploy"] = (
            "planned"
            if lifecycle == "PLANNED"
            else "transporting"
            if lifecycle == "CARRIER_DEPLOYING"
            else "ready_to_deploy"
        )
        views.append(
            PlannedAssignmentView(
                target_id=region.target_id,
                region_id=region.region_id,
                uuv_ids=uuv_ids,
                carrier_id=carrier_ids[0],
                plan_version=region.plan_revision,
                status=status,
            )
        )
    return tuple(views)


def _build_brain_views(
    snapshot: SituationSnapshot,
    events: Sequence[RuntimeEvent],
    plan: TrackingPlan | None,
    *,
    llm_paused: bool = False,
    role_activity: Mapping[str, BrainActivityRecord] | None = None,
    configured_roles: Sequence[Literal["master", "slave", "adversary"]] = (
        "master",
        "slave",
        "adversary",
    ),
) -> tuple[BrainView, ...]:
    """Expose only configured, ledger-backed role activity."""
    del events, plan
    activity = role_activity or {}
    configured = set(configured_roles)
    roles: tuple[Literal["master", "slave", "adversary"], ...] = (
        "master",
        "slave",
        "adversary",
    )
    views: list[BrainView] = []
    for role in roles:
        record = activity.get(role)
        if llm_paused:
            status: Literal[
                "unconfigured", "ready", "running", "succeeded", "degraded", "failed", "paused"
            ] = "paused"
            message = "等待 LLM 重连"
            operation = None
            sim_time_s = snapshot.sim_time_s
            evidence_platform_ids: tuple[str, ...] = ()
        elif role not in configured:
            status = "unconfigured"
            message = "角色未配置"
            operation = None
            sim_time_s = None
            evidence_platform_ids = ()
        elif record is None:
            status = "ready"
            message = "已配置，等待首次调用"
            operation = None
            sim_time_s = None
            evidence_platform_ids = ()
        else:
            status = record.status
            message = record.message
            operation = record.operation
            sim_time_s = record.sim_time_s
            evidence_platform_ids = record.evidence_platform_ids
        views.append(
            BrainView(
                brain_id={
                    "master": "carrier-master",
                    "slave": "group-slave",
                    "adversary": "target-adversary",
                }[role],
                role=role,
                status=status,
                last_update_s=sim_time_s,
                message=message,
                connected_platform_ids=evidence_platform_ids,
                operation=operation,
                evidence_platform_ids=evidence_platform_ids,
            )
        )
    return tuple(views)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _build_estimate(
    report: GroupReport,
    plan: TrackingPlan | None,
    *,
    intent_hypotheses: Mapping[str, IntentHypothesis] | None = None,
    predictions: Mapping[str, PredictedTrackRef] | None = None,
    accepted_predictions: Mapping[str, AcceptedPrediction] | None = None,
    live_authoritative: bool = False,
    execution_snapshot: OperationalExecutionSnapshot | None = None,
    prediction_diffs: Mapping[str, TrajectoryDiffResult] | None = None,
    prediction_gates: Mapping[str, TrajectoryDiffGateState] | None = None,
    world_model_forecasts: Mapping[str, WorldModelForecast] | None = None,
    events: Sequence[RuntimeEvent] = (),
    classification: str = "unknown",
    last_ping_s: int | None = None,
    adversary_summary: AdversaryOperationalSummary | None = None,
    map_bounds: MapBounds = DEFAULT_MAP_BOUNDS,
) -> TargetEstimateView:
    belief = report.belief
    if len(belief.mean) < 2:
        raise ValueError(
            f"belief mean for target {belief.target_id!r} has fewer than 2 components"
        )
    p00, p01, p10, p11 = _position_covariance(belief)
    classification_label = cast(
        Literal["submarine", "decoy", "unknown"],
        classification
        if classification in {"submarine", "decoy", "unknown"}
        else "unknown",
    )
    return TargetEstimateView(
        target_id=belief.target_id,
        mean=_clip_point(belief.mean[0], belief.mean[1], map_bounds),
        covariance_ellipse=_covariance_to_ellipse(p00, p01, p10, p11),
        intent=_build_intent(plan, belief.target_id, intent_hypotheses),
        prediction=_build_prediction(
            predictions.get(belief.target_id) if predictions else None,
            accepted=(accepted_predictions or {}).get(belief.target_id),
            execution_snapshot=execution_snapshot,
            live_authoritative=live_authoritative,
            diff=(prediction_diffs or {}).get(belief.target_id),
            gate=(prediction_gates or {}).get(belief.target_id),
            events=events,
        ),
        world_model=_build_world_model_forecast(
            (world_model_forecasts or {}).get(belief.target_id),
            map_bounds,
        ),
        quality=EstimateQualityView(
            quality_score=report.quality.window_mean,
            # The RMS position error proxy is the covariance trace:
            # E[||x - mean||^2] = p00 + p11 for the 2D Gaussian.
            estimated_rmse_m=math.sqrt(max(0.0, p00 + p11)),
            fim_min_eigenvalue=belief.fim_min_eigenvalue,
            fim_condition=_finite_condition(belief.fim_condition),
        ),
        classification=classification_label,
        last_ping_s=last_ping_s,
        detection_range_m=(
            adversary_summary.detection_range_m
            if adversary_summary is not None
            else 1.0
        ),
        detected_platform_ids=(
            adversary_summary.detected_platform_ids
            if adversary_summary is not None
            else ()
        ),
    )


def _build_adversary_view(summary: AdversaryOperationalSummary) -> AdversaryView:
    return AdversaryView(
        target_id=summary.target_id,
        sim_time_s=summary.sim_time_s,
        detection_range_m=summary.detection_range_m,
        detected_platform_ids=summary.detected_platform_ids,
        trigger_event_ids=summary.trigger_event_ids,
        decision_id=summary.decision_id,
        maneuver=summary.maneuver,
        intent=summary.intent,
        segment=summary.segment,
        speed_mps=summary.speed,
        heading_rad=summary.heading,
        decoy_count=summary.decoy_count,
        confidence=summary.confidence,
        rationale=summary.rationale,
        communications_discipline=summary.communications_discipline,
        decision_status=summary.decision_status,
        escape_region_id=summary.escape_region_id,
        decision_source=summary.decision_source,
        guidance_id=summary.guidance_id,
        guidance_waypoint_xy=(
            Point2D(
                x=summary.guidance_waypoint_xy[0],
                y=summary.guidance_waypoint_xy[1],
            )
            if summary.guidance_waypoint_xy is not None
            else None
        ),
        guidance_speed_mps=summary.guidance_speed_mps,
        guidance_heading_rad=summary.guidance_heading_rad,
        guidance_valid_until_s=summary.guidance_valid_until_s,
        degraded_reason=summary.degraded_reason,
    )


def _position_covariance(belief: TargetBelief) -> tuple[float, float, float, float]:
    """The leading 2x2 position block of the belief covariance."""
    covariance = belief.covariance
    if len(covariance) < 2 or len(covariance[0]) < 2 or len(covariance[1]) < 2:
        raise ValueError(
            f"belief covariance for target {belief.target_id!r} is smaller than 2x2"
        )
    return (covariance[0][0], covariance[0][1], covariance[1][0], covariance[1][1])


def _covariance_to_ellipse(
    p00: float, p01: float, p10: float, p11: float
) -> CovarianceEllipse:
    """Render a 2x2 covariance as ellipse axes (standard deviations) plus rotation.

    The rotation is the eigenvector angle of the largest eigenvalue;
    degenerate covariances are floored to a tiny positive circle so the
    frame contract's strictly-positive-axis validation always passes.
    """
    trace = p00 + p11
    determinant = p00 * p11 - p01 * p10
    half_diff = math.sqrt(max(0.0, trace * trace / 4.0 - determinant))
    lambda_max = trace / 2.0 + half_diff
    lambda_min = max(0.0, trace / 2.0 - half_diff)
    floor = _MIN_AXIS_M * _MIN_AXIS_M
    semimajor = math.sqrt(max(floor, lambda_max))
    semiminor = math.sqrt(max(floor, lambda_min))
    rotation = 0.5 * math.atan2(2.0 * p01, p00 - p11)
    return CovarianceEllipse(
        semimajor_m=semimajor, semiminor_m=semiminor, rotation_rad=rotation
    )


def _build_intent(
    plan: TrackingPlan | None,
    target_id: str,
    intent_hypotheses: Mapping[str, IntentHypothesis] | None = None,
) -> IntentView:
    """Intent label from the plan's per-target intent references.

    The builder inputs carry no confidence for the referenced label, so the
    view renders with confidence ``0.0``; an absent or unknown reference
    renders as ``unknown``.
    """
    hypothesis = intent_hypotheses.get(target_id) if intent_hypotheses else None
    raw = (
        hypothesis.label
        if hypothesis is not None
        else plan.intent_refs.get(target_id)
        if plan is not None
        else None
    )
    label: IntentLabel = cast(
        IntentLabel, raw if raw is not None and raw in _INTENT_LABELS else "unknown"
    )
    return IntentView(
        label=label,
        confidence=hypothesis.confidence if hypothesis is not None else 0.0,
        alternatives=(dict(hypothesis.alternatives) if hypothesis is not None else {}),
        ranked_motives=(hypothesis.ranked_motives if hypothesis is not None else ()),
    )


def _build_prediction(
    prediction: PredictedTrackRef | None,
    *,
    accepted: AcceptedPrediction | None = None,
    execution_snapshot: OperationalExecutionSnapshot | None = None,
    live_authoritative: bool = False,
    diff: TrajectoryDiffResult | None = None,
    gate: TrajectoryDiffGateState | None = None,
    events: Sequence[RuntimeEvent] = (),
) -> PredictionCorridorView | None:
    if accepted is not None:
        if accepted.health.status == "unavailable":
            return None
        if execution_snapshot is not None and accepted.prediction is not None:
            if accepted.prediction.target_id != execution_snapshot.target_id:
                raise ValueError("accepted prediction target ID must match execution target")
            if accepted.prediction.prediction_id != execution_snapshot.prediction_id:
                raise ValueError("accepted prediction ID must match execution prediction ID")
    if live_authoritative and accepted is None:
        return None
    if execution_snapshot is not None:
        authoritative = execution_snapshot.prediction
        points_xy = authoritative.centerline_xy
        radius_m = authoritative.corridor_radius_m
        prediction_id = authoritative.prediction_id
        prediction_revision = authoritative.prediction_revision
        origin_sim_time_s = authoritative.origin_sim_time_s
        horizon_s = authoritative.times_s[-1] - authoritative.origin_sim_time_s
        sample_step_s = authoritative.times_s[0] - authoritative.origin_sim_time_s
        leading_probability = max(authoritative.model_probabilities.values())
        health = _execution_prediction_health(execution_snapshot)
        assessed_confidence: tuple[float, ...] = ()
    else:
        if accepted is not None:
            if accepted.prediction is None:
                return None
            prediction = accepted.prediction
        if prediction is None:
            return None
        points_xy = prediction.points_xy
        radius_m = prediction.corridor_radius_m
        prediction_id = prediction.prediction_id
        prediction_revision = max(1, round(prediction.sim_time_s))
        origin_sim_time_s = float(prediction.sim_time_s)
        horizon_s = prediction.horizon_s
        sample_step_s = prediction.sample_step_s
        leading_probability = max(
            prediction.imm_model_probabilities.values(), default=1.0
        )
        health = _accepted_prediction_health(accepted, prediction)
        assessed_confidence = prediction.point_confidence
    points = tuple(Point2D(x=x, y=y) for x, y in points_xy)
    point_confidence = assessed_confidence or _prediction_point_confidences(
        radius_m,
        len(points),
        leading_probability,
    )
    if len(points) != len(radius_m) or len(points) != len(point_confidence):
        raise ValueError(
            "prediction centerline, radius, and point confidence lengths must match"
        )
    return PredictionCorridorView(
        prediction_id=prediction_id,
        prediction_revision=prediction_revision,
        origin_sim_time_s=origin_sim_time_s,
        health=health,
        horizon_s=horizon_s,
        sample_step_s=sample_step_s,
        centerline_xy=points,
        radius_m=radius_m,
        point_confidence=point_confidence,
        diff=_build_prediction_diff(diff, gate, events),
    )


def _accepted_prediction_health(
    accepted: AcceptedPrediction | None,
    prediction: PredictedTrackRef,
) -> PredictionHealthView:
    if accepted is not None:
        return PredictionHealthView.model_validate(accepted.health.model_dump())
    regime = cast(
        Literal["imm", "bspline", "short_history", "boundary_recovery"],
        (
            prediction.prediction_regime
            if prediction.prediction_regime
            in {"imm", "bspline", "short_history", "boundary_recovery"}
            else "short_history"
        ),
    )
    return PredictionHealthView(
        status="valid" if regime == "imm" else "degraded",
        regime=regime,
        reason_codes=(
            ()
            if prediction.prediction_regime == regime
            else ("prediction_health_not_assessed",)
        ),
        source_track_age_s=0.0,
        clipped_point_fraction=0.0,
        maximum_radius_m=max(prediction.corridor_radius_m, default=0.0),
        raw_prediction_id=prediction.prediction_id,
    )


def _execution_prediction_health(
    execution: OperationalExecutionSnapshot,
) -> PredictionHealthView:
    prediction = execution.prediction
    prediction_degraded = (
        "prediction" in execution.degradation.failed_components
        or prediction.prediction_regime != "imm"
    )
    reasons = tuple(
        reason
        for reason in execution.degradation.reasons
        if not reason.startswith("region_generation_mode:")
    )
    return PredictionHealthView(
        status="degraded" if prediction_degraded else "valid",
        regime=prediction.prediction_regime,
        reason_codes=reasons,
        source_track_age_s=max(
            0.0,
            float(execution.source_sim_time_s)
            - float(execution.target_track.sim_time_s),
        ),
        clipped_point_fraction=min(
            1.0,
            len(prediction.clipping_records)
            / max(1, len(prediction.centerline_xy)),
        ),
        maximum_radius_m=max(prediction.corridor_radius_m, default=0.0),
        raw_prediction_id=prediction.prediction_id,
    )


def _build_world_model_forecast(
    forecast: WorldModelForecast | None,
    map_bounds: MapBounds,
) -> WorldModelForecastView | None:
    if forecast is None:
        return None
    return WorldModelForecastView(
        model_kind=forecast.model_kind,
        model_version=forecast.model_version,
        control_authority=forecast.control_authority,
        as_of_s=forecast.as_of_s,
        source_prediction_id=forecast.source_prediction_id,
        source_observation_ids=forecast.source_observation_ids,
        source_observability_event_ids=forecast.source_observability_event_ids,
        source_plan_revision=forecast.source_plan_revision,
        data_status=forecast.data_status.value,
        trajectory_fallback_used=forecast.trajectory_fallback_used,
        imm_model_probabilities=dict(sorted(forecast.imm_model_probabilities.items())),
        horizons=tuple(
            WorldModelHorizonView(
                name=horizon.name.value,
                start_offset_s=horizon.start_offset_s,
                end_offset_s=horizon.end_offset_s,
                sample_count=horizon.sample_count,
                covered=horizon.covered,
            )
            for horizon in forecast.horizons
        ),
        events=tuple(
            WorldModelEventView(
                event_id=event.event_id,
                event_type=event.event_type.value,
                horizon=event.horizon.value,
                predicted_time_s=event.predicted_time_s,
                time_to_event_s=event.time_to_event_s,
                predicted_position=_clip_point(
                    event.predicted_position_xy[0],
                    event.predicted_position_xy[1],
                    map_bounds,
                ),
                confidence=event.confidence,
                level=event.level,
                rule_id=event.rule_id,
                summary=event.summary,
                evidence=tuple(
                    WorldModelEvidenceView(
                        key=item.key,
                        source=item.source,
                        value=item.value,
                        threshold=item.threshold,
                        unit=item.unit,
                        description=item.description,
                    )
                    for item in event.evidence
                ),
            )
            for event in forecast.events
        ),
        warnings=forecast.warnings,
    )


def _prediction_point_confidences(
    radii_m: Sequence[float],
    point_count: int,
    leading_model_probability: float,
) -> tuple[float, ...]:
    """Convert IMM mode probability and covariance spread to point confidence."""
    if point_count <= 0:
        return ()
    positive_radii = tuple(float(radius) for radius in radii_m if radius > 0.0)
    base_radius = min(positive_radii, default=1.0)
    leading_probability = max(0.0, min(1.0, float(leading_model_probability)))
    fallback_radius = positive_radii[-1] if positive_radii else base_radius
    confidences: list[float] = []
    for index in range(point_count):
        radius = float(radii_m[index]) if index < len(radii_m) else fallback_radius
        relative_peak_density = 1.0 if radius <= 0.0 else (base_radius / radius) ** 2
        confidences.append(
            max(0.0, min(1.0, leading_probability * relative_peak_density))
        )
    return tuple(confidences)


def _build_prediction_diff(
    diff: TrajectoryDiffResult | None,
    gate: TrajectoryDiffGateState | None,
    events: Sequence[RuntimeEvent],
) -> PredictionDiffView | None:
    if diff is None:
        return None
    if diff.gate_transition == "none":
        state = "stable" if diff.status == "comparable" else "unavailable"
    else:
        state = diff.gate_transition
    confirmed_event = next(
        (
            event
            for event in sorted(
                events,
                key=lambda item: (item.sim_time_s, item.event_id),
                reverse=True,
            )
            if event.event_type == "target_intent_changed"
            and event.entity_id == diff.target_id
            and event.payload.get("diff_id") == diff.diff_id
        ),
        None,
    )
    confirmed_intent = None
    resulting_plan_revision = None
    if confirmed_event is not None:
        label = confirmed_event.payload.get("label")
        if isinstance(label, str):
            confirmed_intent = label
        revision = confirmed_event.payload.get("resulting_plan_revision")
        if isinstance(revision, int) and not isinstance(revision, bool) and revision >= 1:
            resulting_plan_revision = revision
    return PredictionDiffView(
        diff_id=diff.diff_id,
        state=state,
        status=diff.status,
        reason=diff.reason,
        absolute_rms_m=diff.absolute_rms_m,
        normalized_rms=diff.normalized_rms,
        absolute_floor_m=diff.absolute_floor_m,
        normalized_threshold=diff.normalized_threshold,
        consecutive_count=diff.consecutive_count,
        confirmation_cycles=diff.confirmation_cycles,
        previous_prediction_id=diff.previous_prediction_id,
        current_prediction_id=diff.current_prediction_id,
        leading_model_changed=diff.leading_model_changed,
        js_distance=diff.js_distance,
        suspicion_event_id=None if gate is None else gate.suspicion_event_id,
        confirmed_intent=confirmed_intent,
        resulting_plan_revision=resulting_plan_revision,
    )


def _build_group(report: GroupReport) -> GroupView:
    quality = report.quality
    return GroupView(
        group_id=report.group_id,
        target_id=report.target_id,
        member_ids=tuple(sorted(report.member_ids)),
        quality=GroupQualityView(
            instant=quality.instant,
            window_mean=quality.window_mean,
            ewma=quality.ewma,
            components=dict(sorted(quality.components.items())),
            hard_guard_reasons=tuple(sorted(quality.hard_guard_reasons)),
        ),
    )


def _build_regional_plan_views(
    plan: TrackingPlan | None,
    reports: Sequence[GroupReport],
    sim_time_s: int,
    events: Sequence[RuntimeEvent] = (),
) -> dict[str, RegionalPlanView]:
    if plan is None:
        return {}
    regional_plans = getattr(plan, "regional_plans", {}) or {}
    groups = tuple(_build_group(report) for report in reports)
    views: dict[str, RegionalPlanView] = {}
    for target_id, regional_plan in sorted(regional_plans.items()):
        ordered_tasks = sorted(
            (
                task
                for task in regional_plan.tasks
                if "region_cap_not_selected" not in task.degraded_reasons
            ),
            key=lambda task: (task.active_window.start_s, task.region_id),
        )
        cells_by_id = {cell.region_id: cell for cell in regional_plan.cells}
        regions = tuple(
            _build_region_task_view(
                task,
                cells_by_id[task.region_id],
                regional_plan=regional_plan,
                index=index,
                groups=groups,
                sim_time_s=sim_time_s,
            )
            for index, task in enumerate(ordered_tasks)
            if task.region_id in cells_by_id
        )
        current, next_region = _handoff_regions(ordered_tasks, sim_time_s)
        views[target_id] = RegionalPlanView(
            target_id=regional_plan.target_id,
            prediction_id=regional_plan.prediction_id,
            revision=regional_plan.plan_revision,
            cell_size_m=regional_plan.cell_size_m,
            grid_spec=regional_plan.grid_spec,
            evidence_ids=tuple(sorted(regional_plan.evidence_ids)),
            current_handoff_region_id=current,
            next_handoff_region_id=next_region,
            causal_event_ids=_regional_causal_event_ids(
                plan, target_id, groups, events
            ),
            llm_hashes=plan.regional_llm_hashes.get(target_id),
            regions=regions,
        )
    return views


def _regional_causal_event_ids(
    plan: TrackingPlan,
    target_id: str,
    groups: Sequence[GroupView],
    events: Sequence[RuntimeEvent],
) -> tuple[str, ...]:
    """Return plan triggers that are causally relevant to one target."""
    group_ids = {group.group_id for group in groups if group.target_id == target_id}
    return tuple(
        event.event_id
        for event in sorted(events, key=lambda item: (item.sim_time_s, item.event_id))
        if event.event_id in plan.trigger_event_ids
        and plan.valid_from_s <= event.sim_time_s <= plan.valid_until_s
        and (
            event.entity_id == target_id
            or event.entity_id in group_ids
            or event.payload.get("target_id") == target_id
        )
    )


def _handoff_regions(
    tasks: Sequence[RegionTask], sim_time_s: int
) -> tuple[str | None, str | None]:
    """Return the task active now and its declared or chronological successor."""
    current = next(
        (
            task
            for task in tasks
            if task.active_window.start_s <= sim_time_s < task.active_window.end_s
        ),
        None,
    )
    if current is not None:
        if current.successor_region_id is not None:
            return current.region_id, current.successor_region_id
        following = next(
            (task for task in tasks if task.active_window.start_s >= current.active_window.end_s),
            None,
        )
        return current.region_id, following.region_id if following is not None else None
    following = next((task for task in tasks if task.active_window.start_s > sim_time_s), None)
    return None, following.region_id if following is not None else None


def _build_region_task_view(
    task: Any,
    cell: Any,
    *,
    regional_plan: TargetRegionPlan,
    index: int,
    groups: Sequence[GroupView],
    sim_time_s: int,
) -> RegionTaskView:
    group = _group_for_region_task(task, groups)
    effect = _build_tracking_effect(task, group, sim_time_s)
    predecessor_ids = (
        (task.predecessor_region_id,)
        if task.predecessor_region_id is not None
        else tuple(cell.predecessor_region_ids)
    )
    successor_ids = (
        (task.successor_region_id,)
        if task.successor_region_id is not None
        else tuple(cell.successor_region_ids)
    )
    return RegionTaskView(
        region_id=task.region_id,
        display_name=f"region_{index + 1}",
        target_id=task.target_id,
        geometry=(
            Point2D(x=cell.min_x, y=cell.min_y),
            Point2D(x=cell.max_x, y=cell.min_y),
            Point2D(x=cell.max_x, y=cell.max_y),
            Point2D(x=cell.min_x, y=cell.max_y),
        ),
        grid_x=cell.grid_x,
        grid_y=cell.grid_y,
        start_time_s=task.active_window.start_s,
        end_time_s=task.active_window.end_s,
        visit_window_index=task.visit_window_index,
        visit_window=(
            cell.visit_windows[task.visit_window_index]
            if task.visit_window_index < len(cell.visit_windows)
            else task.active_window
        ),
        predecessor_region_ids=predecessor_ids,
        successor_region_ids=successor_ids,
        assigned_uuv_ids=tuple(sorted(task.assigned_uuv_ids)),
        tracking_mode=task.tracking_mode,
        uuv_roles=tuple(task.uuv_roles),
        sonar_policy=task.sonar_policy,
        communication=task.communication,
        communication_links=tuple(sorted(task.communication_links)),
        group_id=group.group_id if group is not None else None,
        status=effect.status,
        degraded_reasons=tuple(sorted(task.degraded_reasons)),
        evidence_ids=tuple(
            sorted(set(regional_plan.evidence_ids) | set(cell.evidence_ids) | set(task.evidence_ids))
        ),
        revision=max(regional_plan.plan_revision, task.plan_revision),
        effect=effect,
    )


def _group_for_region_task(
    task: Any, groups: Sequence[GroupView]
) -> GroupView | None:
    assigned = set(task.assigned_uuv_ids)
    candidates = [
        group
        for group in groups
        if group.target_id == task.target_id
        and assigned.issubset(set(group.member_ids))
    ]
    return min(candidates, key=lambda group: group.group_id) if candidates else None


def _build_tracking_effect(
    task: Any, group: GroupView | None, sim_time_s: int
) -> TrackingEffectView:
    assigned_count = len(task.assigned_uuv_ids)
    status: Literal["planned", "active", "handoff_ready", "degraded", "uncovered"]
    if assigned_count == 0:
        status = "uncovered"
        coverage_ratio = 0.0
        quality_score = 0.0
        handoff_progress = 0.0
    else:
        quality_score = group.quality.ewma if group is not None else 0.0
        coverage_ratio = 1.0 if group is not None else 0.0
        in_window = task.active_window.start_s <= sim_time_s < task.active_window.end_s
        if task.degraded_reasons or (
            group is not None and group.quality.hard_guard_reasons
        ):
            status = "degraded"
        elif task.assignment_status == "handed_off":
            status = "handoff_ready"
        elif in_window:
            status = "active"
        else:
            status = "planned"
        handoff_progress = 1.0 if status == "handoff_ready" else 0.0
    return TrackingEffectView(
        status=status,
        coverage_ratio=coverage_ratio,
        quality_score=quality_score,
        handoff_progress=handoff_progress,
        quality_source="group_quality_proxy",
        hard_guard_reasons=(
            tuple(sorted(task.degraded_reasons))
            + (group.quality.hard_guard_reasons if group is not None else ())
        ),
    )


def _build_ray(
    observation: BearingObservation,
    by_uuv: Mapping[str, UUVState],
    map_bounds: MapBounds = DEFAULT_MAP_BOUNDS,
) -> BearingRayView:
    origin = by_uuv.get(observation.uuv_id)
    if origin is None:
        raise ValueError(
            f"bearing ray {observation.observation_id!r} references unknown "
            f"UUV {observation.uuv_id!r}"
        )
    return BearingRayView(
        observation_id=observation.observation_id,
        uuv_id=observation.uuv_id,
        target_id=observation.target_id,
        origin=_clip_point(origin.position_xy[0], origin.position_xy[1], map_bounds),
        azimuth_rad=observation.azimuth_rad,
        variance_rad2=observation.variance_rad2,
        confidence=observation.detection_confidence,
    )


def _build_plan_view(plan: TrackingPlan) -> PlanView:
    affected = sorted(plan.target_priorities)
    if not affected:
        affected = sorted(plan.member_ids_by_target)
    changes: list[str] = []
    if plan.diff is not None:
        changes.extend(
            f"{group} adds {', '.join(members)}"
            for group, members in sorted(plan.diff.members_added.items())
        )
        changes.extend(
            f"{group} removes {', '.join(members)}"
            for group, members in sorted(plan.diff.members_removed.items())
        )
    return PlanView(
        plan_id=plan.plan_id,
        version=plan.revision,
        status=plan.status,
        concept=plan.concept,
        reason="",
        affected_targets=tuple(affected),
        group_changes=tuple(sorted(changes)),
        valid_from_s=plan.valid_from_s,
        valid_until_s=plan.valid_until_s if plan.valid_until_s > 0 else None,
        segment_plan=_segment_labels(plan),
    )


def _segment_labels(plan: TrackingPlan) -> tuple[str, ...]:
    if plan.segment_plan is None:
        return ()
    return tuple(
        f"{segment.group_id}:{segment.start_s}-{segment.end_s}"
        for segment in plan.segment_plan.segments
    )


def _build_ledger_view(decision: DecisionRecord) -> LedgerView:
    if decision.final_plan_id is None:
        outcome: Literal["committed", "degraded", "rejected"] = "rejected"
    elif any(report.degraded for report in decision.verification_records):
        outcome = "degraded"
    else:
        outcome = "committed"
    final_version = (
        decision.final_plan_diff.to_revision
        if decision.final_plan_diff is not None
        else _plan_revision_from_id(decision.final_plan_id)
    )
    return LedgerView(
        decision_id=decision.decision_id,
        sim_time_s=decision.sim_time_s,
        outcome=outcome,
        trigger_event_ids=decision.trigger_event_ids,
        evidence_ids=decision.input_evidence_ids,
        final_plan_id=decision.final_plan_id,
        final_plan_version=final_version,
    )


def _build_plan_timeline(
    decisions: Sequence[DecisionRecord],
    events: Sequence[RuntimeEvent],
    *,
    active_plan: TrackingPlan | None = None,
    current_sim_time_s: int | None = None,
) -> tuple[PlanTimelineView, ...]:
    """Project durable decisions into factor-left/result-right replay rows."""
    events_by_id = {event.event_id: event for event in events}
    rows: list[PlanTimelineView] = []
    for decision in sorted(decisions, key=lambda item: (item.sim_time_s, item.decision_id)):
        factors: list[TimelineFactorView] = []
        for event_id in decision.trigger_event_ids:
            event = events_by_id.get(event_id)
            factors.append(
                TimelineFactorView(
                    kind="event",
                    ref_id=event_id,
                    label=(
                        f"{event.event_type} / {event.level.value}"
                        if event is not None
                        else "trigger event"
                    ),
                    detail=_event_detail(event),
                )
            )
        for evidence_id in decision.input_evidence_ids[:12]:
            factors.append(
                TimelineFactorView(
                    kind="evidence", ref_id=evidence_id, label="估计证据"
                )
            )
        for directive in decision.expert_inputs:
            factors.append(
                TimelineFactorView(
                    kind="directive",
                    ref_id=directive.directive_id,
                    label="人工方案反馈",
                    detail=directive.raw_text[:180],
                )
            )
        if not factors:
            factors.append(
                TimelineFactorView(
                    kind="evidence",
                    ref_id=decision.snapshot_hash or decision.decision_id,
                    label="当前态势快照",
                )
            )
        plan = None
        if decision.final_plan_id is not None:
            diff = decision.final_plan_diff
            version = (
                diff.to_revision
                if diff is not None
                else (_plan_revision_from_id(decision.final_plan_id) or 1)
            )
            changes = () if diff is None else _timeline_group_changes(diff)
            plan = TimelinePlanView(
                plan_id=decision.final_plan_id,
                version=version,
                status="active",
                summary=(diff.summary if diff is not None and diff.summary else "方案已提交并进入执行"),
                group_changes=changes,
            )
        rows.append(
            PlanTimelineView(
                adjustment_id=decision.decision_id,
                sim_time_s=decision.sim_time_s,
                factors=tuple(factors[:24]),
                plan=plan,
            )
        )
    if active_plan is not None and not any(
        row.plan is not None and row.plan.version == active_plan.revision for row in rows
    ):
        source_factors = tuple(
            TimelineFactorView(kind="event", ref_id=event_id, label="方案触发事件")
            for event_id in active_plan.trigger_event_ids
            if event_id in events_by_id
        )
        if not source_factors:
            source_factors = (
                TimelineFactorView(
                    kind="evidence",
                    ref_id=active_plan.plan_id,
                    label="当前活动方案",
                ),
            )
        rows.append(
            PlanTimelineView(
                adjustment_id=f"active-plan:{active_plan.plan_id}",
                sim_time_s=(
                    current_sim_time_s
                    if current_sim_time_s is not None
                    else active_plan.valid_from_s
                ),
                factors=source_factors,
                plan=TimelinePlanView(
                    plan_id=active_plan.plan_id,
                    version=active_plan.revision,
                    status="active",
                    summary="当前活动方案",
                ),
            )
        )
    return tuple(rows[-80:])


def _plan_revision_from_id(plan_id: str | None) -> int | None:
    """Read the revision suffix emitted by the deterministic plan ID."""
    if not isinstance(plan_id, str) or ":plan:" not in plan_id:
        return None
    suffix = plan_id.rsplit(":plan:", 1)[-1]
    return int(suffix) if suffix.isdigit() and int(suffix) > 0 else None


def _timeline_group_changes(diff: Any) -> tuple[str, ...]:
    changes = [
        f"{group} 增加 {', '.join(members)}"
        for group, members in sorted(diff.members_added.items())
    ]
    changes.extend(
        f"{group} 退出 {', '.join(members)}"
        for group, members in sorted(diff.members_removed.items())
    )
    changes.extend(f"航路更新 {member}" for member in diff.waypoints_changed)
    return tuple(changes)


def _event_detail(event: RuntimeEvent | None) -> str:
    if event is None:
        return ""
    message = event.payload.get("message")
    if isinstance(message, str):
        return message[:180]
    hypothesis = event.payload.get("hypothesis")
    return str(hypothesis)[:180] if hypothesis is not None else ""


def _build_event_view(event: RuntimeEvent) -> EventView:
    return EventView(
        event_id=event.event_id,
        sim_time_s=event.sim_time_s,
        event_type=event.event_type,
        level=event.level,
        entity_id=event.entity_id,
        message=(
            str(event.payload["message"])
            if isinstance(event.payload.get("message"), str)
            else ""
        ),
    )


def _map_bounds(map_bounds_xy: Sequence[float] | None) -> MapBounds:
    if map_bounds_xy is None:
        return DEFAULT_MAP_BOUNDS
    values = tuple(float(value) for value in map_bounds_xy)
    if len(values) != 4:
        raise ValueError("map_bounds_xy must contain min_x, max_x, min_y, max_y")
    return MapBounds(min_x=values[0], max_x=values[1], min_y=values[2], max_y=values[3])


def _clip_point(x: float, y: float, bounds: MapBounds = DEFAULT_MAP_BOUNDS) -> Point2D:
    return Point2D(
        x=min(max(x, bounds.min_x), bounds.max_x),
        y=min(max(y, bounds.min_y), bounds.max_y),
    )


def _finite_condition(condition: float) -> float:
    """A finite render of the FIM condition number.

    Non-finite values (the belief default when no FIM estimate exists yet)
    cannot survive a JSONL round trip, so they render as
    ``_FIM_CONDITION_FINITE_CAP``.
    """
    return condition if math.isfinite(condition) else _FIM_CONDITION_FINITE_CAP
