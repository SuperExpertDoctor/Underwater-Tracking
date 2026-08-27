"""Build the authoritative UUV execution snapshot from live public state.

The graph and legacy optimizer still expose audit-shaped models.  This module
is the narrow adapter that turns their public prediction and intent references
into the immutable execution contract consumed by the mission controller and
all live transports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from underwater_tracking.domain.agent_models import IntentHypothesis, PredictedTrackRef
from underwater_tracking.domain.execution_models import (
    DeterministicIntentState,
    ExecutionDegradation,
    GlobalTargetTrackView,
    IMMModelForecast,
    IMMPredictedTrack,
    OperationalExecutionSnapshot,
    PlanSource,
    ReserveUUVState,
    TaskGroupAssignment,
)
from underwater_tracking.domain.mission_models import (
    RegionLifecycle,
    RegionMissionState,
    UUVResourceState,
)
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.intent.deterministic import ConfirmedIntentRevision
from underwater_tracking.planning.dynamic_regions import (
    DynamicRegionChain,
    build_dynamic_region_chain,
)
from underwater_tracking.planning.task_groups import allocate_four_task_groups


def build_execution_snapshot(
    *,
    situation: SituationSnapshot,
    target_track: GlobalTargetTrackView,
    prediction: PredictedTrackRef,
    intent: DeterministicIntentState | ConfirmedIntentRevision | IntentHypothesis | None,
    uuv_resources: Mapping[str, UUVResourceState] | Sequence[UUVResourceState],
    execution_revision: int,
    prediction_revision: int | None = None,
    previous: OperationalExecutionSnapshot | None = None,
    mission_regions: Sequence[RegionMissionState] = (),
    expert_request_version: int = 0,
    plan_source: PlanSource = "deterministic",
) -> OperationalExecutionSnapshot:
    """Build one complete four-region, four-group execution snapshot."""

    if execution_revision < 1:
        raise ValueError("execution_revision must be positive")
    if target_track.target_id != prediction.target_id:
        raise ValueError("target track and prediction targets must match")
    if not target_track.target_id:
        raise ValueError("target track must have a target ID")
    if situation.map_bounds_xy is None:
        raise ValueError("execution snapshot requires map bounds")

    selected_prediction_revision = max(
        1,
        int(prediction_revision or prediction.sim_time_s or 1),
    )
    if previous is not None and prediction.prediction_id == previous.prediction_id:
        selected_prediction_revision = max(
            selected_prediction_revision,
            previous.prediction_revision,
        )
    execution_prediction = _as_imm_prediction(
        prediction,
        target_track,
        prediction_revision=selected_prediction_revision,
    )
    execution_intent = _as_deterministic_intent(
        intent,
        target_id=target_track.target_id,
        prediction_revision=selected_prediction_revision,
        fallback_evidence_ids=(
            *target_track.source_event_ids,
            prediction.prediction_id,
        ),
    )

    previous_chain = _previous_chain(previous)
    chain = build_dynamic_region_chain(
        execution_prediction,
        execution_revision=execution_revision,
        map_bounds_xy=tuple(float(value) for value in situation.map_bounds_xy),
        previous_chain=previous_chain,
    )
    allocation = allocate_four_task_groups(
        chain,
        uuv_resources,
        execution_revision=execution_revision,
        previous_assignments=(previous.task_groups if previous is not None else ()),
    )
    if len(allocation.assignments) != 4:
        raise ValueError(
            "four-region execution requires four complete two-UUV task groups"
        )

    mission_by_region = {region.region_id: region for region in mission_regions}
    bound_regions = tuple(
        region.model_copy(
            update={
                "status": _region_status(
                    mission_by_region.get(region.region_id)
                ),
                "evidence_ids": _unique(
                    *region.evidence_ids,
                    *target_track.source_event_ids,
                ),
            }
        )
        for region in allocation.bound_regions
    )
    regions_by_id = {region.region_id: region for region in bound_regions}
    groups = tuple(
        group.model_copy(
            update={
                "status": _group_status(regions_by_id[group.region_id].status),
                "evidence_ids": _unique(
                    *group.evidence_ids,
                    *target_track.source_event_ids,
                ),
            }
        )
        for group in allocation.assignments
    )
    reserves = _resource_episodes(allocation.reserve_uuvs, uuv_resources)

    current_region_id = _current_region_id(bound_regions)
    current_index = next(
        index
        for index, region in enumerate(bound_regions)
        if region.region_id == current_region_id
    )
    next_region_id = (
        bound_regions[current_index + 1].region_id
        if current_index + 1 < len(bound_regions)
        else current_region_id
    )
    degradation_reasons = list(allocation.degradation_reasons)
    if prediction.fallback_used:
        degradation_reasons.append(
            f"prediction_fallback:{prediction.fallback_reason or 'short_history'}"
        )
    degradation_reasons = list(dict.fromkeys(degradation_reasons))
    evidence_ids = _unique(
        *target_track.source_event_ids,
        *prediction.source_belief_history_ids,
        prediction.prediction_id,
        *execution_intent.evidence_ids,
        *(
            evidence_id
            for region in bound_regions
            for evidence_id in region.evidence_ids
        ),
        *(
            evidence_id
            for group in groups
            for evidence_id in group.evidence_ids
        ),
    )
    valid_from_s = float(situation.sim_time_s)
    valid_until_s = max(
        valid_from_s + 1.0,
        *(float(region.end_s) for region in bound_regions),
    )
    return OperationalExecutionSnapshot(
        scenario_id=situation.scenario_id,
        target_id=target_track.target_id,
        execution_revision=execution_revision,
        source_snapshot_revision=situation.snapshot_revision,
        source_sim_time_s=float(target_track.sim_time_s),
        prediction_revision=selected_prediction_revision,
        prediction_id=execution_prediction.prediction_id,
        intent_revision=execution_intent.intent_revision,
        expert_request_version=expert_request_version,
        generated_at_s=float(situation.sim_time_s),
        valid_from_s=valid_from_s,
        valid_until_s=valid_until_s,
        plan_source=plan_source,
        target_track=target_track,
        prediction=execution_prediction,
        intent=execution_intent,
        regions=bound_regions,
        task_groups=groups,
        reserve_uuvs=reserves,
        current_region_id=current_region_id,
        next_region_id=next_region_id,
        evidence_ids=evidence_ids,
        degradation=ExecutionDegradation(
            status="degraded" if degradation_reasons else "nominal",
            reasons=tuple(degradation_reasons),
            active_plan_preserved=False,
            failed_components=(
                ("prediction",) if prediction.fallback_used else ()
            ),
        ),
        base_execution_revision=(
            previous.execution_revision if previous is not None else None
        ),
    )


def _as_imm_prediction(
    prediction: PredictedTrackRef,
    target_track: GlobalTargetTrackView,
    *,
    prediction_revision: int,
) -> IMMPredictedTrack:
    step_s = max(float(prediction.sample_step_s), 1.0)
    origin_s = max(float(target_track.sim_time_s), 0.0)
    raw_points = tuple(
        (float(point[0]), float(point[1])) for point in prediction.points_xy
    )
    raw_times = tuple(float(value) for value in prediction.times_s)
    if not raw_points:
        count = max(1, int(float(prediction.horizon_s) // step_s))
        raw_points = tuple(
            (
                target_track.position_xy[0] + target_track.velocity_xy[0] * step_s * index,
                target_track.position_xy[1] + target_track.velocity_xy[1] * step_s * index,
            )
            for index in range(1, count + 1)
        )
    points = raw_points
    times: list[float] = []
    previous_time = origin_s
    for index in range(len(points)):
        candidate = raw_times[index] if index < len(raw_times) else previous_time + step_s
        if not isfinite(candidate) or candidate <= previous_time:
            candidate = previous_time + step_s
        times.append(candidate)
        previous_time = candidate

    radii = tuple(
        max(1.0, abs(float(prediction.corridor_radius_m[index])))
        if index < len(prediction.corridor_radius_m)
        and isfinite(float(prediction.corridor_radius_m[index]))
        else 1.0
        for index in range(len(points))
    )
    covariances = tuple(
        _covariance_at(prediction, index, radii[index])
        for index in range(len(points))
    )
    probabilities = _probabilities(prediction.imm_model_probabilities)
    branches = (
        prediction.imm_model_states
        if len(prediction.imm_model_states) == 3
        and {branch.model_name for branch in prediction.imm_model_states}
        == {"CV", "CT_LEFT", "CT_RIGHT"}
        else _default_branches(target_track, probabilities)
    )
    source_ids = _unique(
        *prediction.source_belief_history_ids,
        *target_track.source_event_ids,
    )
    regime = "imm" if prediction.prediction_regime == "imm" else "short_history"
    return IMMPredictedTrack(
        prediction_id=prediction.prediction_id,
        prediction_revision=prediction_revision,
        target_id=target_track.target_id,
        origin_sim_time_s=origin_s,
        times_s=tuple(times),
        centerline_xy=points,
        covariance_xy=covariances,
        corridor_radius_m=radii,
        model_branches=tuple(branches),
        model_probabilities=probabilities,
        clipping_records=_unique(
            *prediction.clipping_records,
            *prediction.imm_clipping_records,
        ),
        source_track_revision=target_track.track_revision,
        source_observation_ids=source_ids,
        prediction_regime=regime,
    )


def _default_branches(
    track: GlobalTargetTrackView,
    probabilities: Mapping[str, float],
) -> tuple[IMMModelForecast, ...]:
    mean = (
        track.position_xy[0],
        track.position_xy[1],
        track.velocity_xy[0],
        track.velocity_xy[1],
        track.turn_rate_rad_s,
    )
    covariance = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(5))
        for row in range(5)
    )
    return tuple(
        IMMModelForecast(
            model_name=name,
            state_mean=mean,
            state_covariance=covariance,
            model_probability=probabilities[name],
            likelihood=1.0,
            source_observation_ids=track.source_event_ids,
        )
        for name in ("CV", "CT_LEFT", "CT_RIGHT")
    )


def _covariance_at(
    prediction: PredictedTrackRef,
    index: int,
    radius: float,
) -> tuple[float, float, float, float]:
    if index < len(prediction.imm_covariance_xy):
        value = prediction.imm_covariance_xy[index]
        if len(value) == 4:
            values = tuple(float(item) for item in value)
            if all(isfinite(item) for item in values):
                return values  # type: ignore[return-value]
    variance = max(1.0, radius * radius)
    return (variance, 0.0, 0.0, variance)


def _probabilities(raw: Mapping[str, float]) -> dict[str, float]:
    result = {"CV": 0.0, "CT_LEFT": 0.0, "CT_RIGHT": 0.0}
    for key, value in raw.items():
        normalized = str(key).casefold().replace("-", "_")
        destination = {
            "cv": "CV",
            "constant_velocity": "CV",
            "left": "CT_LEFT",
            "left_turn": "CT_LEFT",
            "ct_left": "CT_LEFT",
            "right": "CT_RIGHT",
            "right_turn": "CT_RIGHT",
            "ct_right": "CT_RIGHT",
        }.get(normalized)
        if destination is not None and isfinite(float(value)) and float(value) >= 0:
            result[destination] += float(value)
    total = sum(result.values())
    if total <= 1e-12:
        return {key: 1.0 / 3.0 for key in result}
    return {key: value / total for key, value in result.items()}


def _as_deterministic_intent(
    intent: DeterministicIntentState | ConfirmedIntentRevision | IntentHypothesis | None,
    *,
    target_id: str,
    prediction_revision: int,
    fallback_evidence_ids: Sequence[str],
) -> DeterministicIntentState:
    if isinstance(intent, DeterministicIntentState):
        return intent.model_copy(
            update={
                "target_id": target_id,
                "prediction_revision": prediction_revision,
                "evidence_ids": _unique(*intent.evidence_ids, *fallback_evidence_ids),
            }
        )
    if isinstance(intent, ConfirmedIntentRevision):
        label = intent.intent_label
        confidence = intent.confidence
        revision = intent.intent_revision
        rule_version = intent.rule_version
        features = _numeric_mapping(intent.features)
        thresholds = _numeric_mapping(intent.thresholds)
        evidence_ids = intent.evidence_ids
    elif isinstance(intent, IntentHypothesis):
        label = intent.label
        confidence = intent.confidence
        revision = 1
        rule_version = "deterministic-intent-v1"
        features = {}
        thresholds = {}
        evidence_ids = intent.evidence_ids
    else:
        label = "unknown"
        confidence = 0.0
        revision = 1
        rule_version = "deterministic-intent-v1"
        features = {}
        thresholds = {}
        evidence_ids = ()
    return DeterministicIntentState(
        target_id=target_id,
        intent_label=label,
        confidence=confidence,
        intent_revision=max(1, revision),
        prediction_revision=prediction_revision,
        rule_version=rule_version,
        features=features,
        thresholds=thresholds,
        evidence_ids=_unique(*evidence_ids, *fallback_evidence_ids),
    )


def _numeric_mapping(values: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(numeric):
            result[str(key)] = numeric
    return result


def _previous_chain(
    previous: OperationalExecutionSnapshot | None,
) -> DynamicRegionChain | None:
    if previous is None:
        return None
    return DynamicRegionChain(
        target_id=previous.target_id,
        prediction_id=previous.prediction_id,
        execution_revision=previous.execution_revision,
        geometry_revision=max(
            region.geometry_revision for region in previous.regions
        ),
        regions=previous.regions,
    )


def _resource_episodes(
    reserves: Sequence[ReserveUUVState],
    resources: Mapping[str, UUVResourceState] | Sequence[UUVResourceState],
) -> tuple[ReserveUUVState, ...]:
    by_id = (
        dict(resources)
        if isinstance(resources, Mapping)
        else {resource.uuv_id: resource for resource in resources}
    )
    return tuple(
        reserve.model_copy(
            update={
                "resource_episode": by_id[reserve.uuv_id].resource_episode
                if reserve.uuv_id in by_id
                else reserve.resource_episode
            }
        )
        for reserve in reserves
    )


def _region_status(region: RegionMissionState | None) -> str:
    if region is None:
        return "planned"
    return {
        RegionLifecycle.PLANNED: "planned",
        RegionLifecycle.CARRIER_DEPLOYING: "prepositioning",
        RegionLifecycle.ACTIVE_SCAN: "active",
        RegionLifecycle.PASSIVE_TRACK: "passive",
        RegionLifecycle.HANDOFF_PENDING: "handoff_pending",
        RegionLifecycle.TRACKING_COMPLETED: "monitoring_complete",
        RegionLifecycle.CARRIER_RECOVERY: "handoff_completed",
        RegionLifecycle.RECOVERED: "monitoring_complete",
        RegionLifecycle.DEGRADED: "degraded",
        RegionLifecycle.UNCOVERED: "uncovered",
    }.get(region.lifecycle, "planned")


def _group_status(region_status: str) -> str:
    return {
        "active": "active",
        "passive": "active",
        "handoff_pending": "handoff_pending",
        "monitoring_complete": "complete",
        "degraded": "degraded",
        "uncovered": "degraded",
    }.get(region_status, "prepositioning")


def _current_region_id(regions: Sequence[Any]) -> str:
    for region in regions:
        if region.status in {"active", "passive", "handoff_pending"}:
            return region.region_id
    return regions[0].region_id


def _unique(*values: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value and value.strip()))


__all__ = ["build_execution_snapshot"]
