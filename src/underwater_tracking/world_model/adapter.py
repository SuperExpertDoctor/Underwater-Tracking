"""Adapters from the live operational contracts into the rule world model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import cos, isfinite, sin

from underwater_tracking.domain.agent_models import PredictedTrackRef, TrackingPlan
from underwater_tracking.domain.models import (
    DeploymentState,
    GroupReport,
    SituationSnapshot,
    UUVStatus,
)
from underwater_tracking.world_model.config import DEFAULT_WORLD_MODEL_CONFIG
from underwater_tracking.world_model.models import (
    ImmBeliefInput,
    RuleWorldModelConfig,
    RuleWorldModelInput,
    TrackingContextInput,
    TrajectoryForecastInput,
    UuvForecastInput,
    WorldModelForecast,
)
from underwater_tracking.world_model.rules import predict_future_events


PlannedUuvSample = tuple[float, float, float]


def build_world_model_input(
    snapshot: SituationSnapshot,
    prediction: PredictedTrackRef,
    *,
    previous_contact_count: int | None = None,
    association_confidence: float | None = None,
    previous_association_confidence: float | None = None,
    association_entropy: float | None = None,
    previous_association_entropy: float | None = None,
    communication_ok_by_uuv: Mapping[str, bool] | None = None,
    planned_uuv_tracks: Mapping[str, Sequence[PlannedUuvSample]] | None = None,
    active_plan: TrackingPlan | None = None,
) -> RuleWorldModelInput:
    """Build a truth-safe rule input from one public situation snapshot.

    ``planned_uuv_tracks`` is optional and contains absolute
    ``(time_s, x, y)`` samples.  When omitted, the predictor projects each
    UUV at its current public heading and speed.  Association trend values
    are optional because the current ``SituationSnapshot`` contract does not
    yet expose them directly.
    """

    if prediction.target_id == "":
        raise ValueError("prediction target_id must not be empty")
    if not prediction.times_s or not prediction.points_xy:
        raise ValueError("prediction must contain sampled future points")
    report = _group_report(snapshot, prediction.target_id)
    if report is None:
        raise ValueError(f"no group report for target {prediction.target_id!r}")
    position_xy = _belief_position(report)
    velocity_xy = _belief_velocity(report, prediction, position_xy)
    communication = _communication_status(snapshot)
    if communication_ok_by_uuv is not None:
        communication.update(communication_ok_by_uuv)
    planned = (
        dict(planned_uuv_tracks)
        if planned_uuv_tracks is not None
        else planned_uuv_tracks_from_plan(active_plan, as_of_s=float(snapshot.sim_time_s))
    )
    observability_hypotheses, observability_event_ids = _observability_evidence(
        snapshot,
        prediction.target_id,
    )
    member_ids = set(report.member_ids)
    uuvs: list[UuvForecastInput] = []
    for uuv in sorted(snapshot.uuvs, key=lambda item: item.uuv_id):
        if member_ids and uuv.uuv_id not in member_ids:
            continue
        samples = tuple(
            sorted(
                (
                    (float(time_s), float(x), float(y))
                    for time_s, x, y in planned.get(uuv.uuv_id, ())
                ),
                key=lambda item: item[0],
            )
        )
        healthy = (
            uuv.status is not UUVStatus.FAILED
            and uuv.deployment_state is DeploymentState.DEPLOYED
            and uuv.capability.passive_sonar_available
        )
        uuvs.append(
            UuvForecastInput(
                uuv_id=uuv.uuv_id,
                position_xy=(float(uuv.position_xy[0]), float(uuv.position_xy[1])),
                velocity_xy_mps=(
                    float(uuv.speed_mps * cos(uuv.heading_rad)),
                    float(uuv.speed_mps * sin(uuv.heading_rad)),
                ),
                passive_range_m=float(uuv.capability.passive_range_m),
                bearing_variance_rad2=float(uuv.capability.bearing_variance_rad2),
                energy_fraction=float(uuv.energy_fraction),
                healthy=healthy,
                communication_ok=bool(communication.get(uuv.uuv_id, True)),
                state_age_s=0.0,
                planned_times_s=tuple(item[0] for item in samples),
                planned_points_xy=tuple((item[1], item[2]) for item in samples),
            )
        )
    return RuleWorldModelInput(
        scenario_id=snapshot.scenario_id,
        target_id=prediction.target_id,
        as_of_s=float(snapshot.sim_time_s),
        belief=ImmBeliefInput(
            position_xy=position_xy,
            velocity_xy_mps=velocity_xy,
            turn_rate_rad_s=(
                float(report.belief.mean[4]) if len(report.belief.mean) > 4 else 0.0
            ),
            covariance_trace_m2=_position_covariance_trace(report),
            model_probabilities=_normalized_model_probabilities(
                prediction.imm_model_probabilities or report.belief.model_probabilities
            ),
        ),
        trajectory=TrajectoryForecastInput(
            prediction_id=prediction.prediction_id,
            times_s=tuple(float(value) for value in prediction.times_s),
            points_xy=tuple(
                (float(point[0]), float(point[1])) for point in prediction.points_xy
            ),
            corridor_radius_m=tuple(
                float(value) for value in prediction.corridor_radius_m
            ),
            fallback_used=prediction.fallback_used,
            fallback_reason=prediction.fallback_reason,
        ),
        uuvs=tuple(uuvs),
        tracking=TrackingContextInput(
            quality_ewma=float(report.quality.ewma),
            current_contact_count=len(snapshot.contacts),
            previous_contact_count=previous_contact_count,
            association_confidence=association_confidence,
            previous_association_confidence=previous_association_confidence,
            association_entropy=association_entropy,
            previous_association_entropy=previous_association_entropy,
            observability_hypotheses=observability_hypotheses,
        ),
        map_bounds_xy=(
            (
                float(snapshot.map_bounds_xy[0]),
                float(snapshot.map_bounds_xy[1]),
                float(snapshot.map_bounds_xy[2]),
                float(snapshot.map_bounds_xy[3]),
            )
            if snapshot.map_bounds_xy is not None
            else None
        ),
        source_observation_ids=tuple(report.belief.source_observation_ids),
        source_observability_event_ids=observability_event_ids,
        source_plan_revision=active_plan.revision if active_plan is not None else None,
    )


def predict_snapshot_events(
    snapshot: SituationSnapshot,
    prediction: PredictedTrackRef,
    *,
    config: RuleWorldModelConfig = DEFAULT_WORLD_MODEL_CONFIG,
    previous_contact_count: int | None = None,
    association_confidence: float | None = None,
    previous_association_confidence: float | None = None,
    association_entropy: float | None = None,
    previous_association_entropy: float | None = None,
    communication_ok_by_uuv: Mapping[str, bool] | None = None,
    planned_uuv_tracks: Mapping[str, Sequence[PlannedUuvSample]] | None = None,
    active_plan: TrackingPlan | None = None,
) -> WorldModelForecast:
    """Adapt one snapshot and immediately run the deterministic predictor."""

    inputs = build_world_model_input(
        snapshot,
        prediction,
        previous_contact_count=previous_contact_count,
        association_confidence=association_confidence,
        previous_association_confidence=previous_association_confidence,
        association_entropy=association_entropy,
        previous_association_entropy=previous_association_entropy,
        communication_ok_by_uuv=communication_ok_by_uuv,
        planned_uuv_tracks=planned_uuv_tracks,
        active_plan=active_plan,
    )
    return predict_future_events(inputs, config)


def build_world_model_forecasts(
    snapshot: SituationSnapshot,
    predictions: Mapping[str, PredictedTrackRef],
    *,
    config: RuleWorldModelConfig = DEFAULT_WORLD_MODEL_CONFIG,
    active_plan: TrackingPlan | None = None,
) -> dict[str, WorldModelForecast]:
    """Build one deterministic, read-only event forecast per tracked target."""

    report_target_ids = {report.target_id for report in snapshot.group_reports}
    forecasts: dict[str, WorldModelForecast] = {}
    for target_id, prediction in sorted(predictions.items()):
        if target_id not in report_target_ids:
            continue
        forecasts[target_id] = predict_snapshot_events(
            snapshot,
            prediction,
            config=config,
            active_plan=active_plan,
        )
    return forecasts


def planned_uuv_tracks_from_plan(
    plan: TrackingPlan | None,
    *,
    as_of_s: float,
) -> dict[str, tuple[PlannedUuvSample, ...]]:
    """Expose future committed waypoints as absolute samples for projection."""

    if plan is None:
        return {}
    tracks: dict[str, tuple[PlannedUuvSample, ...]] = {}
    for uuv_id, waypoints in sorted(plan.waypoints_by_member.items()):
        samples: list[PlannedUuvSample] = []
        last_time = as_of_s
        for waypoint in waypoints:
            time_s = float(waypoint.arrive_at_s)
            if time_s <= last_time:
                continue
            samples.append((time_s, float(waypoint.x), float(waypoint.y)))
            last_time = time_s
        if samples:
            tracks[uuv_id] = tuple(samples)
    return tracks


def _group_report(snapshot: SituationSnapshot, target_id: str) -> GroupReport | None:
    return next(
        (report for report in snapshot.group_reports if report.target_id == target_id),
        None,
    )


def _communication_status(snapshot: SituationSnapshot) -> dict[str, bool]:
    platform_snapshot = snapshot.platform_snapshot
    if platform_snapshot is None:
        return {}
    return {
        platform.platform_id: bool(platform.master_connected)
        for platform in platform_snapshot.roster.uuvs
    }


def _observability_evidence(
    snapshot: SituationSnapshot,
    target_id: str,
) -> tuple[dict[str, float], tuple[str, ...]]:
    hypotheses: dict[str, float] = {}
    source_ids: list[str] = []
    for runtime_event in snapshot.pending_events:
        if not runtime_event.event_type.startswith("observability_"):
            continue
        raw_events = runtime_event.payload.get("events", ())
        if not isinstance(raw_events, (list, tuple)):
            continue
        for raw in raw_events:
            if not isinstance(raw, Mapping) or raw.get("track_id") != target_id:
                continue
            if bool(raw.get("recovery", False)):
                continue
            hypothesis = raw.get("hypothesis")
            event_id = raw.get("event_id")
            confidence = raw.get("confidence")
            if not isinstance(hypothesis, str) or not hypothesis:
                continue
            if not isinstance(confidence, (int, float)) or not isfinite(float(confidence)):
                continue
            bounded = min(1.0, max(0.0, float(confidence)))
            hypotheses[hypothesis] = max(hypotheses.get(hypothesis, 0.0), bounded)
            if isinstance(event_id, str) and event_id:
                source_ids.append(event_id)
    return dict(sorted(hypotheses.items())), tuple(sorted(set(source_ids)))


def _belief_position(report: GroupReport) -> tuple[float, float]:
    if len(report.belief.mean) < 2:
        raise ValueError("target belief mean must contain x and y")
    position = (float(report.belief.mean[0]), float(report.belief.mean[1]))
    if not all(isfinite(value) for value in position):
        raise ValueError("target belief position must be finite")
    return position


def _belief_velocity(
    report: GroupReport,
    prediction: PredictedTrackRef,
    position_xy: tuple[float, float],
) -> tuple[float, float]:
    if len(report.belief.mean) >= 4:
        velocity = (float(report.belief.mean[2]), float(report.belief.mean[3]))
        if all(isfinite(value) for value in velocity):
            return velocity
    delta_s = float(prediction.times_s[0] - report.belief.sim_time_s)
    if delta_s <= 0.0:
        return 0.0, 0.0
    first = prediction.points_xy[0]
    return (
        float(first[0] - position_xy[0]) / delta_s,
        float(first[1] - position_xy[1]) / delta_s,
    )


def _position_covariance_trace(report: GroupReport) -> float:
    covariance = report.belief.covariance
    diagonal = [
        float(covariance[index][index])
        for index in range(min(2, len(covariance)))
        if index < len(covariance[index])
    ]
    trace = sum(diagonal)
    if not isfinite(trace) or trace < 0.0:
        raise ValueError("target position covariance trace must be finite and non-negative")
    return trace


def _normalized_model_probabilities(raw: Mapping[str, float]) -> dict[str, float]:
    if not raw:
        return {"cv": 1.0, "left_turn": 0.0, "right_turn": 0.0}
    values = {str(name): float(value) for name, value in raw.items()}
    if any(not isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("IMM model probabilities must be finite and non-negative")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("IMM model probabilities must have positive mass")
    normalized = {name: value / total for name, value in values.items()}
    # Make the final value absorb harmless floating-point summation drift.
    last_name = max(normalized)
    normalized[last_name] += 1.0 - sum(normalized.values())
    return dict(sorted(normalized.items()))
