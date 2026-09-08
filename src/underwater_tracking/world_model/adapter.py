"""Adapters from the live operational contracts into the rule world model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import cos, isfinite, log, sin

from underwater_tracking.domain.agent_models import PredictedTrackRef, TrackingPlan
from underwater_tracking.domain.execution_models import OperationalExecutionSnapshot
from underwater_tracking.domain.prediction_models import AcceptedPrediction
from underwater_tracking.tracking.public_estimate import assess_public_estimate
from underwater_tracking.domain.models import (
    ContactClassification,
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
    DataStatus,
    HorizonCoverage,
)
from underwater_tracking.world_model.rules import predict_future_events


PlannedUuvSample = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ContactAssociationSnapshot:
    """Public contact-count and association ambiguity proxy for one target."""

    contact_count: int
    target_confidence: float | None
    normalized_entropy: float | None


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
    source_plan_revision: int | None = None,
    execution_snapshot: OperationalExecutionSnapshot | None = None,
    accepted_prediction: AcceptedPrediction | None = None,
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
    health = assess_public_estimate(report.belief, snapshot.sim_time_s)
    if health.status in {"expired", "unavailable"}:
        raise ValueError(";".join(health.reason_codes))
    if prediction.source_track_revision != report.belief.track_revision:
        raise ValueError("prediction_source_track_revision_mismatch")
    if (
        prediction.prediction_revision is None
        or prediction.valid_until_s is None
        or prediction.generated_at_s is None
    ):
        raise ValueError("prediction_provenance_missing")
    if snapshot.sim_time_s >= prediction.valid_until_s:
        raise ValueError("prediction_expired")
    if prediction.generated_at_s > snapshot.sim_time_s:
        raise ValueError("prediction_generation_time_invalid")
    if accepted_prediction is not None and (
        accepted_prediction.health.status == "unavailable"
        or accepted_prediction.prediction is None
        or accepted_prediction.prediction.prediction_id != prediction.prediction_id
    ):
        raise ValueError("accepted_prediction_unavailable_or_mismatched")
    owner_id = None
    region = None
    source_reasons = list(health.reason_codes)
    if accepted_prediction is not None and accepted_prediction.health.status == "degraded":
        source_reasons.extend(
            accepted_prediction.health.reason_codes or ("accepted_prediction_degraded",)
        )
    member_ids = set(report.member_ids)
    if execution_snapshot is not None:
        if (
            execution_snapshot.target_id != prediction.target_id
            or execution_snapshot.prediction_id != prediction.prediction_id
            or execution_snapshot.prediction_revision != prediction.prediction_revision
            or execution_snapshot.target_track.track_revision != prediction.source_track_revision
        ):
            raise ValueError("execution_prediction_identity_mismatch")
        if snapshot.sim_time_s >= execution_snapshot.valid_until_s:
            raise ValueError("execution_context_expired")
        source_plan_revision = execution_snapshot.execution_revision
        owner_id = execution_snapshot.tracking_control.tracking_owner_group_id
        owner = next(
            (
                group
                for group in execution_snapshot.task_groups
                if group.group_instance_id == owner_id
            ),
            None,
        )
        member_ids = set(owner.member_uuv_ids) if owner else set()
        region = next(
            (
                region
                for region in execution_snapshot.regions
                if owner is not None and region.region_id == owner.region_id
            ),
            None,
        )
        if owner is None:
            source_reasons.append("tracking_owner_unassigned")
    position_xy = _belief_position(report)
    velocity_xy = _belief_velocity(report, prediction, position_xy)
    communication = _communication_status(snapshot)
    if communication_ok_by_uuv is not None:
        communication.update(communication_ok_by_uuv)
    planned = (
        {}
        if execution_snapshot is not None
        else (
            dict(planned_uuv_tracks)
            if planned_uuv_tracks is not None
            else planned_uuv_tracks_from_plan(active_plan, as_of_s=float(snapshot.sim_time_s))
        )
    )
    observability_hypotheses, observability_event_ids = _observability_evidence(
        snapshot,
        prediction.target_id,
    )
    uuvs: list[UuvForecastInput] = []
    for uuv in sorted(snapshot.uuvs, key=lambda item: item.uuv_id):
        if uuv.uuv_id not in member_ids:
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
                state_time_s=float(snapshot.sim_time_s),
                planned_times_s=tuple(item[0] for item in samples),
                planned_points_xy=tuple((item[1], item[2]) for item in samples),
            )
        )
    return RuleWorldModelInput(
        scenario_id=snapshot.scenario_id,
        target_id=prediction.target_id,
        as_of_s=float(snapshot.sim_time_s),
        source_track_revision=prediction.source_track_revision,
        prediction_revision=prediction.prediction_revision,
        last_observed_at_s=report.belief.last_observed_at_s,
        generated_at_s=float(snapshot.sim_time_s),
        valid_until_s=min(
            float(health.valid_until_s),
            prediction.valid_until_s,
            execution_snapshot.valid_until_s
            if execution_snapshot is not None
            else prediction.valid_until_s,
        ),
        source_prediction_id=prediction.prediction_id,
        owner_group_id=owner_id,
        source_group_id=report.group_id,
        region_id=region.region_id if region else None,
        region_geometry_revision=region.geometry_revision if region else None,
        task_region_bounds_xy=(
            min(x for x, y in region.geometry),
            max(x for x, y in region.geometry),
            min(y for x, y in region.geometry),
            max(y for x, y in region.geometry),
        )
        if region
        else None,
        source_status="degraded" if health.status == "degraded" or source_reasons else "current",
        source_reason_codes=tuple(source_reasons),
        belief=ImmBeliefInput(
            position_xy=position_xy,
            velocity_xy_mps=velocity_xy,
            turn_rate_rad_s=(float(report.belief.mean[4]) if len(report.belief.mean) > 4 else 0.0),
            covariance_trace_m2=_position_covariance_trace(report),
            model_probabilities=_normalized_model_probabilities(
                prediction.imm_model_probabilities or report.belief.model_probabilities
            ),
        ),
        trajectory=TrajectoryForecastInput(
            prediction_id=prediction.prediction_id,
            times_s=tuple(
                float(value) for value in prediction.times_s if value > snapshot.sim_time_s
            ),
            points_xy=tuple(
                (float(point[0]), float(point[1]))
                for time, point in zip(prediction.times_s, prediction.points_xy, strict=True)
                if time > snapshot.sim_time_s
            ),
            corridor_radius_m=tuple(
                float(value)
                for time, value in zip(
                    prediction.times_s, prediction.corridor_radius_m, strict=True
                )
                if time > snapshot.sim_time_s
            ),
            fallback_used=prediction.fallback_used,
            fallback_reason=prediction.fallback_reason,
            prediction_regime=prediction.prediction_regime,
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
            if snapshot.map_bounds_xy is not None and execution_snapshot is None
            else None
        ),
        source_observation_ids=tuple(report.belief.source_observation_ids),
        source_observability_event_ids=observability_event_ids,
        source_plan_revision=(
            source_plan_revision
            if source_plan_revision is not None
            else active_plan.revision
            if active_plan is not None
            else None
        ),
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
    source_plan_revision: int | None = None,
    execution_snapshot: OperationalExecutionSnapshot | None = None,
    accepted_prediction: AcceptedPrediction | None = None,
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
        source_plan_revision=source_plan_revision,
        execution_snapshot=execution_snapshot,
        accepted_prediction=accepted_prediction,
    )
    return predict_future_events(inputs, config)


def build_world_model_forecasts(
    snapshot: SituationSnapshot,
    predictions: Mapping[str, PredictedTrackRef],
    *,
    config: RuleWorldModelConfig = DEFAULT_WORLD_MODEL_CONFIG,
    active_plan: TrackingPlan | None = None,
    previous_tracking_by_target: Mapping[str, ContactAssociationSnapshot] | None = None,
    source_plan_revision: int | None = None,
    execution_snapshot: OperationalExecutionSnapshot | None = None,
    accepted_predictions: Mapping[str, AcceptedPrediction] | None = None,
) -> dict[str, WorldModelForecast]:
    """Build one deterministic, read-only event forecast per tracked target."""

    report_target_ids = {report.target_id for report in snapshot.group_reports}
    if execution_snapshot is not None:
        report_target_ids.add(execution_snapshot.target_id)
    forecasts: dict[str, WorldModelForecast] = {}
    for target_id in sorted(report_target_ids):
        prediction = predictions.get(target_id)
        accepted = (accepted_predictions or {}).get(target_id)
        if accepted_predictions is not None:
            prediction = accepted.prediction if accepted is not None else None
        if execution_snapshot is not None and execution_snapshot.target_id == target_id:
            prediction = prediction_from_execution(execution_snapshot)
            accepted = None  # Authoritative execution is assessed against its own source/TTL below.
        if prediction is None:
            forecasts[target_id] = unavailable_forecast(
                snapshot, target_id, None, "accepted_prediction_missing", config=config
            )
            continue
        current_tracking = contact_association_snapshot(snapshot, target_id)
        previous_tracking = (previous_tracking_by_target or {}).get(target_id)
        try:
            forecasts[target_id] = predict_snapshot_events(
                snapshot,
                prediction,
                config=config,
                previous_contact_count=(
                    previous_tracking.contact_count if previous_tracking is not None else None
                ),
                association_confidence=current_tracking.target_confidence,
                previous_association_confidence=(
                    previous_tracking.target_confidence if previous_tracking is not None else None
                ),
                association_entropy=current_tracking.normalized_entropy,
                previous_association_entropy=(
                    previous_tracking.normalized_entropy if previous_tracking is not None else None
                ),
                active_plan=active_plan,
                source_plan_revision=source_plan_revision,
                execution_snapshot=execution_snapshot,
                accepted_prediction=accepted,
            )
        except ValueError as exc:
            forecasts[target_id] = unavailable_forecast(
                snapshot,
                target_id,
                prediction,
                str(exc),
                config=config,
                execution_snapshot=execution_snapshot,
            )
    return forecasts


def prediction_from_execution(execution: OperationalExecutionSnapshot) -> PredictedTrackRef:
    """Read the same IMM/cubic B-spline arrays published by the execution frame."""
    prediction = execution.prediction
    times = prediction.times_s
    points = prediction.centerline_xy
    radii = prediction.corridor_radius_m
    regime = prediction.prediction_regime
    if prediction.bspline_times_s and prediction.bspline_centerline_xy:
        import numpy as np

        times, points = prediction.bspline_times_s, prediction.bspline_centerline_xy
        radii = tuple(float(value) for value in np.interp(times, prediction.times_s, radii))
        regime = "bspline"
    elif regime == "bspline":
        regime = "imm"
    return PredictedTrackRef(
        prediction_id=prediction.prediction_id,
        target_id=prediction.target_id,
        sim_time_s=int(prediction.origin_sim_time_s),
        horizon_s=prediction.times_s[-1] - prediction.origin_sim_time_s,
        sample_step_s=prediction.times_s[0] - prediction.origin_sim_time_s,
        times_s=times,
        points_xy=points,
        corridor_radius_m=radii,
        imm_times_s=prediction.times_s,
        imm_centerline_xy=prediction.centerline_xy,
        imm_corridor_radius_m=prediction.corridor_radius_m,
        bspline_times_s=prediction.bspline_times_s,
        bspline_centerline_xy=prediction.bspline_centerline_xy,
        imm_model_probabilities=dict(prediction.model_probabilities),
        prediction_regime=regime,
        fallback_used=regime in {"short_history", "boundary_recovery"},
        fallback_reason=regime if regime in {"short_history", "boundary_recovery"} else None,
        source_track_revision=prediction.source_track_revision,
        prediction_revision=prediction.prediction_revision,
        last_observed_at_s=prediction.last_observed_at_s,
        generated_at_s=prediction.generated_at_s,
        valid_until_s=prediction.valid_until_s,
    )


def unavailable_forecast(
    snapshot: SituationSnapshot,
    target_id: str,
    prediction: PredictedTrackRef | None,
    reason: str,
    *,
    config: RuleWorldModelConfig = DEFAULT_WORLD_MODEL_CONFIG,
    execution_snapshot: OperationalExecutionSnapshot | None = None,
) -> WorldModelForecast:
    """Missing/invalid input is a visible unavailable result, never an empty success."""
    report = _group_report(snapshot, target_id)
    belief = report.belief if report is not None else None
    return WorldModelForecast(
        scenario_id=snapshot.scenario_id,
        target_id=target_id,
        as_of_s=snapshot.sim_time_s,
        generated_at_s=snapshot.sim_time_s,
        source_prediction_id=prediction.prediction_id if prediction else f"unavailable:{target_id}",
        source_track_revision=(belief.track_revision or None) if belief else None,
        prediction_revision=prediction.prediction_revision if prediction else None,
        last_observed_at_s=belief.last_observed_at_s if belief else None,
        valid_until_s=prediction.valid_until_s if prediction else None,
        source_observation_ids=belief.source_observation_ids if belief else (),
        source_plan_revision=execution_snapshot.execution_revision if execution_snapshot else None,
        owner_group_id=execution_snapshot.tracking_control.tracking_owner_group_id
        if execution_snapshot
        else None,
        source_group_id=report.group_id if report else None,
        data_status=DataStatus.EXPIRED if "expired" in reason else DataStatus.UNAVAILABLE,
        trajectory_fallback_used=prediction.fallback_used if prediction else False,
        imm_model_probabilities={},
        events=(),
        warnings=(reason,),
        horizons=tuple(
            HorizonCoverage(
                name=h.name,
                start_offset_s=h.start_offset_s,
                end_offset_s=h.end_offset_s,
                sample_count=0,
                covered=False,
            )
            for h in config.horizons
        ),
    )


def contact_association_snapshot(
    snapshot: SituationSnapshot,
    target_id: str,
) -> ContactAssociationSnapshot:
    """Derive a transparent ambiguity proxy from public contacts only.

    This is not a learned association probability.  Detection confidence is
    used when public bearing rays exist; otherwise the public operational
    classification supplies a conservative fixed weight.  The normalized
    entropy reports how evenly the available evidence is split across contacts.
    """

    contacts = tuple(getattr(snapshot, "contacts", ()) or ())
    weighted = tuple(
        (str(contact.contact_id), _contact_evidence_weight(contact)) for contact in contacts
    )
    positive = tuple((contact_id, weight) for contact_id, weight in weighted if weight > 0.0)
    total = sum(weight for _, weight in positive)
    if total <= 0.0:
        return ContactAssociationSnapshot(
            contact_count=len(contacts),
            target_confidence=None,
            normalized_entropy=None,
        )
    target_weight = sum(weight for contact_id, weight in positive if contact_id == target_id)
    probabilities = tuple(weight / total for _, weight in positive)
    entropy = (
        0.0
        if len(probabilities) <= 1
        else -sum(value * log(value) for value in probabilities) / log(len(probabilities))
    )
    return ContactAssociationSnapshot(
        contact_count=len(contacts),
        target_confidence=min(1.0, max(0.0, target_weight / total)),
        normalized_entropy=min(1.0, max(0.0, entropy)),
    )


def _contact_evidence_weight(contact: object) -> float:
    rays = tuple(getattr(contact, "bearing_rays", ()) or ())
    confidences = tuple(
        float(ray.detection_confidence)
        for ray in rays
        if not bool(getattr(ray, "is_false_alarm", False))
    )
    if confidences:
        return sum(confidences) / len(confidences)
    classification = getattr(
        contact,
        "classification",
        ContactClassification.UNVERIFIED,
    )
    return {
        ContactClassification.SUBMARINE: 1.0,
        ContactClassification.UNVERIFIED: 0.5,
        ContactClassification.DECOY: 0.1,
    }.get(classification, 0.5)


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
