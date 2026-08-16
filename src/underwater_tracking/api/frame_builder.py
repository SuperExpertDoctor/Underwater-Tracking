# src/underwater_tracking/api/frame_builder.py
"""Pure adapter from estimator-visible runtime state to ``OperationalFrame``.

``build_operational_frame`` maps only the state the estimator and planner
produce for the operator: UUV states, target beliefs (covariance rendered
as ellipse axes plus rotation), bearing observations carried by sonar
contacts, group reports, runtime events, the committed plan, the decision
ledger tail, and metrics. All geometry is clipped to ``DEFAULT_MAP_BOUNDS``
and every entity list is sorted by stable ID, so a given run state always
produces byte-identical frames.

Intent hypotheses, predicted corridors, applied reservations, and breadcrumbs
are optional inputs so the adapter remains useful for a cold-start frame but
the live publisher can carry the full estimator state once it is available.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, cast, get_args

from underwater_tracking.domain import (
    BearingRayView,
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
    OperationalFrame,
    PlanView,
    Point2D,
    PredictionCorridorView,
    TargetEstimateView,
    UUVView,
)
from underwater_tracking.domain.agent_models import (
    DecisionRecord,
    ExpertDirective,
    IntentHypothesis,
    IntentLabel,
    PredictedTrackRef,
    TrackingPlan,
)
from underwater_tracking.domain.models import (
    BearingObservation,
    CarrierState,
    GroupReport,
    RuntimeEvent,
    SituationSnapshot,
    TargetBelief,
    UUVState,
)

# The tactical map region the frame clips geometry to: a square that
# comfortably covers the +/-800 m entity spawn span and the 3000 m
# active-sonar range with margin.
DEFAULT_MAP_BOUNDS = MapBounds(min_x=-4000.0, min_y=-4000.0, max_x=4000.0, max_y=4000.0)

# Floor for the semiminor axis of a degenerate covariance (meters); the
# frame contract requires strictly positive axes.
_MIN_AXIS_M = 1.0e-3

# Non-finite FIM condition numbers (the belief default when no FIM estimate
# exists yet) render as this finite cap: JSONL cannot carry non-finite
# floats through a pydantic round trip.
_FIM_CONDITION_FINITE_CAP = 1.0e6

_INTENT_LABELS: frozenset[str] = frozenset(get_args(IntentLabel))


def build_operational_frame(
    snapshot: SituationSnapshot,
    plan: TrackingPlan | None,
    ledger_tail: Sequence[DecisionRecord],
    events: Sequence[RuntimeEvent],
    metrics: Sequence[MetricView],
    *,
    intent_hypotheses: Mapping[str, IntentHypothesis] | None = None,
    predictions: Mapping[str, PredictedTrackRef] | None = None,
    applied_directives: Sequence[ExpertDirective] = (),
    breadcrumbs: Mapping[str, Sequence[tuple[float, float]]] | None = None,
) -> OperationalFrame:
    """Build one validated operational frame from estimator-visible state.

    ``frame_id`` mirrors the snapshot revision; ``plan_version`` mirrors
    the committed plan's revision (``0`` with no plan), and the rendered
    active plan carries the same version so the frame contract's
    consistency validator passes. ``events`` and ``ledger_tail`` are mapped
    from the caller's arguments, not from the snapshot's pending-events
    field.
    """
    by_uuv = {state.uuv_id: state for state in snapshot.uuvs}
    reports = sorted(snapshot.group_reports, key=lambda report: report.target_id)
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
            classification=classification_by_target.get(report.target_id, "unknown"),
            last_ping_s=latest_ping_by_target.get(report.target_id),
        )
        for report in reports
    )
    groups = tuple(_build_group(report) for report in reports)
    rays = tuple(
        _build_ray(observation, by_uuv)
        for contact in sorted(snapshot.contacts, key=lambda c: c.contact_id)
        for observation in sorted(contact.bearing_rays, key=lambda o: o.observation_id)
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
    return OperationalFrame(
        frame_id=snapshot.snapshot_revision,
        sim_time_s=snapshot.sim_time_s,
        plan_version=plan.revision if plan is not None else 0,
        map_bounds=DEFAULT_MAP_BOUNDS,
        carrier=_build_carrier_view(snapshot.carrier),
        uuvs=uuv_views,
        target_estimates=estimates,
        bearing_rays=rays,
        groups=groups,
        events=event_views,
        plans=plan_views,
        ledger=ledger_views,
        metrics=metric_views,
    )


def _build_uuv_view(
    state: UUVState,
    plan: TrackingPlan | None,
    *,
    breadcrumbs: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    active_pingers: set[str] | None = None,
    reserved_uuvs: set[str] | None = None,
) -> UUVView:
    waypoints = plan.waypoints_by_member.get(state.uuv_id, ()) if plan is not None else ()
    current_waypoint = _clip_point(waypoints[0].x, waypoints[0].y) if waypoints else None
    trail = breadcrumbs.get(state.uuv_id, ()) if breadcrumbs is not None else ()
    return UUVView(
        uuv_id=state.uuv_id,
        status=state.status,
        deployment_state=state.deployment_state,
        position=_clip_point(state.position_xy[0], state.position_xy[1]),
        heading_rad=state.heading_rad,
        speed_mps=state.speed_mps,
        energy_fraction=state.energy_fraction,
        group_id=state.group_id,
        current_waypoint=current_waypoint,
        breadcrumb=tuple(_clip_point(x, y) for x, y in trail),
        sensor_mode=(
            "active"
            if state.uuv_id in (active_pingers or set())
            else state.sensor_mode
        ),
        reserved=state.reserved or state.uuv_id in (reserved_uuvs or set()),
    )


def _build_carrier_view(carrier: CarrierState | None) -> CarrierView | None:
    if carrier is None:
        return None
    return CarrierView(
        carrier_id=carrier.carrier_id,
        position=_clip_point(carrier.position_xy[0], carrier.position_xy[1]),
        heading_rad=carrier.heading_rad,
        speed_mps=carrier.speed_mps,
        status=carrier.status,
        onboard_uuv_ids=tuple(sorted(carrier.onboard_uuv_ids)),
        deployed_uuv_ids=tuple(sorted(carrier.deployed_uuv_ids)),
        returning_uuv_ids=tuple(sorted(carrier.returning_uuv_ids)),
    )


def _build_estimate(
    report: GroupReport,
    plan: TrackingPlan | None,
    *,
    intent_hypotheses: Mapping[str, IntentHypothesis] | None = None,
    predictions: Mapping[str, PredictedTrackRef] | None = None,
    classification: str = "unknown",
    last_ping_s: int | None = None,
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
        mean=_clip_point(belief.mean[0], belief.mean[1]),
        covariance_ellipse=_covariance_to_ellipse(p00, p01, p10, p11),
        intent=_build_intent(plan, belief.target_id, intent_hypotheses),
        prediction=_build_prediction(predictions.get(belief.target_id) if predictions else None),
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
    )


def _build_prediction(prediction: PredictedTrackRef | None) -> PredictionCorridorView | None:
    if prediction is None:
        return None
    points = tuple(_clip_point(x, y) for x, y in prediction.points_xy)
    return PredictionCorridorView(
        horizon_s=prediction.horizon_s,
        sample_step_s=prediction.sample_step_s,
        centerline_xy=points,
        radius_m=prediction.corridor_radius_m,
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


def _build_ray(
    observation: BearingObservation, by_uuv: Mapping[str, UUVState]
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
        origin=_clip_point(origin.position_xy[0], origin.position_xy[1]),
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
        else None
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


def _clip_point(x: float, y: float) -> Point2D:
    bounds = DEFAULT_MAP_BOUNDS
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
