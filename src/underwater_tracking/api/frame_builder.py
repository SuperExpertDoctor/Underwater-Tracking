# src/underwater_tracking/api/frame_builder.py
"""Pure adapter from estimator-visible runtime state to ``OperationalFrame``.

``build_operational_frame`` maps only the state the estimator and planner
produce for the operator: UUV states, target beliefs (covariance rendered
as ellipse axes plus rotation), bearing observations carried by sonar
contacts, group reports, runtime events, the committed plan, the decision
ledger tail, and metrics. All geometry is clipped to ``DEFAULT_MAP_BOUNDS``
and every entity list is sorted by stable ID, so a given run state always
produces byte-identical frames.

Two frame slots have no source in the builder inputs and render
conservatively: the prediction corridor (``None``) and intent confidence
(``0.0``, with the label taken from the plan's per-target intent
references when present, ``unknown`` otherwise). Wiring the richer intent
hypotheses and predicted tracks through the runtime port is a later stage.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, cast, get_args

from underwater_tracking.domain import (
    BearingRayView,
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
    TargetEstimateView,
    UUVView,
)
from underwater_tracking.domain.agent_models import DecisionRecord, IntentLabel, TrackingPlan
from underwater_tracking.domain.models import (
    BearingObservation,
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
) -> OperationalFrame:
    """Build one validated operational frame from estimator-visible state.

    ``frame_id`` mirrors the snapshot revision; ``plan_version`` mirrors
    the committed plan's revision (``0`` with no plan), and the rendered
    active plan carries the same version so the frame contract's
    consistency validator passes. ``events`` and ``ledger_tail`` are mapped
    from the caller's arguments, not from the snapshot's pending-events
    field.
    """
    uuv_views = tuple(
        _build_uuv_view(state, plan)
        for state in sorted(snapshot.uuvs, key=lambda state: state.uuv_id)
    )
    by_uuv = {state.uuv_id: state for state in snapshot.uuvs}
    reports = sorted(snapshot.group_reports, key=lambda report: report.target_id)
    estimates = tuple(_build_estimate(report, plan) for report in reports)
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
        uuvs=uuv_views,
        target_estimates=estimates,
        bearing_rays=rays,
        groups=groups,
        events=event_views,
        plans=plan_views,
        ledger=ledger_views,
        metrics=metric_views,
    )


def _build_uuv_view(state: UUVState, plan: TrackingPlan | None) -> UUVView:
    waypoints = plan.waypoints_by_member.get(state.uuv_id, ()) if plan is not None else ()
    current_waypoint = _clip_point(waypoints[0].x, waypoints[0].y) if waypoints else None
    return UUVView(
        uuv_id=state.uuv_id,
        status=state.status,
        position=_clip_point(state.position_xy[0], state.position_xy[1]),
        heading_rad=state.heading_rad,
        speed_mps=state.speed_mps,
        energy_fraction=state.energy_fraction,
        group_id=state.group_id,
        current_waypoint=current_waypoint,
        breadcrumb=(),
    )


def _build_estimate(report: GroupReport, plan: TrackingPlan | None) -> TargetEstimateView:
    belief = report.belief
    if len(belief.mean) < 2:
        raise ValueError(
            f"belief mean for target {belief.target_id!r} has fewer than 2 components"
        )
    p00, p01, p10, p11 = _position_covariance(belief)
    return TargetEstimateView(
        target_id=belief.target_id,
        mean=_clip_point(belief.mean[0], belief.mean[1]),
        covariance_ellipse=_covariance_to_ellipse(p00, p01, p10, p11),
        intent=_build_intent(plan, belief.target_id),
        prediction=None,
        quality=EstimateQualityView(
            quality_score=report.quality.window_mean,
            # The RMS position error proxy is the covariance trace:
            # E[||x - mean||^2] = p00 + p11 for the 2D Gaussian.
            estimated_rmse_m=math.sqrt(max(0.0, p00 + p11)),
            fim_min_eigenvalue=belief.fim_min_eigenvalue,
            fim_condition=_finite_condition(belief.fim_condition),
        ),
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


def _build_intent(plan: TrackingPlan | None, target_id: str) -> IntentView:
    """Intent label from the plan's per-target intent references.

    The builder inputs carry no confidence for the referenced label, so the
    view renders with confidence ``0.0``; an absent or unknown reference
    renders as ``unknown``.
    """
    raw = plan.intent_refs.get(target_id) if plan is not None else None
    label: IntentLabel = cast(
        IntentLabel, raw if raw is not None and raw in _INTENT_LABELS else "unknown"
    )
    return IntentView(label=label, confidence=0.0, alternatives={})


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
        message="",
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
