# src/underwater_tracking/api/frame_builder.py
"""Pure adapter from estimator-visible runtime state to ``OperationalFrame``.

``build_operational_frame`` maps only the state the estimator and planner
produce for the operator: UUV/USV states, public communication links, target
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
    OperationalSchemeView,
    PlanTimelineView,
    PlanView,
    Point2D,
    PredictionCorridorView,
    TargetEstimateView,
    TimelineFactorView,
    TimelinePlanView,
    IntelligenceView,
    UUVView,
)
from underwater_tracking.domain.platforms import (
    PlatformKind,
    PlatformSnapshot,
    UUVPlatformState,
    USVPlatformState,
)
from underwater_tracking.domain.ui_models import (
    BrainView,
    CommunicationLinkView,
    USVView,
)
from underwater_tracking.domain.agent_models import (
    DecisionRecord,
    ExpertDirective,
    IntentHypothesis,
    IntentLabel,
    PlanAdjustmentSuggestion,
    PredictedTrackRef,
    TrackingPlan,
)
from underwater_tracking.domain.models import (
    BearingObservation,
    CarrierState,
    GroupReport,
    IntelligenceReport,
    OperationalScheme,
    RuntimeEvent,
    SituationSnapshot,
    TargetBelief,
    UUVState,
)
from underwater_tracking.domain.adversary_models import AdversaryOperationalSummary

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
    applied_directives: Sequence[ExpertDirective] = (),
    breadcrumbs: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    map_bounds_xy: Sequence[float] | None = None,
    llm_paused: bool = False,
    plan_adjustment_suggestions: Sequence[PlanAdjustmentSuggestion] = (),
) -> OperationalFrame:
    """Build one validated operational frame from estimator-visible state.

    ``frame_id`` mirrors the snapshot revision; ``plan_version`` mirrors
    the committed plan's revision (``0`` with no plan), and the rendered
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
    usv_views, link_views, peers_by_platform = _build_platform_views(
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
            classification=classification_by_target.get(report.target_id, "unknown"),
            last_ping_s=latest_ping_by_target.get(report.target_id),
            adversary_summary=adversary_by_target.get(report.target_id),
            map_bounds=map_bounds,
        )
        for report in reports
    )
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
    return OperationalFrame(
        frame_id=snapshot.snapshot_revision,
        sim_time_s=snapshot.sim_time_s,
        plan_version=plan.revision if plan is not None else 0,
        map_bounds=map_bounds,
        carrier=_build_carrier_view(
            snapshot.carrier,
            snapshot.platform_snapshot.carrier.support_radius_m
            if snapshot.platform_snapshot is not None
            else None,
            map_bounds,
        ),
        uuvs=uuv_views,
        usvs=usv_views,
        communication_links=link_views,
        brains=_build_brain_views(snapshot, events, plan, llm_paused=llm_paused),
        adversaries=tuple(
            _build_adversary_view(summary)
            for summary in sorted(
                snapshot.adversary_summaries, key=lambda item: item.target_id
            )
        ),
        target_estimates=estimates,
        bearing_rays=rays,
        groups=groups,
        events=event_views,
        plans=plan_views,
        ledger=ledger_views,
        metrics=metric_views,
        scheme=_build_scheme_view(snapshot.operational_scheme, snapshot.sim_time_s),
        intelligence=_build_intelligence_views(
            snapshot.intelligence_reports, snapshot.sim_time_s
        ),
        plan_timeline=_build_plan_timeline(ledger_tail, events),
        plan_adjustment_suggestions=tuple(plan_adjustment_suggestions),
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
        position=_clip_point(state.position_xy[0], state.position_xy[1], map_bounds),
        heading_rad=state.heading_rad,
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
        position=_clip_point(carrier.position_xy[0], carrier.position_xy[1], map_bounds),
        heading_rad=carrier.heading_rad,
        speed_mps=carrier.speed_mps,
        status=carrier.status,
        onboard_uuv_ids=tuple(sorted(carrier.onboard_uuv_ids)),
        deployed_uuv_ids=tuple(sorted(carrier.deployed_uuv_ids)),
        returning_uuv_ids=tuple(sorted(carrier.returning_uuv_ids)),
        support_radius_m=support_radius_m,
    )


def _build_platform_views(
    snapshot: PlatformSnapshot | None,
    map_bounds: MapBounds = DEFAULT_MAP_BOUNDS,
) -> tuple[
    tuple[USVView, ...],
    tuple[CommunicationLinkView, ...],
    dict[str, tuple[str, ...]],
]:
    """Project platform-core state into public node/link views.

    ``PlatformSnapshot`` contains capabilities and positions, but no target
    state.  Disconnected links are reconstructed from the same public range
    contract used by the connectivity core, so the UI can explain a missing
    relay without receiving hidden simulation state.
    """
    if snapshot is None:
        return (), (), {}

    platform_positions: dict[str, tuple[float, float]] = {
        snapshot.carrier.carrier_id: snapshot.carrier.position_xy,
    }
    platform_positions.update(
        {state.platform_id: state.position_xy for state in snapshot.roster.usvs}
    )
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
    carrier_id = snapshot.carrier.carrier_id
    for usv in snapshot.roster.usvs:
        candidates.append(
            (
                carrier_id,
                usv.platform_id,
                "surface",
                usv.capability.communications.surface_range_m,
                _distance(platform_positions[carrier_id], usv.position_xy),
            )
        )
    ordered = (*snapshot.roster.usvs, *snapshot.roster.uuvs)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left.capability.kind is PlatformKind.USV and right.capability.kind is PlatformKind.USV:
                medium: Literal["surface", "acoustic"] = "surface"
                limit = min(
                    left.capability.communications.surface_range_m,
                    right.capability.communications.surface_range_m,
                )
            else:
                medium = "acoustic"
                limit = min(
                    left.capability.communications.acoustic_range_m,
                    right.capability.communications.acoustic_range_m,
                )
            candidates.append(
                (
                    left.platform_id,
                    right.platform_id,
                    medium,
                    limit,
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
                relay=connected and (
                    source_id == carrier_id
                    or target_id == carrier_id
                    or source_id in {state.platform_id for state in snapshot.roster.usvs}
                    or target_id in {state.platform_id for state in snapshot.roster.usvs}
                ),
            )
        )

    uuv_ids = {state.platform_id for state in snapshot.roster.uuvs}
    usv_views = tuple(
        _build_usv_view(
            state,
            carrier_id=carrier_id,
            carrier_position=snapshot.carrier.position_xy,
            connected_peer_ids=peers.get(state.platform_id, set()),
            connected_uuv_ids=uuv_ids & peers.get(state.platform_id, set()),
            map_bounds=map_bounds,
        )
        for state in sorted(snapshot.roster.usvs, key=lambda item: item.platform_id)
    )
    return (
        usv_views,
        tuple(sorted(link_views, key=lambda link: (link.source_id, link.target_id))),
        {platform_id: tuple(sorted(peer_ids)) for platform_id, peer_ids in peers.items()},
    )


def _build_usv_view(
    state: USVPlatformState,
    *,
    carrier_id: str,
    carrier_position: tuple[float, float],
    connected_peer_ids: set[str],
    connected_uuv_ids: set[str],
    map_bounds: MapBounds = DEFAULT_MAP_BOUNDS,
) -> USVView:
    distance_to_carrier = _distance(state.position_xy, carrier_position)
    return USVView(
        usv_id=state.platform_id,
        position=_clip_point(*state.position_xy, map_bounds),
        heading_rad=state.heading_rad,
        speed_mps=state.speed_mps,
        energy_fraction=state.energy_fraction,
        deployment_state=state.deployment_state,
        sensor_mode=state.sensor_mode,
        distance_to_carrier_m=distance_to_carrier,
        passive_range_m=state.capability.sonar.passive_range_m,
        active_range_m=min(
            state.capability.sonar.active_source_range_m,
            state.capability.sonar.active_receive_range_m,
        ),
        active_capable=state.capability.sonar.active_capable,
        communication_range_m=state.capability.communications.surface_range_m,
        relay_active=(
            carrier_id in connected_peer_ids and bool(connected_uuv_ids)
        ),
        connected=bool(connected_peer_ids),
        connected_peer_ids=tuple(sorted(connected_peer_ids)),
    )


def _build_brain_views(
    snapshot: SituationSnapshot,
    events: Sequence[RuntimeEvent],
    plan: TrackingPlan | None,
    *,
    llm_paused: bool = False,
) -> tuple[BrainView, ...]:
    """Expose decision-role data flow without exposing hidden target state."""
    latest_event = max(events, key=lambda event: (event.sim_time_s, event.event_id), default=None)
    event_type = latest_event.event_type.lower() if latest_event is not None else ""
    paused = llm_paused or any(
        token in event_type for token in ("llm_error", "brain_paused", "agent_paused")
    )
    status: Literal["online", "paused", "degraded", "unknown"] = (
        "paused" if paused else "online"
    )
    status_message = "等待 LLM 重连" if paused else None
    return (
        BrainView(
            brain_id="carrier-master",
            role="master",
            status=(
                status
                if paused or plan is not None or snapshot.operational_scheme is not None
                else "unknown"
            ),
            last_update_s=snapshot.sim_time_s,
            message=(
                status_message
                or ("全局资源调度与人机协同" if plan is not None else "等待主脑决策")
            ),
            connected_platform_ids=tuple(sorted(state.uuv_id for state in snapshot.uuvs)),
        ),
        BrainView(
            brain_id="group-slave",
            role="slave",
            status=status if paused or snapshot.uuvs else "unknown",
            last_update_s=snapshot.sim_time_s if paused or snapshot.uuvs else None,
            message=status_message or ("编组声纳即时决策" if snapshot.uuvs else "等待编组"),
            connected_platform_ids=tuple(sorted(state.uuv_id for state in snapshot.uuvs)),
        ),
        BrainView(
            brain_id="target-adversary",
            role="adversary",
            status=status if paused or snapshot.group_reports else "unknown",
            last_update_s=snapshot.sim_time_s if paused or snapshot.group_reports else None,
            message=(
                status_message
                or (
                    "基于估计态势建模对手意图"
                    if snapshot.group_reports
                    else "等待目标估计"
                )
            ),
            connected_platform_ids=tuple(sorted(state.platform_id for state in snapshot.platform_snapshot.roster.uuvs))
            if snapshot.platform_snapshot is not None
            else (),
        ),
    )


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _build_estimate(
    report: GroupReport,
    plan: TrackingPlan | None,
    *,
    intent_hypotheses: Mapping[str, IntentHypothesis] | None = None,
    predictions: Mapping[str, PredictedTrackRef] | None = None,
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


def _build_prediction(
    prediction: PredictedTrackRef | None,
    map_bounds: MapBounds = DEFAULT_MAP_BOUNDS,
) -> PredictionCorridorView | None:
    if prediction is None:
        return None
    points = tuple(_clip_point(x, y, map_bounds) for x, y in prediction.points_xy)
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


def _build_plan_timeline(
    decisions: Sequence[DecisionRecord], events: Sequence[RuntimeEvent]
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
        for query_id in decision.knowledge_query_ids:
            factors.append(
                TimelineFactorView(
                    kind="knowledge",
                    ref_id=query_id,
                    label="本体专家知识",
                    detail="已注入策略 LLM 的外部专家证据",
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
            version = diff.to_revision if diff is not None else 1
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
    return tuple(rows[-80:])


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
