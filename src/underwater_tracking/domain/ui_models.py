# src/underwater_tracking/domain/ui_models.py
"""Versioned UI frame contracts with strict truth isolation (spec 17.3).

``OperationalFrame`` and its view models carry only estimator-visible
state: positions, covariances, bearings, groups, plans, events, ledger
rows and metrics.  They must never contain target truth — that gate is
enforced by ``tests/api/test_frame_contracts.py`` at the schema level and
by ``tests/api/test_truth_isolation.py`` at the route level (task 10).

Target truth lives exclusively in ``EvaluationFrame``, a standalone model
(no inheritance from ``OperationalFrame``) that wraps ``TargetTruth``
dataclasses together with the identifiers pairing it to the operational
run it was collected under.  It is served only by separately enabled
evaluation routes.
"""
from __future__ import annotations

from math import pi
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from underwater_tracking.domain.agent_models import (
    Concept,
    IntentLabel,
    PlanAdjustmentSuggestion,
    PlanStatus,
)
from underwater_tracking.domain.models import (
    CarrierStatus,
    DeploymentState,
    EventLevel,
    IntelligenceSource,
    StrictModel,
    UUVStatus,
)
from underwater_tracking.domain.regional_models import (
    CommunicationRequirement,
    GridSpec,
    SonarPolicy,
    TimeWindow,
)
from underwater_tracking.domain.relationships import (
    expected_carrier_status,
    normalize_legacy_carrier_relationships,
    normalize_legacy_uuv_deployment_state,
)
from underwater_tracking.domain.truth import TargetTruth


class Point2D(StrictModel):
    x: float
    y: float


class MapBounds(StrictModel):
    """Axis-aligned region the tactical map clips geometry to."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @model_validator(mode="after")
    def axes_ordered(self) -> MapBounds:
        if self.min_x > self.max_x or self.min_y > self.max_y:
            raise ValueError("map bounds require min <= max on both axes")
        return self


class CovarianceEllipse(StrictModel):
    """Covariance rendered as an ellipse: axes in meters and rotation."""

    semimajor_m: float = Field(gt=0)
    semiminor_m: float = Field(gt=0)
    rotation_rad: float

    @field_validator("rotation_rad")
    @classmethod
    def wrap_angle(cls, value: float) -> float:
        return (value + pi) % (2 * pi) - pi

    @model_validator(mode="after")
    def axes_ordered(self) -> CovarianceEllipse:
        if self.semiminor_m > self.semimajor_m:
            raise ValueError("semiminor axis must not exceed semimajor axis")
        return self


class UUVView(StrictModel):
    uuv_id: str
    status: UUVStatus
    deployment_state: DeploymentState = DeploymentState.DEPLOYED
    position: Point2D
    heading_rad: float
    speed_mps: float = Field(ge=0)
    energy_fraction: float = Field(ge=0, le=1)
    remaining_range_m: float = Field(default=0.0, ge=0)
    group_id: str | None = None
    current_waypoint: Point2D | None = None
    breadcrumb: tuple[Point2D, ...] = ()
    sensor_mode: Literal["active", "passive"] = "passive"
    reserved: bool = False
    passive_range_m: float | None = Field(default=None, gt=0)
    active_range_m: float | None = Field(default=None, gt=0)
    active_capable: bool = False
    is_group_leader: bool = False
    master_connected: bool = False
    connected_peer_ids: tuple[str, ...] = ()
    communication_status: Literal["carrier", "relay", "mesh", "disconnected"] = "disconnected"
    tracked_target_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_deployment_state(cls, value: Any) -> Any:
        return normalize_legacy_uuv_deployment_state(value)

    @model_validator(mode="after")
    def status_matches_deployment_state(self) -> UUVView:
        if self.status is UUVStatus.TRACKING and self.deployment_state is DeploymentState.ONBOARD:
            raise ValueError("tracking status cannot be onboard")
        if self.status is UUVStatus.TRACKING and self.deployment_state is DeploymentState.FAILED:
            raise ValueError("tracking status cannot be failed")
        if self.status is UUVStatus.RETURNING and self.deployment_state is not DeploymentState.RETURNING:
            raise ValueError("returning status requires returning deployment_state")
        if self.status is UUVStatus.FAILED and self.deployment_state is not DeploymentState.FAILED:
            raise ValueError("failed status requires failed deployment_state")
        if self.deployment_state is DeploymentState.RETURNING and self.status is not UUVStatus.RETURNING:
            raise ValueError("returning deployment_state requires returning status")
        if self.deployment_state is DeploymentState.FAILED and self.status is not UUVStatus.FAILED:
            raise ValueError("failed deployment_state requires failed status")
        return self


class CarrierView(StrictModel):
    carrier_id: str
    position: Point2D
    heading_rad: float
    speed_mps: float = Field(ge=0)
    status: CarrierStatus = CarrierStatus.TRANSIT
    onboard_uuv_ids: tuple[str, ...] = ()
    deployed_uuv_ids: tuple[str, ...] = ()
    returning_uuv_ids: tuple[str, ...] = ()
    support_radius_m: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def relationship_lists_are_disjoint(self) -> CarrierView:
        raw_lists = (
            self.onboard_uuv_ids,
            self.deployed_uuv_ids,
            self.returning_uuv_ids,
        )
        if any(len(ids) != len(set(ids)) for ids in raw_lists):
            raise ValueError("carrier relationship lists must not contain duplicate IDs")
        lists = tuple(set(ids) for ids in raw_lists)
        if any(left & right for index, left in enumerate(lists) for right in lists[index + 1 :]):
            raise ValueError("carrier relationship lists must be disjoint")
        expected = expected_carrier_status(
            self.speed_mps, self.onboard_uuv_ids, self.deployed_uuv_ids, self.returning_uuv_ids
        )
        if str(self.status) != expected:
            if self.returning_uuv_ids:
                raise ValueError("returning UUVs require recovering status")
            if self.status is CarrierStatus.RECOVERING:
                raise ValueError("recovering status requires returning UUVs")
            if self.status is CarrierStatus.DEPLOYING:
                raise ValueError("deploying status requires onboard and deployed UUVs")
            if self.status is CarrierStatus.STANDBY:
                raise ValueError("standby status requires zero speed")
            if self.status is CarrierStatus.TRANSIT:
                raise ValueError("transit status requires movement")
        return self


class IntentView(StrictModel):
    label: IntentLabel
    confidence: float = Field(ge=0, le=1)
    alternatives: dict[IntentLabel, float] = Field(default_factory=dict)


class PredictionCorridorView(StrictModel):
    """Predicted track centerline with a radius envelope per sample."""

    horizon_s: float = Field(gt=0)
    sample_step_s: float = Field(gt=0)
    centerline_xy: tuple[Point2D, ...] = ()
    radius_m: tuple[float, ...] = ()


class EstimateQualityView(StrictModel):
    """Estimator-visible quality proxies; never true error."""

    quality_score: float = Field(ge=0, le=1)
    estimated_rmse_m: float = Field(ge=0)
    fim_min_eigenvalue: float = Field(ge=0)
    fim_condition: float = Field(ge=0)


class TargetEstimateView(StrictModel):
    target_id: str
    mean: Point2D
    covariance_ellipse: CovarianceEllipse
    intent: IntentView
    prediction: PredictionCorridorView | None = None
    quality: EstimateQualityView
    classification: Literal["submarine", "decoy", "unknown"] = "unknown"
    last_ping_s: int | None = None
    detection_range_m: float = Field(default=1.0, gt=0)
    detected_platform_ids: tuple[str, ...] = ()


class BearingRayView(StrictModel):
    observation_id: str
    uuv_id: str
    target_id: str
    origin: Point2D
    azimuth_rad: float
    variance_rad2: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)

    @field_validator("azimuth_rad")
    @classmethod
    def wrap_angle(cls, value: float) -> float:
        return (value + pi) % (2 * pi) - pi


class GroupQualityView(StrictModel):
    instant: float = Field(ge=0, le=1)
    window_mean: float = Field(ge=0, le=1)
    ewma: float = Field(ge=0, le=1)
    components: dict[str, float] = Field(default_factory=dict)
    hard_guard_reasons: tuple[str, ...] = ()


class GroupView(StrictModel):
    group_id: str
    target_id: str
    member_ids: tuple[str, ...] = ()
    quality: GroupQualityView


class TrackingEffectView(StrictModel):
    status: Literal["planned", "active", "handoff_ready", "degraded", "uncovered"]
    coverage_ratio: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)
    handoff_progress: float = Field(ge=0, le=1)
    quality_source: Literal["group_quality_proxy", "region_telemetry"]
    hard_guard_reasons: tuple[str, ...] = ()
    expert_feedback_ids: tuple[str, ...] = ()


class RegionTaskView(StrictModel):
    region_id: str
    display_name: str
    target_id: str
    geometry: tuple[Point2D, ...]
    grid_x: int | None = None
    grid_y: int | None = None
    start_time_s: int = Field(ge=0)
    end_time_s: int = Field(gt=0)
    visit_window_index: int = Field(default=0, ge=0)
    visit_window: TimeWindow | None = None
    predecessor_region_ids: tuple[str, ...] = ()
    successor_region_ids: tuple[str, ...] = ()
    assigned_uuv_ids: tuple[str, ...] = ()
    assigned_usv_ids: tuple[str, ...] = ()
    tracking_mode: Literal[
        "uuv_primary_usv_relay", "heuristic_uuv", "heuristic_usv"
    ]
    uuv_roles: tuple[str, ...] = ()
    usv_role: str | None = None
    sonar_policy: SonarPolicy | None = None
    communication: CommunicationRequirement | None = None
    communication_links: tuple[str, ...] = ()
    relay_usv_ids: tuple[str, ...] = ()
    group_id: str | None = None
    status: Literal[
        "planned", "active", "handoff_ready", "handed_off", "degraded", "uncovered"
    ]
    degraded_reasons: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    revision: int = Field(default=1, ge=1)
    effect: TrackingEffectView


class RegionalPlanView(StrictModel):
    target_id: str
    prediction_id: str
    revision: int = Field(ge=1)
    cell_size_m: float = Field(gt=0)
    # Optional so JSONL frames written before regional detail support remain replayable.
    grid_spec: GridSpec | None = None
    evidence_ids: tuple[str, ...] = ()
    current_handoff_region_id: str | None = None
    next_handoff_region_id: str | None = None
    causal_event_ids: tuple[str, ...] = ()
    llm_hashes: tuple[str, str] | None = None
    regions: tuple[RegionTaskView, ...] = ()


class EventView(StrictModel):
    event_id: str
    sim_time_s: int = Field(ge=0)
    event_type: str
    level: EventLevel
    entity_id: str | None = None
    message: str = ""


class OperationalSchemeView(StrictModel):
    """Bounded, operator-facing view of the active operational scheme."""

    scheme_id: str
    version: int = Field(ge=1)
    valid_from_s: int = Field(ge=0)
    valid_until_s: int = Field(ge=0)
    target_priorities: dict[str, float] = Field(default_factory=dict)
    minimum_quality: dict[str, float] = Field(default_factory=dict)
    constraints: tuple[str, ...] = ()


class IntelligenceView(StrictModel):
    """Compact source-attributed intelligence summary for the command center."""

    report_id: str
    source: IntelligenceSource
    target_id: str
    confidence: float = Field(ge=0, le=1)
    issued_at_s: int = Field(ge=0)
    valid_until_s: int = Field(ge=0)
    content_summary: str | None = None


class USVView(StrictModel):
    """Estimator-visible surface node and relay status."""

    usv_id: str
    position: Point2D
    heading_rad: float
    speed_mps: float = Field(ge=0)
    energy_fraction: float = Field(ge=0, le=1)
    deployment_state: Literal["onboard", "deployed", "returning", "failed"]
    sensor_mode: Literal["active", "passive"] = "passive"
    distance_to_carrier_m: float = Field(ge=0)
    passive_range_m: float = Field(gt=0)
    active_range_m: float = Field(gt=0)
    active_capable: bool
    communication_range_m: float = Field(default=1.0, gt=0)
    relay_active: bool = False
    connected: bool = False
    connected_peer_ids: tuple[str, ...] = ()


class CommunicationLinkView(StrictModel):
    """A public link candidate, including distance-based disconnect state."""

    source_id: str
    target_id: str
    medium: Literal["surface", "acoustic"]
    distance_m: float = Field(ge=0)
    limit_m: float = Field(gt=0)
    status: Literal["connected", "disconnected"]
    relay: bool = False


class RegionAssignmentView(StrictModel):
    """One platform role rendered inside a regional timeline row."""

    platform_id: str = Field(min_length=1)
    platform_kind: Literal["uuv", "usv"]
    role: str = Field(min_length=1)
    start_offset_s: float = Field(allow_inf_nan=False)
    end_offset_s: float = Field(allow_inf_nan=False)
    sonar_mode: Literal["passive", "active"] = "passive"

    @model_validator(mode="after")
    def ordered_offsets(self) -> RegionAssignmentView:
        if self.end_offset_s < self.start_offset_s:
            raise ValueError("end_offset_s must not precede start_offset_s")
        return self


class RegionTimelineView(StrictModel):
    """Estimator-safe regional handoff row for live and replay frames."""

    region_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    center: Point2D
    bounds: MapBounds
    start_offset_s: float = Field(allow_inf_nan=False)
    end_offset_s: float = Field(allow_inf_nan=False)
    status: Literal["planned", "active", "handed_off", "degraded", "uncovered"]
    coverage_mode: Literal["required", "reserve", "optional"] = "required"
    priority: float = Field(ge=0, allow_inf_nan=False)
    occupancy_likelihood: float = Field(ge=0, le=1, allow_inf_nan=False)
    uuv_assignments: tuple[RegionAssignmentView, ...] = ()
    usv_assignments: tuple[RegionAssignmentView, ...] = ()
    communication_links: tuple[CommunicationLinkView, ...] = ()
    handoff_from: str | None = None
    handoff_to: str | None = None
    evidence_ids: tuple[str, ...] = ()
    degraded_reasons: tuple[str, ...] = ()
    plan_revision: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def ordered_offsets(self) -> RegionTimelineView:
        if self.end_offset_s < self.start_offset_s:
            raise ValueError("end_offset_s must not precede start_offset_s")
        return self


class BrainView(StrictModel):
    """Operational data-flow status for the three decision roles."""

    brain_id: str
    role: Literal["master", "slave", "adversary"]
    status: Literal["online", "paused", "degraded", "unknown"]
    last_update_s: int | None = Field(default=None, ge=0)
    message: str = ""
    connected_platform_ids: tuple[str, ...] = ()


class AdversaryView(StrictModel):
    """Operator-safe target brain decision and self-detection summary."""

    target_id: str
    sim_time_s: int = Field(ge=0)
    detection_range_m: float = Field(gt=0)
    detected_platform_ids: tuple[str, ...] = ()
    trigger_event_ids: tuple[str, ...] = ()
    decision_id: str | None = None
    maneuver: str | None = None
    intent: str | None = None
    segment: str | None = None
    speed_mps: float | None = Field(default=None, ge=0)
    heading_rad: float | None = None
    decoy_count: int = Field(default=0, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = None
    communications_discipline: str | None = None
    decision_status: Literal["unknown", "inconclusive", "contact_maintained", "contact_lost"] = "unknown"


class PlanView(StrictModel):
    """One plan as rendered to the operator (current or candidate).

    ``version`` is the version the frame's ``plan_version`` must agree
    with whenever this plan is the active one.
    """

    plan_id: str
    version: int = Field(ge=1)
    status: PlanStatus
    concept: Concept = "hold_current"
    reason: str = ""
    affected_targets: tuple[str, ...] = ()
    group_changes: tuple[str, ...] = ()
    valid_from_s: int = Field(default=0, ge=0)
    valid_until_s: int | None = None
    segment_plan: tuple[str, ...] = ()


class LedgerView(StrictModel):
    """One traceable decision row for the decision ledger."""

    decision_id: str
    sim_time_s: int = Field(ge=0)
    outcome: Literal["committed", "degraded", "rejected"] = "committed"
    trigger_event_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    final_plan_id: str | None = None
    final_plan_version: int | None = None


class TimelineFactorView(StrictModel):
    """One left-hand factor that caused a plan adjustment."""

    kind: Literal["event", "evidence", "directive", "knowledge"]
    ref_id: str
    label: str
    detail: str = ""


class TimelinePlanView(StrictModel):
    """One right-hand result of a plan adjustment."""

    plan_id: str
    version: int = Field(ge=1)
    status: PlanStatus
    summary: str = ""
    group_changes: tuple[str, ...] = ()


class PlanTimelineView(StrictModel):
    """Factor-to-plan relationship for historical battle replay."""

    adjustment_id: str
    sim_time_s: int = Field(ge=0)
    factors: tuple[TimelineFactorView, ...] = ()
    plan: TimelinePlanView | None = None


class MetricView(StrictModel):
    metric_id: str
    label: str = ""
    value: float
    unit: str = ""
    threshold: float | None = None
    window_s: int = Field(default=0, ge=0)
    series: tuple[float, ...] = ()
    status: str = "OK"
    mean_window: float | None = None
    worst_window: float | None = None
    trend_per_sec: float | None = None
    valid_fraction: float | None = Field(default=None, ge=0, le=1)
    reason: str = ""


class OperationalFrame(StrictModel):
    """Versioned operational snapshot broadcast over the live wire.

    ``plan_version`` is the committed plan version the frame renders; it
    must agree with the version of the active plan carried in ``plans``.
    """

    schema_version: str = "1.0"
    frame_id: int = Field(ge=0)
    sim_time_s: int = Field(ge=0)
    physics_step_s: int = Field(default=5, gt=0)
    plan_version: int = Field(ge=0)
    map_bounds: MapBounds
    carrier: CarrierView | None = None
    uuvs: tuple[UUVView, ...] = ()
    usvs: tuple[USVView, ...] = ()
    communication_links: tuple[CommunicationLinkView, ...] = ()
    brains: tuple[BrainView, ...] = ()
    adversaries: tuple[AdversaryView, ...] = ()
    target_estimates: tuple[TargetEstimateView, ...] = ()
    bearing_rays: tuple[BearingRayView, ...] = ()
    groups: tuple[GroupView, ...] = ()
    regional_plans: dict[str, RegionalPlanView] = Field(default_factory=dict)
    events: tuple[EventView, ...] = ()
    plans: tuple[PlanView, ...] = ()
    ledger: tuple[LedgerView, ...] = ()
    metrics: tuple[MetricView, ...] = ()
    scheme: OperationalSchemeView | None = None
    intelligence: tuple[IntelligenceView, ...] = ()
    plan_timeline: tuple[PlanTimelineView, ...] = ()
    region_timeline: tuple[RegionTimelineView, ...] = ()
    plan_adjustment_suggestions: tuple[PlanAdjustmentSuggestion, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_carrier_relationships(cls, value: Any) -> Any:
        return normalize_legacy_carrier_relationships(value)

    @model_validator(mode="after")
    def plan_version_matches_active_plan(self) -> OperationalFrame:
        for plan in self.plans:
            if plan.status == "active" and plan.version != self.plan_version:
                raise ValueError(
                    f"frame plan_version {self.plan_version} does not match active "
                    f"plan {plan.plan_id!r} version {plan.version}"
                )
        return self

    @model_validator(mode="after")
    def carrier_relationships_match_uuvs(self) -> OperationalFrame:
        if self.carrier is None:
            return self
        uuvs_by_id = {uuv.uuv_id: uuv for uuv in self.uuvs}
        relationships = {
            DeploymentState.ONBOARD: self.carrier.onboard_uuv_ids,
            DeploymentState.DEPLOYED: self.carrier.deployed_uuv_ids,
            DeploymentState.RETURNING: self.carrier.returning_uuv_ids,
        }
        if any(len(ids) != len(set(ids)) for ids in relationships.values()):
            raise ValueError("carrier relationship lists must not contain duplicate IDs")
        relationship_sets = tuple(set(ids) for ids in relationships.values())
        if any(
            left & right
            for index, left in enumerate(relationship_sets)
            for right in relationship_sets[index + 1 :]
        ):
            raise ValueError("carrier relationship lists must be disjoint")
        listed_ids = {uuv_id for ids in relationships.values() for uuv_id in ids}
        for expected_state, ids in relationships.items():
            for uuv_id in ids:
                uuv = uuvs_by_id.get(uuv_id)
                if uuv is None:
                    raise ValueError(f"carrier lists unknown UUV {uuv_id!r}")
                if (
                    uuv.status is UUVStatus.RETURNING
                    and uuv.deployment_state is not DeploymentState.RETURNING
                ) or (
                    uuv.status is UUVStatus.FAILED
                    and uuv.deployment_state is not DeploymentState.FAILED
                ):
                    raise ValueError(f"uuv {uuv_id!r} status contradicts deployment_state")
                if uuv.status is UUVStatus.FAILED or uuv.deployment_state is DeploymentState.FAILED:
                    raise ValueError(f"carrier lists must omit failed UUV {uuv_id!r}")
                if uuv.deployment_state is not expected_state:
                    raise ValueError(
                        f"carrier list {expected_state.value!r} contains {uuv_id!r} "
                        f"with deployment_state {uuv.deployment_state.value!r}"
                    )
        for uuv in self.uuvs:
            if uuv.status is UUVStatus.FAILED or uuv.deployment_state is DeploymentState.FAILED:
                if uuv.uuv_id in listed_ids:
                    raise ValueError(f"carrier lists must omit failed UUV {uuv.uuv_id!r}")
                continue
            if uuv.uuv_id not in listed_ids:
                raise ValueError(f"carrier lists omit non-failed UUV {uuv.uuv_id!r}")
        return self


class EvaluationFrame(StrictModel):
    """Truth-only frame paired to the operational run it evaluates.

    Standalone by design: it never inherits from ``OperationalFrame`` so
    the operational contract cannot structurally admit truth fields.
    ``scenario_id``, ``run_id`` and ``plan_version`` pair this frame to
    the operational run it was collected under.
    """

    schema_version: str = "1.0"
    frame_id: int = Field(ge=0)
    sim_time_s: int = Field(ge=0)
    scenario_id: str
    run_id: str
    plan_version: int = Field(ge=0)
    targets: tuple[TargetTruth, ...] = ()
