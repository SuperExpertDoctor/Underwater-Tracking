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
    IntentMotive,
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


OperationalStage = Literal[
    "task_execution",
    "event_trigger",
    "human_feedback",
    "dynamic_adjustment",
]


class OperationalThinkingSummary(StrictModel):
    """Bounded operator rationale for one planning epoch."""

    epoch_id: str = Field(min_length=1, max_length=240)
    plan_version: int = Field(ge=0)
    trigger: Literal["initialization", "critical_event", "expert_feedback"]
    summary: str = Field(min_length=1, max_length=240)
    source_event_ids: tuple[str, ...] = Field(default=(), max_length=32)


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


class PredictionGridCellView(StrictModel):
    """Estimator-safe evidence for one deterministic prediction-grid cell."""

    region_id: str
    target_id: str
    revision: int = Field(ge=1)
    grid_x: int
    grid_y: int
    bounds: MapBounds
    probability: float = Field(ge=0, le=1)
    first_entry_s: int = Field(ge=0)
    last_exit_s: int = Field(ge=0)
    imm_model_probabilities: dict[str, float] = Field(default_factory=dict)
    covariance_summary: tuple[float, float, float]
    intent_label: str
    intent_confidence: float = Field(ge=0, le=1)


class PredictionGridView(StrictModel):
    """Versioned grid evidence rendered beneath the mission regions."""

    target_id: str
    revision: int = Field(ge=1)
    origin: Point2D
    cell_size_m: float = Field(gt=0)
    centerline_region_ids: tuple[str, ...] = ()
    cells: tuple[PredictionGridCellView, ...] = ()


class RegionalMissionView(StrictModel):
    """UUV-only executable region state for live and replay consumers."""

    region_id: str
    target_id: str
    cell_ids: tuple[str, ...] = ()
    geometry: tuple[Point2D, ...] = ()
    entry_s: int = Field(ge=0)
    exit_s: int = Field(gt=0)
    lifecycle: Literal[
        "PLANNED",
        "CARRIER_DEPLOYING",
        "ACTIVE_SCAN",
        "PASSIVE_TRACK",
        "HANDOFF_PENDING",
        "TRACKING_COMPLETED",
        "CARRIER_RECOVERY",
        "RECOVERED",
        "DEGRADED",
        "UNCOVERED",
    ]
    active_scan_uuv_ids: tuple[str, ...] = ()
    passive_track_uuv_ids: tuple[str, ...] = ()
    reserve_uuv_ids: tuple[str, ...] = ()
    coverage: float = Field(ge=0, le=1)
    tracking_quality: float = Field(ge=0, le=1)
    handoff_from: str | None = None
    handoff_to: str | None = None
    carrier_task_id: str | None = None
    carrier_id: str | None = None
    degraded_reasons: tuple[str, ...] = ()
    plan_revision: int = Field(default=1, ge=1)


class CarrierMissionView(StrictModel):
    """Carrier logistics state; the carrier is never a sensor platform."""

    carrier_id: str
    role: Literal["carrier", "mother_ship"] = "carrier"
    home_battle_group_id: str
    mission_type: Literal["DEPLOY", "RECOVER", "DEPLOY_AND_RECOVER"]
    route_status: Literal[
        "TO_DEPLOY",
        "DEPLOYING",
        "EN_ROUTE_NEXT_DEPLOY",
        "RETURNING_TO_FLEET",
        "RECOVERING",
        "RENDEZVOUS_BLOCKED",
        "COMPLETE",
        "FAILED",
    ]
    route: tuple[Point2D, ...] = ()
    stop_ids: tuple[str, ...] = ()
    onboard_uuv_ids: tuple[str, ...] = ()
    ready_uuv_ids: tuple[str, ...] = ()
    reserved_uuv_ids: tuple[str, ...] = ()
    recoverable_uuv_ids: tuple[str, ...] = ()


class MissionEventView(StrictModel):
    """Structured lifecycle event projected from MissionController."""

    event_id: str
    sim_time_s: int = Field(ge=0)
    event_type: str
    level: EventLevel
    entity_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


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
    physically_exposed: bool = True
    display_opacity: float = Field(default=1.0, ge=0, le=1)
    position: Point2D
    heading_rad: float
    sensor_heading_rad: float | None = None
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
        if self.status in {UUVStatus.TRACK, UUVStatus.SCAN} and (
            self.deployment_state is not DeploymentState.DEPLOYED
        ):
            raise ValueError("track and scan status require deployed deployment_state")
        if self.deployment_state in {DeploymentState.RETURNING, DeploymentState.FAILED} and (
            self.status is not UUVStatus.UNAVAILABLE
        ):
            raise ValueError("returning and failed deployment states require unavailable status")
        return self


class UUVResourceView(StrictModel):
    """Planner-visible UUV endurance and capability telemetry."""

    uuv_id: str
    carrier_id: str | None = None
    mileage_m: float = Field(ge=0)
    energy_fraction: float = Field(ge=0, le=1)
    healthy: bool = True
    capability_active: bool = True
    deployment_state: str
    resource_episode: int = Field(ge=0)


class CarrierView(StrictModel):
    carrier_id: str
    role: Literal["carrier", "mother_ship"] = "carrier"
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
    ranked_motives: tuple[IntentMotive, ...] = ()


class PredictionDiffView(StrictModel):
    diff_id: str
    state: Literal[
        "stable",
        "accumulating",
        "suspected",
        "verifying",
        "confirmed",
        "reset",
        "unavailable",
    ]
    status: str
    reason: str | None = None
    absolute_rms_m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    normalized_rms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    absolute_floor_m: float = Field(gt=0, allow_inf_nan=False)
    normalized_threshold: float = Field(gt=0, allow_inf_nan=False)
    consecutive_count: int = Field(ge=0)
    confirmation_cycles: int = Field(ge=1)
    previous_prediction_id: str | None = None
    current_prediction_id: str
    leading_model_changed: bool = False
    js_distance: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    suspicion_event_id: str | None = None
    confirmed_intent: str | None = None
    resulting_plan_revision: int | None = Field(default=None, ge=1)


class PredictionHealthView(StrictModel):
    """Transport health for an assessed prediction or a legacy replay."""

    status: Literal["valid", "degraded", "unavailable", "legacy_unknown"]
    regime: Literal[
        "imm",
        "bspline",
        "short_history",
        "boundary_recovery",
        "legacy_unknown",
    ]
    reason_codes: tuple[str, ...] = ()
    source_track_age_s: float = Field(ge=0, allow_inf_nan=False)
    clipped_point_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    maximum_radius_m: float = Field(ge=0, allow_inf_nan=False)
    raw_prediction_id: str | None = None


class PredictionCorridorView(StrictModel):
    """Predicted track centerline with a radius envelope per sample."""

    prediction_id: str = Field(min_length=1)
    prediction_revision: int = Field(ge=1)
    origin_sim_time_s: float = Field(ge=0, allow_inf_nan=False)
    health: PredictionHealthView
    horizon_s: float = Field(gt=0)
    sample_step_s: float = Field(gt=0)
    centerline_xy: tuple[Point2D, ...] = ()
    radius_m: tuple[float, ...] = ()
    point_confidence: tuple[float, ...] = ()
    diff: PredictionDiffView | None = None

    @model_validator(mode="after")
    def prediction_arrays_match(self) -> PredictionCorridorView:
        size = len(self.centerline_xy)
        if len(self.radius_m) != size or len(self.point_confidence) != size:
            raise ValueError(
                "prediction centerline, radius, and point confidence lengths must match"
            )
        return self


class WorldModelEvidenceView(StrictModel):
    key: str
    source: Literal[
        "imm",
        "bspline",
        "tracking_context",
        "uuv_projection",
        "map_bounds",
        "observability",
    ]
    value: float = Field(allow_inf_nan=False)
    threshold: float | None = Field(default=None, allow_inf_nan=False)
    unit: str = "1"
    description: str


class WorldModelEventView(StrictModel):
    event_id: str
    event_type: str
    horizon: Literal["H1", "H2", "H3", "H4"]
    predicted_time_s: float = Field(ge=0, allow_inf_nan=False)
    time_to_event_s: float = Field(ge=0, allow_inf_nan=False)
    predicted_position: Point2D
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    level: EventLevel
    rule_id: str
    summary: str
    evidence: tuple[WorldModelEvidenceView, ...] = ()


class WorldModelHorizonView(StrictModel):
    name: Literal["H1", "H2", "H3", "H4"]
    start_offset_s: float = Field(ge=0, allow_inf_nan=False)
    end_offset_s: float = Field(gt=0, allow_inf_nan=False)
    sample_count: int = Field(ge=0)
    covered: bool


class WorldModelForecastView(StrictModel):
    model_kind: Literal["rule_demo"] = "rule_demo"
    model_version: str
    control_authority: Literal[False] = False
    as_of_s: float = Field(ge=0, allow_inf_nan=False)
    source_prediction_id: str
    source_observation_ids: tuple[str, ...] = ()
    source_observability_event_ids: tuple[str, ...] = ()
    source_plan_revision: int | None = Field(default=None, ge=1)
    data_status: Literal["ready", "degraded"]
    trajectory_fallback_used: bool
    imm_model_probabilities: dict[str, float] = Field(default_factory=dict)
    horizons: tuple[WorldModelHorizonView, ...] = ()
    events: tuple[WorldModelEventView, ...] = ()
    warnings: tuple[str, ...] = ()


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
    world_model: WorldModelForecastView | None = None
    quality: EstimateQualityView
    classification: Literal["submarine", "decoy", "unknown"] = "unknown"
    last_ping_s: int | None = None
    estimated_depth_m: float | None = Field(default=None, ge=0)
    depth_uncertainty_m: float | None = Field(default=None, ge=0)
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
    tracking_mode: Literal["heuristic_uuv"]
    uuv_roles: tuple[str, ...] = ()
    sonar_policy: SonarPolicy | None = None
    communication: CommunicationRequirement | None = None
    communication_links: tuple[str, ...] = ()
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
    platform_kind: Literal["uuv"]
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
    status: Literal[
        "unconfigured",
        "ready",
        "running",
        "succeeded",
        "degraded",
        "failed",
        # Legacy replay values remain readable; live publishers never create them.
        "online",
        "paused",
        "unknown",
    ]
    last_update_s: int | None = Field(default=None, ge=0)
    message: str = ""
    connected_platform_ids: tuple[str, ...] = ()
    operation: str | None = None
    evidence_platform_ids: tuple[str, ...] = ()


class PlannedAssignmentView(StrictModel):
    """A mission-controller assignment that may still be onboard."""

    target_id: str
    region_id: str
    uuv_ids: tuple[str, ...]
    carrier_id: str
    plan_version: int = Field(ge=0)
    status: Literal["planned", "transporting", "ready_to_deploy"]


class ExecutionGroupView(StrictModel):
    """A physically exposed UUV execution group."""

    group_id: str
    target_id: str
    region_id: str
    member_ids: tuple[str, ...]
    mode: Literal["active_scan", "passive_track", "returning"]


class ExecutionRegionView(StrictModel):
    """One stable executable region projected from the authoritative snapshot."""

    region_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    slot_index: int = Field(ge=1, le=4)
    execution_revision: int = Field(ge=1)
    prediction_id: str = Field(min_length=1)
    geometry: tuple[Point2D, ...] = Field(min_length=3)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    geometry_revision: int = Field(ge=1)
    predecessor_region_id: str | None = None
    successor_region_id: str | None = None
    handoff_start_s: float | None = Field(default=None, ge=0)
    handoff_end_s: float | None = Field(default=None, gt=0)
    status: Literal[
        "planned",
        "prepositioning",
        "active",
        "passive",
        "handoff_pending",
        "handoff_completed",
        "monitoring_complete",
        "degraded",
        "uncovered",
    ] = "planned"
    task_group_id: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_window(self) -> ExecutionRegionView:
        if self.end_s <= self.start_s:
            raise ValueError("execution region end_s must be after start_s")
        return self


class TaskGroupView(StrictModel):
    """One two-UUV execution group projected for the operator."""

    task_group_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    region_id: str = Field(min_length=1)
    execution_revision: int = Field(ge=1)
    member_uuv_ids: tuple[str, ...] = Field(min_length=2, max_length=2)
    active_verifier_uuv_id: str = Field(min_length=1)
    passive_tracker_uuv_id: str = Field(min_length=1)
    status: Literal[
        "prepositioning",
        "active",
        "handoff_pending",
        "replacing",
        "degraded",
        "complete",
    ] = "prepositioning"
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_group_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for source, target in (
            ("group_id", "task_group_id"),
            ("uuv_ids", "member_uuv_ids"),
            ("active_uuv_id", "active_verifier_uuv_id"),
            ("passive_uuv_id", "passive_tracker_uuv_id"),
        ):
            if target not in normalized and source in normalized:
                normalized[target] = normalized.pop(source)
        return normalized

    @model_validator(mode="after")
    def validate_roles(self) -> TaskGroupView:
        members = set(self.member_uuv_ids)
        if len(members) != 2:
            raise ValueError("execution task group members must be distinct")
        if {self.active_verifier_uuv_id, self.passive_tracker_uuv_id} != members:
            raise ValueError("execution task group roles must cover both members")
        return self

    @property
    def group_id(self) -> str:
        return self.task_group_id

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self.member_uuv_ids


class ExecutionView(StrictModel):
    """The single execution projection carried by a live or replay frame."""

    target_id: str = Field(min_length=1)
    prediction_id: str = Field(min_length=1)
    execution_revision: int = Field(ge=1)
    source_snapshot_revision: int = Field(ge=0)
    prediction_revision: int = Field(ge=1)
    intent_revision: int = Field(ge=1)
    data_age_s: float = Field(ge=0)
    valid_from_s: float = Field(ge=0, allow_inf_nan=False)
    valid_until_s: float = Field(gt=0, allow_inf_nan=False)
    health_status: Literal["current", "degraded", "expired", "failed"]
    health_reasons: tuple[str, ...] = ()
    region_generation_mode: Literal[
        "imm",
        "degraded_prediction",
        "boundary_recovery",
        "reprojected_previous",
    ]
    plan_source: Literal["deterministic", "llm_optimized", "human_revised"]
    current_region_id: str = Field(min_length=1)
    next_region_id: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    regions: tuple[ExecutionRegionView, ...] = Field(min_length=4, max_length=4)
    task_groups: tuple[TaskGroupView, ...] = Field(min_length=4, max_length=4)
    reserve_uuv_ids: tuple[str, ...] = ()
    degraded: bool = False
    degradation_reasons: tuple[str, ...] = ()
    active_plan_preserved: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_data_status(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "data_status" not in value:
            return value
        normalized = dict(value)
        legacy_status = normalized.pop("data_status")
        normalized.setdefault(
            "health_status",
            {
                "current": "current",
                "stale": "degraded",
                "unavailable": "failed",
            }.get(legacy_status, legacy_status),
        )
        return normalized

    @model_validator(mode="after")
    def validate_consistency(self) -> ExecutionView:
        if self.valid_until_s <= self.valid_from_s:
            raise ValueError("execution validity interval must be positive")
        expected_ids = tuple(
            f"{self.target_id}:task:{index:02d}" for index in range(1, 5)
        )
        region_ids = tuple(region.region_id for region in self.regions)
        if region_ids != expected_ids:
            raise ValueError("execution view must contain four ordered stable regions")
        if any(region.target_id != self.target_id for region in self.regions):
            raise ValueError("execution region targets must agree")
        revisions = {region.execution_revision for region in self.regions}
        if revisions != {self.execution_revision}:
            raise ValueError("execution region revisions must agree")
        group_ids = tuple(group.task_group_id for group in self.task_groups)
        if len(set(group_ids)) != 4:
            raise ValueError("execution task groups must be unique")
        if {group.region_id for group in self.task_groups} != set(region_ids):
            raise ValueError("execution task groups must cover all regions")
        if any(group.execution_revision != self.execution_revision for group in self.task_groups):
            raise ValueError("execution task group revisions must agree")
        members = [
            member
            for group in self.task_groups
            for member in group.member_uuv_ids
        ]
        if len(members) != len(set(members)):
            raise ValueError("execution task group members must be disjoint")
        if set(members) & set(self.reserve_uuv_ids):
            raise ValueError("reserve UUVs must be disjoint from task groups")
        if self.current_region_id not in region_ids or self.next_region_id not in region_ids:
            raise ValueError("current and next regions must belong to the chain")
        return self


class FrameConsistencyReport(StrictModel):
    """Machine-readable consistency result for the frame execution projection."""

    valid: bool
    execution_revision: int | None = Field(default=None, ge=1)
    source_snapshot_revision: int | None = Field(default=None, ge=0)
    errors: tuple[str, ...] = ()

    @property
    def degraded(self) -> bool:
        return not self.valid


class BrainActivityRecord(StrictModel):
    """The latest durable activity for one configured decision role."""

    brain_id: str
    role: Literal["master", "slave", "adversary"]
    status: Literal[
        "unconfigured", "ready", "running", "succeeded", "degraded", "failed"
    ]
    operation: str | None = None
    sim_time_s: int | None = Field(default=None, ge=0)
    evidence_platform_ids: tuple[str, ...] = ()
    message: str = ""


class PlanningHealthView(StrictModel):
    """Non-blocking planning lifecycle status exposed by the health API."""

    status: Literal[
        "idle", "queued", "running", "committed", "invalidated", "rejected", "failed", "awaiting_retry", "degraded"
    ]
    epoch_id: str | None = None
    base_physics_revision: int | None = Field(default=None, ge=0)
    current_physics_revision: int | None = Field(default=None, ge=0)
    latest_physics_revision: int | None = Field(default=None, ge=0)
    base_sim_time_s: int | None = Field(default=None, ge=0)
    current_sim_time_s: int | None = Field(default=None, ge=0)
    latest_sim_time_s: int | None = Field(default=None, ge=0)
    data_age_s: int | None = Field(default=None, ge=0)
    deadline_utc_ms: int | None = Field(default=None, ge=0)
    node: str | None = None
    attempt: int = Field(default=0, ge=0)
    planning_epoch_invariant_failures: int = Field(default=0, ge=0)
    queued_event_count: int = Field(default=0, ge=0)
    last_result_status: str | None = None
    last_error: str | None = None


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
    escape_region_id: str | None = None
    decision_source: Literal["llm", "mission_route", "boundary_avoidance", "safe_hold"] | None = None
    guidance_id: str | None = None
    guidance_waypoint_xy: Point2D | None = None
    guidance_speed_mps: float | None = Field(default=None, ge=0)
    guidance_heading_rad: float | None = None
    guidance_valid_until_s: int | None = Field(default=None, ge=0)
    degraded_reason: str | None = None


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

    kind: Literal["event", "evidence", "directive"]
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
    scenario_id: str | None = None
    frame_id: int = Field(ge=0)
    sim_time_s: int = Field(ge=0)
    physics_step_s: int = Field(default=5, gt=0)
    plan_version: int = Field(ge=0)
    run_phase: Literal[
        "created",
        "bootstrap_planning",
        "awaiting_retry",
        "running",
        "completed",
        "stopping",
        "stopped",
        "failed",
    ] = "running"
    planning_snapshot_revision: int | None = Field(default=None, ge=0)
    planning_sim_time_s: int | None = Field(default=None, ge=0)
    planning_data_age_s: int | None = Field(default=None, ge=0)
    planning_data_status: Literal["current", "stale", "unavailable"] = "unavailable"
    operational_stage_flags: tuple[OperationalStage, ...] = ()
    # Operator-safe rationale summary; this is deliberately not raw model
    # chain-of-thought or an unbounded prompt/response transcript.
    llm_thinking: str | None = None
    llm_thinking_trigger: str | None = None
    llm_thinking_epoch_id: str | None = None
    llm_thinking_source_event_ids: tuple[str, ...] = ()
    uuv_only: bool = False
    map_bounds: MapBounds
    planning: PlanningHealthView | None = None
    execution: ExecutionView | None = None
    execution_consistency: FrameConsistencyReport | None = None
    # Audit identifiers are safe to expose; private event payloads remain out
    # of the blue-planning event stream.
    operator_audit_event_ids: tuple[str, ...] = ()
    carrier: CarrierView | None = None
    carriers: tuple[CarrierView, ...] = ()
    uuvs: tuple[UUVView, ...] = ()
    communication_links: tuple[CommunicationLinkView, ...] = ()
    brains: tuple[BrainView, ...] = ()
    planned_assignments: tuple[PlannedAssignmentView, ...] = ()
    execution_groups: tuple[ExecutionGroupView, ...] = ()
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
    prediction_grids: tuple[PredictionGridView, ...] = ()
    regional_missions: tuple[RegionalMissionView, ...] = ()
    carrier_missions: tuple[CarrierMissionView, ...] = ()
    mission_events: tuple[MissionEventView, ...] = ()
    uuv_mission_modes: dict[str, str] = Field(default_factory=dict)
    uuv_resources: tuple[UUVResourceView, ...] = ()

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
    def execution_projection_is_consistent(self) -> OperationalFrame:
        if self.execution is None:
            if self.execution_consistency is not None:
                raise ValueError("execution consistency requires an execution view")
            return self
        if self.execution_consistency is not None:
            if not self.execution_consistency.valid:
                raise ValueError("invalid execution consistency report cannot be live")
            if (
                self.execution_consistency.execution_revision
                != self.execution.execution_revision
            ):
                raise ValueError("execution consistency revision must match execution")
        matching_estimate = next(
            (
                estimate
                for estimate in self.target_estimates
                if estimate.target_id == self.execution.target_id
            ),
            None,
        )
        if matching_estimate is not None and matching_estimate.prediction is not None:
            prediction = matching_estimate.prediction
            if prediction.prediction_revision != self.execution.prediction_revision:
                raise ValueError(
                    "prediction revision must match the execution prediction revision"
                )
            execution_prediction_ids = {
                region.prediction_id for region in self.execution.regions
            }
            if execution_prediction_ids != {prediction.prediction_id}:
                raise ValueError("prediction ID must match the execution prediction ID")
        return self

    @model_validator(mode="after")
    def carrier_relationships_match_uuvs(self) -> OperationalFrame:
        carriers = self.carriers or ((self.carrier,) if self.carrier is not None else ())
        if not carriers:
            return self
        uuvs_by_id = {uuv.uuv_id: uuv for uuv in self.uuvs}
        listed_ids: set[str] = set()
        for carrier in carriers:
            relationships = {
                DeploymentState.ONBOARD: carrier.onboard_uuv_ids,
                DeploymentState.DEPLOYED: carrier.deployed_uuv_ids,
                DeploymentState.RETURNING: carrier.returning_uuv_ids,
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
            carrier_listed_ids = {
                uuv_id for ids in relationships.values() for uuv_id in ids
            }
            if listed_ids & carrier_listed_ids:
                raise ValueError("carrier relationship lists must be disjoint across carriers")
            listed_ids.update(carrier_listed_ids)
            for expected_state, ids in relationships.items():
                for uuv_id in ids:
                    uuv = uuvs_by_id.get(uuv_id)
                    if uuv is None:
                        raise ValueError(f"carrier lists unknown UUV {uuv_id!r}")
                    if uuv.deployment_state is DeploymentState.FAILED:
                        raise ValueError(f"carrier lists must omit failed UUV {uuv_id!r}")
                    if uuv.deployment_state is not expected_state:
                        raise ValueError(
                            f"carrier list {expected_state.value!r} contains {uuv_id!r} "
                            f"with deployment_state {uuv.deployment_state.value!r}"
                        )
        for uuv in self.uuvs:
            if uuv.deployment_state is DeploymentState.FAILED:
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
