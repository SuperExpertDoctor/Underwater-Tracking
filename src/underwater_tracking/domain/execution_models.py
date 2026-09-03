"""Authoritative contracts for UUV-only target execution.

The models in this module deliberately sit above the legacy planning models.
They describe one immutable, internally consistent execution decision that can
be consumed by the mission controller and every live-frame transport.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from itertools import pairwise
from math import isclose, isfinite
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator

from underwater_tracking.domain.models import StrictModel


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Point2 = tuple[FiniteFloat, FiniteFloat]
Covariance2 = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]

IntentLabel = Literal[
    "transit",
    "patrol",
    "loiter",
    "evade",
    "approach",
    "withdraw",
    "unknown",
]
PredictionRegime = Literal["imm", "bspline", "short_history", "boundary_recovery"]
PlanSource = Literal["deterministic", "llm_optimized", "human_revised"]
FreshnessStatus = Literal["fresh", "stale", "unknown"]
RegionStatus = Literal[
    "planned",
    "prepositioning",
    "active",
    "passive",
    "handoff_pending",
    "handoff_completed",
    "monitoring_complete",
    "degraded",
    "uncovered",
]
TaskGroupStatus = Literal[
    "prepositioning",
    "active",
    "handoff_pending",
    "replacing",
    "degraded",
    "complete",
]


class ExecutionModel(StrictModel):
    """Strict immutable base for authoritative execution data."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class GlobalTrackSample(ExecutionModel):
    """One executed target state retained by the public global track."""

    sim_time_s: NonNegativeFloat
    position_xy: Point2
    velocity_xy: Point2 = (0.0, 0.0)


class GlobalTargetTrackView(ExecutionModel):
    target_id: str = Field(min_length=1)
    track_revision: int = Field(ge=1)
    sim_time_s: NonNegativeFloat
    position_xy: Point2
    velocity_xy: Point2
    heading_rad: FiniteFloat
    acceleration_xy: Point2
    turn_rate_rad_s: FiniteFloat
    bounded_history: tuple[GlobalTrackSample, ...] = Field(min_length=1)
    source_event_ids: tuple[str, ...] = Field(min_length=1)
    freshness_status: FreshnessStatus = "fresh"

    @model_validator(mode="before")
    @classmethod
    def normalize_history(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        history = value.get("bounded_history", ())
        normalized: list[object] = []
        for sample in history:
            if isinstance(sample, Mapping):
                normalized.append(sample)
            elif isinstance(sample, (tuple, list)) and len(sample) >= 3:
                normalized.append(
                    {
                        "sim_time_s": sample[0],
                        "position_xy": (sample[1], sample[2]),
                    }
                )
            else:
                normalized.append(sample)
        return {**value, "bounded_history": tuple(normalized)}

    @model_validator(mode="after")
    def validate_history(self) -> GlobalTargetTrackView:
        times = tuple(sample.sim_time_s for sample in self.bounded_history)
        if any(right <= left for left, right in pairwise(times)):
            raise ValueError("global track history timestamps must be strictly increasing")
        if times[-1] > self.sim_time_s:
            raise ValueError("global track history cannot be newer than the track")
        if any(not event_id.strip() for event_id in self.source_event_ids):
            raise ValueError("global track source event IDs must not be empty")
        return self


class IMMModelForecast(ExecutionModel):
    """Public state and likelihood evidence for one IMM branch."""

    model_name: Literal["CV", "CT_LEFT", "CT_RIGHT"]
    state_mean: tuple[FiniteFloat, ...] = Field(min_length=5)
    state_covariance: tuple[tuple[FiniteFloat, ...], ...] = Field(min_length=5)
    model_probability: UnitFloat
    innovation: tuple[FiniteFloat, ...] = ()
    likelihood: NonNegativeFloat
    source_observation_ids: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def normalize_branch_aliases(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for source, target in (
            ("model", "model_name"),
            ("model_id", "model_name"),
            ("probability", "model_probability"),
            ("covariance", "state_covariance"),
        ):
            if target not in normalized and source in normalized:
                normalized[target] = normalized.pop(source)
        return normalized

    @model_validator(mode="after")
    def validate_state(self) -> IMMModelForecast:
        rows = len(self.state_covariance)
        if rows != len(self.state_mean) or any(len(row) != rows for row in self.state_covariance):
            raise ValueError("IMM state covariance must be square and match state_mean")
        for row_index, row in enumerate(self.state_covariance):
            for col_index, value in enumerate(row):
                if not isfinite(value):
                    raise ValueError("IMM state covariance must be finite")
                if not isclose(value, self.state_covariance[col_index][row_index], abs_tol=1e-9):
                    raise ValueError("IMM state covariance must be symmetric")
        return self

    @property
    def model(self) -> str:
        """Compatibility accessor for callers that call the branch a model."""

        return self.model_name


class IMMPredictedTrack(ExecutionModel):
    """Moment-matched forecast over all three operational IMM branches."""

    prediction_id: str = Field(min_length=1)
    prediction_revision: int = Field(ge=1)
    target_id: str = Field(min_length=1)
    origin_sim_time_s: NonNegativeFloat
    times_s: tuple[NonNegativeFloat, ...] = Field(min_length=1)
    centerline_xy: tuple[Point2, ...] = Field(min_length=1)
    covariance_xy: tuple[Covariance2, ...] = Field(min_length=1)
    corridor_radius_m: tuple[PositiveFloat, ...] = Field(min_length=1)
    model_branches: tuple[IMMModelForecast, ...] = Field(min_length=3)
    model_probabilities: Mapping[str, UnitFloat]
    clipping_records: tuple[str, ...] = ()
    source_track_revision: int = Field(ge=1)
    source_observation_ids: tuple[str, ...] = ()
    prediction_regime: PredictionRegime
    bspline_times_s: tuple[NonNegativeFloat, ...] = ()
    bspline_centerline_xy: tuple[Point2, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def normalize_prediction_aliases(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for source, target in (
            ("model_forecasts", "model_branches"),
            ("points_xy", "centerline_xy"),
        ):
            if target not in normalized and source in normalized:
                normalized[target] = normalized.pop(source)
        corridor = normalized.get("corridor_radius_m")
        centerline = normalized.get("centerline_xy", ())
        if isinstance(corridor, (int, float)) and centerline:
            normalized["corridor_radius_m"] = tuple(float(corridor) for _ in centerline)
        return normalized

    @model_validator(mode="after")
    def validate_forecast(self) -> IMMPredictedTrack:
        size = len(self.times_s)
        if len(self.centerline_xy) != size:
            raise ValueError("IMM centerline and times must have equal lengths")
        if len(self.covariance_xy) != size or len(self.corridor_radius_m) != size:
            raise ValueError("IMM forecast arrays must have equal lengths")
        if any(right <= left for left, right in zip(self.times_s, self.times_s[1:])):
            raise ValueError("IMM forecast times must be strictly increasing")
        if self.times_s[0] <= self.origin_sim_time_s:
            raise ValueError("IMM forecast must begin after origin_sim_time_s")
        if len(self.model_branches) != 3:
            raise ValueError("IMM forecast must expose CV, CT_LEFT and CT_RIGHT branches")
        names = tuple(branch.model_name for branch in self.model_branches)
        if set(names) != {"CV", "CT_LEFT", "CT_RIGHT"}:
            raise ValueError("IMM forecast branches must be CV, CT_LEFT and CT_RIGHT")
        if self.bspline_times_s or self.bspline_centerline_xy:
            if len(self.bspline_times_s) != len(self.bspline_centerline_xy):
                raise ValueError("B-spline times and centerline must have equal lengths")
            if any(
                right <= left
                for left, right in zip(self.bspline_times_s, self.bspline_times_s[1:])
            ):
                raise ValueError("B-spline forecast times must be strictly increasing")
        probability_sum = sum(self.model_probabilities.values())
        if not isclose(probability_sum, 1.0, abs_tol=1e-6):
            raise ValueError("IMM model probabilities must sum to one")
        if set(self.model_probabilities) != set(names):
            raise ValueError("IMM model probabilities must name every branch")
        return self


class DeterministicIntentState(ExecutionModel):
    """Rule-derived intent that remains valid when LLM work is unavailable."""

    target_id: str = Field(min_length=1)
    intent_label: IntentLabel
    confidence: UnitFloat
    intent_revision: int = Field(ge=1)
    prediction_revision: int = Field(ge=1)
    rule_version: str = Field(min_length=1)
    features: Mapping[str, FiniteFloat] = Field(default_factory=dict)
    thresholds: Mapping[str, NonNegativeFloat] = Field(default_factory=dict)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_revision_alias(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "intent_revision" not in value and "revision" in value:
            return {**value, "intent_revision": value["revision"]}
        return value


class TaskGroupLifecycle(str, Enum):
    ENTERING = "entering"
    ACTIVE_SCAN = "active_scan"
    PASSIVE_TRACK = "passive_track"
    DEDICATED_TRACK = "dedicated_track"
    DEDICATED_RELEASE_PENDING = "dedicated_release_pending"
    EXITING = "exiting"
    DISAPPEARED = "disappeared"


class GroupSensorMode(str, Enum):
    ACTIVE = "active"
    PASSIVE = "passive"
    OFF = "off"


class TrackingControlState(ExecutionModel):
    """Group-level ownership control for one target's live execution."""

    mode: Literal["regional", "dedicated"] = "regional"
    tracking_owner_group_id: str | None = None
    pending_successor_group_id: str | None = None
    dedicated_release_triggered_at_m: NonNegativeFloat | None = None
    dedicated_release_reason: str | None = None
    source_event_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_group_references(self) -> TrackingControlState:
        if (
            self.tracking_owner_group_id is not None
            and not self.tracking_owner_group_id.strip()
        ):
            raise ValueError("tracking owner group ID must not be empty")
        if (
            self.pending_successor_group_id is not None
            and not self.pending_successor_group_id.strip()
        ):
            raise ValueError("pending successor group ID must not be empty")
        if (
            self.tracking_owner_group_id is not None
            and self.tracking_owner_group_id == self.pending_successor_group_id
        ):
            raise ValueError("tracking owner and pending successor must differ")
        if any(not event_id.strip() for event_id in self.source_event_ids):
            raise ValueError("tracking control source event IDs must not be empty")
        return self


class ExecutionRegion(ExecutionModel):
    """One stable task-region slot in the four-region execution chain."""

    region_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    slot_index: int = Field(ge=1, le=4)
    execution_revision: int = Field(ge=1)
    prediction_id: str = Field(min_length=1)
    geometry: tuple[Point2, ...] = Field(min_length=3)
    center: Point2 = (0.0, 0.0)
    side_length_m: PositiveFloat = 1.0
    centerline_indices: tuple[int, ...] = Field(min_length=1)
    start_s: NonNegativeFloat
    end_s: PositiveFloat
    geometry_revision: int = Field(ge=1)
    predecessor_region_id: str | None = None
    successor_region_id: str | None = None
    handoff_start_s: NonNegativeFloat | None = None
    handoff_end_s: PositiveFloat | None = None
    status: RegionStatus = "planned"
    task_group_id: str | None = None
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_region_aliases(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for source, target in (
            ("polygon", "geometry"),
            ("start_time_s", "start_s"),
            ("end_time_s", "end_s"),
            ("region_revision", "geometry_revision"),
        ):
            if target not in normalized and source in normalized:
                normalized[target] = normalized.pop(source)
        geometry = normalized.get("geometry")
        if isinstance(geometry, (tuple, list)) and len(geometry) >= 4:
            points = tuple((float(point[0]), float(point[1])) for point in geometry)
            min_x = min(point[0] for point in points)
            max_x = max(point[0] for point in points)
            min_y = min(point[1] for point in points)
            max_y = max(point[1] for point in points)
            normalized.setdefault("center", ((min_x + max_x) / 2, (min_y + max_y) / 2))
            normalized.setdefault("side_length_m", max(max_x - min_x, max_y - min_y))
        return normalized

    @model_validator(mode="after")
    def validate_region(self) -> ExecutionRegion:
        expected_id = f"{self.target_id}:task:{self.slot_index:02d}"
        if self.region_id != expected_id:
            raise ValueError("region_id must be the stable target task slot ID")
        if self.end_s <= self.start_s:
            raise ValueError("region end_s must be after start_s")
        if self.predecessor_region_id == self.region_id or self.successor_region_id == self.region_id:
            raise ValueError("region topology cannot self-reference")
        if self.handoff_start_s is not None and self.handoff_end_s is not None:
            if self.handoff_end_s <= self.handoff_start_s:
                raise ValueError("handoff interval must be positive")
            if self.handoff_start_s < self.start_s or self.handoff_end_s > self.end_s:
                raise ValueError("handoff interval must be inside the region window")
        if len(self.geometry) != 4:
            raise ValueError("execution region geometry must contain exactly four square corners")
        points = tuple((float(point[0]), float(point[1])) for point in self.geometry)
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        width = max_x - min_x
        height = max_y - min_y
        if not isclose(width, height, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("execution region geometry must be square")
        expected = {
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        }
        if set(points) != expected:
            raise ValueError("execution region geometry must be an axis-aligned square")
        expected_center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        if not all(
            isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-7)
            for actual, expected_value in zip(self.center, expected_center)
        ):
            raise ValueError("execution region center must match its square geometry")
        if not isclose(self.side_length_m, width, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("execution region side length must match its square geometry")
        return self


class TaskGroupAssignment(ExecutionModel):
    """Exactly one two-UUV group assigned to one execution-region slot."""

    task_group_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    region_id: str = Field(min_length=1)
    execution_revision: int = Field(ge=1)
    member_uuv_ids: tuple[str, str]
    active_verifier_uuv_id: str = Field(min_length=1)
    passive_tracker_uuv_id: str = Field(min_length=1)
    status: TaskGroupStatus = "prepositioning"
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_group_aliases(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
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
    def validate_group(self) -> TaskGroupAssignment:
        members = set(self.member_uuv_ids)
        if len(members) != 2:
            raise ValueError("task group must contain exactly two distinct UUVs")
        if {self.active_verifier_uuv_id, self.passive_tracker_uuv_id} != members:
            raise ValueError("task group roles must cover both member UUVs")
        if self.active_verifier_uuv_id == self.passive_tracker_uuv_id:
            raise ValueError("task group active and passive UUVs must differ")
        expected_target_prefix = f"{self.target_id}:task:"
        if not self.region_id.startswith(expected_target_prefix):
            raise ValueError("task group region must belong to its target")
        return self


class TaskGroupInstance(ExecutionModel):
    """One deployment-aware three-UUV execution instance.

    ``TaskGroupAssignment`` remains available for legacy planning payloads;
    this model is the live UUV-only runtime contract.
    """

    group_instance_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    region_id: str = Field(min_length=1)
    deployment_revision: int = Field(ge=1)
    member_uuv_ids: tuple[str, ...] = Field(min_length=3, max_length=3)
    lifecycle: TaskGroupLifecycle = TaskGroupLifecycle.ENTERING
    sensor_mode: GroupSensorMode = GroupSensorMode.ACTIVE
    ownership_status: str = Field(min_length=1)
    entry_boundary_point: Point2 | None = None
    exit_boundary_point: Point2 | None = None
    source_group_instance_id: str | None = None
    reason: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def require_three_members(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "member_uuv_ids" in value:
            members = value["member_uuv_ids"]
            if not isinstance(members, (tuple, list)) or len(members) != 3:
                raise ValueError("task group instance must contain exactly three members")
        return value

    @model_validator(mode="after")
    def validate_instance(self) -> TaskGroupInstance:
        if len(set(self.member_uuv_ids)) != 3:
            raise ValueError("task group instance member IDs must be unique")
        if self.source_group_instance_id == self.group_instance_id:
            raise ValueError("task group instance cannot source itself")
        required_sensor_mode = {
            TaskGroupLifecycle.ACTIVE_SCAN: GroupSensorMode.ACTIVE,
            TaskGroupLifecycle.PASSIVE_TRACK: GroupSensorMode.PASSIVE,
            TaskGroupLifecycle.DEDICATED_TRACK: GroupSensorMode.PASSIVE,
            TaskGroupLifecycle.DEDICATED_RELEASE_PENDING: GroupSensorMode.PASSIVE,
        }.get(self.lifecycle)
        if required_sensor_mode is not None and self.sensor_mode is not required_sensor_mode:
            raise ValueError(
                f"{self.lifecycle.value} requires matching sensor mode {required_sensor_mode.value}"
            )
        if self.lifecycle is TaskGroupLifecycle.DISAPPEARED and self.sensor_mode is not GroupSensorMode.OFF:
            raise ValueError("disappeared task group instance requires sensor mode off")
        if any(not evidence_id.strip() for evidence_id in self.evidence_ids):
            raise ValueError("task group instance evidence IDs must not be empty")
        return self


class ReserveUUVState(ExecutionModel):
    """A non-spatial UUV waiting for a deterministic boundary replacement."""

    uuv_id: str = Field(min_length=1)
    status: Literal["reserve", "entering", "exiting", "unavailable"] = "reserve"
    priority: NonNegativeFloat = 0.0
    resource_episode: int = Field(default=0, ge=0)


class ExecutionDegradation(ExecutionModel):
    """Explicit health state for a snapshot or a preserved previous plan."""

    status: Literal["nominal", "degraded"] = "nominal"
    reasons: tuple[str, ...] = ()
    active_plan_preserved: bool = False
    failed_components: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def normalize_degraded_flag(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "status" not in value and "degraded" in value:
            return {
                **value,
                "status": "degraded" if value["degraded"] else "nominal",
            }
        return value

    @property
    def degraded(self) -> bool:
        return self.status == "degraded"


class ExecutionContextRef(ExecutionModel):
    """The immutable execution coordinates shared by every operator surface."""

    scenario_id: str = Field(min_length=1)
    execution_revision: int = Field(ge=1)
    frame_id: int = Field(ge=0)
    source_snapshot_revision: int = Field(ge=0)
    target_id: str = Field(min_length=1)
    prediction_id: str = Field(min_length=1)
    prediction_revision: int = Field(ge=1)
    intent_revision: int = Field(ge=1)
    region_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    task_group_ids: tuple[str, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_stable_slots(self) -> ExecutionContextRef:
        expected_regions = tuple(
            f"{self.target_id}:task:{index:02d}" for index in range(1, 5)
        )
        if self.region_ids != expected_regions:
            raise ValueError("execution context must contain four stable region slots")
        if len(set(self.task_group_ids)) != 4:
            raise ValueError("execution context task groups must be unique")
        return self

    @classmethod
    def from_snapshot(
        cls,
        snapshot: OperationalExecutionSnapshot,
        *,
        frame_id: int | None = None,
    ) -> ExecutionContextRef:
        """Build a transport-neutral reference without exposing full state."""

        return cls(
            scenario_id=snapshot.scenario_id,
            execution_revision=snapshot.execution_revision,
            frame_id=(
                frame_id
                if frame_id is not None
                else (
                    snapshot.frame_id
                    if snapshot.frame_id is not None
                    else snapshot.source_snapshot_revision
                )
            ),
            source_snapshot_revision=snapshot.source_snapshot_revision,
            target_id=snapshot.target_id,
            prediction_id=snapshot.prediction_id,
            prediction_revision=snapshot.prediction_revision,
            intent_revision=snapshot.intent_revision,
            region_ids=tuple(region.region_id for region in snapshot.regions),
            task_group_ids=tuple(
                (
                    group.group_instance_id
                    if isinstance(group, TaskGroupInstance)
                    else group.task_group_id
                )
                for group in snapshot.task_groups
            ),
        )


ContributionKind = Literal["algorithm", "llm", "human"]


class ExecutionContribution(ExecutionModel):
    """One bounded, operator-safe explanation contribution."""

    contributor: ContributionKind
    component: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=2000)
    evidence_ids: tuple[str, ...] = ()


class EvidenceReference(ExecutionModel):
    """A read-only, resolved evidence reference safe for the assistant UI."""

    evidence_id: str = Field(min_length=1, max_length=240)
    source_type: str = Field(min_length=1, max_length=120)
    scenario_id: str = Field(min_length=1, max_length=240)
    summary: str = Field(min_length=1, max_length=4000)
    execution_revision: int | None = Field(default=None, ge=1)
    frame_id: int | None = Field(default=None, ge=0)
    source_event_id: str | None = Field(default=None, max_length=240)
    source_decision_id: str | None = Field(default=None, max_length=240)


class EvidenceResolution(ExecutionModel):
    """Result of a bounded read-only evidence lookup."""

    requested_evidence_ids: tuple[str, ...] = ()
    resolved: tuple[EvidenceReference, ...] = ()
    unresolved_evidence: tuple[str, ...] = ()
    read_only: bool = True
    execution_revision: int | None = Field(default=None, ge=1)
    frame_id: int | None = Field(default=None, ge=0)


class ExecutionDecisionRecord(ExecutionModel):
    """Auditable explanation metadata for one committed execution revision."""

    decision_id: str = Field(min_length=1, max_length=240)
    scenario_id: str = Field(min_length=1, max_length=240)
    execution_revision: int = Field(ge=1)
    frame_id: int = Field(ge=0)
    target_id: str = Field(min_length=1)
    prediction_id: str = Field(min_length=1)
    intent_label: IntentLabel
    current_region_id: str = Field(min_length=1)
    next_region_id: str = Field(min_length=1)
    region_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    task_group_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    evidence_ids: tuple[str, ...] = ()
    unresolved_evidence: tuple[str, ...] = ()
    rationale: str = Field(min_length=1, max_length=8000)
    recent_adjustment: str = Field(min_length=1, max_length=1000)
    algorithm_contributions: tuple[ExecutionContribution, ...] = ()
    llm_contributions: tuple[ExecutionContribution, ...] = ()
    human_contributions: tuple[ExecutionContribution, ...] = ()

    @model_validator(mode="after")
    def validate_record_slots(self) -> ExecutionDecisionRecord:
        expected_regions = tuple(
            f"{self.target_id}:task:{index:02d}" for index in range(1, 5)
        )
        if self.region_ids != expected_regions:
            raise ValueError("execution decision must contain four stable region slots")
        if len(set(self.task_group_ids)) != 4:
            raise ValueError("execution decision task groups must be unique")
        return self


class OperationalExecutionSnapshot(ExecutionModel):
    """The single authoritative UUV-only execution decision."""

    scenario_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    execution_revision: int = Field(ge=1)
    source_snapshot_revision: int = Field(ge=0)
    source_sim_time_s: NonNegativeFloat
    prediction_revision: int = Field(ge=1)
    prediction_id: str = Field(min_length=1)
    intent_revision: int = Field(ge=1)
    expert_request_version: int = Field(default=0, ge=0)
    generated_at_s: NonNegativeFloat
    valid_from_s: NonNegativeFloat
    valid_until_s: PositiveFloat
    plan_source: PlanSource
    target_track: GlobalTargetTrackView
    prediction: IMMPredictedTrack
    intent: DeterministicIntentState
    regions: tuple[ExecutionRegion, ...] = Field(min_length=4, max_length=4)
    task_groups: tuple[TaskGroupInstance | TaskGroupAssignment, ...] = Field(min_length=1)
    reserve_uuvs: tuple[ReserveUUVState, ...] = ()
    tracking_control: TrackingControlState = Field(default_factory=TrackingControlState)
    current_region_id: str = Field(min_length=1)
    next_region_id: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    degradation: ExecutionDegradation = Field(default_factory=ExecutionDegradation)
    frame_id: int | None = Field(default=None, ge=0)
    base_execution_revision: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_snapshot(self) -> OperationalExecutionSnapshot:
        region_ids = tuple(region.region_id for region in self.regions)
        expected_region_ids = tuple(f"{self.target_id}:task:{index:02d}" for index in range(1, 5))
        if region_ids != expected_region_ids:
            raise ValueError("execution snapshot must contain four stable ordered region slots")
        if self.valid_until_s <= self.valid_from_s:
            raise ValueError("execution snapshot validity interval must be positive")
        if self.generated_at_s < self.source_sim_time_s:
            raise ValueError("execution snapshot cannot be generated before its source time")
        if self.prediction.target_id != self.target_id or self.intent.target_id != self.target_id:
            raise ValueError("execution target IDs must agree")
        if self.target_track.target_id != self.target_id:
            raise ValueError("target track target ID must agree with execution snapshot")
        if self.prediction.prediction_id != self.prediction_id:
            raise ValueError("prediction ID must agree with execution snapshot")
        if self.prediction.prediction_revision != self.prediction_revision:
            raise ValueError("prediction revision must agree with execution snapshot")
        if self.intent.intent_revision != self.intent_revision:
            raise ValueError("intent revision must agree with execution snapshot")
        if self.intent.prediction_revision != self.prediction_revision:
            raise ValueError("intent must refer to the current prediction revision")
        if self.prediction.source_track_revision != self.target_track.track_revision:
            raise ValueError("prediction must refer to the current global track revision")

        if any(region.execution_revision != self.execution_revision for region in self.regions):
            raise ValueError("all execution regions must use execution_revision")
        if any(region.target_id != self.target_id for region in self.regions):
            raise ValueError("all execution regions must use the snapshot target")
        if any(region.prediction_id != self.prediction_id for region in self.regions):
            raise ValueError("all execution regions must use prediction_id")
        if any(
            region.predecessor_region_id != (region_ids[index - 1] if index else None)
            or region.successor_region_id != (region_ids[index + 1] if index < 3 else None)
            for index, region in enumerate(self.regions)
        ):
            raise ValueError("execution region predecessor/successor topology is incomplete")
        if self.current_region_id not in region_ids or self.next_region_id not in region_ids:
            raise ValueError("current and next regions must belong to the execution chain")

        runtime_groups = tuple(
            group for group in self.task_groups if isinstance(group, TaskGroupInstance)
        )
        legacy_groups = tuple(
            group for group in self.task_groups if isinstance(group, TaskGroupAssignment)
        )
        if runtime_groups and legacy_groups:
            raise ValueError("execution task groups cannot mix runtime and legacy instances")

        if runtime_groups:
            group_ids = tuple(group.group_instance_id for group in runtime_groups)
            if len(set(group_ids)) != len(group_ids):
                raise ValueError("execution task group instance IDs must be unique")
            if len(runtime_groups) > 8:
                raise ValueError("execution snapshot allows at most eight runtime task groups")
            if any(group.target_id != self.target_id for group in runtime_groups):
                raise ValueError("all runtime task groups must use the snapshot target")
            if any(group.region_id not in set(region_ids) for group in runtime_groups):
                raise ValueError("runtime task group region must belong to the snapshot")
            owner_groups = tuple(
                group for group in runtime_groups if group.ownership_status == "owner"
            )
            if len(owner_groups) > 1:
                raise ValueError("execution snapshot allows at most one tracking owner")
            owner_id = self.tracking_control.tracking_owner_group_id
            if owner_id is not None and owner_id not in set(group_ids):
                raise ValueError("tracking owner group must exist in execution task groups")
            if owner_groups and owner_groups[0].group_instance_id != owner_id:
                raise ValueError("task group owner status must match tracking control")
            if (
                self.tracking_control.mode == "dedicated"
                and owner_id is None
            ):
                raise ValueError("dedicated execution requires a tracking owner")
        elif legacy_groups:
            group_ids = tuple(group.task_group_id for group in legacy_groups)
            if len(set(group_ids)) != 4:
                raise ValueError("execution task group IDs must be unique")
            if any(group.execution_revision != self.execution_revision for group in legacy_groups):
                raise ValueError("all task groups must use execution_revision")
            if any(group.target_id != self.target_id for group in legacy_groups):
                raise ValueError("all task groups must use the snapshot target")
            if {group.region_id for group in legacy_groups} != set(region_ids):
                raise ValueError("exactly one task group is required for every execution region")
            region_group_ids = {region.task_group_id for region in self.regions}
            if region_group_ids != set(group_ids):
                raise ValueError("every execution region must name its task group")
        else:
            raise ValueError("execution snapshot requires at least one task group")
        all_members: list[str] = []
        for group in self.task_groups:
            all_members.extend(group.member_uuv_ids)
        if len(all_members) != len(set(all_members)):
            raise ValueError("a UUV cannot belong to more than one task group")
        reserve_ids = tuple(reserve.uuv_id for reserve in self.reserve_uuvs)
        if len(reserve_ids) != len(set(reserve_ids)):
            raise ValueError("reserve UUV IDs must be unique")
        if set(all_members) & set(reserve_ids):
            raise ValueError("a UUV cannot be both an execution member and a reserve")
        all_evidence = self.evidence_ids
        if len(all_evidence) != len(set(all_evidence)):
            raise ValueError("execution evidence IDs must be unique")
        if any(not evidence_id.strip() for evidence_id in all_evidence):
            raise ValueError("execution evidence IDs must not be empty")
        if self.base_execution_revision is not None and self.base_execution_revision >= self.execution_revision:
            raise ValueError("base execution revision must precede execution_revision")
        return self


__all__ = [
    "ContributionKind",
    "DeterministicIntentState",
    "EvidenceReference",
    "EvidenceResolution",
    "ExecutionContextRef",
    "ExecutionContribution",
    "ExecutionDecisionRecord",
    "ExecutionDegradation",
    "ExecutionRegion",
    "GlobalTargetTrackView",
    "GlobalTrackSample",
    "GroupSensorMode",
    "IMMModelForecast",
    "IMMPredictedTrack",
    "OperationalExecutionSnapshot",
    "ReserveUUVState",
    "TaskGroupAssignment",
    "TaskGroupInstance",
    "TaskGroupLifecycle",
    "TrackingControlState",
]
