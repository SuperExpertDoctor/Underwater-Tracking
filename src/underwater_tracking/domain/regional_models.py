from __future__ import annotations

from collections.abc import Mapping
from math import isclose
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator

from underwater_tracking.domain.models import StrictModel

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


def _orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _on_segment(
    first: tuple[float, float],
    second: tuple[float, float],
    point: tuple[float, float],
) -> bool:
    return (
        min(first[0], second[0]) <= point[0] <= max(first[0], second[0])
        and min(first[1], second[1]) <= point[1] <= max(first[1], second[1])
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    first_orientation = _orientation(first_start, first_end, second_start)
    second_orientation = _orientation(first_start, first_end, second_end)
    third_orientation = _orientation(second_start, second_end, first_start)
    fourth_orientation = _orientation(second_start, second_end, first_end)
    tolerance = 1e-9
    if (
        ((first_orientation > tolerance and second_orientation < -tolerance)
         or (first_orientation < -tolerance and second_orientation > tolerance))
        and ((third_orientation > tolerance and fourth_orientation < -tolerance)
             or (third_orientation < -tolerance and fourth_orientation > tolerance))
    ):
        return True
    return (
        abs(first_orientation) <= tolerance
        and _on_segment(first_start, first_end, second_start)
        or abs(second_orientation) <= tolerance
        and _on_segment(first_start, first_end, second_end)
        or abs(third_orientation) <= tolerance
        and _on_segment(second_start, second_end, first_start)
        or abs(fourth_orientation) <= tolerance
        and _on_segment(second_start, second_end, first_end)
    )


def _is_simple_perimeter(points: tuple[tuple[float, float], ...]) -> bool:
    edge_count = len(points)
    for first_index in range(edge_count):
        first_start = points[first_index]
        first_end = points[(first_index + 1) % edge_count]
        for second_index in range(first_index + 1, edge_count):
            if second_index in {
                first_index,
                (first_index + 1) % edge_count,
                (first_index - 1) % edge_count,
            }:
                continue
            second_start = points[second_index]
            second_end = points[(second_index + 1) % edge_count]
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return False
    return True
PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

UUVRole = Literal["passive_tracker", "active_verifier", "handoff_reserve"]
RegionRole = UUVRole
RegionAssignmentStatus = Literal["planned", "active", "handed_off", "degraded", "uncovered"]
RegionCoverageMode = Literal["required", "reserve", "optional"]
UUVRegionalTrackingMode = Literal["active_scan", "passive_track", "handoff_reserve"]
RegionTrackingMode = Literal["heuristic_uuv"]


class RegionSlotPolicy(StrictModel):
    """LLM-editable semantics for one existing execution-region slot.

    Geometry, task-group membership, and physical routes deliberately do not
    belong to this model.  They remain deterministic outputs of the execution
    planner and therefore cannot be smuggled into an LLM revision.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    region_id: str = Field(min_length=1)
    slot_index: int = Field(ge=1, le=4)
    priority: UnitFloat = 0.5
    window_start_ratio: UnitFloat = 0.0
    window_end_ratio: UnitFloat = 1.0
    width_scale: float = Field(default=1.0, ge=0.5, le=2.0, allow_inf_nan=False)
    overlap_ratio: float = Field(default=0.1, ge=0.0, le=0.35, allow_inf_nan=False)
    tracking_mode: UUVRegionalTrackingMode = "passive_track"
    sonar_mode: Literal["passive", "active", "passive_then_active"] = "passive"
    task_group_role: UUVRole = "passive_tracker"
    reserve_priority: UnitFloat = 0.0
    rationale: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_semantic_aliases(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        aliases = {
            "slot": "slot_index",
            "priority_score": "priority",
            "start_ratio": "window_start_ratio",
            "end_ratio": "window_end_ratio",
            "time_window_start_ratio": "window_start_ratio",
            "time_window_end_ratio": "window_end_ratio",
            "handoff_overlap_ratio": "overlap_ratio",
            "role": "task_group_role",
            "group_role": "task_group_role",
        }
        for source, target in aliases.items():
            if target not in normalized and source in normalized:
                normalized[target] = normalized.pop(source)
        window = normalized.pop("time_window_ratio", None)
        if (
            window is not None
            and isinstance(window, (tuple, list))
            and len(window) == 2
        ):
            normalized.setdefault("window_start_ratio", window[0])
            normalized.setdefault("window_end_ratio", window[1])
        return normalized

    @model_validator(mode="after")
    def validate_window(self) -> RegionSlotPolicy:
        if self.window_end_ratio <= self.window_start_ratio:
            raise ValueError("region slot time window must have positive duration")
        expected_id = f"{self.region_id.split(':task:')[0]}:task:{self.slot_index:02d}"
        if ":task:" in self.region_id and self.region_id != expected_id:
            raise ValueError("region slot ID must agree with its slot index")
        return self


class ExecutionStrategyProposal(StrictModel):
    """Constrained semantic delta proposed for one target's execution chain."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    target_id: str = Field(min_length=1)
    base_execution_revision: int = Field(ge=0)
    resource_revision: int = Field(default=0, ge=0)
    manual_revision: int = Field(default=0, ge=0)
    region_slots: tuple[RegionSlotPolicy, ...] = Field(min_length=4, max_length=4)
    intent_explanation: str = ""
    recommendation: Literal["hold_current", "revise"] = "revise"
    hold_current: bool = False
    rationale: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_proposal_aliases(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for source, target in (
            ("revision", "base_execution_revision"),
            ("base_revision", "base_execution_revision"),
            ("slots", "region_slots"),
            ("policies", "region_slots"),
            ("region_policies", "region_slots"),
            ("decision", "recommendation"),
        ):
            if target not in normalized and source in normalized:
                normalized[target] = normalized.pop(source)
        if normalized.get("recommendation") == "hold":
            normalized["recommendation"] = "hold_current"
        if normalized.get("recommendation") == "hold_current":
            normalized.setdefault("hold_current", True)
        return normalized

    @model_validator(mode="after")
    def validate_topology(self) -> ExecutionStrategyProposal:
        expected_ids = tuple(
            f"{self.target_id}:task:{index:02d}" for index in range(1, 5)
        )
        actual_ids = tuple(slot.region_id for slot in self.region_slots)
        if actual_ids != expected_ids:
            raise ValueError(
                "execution strategy must address the four existing target task slots"
            )
        if tuple(slot.slot_index for slot in self.region_slots) != (1, 2, 3, 4):
            raise ValueError("execution strategy slots must be ordered 1 through 4")
        if self.hold_current != (self.recommendation == "hold_current"):
            raise ValueError("hold_current must agree with recommendation")
        return self


StrategyHealthStatus = Literal[
    "running",
    "validated",
    "committed",
    "invalid_output",
    "provider_timeout",
    "provider_unavailable",
    "stale",
    "resource_conflict",
    "geometry_rejected",
    "preserving_active_plan",
]


class StrategyValidationReport(StrictModel):
    """Auditable result of validating a semantic strategy revision."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    status: StrategyHealthStatus
    valid: bool = False
    proposal: ExecutionStrategyProposal | None = None
    target_id: str = ""
    request_hash: str = ""
    response_hash: str = ""
    model_id: str = ""
    prompt_version: str = ""
    base_execution_revision: int | None = None
    preserved_execution_revision: int | None = None
    active_plan_preserved: bool = True
    errors: tuple[str, ...] = ()
    rejected_fields: tuple[str, ...] = ()
    accepted_region_ids: tuple[str, ...] = ()
    failed_fields: tuple[str, ...] = ()
    retry_condition: str | None = None

    @property
    def degraded(self) -> bool:
        return not self.valid or self.status not in {"validated", "committed"}


class GridSpec(StrictModel):
    origin_xy: tuple[FiniteFloat, FiniteFloat] = (0.0, 0.0)
    map_coordinate_convention: str = "global_xy_m"
    target_grid_cells: int = Field(default=64, ge=1)
    min_cell_size_m: PositiveFinite = 125.0
    max_cell_size_m: PositiveFinite = 2_000.0
    cell_size_rounding_m: PositiveFinite = 50.0
    lateral_half_width_cells: int = Field(default=2, ge=0)
    max_uncertainty_margin_cells: int = Field(default=1, ge=0)
    require_uuv_per_region: bool = False

    @model_validator(mode="after")
    def validate_limits(self) -> GridSpec:
        if self.max_cell_size_m < self.min_cell_size_m:
            raise ValueError("max_cell_size_m must be >= min_cell_size_m")
        return self


class TimeWindow(StrictModel):
    start_s: int = Field(ge=0)
    end_s: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> TimeWindow:
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be after start_s")
        return self


class TaskRegionProposal(StrictModel):
    """LLM-selected rectangular task region in the shared global XY frame."""

    lower_left_xy: tuple[FiniteFloat, FiniteFloat]
    upper_right_xy: tuple[FiniteFloat, FiniteFloat]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> TaskRegionProposal:
        if (
            self.lower_left_xy[0] >= self.upper_right_xy[0]
            or self.lower_left_xy[1] >= self.upper_right_xy[1]
        ):
            raise ValueError("task region upper-right must be northeast of lower-left")
        return self


class TaskRegionProposalSet(StrictModel):
    """Four coordinate-only task regions selected for one target forecast."""

    regions: tuple[TaskRegionProposal, ...] = Field(min_length=4, max_length=4)


class TaskRegion(StrictModel):
    """Validated task region with its planner-owned 1 km grid cells."""

    region_id: str = Field(min_length=1)
    lower_left_xy: tuple[FiniteFloat, FiniteFloat]
    upper_right_xy: tuple[FiniteFloat, FiniteFloat]
    cell_ids: tuple[str, ...] = Field(min_length=1)
    active_window: TimeWindow
    required_uuv_count: int = Field(ge=1, le=4)
    rationale: str = Field(min_length=1)


class RegionalMissionCandidate(StrictModel):
    """Deterministic region candidate exposed to the UUV strategy LLM.

    Candidate geometry is generated by the planner.  The semantic model
    intentionally contains no platform assignment fields, so an LLM can
    select only from an immutable candidate set and cannot invent a route.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    cell_ids: tuple[str, ...] = Field(min_length=1)
    time_window: TimeWindow
    perimeter_points: tuple[tuple[FiniteFloat, FiniteFloat], ...] = Field(min_length=4)
    required_uuv_count: int = Field(default=0, ge=0, le=4)
    predecessor_candidate_ids: tuple[str, ...] = ()
    successor_candidate_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_candidate_geometry(self) -> RegionalMissionCandidate:
        if len(self.cell_ids) != len(set(self.cell_ids)):
            raise ValueError("candidate cell IDs must be unique")
        if len(self.perimeter_points) != len(set(self.perimeter_points)):
            raise ValueError("candidate perimeter points must be unique")
        if min(self.perimeter_points) != self.perimeter_points[0]:
            raise ValueError("candidate perimeter points must start at the minimum vertex")
        if not _is_simple_perimeter(self.perimeter_points):
            raise ValueError("candidate perimeter points must form a simple boundary")
        if len(self.predecessor_candidate_ids) != len(set(self.predecessor_candidate_ids)):
            raise ValueError("candidate predecessor IDs must be unique")
        if len(self.successor_candidate_ids) != len(set(self.successor_candidate_ids)):
            raise ValueError("candidate successor IDs must be unique")
        return self


class UUVRegionalPolicyDecision(StrictModel):
    """Topology-free policy selected by the regional strategy LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    coverage_mode: RegionCoverageMode
    tracking_mode: UUVRegionalTrackingMode
    priority: float = Field(ge=0, allow_inf_nan=False)
    required_quality: UnitFloat
    active_scan_uuv_count: int = Field(default=1, ge=0)
    passive_track_uuv_count: int = Field(default=1, ge=0)
    reserve_uuv_count: int = Field(default=0, ge=0)
    optional_uuv_count: int = Field(default=0, ge=0)
    assigned_uuv_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_uuv_members(self) -> UUVRegionalPolicyDecision:
        if len(self.assigned_uuv_ids) != len(set(self.assigned_uuv_ids)):
            raise ValueError("UUV policy assignments must be unique")
        return self


class UUVRegionalStrategyDecisionSet(StrictModel):
    """Strict live response contract for topology-free UUV decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policies: tuple[UUVRegionalPolicyDecision, ...] = ()


class UUVRegionalPolicy(StrictModel):
    """Deterministically resolved UUV policy, including immutable topology."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    coverage_mode: RegionCoverageMode
    tracking_mode: UUVRegionalTrackingMode
    priority: float = Field(ge=0, allow_inf_nan=False)
    required_quality: UnitFloat
    active_scan_uuv_count: int = Field(default=1, ge=0)
    passive_track_uuv_count: int = Field(default=1, ge=0)
    reserve_uuv_count: int = Field(default=0, ge=0)
    optional_uuv_count: int = Field(default=0, ge=0)
    assigned_uuv_ids: tuple[str, ...] = ()
    predecessor_candidate_id: str | None = None
    successor_candidate_id: str | None = None
    rationale: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_uuv_members(self) -> UUVRegionalPolicy:
        if len(self.assigned_uuv_ids) != len(set(self.assigned_uuv_ids)):
            raise ValueError("UUV policy assignments must be unique")
        return self


class UUVRegionalStrategySet(StrictModel):
    """Strict, candidate-only output contract for new UUV-only runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policies: tuple[UUVRegionalPolicy, ...] = ()
    request_hash: str = ""
    response_hash: str = ""


class RegionCell(StrictModel):
    region_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    grid_x: int
    grid_y: int
    min_x: FiniteFloat
    max_x: FiniteFloat
    min_y: FiniteFloat
    max_y: FiniteFloat
    center_xy: tuple[FiniteFloat, FiniteFloat]
    predicted_target_xy: tuple[FiniteFloat, FiniteFloat] | None = None
    cell_size_m: PositiveFinite
    first_entry_s: int = Field(ge=0)
    last_exit_s: int = Field(ge=0)
    visit_windows: tuple[TimeWindow, ...] = ()
    occupancy_likelihood: UnitFloat = 0.0
    intent_labels: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    predecessor_region_ids: tuple[str, ...] = ()
    successor_region_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_geometry(self) -> RegionCell:
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        if width <= 0 or height <= 0:
            raise ValueError("region bounds must have positive area")
        if not isclose(width, height, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("region cell must be square")
        expected = f"{self.target_id}:cell:{self.grid_x}:{self.grid_y}"
        if self.region_id != expected:
            raise ValueError("region_id must be deterministic from target and grid coordinates")
        expected_center = ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)
        if not all(
            isclose(actual, expected, abs_tol=1e-7)
            for actual, expected in zip(self.center_xy, expected_center)
        ):
            raise ValueError("center_xy must be the center of the region bounds")
        if self.last_exit_s < self.first_entry_s:
            raise ValueError("last_exit_s must not precede first_entry_s")
        return self


class SonarPolicy(StrictModel):
    passive_required: bool = True
    active_allowed: bool = False
    active_mode: Literal["none", "probe", "continuous"] = "none"
    active_cooldown_s: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_active_mode(self) -> SonarPolicy:
        if not self.passive_required:
            raise ValueError("passive sonar must remain required")
        if not self.active_allowed and self.active_mode != "none":
            raise ValueError("active mode requires active_allowed")
        return self


class CommunicationRequirement(StrictModel):
    carrier_to_uuv: bool = True
    acoustic_link_required: bool = True


class RegionTask(StrictModel):
    region_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    active_window: TimeWindow
    visit_window_index: int = Field(default=0, ge=0)
    priority: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    required_quality: UnitFloat = 0.0
    coverage_mode: RegionCoverageMode = "required"
    # LLM-provided explanatory metadata, not allocation targets. The explicit
    # member IDs below are authoritative whenever a regional policy is used.
    required_uuv_count: int = Field(default=0, ge=0)
    tracking_mode: RegionTrackingMode = "heuristic_uuv"
    uuv_roles: tuple[UUVRole, ...] = ()
    sonar_policy: SonarPolicy = SonarPolicy()
    communication: CommunicationRequirement = CommunicationRequirement()
    predecessor_region_id: str | None = None
    successor_region_id: str | None = None
    assigned_uuv_ids: tuple[str, ...] = ()
    assignment_status: RegionAssignmentStatus = "planned"
    communication_links: tuple[str, ...] = ()
    current_sonar_mode: Literal["passive", "active"] = "passive"
    evidence_ids: tuple[str, ...] = ()
    plan_revision: int = Field(default=1, ge=1)
    degraded_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_roles(self) -> RegionTask:
        if not self.sonar_policy.passive_required:
            raise ValueError("passive sonar must remain required")
        return self


class RegionalPolicy(StrictModel):
    region_id: str = Field(min_length=1)
    coverage_mode: RegionCoverageMode
    priority: float = Field(ge=0, allow_inf_nan=False)
    required_quality: UnitFloat
    required_uuv_count: int = Field(ge=0)
    uuv_roles: tuple[UUVRole, ...] = ()
    sonar_policy: SonarPolicy
    communication: CommunicationRequirement
    predecessor_region_id: str | None = None
    successor_region_id: str | None = None
    tracking_mode: RegionTrackingMode = "heuristic_uuv"
    assigned_uuv_ids: tuple[str, ...]
    rationale: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_policy_roles(self) -> RegionalPolicy:
        return self


class RegionalStrategySet(StrictModel):
    policies: tuple[RegionalPolicy, ...] = ()
    request_hash: str = ""
    response_hash: str = ""


class TargetRegionPlan(StrictModel):
    target_id: str = Field(min_length=1)
    grid_spec: GridSpec
    cell_size_m: PositiveFinite
    cells: tuple[RegionCell, ...] = ()
    tasks: tuple[RegionTask, ...] = ()
    task_regions: tuple[TaskRegion, ...] = ()
    prediction_id: str = Field(min_length=1)
    intent_label: str = Field(min_length=1)
    intent_confidence: UnitFloat
    evidence_ids: tuple[str, ...] = ()
    fallback_used: bool = False
    fallback_reason: str | None = None
    plan_revision: int = Field(default=1, ge=1)

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(cell.region_id for cell in self.cells)

    @model_validator(mode="after")
    def validate_task_coverage(self) -> TargetRegionPlan:
        cell_ids = self.region_ids
        task_ids = tuple(task.region_id for task in self.tasks)
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("region cell IDs must be unique")
        if set(task_ids) != set(cell_ids) or len(task_ids) != len(cell_ids):
            raise ValueError("exactly one task is required for every region cell")
        if any(task.target_id != self.target_id for task in self.tasks):
            raise ValueError("region task target IDs must match the plan target")
        if len({region.region_id for region in self.task_regions}) != len(self.task_regions):
            raise ValueError("task region IDs must be unique")
        if any(
            not set(region.cell_ids).issubset(cell_ids)
            for region in self.task_regions
        ):
            raise ValueError("task region references an unknown cell")
        for left_index, left in enumerate(self.cells):
            for right in self.cells[left_index + 1:]:
                if (
                    min(left.max_x, right.max_x) > max(left.min_x, right.min_x)
                    and min(left.max_y, right.max_y) > max(left.min_y, right.min_y)
                ):
                    raise ValueError("region cells must not overlap")
        return self
