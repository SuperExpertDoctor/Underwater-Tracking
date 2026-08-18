from __future__ import annotations

from math import isclose
from typing import Annotated, Literal

from pydantic import Field, model_validator

from underwater_tracking.domain.models import StrictModel

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

UUVRole = Literal["passive_tracker", "active_verifier", "handoff_reserve"]
USVRole = Literal["surface_relay", "active_tracker", "relay_and_tracker", "handoff_reserve"]
RegionRole = UUVRole | USVRole
RegionAssignmentStatus = Literal["planned", "active", "handed_off", "degraded", "uncovered"]
RegionCoverageMode = Literal["required", "reserve", "optional"]
RegionTrackingMode = Literal[
    "uuv_primary_usv_relay",
    "heuristic_uuv",
    "heuristic_usv",
]


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
    require_usv_per_region: bool = False
    relay_overlap_policy: Literal["forbid", "adjacent_connected"] = "adjacent_connected"

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
    usv_relay_required: bool = True
    acoustic_link_required: bool = True
    relay_overlap_policy: Literal["forbid", "adjacent_connected"] = "adjacent_connected"


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
    required_usv_count: int = Field(default=0, ge=0)
    tracking_mode: RegionTrackingMode = "uuv_primary_usv_relay"
    uuv_roles: tuple[UUVRole, ...] = ()
    usv_role: USVRole | None = None
    sonar_policy: SonarPolicy = SonarPolicy()
    communication: CommunicationRequirement = CommunicationRequirement()
    predecessor_region_id: str | None = None
    successor_region_id: str | None = None
    assigned_uuv_ids: tuple[str, ...] = ()
    assigned_usv_ids: tuple[str, ...] = ()
    assignment_status: RegionAssignmentStatus = "planned"
    communication_links: tuple[str, ...] = ()
    current_sonar_mode: Literal["passive", "active"] = "passive"
    evidence_ids: tuple[str, ...] = ()
    plan_revision: int = Field(default=1, ge=1)
    degraded_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_roles(self) -> RegionTask:
        if (
            self.tracking_mode == "uuv_primary_usv_relay"
            and self.assigned_usv_ids
            and self.usv_role not in {"surface_relay", "handoff_reserve"}
        ):
            raise ValueError("uuv_primary_usv_relay tasks require relay-only USV roles")
        if not self.sonar_policy.passive_required:
            raise ValueError("passive sonar must remain required")
        return self


class RegionalPolicy(StrictModel):
    region_id: str = Field(min_length=1)
    coverage_mode: RegionCoverageMode
    priority: float = Field(ge=0, allow_inf_nan=False)
    required_quality: UnitFloat
    required_uuv_count: int = Field(ge=0)
    required_usv_count: int = Field(ge=0)
    uuv_roles: tuple[UUVRole, ...] = ()
    usv_role: USVRole | None = None
    sonar_policy: SonarPolicy
    communication: CommunicationRequirement
    predecessor_region_id: str | None = None
    successor_region_id: str | None = None
    tracking_mode: RegionTrackingMode = "uuv_primary_usv_relay"
    assigned_uuv_ids: tuple[str, ...] = ()
    assigned_usv_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_policy_roles(self) -> RegionalPolicy:
        if self.tracking_mode == "heuristic_uuv" and self.assigned_usv_ids:
            raise ValueError("heuristic_uuv policy cannot mix USV members")
        if self.tracking_mode == "heuristic_usv" and self.assigned_uuv_ids:
            raise ValueError("heuristic_usv policy cannot mix UUV members")
        if (
            self.tracking_mode == "uuv_primary_usv_relay"
            and self.assigned_usv_ids
            and self.usv_role not in {"surface_relay", "handoff_reserve"}
        ):
            raise ValueError("uuv_primary_usv_relay policy requires relay-only USV roles")
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
        for left_index, left in enumerate(self.cells):
            for right in self.cells[left_index + 1:]:
                if (
                    min(left.max_x, right.max_x) > max(left.min_x, right.min_x)
                    and min(left.max_y, right.max_y) > max(left.min_y, right.min_y)
                ):
                    raise ValueError("region cells must not overlap")
        return self
