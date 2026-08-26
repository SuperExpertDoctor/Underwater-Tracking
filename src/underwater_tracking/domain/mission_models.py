from __future__ import annotations

from enum import Enum
from math import isclose
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from underwater_tracking.domain.execution_models import ReserveUUVState, TaskGroupAssignment
from underwater_tracking.domain.models import StrictModel

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class UUVMissionMode(str, Enum):
    ONBOARD = "ONBOARD"
    TRANSIT_TO_REGION = "TRANSIT_TO_REGION"
    ACTIVE_SCAN = "ACTIVE_SCAN"
    PASSIVE_TRACK = "PASSIVE_TRACK"
    DEDICATED_TRACK = "DEDICATED_TRACK"
    RETURN_TO_REGION = "RETURN_TO_REGION"
    RETURN_REQUIRED = "RETURN_REQUIRED"
    RECOVERING = "RECOVERING"
    FAILED = "FAILED"


class UUVResourceState(StrictModel):
    """Live resource facts consumed by planning and mission execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uuv_id: str = Field(min_length=1)
    carrier_id: str | None = None
    mileage_m: FiniteFloat = Field(ge=0)
    energy_fraction: UnitFloat
    healthy: bool = True
    capability_active: bool = True
    deployment_state: str = Field(min_length=1)
    resource_episode: int = Field(default=0, ge=0)


class RegionLifecycle(str, Enum):
    PLANNED = "PLANNED"
    CARRIER_DEPLOYING = "CARRIER_DEPLOYING"
    ACTIVE_SCAN = "ACTIVE_SCAN"
    PASSIVE_TRACK = "PASSIVE_TRACK"
    HANDOFF_PENDING = "HANDOFF_PENDING"
    TRACKING_COMPLETED = "TRACKING_COMPLETED"
    CARRIER_RECOVERY = "CARRIER_RECOVERY"
    RECOVERED = "RECOVERED"
    DEGRADED = "DEGRADED"
    UNCOVERED = "UNCOVERED"


class CarrierRouteStatus(str, Enum):
    TO_DEPLOY = "TO_DEPLOY"
    DEPLOYING = "DEPLOYING"
    EN_ROUTE_NEXT_DEPLOY = "EN_ROUTE_NEXT_DEPLOY"
    RETURNING_TO_FLEET = "RETURNING_TO_FLEET"
    RECOVERING = "RECOVERING"
    RENDEZVOUS_BLOCKED = "RENDEZVOUS_BLOCKED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class CarrierExecutionMode(str, Enum):
    """Private physical execution mode for one carrier entity."""

    FORMATION_FOLLOW = "FORMATION_FOLLOW"
    MISSION_ROUTE = "MISSION_ROUTE"
    RENDEZVOUS_RETURN = "RENDEZVOUS_RETURN"


class MissionCandidate(StrictModel):
    """Planner-owned region candidate used by the deterministic optimizer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    entry_s: int = Field(ge=0)
    exit_s: int = Field(gt=0)
    probability: UnitFloat
    perimeter_points: tuple[tuple[FiniteFloat, FiniteFloat], ...] = Field(min_length=4)
    active_scan_uuv_count: int = Field(default=1, ge=0)
    passive_track_uuv_count: int = Field(default=1, ge=0)
    reserve_uuv_count: int = Field(default=0, ge=0)
    optional_uuv_count: int = Field(default=0, ge=0)
    priority: UnitFloat = 0.0
    predecessor_candidate_ids: tuple[str, ...] = ()
    successor_candidate_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_candidate(self) -> MissionCandidate:
        if self.exit_s <= self.entry_s:
            raise ValueError("candidate exit_s must be after entry_s")
        if len(self.perimeter_points) != len(set(self.perimeter_points)):
            raise ValueError("candidate perimeter points must be unique")
        if len(self.predecessor_candidate_ids) != len(set(self.predecessor_candidate_ids)):
            raise ValueError("candidate predecessor IDs must be unique")
        if len(self.successor_candidate_ids) != len(set(self.successor_candidate_ids)):
            raise ValueError("candidate successor IDs must be unique")
        return self


class PredictionGridCell(StrictModel):
    target_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    grid_x: int
    grid_y: int
    min_x: FiniteFloat
    max_x: FiniteFloat
    min_y: FiniteFloat
    max_y: FiniteFloat
    cell_size_m: PositiveFloat = 1.0
    probability: UnitFloat
    first_entry_s: int = Field(ge=0)
    last_exit_s: int = Field(ge=0)
    imm_model_probabilities: dict[str, UnitFloat] = Field(default_factory=dict)
    covariance_summary: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    intent_label: str = Field(min_length=1)
    intent_confidence: UnitFloat
    region_id: str = ""

    @model_validator(mode="before")
    @classmethod
    def default_region_id(cls, value: object) -> object:
        if isinstance(value, dict) and not value.get("region_id"):
            value = {
                **value,
                "region_id": (
                    f"{value['target_id']}:r{value['revision']}:cell:"
                    f"{value['grid_x']}:{value['grid_y']}"
                ),
            }
        return value

    @model_validator(mode="after")
    def validate_geometry(self) -> PredictionGridCell:
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        if width <= 0 or height <= 0:
            raise ValueError("prediction grid cell bounds must have positive area")
        if not isclose(width, height, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("prediction grid cell must be square")
        if not isclose(width, self.cell_size_m, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("prediction grid cell size must match its bounds")
        if self.last_exit_s < self.first_entry_s:
            raise ValueError("prediction grid time window is reversed")
        expected = f"{self.target_id}:r{self.revision}:cell:{self.grid_x}:{self.grid_y}"
        if self.region_id != expected:
            raise ValueError("prediction grid region ID is not deterministic")
        return self


class PredictionGrid(StrictModel):
    target_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    origin: tuple[FiniteFloat, FiniteFloat]
    cell_size_m: PositiveFloat
    cells: tuple[PredictionGridCell, ...] = ()
    centerline_region_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_cells(self) -> PredictionGrid:
        ids = [cell.region_id for cell in self.cells]
        if len(ids) != len(set(ids)):
            raise ValueError("prediction grid region IDs must be unique")
        for cell in self.cells:
            if cell.target_id != self.target_id or cell.revision != self.revision:
                raise ValueError("prediction grid cell identity does not match its grid")
            if not isclose(cell.cell_size_m, self.cell_size_m, rel_tol=0.0, abs_tol=1e-7):
                raise ValueError("prediction grid cells must share the grid cell size")
        if not set(self.centerline_region_ids).issubset(ids):
            raise ValueError("prediction grid centerline contains an unknown cell")
        return self

    def cell(self, grid_x: int, grid_y: int) -> PredictionGridCell:
        for cell in self.cells:
            if cell.grid_x == grid_x and cell.grid_y == grid_y:
                return cell
        raise KeyError((grid_x, grid_y))


class RegionMissionState(StrictModel):
    region_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    lifecycle: RegionLifecycle = RegionLifecycle.PLANNED
    active_scan_uuv_ids: tuple[str, ...] = ()
    passive_track_uuv_ids: tuple[str, ...] = ()
    reserve_uuv_ids: tuple[str, ...] = ()
    coverage: UnitFloat = 0.0
    tracking_quality: UnitFloat = 0.0
    entry_confirmations: int = Field(default=0, ge=0)
    handoff_from: str | None = None
    handoff_to: str | None = None
    carrier_task_id: str | None = None
    plan_revision: int = Field(default=1, ge=1)
    degraded_reasons: tuple[str, ...] = ()
    region_polygon: tuple[tuple[FiniteFloat, FiniteFloat], ...] = ()
    scan_waypoints: tuple[tuple[FiniteFloat, FiniteFloat], ...] = ()
    scan_waypoints_by_uuv: dict[str, tuple[tuple[FiniteFloat, FiniteFloat], ...]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def uuv_assignments_are_disjoint(self) -> RegionMissionState:
        groups = (
            self.active_scan_uuv_ids,
            self.passive_track_uuv_ids,
            self.reserve_uuv_ids,
        )
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("region UUV assignments must be unique")
        if any(
            set(left) & set(right)
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        ):
            raise ValueError("region UUV assignments overlap")
        return self


class AcceptedHandoffObservation(StrictModel):
    """One current-cycle passive bearing accepted by a successor group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str = Field(min_length=1)
    observer_uuv_id: str = Field(min_length=1)
    observed_at_s: int = Field(ge=0)


class HandoffEvidence(StrictModel):
    """Typed, current-cycle evidence required to complete a region handoff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    predecessor_region_id: str = Field(min_length=1)
    successor_region_id: str = Field(min_length=1)
    plan_revision: int = Field(ge=1)
    observation_cycle_s: int = Field(ge=0)
    required_uuv_ids: tuple[str, ...]
    deployed_uuv_ids: tuple[str, ...]
    healthy_uuv_ids: tuple[str, ...]
    passive_mode_uuv_ids: tuple[str, ...]
    accepted_observations: tuple[AcceptedHandoffObservation, ...]
    hard_guard_reasons: tuple[str, ...] = ()
    blocked_reason: str | None = None

    @model_validator(mode="after")
    def validate_observation_membership(self) -> HandoffEvidence:
        for field_name in (
            "required_uuv_ids",
            "deployed_uuv_ids",
            "healthy_uuv_ids",
            "passive_mode_uuv_ids",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} IDs must be unique")
        observation_ids = tuple(
            observation.observation_id for observation in self.accepted_observations
        )
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("accepted observation IDs must be unique")
        required = set(self.required_uuv_ids)
        passive = set(self.passive_mode_uuv_ids)
        for observation in self.accepted_observations:
            if observation.observer_uuv_id not in required or observation.observer_uuv_id not in passive:
                raise ValueError(
                    "accepted observation observer must be a required passive UUV"
                )
            if observation.observed_at_s != self.observation_cycle_s:
                raise ValueError("accepted observation cycle must match observation cycle")
        return self

    def is_complete(self, *, group_min_size: int) -> bool:
        """Return whether this evidence satisfies the runtime handoff guards."""
        if group_min_size < 1:
            raise ValueError("group_min_size must be positive")
        required = set(self.required_uuv_ids)
        observers = {
            observation.observer_uuv_id for observation in self.accepted_observations
        }
        return (
            self.blocked_reason is None
            and not self.hard_guard_reasons
            and bool(required)
            and required.issubset(self.deployed_uuv_ids)
            and required.issubset(self.healthy_uuv_ids)
            and required.issubset(self.passive_mode_uuv_ids)
            and len(observers) >= group_min_size
        )


class CarrierMissionModel(StrictModel):
    carrier_id: str = Field(min_length=1)
    role: Literal["carrier", "mother_ship"] = "carrier"
    home_battle_group_id: str = Field(min_length=1)
    mission_type: Literal["DEPLOY", "RECOVER", "DEPLOY_AND_RECOVER"] = "DEPLOY_AND_RECOVER"
    route_status: CarrierRouteStatus = CarrierRouteStatus.TO_DEPLOY
    route_xy: tuple[tuple[FiniteFloat, FiniteFloat], ...] = ()
    stop_ids: tuple[str, ...] = ()
    stop_indices: tuple[int, ...] = ()
    stop_windows: tuple[tuple[int, int], ...] = ()
    onboard_uuv_ids: tuple[str, ...] = ()
    ready_uuv_ids: tuple[str, ...] = ()
    reserved_uuv_ids: tuple[str, ...] = ()
    recoverable_uuv_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def inventories_are_disjoint(self) -> CarrierMissionModel:
        if self.stop_windows:
            if len(self.stop_windows) != len(self.stop_ids):
                raise ValueError("carrier route stop windows must match stop IDs")
            if any(
                entry_s < 0 or exit_s <= entry_s
                for entry_s, exit_s in self.stop_windows
            ):
                raise ValueError("carrier route stop windows must be ordered")
        if self.stop_indices:
            if len(self.stop_indices) != len(self.stop_ids):
                raise ValueError("carrier route stop indices must match stop IDs")
            if not self.route_xy:
                raise ValueError("carrier route stop indices require route points")
            if len(self.stop_indices) != len(set(self.stop_indices)):
                raise ValueError("carrier route stop indices must be unique")
            if any(
                index <= 0 or index >= len(self.route_xy) - 1
                for index in self.stop_indices
            ):
                raise ValueError("carrier route stop index must identify an interior route point")
        groups = (
            self.onboard_uuv_ids,
            self.ready_uuv_ids,
            self.reserved_uuv_ids,
            self.recoverable_uuv_ids,
        )
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("carrier inventory IDs must be unique")
        if any(
            set(left) & set(right)
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        ):
            raise ValueError("carrier inventory groups overlap")
        return self

    @property
    def total_uuv_capacity(self) -> int:
        return sum(
            len(group)
            for group in (
                self.onboard_uuv_ids,
                self.ready_uuv_ids,
                self.reserved_uuv_ids,
                self.recoverable_uuv_ids,
            )
        )

    @property
    def ready_uuv_count(self) -> int:
        return len(self.ready_uuv_ids)

    @property
    def reserved_uuv_count(self) -> int:
        return len(self.reserved_uuv_ids)

    @property
    def recoverable_uuv_count(self) -> int:
        return len(self.recoverable_uuv_ids)


class UUVMissionBatch(StrictModel):
    """One deterministic carrier-to-region UUV deployment batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    carrier_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    uuv_ids: tuple[str, ...] = Field(min_length=1)
    active_scan_uuv_ids: tuple[str, ...] = ()
    passive_track_uuv_ids: tuple[str, ...] = ()
    reserve_uuv_ids: tuple[str, ...] = ()
    deployment_point: tuple[FiniteFloat, FiniteFloat] | None = None
    recovery_point: tuple[FiniteFloat, FiniteFloat] | None = None
    entry_s: int = Field(ge=0)
    exit_s: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_assignments(self) -> UUVMissionBatch:
        groups = (
            self.active_scan_uuv_ids,
            self.passive_track_uuv_ids,
            self.reserve_uuv_ids,
        )
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("mission batch UUV assignments must be unique")
        if any(
            set(left) & set(right)
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        ):
            raise ValueError("mission batch UUV assignments overlap")
        assigned = set().union(*groups)
        if assigned != set(self.uuv_ids):
            raise ValueError("mission batch UUV IDs must match its role assignments")
        if self.exit_s <= self.entry_s:
            raise ValueError("mission batch exit_s must be after entry_s")
        return self


class ExecutableMissionPlan(StrictModel):
    """Verified UUV batches, reserves, region assignments, and carrier work."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int = Field(ge=1)
    uuv_batches_by_carrier: dict[str, tuple[UUVMissionBatch, ...]] = Field(
        default_factory=dict
    )
    reserved_uuv_ids: tuple[str, ...] = ()
    region_assignments: tuple[RegionMissionState, ...] = ()
    carrier_missions: dict[str, CarrierMissionModel] = Field(default_factory=dict)
    degraded_reasons: tuple[str, ...] = ()
    resource_episode_by_uuv: dict[str, int] = Field(default_factory=dict)
    task_groups: tuple[TaskGroupAssignment, ...] = ()
    reserve_uuvs: tuple[ReserveUUVState, ...] = ()

    @model_validator(mode="after")
    def validate_plan_membership(self) -> ExecutableMissionPlan:
        if any(episode < 0 for episode in self.resource_episode_by_uuv.values()):
            raise ValueError("resource episodes must be non-negative")
        if len(self.reserved_uuv_ids) != len(set(self.reserved_uuv_ids)):
            raise ValueError("executable plan reserve UUV IDs must be unique")
        batch_ids: set[str] = set()
        for carrier_id, batches in self.uuv_batches_by_carrier.items():
            for batch in batches:
                if batch.carrier_id != carrier_id:
                    raise ValueError("mission batch carrier ID disagrees with its index")
                overlap = batch_ids.intersection(batch.uuv_ids)
                if overlap:
                    raise ValueError(f"UUV appears in multiple mission batches: {sorted(overlap)}")
                batch_ids.update(batch.uuv_ids)
        overlap = batch_ids.intersection(self.reserved_uuv_ids)
        if overlap:
            raise ValueError(f"UUV is both deployed and reserved: {sorted(overlap)}")
        task_group_members = tuple(
            member
            for group in self.task_groups
            for member in group.member_uuv_ids
        )
        if len(task_group_members) != len(set(task_group_members)):
            raise ValueError("UUV appears in multiple execution task groups")
        reserve_state_ids = tuple(reserve.uuv_id for reserve in self.reserve_uuvs)
        if len(reserve_state_ids) != len(set(reserve_state_ids)):
            raise ValueError("execution reserve UUV IDs must be unique")
        if set(task_group_members) & set(reserve_state_ids):
            raise ValueError("UUV is both a task group member and execution reserve")
        if len(self.region_assignments) != len(
            {assignment.region_id for assignment in self.region_assignments}
        ):
            raise ValueError("executable plan region IDs must be unique")
        assignment_ids: set[str] = set()
        for assignment in self.region_assignments:
            assigned = {
                *assignment.active_scan_uuv_ids,
                *assignment.passive_track_uuv_ids,
                *assignment.reserve_uuv_ids,
            }
            overlap = assignment_ids.intersection(assigned)
            if overlap:
                raise ValueError(
                    f"UUV appears in multiple region assignments: {sorted(overlap)}"
                )
            assignment_ids.update(assigned)
        return self

    @property
    def batches(self) -> tuple[UUVMissionBatch, ...]:
        return tuple(
            batch
            for carrier_id in sorted(self.uuv_batches_by_carrier)
            for batch in self.uuv_batches_by_carrier[carrier_id]
        )

    @property
    def all_uuv_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *self.reserved_uuv_ids,
                    *(uuv_id for batch in self.batches for uuv_id in batch.uuv_ids),
                    *(
                        uuv_id
                        for assignment in self.region_assignments
                        for uuv_id in (
                            *assignment.active_scan_uuv_ids,
                            *assignment.passive_track_uuv_ids,
                            *assignment.reserve_uuv_ids,
                        )
                    ),
                    *(
                        uuv_id
                        for carrier in self.carrier_missions.values()
                        for uuv_id in (
                            *carrier.onboard_uuv_ids,
                            *carrier.ready_uuv_ids,
                            *carrier.reserved_uuv_ids,
                            *carrier.recoverable_uuv_ids,
                        )
                    ),
                }
            )
        )

    @property
    def assignments_by_candidate(self) -> dict[str, RegionMissionState]:
        return {assignment.region_id: assignment for assignment in self.region_assignments}


_REGION_TRANSITIONS: dict[RegionLifecycle, frozenset[RegionLifecycle]] = {
    RegionLifecycle.PLANNED: frozenset({RegionLifecycle.CARRIER_DEPLOYING, RegionLifecycle.DEGRADED}),
    RegionLifecycle.CARRIER_DEPLOYING: frozenset({RegionLifecycle.ACTIVE_SCAN, RegionLifecycle.DEGRADED}),
    RegionLifecycle.ACTIVE_SCAN: frozenset({RegionLifecycle.PASSIVE_TRACK, RegionLifecycle.DEGRADED, RegionLifecycle.UNCOVERED}),
    RegionLifecycle.PASSIVE_TRACK: frozenset({RegionLifecycle.HANDOFF_PENDING, RegionLifecycle.TRACKING_COMPLETED, RegionLifecycle.DEGRADED}),
    RegionLifecycle.HANDOFF_PENDING: frozenset({RegionLifecycle.TRACKING_COMPLETED, RegionLifecycle.DEGRADED}),
    RegionLifecycle.TRACKING_COMPLETED: frozenset({RegionLifecycle.CARRIER_RECOVERY}),
    RegionLifecycle.CARRIER_RECOVERY: frozenset({RegionLifecycle.RECOVERED, RegionLifecycle.DEGRADED}),
    RegionLifecycle.RECOVERED: frozenset(),
    RegionLifecycle.DEGRADED: frozenset({RegionLifecycle.CARRIER_DEPLOYING, RegionLifecycle.RECOVERED}),
    RegionLifecycle.UNCOVERED: frozenset({RegionLifecycle.CARRIER_DEPLOYING, RegionLifecycle.DEGRADED}),
}


def validate_region_transition(current: RegionLifecycle, next_state: RegionLifecycle) -> bool:
    return next_state in _REGION_TRANSITIONS[current]
