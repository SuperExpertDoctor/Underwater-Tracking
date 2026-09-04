"""Legacy allocation and UUV-only task-group policy contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field, model_validator

from underwater_tracking.domain.execution_models import (
    ExecutionModel,
    ExecutionRegion,
    ReserveUUVState,
    TaskGroupAssignment,
)
from underwater_tracking.planning.dynamic_regions import (
    DynamicRegionChain,
    LegacyExecutionRegion,
)


class TaskGroupPolicy(ExecutionModel):
    """Fixed resource policy for one target's four regional slots."""

    group_count: int = Field(default=4, ge=1)
    group_size: int = Field(default=2, ge=2, le=2)
    reserve_count: int = Field(default=4, ge=0)
    active_role: str = "active_verifier"
    passive_role: str = "passive_tracker"

    @property
    def required_uuv_count(self) -> int:
        return self.group_count * self.group_size + self.reserve_count


class UUVTaskGroupPolicy(ExecutionModel):
    """Fixed live-runtime policy for four three-UUV groups and no reserves."""

    group_count: int = Field(default=4, ge=4, le=4)
    group_size: int = Field(default=3, ge=3, le=3)

    @property
    def required_uuv_count(self) -> int:
        return self.group_count * self.group_size


class TaskGroupAllocation(ExecutionModel):
    """One allocation result, including explicit degradation and bound regions."""

    target_id: str = Field(min_length=1)
    execution_revision: int = Field(ge=1)
    assignments: tuple[TaskGroupAssignment, ...] = ()
    reserve_uuvs: tuple[ReserveUUVState, ...] = ()
    bound_regions: tuple[ExecutionRegion | LegacyExecutionRegion, ...] = ()
    unallocated_uuv_ids: tuple[str, ...] = ()
    degraded: bool = False
    degradation_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_allocation(self) -> TaskGroupAllocation:
        assignment_regions = tuple(assignment.region_id for assignment in self.assignments)
        if len(assignment_regions) != len(set(assignment_regions)):
            raise ValueError("task group assignments must use unique regions")
        members = tuple(
            member
            for assignment in self.assignments
            for member in assignment.member_uuv_ids
        )
        if len(members) != len(set(members)):
            raise ValueError("task group allocation cannot reuse a UUV")
        reserve_ids = tuple(reserve.uuv_id for reserve in self.reserve_uuvs)
        if len(reserve_ids) != len(set(reserve_ids)):
            raise ValueError("task group reserve IDs must be unique")
        if set(members) & set(reserve_ids):
            raise ValueError("task group members and reserves must be disjoint")
        return self

    @property
    def assigned_uuv_ids(self) -> tuple[str, ...]:
        return tuple(
            member
            for assignment in self.assignments
            for member in assignment.member_uuv_ids
        )

    @property
    def reserve_uuv_ids(self) -> tuple[str, ...]:
        return tuple(reserve.uuv_id for reserve in self.reserve_uuvs)

    @property
    def groups(self) -> tuple[TaskGroupAssignment, ...]:
        """Compatibility alias for callers that call assignments groups."""
        return self.assignments


@dataclass(slots=True)
class ReplacementQueue:
    """Deterministic FIFO-by-priority reserve queue for boundary replacement."""

    reserves: tuple[ReserveUUVState, ...] = ()
    _available: list[ReserveUUVState] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._available = sorted(
            self.reserves,
            key=lambda reserve: (-reserve.priority, reserve.uuv_id),
        )

    @property
    def reserve_uuvs(self) -> tuple[ReserveUUVState, ...]:
        return tuple(self._available)

    def peek(self) -> ReserveUUVState | None:
        return self._available[0] if self._available else None

    def acquire(
        self,
        *,
        region_id: str,
        failed_uuv_id: str,
        execution_revision: int,
    ) -> ReserveUUVState | None:
        """Move the highest-priority reserve to an entering replacement state."""
        del failed_uuv_id
        if not region_id or execution_revision < 1:
            raise ValueError("replacement region and revision are required")
        if not self._available:
            return None
        reserve = self._available.pop(0)
        return reserve.model_copy(
            update={
                "status": "entering",
                "resource_episode": reserve.resource_episode + 1,
            }
        )

    def requeue(self, reserve: ReserveUUVState) -> None:
        """Return a reserve after a failed replacement attempt."""
        if any(item.uuv_id == reserve.uuv_id for item in self._available):
            return
        self._available.append(reserve.model_copy(update={"status": "reserve"}))
        self._available.sort(key=lambda item: (-item.priority, item.uuv_id))


class TaskGroupAllocator:
    """Allocate stable two-member groups while preserving healthy continuity."""

    def __init__(self, policy: TaskGroupPolicy | None = None) -> None:
        self.policy = policy or TaskGroupPolicy()

    def allocate(
        self,
        regions: DynamicRegionChain | Sequence[ExecutionRegion],
        uuv_resources: Mapping[str, Any] | Sequence[Any],
        *,
        execution_revision: int | None = None,
        previous_assignments: Sequence[TaskGroupAssignment] = (),
    ) -> TaskGroupAllocation:
        region_values = _regions(regions)
        if len(region_values) != self.policy.group_count:
            raise ValueError("four task groups require four execution regions")
        target_id = region_values[0].target_id
        revision = execution_revision or region_values[0].execution_revision
        resource_rows = _resources(uuv_resources)
        if len(resource_rows) != len({row[0] for row in resource_rows}):
            raise ValueError("duplicate UUV resource IDs")
        eligible = [row for row in resource_rows if row[1]]
        eligible_ids = tuple(sorted(row[0] for row in eligible))
        previous_by_region = {assignment.region_id: assignment for assignment in previous_assignments}
        used: set[str] = set()
        assignments: list[TaskGroupAssignment] = []
        reasons: list[str] = []
        for index, region in enumerate(region_values, start=1):
            previous = previous_by_region.get(region.region_id)
            retained = [] if previous is None else [
                member
                for member in previous.member_uuv_ids
                if member in eligible_ids and member not in used
            ]
            members = list(retained)
            members.extend(
                uuv_id
                for uuv_id in eligible_ids
                if uuv_id not in used and uuv_id not in members
            )
            members = members[: self.policy.group_size]
            if len(members) < self.policy.group_size:
                reasons.append(f"{region.region_id}:insufficient_two_uuv_members")
                continue
            used.update(members)
            group_id = f"{target_id}:task-group:{index:02d}"
            evidence_ids = tuple(
                sorted({*region.evidence_ids, f"allocation:{revision}:{group_id}"})
            )
            assignments.append(
                TaskGroupAssignment(
                    task_group_id=group_id,
                    target_id=target_id,
                    region_id=region.region_id,
                    execution_revision=revision,
                    member_uuv_ids=(members[0], members[1]),
                    active_verifier_uuv_id=members[0],
                    passive_tracker_uuv_id=members[1],
                    status="prepositioning",
                    evidence_ids=evidence_ids,
                )
            )
        if len(eligible_ids) < self.policy.required_uuv_count:
            reasons.append(
                f"resources:{len(eligible_ids)}_available_below_{self.policy.required_uuv_count}_required"
            )
        reserve_ids = ()
        if len(eligible_ids) >= self.policy.required_uuv_count:
            reserve_ids = tuple(uuv_id for uuv_id in eligible_ids if uuv_id not in used)
        reserves = tuple(
            ReserveUUVState(
                uuv_id=uuv_id,
                priority=float(self.policy.reserve_count - index),
            )
            for index, uuv_id in enumerate(reserve_ids)
        )
        assignment_by_region = {assignment.region_id: assignment for assignment in assignments}
        bound_regions = tuple(
            region.model_copy(
                update={
                    "execution_revision": revision,
                    "task_group_id": assignment_by_region[region.region_id].task_group_id
                    if region.region_id in assignment_by_region
                    else None,
                }
            )
            for region in region_values
        )
        unallocated = tuple(uuv_id for uuv_id in eligible_ids if uuv_id not in used and uuv_id not in reserve_ids)
        return TaskGroupAllocation(
            target_id=target_id,
            execution_revision=revision,
            assignments=tuple(assignments),
            reserve_uuvs=reserves,
            bound_regions=bound_regions,
            unallocated_uuv_ids=unallocated,
            degraded=bool(reasons),
            degradation_reasons=tuple(dict.fromkeys(reasons)),
        )


def allocate_four_task_groups(
    regions: DynamicRegionChain | Sequence[ExecutionRegion],
    uuv_ids: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    uuv_resources: Mapping[str, Any] | Sequence[Any] | None = None,
    execution_revision: int | None = None,
    previous_assignments: Sequence[TaskGroupAssignment] = (),
    policy: TaskGroupPolicy | None = None,
) -> TaskGroupAllocation:
    """Allocate four two-UUV groups and, with 12 healthy UUVs, four reserves."""
    resources = uuv_resources if uuv_resources is not None else uuv_ids
    if resources is None:
        raise ValueError("UUV resources are required")
    return TaskGroupAllocator(policy).allocate(
        regions,
        resources,
        execution_revision=execution_revision,
        previous_assignments=previous_assignments,
    )


def _regions(
    regions: DynamicRegionChain | Sequence[ExecutionRegion],
) -> tuple[ExecutionRegion, ...]:
    if isinstance(regions, DynamicRegionChain):
        return regions.regions
    return tuple(sorted(regions, key=lambda region: region.slot_index))


def _resources(
    resources: Mapping[str, Any] | Sequence[Any],
) -> tuple[tuple[str, bool], ...]:
    values = resources.values() if isinstance(resources, Mapping) else resources
    rows: list[tuple[str, bool]] = []
    for value in values:
        if isinstance(value, str):
            rows.append((value, True))
            continue
        uuv_id = str(getattr(value, "uuv_id", getattr(value, "platform_id", "")))
        if not uuv_id:
            raise ValueError("every UUV resource must have an ID")
        healthy = bool(getattr(value, "healthy", True))
        energy = float(getattr(value, "energy_fraction", 1.0))
        deployment = str(getattr(value, "deployment_state", "deployed")).casefold()
        healthy = healthy and energy > 0.0 and deployment not in {"failed", "unavailable"}
        rows.append((uuv_id, healthy))
    return tuple(rows)


__all__ = [
    "ReplacementQueue",
    "TaskGroupAllocation",
    "TaskGroupAllocator",
    "TaskGroupPolicy",
    "UUVTaskGroupPolicy",
    "allocate_four_task_groups",
]
