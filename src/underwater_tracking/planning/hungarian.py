from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

from underwater_tracking.domain.models import StrictModel
from underwater_tracking.planning.astar import AStarRoutePlanner, Bounds
from underwater_tracking.planning.carrier_tasks import CarrierServiceTask

Finite = Annotated[float, Field(allow_inf_nan=False)]


class VirtualServiceSlot(StrictModel):
    """A carrier route insertion slot used by the deterministic matcher."""

    slot_id: str = Field(min_length=1)
    carrier_id: str = Field(min_length=1)
    current_xy: tuple[Finite, Finite]
    home_xy: tuple[Finite, Finite]
    ready_uuv_count: int = Field(ge=0)
    healthy: bool = True
    committed_stop_points: tuple[tuple[Finite, Finite], ...] = ()


class CarrierSlotAssignment(StrictModel):
    task_id: str = Field(min_length=1)
    slot_id: str = Field(min_length=1)
    cost: float = Field(ge=0, allow_inf_nan=False)


class HungarianMatchResult(StrictModel):
    assignments: tuple[CarrierSlotAssignment, ...] = ()
    unassigned_task_ids: tuple[str, ...] = ()


class HungarianMatcher:
    """Match service tasks to feasible slots with deterministic tie-breaking."""

    def __init__(
        self,
        *,
        route_planner: AStarRoutePlanner | None = None,
        forbidden_regions: Sequence[Bounds] = (),
        map_bounds: Bounds = (-10_000.0, 10_000.0, -10_000.0, 10_000.0),
    ) -> None:
        self._route_planner = route_planner or AStarRoutePlanner(grid_size_m=1.0)
        self._forbidden_regions = tuple(forbidden_regions)
        self._map_bounds = map_bounds

    def match(
        self,
        tasks: Sequence[CarrierServiceTask],
        virtual_service_slots: Sequence[VirtualServiceSlot],
    ) -> HungarianMatchResult:
        ordered_tasks = tuple(sorted(tasks, key=lambda task: task.task_id))
        ordered_slots = tuple(sorted(virtual_service_slots, key=lambda slot: slot.slot_id))
        costs: dict[tuple[int, int], float] = {}
        for task_index, task in enumerate(ordered_tasks):
            for slot_index, slot in enumerate(ordered_slots):
                cost = self._cost(task, slot)
                if cost is not None:
                    costs[(task_index, slot_index)] = cost

        _, _, assignments = self._search(ordered_tasks, ordered_slots, costs, 0, 0)
        assignments = tuple(
            sorted(assignments, key=lambda assignment: assignment.task_id)
        )
        assigned_task_ids = {assignment.task_id for assignment in assignments}
        return HungarianMatchResult(
            assignments=assignments,
            unassigned_task_ids=tuple(
                task.task_id for task in ordered_tasks if task.task_id not in assigned_task_ids
            ),
        )

    def _search(
        self,
        tasks: tuple[CarrierServiceTask, ...],
        slots: tuple[VirtualServiceSlot, ...],
        costs: dict[tuple[int, int], float],
        task_index: int,
        used_slots: int,
    ) -> tuple[int, float, tuple[CarrierSlotAssignment, ...]]:
        if task_index >= len(tasks):
            return 0, 0.0, ()
        best = self._search(tasks, slots, costs, task_index + 1, used_slots)
        task = tasks[task_index]
        for slot_index, slot in enumerate(slots):
            if used_slots & (1 << slot_index):
                continue
            cost = costs.get((task_index, slot_index))
            if cost is None:
                continue
            matched, total_cost, suffix = self._search(
                tasks,
                slots,
                costs,
                task_index + 1,
                used_slots | (1 << slot_index),
            )
            candidate = (
                matched + 1,
                total_cost + cost,
                (
                    CarrierSlotAssignment(
                        task_id=task.task_id,
                        slot_id=slot.slot_id,
                        cost=cost,
                    ),
                    *suffix,
                ),
            )
            if _better(candidate, best):
                best = candidate
        return best

    def _cost(
        self,
        task: CarrierServiceTask,
        slot: VirtualServiceSlot,
    ) -> float | None:
        if not slot.healthy or slot.ready_uuv_count < task.required_uuv_count:
            return None
        route = self._route_planner.plan(
            slot.current_xy,
            (*slot.committed_stop_points, task.point),
            slot.home_xy,
            self._forbidden_regions,
            self._map_bounds,
        )
        if route is None:
            return None
        return route.distance_m


def _better(
    candidate: tuple[int, float, tuple[CarrierSlotAssignment, ...]],
    current: tuple[int, float, tuple[CarrierSlotAssignment, ...]],
) -> bool:
    candidate_matched, candidate_cost, candidate_assignments = candidate
    current_matched, current_cost, current_assignments = current
    if candidate_matched != current_matched:
        return candidate_matched > current_matched
    if abs(candidate_cost - current_cost) > 1e-9:
        return candidate_cost < current_cost
    return tuple(
        (assignment.task_id, assignment.slot_id)
        for assignment in candidate_assignments
    ) < tuple(
        (assignment.task_id, assignment.slot_id)
        for assignment in current_assignments
    )
