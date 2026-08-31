from __future__ import annotations

from collections.abc import Sequence
from math import hypot
from typing import Annotated

from pydantic import Field

from underwater_tracking.domain.models import StrictModel
from underwater_tracking.planning.astar import AStarRoutePlanner, Bounds, RoutePlan
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
    current_time_s: int = Field(default=0, ge=0)
    speed_mps: Finite = Field(default=5.0, gt=0)
    minimum_ready_uuv_count: int = Field(default=0, ge=0)
    future_reserve_uuv_count: int = Field(default=0, ge=0)


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
        if task.carrier_id != slot.carrier_id or not slot.healthy:
            return None
        deploy_count = task.required_uuv_count if task.task_type == "deploy" else 0
        remaining_ready = slot.ready_uuv_count - deploy_count
        if remaining_ready < 0 or remaining_ready < slot.minimum_ready_uuv_count:
            return None
        future_reserve_loss = max(
            0,
            slot.future_reserve_uuv_count - remaining_ready,
        )
        if future_reserve_loss:
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
        arrival_s = slot.current_time_s + _distance_to_new_stop(route) / slot.speed_mps
        if arrival_s > task.exit_s + 1e-9:
            return None
        eta_slack_s = max(0.0, task.entry_s - arrival_s)
        # Keep the primary objective route distance, while making the
        # operational tie-breaks explicit and deterministic: early arrival
        # is penalized by the time spent waiting, larger batches consume more
        # future capacity, and a small inverse-ready term prefers preserving
        # slots with more remaining inventory.
        service_cost = 0.01 * task.required_uuv_count
        ready_cost = 0.001 / (remaining_ready + 1)
        return route.distance_m + eta_slack_s * slot.speed_mps + service_cost + ready_cost


def _distance_to_new_stop(route: RoutePlan) -> float:
    """Return the route distance through the newly inserted final stop."""
    stop_index = route.stop_indices[-1] if route.stop_indices else len(route.points) - 1
    return sum(
        hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(route.points[: stop_index + 1], route.points[1 : stop_index + 1])
    )


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
