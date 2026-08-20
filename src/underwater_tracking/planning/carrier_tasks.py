from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import hypot
from typing import Annotated, Literal

from pydantic import Field, model_validator

from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    CarrierRouteStatus,
    ExecutableMissionPlan,
)
from underwater_tracking.domain.models import StrictModel
from underwater_tracking.planning.astar import AStarRoutePlanner, Bounds, RoutePlan

Finite = Annotated[float, Field(allow_inf_nan=False)]
Point = tuple[Finite, Finite]
CarrierTaskType = Literal["deploy", "recover"]


class CarrierServiceTask(StrictModel):
    """One carrier stop for deploying or recovering a UUV batch."""

    carrier_id: str = Field(default="carrier_01", min_length=1)
    task_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    task_type: CarrierTaskType
    point: Point
    required_uuv_count: int = Field(gt=0)
    entry_s: int = Field(ge=0)
    exit_s: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered_window(self) -> CarrierServiceTask:
        if self.exit_s <= self.entry_s:
            raise ValueError("carrier task exit_s must follow entry_s")
        return self


class CarrierTaskPlanner:
    """Expand executable UUV batches into perimeter deployment/recovery stops."""

    def __init__(self, *, route_planner: AStarRoutePlanner | None = None) -> None:
        self._route_planner = route_planner or AStarRoutePlanner(grid_size_m=1.0)

    def build_tasks(
        self,
        plan: ExecutableMissionPlan,
        carriers: Sequence[CarrierMissionModel],
    ) -> tuple[CarrierServiceTask, ...]:
        carrier_ids = {carrier.carrier_id for carrier in carriers}
        carrier_by_id = {carrier.carrier_id: carrier for carrier in carriers}
        tasks: list[CarrierServiceTask] = []
        for carrier_id, batches in sorted(plan.uuv_batches_by_carrier.items()):
            if carrier_id not in carrier_ids:
                raise ValueError(f"unknown carrier {carrier_id}")
            for batch in batches:
                if batch.deployment_point is None or batch.recovery_point is None:
                    raise ValueError(
                        f"batch {batch.candidate_id} is missing perimeter deployment/recovery points"
                    )
                carrier = carrier_by_id[carrier_id]
                available_uuv_ids = {
                    *carrier.onboard_uuv_ids,
                    *carrier.ready_uuv_ids,
                    *carrier.reserved_uuv_ids,
                    *carrier.recoverable_uuv_ids,
                }
                if not set(batch.uuv_ids).issubset(available_uuv_ids):
                    missing = sorted(set(batch.uuv_ids) - available_uuv_ids)
                    raise ValueError(
                        f"carrier {carrier_id} does not own batch UUVs: {missing}"
                    )
                count = len(batch.uuv_ids)
                tasks.extend(
                    (
                        CarrierServiceTask(
                            carrier_id=carrier_id,
                            task_id=f"deploy:{batch.candidate_id}",
                            candidate_id=batch.candidate_id,
                            task_type="deploy",
                            point=batch.deployment_point,
                            required_uuv_count=count,
                            entry_s=batch.entry_s,
                            exit_s=batch.exit_s,
                        ),
                        CarrierServiceTask(
                            carrier_id=carrier_id,
                            task_id=f"recover:{batch.candidate_id}",
                            candidate_id=batch.candidate_id,
                            task_type="recover",
                            point=batch.recovery_point,
                            required_uuv_count=count,
                            entry_s=batch.exit_s,
                            exit_s=batch.exit_s + max(1, batch.exit_s - batch.entry_s),
                        ),
                    )
                )
        return tuple(
            sorted(tasks, key=lambda task: (task.entry_s, task.carrier_id, task.task_id))
        )

    def build_routes(
        self,
        plan: ExecutableMissionPlan,
        carriers: Sequence[CarrierMissionModel],
        *,
        current_positions: Mapping[str, Point],
        home_positions: Mapping[str, Point],
        forbidden_regions: Sequence[Bounds] = (),
        map_bounds: Bounds = (-10_000.0, 10_000.0, -10_000.0, 10_000.0),
        current_time_s: int = 0,
        speed_mps_by_carrier: Mapping[str, float] | None = None,
    ) -> dict[str, CarrierMissionModel]:
        """Materialize complete deterministic routes for every carrier."""
        if current_time_s < 0:
            raise ValueError("current_time_s must be non-negative")
        carrier_by_id = {carrier.carrier_id: carrier for carrier in carriers}
        tasks = self.build_tasks(plan, carriers)
        routes: dict[str, CarrierMissionModel] = {}
        for carrier_id in sorted(carrier_by_id):
            carrier = carrier_by_id[carrier_id]
            if carrier_id not in current_positions or carrier_id not in home_positions:
                raise ValueError(f"missing position for carrier {carrier_id}")
            candidate_ids = {
                batch.candidate_id
                for batch in plan.uuv_batches_by_carrier.get(carrier_id, ())
            }
            carrier_tasks = tuple(
                task
                for task in tasks
                if task.carrier_id == carrier_id and task.candidate_id in candidate_ids
            )
            speed_mps = (
                speed_mps_by_carrier.get(carrier_id, 5.0)
                if speed_mps_by_carrier is not None
                else 5.0
            )
            if speed_mps <= 0.0:
                raise ValueError(f"carrier {carrier_id} speed must be positive")
            self._validate_service_slots(
                carrier,
                carrier_tasks,
                current_positions[carrier_id],
                home_positions[carrier_id],
                forbidden_regions,
                map_bounds,
                current_time_s,
                speed_mps,
            )
            route = self._route_planner.plan(
                current_positions[carrier_id],
                tuple(task.point for task in carrier_tasks),
                home_positions[carrier_id],
                forbidden_regions,
                map_bounds,
            )
            if route is None:
                raise ValueError(f"no complete carrier route for {carrier_id}")
            routes[carrier_id] = carrier.model_copy(
                update={
                    "route_status": CarrierRouteStatus.TO_DEPLOY,
                    "route_xy": route.points,
                    "stop_ids": tuple(task.task_id for task in carrier_tasks),
                    "stop_indices": route.stop_indices,
                    "stop_windows": tuple(
                        (task.entry_s, task.exit_s) for task in carrier_tasks
                    ),
                }
            )
        return routes

    def _validate_service_slots(
        self,
        carrier: CarrierMissionModel,
        tasks: Sequence[CarrierServiceTask],
        current_position: Point,
        home_position: Point,
        forbidden_regions: Sequence[Bounds],
        map_bounds: Bounds,
        current_time_s: int,
        speed_mps: float,
    ) -> None:
        """Validate each planned stop through the production matcher.

        A task is matched against a virtual slot representing the same
        carrier after all earlier stops have been committed.  The final full
        route is still rebuilt below; the matcher is the capacity/ETA gate,
        while the route rebuild is the authoritative ordering and geometry
        check.
        """
        if not tasks:
            return
        from underwater_tracking.planning.hungarian import (
            HungarianMatcher,
            VirtualServiceSlot,
        )

        ready_count = len(carrier.onboard_uuv_ids) + len(carrier.ready_uuv_ids)
        slots: list[VirtualServiceSlot] = []
        committed_points: list[Point] = []
        for index, task in enumerate(tasks):
            slots.append(
                VirtualServiceSlot(
                    slot_id=f"{carrier.carrier_id}.task.{index:04d}",
                    carrier_id=carrier.carrier_id,
                    current_xy=current_position,
                    home_xy=home_position,
                    ready_uuv_count=ready_count,
                    committed_stop_points=tuple(committed_points),
                    current_time_s=current_time_s,
                    speed_mps=speed_mps,
                    minimum_ready_uuv_count=0,
                    future_reserve_uuv_count=0,
                )
            )
            if task.task_type == "deploy":
                ready_count -= task.required_uuv_count
            else:
                ready_count += task.required_uuv_count
            committed_points.append(task.point)

        result = HungarianMatcher(
            route_planner=self._route_planner,
            forbidden_regions=forbidden_regions,
            map_bounds=map_bounds,
        ).match(tasks, slots)
        if result.unassigned_task_ids:
            raise ValueError(
                "carrier service tasks are infeasible: "
                f"{list(result.unassigned_task_ids)}"
            )

        route = self._route_planner.plan(
            current_position,
            tuple(task.point for task in tasks),
            home_position,
            forbidden_regions,
            map_bounds,
        )
        if route is None:
            raise ValueError(f"no complete carrier route for {carrier.carrier_id}")
        self.validate_route_windows(route, tasks, current_time_s, speed_mps)

    @staticmethod
    def validate_route_windows(
        route: RoutePlan,
        tasks: Sequence[CarrierServiceTask],
        current_time_s: int,
        speed_mps: float,
    ) -> None:
        """Reject a complete route whose sequential service ETA misses a window."""
        points = route.points
        stop_indices = route.stop_indices
        elapsed_s = float(current_time_s)
        previous_index = 0
        for task, stop_index in zip(tasks, stop_indices, strict=True):
            distance_m = sum(
                hypot(right[0] - left[0], right[1] - left[1])
                for left, right in zip(
                    points[previous_index : stop_index + 1],
                    points[previous_index + 1 : stop_index + 1],
                )
            )
            arrival_s = elapsed_s + distance_m / speed_mps
            if arrival_s > task.exit_s + 1e-9:
                raise ValueError(
                    f"carrier task {task.task_id} misses time window "
                    f"at {arrival_s:.3f}s > {task.exit_s}s"
                )
            elapsed_s = max(arrival_s, float(task.entry_s))
            previous_index = stop_index
