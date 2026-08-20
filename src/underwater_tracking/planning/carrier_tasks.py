from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import Field, model_validator

from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    CarrierRouteStatus,
    ExecutableMissionPlan,
)
from underwater_tracking.domain.models import StrictModel
from underwater_tracking.planning.astar import AStarRoutePlanner, Bounds

Finite = Annotated[float, Field(allow_inf_nan=False)]
Point = tuple[Finite, Finite]
CarrierTaskType = Literal["deploy", "recover"]


class CarrierServiceTask(StrictModel):
    """One carrier stop for deploying or recovering a UUV batch."""

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
        tasks: list[CarrierServiceTask] = []
        for carrier_id, batches in sorted(plan.uuv_batches_by_carrier.items()):
            if carrier_id not in carrier_ids:
                raise ValueError(f"unknown carrier {carrier_id}")
            for batch in batches:
                if batch.deployment_point is None or batch.recovery_point is None:
                    raise ValueError(
                        f"batch {batch.candidate_id} is missing perimeter deployment/recovery points"
                    )
                count = len(batch.uuv_ids)
                tasks.extend(
                    (
                        CarrierServiceTask(
                            task_id=f"deploy:{batch.candidate_id}",
                            candidate_id=batch.candidate_id,
                            task_type="deploy",
                            point=batch.deployment_point,
                            required_uuv_count=count,
                            entry_s=batch.entry_s,
                            exit_s=batch.exit_s,
                        ),
                        CarrierServiceTask(
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
        return tuple(sorted(tasks, key=lambda task: (task.entry_s, task.task_id)))

    def build_routes(
        self,
        plan: ExecutableMissionPlan,
        carriers: Sequence[CarrierMissionModel],
        *,
        current_positions: Mapping[str, Point],
        home_positions: Mapping[str, Point],
        forbidden_regions: Sequence[Bounds] = (),
        map_bounds: Bounds = (-10_000.0, 10_000.0, -10_000.0, 10_000.0),
    ) -> dict[str, CarrierMissionModel]:
        """Materialize complete deterministic routes for every carrier."""
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
                if task.candidate_id in candidate_ids
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
                }
            )
        return routes
