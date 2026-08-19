from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field, model_validator

from underwater_tracking.domain.mission_models import CarrierMissionModel, ExecutableMissionPlan
from underwater_tracking.domain.models import StrictModel

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
