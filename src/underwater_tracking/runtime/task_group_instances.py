"""Deterministic three-UUV deployment instances and region proposals."""

from __future__ import annotations

from dataclasses import dataclass, field

from underwater_tracking.domain.execution_models import (
    ExecutionRegion,
    GroupSensorMode,
    TaskGroupInstance,
    TaskGroupLifecycle,
)


@dataclass(frozen=True, slots=True)
class AlwaysAvailableTaskGroupFactory:
    """Create reproducible three-member groups without a reserve inventory."""

    scenario_id: str

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must not be empty")

    def create(
        self,
        region: ExecutionRegion | str | None = None,
        deployment_revision: int | None = None,
        reason: str | None = None,
        sensor_mode: GroupSensorMode | str | None = None,
        *,
        target_id: str | None = None,
        region_id: str | None = None,
    ) -> TaskGroupInstance:
        if region is not None and region_id is not None:
            raise ValueError("region and region_id cannot both be provided")
        if region is None and region_id is None:
            raise ValueError("region or region_id is required")
        if deployment_revision is None or deployment_revision < 1:
            raise ValueError("deployment_revision must be positive")
        if reason is None or not reason.strip():
            raise ValueError("reason must not be empty")
        if sensor_mode is None:
            raise ValueError("sensor_mode is required")

        region_value = region_id if region is None or isinstance(region, str) else region.region_id
        if isinstance(region, str) and region_id is None:
            region_value = region
        if not region_value or not region_value.strip():
            raise ValueError("region ID must not be empty")
        resolved_target_id = (
            target_id
            or (region.target_id if isinstance(region, ExecutionRegion) else None)
            or region_value.partition(":task:")[0]
        )
        if not resolved_target_id.strip():
            raise ValueError("target_id must not be empty")
        group_id = (
            f"{self.scenario_id}:{region_value}:deploy:{deployment_revision:06d}"
        )
        return TaskGroupInstance(
            group_instance_id=group_id,
            target_id=resolved_target_id,
            region_id=region_value,
            deployment_revision=deployment_revision,
            member_uuv_ids=tuple(
                f"{group_id}:member:{index:02d}" for index in range(1, 4)
            ),
            lifecycle=TaskGroupLifecycle.ENTERING,
            sensor_mode=sensor_mode,
            ownership_status="candidate",
            reason=reason,
            evidence_ids=(f"{group_id}:created",),
        )


@dataclass(slots=True)
class RegionTransitionQueue:
    """Keep only the newest pending region proposal for each stable slot."""

    _pending_by_slot: dict[int, ExecutionRegion] = field(default_factory=dict)

    def offer(self, region: ExecutionRegion) -> None:
        pending = self._pending_by_slot.get(region.slot_index)
        candidate_revision = (region.execution_revision, region.geometry_revision)
        pending_revision = (
            (pending.execution_revision, pending.geometry_revision)
            if pending is not None
            else None
        )
        if pending_revision is None or candidate_revision > pending_revision:
            self._pending_by_slot[region.slot_index] = region

    def pop_latest(self, slot: int) -> ExecutionRegion | None:
        return self._pending_by_slot.pop(slot, None)


__all__ = ["AlwaysAvailableTaskGroupFactory", "RegionTransitionQueue"]
