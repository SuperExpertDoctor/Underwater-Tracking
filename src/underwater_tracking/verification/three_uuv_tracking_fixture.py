"""Deterministic HTTP-driven fixture for the three-UUV live acceptance run."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from underwater_tracking.domain.execution_models import (
    GroupSensorMode,
    OperationalExecutionSnapshot,
    TaskGroupInstance,
    TaskGroupLifecycle,
    TrackingControlState,
)
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.verification.live_demo import validate_uuv_only_frame


THREE_UUV_ACCEPTANCE_STAGES = (
    "active_scan",
    "passive_track",
    "regional_handoff",
    "dedicated_track",
    "dedicated_steady",
    "dedicated_restore_pending",
    "regional_restore",
    "regional_final",
    "parallel_replacement",
)


class ThreeUuvTrackingModesFixture:
    """Drive the real engine through the committed three-UUV acceptance path."""

    def __init__(
        self,
        *,
        engine: Any,
        loop: Any,
        controller: MissionController,
    ) -> None:
        self._engine = engine
        self._loop = loop
        self._controller = controller
        self._stage_index = 0
        initial = self._published_frame()
        execution = initial.get("execution")
        if not isinstance(execution, Mapping):
            raise TypeError("three-UUV fixture requires an executable baseline")
        groups = execution.get("task_groups")
        if not isinstance(groups, list) or len(groups) != 4:
            raise RuntimeError("three-UUV fixture requires four baseline groups")
        self._deployed_by_group = {
            str(group["group_instance_id"]): tuple(
                str(member) for member in group["member_uuv_ids"]
            )
            for group in groups
            if isinstance(group, Mapping)
            and isinstance(group.get("group_instance_id"), str)
            and isinstance(group.get("member_uuv_ids"), list)
        }
        if len(self._deployed_by_group) != 4 or any(
            len(members) != 3 for members in self._deployed_by_group.values()
        ):
            raise RuntimeError("three-UUV fixture baseline group cardinality is invalid")
        self._region_ids = tuple(
            str(region["region_id"])
            for region in execution.get("regions", [])
            if isinstance(region, Mapping) and isinstance(region.get("region_id"), str)
        )
        if len(self._region_ids) != 4:
            raise RuntimeError("three-UUV fixture requires four baseline regions")
        self._dedicated_owner: TaskGroupInstance | None = None

    def advance(self, stage: str) -> dict[str, object]:
        """Advance exactly one named acceptance stage and return its real frame."""
        if stage not in THREE_UUV_ACCEPTANCE_STAGES:
            raise ValueError(f"unknown three-UUV acceptance stage {stage!r}")
        expected = THREE_UUV_ACCEPTANCE_STAGES[self._stage_index : self._stage_index + 1]
        if not expected or stage != expected[0]:
            raise ValueError(
                "three-UUV acceptance stages must be advanced in order; "
                f"expected {expected[0] if expected else 'complete'!r}"
            )
        if stage == "active_scan":
            frame = self._publish_observation(
                10,
                {"deployed_uuv_ids": self._deployed_by_group},
            )
        elif stage == "passive_track":
            frame = self._publish_observation(
                20,
                {"entry_probability": {self._region_ids[0]: 0.95}},
            )
            frame = self._publish_observation(
                30,
                {"entry_probability": {self._region_ids[0]: 0.95}},
            )
        elif stage == "regional_handoff":
            successor = self._group_for_region(self._region_ids[1])
            self._publish_observation(
                40,
                {"entry_probability": {self._region_ids[1]: 0.95}},
            )
            frame = self._publish_observation(
                50,
                {
                    "entry_probability": {self._region_ids[1]: 0.95},
                    "deployed_uuv_ids": {
                        successor.group_instance_id: successor.member_uuv_ids
                    },
                    "passive_observer_ids": {
                        successor.group_instance_id: successor.member_uuv_ids
                    },
                },
            )
        elif stage == "dedicated_track":
            owner_id = self._tracking_owner_id()
            if not self._controller.set_dedicated_owner("target_00", owner_id):
                raise RuntimeError("dedicated acceptance owner could not be installed")
            self._dedicated_owner = self._group_by_id(owner_id)
            frame = self._publish_current_state(sim_time_s=55)
        elif stage == "dedicated_steady":
            exiting = tuple(
                group
                for group in self._controller.snapshot().task_groups
                if group.lifecycle is TaskGroupLifecycle.EXITING
            )
            if len(exiting) != 3:
                raise RuntimeError("dedicated acceptance did not create three exiting groups")
            for group in exiting:
                self._complete_group_exit(group, sim_time_s=60)
            frame = self._publish_observation(
                60,
                {
                    "disappeared_group_instance_ids": tuple(
                        group.group_instance_id for group in exiting
                    )
                },
            )
        elif stage == "dedicated_restore_pending":
            owner = self._dedicated_owner
            if owner is None:
                raise RuntimeError("dedicated acceptance owner is unavailable")
            frame = self._publish_observation(
                70,
                {
                    "mileage_m": {
                        member_id: 45_000.0 for member_id in owner.member_uuv_ids
                    }
                },
            )
        elif stage == "regional_restore":
            pending_id = self._pending_successor_id()
            pending = self._group_by_id(pending_id)
            frame = self._publish_observation(
                80,
                {
                    "deployed_uuv_ids": {
                        pending.group_instance_id: pending.member_uuv_ids
                    },
                    "passive_observer_ids": {
                        pending.group_instance_id: pending.member_uuv_ids
                    },
                },
            )
        elif stage == "regional_final":
            owner = self._dedicated_owner
            if owner is None:
                raise RuntimeError("dedicated acceptance owner is unavailable")
            self._complete_group_exit(owner, sim_time_s=90)
            frame = self._publish_observation(
                90,
                {"disappeared_group_instance_ids": (owner.group_instance_id,)},
            )
        else:
            frame = self._publish_parallel_replacement()
        self._stage_index += 1
        return frame

    def _published_frame(self) -> dict[str, object]:
        frame = self._loop.hub.snapshot()
        if frame is None:
            raise RuntimeError("three-UUV fixture publisher has no current frame")
        payload = frame.model_dump(mode="json")
        violations = validate_uuv_only_frame(payload)
        if violations:
            raise RuntimeError(f"three-UUV fixture published an invalid frame: {violations}")
        return payload

    def _publish_observation(
        self, sim_time_s: int, observations: Mapping[str, object]
    ) -> dict[str, object]:
        self._controller.advance(sim_time_s, observations)
        self._engine._clock.sim_time_s = sim_time_s
        self._engine._reconcile_uuv_mission_state()
        self._loop.publish_latest()
        return self._published_frame()

    def _publish_current_state(self, *, sim_time_s: int | None = None) -> dict[str, object]:
        if sim_time_s is not None:
            self._engine._clock.sim_time_s = sim_time_s
        self._engine._reconcile_uuv_mission_state()
        self._loop.publish_latest()
        return self._published_frame()

    def _group_for_region(self, region_id: str) -> TaskGroupInstance:
        for group in self._controller.snapshot().task_groups:
            if group.region_id == region_id:
                return group
        raise RuntimeError(f"three-UUV fixture region has no group: {region_id}")

    def _group_by_id(self, group_id: str) -> TaskGroupInstance:
        for group in self._controller.snapshot().task_groups:
            if group.group_instance_id == group_id:
                return group
        raise RuntimeError(f"three-UUV fixture group is unavailable: {group_id}")

    def _tracking_owner_id(self) -> str:
        owner_id = self._controller.snapshot().tracking_control.tracking_owner_group_id
        if not isinstance(owner_id, str) or not owner_id:
            raise RuntimeError("three-UUV fixture has no tracking owner")
        return owner_id

    def _pending_successor_id(self) -> str:
        successor_id = self._controller.snapshot().tracking_control.pending_successor_group_id
        if not isinstance(successor_id, str) or not successor_id:
            raise RuntimeError("three-UUV fixture has no pending successor")
        return successor_id

    def _complete_group_exit(self, group: TaskGroupInstance, *, sim_time_s: int) -> None:
        for member_id in group.member_uuv_ids:
            exit_point = self._engine._boundary_exit_points.get(member_id)
            if exit_point is None:
                raise RuntimeError(f"three-UUV fixture has no exit point for {member_id}")
            self._engine._uuvs[member_id].position_xy = exit_point
            if not self._engine.complete_boundary_exit(
                member_id,
                sim_time_s=sim_time_s,
            ):
                raise RuntimeError(f"three-UUV fixture boundary exit failed for {member_id}")

    def _publish_parallel_replacement(self) -> dict[str, object]:
        coordinator = self._loop._execution_coordinator
        current = coordinator.current
        if current is None:
            raise RuntimeError("three-UUV fixture has no current execution snapshot")
        candidate = _parallel_replacement_candidate(current)

        def apply_candidate(snapshot: OperationalExecutionSnapshot) -> bool:
            applied = self._controller.reconcile_execution_snapshot(snapshot)
            self._engine._reconcile_uuv_mission_state()
            return applied.plan_revision == snapshot.execution_revision

        result = coordinator.commit(candidate, apply=apply_candidate)
        if not result.committed:
            raise RuntimeError(result.reason or "parallel replacement was rejected")
        return self._publish_current_state()


def _parallel_replacement_candidate(
    snapshot: OperationalExecutionSnapshot,
) -> OperationalExecutionSnapshot:
    current_groups = tuple(
        group for group in snapshot.task_groups if isinstance(group, TaskGroupInstance)
    )
    if len(current_groups) != 4:
        raise RuntimeError("regional baseline must contain four runtime groups")
    revision = snapshot.execution_revision + 1
    deployment_revision = max(
        group.deployment_revision for group in current_groups
    ) + 1
    old_by_region = {group.region_id: group for group in current_groups}
    owner_region = next(
        group.region_id
        for group in current_groups
        if group.group_instance_id
        == snapshot.tracking_control.tracking_owner_group_id
    )
    new_groups: list[TaskGroupInstance] = []
    for region in snapshot.regions:
        old = old_by_region[region.region_id]
        lifecycle = (
            TaskGroupLifecycle.PASSIVE_TRACK
            if region.region_id == owner_region
            else TaskGroupLifecycle.ACTIVE_SCAN
        )
        sensor_mode = (
            GroupSensorMode.PASSIVE
            if lifecycle is TaskGroupLifecycle.PASSIVE_TRACK
            else GroupSensorMode.ACTIVE
        )
        new_groups.append(
            old.model_copy(
                update={
                    "group_instance_id": (
                        f"{region.region_id}:acceptance-replacement:"
                        f"deploy:{deployment_revision:06d}"
                    ),
                    "deployment_revision": deployment_revision,
                    "member_uuv_ids": tuple(
                        f"{region.region_id}:acceptance-replacement:member:{index:02d}"
                        for index in range(1, 4)
                    ),
                    "lifecycle": lifecycle,
                    "sensor_mode": sensor_mode,
                    "ownership_status": (
                        "owner" if region.region_id == owner_region else "candidate"
                    ),
                    "source_group_instance_id": old.group_instance_id,
                    "reason": "parallel_geometry_replacement",
                }
            )
        )
    shifted_regions = tuple(
        region.model_copy(
            update={
                "execution_revision": revision,
                "geometry": tuple(
                    (x + 100.0, y + 100.0) for x, y in region.geometry
                ),
                "center": (region.center[0] + 100.0, region.center[1] + 100.0),
                "geometry_revision": region.geometry_revision + 1,
                "task_group_id": new_groups[region.slot_index - 1].group_instance_id,
            }
        )
        for region in snapshot.regions
    )
    new_owner_id = next(
        group.group_instance_id
        for group in new_groups
        if group.region_id == owner_region
    )
    return snapshot.model_copy(
        deep=True,
        update={
            "execution_revision": revision,
            "base_execution_revision": snapshot.execution_revision,
            "regions": shifted_regions,
            "task_groups": tuple(new_groups),
            "tracking_control": TrackingControlState(
                mode="regional",
                tracking_owner_group_id=new_owner_id,
                source_event_ids=snapshot.tracking_control.source_event_ids,
            ),
        },
    )
