"""Repository-native acceptance of the three-UUV tracking state machine."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from underwater_tracking.cli import _AgentLoop, _mission_controller_for
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.execution_models import (
    GroupSensorMode,
    OperationalExecutionSnapshot,
    TaskGroupInstance,
    TaskGroupLifecycle,
    TrackingControlState,
)
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.verification.live_demo import validate_uuv_only_frame
from underwater_tracking.verification.uuv_tracking_coverage_runner import NoNetworkLLM


def _group_by_region(
    groups: tuple[TaskGroupInstance, ...],
) -> dict[str, TaskGroupInstance]:
    return {group.region_id: group for group in groups}


class _RuntimeAcceptanceHarness:
    """Drive the production engine and publisher with explicit observations."""

    def __init__(self, tmp_path: Path) -> None:
        config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
        controller = _mission_controller_for(config)
        if controller is None:
            raise RuntimeError("UUV-only scenario did not create a mission controller")
        self.controller: MissionController = controller
        self.loop = _AgentLoop(
            config,
            database_path=tmp_path / "agent.db",
            llm={"master": NoNetworkLLM()},
            run_id="three-uuv-tracking-modes-acceptance",
            steps=64,
            seed=20260904,
        )
        self.engine = SimulationEngine(
            config,
            seed=20260904,
            output_dir=tmp_path / "frames",
            mission_controller=controller,
        )
        try:
            self.loop.attach(self.engine)
            if self.loop.install_deterministic_baseline(
                self.engine.publication_situation()
            ) is None:
                raise RuntimeError("deterministic baseline was not installed")
            self.loop.publish_latest()
        except BaseException:
            self.loop.close(timeout_s=30.0)
            raise

    def publish_observation(
        self,
        sim_time_s: int,
        observations: Mapping[str, object],
    ) -> dict[str, object]:
        self.controller.advance(sim_time_s, observations)
        self.engine._reconcile_uuv_mission_state()
        self.loop.publish_latest()
        frame = self.loop.hub.snapshot()
        if frame is None:
            raise AssertionError("publisher did not produce an operational frame")
        payload = frame.model_dump(mode="json")
        violations = validate_uuv_only_frame(payload)
        if violations:
            execution = payload.get("execution")
            execution_groups = (
                execution.get("task_groups", [])
                if isinstance(execution, dict)
                else []
            )
            group_summary = [
                (
                    group.get("group_instance_id"),
                    group.get("region_id"),
                    group.get("lifecycle"),
                    group.get("sensor_mode"),
                )
                for group in execution_groups
                if isinstance(group, dict)
            ]
            uuv_summary = [
                (
                    item.get("uuv_id"),
                    item.get("group_instance_id"),
                    item.get("group_lifecycle"),
                    item.get("sensor_mode"),
                )
                for item in payload.get("uuvs", [])
                if isinstance(item, dict)
            ]
            raise AssertionError(
                f"invalid published frame: {violations}; "
                f"groups={group_summary}; "
                f"uuvs={uuv_summary}; "
                f"carrier_errors={self.loop.carrier_error_details}; "
                f"controller={self.controller.snapshot().task_groups}"
            )
        if self.loop.carrier_error_count:
            raise AssertionError(self.loop.carrier_error_details)
        return payload

    def publish_current_state(self) -> dict[str, object]:
        self.engine._reconcile_uuv_mission_state()
        self.loop.publish_latest()
        frame = self.loop.hub.snapshot()
        if frame is None:
            raise AssertionError("publisher did not produce an operational frame")
        payload = frame.model_dump(mode="json")
        violations = validate_uuv_only_frame(payload)
        if violations:
            execution = payload.get("execution")
            execution_groups = (
                execution.get("task_groups", [])
                if isinstance(execution, dict)
                else []
            )
            group_summary = [
                (
                    group.get("group_instance_id"),
                    group.get("region_id"),
                    group.get("lifecycle"),
                    group.get("sensor_mode"),
                )
                for group in execution_groups
                if isinstance(group, dict)
            ]
            uuv_summary = [
                (
                    item.get("uuv_id"),
                    item.get("group_instance_id"),
                    item.get("group_lifecycle"),
                    item.get("sensor_mode"),
                )
                for item in payload.get("uuvs", [])
                if isinstance(item, dict)
            ]
            raise AssertionError(
                f"invalid published frame: {violations}; "
                f"groups={group_summary}; uuvs={uuv_summary}"
            )
        return payload

    def complete_group_exit(self, group: TaskGroupInstance, sim_time_s: int) -> None:
        for member_id in group.member_uuv_ids:
            exit_point = self.engine._boundary_exit_points.get(member_id)
            if exit_point is None:
                continue
            self.engine._uuvs[member_id].position_xy = exit_point
            if not self.engine.complete_boundary_exit(
                member_id,
                sim_time_s=sim_time_s,
            ):
                raise AssertionError(f"boundary exit did not complete for {member_id}")

    def close(self) -> None:
        self.engine.logger.close()
        if not self.loop.close(timeout_s=30.0):
            raise AssertionError("agent loop did not close cleanly")


def _event_types(payloads: list[dict[str, object]]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for payload in payloads:
        for key in ("events", "mission_events"):
            raw_events = payload.get(key)
            if not isinstance(raw_events, list):
                continue
            for raw_event in raw_events:
                if not isinstance(raw_event, dict):
                    continue
                event_id = raw_event.get("event_id")
                event_type = raw_event.get("event_type")
                if (
                    isinstance(event_id, str)
                    and event_id not in seen
                    and isinstance(event_type, str)
                ):
                    seen.add(event_id)
                    result.append(event_type)
    return tuple(result)


def _mission_event_types(payloads: list[dict[str, object]]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for payload in payloads:
        raw_events = payload.get("mission_events")
        if not isinstance(raw_events, list):
            continue
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            event_id = raw_event.get("event_id")
            event_type = raw_event.get("event_type")
            if (
                isinstance(event_id, str)
                and event_id not in seen
                and isinstance(event_type, str)
            ):
                seen.add(event_id)
                result.append(event_type)
    return tuple(result)


def _ordered_subsequence(
    values: tuple[str, ...], expected: tuple[str, ...]
) -> bool:
    position = 0
    for value in values:
        if position < len(expected) and value == expected[position]:
            position += 1
    return position == len(expected)


def _runtime_groups(payload: Mapping[str, object]) -> list[dict[str, object]]:
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        raise TypeError("execution projection is missing")
    groups = execution.get("task_groups")
    if not isinstance(groups, list):
        raise TypeError("runtime task groups are missing")
    return groups


def _visible_uuv_count(payload: Mapping[str, object]) -> int:
    groups = _runtime_groups(payload)
    visible_group_ids = {
        group["group_instance_id"]
        for group in groups
        if group.get("lifecycle") != "disappeared"
    }
    uuvs = payload.get("uuvs")
    if not isinstance(uuvs, list):
        return 0
    return sum(
        item.get("physically_exposed") is True
        and item.get("group_instance_id") in visible_group_ids
        for item in uuvs
        if isinstance(item, dict)
    )


def _assert_group_shape(
    payload: Mapping[str, object],
    *,
    expected_count: int,
    expected_lifecycles: set[str] | None = None,
) -> None:
    groups = _runtime_groups(payload)
    assert len(groups) == expected_count
    members = [
        member
        for group in groups
        for member in group.get("member_uuv_ids", [])
    ]
    assert all(len(group.get("member_uuv_ids", [])) == 3 for group in groups)
    assert len(members) == len(set(members)) == expected_count * 3
    if expected_lifecycles is not None:
        assert {str(group["lifecycle"]) for group in groups} == expected_lifecycles


def _parallel_replacement_candidate(
    snapshot: OperationalExecutionSnapshot,
) -> OperationalExecutionSnapshot:
    current_groups = tuple(
        group for group in snapshot.task_groups if isinstance(group, TaskGroupInstance)
    )
    if len(current_groups) != 4:
        raise AssertionError("regional baseline must contain four runtime groups")
    revision = snapshot.execution_revision + 1
    deployment_revision = max(
        group.deployment_revision for group in current_groups
    ) + 1
    old_by_region = _group_by_region(current_groups)
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
                "geometry": tuple((x + 100.0, y + 100.0) for x, y in region.geometry),
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


def test_three_uuv_modes_and_parallel_replacement_are_published_end_to_end(
    tmp_path: Path,
) -> None:
    harness = _RuntimeAcceptanceHarness(tmp_path)
    published: list[dict[str, object]] = []
    try:
        initial = harness.loop.hub.snapshot()
        assert initial is not None
        initial_payload = initial.model_dump(mode="json")
        published.append(initial_payload)
        initial_execution = initial_payload["execution"]
        assert isinstance(initial_execution, dict)
        initial_groups = initial_execution["task_groups"]
        assert isinstance(initial_groups, list)
        assert len(initial_groups) == 4
        deployed_by_group = {
            group["group_instance_id"]: tuple(group["member_uuv_ids"])
            for group in initial_groups
        }
        region_ids = tuple(
            region["region_id"]
            for region in initial_execution["regions"]
            if isinstance(region, dict)
        )
        assert len(region_ids) == 4

        published.append(
            harness.publish_observation(
                10,
                {"deployed_uuv_ids": deployed_by_group},
            )
        )
        for sim_time_s in (20, 30):
            published.append(
                harness.publish_observation(
                    sim_time_s,
                    {"entry_probability": {region_ids[0]: 0.95}},
                )
            )
        passive_frame = published[-1]
        _assert_group_shape(passive_frame, expected_count=4)
        passive_groups = _runtime_groups(passive_frame)
        assert sum(group["lifecycle"] == "passive_track" for group in passive_groups) == 1
        owner_id = passive_frame["execution"]["tracking_control"][
            "tracking_owner_group_id"
        ]
        assert owner_id == next(
            group["group_instance_id"]
            for group in passive_groups
            if group["region_id"] == region_ids[0]
        )

        successor_id = next(
            group["group_instance_id"]
            for group in passive_groups
            if group["region_id"] == region_ids[1]
        )
        successor_members = tuple(
            next(group for group in passive_groups if group["group_instance_id"] == successor_id)[
                "member_uuv_ids"
            ]
        )
        for sim_time_s in (40, 50):
            observations: dict[str, object] = {
                "entry_probability": {region_ids[1]: 0.95},
            }
            if sim_time_s == 50:
                observations.update(
                    {
                        "deployed_uuv_ids": {successor_id: successor_members},
                        "passive_observer_ids": {successor_id: successor_members},
                    }
                )
            published.append(harness.publish_observation(sim_time_s, observations))
        regional_handoff_frame = published[-1]
        assert regional_handoff_frame["execution"]["tracking_control"][
            "tracking_owner_group_id"
        ] == successor_id

        assert harness.controller.set_dedicated_owner("target_00", successor_id)
        published.append(harness.publish_current_state())
        dedicated_frame = published[-1]
        _assert_group_shape(dedicated_frame, expected_count=4)
        assert dedicated_frame["execution"]["tracking_control"]["mode"] == "dedicated"
        assert _visible_uuv_count(dedicated_frame) == 12

        exiting_groups = tuple(
            group
            for group in harness.controller.snapshot().task_groups
            if group.lifecycle is TaskGroupLifecycle.EXITING
        )
        assert len(exiting_groups) == 3
        for group in exiting_groups:
            harness.complete_group_exit(group, sim_time_s=60)
        published.append(
            harness.publish_observation(
                60,
                {
                    "disappeared_group_instance_ids": tuple(
                        group.group_instance_id for group in exiting_groups
                    )
                },
            )
        )
        dedicated_steady_frame = published[-1]
        _assert_group_shape(dedicated_steady_frame, expected_count=1)
        assert dedicated_steady_frame["execution"]["tracking_control"]["mode"] == "dedicated"
        assert _visible_uuv_count(dedicated_steady_frame) == 3
        assert sum(
            uuv.get("sensor_mode") == "active"
            for uuv in dedicated_steady_frame["uuvs"]
            if isinstance(uuv, dict)
            and uuv.get("group_instance_id")
        ) == 0

        owner = next(
            group
            for group in harness.controller.snapshot().task_groups
            if group.group_instance_id
            == dedicated_steady_frame["execution"]["tracking_control"][
                "tracking_owner_group_id"
            ]
        )
        published.append(
            harness.publish_observation(
                70,
                {
                    "mileage_m": {
                        member_id: 45_000.0 for member_id in owner.member_uuv_ids
                    }
                },
            )
        )
        restore_frame = published[-1]
        _assert_group_shape(restore_frame, expected_count=5)
        assert restore_frame["execution"]["tracking_control"]["mode"] == "dedicated"
        assert _visible_uuv_count(restore_frame) == 15
        pending_successor_id = restore_frame["execution"]["tracking_control"][
            "pending_successor_group_id"
        ]
        assert isinstance(pending_successor_id, str)
        restore_groups = _runtime_groups(restore_frame)
        pending_successor = next(
            group
            for group in restore_groups
            if group["group_instance_id"] == pending_successor_id
        )
        published.append(
            harness.publish_observation(
                80,
                {
                    "deployed_uuv_ids": {
                        pending_successor_id: tuple(pending_successor["member_uuv_ids"])
                    },
                    "passive_observer_ids": {
                        pending_successor_id: tuple(pending_successor["member_uuv_ids"])
                    },
                },
            )
        )
        restored_frame = published[-1]
        assert restored_frame["execution"]["tracking_control"]["mode"] == "regional"
        assert restored_frame["execution"]["tracking_control"][
            "tracking_owner_group_id"
        ] == pending_successor_id
        restored_owner = next(
            group
            for group in harness.controller.snapshot().task_groups
            if group.group_instance_id == pending_successor_id
        )
        harness.complete_group_exit(owner, sim_time_s=90)
        published.append(
            harness.publish_observation(
                90,
                {
                    "disappeared_group_instance_ids": (owner.group_instance_id,)
                },
            )
        )
        final_regional_frame = published[-1]
        _assert_group_shape(final_regional_frame, expected_count=4)
        assert _visible_uuv_count(final_regional_frame) == 12
        assert restored_owner.member_uuv_ids

        current = harness.loop._execution_coordinator.current
        assert current is not None
        candidate = _parallel_replacement_candidate(current)

        def apply_candidate(snapshot: OperationalExecutionSnapshot) -> bool:
            applied = harness.controller.reconcile_execution_snapshot(snapshot)
            harness.engine._reconcile_uuv_mission_state()
            return applied.plan_revision == snapshot.execution_revision

        result = harness.loop._execution_coordinator.commit(
            candidate,
            apply=apply_candidate,
        )
        assert result.committed, result.reason
        published.append(harness.publish_current_state())
        parallel_frame = published[-1]
        _assert_group_shape(parallel_frame, expected_count=8)
        assert _visible_uuv_count(parallel_frame) == 24
        assert len(parallel_frame["execution"]["replacements"]) == 4
        assert parallel_frame["execution"]["tracking_control"]["mode"] == "regional"

        expected_events = (
            "task_group_entering",
            "active_scan_started",
            "passive_track_started",
            "tracking_ownership_transferred",
            "dedicated_tracking_started",
            "task_group_exiting",
            "task_group_disappeared",
            "dedicated_release_threshold_reached",
            "regional_mode_restored",
        )
        mission_event_types = _mission_event_types(published)
        assert _ordered_subsequence(mission_event_types, expected_events), (
            mission_event_types
        )
        assert mission_event_types.index("tracking_ownership_transferred") < (
            mission_event_types.index("task_group_exiting")
        )
        assert "region_replacement_started" in mission_event_types

        all_execution_frames = [payload["execution"] for payload in published]
        for execution in all_execution_frames:
            assert execution["tracking_policy"]["region_count"] == 4
            assert execution["tracking_policy"]["task_group_size"] == 3
            assert execution["tracking_policy"]["task_region_side_m"] == 2000.0
            assert execution["tracking_policy"]["target_detection_radius_m"] == 1000.0
            assert execution["tracking_policy"]["uuv_active_detection_radius_m"] == 600.0
            assert execution["tracking_policy"]["uuv_passive_detection_radius_m"] == 600.0
    finally:
        harness.close()
