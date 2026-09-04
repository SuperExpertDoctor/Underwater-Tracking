import pytest

from underwater_tracking.config.models import TrackingPolicyConfig
from tests.domain.test_execution_models import _snapshot as _execution_snapshot
from underwater_tracking.cli import _mission_controller_for
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.execution_models import (
    GroupSensorMode,
    OperationalExecutionSnapshot,
    TaskGroupAssignment,
    TaskGroupInstance,
    TaskGroupLifecycle,
    TrackingControlState,
)
from underwater_tracking.domain.mission_models import (
    AcceptedHandoffObservation,
    CarrierMissionModel,
    ExecutableMissionPlan,
    HandoffEvidence,
    MissionSnapshot,
    RegionLifecycle,
    RegionMissionState,
    UUVMissionBatch,
    UUVMissionMode,
)
from underwater_tracking.runtime.mission_controller import (
    MissionController,
    MissionSnapshot as RuntimeMissionSnapshot,
    _select_runtime_group,
)


def test_controller_rejects_expired_execution_snapshot_before_assignment() -> None:
    controller = MissionController(scenario_id="S1", execution_hard_stale_s=900)
    controller.advance(901, {})
    snapshot = _execution_snapshot().model_copy(
        update={"valid_from_s": 0.0, "valid_until_s": 450.0}
    )

    assert controller.apply_execution_snapshot(snapshot) is False
    assert controller.snapshot().plan_revision == 0


def test_controller_rejects_failed_execution_snapshot_model() -> None:
    controller = MissionController(scenario_id="S1")
    invalid = {
        **_execution_snapshot().model_dump(mode="python"),
        "valid_until_s": 0.0,
    }

    assert controller.apply_execution_snapshot(invalid) is False  # type: ignore[arg-type]
    assert controller.snapshot().plan_revision == 0


def test_controller_rejects_not_yet_valid_execution_snapshot() -> None:
    controller = MissionController(scenario_id="S1", execution_hard_stale_s=900)
    snapshot = _execution_snapshot().model_copy(
        update={"valid_from_s": 100.0, "valid_until_s": 550.0}
    )

    assert controller.apply_execution_snapshot(snapshot) is False
    assert controller.snapshot().plan_revision == 0


def test_execution_snapshot_status_overrides_rolling_lifecycle_progress() -> None:
    controller = MissionController(scenario_id="S1")
    initial = _execution_snapshot()
    controller.advance(int(initial.valid_from_s), {})
    assert controller.apply_execution_snapshot(initial) is True
    assert controller.snapshot().regions[0].lifecycle is RegionLifecycle.ACTIVE_SCAN

    planned = initial.model_copy(
        deep=True,
        update={
            "execution_revision": initial.execution_revision + 1,
            "base_execution_revision": initial.execution_revision,
            "regions": tuple(
                region.model_copy(
                    update={
                        "execution_revision": initial.execution_revision + 1,
                        "status": "planned",
                    }
                )
                for region in initial.regions
            ),
            "task_groups": tuple(
                group.model_copy(
                    update={"execution_revision": initial.execution_revision + 1}
                )
                for group in initial.task_groups
            ),
        },
    )

    assert controller.apply_execution_snapshot(planned) is True
    assert controller.snapshot().regions[0].lifecycle is RegionLifecycle.ACTIVE_SCAN


def test_execution_snapshot_uses_group_lifecycle_for_all_members() -> None:
    controller = MissionController(scenario_id="S1")
    initial = _execution_snapshot()
    controller.advance(int(initial.valid_from_s), {})
    passive = initial.model_copy(
        deep=True,
        update={
            "regions": tuple(
                region.model_copy(update={"status": "passive"})
                for region in initial.regions
            ),
            "task_groups": tuple(
                group.model_copy(
                    update={
                        "lifecycle": TaskGroupLifecycle.PASSIVE_TRACK,
                        "sensor_mode": GroupSensorMode.PASSIVE,
                    }
                )
                for group in initial.task_groups
            ),
        },
    )

    assert controller.apply_execution_snapshot(passive) is True

    snapshot = controller.snapshot()
    for group in passive.task_groups:
        assert all(
            snapshot.uuv_modes[uuv_id] is UUVMissionMode.PASSIVE_TRACK
            for uuv_id in group.member_uuv_ids
        )


def test_execution_snapshot_preserves_recovered_region_lifecycle() -> None:
    controller = MissionController(scenario_id="S1")
    initial = _execution_snapshot()
    controller.advance(int(initial.valid_from_s), {})
    assert controller.apply_execution_snapshot(initial) is True
    controller._regions[initial.regions[0].region_id] = controller._regions[
        initial.regions[0].region_id
    ].model_copy(update={"lifecycle": RegionLifecycle.RECOVERED})

    refreshed = initial.model_copy(
        deep=True,
        update={
            "execution_revision": initial.execution_revision + 1,
            "base_execution_revision": initial.execution_revision,
            "regions": tuple(
                region.model_copy(
                    update={
                        "execution_revision": initial.execution_revision + 1,
                        "status": "monitoring_complete"
                        if region.region_id == initial.regions[0].region_id
                        else region.status,
                    }
                )
                for region in initial.regions
            ),
            "task_groups": tuple(
                group.model_copy(
                    update={"execution_revision": initial.execution_revision + 1}
                )
                for group in initial.task_groups
            ),
        },
    )

    assert controller.apply_execution_snapshot(refreshed) is True
    assert controller.snapshot().regions[0].lifecycle is RegionLifecycle.RECOVERED


def test_default_live_controller_registers_authoritative_onboard_inventory() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = _mission_controller_for(config)
    assert controller is not None

    snapshot = controller.snapshot()
    assert len(snapshot.uuv_resources) == 12
    assert set(snapshot.uuv_modes.values()) == {UUVMissionMode.ONBOARD}
    assert {
        resource.carrier_id for resource in snapshot.uuv_resources.values()
    } == {"carrier_02", "carrier_03", "carrier_04"}
    assert all(resource.mileage_m == 0.0 for resource in snapshot.uuv_resources.values())


def test_uuv_only_controller_uses_tracking_policy_when_legacy_fields_change() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    policy = config.scenario.tracking_policy.model_copy(
        update={
            "max_uuv_mileage_m": 1_000.0,
            "dedicated_release_remaining_mileage_m": 200.0,
        }
    )
    config = config.model_copy(
        update={
            "scenario": config.scenario.model_copy(
                update={
                    "tracking_policy": policy,
                    "region_entry_probability_threshold": 0.99,
                    "region_transition_confirm_cycles": 7,
                    "resource_warning_mileage_fraction": 0.99,
                }
            )
        }
    )
    controller = _mission_controller_for(config)
    assert controller is not None

    controller.apply_verified_plan(plan())
    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})
    controller.advance(20, {"entry_probability": {"R1": 0.8}})
    transitioned = controller.advance(30, {"entry_probability": {"R1": 0.8}})

    assert transitioned.regions[0].lifecycle is RegionLifecycle.PASSIVE_TRACK
    assert controller.max_uuv_mileage_m == 1_000.0
    assert controller.resource_warning_mileage_m == 200.0


def test_configured_uuv_owner_cannot_change_on_observation() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = _mission_controller_for(config)
    assert controller is not None
    before = controller.snapshot().uuv_resources["uuv_00"]

    controller.advance(
        30,
        {
            "mileage_m": {"uuv_00": 10.0},
            "deployment_state": {"uuv_00": "onboard"},
        },
    )

    after = controller.snapshot().uuv_resources["uuv_00"]
    assert after.carrier_id == before.carrier_id
    assert after.mileage_m == 10.0


def _runtime_group(
    slot: int,
    *,
    lifecycle: TaskGroupLifecycle = TaskGroupLifecycle.ACTIVE_SCAN,
    ownership_status: str = "candidate",
    deployment_revision: int = 9,
) -> TaskGroupInstance:
    region_id = f"target_00:task:{slot:02d}"
    group_id = f"S1:{region_id}:deploy:{deployment_revision:06d}"
    sensor_mode = (
        GroupSensorMode.PASSIVE
        if lifecycle
        in {
            TaskGroupLifecycle.PASSIVE_TRACK,
            TaskGroupLifecycle.DEDICATED_TRACK,
            TaskGroupLifecycle.DEDICATED_RELEASE_PENDING,
        }
        else GroupSensorMode.ACTIVE
    )
    return TaskGroupInstance(
        group_instance_id=group_id,
        target_id="target_00",
        region_id=region_id,
        deployment_revision=deployment_revision,
        member_uuv_ids=tuple(
            f"{group_id}:member:{member_index:02d}"
            for member_index in range(1, 4)
        ),
        lifecycle=lifecycle,
        sensor_mode=sensor_mode,
        ownership_status=ownership_status,
        reason="initial_deployment",
        evidence_ids=(f"{group_id}:evidence",),
    )


def _runtime_execution_snapshot(
    *,
    r1_lifecycle: TaskGroupLifecycle = TaskGroupLifecycle.ACTIVE_SCAN,
    r1_ownership_status: str = "candidate",
    r2_lifecycle: TaskGroupLifecycle = TaskGroupLifecycle.ACTIVE_SCAN,
    r2_ownership_status: str = "candidate",
    r3_lifecycle: TaskGroupLifecycle = TaskGroupLifecycle.ACTIVE_SCAN,
    r3_ownership_status: str = "candidate",
    r4_lifecycle: TaskGroupLifecycle = TaskGroupLifecycle.ACTIVE_SCAN,
    r4_ownership_status: str = "candidate",
    deployment_revision: int = 9,
) -> OperationalExecutionSnapshot:
    base = _execution_snapshot()
    groups = tuple(
        _runtime_group(
            slot,
            lifecycle={
                1: r1_lifecycle,
                2: r2_lifecycle,
            }.get(slot, TaskGroupLifecycle.ACTIVE_SCAN),
            ownership_status={
                1: r1_ownership_status,
                2: r2_ownership_status,
                3: r3_ownership_status,
                4: r4_ownership_status,
            }.get(slot, "candidate"),
            deployment_revision=deployment_revision,
        )
        for slot in range(1, 5)
    )
    status_by_lifecycle = {
        TaskGroupLifecycle.ACTIVE_SCAN: "active",
        TaskGroupLifecycle.PASSIVE_TRACK: "passive",
    }
    regions = tuple(
        region.model_copy(
            update={
                "task_group_id": groups[index].group_instance_id,
                "status": status_by_lifecycle.get(
                    groups[index].lifecycle,
                    "planned",
                ),
            }
        )
        for index, region in enumerate(base.regions)
    )
    owner_id = (
        groups[0].group_instance_id
        if r1_ownership_status == "owner"
        else None
    )
    return base.model_copy(
        deep=True,
        update={
            "regions": regions,
            "task_groups": groups,
            "tracking_policy": TrackingPolicyConfig(),
            "tracking_control": TrackingControlState(
                mode="regional",
                tracking_owner_group_id=owner_id,
            ),
        },
    )


def _runtime_controller(**snapshot_updates: object) -> MissionController:
    controller = MissionController(
        scenario_id="S1",
        region_entry_probability_threshold=0.70,
        region_transition_confirm_cycles=2,
        dedicated_release_remaining_mileage_m=7_000.0,
    )
    snapshot = _runtime_execution_snapshot(**snapshot_updates)
    controller.advance(int(snapshot.valid_from_s), {})
    assert controller.apply_execution_snapshot(snapshot) is True
    return controller


def _runtime_group_from_snapshot(
    controller: MissionController,
    region_id: str,
) -> TaskGroupInstance:
    return next(
        group
        for group in controller.snapshot().task_groups
        if group.region_id == region_id
    )


def _runtime_replacement_snapshot(
    base: OperationalExecutionSnapshot,
    *,
    revision: int,
    shifted_slots: tuple[int, ...] = (0, 1, 2, 3),
    shift_m: float = 100.0,
) -> OperationalExecutionSnapshot:
    groups = tuple(
        _runtime_group(slot, deployment_revision=revision)
        for slot in range(1, 5)
    )
    shifted_regions = []
    for region in base.regions:
        if region.slot_index not in shifted_slots:
            shifted_regions.append(
                region.model_copy(update={"execution_revision": revision})
            )
            continue
        center = (region.center[0] + shift_m, region.center[1] + shift_m)
        geometry = tuple((x + shift_m, y + shift_m) for x, y in region.geometry)
        shifted_regions.append(
            region.model_copy(
                update={
                    "center": center,
                    "geometry": geometry,
                    "geometry_revision": region.geometry_revision + 1,
                    "execution_revision": revision,
                "task_group_id": groups[region.slot_index].group_instance_id,
                }
            )
        )
    return base.model_copy(
        deep=True,
        update={
            "execution_revision": revision,
            "base_execution_revision": base.execution_revision,
            "regions": tuple(shifted_regions),
            "task_groups": groups,
            "tracking_control": TrackingControlState(mode="regional"),
        },
    )


def test_runtime_group_entry_switches_all_three_members_to_passive() -> None:
    controller = _runtime_controller()
    group = _runtime_group_from_snapshot(controller, "target_00:task:01")

    controller.observe(
        {"region_entry_probabilities": {group.region_id: 0.70}}
    )
    assert _runtime_group_from_snapshot(
        controller, group.region_id
    ).lifecycle is TaskGroupLifecycle.ACTIVE_SCAN

    snapshot = controller.observe(
        {"region_entry_probabilities": {group.region_id: 0.81}}
    )
    tracked = _runtime_group_from_snapshot(controller, group.region_id)
    assert tracked.lifecycle is TaskGroupLifecycle.PASSIVE_TRACK
    assert tracked.sensor_mode is GroupSensorMode.PASSIVE
    assert all(
        snapshot.uuv_modes[uuv_id] is UUVMissionMode.PASSIVE_TRACK
        for uuv_id in tracked.member_uuv_ids
    )


def test_runtime_entry_confirmation_resets_on_below_threshold_and_missing_cycle() -> None:
    controller = _runtime_controller()
    region_id = "target_00:task:01"

    controller.observe({"region_entry_probabilities": {region_id: 0.69}})
    assert controller.snapshot().regions[0].entry_confirmations == 0
    controller.observe({"region_entry_probabilities": {region_id: 0.80}})
    assert controller.snapshot().regions[0].entry_confirmations == 1
    controller.observe({})
    assert controller.snapshot().regions[0].entry_confirmations == 0
    controller.observe({"region_entry_probabilities": {region_id: 0.80}})
    assert controller.snapshot().regions[0].entry_confirmations == 1
    assert _runtime_group_from_snapshot(
        controller, region_id
    ).lifecycle is TaskGroupLifecycle.ACTIVE_SCAN


def test_runtime_handoff_keeps_owner_until_all_three_current_observers_exist() -> None:
    controller = _runtime_controller(
        r1_lifecycle=TaskGroupLifecycle.PASSIVE_TRACK,
        r1_ownership_status="owner",
    )
    r1 = _runtime_group_from_snapshot(controller, "target_00:task:01")
    r2 = _runtime_group_from_snapshot(controller, "target_00:task:02")

    for probability in (0.80, 0.81):
        controller.observe(
            {
                "region_entry_probabilities": {r2.region_id: probability},
                "deployed_uuv_ids": {r2.region_id: r2.member_uuv_ids},
                "passive_observer_ids": {
                    r2.region_id: r2.member_uuv_ids[:2]
                },
            }
        )
    assert controller.snapshot().tracking_control.tracking_owner_group_id == r1.group_instance_id
    assert _runtime_group_from_snapshot(
        controller, r1.region_id
    ).lifecycle is TaskGroupLifecycle.PASSIVE_TRACK

    snapshot = controller.observe(
        {
            "passive_observer_ids": {r2.region_id: r2.member_uuv_ids},
            "deployed_uuv_ids": {r2.region_id: r2.member_uuv_ids},
        }
    )
    event_types = [event.event_type for event in snapshot.events]
    assert event_types.index("tracking_ownership_transferred") < event_types.index(
        "task_group_exiting"
    )
    assert snapshot.tracking_control.tracking_owner_group_id == r2.group_instance_id
    assert _runtime_group_from_snapshot(
        controller, r1.region_id
    ).lifecycle is TaskGroupLifecycle.EXITING


def test_runtime_handoff_waiting_preserves_owner_for_missing_successor() -> None:
    controller = _runtime_controller(
        r1_lifecycle=TaskGroupLifecycle.PASSIVE_TRACK,
        r1_ownership_status="owner",
    )
    owner_id = controller.snapshot().tracking_control.tracking_owner_group_id

    snapshot = controller.observe(
        {"target_exit_predicted": "target_00:task:01"}
    )
    assert snapshot.tracking_control.tracking_owner_group_id == owner_id
    assert _runtime_group_from_snapshot(
        controller, "target_00:task:01"
    ).lifecycle is TaskGroupLifecycle.PASSIVE_TRACK
    assert any(
        event.event_type == "handoff_waiting_for_passive_observation"
        for event in snapshot.events
    )


def test_runtime_handoff_requires_deployment_observation_for_all_successor_members() -> None:
    controller = _runtime_controller(
        r1_lifecycle=TaskGroupLifecycle.PASSIVE_TRACK,
        r1_ownership_status="owner",
        r2_lifecycle=TaskGroupLifecycle.PASSIVE_TRACK,
    )
    owner_id = controller.snapshot().tracking_control.tracking_owner_group_id
    successor = _runtime_group_from_snapshot(controller, "target_00:task:02")

    snapshot = controller.observe(
        {"passive_observer_ids": {successor.region_id: successor.member_uuv_ids}}
    )

    assert snapshot.tracking_control.tracking_owner_group_id == owner_id
    assert not any(
        event.event_type == "tracking_ownership_transferred"
        for event in snapshot.events
    )


def test_runtime_handoff_accepts_only_adjacent_successor_group() -> None:
    controller = _runtime_controller(
        r1_lifecycle=TaskGroupLifecycle.PASSIVE_TRACK,
        r1_ownership_status="owner",
        r2_lifecycle=TaskGroupLifecycle.ACTIVE_SCAN,
        r3_lifecycle=TaskGroupLifecycle.PASSIVE_TRACK,
    )
    owner_id = controller.snapshot().tracking_control.tracking_owner_group_id
    successor = _runtime_group_from_snapshot(controller, "target_00:task:03")

    snapshot = controller.observe(
        {
            "passive_observer_ids": {successor.region_id: successor.member_uuv_ids},
            "deployed_uuv_ids": {successor.region_id: successor.member_uuv_ids},
        }
    )

    assert snapshot.tracking_control.tracking_owner_group_id == owner_id
    assert not any(
        event.event_type == "tracking_ownership_transferred"
        for event in snapshot.events
    )


def test_runtime_projection_prefers_incoming_group_over_exiting_group() -> None:
    outgoing = _runtime_group(
        1,
        lifecycle=TaskGroupLifecycle.EXITING,
        deployment_revision=8,
    )
    incoming = _runtime_group(
        1,
        lifecycle=TaskGroupLifecycle.ACTIVE_SCAN,
        deployment_revision=9,
    )

    assert _select_runtime_group((outgoing, incoming), owner_group_id=None) == incoming


def test_dedicated_owner_entry_requires_current_passive_owner_and_locks_whole_group() -> None:
    controller = _runtime_controller(
        r1_lifecycle=TaskGroupLifecycle.PASSIVE_TRACK,
        r1_ownership_status="owner",
    )
    owner = _runtime_group_from_snapshot(controller, "target_00:task:01")

    assert controller.set_dedicated_owner("target_00", owner.group_instance_id) is True

    snapshot = controller.snapshot()
    assert snapshot.tracking_control.mode == "dedicated"
    assert snapshot.tracking_control.tracking_owner_group_id == owner.group_instance_id
    assert _runtime_group_from_snapshot(
        controller, owner.region_id
    ).lifecycle is TaskGroupLifecycle.DEDICATED_TRACK
    assert all(
        group.lifecycle is TaskGroupLifecycle.EXITING
        for group in snapshot.task_groups
        if group.group_instance_id != owner.group_instance_id
    )
    assert all(
        snapshot.uuv_modes[member_id] is UUVMissionMode.DEDICATED_TRACK
        for member_id in owner.member_uuv_ids
    )
    dedicated_event = next(
        event
        for event in snapshot.events
        if event.event_type == "dedicated_tracking_started"
    )
    assert dedicated_event.event_id.endswith(":d9")
    assert dedicated_event.payload["deployment_revision"] == owner.deployment_revision
    assert dedicated_event.payload["mode"] == TaskGroupLifecycle.DEDICATED_TRACK.value
    assert tuple(dedicated_event.payload["member_uuv_ids"]) == owner.member_uuv_ids


def test_dedicated_release_is_triggered_once_at_remaining_mileage_threshold() -> None:
    controller = _runtime_controller(
        r1_lifecycle=TaskGroupLifecycle.PASSIVE_TRACK,
        r1_ownership_status="owner",
    )
    owner = _runtime_group_from_snapshot(controller, "target_00:task:01")
    assert controller.set_dedicated_owner("target_00", owner.group_instance_id) is True

    first = controller.observe(
        {"mileage_m": {owner.member_uuv_ids[0]: 43_000.0}}
    )
    second = controller.observe(
        {"mileage_m": {owner.member_uuv_ids[0]: 43_000.0}}
    )

    assert first.tracking_control.mode == "dedicated"
    assert next(
        group
        for group in first.task_groups
        if group.group_instance_id == owner.group_instance_id
    ).lifecycle is TaskGroupLifecycle.DEDICATED_RELEASE_PENDING
    threshold_event = next(
        event
        for event in second.events
        if event.event_type == "dedicated_release_threshold_reached"
    )
    assert threshold_event.event_id.endswith(":d9")
    assert threshold_event.payload["deployment_revision"] == owner.deployment_revision
    assert threshold_event.payload["mode"] == TaskGroupLifecycle.DEDICATED_RELEASE_PENDING.value


def test_dedicated_owner_entry_rejects_non_owner_or_non_passive_groups() -> None:
    controller = _runtime_controller()
    candidate = _runtime_group_from_snapshot(controller, "target_00:task:02")

    assert controller.set_dedicated_owner("target_00", candidate.group_instance_id) is False
    assert controller.snapshot().tracking_control.mode == "regional"


def test_dedicated_release_restores_latest_four_groups_after_passive_observation() -> None:
    controller = _runtime_controller(
        r1_lifecycle=TaskGroupLifecycle.PASSIVE_TRACK,
        r1_ownership_status="owner",
    )
    owner = _runtime_group_from_snapshot(controller, "target_00:task:01")
    assert controller.set_dedicated_owner("target_00", owner.group_instance_id) is True

    pending = controller.observe(
        {"mileage_m": {owner.member_uuv_ids[0]: 43_000.0}}
    )
    successor_id = pending.tracking_control.pending_successor_group_id
    assert successor_id is not None
    successor = next(
        group
        for group in pending.task_groups
        if group.group_instance_id == successor_id
    )

    restored = controller.observe(
        {
            "deployed_uuv_ids": {successor.group_instance_id: successor.member_uuv_ids},
            "passive_observer_ids": {
                successor.group_instance_id: successor.member_uuv_ids
            },
        }
    )

    assert restored.tracking_control.mode == "regional"
    assert restored.tracking_control.tracking_owner_group_id == successor_id
    assert next(
        group for group in restored.task_groups if group.group_instance_id == successor_id
    ).lifecycle is TaskGroupLifecycle.PASSIVE_TRACK
    assert next(
        group for group in restored.task_groups if group.group_instance_id == owner.group_instance_id
    ).lifecycle is TaskGroupLifecycle.EXITING
    assert any(
        event.event_type == "regional_mode_restored"
        for event in restored.events
    )


def test_dedicated_refresh_does_not_release_before_mileage_threshold() -> None:
    controller = _runtime_controller(
        r1_lifecycle=TaskGroupLifecycle.PASSIVE_TRACK,
        r1_ownership_status="owner",
    )
    owner_id = controller.snapshot().tracking_control.tracking_owner_group_id
    owner = _runtime_group_from_snapshot(controller, "target_00:task:01")
    assert controller.set_dedicated_owner("target_00", owner.group_instance_id) is True
    current = controller.runtime_execution_snapshot(_execution_snapshot())
    refreshed = current.model_copy(
        deep=True,
        update={
            "execution_revision": current.execution_revision + 1,
            "base_execution_revision": current.execution_revision,
            "regions": tuple(
                region.model_copy(
                    update={"execution_revision": current.execution_revision + 1}
                )
                for region in current.regions
            ),
        },
    )

    controller.reconcile_execution_snapshot(refreshed)

    assert controller.snapshot().tracking_control.mode == "dedicated"
    assert controller.snapshot().tracking_control.tracking_owner_group_id == owner_id


def test_runtime_snapshot_reconcile_preserves_group_progress_across_refresh() -> None:
    controller = _runtime_controller(
        r1_lifecycle=TaskGroupLifecycle.PASSIVE_TRACK,
        r1_ownership_status="owner",
    )
    current = controller.snapshot()
    owner_id = current.tracking_control.tracking_owner_group_id
    candidate = _runtime_execution_snapshot(
        deployment_revision=9,
    ).model_copy(
        deep=True,
        update={
            "execution_revision": 10,
            "base_execution_revision": current.plan_revision,
            "regions": tuple(
                region.model_copy(update={"execution_revision": 10})
                for region in _runtime_execution_snapshot().regions
            ),
        },
    )

    reconciled = controller.reconcile_execution_snapshot(candidate)

    assert reconciled.plan_revision == 10
    assert reconciled.tracking_control.tracking_owner_group_id == owner_id
    assert _runtime_group_from_snapshot(
        controller, "target_00:task:01"
    ).lifecycle is TaskGroupLifecycle.PASSIVE_TRACK


def test_runtime_reconcile_reuses_groups_when_only_prediction_metadata_changes() -> None:
    controller = _runtime_controller()
    before = controller.snapshot()
    before_ids = {
        group.region_id: group.group_instance_id for group in before.task_groups
    }
    base = controller.runtime_execution_snapshot(_execution_snapshot())
    candidate = base.model_copy(
        deep=True,
        update={
            "execution_revision": base.execution_revision + 1,
            "base_execution_revision": base.execution_revision,
            "prediction_id": "prediction:metadata-only",
            "regions": tuple(
                region.model_copy(
                    update={
                        "execution_revision": base.execution_revision + 1,
                        "prediction_id": "prediction:metadata-only",
                    }
                )
                for region in base.regions
            ),
        },
    )

    reconciled = controller.reconcile_execution_snapshot(candidate)

    assert len(reconciled.task_groups) == 4
    assert {
        group.region_id: group.group_instance_id for group in reconciled.task_groups
    } == before_ids
    assert controller.replacement_states == ()


def test_four_changed_regions_keep_old_and_new_groups_visible() -> None:
    controller = _runtime_controller()
    current = controller.runtime_execution_snapshot(_execution_snapshot())
    candidate = _runtime_replacement_snapshot(
        current,
        revision=current.execution_revision + 1,
    )

    reconciled = controller.reconcile_execution_snapshot(candidate)
    waterborne = [
        group
        for group in reconciled.task_groups
        if group.lifecycle is not TaskGroupLifecycle.DISAPPEARED
    ]

    assert len(waterborne) == 8
    assert sum(
        group.lifecycle is TaskGroupLifecycle.ENTERING for group in waterborne
    ) == 4
    assert sum(
        group.lifecycle is TaskGroupLifecycle.EXITING for group in waterborne
    ) == 4
    assert len(controller.replacement_states) == 4


def test_region_replacement_completion_releases_outgoing_resources_and_identity() -> None:
    controller = _runtime_controller()
    current = controller.runtime_execution_snapshot(_execution_snapshot())
    candidate = _runtime_replacement_snapshot(
        current,
        revision=current.execution_revision + 1,
        shifted_slots=(0,),
    )
    reconciled = controller.reconcile_execution_snapshot(candidate)
    outgoing = next(
        group
        for group in reconciled.task_groups
        if group.region_id == "target_00:task:01"
        and group.lifecycle is TaskGroupLifecycle.EXITING
    )
    incoming = next(
        group
        for group in reconciled.task_groups
        if group.region_id == "target_00:task:01"
        and group.lifecycle is not TaskGroupLifecycle.EXITING
    )
    controller.observe(
        {
            "mileage_m": {
                member_id: 321.0 for member_id in outgoing.member_uuv_ids
            }
        }
    )
    before_episodes = {
        member_id: controller.snapshot().resource_episode_by_uuv.get(member_id, 0)
        for member_id in outgoing.member_uuv_ids
    }

    assert controller.complete_region_replacement(
        outgoing.region_id,
        incoming_group_id=incoming.group_instance_id,
    ) is True

    snapshot = controller.snapshot()
    assert outgoing.group_instance_id not in {
        group.group_instance_id for group in snapshot.task_groups
    }
    assert controller.replacement_states == ()
    for member_id in outgoing.member_uuv_ids:
        assert snapshot.uuv_modes[member_id] is UUVMissionMode.ONBOARD
        assert snapshot.uuv_resources[member_id].mileage_m == 0.0
        assert snapshot.uuv_resources[member_id].energy_fraction == 1.0
        assert snapshot.resource_episode_by_uuv[member_id] == before_episodes[member_id] + 1
    disappeared = next(
        event for event in snapshot.events
        if event.event_type == "task_group_disappeared"
        and event.entity_id == outgoing.group_instance_id
    )
    assert disappeared.payload["group_instance_id"] == outgoing.group_instance_id
    assert tuple(disappeared.payload["member_uuv_ids"]) == outgoing.member_uuv_ids


def test_runtime_reconcile_keeps_one_pair_and_latest_pending_region() -> None:
    controller = _runtime_controller()
    current = controller.runtime_execution_snapshot(_execution_snapshot())
    first = _runtime_replacement_snapshot(
        current,
        revision=current.execution_revision + 1,
        shift_m=100.0,
    )
    second = _runtime_replacement_snapshot(
        first,
        revision=first.execution_revision + 1,
        shift_m=200.0,
    )
    third = _runtime_replacement_snapshot(
        second,
        revision=second.execution_revision + 1,
        shift_m=300.0,
    )

    controller.reconcile_execution_snapshot(first)
    first_ids = {group.group_instance_id for group in controller.snapshot().task_groups}
    controller.reconcile_execution_snapshot(second)
    controller.reconcile_execution_snapshot(third)

    assert len(controller.snapshot().task_groups) == 8
    assert first_ids.issubset(
        {group.group_instance_id for group in controller.snapshot().task_groups}
    )
    assert all(
        state.latest_pending_region is not None
        and state.target_geometry_revision
        == state.latest_pending_region.geometry_revision
        for state in controller.replacement_states
    )
    assert all(
        state.latest_pending_region is not None
        and state.latest_pending_region.execution_revision == third.execution_revision
        for state in controller.replacement_states
    )


def test_runtime_execution_snapshot_projects_controller_groups_and_control() -> None:
    controller = _runtime_controller()
    base = controller.runtime_execution_snapshot(_execution_snapshot())

    controller.reconcile_execution_snapshot(
        _runtime_replacement_snapshot(
            base,
            revision=base.execution_revision + 1,
            shifted_slots=(1,),
        )
    )
    projected = controller.runtime_execution_snapshot(
        _runtime_replacement_snapshot(
            base,
            revision=base.execution_revision + 1,
            shifted_slots=(1,),
        )
    )

    assert projected.execution_revision == controller.execution_revision
    assert projected.task_groups == controller.snapshot().task_groups
    assert projected.tracking_control == controller.snapshot().tracking_control


def plan(*, revision: int = 1, include_successor: bool = False) -> ExecutableMissionPlan:
    assignments = [
        RegionMissionState(
            region_id="R1",
            target_id="T1",
            active_scan_uuv_ids=("U1",),
            passive_track_uuv_ids=("U2",),
            reserve_uuv_ids=("U3",),
            handoff_to="R2" if include_successor else None,
            plan_revision=revision,
        )
    ]
    batches = [
        UUVMissionBatch(
            carrier_id="carrier_01",
            candidate_id="R1",
            uuv_ids=("U1", "U2"),
            active_scan_uuv_ids=("U1",),
            passive_track_uuv_ids=("U2",),
            deployment_point=(0.0, 100.0),
            recovery_point=(100.0, 100.0),
            entry_s=10,
            exit_s=100,
        )
    ]
    if include_successor:
        assignments.append(
            RegionMissionState(
                region_id="R2",
                target_id="T1",
                active_scan_uuv_ids=("U4",),
                passive_track_uuv_ids=("U5",),
                handoff_from="R1",
                plan_revision=revision,
            )
        )
        batches.append(
            UUVMissionBatch(
                carrier_id="carrier_01",
                candidate_id="R2",
                uuv_ids=("U4", "U5"),
                active_scan_uuv_ids=("U4",),
                passive_track_uuv_ids=("U5",),
                deployment_point=(200.0, 100.0),
                recovery_point=(300.0, 100.0),
                entry_s=110,
                exit_s=200,
            )
        )
    return ExecutableMissionPlan(
        revision=revision,
        uuv_batches_by_carrier={"carrier_01": tuple(batches)},
        region_assignments=tuple(assignments),
        carrier_missions={
            "carrier_01": CarrierMissionModel(
                carrier_id="carrier_01",
                home_battle_group_id="home",
                ready_uuv_ids=("U1", "U2", "U3", "U4", "U5"),
            )
        },
    )


def test_controller_applies_only_newer_verified_plan_and_returns_immutable_snapshot() -> None:
    controller = MissionController(scenario_id="S1")

    assert controller.apply_verified_plan(plan(revision=2)) is True
    before = controller.snapshot()
    assert before.plan_revision == 2
    assert before.regions[0].lifecycle is RegionLifecycle.PLANNED
    assert before.uuv_modes["U1"] is UUVMissionMode.TRANSIT_TO_REGION
    assert controller.apply_verified_plan(plan(revision=2)) is False
    assert controller.apply_verified_plan(plan(revision=1)) is False
    assert controller.snapshot() == before


def test_entry_probability_requires_two_confirmations_before_passive_track() -> None:
    controller = MissionController(
        scenario_id="S1",
        region_entry_probability_threshold=0.70,
        region_transition_confirm_cycles=2,
    )
    controller.apply_verified_plan(plan())

    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})
    deployed = controller.snapshot()
    assert deployed.regions[0].lifecycle is RegionLifecycle.ACTIVE_SCAN
    assert deployed.uuv_modes["U1"] is UUVMissionMode.ACTIVE_SCAN
    assert deployed.uuv_modes["U2"] is UUVMissionMode.PASSIVE_TRACK
    controller.advance(20, {"entry_probability": {"R1": 0.8}})
    assert controller.snapshot().regions[0].lifecycle is RegionLifecycle.ACTIVE_SCAN
    controller.advance(30, {"entry_probability": {"R1": 0.8}})

    snapshot = controller.snapshot()
    assert snapshot.regions[0].lifecycle is RegionLifecycle.PASSIVE_TRACK
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.PASSIVE_TRACK
    assert [event.event_type for event in snapshot.events] == ["target_entered_region"]


def test_terminal_region_keeps_tracking_until_a_replacement_plan_arrives() -> None:
    controller = MissionController(
        scenario_id="S1",
        region_entry_probability_threshold=0.70,
        region_transition_confirm_cycles=2,
    )
    controller.apply_verified_plan(plan())
    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})
    controller.advance(20, {"entry_probability": {"R1": 0.8}})
    controller.advance(30, {"entry_probability": {"R1": 0.8}})

    snapshot = controller.advance(40, {"target_exit_predicted": "R1"})

    assert snapshot.regions[0].lifecycle is RegionLifecycle.PASSIVE_TRACK
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.PASSIVE_TRACK
    assert snapshot.uuv_modes["U2"] is UUVMissionMode.PASSIVE_TRACK
    assert snapshot.carrier_missions["carrier_01"].recoverable_uuv_ids == ()


def test_active_scan_group_stays_waterborne_while_successor_handoff_is_pending() -> None:
    controller = MissionController(scenario_id="S1")
    controller.apply_verified_plan(plan(include_successor=True))
    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})

    snapshot = controller.advance(20, {"target_exit_predicted": "R1"})

    region = next(item for item in snapshot.regions if item.region_id == "R1")
    assert region.lifecycle is RegionLifecycle.ACTIVE_SCAN
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.ACTIVE_SCAN
    assert snapshot.uuv_modes["U2"] is UUVMissionMode.PASSIVE_TRACK


def test_active_scan_requires_typed_handoff_evidence_before_passive_track() -> None:
    controller = MissionController(
        scenario_id="S1",
        region_transition_confirm_cycles=1,
    )
    controller.apply_verified_plan(plan(include_successor=True))
    controller.advance(
        10,
        {
            "deployed_uuv_ids": {
                "R1": ("U1", "U2"),
                "R2": ("U4", "U5"),
            }
        },
    )
    controller.advance(
        20,
        {
            "entry_probability": {"R1": 0.9, "R2": 0.9},
            "target_exit_predicted": "R1",
        },
    )

    snapshot = controller.advance(30, {"target_exit_predicted": "R1"})

    regions = {region.region_id: region for region in snapshot.regions}
    assert regions["R1"].lifecycle is RegionLifecycle.ACTIVE_SCAN
    assert regions["R2"].lifecycle is RegionLifecycle.PASSIVE_TRACK
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.ACTIVE_SCAN
    assert snapshot.uuv_modes["U2"] is UUVMissionMode.PASSIVE_TRACK


def test_active_scan_handoff_completes_with_current_cycle_evidence() -> None:
    controller = MissionController(
        scenario_id="S1",
        group_min_size=2,
        region_transition_confirm_cycles=1,
    )
    controller.apply_verified_plan(plan(include_successor=True))
    controller.advance(
        10,
        {
            "deployed_uuv_ids": {
                "R1": ("U1", "U2"),
                "R2": ("U4", "U5"),
            }
        },
    )
    controller.advance(20, {"entry_probability": {"R2": 0.9}})

    snapshot = controller.advance(
        30,
        {
            "target_exit_predicted": "R1",
            "handoff_evidence": {"R1": _typed_handoff_evidence()},
        },
    )

    regions = {region.region_id: region for region in snapshot.regions}
    assert regions["R1"].lifecycle is RegionLifecycle.TRACKING_COMPLETED
    assert regions["R2"].lifecycle is RegionLifecycle.PASSIVE_TRACK
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.uuv_modes["U2"] is UUVMissionMode.RETURN_REQUIRED


def test_missing_or_invalid_entry_probability_resets_confirmation() -> None:
    controller = MissionController(
        scenario_id="S1",
        region_entry_probability_threshold=0.70,
        region_transition_confirm_cycles=2,
    )
    controller.apply_verified_plan(plan())
    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})

    controller.advance(20, {"entry_probability": {"R1": 0.8}})
    assert controller.snapshot().regions[0].entry_confirmations == 1
    controller.advance(30, {})
    assert controller.snapshot().regions[0].entry_confirmations == 0
    assert controller.snapshot().regions[0].lifecycle is RegionLifecycle.ACTIVE_SCAN

    controller.advance(40, {"entry_probability": {"R1": float("nan")}})
    assert controller.snapshot().regions[0].entry_confirmations == 0
    controller.advance(50, {"entry_probability": {"R1": 0.8}})
    snapshot = controller.snapshot()
    assert snapshot.regions[0].entry_confirmations == 1
    assert snapshot.regions[0].lifecycle is RegionLifecycle.ACTIVE_SCAN


def test_handoff_activates_successor_before_predecessor_closes() -> None:
    controller = _prepare_handoff_controller()
    controller.advance(30, {"handoff_evidence": {"R1": _typed_handoff_evidence()}})

    regions = {region.region_id: region for region in controller.snapshot().regions}
    assert regions["R2"].lifecycle is RegionLifecycle.PASSIVE_TRACK
    assert regions["R1"].lifecycle is RegionLifecycle.TRACKING_COMPLETED
    assert controller.snapshot().uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert controller.snapshot().uuv_modes["U2"] is UUVMissionMode.RETURN_REQUIRED
    assert controller.snapshot().events[-1].event_type == "handoff_completed"


def _typed_handoff_evidence(
    *,
    plan_revision: int = 1,
    observation_cycle_s: int = 30,
    successor_region_id: str = "R2",
    deployed_uuv_ids: tuple[str, ...] = ("U4", "U5"),
    healthy_uuv_ids: tuple[str, ...] = ("U4", "U5"),
    passive_mode_uuv_ids: tuple[str, ...] = ("U4", "U5"),
    accepted_observations: tuple[AcceptedHandoffObservation, ...] = (
        AcceptedHandoffObservation(
            observation_id="obs-u4",
            observer_uuv_id="U4",
            observed_at_s=30,
        ),
        AcceptedHandoffObservation(
            observation_id="obs-u5",
            observer_uuv_id="U5",
            observed_at_s=30,
        ),
    ),
    hard_guard_reasons: tuple[str, ...] = (),
    blocked_reason: str | None = None,
) -> HandoffEvidence:
    return HandoffEvidence(
        predecessor_region_id="R1",
        successor_region_id=successor_region_id,
        plan_revision=plan_revision,
        observation_cycle_s=observation_cycle_s,
        required_uuv_ids=("U4", "U5"),
        deployed_uuv_ids=deployed_uuv_ids,
        healthy_uuv_ids=healthy_uuv_ids,
        passive_mode_uuv_ids=passive_mode_uuv_ids,
        accepted_observations=accepted_observations,
        hard_guard_reasons=hard_guard_reasons,
        blocked_reason=blocked_reason,
    )


def _prepare_handoff_controller() -> MissionController:
    controller = MissionController(
        scenario_id="S1",
        group_min_size=2,
        region_transition_confirm_cycles=1,
    )
    controller.apply_verified_plan(plan(include_successor=True))
    controller.advance(
        10,
        {
            "deployed_uuv_ids": {"R1": ("U1", "U2"), "R2": ("U4", "U5")},
            "entry_probability": {"R1": 0.9, "R2": 0.9},
        },
    )
    controller.advance(20, {"target_exit_predicted": "R1"})
    assert controller.snapshot().regions[0].lifecycle is RegionLifecycle.HANDOFF_PENDING
    return controller


def test_typed_handoff_evidence_completes_once_with_exact_source_ids() -> None:
    controller = _prepare_handoff_controller()

    snapshot = controller.advance(
        30,
        {"handoff_evidence": {"R1": _typed_handoff_evidence()}},
    )

    regions = {region.region_id: region for region in snapshot.regions}
    assert regions["R2"].lifecycle is RegionLifecycle.PASSIVE_TRACK
    assert regions["R1"].lifecycle is RegionLifecycle.TRACKING_COMPLETED
    event = next(event for event in snapshot.events if event.event_type == "handoff_completed")
    assert event.payload["source_observation_ids"] == ("obs-u4", "obs-u5")
    assert event.payload["plan_revision"] == 1


def test_handoff_accepts_successor_already_pending_its_next_handoff() -> None:
    controller = _prepare_handoff_controller()
    controller._regions["R2"] = controller._regions["R2"].model_copy(
        update={"lifecycle": RegionLifecycle.HANDOFF_PENDING}
    )

    snapshot = controller.advance(
        30,
        {"handoff_evidence": {"R1": _typed_handoff_evidence()}},
    )

    regions = {region.region_id: region for region in snapshot.regions}
    assert regions["R1"].lifecycle is RegionLifecycle.TRACKING_COMPLETED
    assert regions["R2"].lifecycle is RegionLifecycle.HANDOFF_PENDING
    assert any(event.event_type == "handoff_completed" for event in snapshot.events)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("plan_revision", 2),
        ("observation_cycle_s", 20),
        ("successor_region_id", "R9"),
        ("deployed_uuv_ids", ("U4",)),
        ("healthy_uuv_ids", ("U4",)),
        ("passive_mode_uuv_ids", ("U4",)),
        ("accepted_observations", ()),
        ("hard_guard_reasons", ("covariance_unbounded",)),
    ),
)
def test_incomplete_typed_handoff_evidence_keeps_pending(
    field: str,
    value: object,
) -> None:
    controller = _prepare_handoff_controller()
    if field == "observation_cycle_s":
        evidence = _typed_handoff_evidence().model_copy(update={field: value})
    elif field == "passive_mode_uuv_ids":
        evidence = _typed_handoff_evidence(
            passive_mode_uuv_ids=("U4",),
            accepted_observations=(
                AcceptedHandoffObservation(
                    observation_id="obs-u4",
                    observer_uuv_id="U4",
                    observed_at_s=30,
                ),
            ),
        )
    else:
        evidence = _typed_handoff_evidence(**{field: value})

    snapshot = controller.advance(30, {"handoff_evidence": {"R1": evidence}})

    regions = {region.region_id: region for region in snapshot.regions}
    assert regions["R1"].lifecycle is RegionLifecycle.HANDOFF_PENDING
    assert regions["R2"].lifecycle is RegionLifecycle.PASSIVE_TRACK
    assert not any(event.event_type == "handoff_completed" for event in snapshot.events)


def test_blocked_typed_handoff_degrades_once_with_source_ids() -> None:
    controller = _prepare_handoff_controller()
    evidence = _typed_handoff_evidence(blocked_reason="successor_unavailable")

    snapshot = controller.advance(30, {"handoff_evidence": {"R1": evidence}})
    snapshot = controller.advance(40, {"handoff_evidence": {"R1": evidence}})

    region = next(region for region in snapshot.regions if region.region_id == "R1")
    assert region.lifecycle is RegionLifecycle.DEGRADED
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.uuv_modes["U2"] is UUVMissionMode.RETURN_REQUIRED
    blocked = [event for event in snapshot.events if event.event_type == "handoff_blocked"]
    assert len(blocked) == 1
    assert blocked[0].payload["plan_revision"] == 1
    assert blocked[0].payload["source_observation_ids"] == ("obs-u4", "obs-u5")
    assert not any(event.event_type == "handoff_completed" for event in snapshot.events)


def test_handoff_does_not_reopen_a_degraded_successor() -> None:
    controller = _prepare_handoff_controller()

    snapshot = controller.advance(
        30,
        {
            "failed_uuv_ids": ("U4",),
            "handoff_evidence": {"R1": _typed_handoff_evidence()},
        },
    )

    regions = {region.region_id: region for region in snapshot.regions}
    assert regions["R1"].lifecycle is RegionLifecycle.HANDOFF_PENDING
    assert regions["R2"].lifecycle is RegionLifecycle.DEGRADED
    assert not any(event.event_type == "handoff_completed" for event in snapshot.events)


def test_mileage_exhaustion_is_idempotent_and_rotates_at_the_boundary() -> None:
    controller = MissionController(scenario_id="S1", max_uuv_mileage_m=1_000.0)
    controller.apply_verified_plan(plan())

    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})
    controller.advance(20, {"mileage_m": {"U1": 1_001.0}})
    controller.advance(30, {"mileage_m": {"U1": 1_001.0}})

    snapshot = controller.snapshot()
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.carrier_missions["carrier_01"].recoverable_uuv_ids == ()
    assert [event.event_type for event in snapshot.events].count("uuv_range_exhausted") == 1


def test_resource_warning_is_observed_before_hard_exhaustion() -> None:
    controller = MissionController(
        scenario_id="S1",
        max_uuv_mileage_m=1_000.0,
        resource_warning_mileage_fraction=0.20,
    )
    controller.apply_verified_plan(plan())

    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})
    controller.advance(20, {"mileage_m": {"U1": 201.0}})
    controller.advance(30, {"mileage_m": {"U1": 201.0}})

    snapshot = controller.snapshot()
    warnings = [
        event for event in snapshot.events if event.event_type == "endurance_threshold_crossed"
    ]
    assert len(warnings) == 1
    assert warnings[0].payload["warning_mileage_m"] == 200.0
    assert snapshot.uuv_modes["U1"] is not UUVMissionMode.RETURN_REQUIRED


def test_dedicated_group_returns_to_region_and_rejoins_normal_scan() -> None:
    controller = MissionController(scenario_id="S1", max_uuv_mileage_m=1_000.0)
    controller.apply_verified_plan(plan())

    assert controller.set_dedicated_group("T1", ("U1",)) is True
    deployed = controller.advance(
        10,
        {"deployed_uuv_ids": {"R1": ("U1", "U2")}},
    )
    assert deployed.uuv_modes["U1"] is UUVMissionMode.DEDICATED_TRACK
    assert deployed.dedicated_target_by_uuv["U1"] == "T1"

    exhausted = controller.advance(20, {"mileage_m": {"U1": 1_001.0}})
    assert exhausted.uuv_modes["U1"] is UUVMissionMode.RETURN_TO_REGION
    assert exhausted.carrier_missions["carrier_01"].recoverable_uuv_ids == ()

    returned = controller.advance(
        30,
        {
            "returned_to_region_uuv_ids": ("U1",),
            "mileage_m": {"U1": 1_001.0},
        },
    )
    assert returned.uuv_modes["U1"] is UUVMissionMode.ACTIVE_SCAN
    assert "U1" not in returned.dedicated_target_by_uuv
    assert returned.uuv_resources["U1"].mileage_m == 0.0
    assert any(event.event_type == "dedicated_mode_released" for event in returned.events)


def test_dedicated_group_exits_other_deployed_target_regions() -> None:
    controller = MissionController(scenario_id="S1")
    controller.apply_verified_plan(plan(include_successor=True))
    controller.advance(
        10,
        {"deployed_uuv_ids": {"R1": ("U1", "U2"), "R2": ("U4", "U5")}},
    )

    assert controller.set_dedicated_group("T1", ("U1", "U2")) is True

    snapshot = controller.snapshot()
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.DEDICATED_TRACK
    assert snapshot.uuv_modes["U2"] is UUVMissionMode.DEDICATED_TRACK
    assert snapshot.uuv_modes["U4"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.uuv_modes["U5"] is UUVMissionMode.RETURN_REQUIRED
    assert any(
        event.event_type == "dedicated_group_regional_exit_requested"
        and event.entity_id == "T1"
        and event.payload["uuv_ids"] == ("U4", "U5")
        for event in snapshot.events
    )


def test_dedicated_group_exits_before_range_exhaustion_to_preserve_return_reserve() -> None:
    controller = MissionController(
        scenario_id="S1",
        max_uuv_mileage_m=1_000.0,
        resource_warning_mileage_fraction=0.20,
    )
    controller.apply_verified_plan(plan())
    assert controller.set_dedicated_group("T1", ("U1",)) is True

    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})
    still_tracking = controller.advance(20, {"mileage_m": {"U1": 201.0}})
    assert still_tracking.uuv_modes["U1"] is UUVMissionMode.DEDICATED_TRACK

    returning = controller.advance(30, {"mileage_m": {"U1": 801.0}})
    assert returning.uuv_modes["U1"] is UUVMissionMode.RETURN_TO_REGION
    assert any(
        event.event_type == "uuv_dedicated_return_to_region"
        and event.payload["reason"] == "dedicated_range_reserve"
        for event in returning.events
    )


def test_dedicated_group_failure_releases_the_mode_for_regional_replan() -> None:
    controller = MissionController(scenario_id="S1")
    controller.apply_verified_plan(plan())
    assert controller.set_dedicated_group("T1", ("U1", "U2")) is True
    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})

    failed = controller.advance(20, {"failed_uuv_ids": ("U1",)})

    assert failed.uuv_modes["U1"] is UUVMissionMode.FAILED
    assert failed.uuv_modes["U2"] is UUVMissionMode.PASSIVE_TRACK
    assert failed.dedicated_target_by_uuv == {}
    assert any(
        event.event_type == "dedicated_mode_released"
        and event.entity_id == "U1"
        and event.payload == {"target_id": "T1", "reason": "member_failure"}
        for event in failed.events
    )


def test_low_range_regional_uuv_is_replaced_without_changing_working_count() -> None:
    controller = MissionController(
        scenario_id="S1",
        max_uuv_mileage_m=1_000.0,
        resource_warning_mileage_fraction=0.20,
    )
    controller.apply_verified_plan(plan())
    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})

    snapshot = controller.advance(20, {"mileage_m": {"U1": 801.0}})

    region = snapshot.regions[0]
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.uuv_modes["U3"] is UUVMissionMode.ACTIVE_SCAN
    assert snapshot.carrier_missions["carrier_01"].recoverable_uuv_ids == ()
    assert region.active_scan_uuv_ids == ("U3",)
    assert region.passive_track_uuv_ids == ("U2",)
    assert region.reserve_uuv_ids == ()
    assert len(region.active_scan_uuv_ids + region.passive_track_uuv_ids) == 2
    replacement = next(
        event for event in snapshot.events if event.event_type == "uuv_boundary_replacement"
    )
    assert replacement.payload == {
        "outgoing_uuv_id": "U1",
        "replacement_uuv_id": "U3",
        "region_id": "R1",
        "role": "active_scan",
        "reason": "uuv_range_reserve",
    }

    cooling = controller.advance(30, {"boundary_exited_uuv_ids": ("U1",)})
    assert cooling.uuv_modes["U1"] is UUVMissionMode.RECOVERING
    assert cooling.uuv_resources["U1"].deployment_state == "unavailable"

    still_cooling = controller.advance(149, {})
    assert still_cooling.uuv_modes["U1"] is UUVMissionMode.RECOVERING

    refueled = controller.advance(150, {})
    assert refueled.uuv_modes["U1"] is UUVMissionMode.ONBOARD
    assert refueled.uuv_resources["U1"].mileage_m == 0.0
    assert refueled.uuv_resources["U1"].energy_fraction == 1.0
    assert any(
        event.event_type == "uuv_refueled_active" and event.entity_id == "U1"
        for event in refueled.events
    )


def test_unavailable_group_reenters_from_boundary_when_no_replacement_exists() -> None:
    no_reserve = plan().model_copy(
        update={
            "region_assignments": (
                plan().region_assignments[0].model_copy(update={"reserve_uuv_ids": ()}),
            ),
            "carrier_missions": {},
        }
    )
    controller = MissionController(
        scenario_id="S1",
        max_uuv_mileage_m=1_000.0,
        resource_warning_mileage_fraction=0.20,
        refuel_cooldown_s=120,
    )
    controller.apply_verified_plan(no_reserve)
    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})
    returning = controller.advance(
        20,
        {"mileage_m": {"U1": 801.0, "U2": 801.0}},
    )
    assert returning.uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert returning.uuv_modes["U2"] is UUVMissionMode.RETURN_REQUIRED

    controller.advance(30, {"boundary_exited_uuv_ids": ("U1", "U2")})
    reentering = controller.advance(150, {})

    assert reentering.uuv_modes["U1"] is UUVMissionMode.ACTIVE_SCAN
    assert reentering.uuv_modes["U2"] is UUVMissionMode.PASSIVE_TRACK
    assert reentering.uuv_resources["U1"].deployment_state == "ACTIVE_SCAN"
    assert reentering.uuv_resources["U2"].deployment_state == "PASSIVE_TRACK"


def test_mileage_exhaustion_is_recorded_after_task_rotation_already_requested_recovery() -> None:
    controller = MissionController(scenario_id="S1", max_uuv_mileage_m=1_000.0)
    controller.apply_verified_plan(plan())

    controller.advance(10, {"recovery_requested_uuv_ids": ("U1",)})
    snapshot = controller.advance(20, {"mileage_m": {"U1": 1_001.0}})

    assert snapshot.uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert [event.event_type for event in snapshot.events].count("uuv_range_exhausted") == 1


def test_failed_uuv_is_removed_from_active_mode_without_fabricated_replacement() -> None:
    controller = MissionController(scenario_id="S1")
    controller.apply_verified_plan(plan())

    controller.advance(10, {"failed_uuv_ids": ("U1",)})

    snapshot = controller.snapshot()
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.FAILED
    assert snapshot.regions[0].lifecycle is RegionLifecycle.DEGRADED
    assert "U99" not in snapshot.uuv_modes
    assert snapshot.events[-1].event_type == "uuv_failed"


def test_recovered_uuv_returns_to_ready_pool_after_health_check() -> None:
    controller = MissionController(scenario_id="S1", max_uuv_mileage_m=1_000.0)
    controller.apply_verified_plan(plan())

    controller.advance(
        10,
        {
            "deployed_uuv_ids": {"R1": ("U1", "U2")},
            "mileage_m": {"U1": 1_001.0},
            "energy_fraction": {"U1": 0.8},
        },
    )
    snapshot = controller.advance(
        20,
        {
            "recovered_uuv_ids": ("U1",),
            "health_check_passed": {"U1": True},
        },
    )

    assert snapshot.uuv_modes["U1"] is UUVMissionMode.ONBOARD
    assert snapshot.carrier_missions["carrier_01"].ready_uuv_ids == (
        "U1",
        "U3",
        "U4",
        "U5",
    )
    assert any(
        event.event_type == "carrier_recovery_completed" and event.entity_id == "U1"
        for event in snapshot.events
    )
    assert snapshot.uuv_resources["U1"].mileage_m == 0.0
    assert snapshot.uuv_resources["U1"].energy_fraction == 1.0


def test_handoff_completion_immediately_queues_predecessor_uuvs_for_recovery() -> None:
    controller = _prepare_handoff_controller()
    snapshot = controller.advance(
        30,
        {"handoff_evidence": {"R1": _typed_handoff_evidence()}},
    )

    assert snapshot.uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.uuv_modes["U2"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.carrier_missions["carrier_01"].recoverable_uuv_ids == ("U1", "U2")


def test_completed_region_rotates_uuvs_at_resource_warning_without_erasing_assignment() -> None:
    controller = MissionController(
        scenario_id="S1",
        group_min_size=2,
        region_transition_confirm_cycles=1,
        max_uuv_mileage_m=1_000.0,
        resource_warning_mileage_fraction=0.20,
    )
    controller.apply_verified_plan(plan(include_successor=True))
    controller.advance(
        10,
        {
            "deployed_uuv_ids": {"R1": ("U1", "U2"), "R2": ("U4", "U5")},
            "entry_probability": {"R1": 0.9, "R2": 0.9},
        },
    )
    controller.advance(20, {"target_exit_predicted": "R1"})
    controller.advance(30, {"handoff_evidence": {"R1": _typed_handoff_evidence()}})

    snapshot = controller.advance(40, {"mileage_m": {"U1": 201.0, "U2": 201.0}})

    assert snapshot.uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.uuv_modes["U2"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.carrier_missions["carrier_01"].recoverable_uuv_ids == ("U1", "U2")
    assert any(
        event.event_type == "endurance_threshold_crossed"
        and event.entity_id == "U1"
        for event in snapshot.events
    )
    completed = next(region for region in snapshot.regions if region.region_id == "R1")
    assert completed.lifecycle is RegionLifecycle.TRACKING_COMPLETED
    assert completed.active_scan_uuv_ids == ("U1",)
    assert completed.passive_track_uuv_ids == ("U2",)


def test_missing_resource_fields_preserve_last_known_values() -> None:
    controller = MissionController(scenario_id="S1")
    controller.apply_verified_plan(plan())

    controller.advance(
        10,
        {
            "mileage_m": {"U1": 123.0},
            "energy_fraction": {"U1": 0.42},
            "uuv_health": {"U1": True},
            "uuv_capability_active": {"U1": True},
            "deployment_state": {"U1": "deployed"},
        },
    )
    snapshot = controller.advance(20, {})

    resource = snapshot.uuv_resources["U1"]
    assert resource.mileage_m == 123.0
    assert resource.energy_fraction == 0.42
    assert resource.healthy is True
    assert resource.capability_active is True
    assert resource.deployment_state == "deployed"


def test_distinct_external_dispatch_events_are_not_deduplicated_by_carrier() -> None:
    controller = MissionController(scenario_id="S1")
    controller.apply_verified_plan(plan())

    controller.advance(
        10,
        {
            "carrier_dispatch_completed": {
                "entity_id": "carrier_01",
                "event_id": "dispatch:R1:10",
                "candidate_id": "R1",
            }
        },
    )
    snapshot = controller.advance(
        20,
        {
            "carrier_dispatch_completed": {
                "entity_id": "carrier_01",
                "event_id": "dispatch:R2:20",
                "candidate_id": "R2",
            }
        },
    )

    assert [
        event.event_type
        for event in snapshot.events
        if event.event_type == "carrier_dispatch_completed"
    ] == ["carrier_dispatch_completed", "carrier_dispatch_completed"]


def test_mission_event_history_and_dedupe_index_are_bounded() -> None:
    controller = MissionController(scenario_id="S1", event_history_limit=3)

    for index in range(8):
        controller._sim_time_s = index
        controller._emit("external_signal", f"uuv-{index}")

    assert len(controller.events) == 3
    assert len(controller._emitted) == 3


def test_health_and_capability_observations_remove_uuv_from_execution() -> None:
    controller = MissionController(scenario_id="S1")
    controller.apply_verified_plan(plan())

    failed = controller.advance(10, {"uuv_health": {"U1": False}})
    assert failed.uuv_modes["U1"] is UUVMissionMode.FAILED
    assert failed.events[-1].event_type == "uuv_failed"

    controller = MissionController(scenario_id="S1")
    controller.apply_verified_plan(plan())
    degraded = controller.advance(10, {"uuv_capability_active": {"U1": False}})
    assert degraded.uuv_modes["U1"] is UUVMissionMode.FAILED
    assert degraded.events[-1].event_type == "uuv_capability_lost"


def test_new_plan_rotates_removed_active_uuvs_into_carrier_recovery() -> None:
    controller = MissionController(scenario_id="S1")
    controller.apply_verified_plan(plan())
    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})

    replacement = plan(revision=2).model_copy(
        update={
            "uuv_batches_by_carrier": {},
            "region_assignments": (),
            "carrier_missions": {
                "carrier_01": CarrierMissionModel(
                    carrier_id="carrier_01",
                    home_battle_group_id="home",
                    ready_uuv_ids=("U3", "U4", "U5"),
                )
            },
        }
    )
    assert controller.apply_verified_plan(replacement) is True

    snapshot = controller.snapshot()
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.uuv_modes["U2"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.carrier_missions["carrier_01"].recoverable_uuv_ids == ("U1", "U2")


def test_new_plan_preserves_return_required_for_still_assigned_uuv() -> None:
    controller = MissionController(scenario_id="S1", max_uuv_mileage_m=1_000.0)
    assert controller.apply_verified_plan(plan()) is True
    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})
    controller.advance(20, {"mileage_m": {"U1": 1_001.0}})

    assert controller.snapshot().uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert controller.apply_verified_plan(plan(revision=2)) is True

    assert controller.snapshot().uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED


def test_carrierless_plan_refresh_preserves_carrier_recovery_metadata() -> None:
    controller = MissionController(scenario_id="S1")
    initial = plan(include_successor=True).model_copy(
        update={
            "uuv_batches_by_carrier": {},
            "task_groups": (
                TaskGroupAssignment(
                    task_group_id="TG-R1",
                    target_id="T1",
                    region_id="T1:task:R1",
                    execution_revision=1,
                    member_uuv_ids=("U1", "U2"),
                    active_verifier_uuv_id="U1",
                    passive_tracker_uuv_id="U2",
                    evidence_ids=("evidence-R1",),
                ),
            ),
        }
    )
    assert controller.apply_verified_plan(initial) is True
    previous_carriers = controller.snapshot().carrier_missions

    carrierless = initial.model_copy(
        deep=True,
        update={"revision": 2, "carrier_missions": {}},
    )

    assert controller.apply_verified_plan(carrierless) is True
    snapshot = controller.snapshot()
    assert snapshot.carrier_missions == previous_carriers
    assert controller._uuv_carrier_ids["U1"] == "carrier_01"


def test_equivalent_new_plan_preserves_region_lifecycle_progress() -> None:
    controller = _prepare_handoff_controller()
    controller.advance(30, {"handoff_evidence": {"R1": _typed_handoff_evidence()}})

    rolling_plan = plan(include_successor=True, revision=2)
    rolling_plan = rolling_plan.model_copy(
        update={
            "region_assignments": tuple(
                region.model_copy(
                    update={
                        "task_group_id": f"rolling:{region.region_id}",
                        "carrier_task_id": f"rolling-task:{region.region_id}",
                    }
                )
                for region in rolling_plan.region_assignments
            )
        }
    )

    assert controller.apply_verified_plan(rolling_plan) is True

    regions = {region.region_id: region for region in controller.snapshot().regions}
    assert regions["R1"].lifecycle is RegionLifecycle.TRACKING_COMPLETED
    assert regions["R2"].lifecycle is RegionLifecycle.PASSIVE_TRACK


def test_rolling_refresh_does_not_preserve_pending_handoff_after_topology_change() -> None:
    controller = _prepare_handoff_controller()
    base_plan = plan(include_successor=True, revision=2)
    rolling_plan = base_plan.model_copy(
        update={
            "region_assignments": tuple(
                region.model_copy(update={"handoff_to": None})
                if region.region_id == "R1"
                else region
                for region in base_plan.region_assignments
            )
        }
    )

    assert controller.apply_verified_plan(rolling_plan) is True
    region = next(
        region for region in controller.snapshot().regions if region.region_id == "R1"
    )
    assert region.lifecycle is RegionLifecycle.PLANNED


def test_recovery_requires_health_check_and_completes_after_all_uuvs_return() -> None:
    controller = _prepare_handoff_controller()
    controller.advance(
        30,
        {"handoff_evidence": {"R1": _typed_handoff_evidence()}},
    )
    controller.advance(35, {"recovery_requested_uuv_ids": ("U1", "U2")})
    controller.advance(40, {"recovering_uuv_ids": ("U1", "U2")})

    pending = controller.advance(50, {"recovered_uuv_ids": ("U1",)})
    assert pending.regions[0].lifecycle is RegionLifecycle.CARRIER_RECOVERY
    assert pending.uuv_modes["U1"] is UUVMissionMode.RECOVERING

    still_recovering = controller.advance(
        60,
        {
            "recovered_uuv_ids": ("U1",),
            "health_check_passed": {"U1": True},
        },
    )
    assert still_recovering.regions[0].lifecycle is RegionLifecycle.CARRIER_RECOVERY

    recovered = controller.advance(
        70,
        {
            "recovered_uuv_ids": ("U2",),
            "health_check_passed": {"U2": True},
        },
    )
    assert recovered.regions[0].lifecycle is RegionLifecycle.RECOVERED


def test_recovered_region_lifecycle_survives_verified_plan_refresh() -> None:
    controller = _prepare_handoff_controller()
    controller.advance(
        30,
        {"handoff_evidence": {"R1": _typed_handoff_evidence()}},
    )
    controller.advance(35, {"recovery_requested_uuv_ids": ("U1", "U2")})
    controller.advance(40, {"recovering_uuv_ids": ("U1", "U2")})
    recovered = controller.advance(
        50,
        {
            "recovered_uuv_ids": ("U1", "U2"),
            "health_check_passed": {"U1": True, "U2": True},
        },
    )
    assert recovered.regions[0].lifecycle is RegionLifecycle.RECOVERED

    assert controller.apply_verified_plan(plan(include_successor=True, revision=2)) is True
    snapshot = controller.snapshot()
    regions = {region.region_id: region for region in snapshot.regions}
    assert regions["R1"].lifecycle is RegionLifecycle.RECOVERED
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.ONBOARD
    assert snapshot.uuv_modes["U2"] is UUVMissionMode.ONBOARD


def test_partial_recovery_acknowledgement_survives_verified_plan_refresh() -> None:
    controller = _prepare_handoff_controller()
    controller.advance(
        30,
        {"handoff_evidence": {"R1": _typed_handoff_evidence()}},
    )
    controller.advance(35, {"recovery_requested_uuv_ids": ("U1", "U2")})
    controller.advance(40, {"recovering_uuv_ids": ("U1", "U2")})
    pending = controller.advance(
        50,
        {
            "recovered_uuv_ids": ("U1",),
            "health_check_passed": {"U1": True},
        },
    )
    assert pending.regions[0].lifecycle is RegionLifecycle.CARRIER_RECOVERY
    assert controller._recovered_uuv_ids_by_region["R1"] == {"U1"}
    assert pending.uuv_modes["U1"] is UUVMissionMode.ONBOARD

    assert controller.apply_verified_plan(plan(include_successor=True, revision=2)) is True
    refreshed = controller.snapshot()
    assert controller._recovered_uuv_ids_by_region["R1"] == {"U1"}
    assert refreshed.uuv_modes["U1"] is UUVMissionMode.ONBOARD

    recovered = controller.advance(
        60,
        {
            "recovered_uuv_ids": ("U2",),
            "health_check_passed": {"U2": True},
        },
    )
    assert recovered.regions[0].lifecycle is RegionLifecycle.RECOVERED


def test_mission_snapshot_domain_projection_is_reexported_by_runtime() -> None:
    group = TaskGroupInstance(
        group_instance_id="S1:T1:task:01:deploy:000001",
        target_id="T1",
        region_id="T1:task:01",
        deployment_revision=1,
        member_uuv_ids=("S1:UUV:01", "S1:UUV:02", "S1:UUV:03"),
        lifecycle=TaskGroupLifecycle.PASSIVE_TRACK,
        sensor_mode=GroupSensorMode.PASSIVE,
        ownership_status="owner",
        reason="initial_deployment",
        evidence_ids=("plan:1",),
    )
    snapshot = MissionSnapshot(
        scenario_id="S1",
        sim_time_s=30,
        plan_revision=1,
        task_groups=(group,),
        tracking_control=TrackingControlState(
            mode="regional",
            tracking_owner_group_id=group.group_instance_id,
        ),
    )

    assert RuntimeMissionSnapshot is MissionSnapshot
    assert snapshot.task_groups == (group,)
    assert snapshot.tracking_control.tracking_owner_group_id == group.group_instance_id


def test_mission_snapshot_mapping_defaults_are_not_shared() -> None:
    first = MissionSnapshot(scenario_id="S1", sim_time_s=0, plan_revision=0)
    second = MissionSnapshot(scenario_id="S2", sim_time_s=0, plan_revision=0)

    assert first.pending_region_revisions is not second.pending_region_revisions
    assert first.uuv_modes is not second.uuv_modes
    assert first.uuv_resources is not second.uuv_resources
    assert first.resource_episode_by_uuv is not second.resource_episode_by_uuv
    assert first.dedicated_target_by_uuv is not second.dedicated_target_by_uuv
    assert first.carrier_missions is not second.carrier_missions
