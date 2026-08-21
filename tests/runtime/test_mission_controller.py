import pytest

from underwater_tracking.domain.mission_models import (
    AcceptedHandoffObservation,
    CarrierMissionModel,
    ExecutableMissionPlan,
    HandoffEvidence,
    RegionLifecycle,
    RegionMissionState,
    UUVMissionBatch,
    UUVMissionMode,
)
from underwater_tracking.runtime.mission_controller import MissionController


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
    assert deployed.uuv_modes["U2"] is UUVMissionMode.ACTIVE_SCAN
    controller.advance(20, {"entry_probability": {"R1": 0.8}})
    assert controller.snapshot().regions[0].lifecycle is RegionLifecycle.ACTIVE_SCAN
    controller.advance(30, {"entry_probability": {"R1": 0.8}})

    snapshot = controller.snapshot()
    assert snapshot.regions[0].lifecycle is RegionLifecycle.PASSIVE_TRACK
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.PASSIVE_TRACK
    assert [event.event_type for event in snapshot.events] == ["target_entered_region"]


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


def test_mileage_exhaustion_is_idempotent_and_enqueues_recovery() -> None:
    controller = MissionController(scenario_id="S1", max_uuv_mileage_m=1_000.0)
    controller.apply_verified_plan(plan())

    controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}})
    controller.advance(20, {"mileage_m": {"U1": 1_001.0}})
    controller.advance(30, {"mileage_m": {"U1": 1_001.0}})

    snapshot = controller.snapshot()
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.carrier_missions["carrier_01"].recoverable_uuv_ids == ("U1",)
    assert [event.event_type for event in snapshot.events].count("uuv_range_exhausted") == 1


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


def test_handoff_marks_predecessor_uuvs_for_carrier_recovery() -> None:
    controller = _prepare_handoff_controller()
    snapshot = controller.advance(
        30,
        {"handoff_evidence": {"R1": _typed_handoff_evidence()}},
    )

    assert snapshot.uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.uuv_modes["U2"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.carrier_missions["carrier_01"].recoverable_uuv_ids == ("U1", "U2")


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


def test_recovery_requires_health_check_and_completes_after_all_uuvs_return() -> None:
    controller = _prepare_handoff_controller()
    controller.advance(
        30,
        {"handoff_evidence": {"R1": _typed_handoff_evidence()}},
    )
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
