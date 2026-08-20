from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    ExecutableMissionPlan,
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
    assert controller.snapshot().regions[0].lifecycle is RegionLifecycle.ACTIVE_SCAN
    controller.advance(20, {"entry_probability": {"R1": 0.8}})
    assert controller.snapshot().regions[0].lifecycle is RegionLifecycle.ACTIVE_SCAN
    controller.advance(30, {"entry_probability": {"R1": 0.8}})

    snapshot = controller.snapshot()
    assert snapshot.regions[0].lifecycle is RegionLifecycle.PASSIVE_TRACK
    assert snapshot.uuv_modes["U1"] is UUVMissionMode.PASSIVE_TRACK
    assert [event.event_type for event in snapshot.events] == ["target_entered_region"]


def test_handoff_activates_successor_before_predecessor_closes() -> None:
    controller = MissionController(scenario_id="S1", region_transition_confirm_cycles=1)
    controller.apply_verified_plan(plan(include_successor=True))
    controller.advance(
        10,
        {
            "deployed_uuv_ids": {"R1": ("U1", "U2"), "R2": ("U4", "U5")},
            "entry_probability": {"R1": 0.9},
        },
    )
    assert controller.snapshot().regions[0].lifecycle is RegionLifecycle.PASSIVE_TRACK

    controller.advance(
        20,
        {
            "handoff_ready": {"R1": "R2"},
            "successor_passive_ready": {"R2": True},
        },
    )

    regions = {region.region_id: region for region in controller.snapshot().regions}
    assert regions["R2"].lifecycle is RegionLifecycle.PASSIVE_TRACK
    assert regions["R1"].lifecycle is RegionLifecycle.TRACKING_COMPLETED
    assert controller.snapshot().events[-1].event_type == "handoff_completed"


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
    snapshot = controller.advance(20, {"recovered_uuv_ids": ("U1",)})

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


def test_handoff_marks_predecessor_uuvs_for_carrier_recovery() -> None:
    controller = MissionController(scenario_id="S1", region_transition_confirm_cycles=1)
    controller.apply_verified_plan(plan(include_successor=True))
    controller.advance(
        10,
        {
            "deployed_uuv_ids": {"R1": ("U1", "U2"), "R2": ("U4", "U5")},
            "entry_probability": {"R1": 0.9},
        },
    )

    snapshot = controller.advance(
        20,
        {
            "handoff_ready": {"R1": "R2"},
            "successor_passive_ready": {"R2": True},
        },
    )

    assert snapshot.uuv_modes["U1"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.uuv_modes["U2"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.carrier_missions["carrier_01"].recoverable_uuv_ids == ("U1", "U2")
