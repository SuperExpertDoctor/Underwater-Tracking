import pytest

from tests.domain.test_execution_models import _snapshot as _execution_snapshot
from underwater_tracking.cli import _mission_controller_for
from underwater_tracking.config.loader import load_app_config
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


def test_equivalent_new_plan_preserves_region_lifecycle_progress() -> None:
    controller = _prepare_handoff_controller()
    controller.advance(30, {"handoff_evidence": {"R1": _typed_handoff_evidence()}})

    assert controller.apply_verified_plan(plan(include_successor=True, revision=2)) is True

    regions = {region.region_id: region for region in controller.snapshot().regions}
    assert regions["R1"].lifecycle is RegionLifecycle.TRACKING_COMPLETED
    assert regions["R2"].lifecycle is RegionLifecycle.PASSIVE_TRACK


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
