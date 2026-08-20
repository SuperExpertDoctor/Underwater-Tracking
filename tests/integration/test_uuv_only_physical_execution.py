from __future__ import annotations

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    ExecutableMissionPlan,
    RegionMissionState,
    UUVMissionBatch,
    UUVMissionMode,
)
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.simulation.engine import SimulationEngine


def _plan(config) -> ExecutableMissionPlan:
    assert config.environment is not None
    primary = config.environment.carrier
    secondary = next(
        carrier for carrier in config.environment.carriers if carrier.platform_id == "carrier_02"
    )
    primary_start = primary.position_xy
    secondary_start = secondary.position_xy
    deployment = (primary_start[0] + 200.0, primary_start[1])
    recovery = (primary_start[0] + 400.0, primary_start[1])
    batch = UUVMissionBatch(
        carrier_id="carrier_01",
        candidate_id="region-1",
        uuv_ids=("uuv_00",),
        active_scan_uuv_ids=("uuv_00",),
        deployment_point=deployment,
        recovery_point=recovery,
        entry_s=0,
        exit_s=100,
    )
    return ExecutableMissionPlan(
        revision=1,
        uuv_batches_by_carrier={"carrier_01": (batch,)},
        region_assignments=(
            RegionMissionState(
                region_id="region-1",
                target_id="target_00",
                active_scan_uuv_ids=("uuv_00",),
            ),
        ),
        carrier_missions={
            "carrier_01": CarrierMissionModel(
                carrier_id="carrier_01",
                home_battle_group_id="carrier_battle_group_01",
                route_xy=(primary_start, deployment, recovery, primary_start),
                stop_ids=("deploy:region-1", "recover:region-1"),
                ready_uuv_ids=("uuv_00",),
            ),
            "carrier_02": CarrierMissionModel(
                carrier_id="carrier_02",
                home_battle_group_id="carrier_battle_group_01",
                route_xy=(secondary_start, secondary_start),
            ),
        },
    )


def test_verified_plan_executes_two_carriers_and_uuv_deployment_recovery() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=20260820, mission_controller=controller)

    assert engine.apply_verified_mission_plan(_plan(config)) is True
    for _ in range(80):
        engine.step()

    carriers = engine.carrier_states()
    assert set(carriers) == {"carrier_01", "carrier_02"}
    assert all(carrier.mission_route_complete for carrier in engine._carrier_entities.values())
    assert any(event.event_type == "uuv_deployed" for event in engine.events())
    assert any(event.event_type == "uuv_recovered" for event in engine.events())
    assert engine.mission_snapshot() is not None
    assert engine.mission_snapshot().uuv_modes["uuv_00"] is UUVMissionMode.ONBOARD


def test_exhausted_uuv_is_recovered_and_sortie_distance_is_reset() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(
        scenario_id=config.scenario.scenario_id,
        max_uuv_mileage_m=1_000.0,
    )
    engine = SimulationEngine(config, seed=20260820, mission_controller=controller)
    plan = _plan(config)
    assert engine.apply_verified_mission_plan(plan) is True
    engine._mission_distance_m["uuv_00"] = 1_000.0

    for _ in range(80):
        engine.step()

    events = engine.events()
    assert any(event.event_type == "uuv_range_exhausted" for event in events)
    assert any(event.event_type == "carrier_recovery_completed" for event in events)
    assert engine.mission_distance("uuv_00") == 0.0


def test_engine_completes_windowed_scan_passive_handoff_and_final_recovery() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(
        scenario_id=config.scenario.scenario_id,
        region_entry_probability_threshold=0.70,
        region_transition_confirm_cycles=2,
    )
    assert config.environment is not None
    primary = config.environment.carrier
    secondary = next(
        carrier
        for carrier in config.environment.carriers
        if carrier.platform_id == "carrier_02"
    )
    start = primary.position_xy
    batches = tuple(
        UUVMissionBatch(
            carrier_id="carrier_01",
            candidate_id=region_id,
            uuv_ids=(active_id, passive_id),
            active_scan_uuv_ids=(active_id,),
            passive_track_uuv_ids=(passive_id,),
            deployment_point=(start[0] + offset, start[1]),
            recovery_point=(start[0] + offset + 100.0, start[1]),
            entry_s=0,
            exit_s=150,
        )
        for region_id, active_id, passive_id, offset in (
            ("R1", "uuv_00", "uuv_01", 100.0),
            ("R2", "uuv_02", "uuv_03", 300.0),
        )
    )
    plan = ExecutableMissionPlan(
        revision=1,
        uuv_batches_by_carrier={"carrier_01": batches},
        region_assignments=(
            RegionMissionState(
                region_id="R1",
                target_id="target_00",
                active_scan_uuv_ids=("uuv_00",),
                passive_track_uuv_ids=("uuv_01",),
                handoff_to="R2",
            ),
            RegionMissionState(
                region_id="R2",
                target_id="target_00",
                active_scan_uuv_ids=("uuv_02",),
                passive_track_uuv_ids=("uuv_03",),
                handoff_from="R1",
            ),
        ),
        carrier_missions={
            "carrier_01": CarrierMissionModel(
                carrier_id="carrier_01",
                home_battle_group_id="carrier_battle_group_01",
                ready_uuv_ids=("uuv_00", "uuv_01", "uuv_02", "uuv_03"),
            ),
            "carrier_02": CarrierMissionModel(
                carrier_id="carrier_02",
                home_battle_group_id="carrier_battle_group_01",
                route_xy=(secondary.position_xy, secondary.position_xy),
            ),
        },
    )
    engine = SimulationEngine(config, seed=20260820, mission_controller=controller)

    assert engine.apply_verified_mission_plan(plan) is True
    lifecycle_trace: list[str] = []
    for _ in range(120):
        engine.step()
        lifecycle_trace.extend(region.lifecycle.value for region in controller.snapshot().regions)

    snapshot = controller.snapshot()
    lifecycles = {region.region_id: region.lifecycle.value for region in snapshot.regions}
    events = {event.event_type for event in engine.events()}
    assert "ACTIVE_SCAN" in lifecycle_trace
    assert "PASSIVE_TRACK" in lifecycle_trace
    assert lifecycles == {"R1": "RECOVERED", "R2": "RECOVERED"}
    assert events >= {
        "carrier_dispatch_completed",
        "target_entered_region",
        "handoff_completed",
        "target_exit_predicted",
        "carrier_recovery_completed",
    }
    assert all(
        snapshot.uuv_modes[uuv_id] is UUVMissionMode.ONBOARD
        for uuv_id in ("uuv_00", "uuv_01", "uuv_02", "uuv_03")
    )
    assert all(carrier.mission_route_complete for carrier in engine._carrier_entities.values())
