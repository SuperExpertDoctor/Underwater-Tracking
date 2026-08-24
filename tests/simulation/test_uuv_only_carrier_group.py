from __future__ import annotations

from math import hypot

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.models import (
    BearingObservation,
    DeploymentState,
    GroupQuality,
    GroupReport,
    TargetBelief,
)
from underwater_tracking.domain.observations import PassiveSonarObservation
from underwater_tracking.domain.mission_models import (
    CarrierExecutionMode,
    CarrierMissionModel,
    CarrierRouteStatus,
    ExecutableMissionPlan,
    RegionLifecycle,
    RegionMissionState,
    UUVResourceState,
    UUVMissionBatch,
    UUVMissionMode,
)
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.simulation.connectivity import has_path
import underwater_tracking.simulation.engine as engine_module
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.simulation.carrier_group import carrier_slot_position


def _carrier_plan(
    config,
    *,
    start_position: tuple[float, float] | None = None,
    rendezvous_position: tuple[float, float] | None = None,
    revision: int = 1,
    entry_s: int = 0,
    exit_s: int = 120,
) -> ExecutableMissionPlan:
    assert config.environment is not None
    mother = next(
        carrier
        for carrier in config.environment.carriers
        if carrier.platform_id == "carrier_02"
    )
    start = mother.position_xy if start_position is None else start_position
    rendezvous = start if rendezvous_position is None else rendezvous_position
    batch = UUVMissionBatch(
        carrier_id="carrier_02",
        candidate_id="target_00:r0",
        uuv_ids=("uuv_00",),
        active_scan_uuv_ids=("uuv_00",),
        deployment_point=(start[0] + 100.0, start[1]),
        recovery_point=(start[0] + 200.0, start[1]),
        entry_s=entry_s,
        exit_s=exit_s,
    )
    return ExecutableMissionPlan(
        revision=revision,
        uuv_batches_by_carrier={"carrier_02": (batch,)},
        region_assignments=(
            RegionMissionState(
                region_id="target_00:r0",
                target_id="target_00",
                active_scan_uuv_ids=("uuv_00",),
            ),
        ),
        carrier_missions={
            carrier.platform_id: CarrierMissionModel(
                carrier_id=carrier.platform_id,
                home_battle_group_id="carrier_battle_group_01",
                ready_uuv_ids=("uuv_00",)
                if carrier.platform_id == "carrier_02"
                else (),
                route_xy=(
                    start,
                    (start[0] + 100.0, start[1]),
                    (start[0] + 200.0, start[1]),
                    rendezvous,
                )
                if carrier.platform_id == "carrier_02"
                else (),
                stop_ids=("deploy:target_00:r0", "recover:target_00:r0")
                if carrier.platform_id == "carrier_02"
                else (),
            )
            for carrier in (config.environment.carrier, *config.environment.carriers)
            if carrier.platform_id
        },
    )


def test_uuv_only_initializes_one_carrier_and_three_mother_ship_support_points() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)

    assert engine._carrier_entities["carrier_01"].role == "carrier"
    assert {
        carrier_id
        for carrier_id, role in engine._carrier_roles.items()
        if role == "mother_ship"
    } == {"carrier_02", "carrier_03", "carrier_04"}
    assert set(engine._uuv_carrier_ids.values()) <= {
        "carrier_02",
        "carrier_03",
        "carrier_04",
    }
    for uuv_id, carrier_id in engine._uuv_carrier_ids.items():
        assert engine._uuvs[uuv_id].position_xy == engine._carrier_entities[carrier_id].position_xy


def test_uuv_only_initial_inventory_uses_configured_mother_ownership() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)
    assert config.environment is not None

    expected_owner_by_uuv = {
        uuv.platform_id: uuv.home_carrier_id for uuv in config.environment.uuvs
    }
    assert engine._waterborne_uuv_ids == set()
    assert set(engine._deployment_states.values()) == {DeploymentState.ONBOARD}
    assert engine._uuv_carrier_ids == expected_owner_by_uuv
    assert engine._carrier_entities["carrier_01"].position_xy == (-8000.0, -8000.0)
    for uuv_id, owner_id in expected_owner_by_uuv.items():
        assert owner_id is not None
        assert engine._uuvs[uuv_id].position_xy == engine._carrier_entities[owner_id].position_xy

    carrier_states = engine.carrier_states()
    assert carrier_states["carrier_01"].onboard_uuv_ids == ()
    assert carrier_states["carrier_02"].onboard_uuv_ids == (
        "uuv_00",
        "uuv_01",
        "uuv_02",
        "uuv_03",
    )
    assert carrier_states["carrier_03"].onboard_uuv_ids == (
        "uuv_04",
        "uuv_05",
        "uuv_06",
        "uuv_07",
    )
    assert carrier_states["carrier_04"].onboard_uuv_ids == (
        "uuv_08",
        "uuv_09",
        "uuv_10",
        "uuv_11",
    )


def test_standby_mothers_follow_rotating_leader_slots_with_bounded_motion() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)
    leader = engine._carrier_entities["carrier_01"]
    mother = engine._carrier_entities["carrier_02"]
    mother.position_xy = (leader.position_xy[0], leader.position_xy[1] - 3000.0)
    before = mother.position_xy

    engine.step()

    assert leader.position_xy != (-8000.0, -8000.0)
    displacement = hypot(
        mother.position_xy[0] - before[0], mother.position_xy[1] - before[1]
    )
    assert displacement <= mother.speed_mps * config.timing.physics_step_s + 1e-9
    assert mother.position_xy[1] > before[1]
    expected_slot = carrier_slot_position(
        leader.position_xy,
        leader.heading_rad,
        (0.0, -1000.0),
    )
    assert hypot(
        mother.position_xy[0] - expected_slot[0],
        mother.position_xy[1] - expected_slot[1],
    ) < 3000.0
    assert mother.execution_mode is CarrierExecutionMode.FORMATION_FOLLOW


def test_operational_mother_slot_stays_world_anchored_when_leader_turns() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)
    leader = engine._carrier_entities["carrier_01"]
    mother = engine._carrier_entities["carrier_03"]
    leader.position_xy = (0.0, 0.0)
    leader.heading_rad = 1.5707963267948966
    mother.position_xy = (1000.0, 0.0)
    mother.execution_mode = CarrierExecutionMode.FORMATION_FOLLOW

    slot = engine._current_carrier_slot_position("carrier_03")

    assert slot == (1000.0, 0.0)
    assert hypot(mother.position_xy[0] - slot[0], mother.position_xy[1] - slot[1]) == 0.0


def test_mission_route_can_end_at_predicted_moving_slot() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    plan = _carrier_plan(config)
    route = plan.carrier_missions["carrier_02"].route_xy
    moving_endpoint = (route[-1][0] + 500.0, route[-1][1])
    mission = plan.carrier_missions["carrier_02"].model_copy(
        update={
            "route_xy": (*route[:-1], moving_endpoint),
        }
    )
    plan = plan.model_copy(
        update={
            "carrier_missions": {
                **plan.carrier_missions,
                "carrier_02": mission,
            }
        }
    )

    assert engine.apply_verified_mission_plan(plan) is True
    assert engine._carrier_entities["carrier_02"].mission_route_xy[-1] == moving_endpoint
    assert engine._carrier_entities["carrier_02"].execution_mode is CarrierExecutionMode.MISSION_ROUTE
    assert engine._carrier_route_status_for("carrier_02") is CarrierRouteStatus.DEPLOYING


def test_uuv_physical_exposure_tracks_deployment_failure_and_recovery() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)

    engine.request_uuv_deployment("uuv_00", reason="test_deploy")
    assert "uuv_00" in engine._waterborne_uuv_ids
    assert engine._uuv_state("uuv_00").physically_exposed is True

    engine.fail_uuv("uuv_00")
    assert engine._deployment_states["uuv_00"] is DeploymentState.FAILED
    assert "uuv_00" in engine._waterborne_uuv_ids
    assert engine._uuv_state("uuv_00").physically_exposed is True

    engine.fail_uuv("uuv_01")
    assert engine._deployment_states["uuv_01"] is DeploymentState.FAILED
    assert "uuv_01" not in engine._waterborne_uuv_ids
    assert engine._uuv_state("uuv_01").physically_exposed is False

    engine.request_uuv_deployment("uuv_02", reason="test_deploy")
    engine.request_uuv_recovery("uuv_02", reason="test_recover")
    engine._complete_uuv_recovery("uuv_02", sim_time_s=30)
    assert engine._deployment_states["uuv_02"] is DeploymentState.ONBOARD
    assert "uuv_02" not in engine._waterborne_uuv_ids
    assert engine._uuv_state("uuv_02").physically_exposed is False


def test_newly_deployed_uuv_and_event_share_publication_boundary() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)

    engine.request_uuv_deployment("uuv_00", reason="test_deploy")
    frame = engine.step()
    uuv = next(item for item in frame["uuvs"] if item["platform_id"] == "uuv_00")

    assert uuv["physically_exposed"] is True
    assert any(
        event.event_type == "uuv_deployed" and event.entity_id == "uuv_00"
        for event in engine.events()
    )


def test_uuv_only_public_situation_exposes_all_carrier_roles() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)

    situation = engine.publication_situation()

    assert tuple(carrier.carrier_id for carrier in situation.carriers) == (
        "carrier_01",
        "carrier_02",
        "carrier_03",
        "carrier_04",
    )
    assert situation.carrier is not None
    assert situation.carrier.carrier_id == "carrier_01"
    assert situation.carrier.role == "carrier"
    assert situation.carrier.onboard_uuv_ids == ()
    assert {carrier.role for carrier in situation.carriers[1:]} == {
        "mother_ship"
    }


def test_uuv_only_connectivity_uses_the_mother_ship_and_fleet_mesh() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)

    engine._deployment_states["uuv_00"] = DeploymentState.DEPLOYED
    engine._uuvs["uuv_00"].position_xy = engine._carrier_entities[
        "carrier_02"
    ].position_xy
    engine._rebuild_connectivity()

    assert has_path(engine._connectivity, "carrier_02", "uuv_00")
    assert has_path(engine._connectivity, "carrier_01", "uuv_00")
    assert any(
        link.source_id == "carrier_02"
        and link.target_id == "uuv_00"
        and link.medium == "acoustic"
        for link in engine._connectivity.links
    )


def test_mother_ship_deploys_recovers_and_returns_to_fleet() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)

    assert engine.apply_verified_mission_plan(_carrier_plan(config)) is True
    for _ in range(100):
        engine.step()

    mother = engine._carrier_entities["carrier_02"]
    assert mother.mission_route_xy == ()
    assert mother.execution_mode is CarrierExecutionMode.FORMATION_FOLLOW
    assert engine._carrier_route_status_for("carrier_02") is CarrierRouteStatus.COMPLETE
    returned = [
        event.event_type == "carrier_returned_to_fleet"
        and event.entity_id == "carrier_02"
        for event in engine.events()
    ]
    assert sum(returned) == 1
    return_event = next(
        event
        for event in engine.events()
        if event.event_type == "carrier_returned_to_fleet"
        and event.entity_id == "carrier_02"
    )
    assert return_event.payload["deployed_uuv_ids"] == ()
    assert return_event.payload["returning_uuv_ids"] == ()
    assert engine.mission_snapshot() is not None
    assert engine.mission_snapshot().uuv_modes["uuv_00"] is UUVMissionMode.ONBOARD


def test_plan_ignores_episode_changes_for_uuvs_removed_from_the_sortie() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    plan = _carrier_plan(config).model_copy(
        update={
            "resource_episode_by_uuv": {
                uuv_id: 0 for uuv_id in engine._uuvs
            }
        }
    )
    controller._resource_episode_by_uuv.update({"uuv_00": 0, "uuv_02": 1})

    assert engine.apply_verified_mission_plan(plan) is True


def test_mother_holds_recovery_stop_until_owned_uuv_is_onboard() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)

    assert engine.apply_verified_mission_plan(_carrier_plan(config)) is True
    for _ in range(30):
        engine.step()
        mother = engine._carrier_entities["carrier_02"]
        if mother.awaiting_release_stop_index is not None:
            break
    else:
        raise AssertionError("carrier did not reach its externally released recovery stop")

    held_position = mother.position_xy
    assert engine._deployment_states["uuv_00"] is DeploymentState.RETURNING
    engine.step()
    assert mother.position_xy == held_position
    assert mother.awaiting_release_stop_index is not None
    assert engine._deployment_states["uuv_00"] in {
        DeploymentState.RETURNING,
        DeploymentState.ONBOARD,
    }


def test_mother_ship_emits_one_return_event_per_voyage() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)

    assert engine.apply_verified_mission_plan(_carrier_plan(config)) is True
    for _ in range(100):
        engine.step()

    mother = engine._carrier_entities["carrier_02"]
    second_plan = _carrier_plan(
        config,
        start_position=mother.position_xy,
        rendezvous_position=engine._current_carrier_slot_position("carrier_02"),
        revision=2,
        entry_s=engine._clock.sim_time_s,
        exit_s=engine._clock.sim_time_s + 120,
    )
    assert engine.apply_verified_mission_plan(second_plan) is True
    for _ in range(100):
        engine.step()

    returned = [
        event
        for event in engine.events()
        if event.event_type == "carrier_returned_to_fleet"
        and event.entity_id == "carrier_02"
    ]
    assert len(returned) == 2
    assert len({event.event_id for event in returned}) == 2


def test_replan_with_new_uuv_rebuilds_partial_active_sortie_route() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)

    first_plan = _carrier_plan(config)
    assert engine.apply_verified_mission_plan(first_plan) is True
    for _ in range(30):
        engine.step()
        if engine._deployment_states["uuv_00"] is DeploymentState.DEPLOYED:
            break
    assert engine._deployment_states["uuv_00"] is DeploymentState.DEPLOYED

    mother = engine._carrier_entities["carrier_02"]
    first_batch = first_plan.batches[0]
    current_time_s = engine._clock.sim_time_s
    second_batch = first_batch.model_copy(
        update={
            "uuv_ids": ("uuv_00", "uuv_01"),
            "active_scan_uuv_ids": ("uuv_00", "uuv_01"),
            "entry_s": current_time_s,
            "exit_s": current_time_s + 120,
        }
    )
    second_region = first_plan.region_assignments[0].model_copy(
        update={"active_scan_uuv_ids": ("uuv_00", "uuv_01")}
    )
    second_mission = first_plan.carrier_missions["carrier_02"].model_copy(
        update={
            "ready_uuv_ids": ("uuv_00", "uuv_01"),
            "route_xy": (
                mother.position_xy,
                (mother.position_xy[0] + 100.0, mother.position_xy[1]),
                (mother.position_xy[0] + 200.0, mother.position_xy[1]),
                mother.position_xy,
            ),
            "stop_ids": (
                "deploy:target_00:r0",
                "recover:target_00:r0",
            ),
            "stop_windows": (
                (current_time_s, current_time_s + 120),
                (current_time_s, current_time_s + 120),
            ),
        }
    )
    second_plan = first_plan.model_copy(
        update={
            "revision": 2,
            "uuv_batches_by_carrier": {"carrier_02": (second_batch,)},
            "region_assignments": (second_region,),
            "carrier_missions": {
                **first_plan.carrier_missions,
                "carrier_02": second_mission,
            },
        }
    )

    assert engine.apply_verified_mission_plan(second_plan) is True
    for _ in range(40):
        engine.step()
        if engine._deployment_states["uuv_01"] is DeploymentState.DEPLOYED:
            break

    assert engine._deployment_states["uuv_01"] is DeploymentState.DEPLOYED
    assert any(
        event.event_type == "uuv_deployed" and event.entity_id == "uuv_01"
        for event in engine.events()
    )


def test_replan_preserves_inflight_handoff_batches_when_new_plan_omits_them() -> None:
    """A rolling plan cannot withdraw a successor that is already in the water."""
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    first_plan = _carrier_plan(config)
    first_batch = first_plan.batches[0].model_copy(
        update={
            "candidate_id": "target_00:predecessor",
            "uuv_ids": ("uuv_00",),
            "active_scan_uuv_ids": (),
            "passive_track_uuv_ids": ("uuv_00",),
            "entry_s": 0,
            "exit_s": 120,
        }
    )
    successor_batch = first_batch.model_copy(
        update={
            "candidate_id": "target_00:successor",
            "uuv_ids": ("uuv_01",),
            "passive_track_uuv_ids": ("uuv_01",),
            "entry_s": 0,
            "exit_s": 120,
        }
    )
    predecessor = first_plan.region_assignments[0].model_copy(
        update={
            "region_id": first_batch.candidate_id,
            "active_scan_uuv_ids": (),
            "passive_track_uuv_ids": ("uuv_00",),
            "handoff_to": successor_batch.candidate_id,
            "lifecycle": RegionLifecycle.PASSIVE_TRACK,
        }
    )
    successor = predecessor.model_copy(
        update={
            "region_id": successor_batch.candidate_id,
            "passive_track_uuv_ids": ("uuv_01",),
            "handoff_from": predecessor.region_id,
            "handoff_to": None,
        }
    )
    carrier_mission = first_plan.carrier_missions["carrier_02"].model_copy(
        update={
            "ready_uuv_ids": ("uuv_00", "uuv_01"),
            "route_xy": (),
            "stop_ids": (),
            "stop_indices": (),
            "stop_windows": (),
        }
    )
    first_plan = first_plan.model_copy(
        update={
            "uuv_batches_by_carrier": {
                "carrier_02": (first_batch, successor_batch),
            },
            "region_assignments": (predecessor, successor),
            "carrier_missions": {
                **first_plan.carrier_missions,
                "carrier_02": carrier_mission,
            },
        }
    )

    assert engine.apply_verified_mission_plan(first_plan) is True
    carrier = engine._carrier_entities["carrier_02"]
    committed_route = carrier.mission_route_xy
    for uuv_id, batch in (
        ("uuv_00", first_batch),
        ("uuv_01", successor_batch),
    ):
        engine._deployment_states[uuv_id] = DeploymentState.DEPLOYED
        assert batch.deployment_point is not None
        engine._uuvs[uuv_id].position_xy = batch.deployment_point

    second_plan = first_plan.model_copy(
        update={
            "revision": 2,
            "uuv_batches_by_carrier": {},
            "region_assignments": (),
            "carrier_missions": {
                carrier_id: mission.model_copy(
                    update={
                        "ready_uuv_ids": (),
                        "recoverable_uuv_ids": (),
                        "route_xy": (),
                        "stop_ids": (),
                        "stop_indices": (),
                        "stop_windows": (),
                    }
                )
                for carrier_id, mission in first_plan.carrier_missions.items()
            },
        }
    )

    assert engine.apply_verified_mission_plan(second_plan) is True
    assert engine._mission_plan is not None
    assert {
        batch.candidate_id for batch in engine._mission_plan.batches
    } == {first_batch.candidate_id, successor_batch.candidate_id}
    assert {
        assignment.region_id for assignment in engine._mission_plan.region_assignments
    } == {first_batch.candidate_id, successor_batch.candidate_id}
    assert engine._carrier_entities["carrier_02"].mission_route_xy == committed_route
    assert engine._deployment_states["uuv_00"] is DeploymentState.DEPLOYED
    assert engine._deployment_states["uuv_01"] is DeploymentState.DEPLOYED


def test_infeasible_rendezvous_retains_safe_route_and_recovers(monkeypatch) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    carrier = engine._carrier_entities["carrier_02"]
    safe_route = (carrier.position_xy, (carrier.position_xy[0] + 100.0, carrier.position_xy[1]))
    carrier.set_mission_route(safe_route, rendezvous_xy=safe_route[-1])
    carrier.execution_mode = CarrierExecutionMode.MISSION_ROUTE
    original_route = carrier.mission_route_xy
    original_solver = engine_module.solve_moving_rendezvous
    monkeypatch.setattr(engine_module, "solve_moving_rendezvous", lambda **_: None)

    engine._update_carrier_rendezvous_tails(0)

    assert carrier.mission_route_xy == original_route
    assert engine._carrier_route_status_for("carrier_02") is CarrierRouteStatus.RENDEZVOUS_BLOCKED
    assert sum(
        event.event_type == "carrier_rendezvous_infeasible"
        and event.entity_id == "carrier_02"
        for event in engine._events
    ) == 1

    monkeypatch.setattr(engine_module, "solve_moving_rendezvous", original_solver)
    engine._update_carrier_rendezvous_tails(0)
    assert engine._carrier_route_status_for("carrier_02") is CarrierRouteStatus.RETURNING_TO_FLEET


def test_failed_recovery_member_blocks_mother_at_recovery_stop() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    assert engine.apply_verified_mission_plan(_carrier_plan(config)) is True
    engine.fail_uuv("uuv_00")

    for _ in range(40):
        engine.step()
        if engine._carrier_route_status_for("carrier_02") is CarrierRouteStatus.FAILED:
            break
    else:
        raise AssertionError("failed recovery member did not block the mother ship")

    carrier = engine._carrier_entities["carrier_02"]
    assert carrier.awaiting_release_stop_index is not None
    assert carrier.execution_mode is CarrierExecutionMode.MISSION_ROUTE
    assert not any(
        event.event_type == "carrier_returned_to_fleet"
        and event.entity_id == "carrier_02"
        for event in engine.events()
    )
    assert any(
        event.event_type == "carrier_recovery_blocked"
        and event.entity_id == "carrier_02"
        for event in engine.events()
    )


def test_replan_without_new_mother_ship_batch_returns_active_route_to_fleet() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)

    assert engine.apply_verified_mission_plan(_carrier_plan(config)) is True
    engine.step()
    assert engine._carrier_entities["carrier_02"].mission_route_complete is False

    empty_plan = ExecutableMissionPlan(
        revision=2,
        carrier_missions={
            carrier.platform_id: CarrierMissionModel(
                carrier_id=carrier.platform_id,
                home_battle_group_id="carrier_battle_group_01",
                role=carrier.role,
            )
            for carrier in (config.environment.carrier, *config.environment.carriers)
            if carrier.platform_id
        },
    )
    assert engine.apply_verified_mission_plan(empty_plan) is True
    for _ in range(100):
        engine.step()

    mother = engine._carrier_entities["carrier_02"]
    assert mother.mission_route_xy == ()
    assert mother.execution_mode is CarrierExecutionMode.FORMATION_FOLLOW
    assert engine._carrier_route_status_for("carrier_02") is CarrierRouteStatus.COMPLETE
    assert sum(
        event.event_type == "carrier_returned_to_fleet"
        and event.entity_id == "carrier_02"
        for event in engine.events()
    ) == 1


def test_infeasible_future_handoff_is_degraded_without_failing_the_run() -> None:
    """A missed future service window must not abort an otherwise valid sortie."""
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    assert config.environment is not None

    # This is the geometry reached by the formation at the 900 s strategic
    # review in the live scenario.  The successor window closes too soon for
    # carrier_02, while the current batch remains reachable.
    engine._clock.sim_time_s = 900
    engine._carrier_entities["carrier_01"].position_xy = (-4400.0, -8000.0)
    engine._carrier_entities["carrier_02"].position_xy = (-3000.0, -8100.0)
    engine._carrier_entities["carrier_03"].position_xy = (-3400.0, -8000.0)
    engine._carrier_entities["carrier_04"].position_xy = (-4400.0, -7000.0)

    current_id = "target_00:cell:-1:-3"
    future_id = "target_00:cell:-3:2"
    current_batches = {
        "carrier_02": UUVMissionBatch(
            carrier_id="carrier_02",
            candidate_id=current_id,
            uuv_ids=("uuv_02",),
            active_scan_uuv_ids=("uuv_02",),
            deployment_point=(-2000.0, -6000.0),
            recovery_point=(0.0, -4000.0),
            entry_s=0,
            exit_s=2100,
        ),
        "carrier_03": UUVMissionBatch(
            carrier_id="carrier_03",
            candidate_id=current_id,
            uuv_ids=("uuv_07",),
            passive_track_uuv_ids=("uuv_07",),
            deployment_point=(-2000.0, -6000.0),
            recovery_point=(0.0, -4000.0),
            entry_s=0,
            exit_s=2100,
        ),
    }
    future_batch = UUVMissionBatch(
        carrier_id="carrier_02",
        candidate_id=future_id,
        uuv_ids=("uuv_01", "uuv_03"),
        active_scan_uuv_ids=("uuv_01",),
        passive_track_uuv_ids=("uuv_03",),
        deployment_point=(-6000.0, 4000.0),
        recovery_point=(-4000.0, 6000.0),
        entry_s=600,
        exit_s=2100,
    )
    assignments = (
        RegionMissionState(
            region_id=current_id,
            target_id="target_00",
            lifecycle=RegionLifecycle.PLANNED,
            active_scan_uuv_ids=("uuv_02",),
            passive_track_uuv_ids=("uuv_07",),
            region_polygon=((-2000.0, -6000.0), (-2000.0, -4000.0), (0.0, -4000.0), (0.0, -6000.0)),
            handoff_to=future_id,
        ),
        RegionMissionState(
            region_id=future_id,
            target_id="target_00",
            lifecycle=RegionLifecycle.PLANNED,
            active_scan_uuv_ids=("uuv_01",),
            passive_track_uuv_ids=("uuv_03",),
            region_polygon=((-6000.0, 4000.0), (-6000.0, 6000.0), (-4000.0, 6000.0), (-4000.0, 4000.0)),
            handoff_from=current_id,
        ),
    )
    missions = {
        carrier.platform_id: CarrierMissionModel(
            carrier_id=carrier.platform_id,
            role=carrier.role,
            home_battle_group_id=config.scenario.home_battle_group_id,
            ready_uuv_ids=tuple(
                uuv_id
                for uuv_id, owner_id in engine._uuv_carrier_ids.items()
                if owner_id == carrier.platform_id
            ),
        )
        for carrier in (config.environment.carrier, *config.environment.carriers)
    }
    plan = ExecutableMissionPlan(
        revision=1,
        uuv_batches_by_carrier={
            **{
                carrier_id: (batch,)
                for carrier_id, batch in current_batches.items()
            },
            "carrier_02": (current_batches["carrier_02"], future_batch),
        },
        region_assignments=assignments,
        carrier_missions=missions,
        resource_episode_by_uuv=engine._mission_resource_episodes(),
    )

    assert engine.apply_verified_mission_plan(plan) is True, engine._last_mission_plan_failure_reason
    assert engine._mission_plan is not None
    assert future_id not in {
        batch.candidate_id for batch in engine._mission_plan.batches
    }
    future_assignment = engine._mission_plan.assignments_by_candidate[future_id]
    assert future_assignment.lifecycle is RegionLifecycle.UNCOVERED
    assert "carrier_route_infeasible" in future_assignment.degraded_reasons
    assert engine._mission_plan.assignments_by_candidate[current_id].handoff_to is None
    assert any(
        event.event_type == "carrier_plan_degraded"
        and event.payload.get("candidate_id") == future_id
        for event in engine.events()
    )


def test_infeasible_unstarted_current_batch_is_degraded_without_failing_the_run() -> None:
    """A not-yet-deployed batch past its physical window must be withdrawn atomically."""
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    assert config.environment is not None

    # This is the live t=900 s geometry.  The carrier reaches the deployment
    # perimeter at 2112.5 s, just after the candidate's 2100 s deadline.
    engine._clock.sim_time_s = 900
    engine._carrier_entities["carrier_01"].position_xy = (-4400.0, -8000.0)
    engine._carrier_entities["carrier_02"].position_xy = (-6600.0, -900.0)
    engine._carrier_entities["carrier_03"].position_xy = (-4000.0, -6000.0)
    engine._carrier_entities["carrier_04"].position_xy = (-4400.0, -7000.0)

    candidate_id = "target_00:cell:-1:-3"
    batch = UUVMissionBatch(
        carrier_id="carrier_02",
        candidate_id=candidate_id,
        uuv_ids=("uuv_00", "uuv_02"),
        active_scan_uuv_ids=("uuv_00",),
        passive_track_uuv_ids=("uuv_02",),
        deployment_point=(-2000.0, -6000.0),
        recovery_point=(0.0, -4000.0),
        entry_s=0,
        exit_s=2100,
    )
    missions = {
        carrier.platform_id: CarrierMissionModel(
            carrier_id=carrier.platform_id,
            role=carrier.role,
            home_battle_group_id=config.scenario.home_battle_group_id,
            ready_uuv_ids=tuple(
                uuv_id
                for uuv_id, owner_id in engine._uuv_carrier_ids.items()
                if owner_id == carrier.platform_id
            ),
        )
        for carrier in (config.environment.carrier, *config.environment.carriers)
    }
    plan = ExecutableMissionPlan(
        revision=3,
        uuv_batches_by_carrier={"carrier_02": (batch,)},
        region_assignments=(
            RegionMissionState(
                region_id=candidate_id,
                target_id="target_00",
                lifecycle=RegionLifecycle.PLANNED,
                active_scan_uuv_ids=("uuv_00",),
                passive_track_uuv_ids=("uuv_02",),
                region_polygon=(
                    (-2000.0, -6000.0),
                    (-2000.0, -4000.0),
                    (0.0, -4000.0),
                    (0.0, -6000.0),
                ),
            ),
        ),
        carrier_missions=missions,
        resource_episode_by_uuv=engine._mission_resource_episodes(),
    )

    assert engine.apply_verified_mission_plan(plan) is True, engine._last_mission_plan_failure_reason
    assert engine._mission_plan is not None
    assert candidate_id not in {batch.candidate_id for batch in engine._mission_plan.batches}
    assignment = engine._mission_plan.assignments_by_candidate[candidate_id]
    assert assignment.lifecycle is RegionLifecycle.UNCOVERED
    assert "carrier_route_infeasible" in assignment.degraded_reasons
    assert any(
        event.event_type == "carrier_plan_degraded"
        and event.payload.get("candidate_id") == candidate_id
        for event in engine.events()
    )
    degraded_event_count = sum(
        event.event_type == "carrier_plan_degraded"
        and event.payload.get("candidate_id") == candidate_id
        for event in engine.events()
    )

    # The planner may reissue the same candidate before its immutable service
    # window closes.  Route feedback must withdraw it before another A* run.
    assert engine.apply_verified_mission_plan(plan) is True, engine._last_mission_plan_failure_reason
    assert sum(
        event.event_type == "carrier_plan_degraded"
        and event.payload.get("candidate_id") == candidate_id
        for event in engine.events()
    ) == degraded_event_count


def test_active_carrier_route_survives_expired_window_in_new_plan() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)

    first_plan = _carrier_plan(config, exit_s=600)
    assert engine.apply_verified_mission_plan(first_plan) is True
    carrier = engine._carrier_entities["carrier_02"]
    original_route = carrier.mission_route_xy
    assert original_route
    assert carrier.remaining_committed_stops()

    engine._clock.sim_time_s = 900
    engine._deployment_states["uuv_00"] = DeploymentState.DEPLOYED
    engine._waterborne_uuv_ids.add("uuv_00")
    updated_missions = dict(first_plan.carrier_missions)
    updated_missions["carrier_02"] = updated_missions["carrier_02"].model_copy(
        update={
            "ready_uuv_ids": (),
            "route_xy": (),
            "stop_ids": (),
            "stop_indices": (),
            "stop_windows": (),
        }
    )
    second_plan = first_plan.model_copy(
        update={
            "revision": 2,
            "carrier_missions": updated_missions,
        }
    )

    assert engine.apply_verified_mission_plan(second_plan) is True
    assert engine._last_mission_plan_failure_reason is None
    assert carrier.mission_route_xy == original_route
    assert carrier.remaining_committed_stops()


def test_route_error_skips_active_batch_and_degrades_later_unstarted_batch() -> None:
    """A multi-task route error must preserve waterborne members."""
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)

    base_plan = _carrier_plan(config)
    current_batch = base_plan.batches[0]
    future_id = "target_00:r1"
    future_batch = current_batch.model_copy(
        update={
            "candidate_id": future_id,
            "uuv_ids": ("uuv_01",),
            "active_scan_uuv_ids": ("uuv_01",),
        }
    )
    future_assignment = base_plan.region_assignments[0].model_copy(
        update={
            "region_id": future_id,
            "active_scan_uuv_ids": ("uuv_01",),
        }
    )
    plan = base_plan.model_copy(
        update={
            "uuv_batches_by_carrier": {
                "carrier_02": (current_batch, future_batch),
            },
            "region_assignments": (
                base_plan.region_assignments[0],
                future_assignment,
            ),
        }
    )
    engine._deployment_states["uuv_00"] = DeploymentState.DEPLOYED

    degraded = engine._degrade_infeasible_unstarted_batch(
        plan,
        carrier_id="carrier_02",
        error=ValueError(
            "carrier service tasks are infeasible: "
            "['deploy:target_00:r0', 'deploy:target_00:r1']"
        ),
    )

    assert degraded is not None
    candidate_id, adjusted_plan = degraded
    assert candidate_id == future_id
    assert {batch.candidate_id for batch in adjusted_plan.batches} == {
        current_batch.candidate_id
    }


def test_uuv_only_rejects_uuv_logistics_assigned_to_the_carrier_hull() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    plan = _carrier_plan(config)
    batch = plan.batches[0].model_copy(update={"carrier_id": "carrier_01"})
    carrier_missions = {
        carrier_id: mission.model_copy(
            update={
                "ready_uuv_ids": ("uuv_00",)
                if carrier_id == "carrier_01"
                else (),
            }
        )
        for carrier_id, mission in plan.carrier_missions.items()
    }
    invalid = plan.model_copy(
        update={
            "uuv_batches_by_carrier": {"carrier_01": (batch,)},
            "carrier_missions": carrier_missions,
        }
    )

    assert engine.apply_verified_mission_plan(invalid) is False


def test_uuv_only_rejects_cross_mother_ship_reassignment() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    assert config.environment is not None
    mother = next(
        carrier
        for carrier in config.environment.carriers
        if carrier.platform_id == "carrier_03"
    )
    x, y = mother.position_xy
    batch = _carrier_plan(config).batches[0].model_copy(
        update={
            "carrier_id": "carrier_03",
            "deployment_point": (x + 100.0, y),
            "recovery_point": (x + 200.0, y),
        }
    )
    carrier_missions = {
        carrier.platform_id: CarrierMissionModel(
            carrier_id=carrier.platform_id,
            home_battle_group_id="carrier_battle_group_01",
            ready_uuv_ids=("uuv_00",) if carrier.platform_id == "carrier_03" else (),
            route_xy=(
                (x, y),
                (x + 100.0, y),
                (x + 200.0, y),
                (x, y),
            )
            if carrier.platform_id == "carrier_03"
            else (),
            stop_ids=("deploy:target_00:r0", "recover:target_00:r0")
            if carrier.platform_id == "carrier_03"
            else (),
        )
        for carrier in (config.environment.carrier, *config.environment.carriers)
        if carrier.platform_id
    }
    invalid = _carrier_plan(config).model_copy(
        update={
            "uuv_batches_by_carrier": {"carrier_03": (batch,)},
            "carrier_missions": carrier_missions,
        }
    )

    assert engine.apply_verified_mission_plan(invalid) is False
    assert engine._uuv_carrier_ids["uuv_00"] == "carrier_02"


def test_normal_mode_routes_active_uuvs_and_emits_region_scan_ping() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    assert config.environment is not None
    mother = next(
        carrier
        for carrier in config.environment.carriers
        if carrier.platform_id == "carrier_02"
    )
    x, y = mother.position_xy
    plan = _carrier_plan(config)
    region = plan.region_assignments[0].model_copy(
        update={
            "region_polygon": ((x + 90.0, y - 50.0), (x + 250.0, y - 50.0),
                               (x + 250.0, y + 50.0), (x + 90.0, y + 50.0)),
            "scan_waypoints": ((x + 100.0, y - 40.0), (x + 240.0, y - 40.0)),
            "scan_waypoints_by_uuv": {
                "uuv_00": ((x + 100.0, y - 40.0), (x + 240.0, y - 40.0)),
            },
        }
    )
    plan = plan.model_copy(update={"region_assignments": (region,)})
    assert engine.apply_verified_mission_plan(plan) is True

    engine._deployment_states["uuv_00"] = DeploymentState.DEPLOYED
    engine._uuvs["uuv_00"].position_xy = (x + 100.0, y - 40.0)
    controller.advance(
        0,
        {"deployed_uuv_ids": {region.region_id: ("uuv_00",)}},
    )
    engine._reconcile_uuv_mission_state()
    engine._contact_state["target_00"]["position_xy"] = (x + 500.0, y - 40.0)
    engine._plan_mission_waypoints(controller.snapshot())

    assert engine._sensor_modes["uuv_00"] == "active"
    assert engine._ping_targets["uuv_00"] == "target_00"
    assert tuple(engine._uuvs["uuv_00"].waypoints) == region.scan_waypoints_by_uuv["uuv_00"]

    engine._process_pings(30)
    assert any(
        event.event_type == "active_ping"
        and event.entity_id == "target_00"
        and event.payload.get("uuv_id") == "uuv_00"
        for event in engine._events
    )


def test_normal_mode_routes_all_region_members_before_target_entry() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    assert config.environment is not None
    mother = next(
        carrier
        for carrier in config.environment.carriers
        if carrier.platform_id == "carrier_02"
    )
    x, y = mother.position_xy
    plan = _carrier_plan(config)
    batch = plan.batches[0].model_copy(
        update={
            "uuv_ids": ("uuv_00", "uuv_03"),
            "passive_track_uuv_ids": ("uuv_03",),
        }
    )
    region = plan.region_assignments[0].model_copy(
        update={
            "passive_track_uuv_ids": ("uuv_03",),
            "scan_waypoints": (
                (x + 100.0, y - 40.0),
                (x + 240.0, y - 40.0),
            ),
            "scan_waypoints_by_uuv": {
                "uuv_00": ((x + 100.0, y - 40.0), (x + 240.0, y - 40.0)),
                "uuv_03": ((x + 100.0, y + 40.0), (x + 240.0, y + 40.0)),
            },
        }
    )
    carrier_mission = plan.carrier_missions["carrier_02"].model_copy(
        update={"ready_uuv_ids": ("uuv_00", "uuv_03")}
    )
    plan = plan.model_copy(
        update={
            "uuv_batches_by_carrier": {"carrier_02": (batch,)},
            "region_assignments": (region,),
            "carrier_missions": {
                **plan.carrier_missions,
                "carrier_02": carrier_mission,
            },
        }
    )
    assert engine.apply_verified_mission_plan(plan) is True

    for uuv_id, point in {
        "uuv_00": (x + 100.0, y - 40.0),
        "uuv_03": (x + 100.0, y + 40.0),
    }.items():
        engine._deployment_states[uuv_id] = DeploymentState.DEPLOYED
        engine._uuvs[uuv_id].position_xy = point
    controller.advance(
        0,
        {"deployed_uuv_ids": {region.region_id: ("uuv_00", "uuv_03")}},
    )
    engine._reconcile_uuv_mission_state()
    engine._plan_mission_waypoints(controller.snapshot())

    assert controller.snapshot().uuv_modes["uuv_00"] is UUVMissionMode.ACTIVE_SCAN
    assert controller.snapshot().uuv_modes["uuv_03"] is UUVMissionMode.PASSIVE_TRACK
    assert tuple(engine._uuvs["uuv_03"].waypoints) == (
        region.scan_waypoints_by_uuv["uuv_03"][0],
    )
    assert engine._sensor_modes["uuv_03"] == "passive"


def test_region_entry_uses_public_belief_mass_and_omits_invalid_mass() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    config = config.model_copy(
        update={
            "scenario": config.scenario.model_copy(
                update={"region_entry_buffer_m": 0.0}
            )
        }
    )
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    plan = _carrier_plan(config)
    region = plan.region_assignments[0].model_copy(
        update={
            "region_polygon": (
                (-1.0, -1.0),
                (1.0, -1.0),
                (1.0, 1.0),
                (-1.0, 1.0),
            )
        }
    )
    plan = plan.model_copy(update={"region_assignments": (region,)})
    assert engine.apply_verified_mission_plan(plan) is True
    engine.request_uuv_deployment("uuv_00", reason="test_observation")
    engine.activate_execution_group(
        target_id="target_00",
        region_id=region.region_id,
        member_ids=("uuv_00",),
    )
    engine._fuse_execution_group_observations(
        0,
        (
            PassiveSonarObservation(
                observation_id="entry-test-uuv-00",
                scenario_id=config.scenario.scenario_id,
                sim_time_s=0,
                observer_id="uuv_00",
                target_id="target_00",
                azimuth_rad=0.1,
                variance_rad2=0.01,
                detection_confidence=0.9,
                snr_db=8.0,
            ),
        ),
    )
    snapshot = controller.snapshot().model_copy(update={"regions": (region,)})
    report = engine._latest_reports["target_00"]
    engine._latest_reports["target_00"] = report.model_copy(
        update={
            "belief": report.belief.model_copy(
                update={
                    "mean": (0.0, 0.0),
                    "covariance": ((1.0, 0.0), (0.0, 1.0)),
                }
            )
        }
    )

    probabilities = engine._mission_entry_probabilities(0, snapshot)

    probability = probabilities[region.region_id]
    assert 0.45 < probability < 0.55
    assert probability not in (0.0, 1.0)

    engine._latest_reports["target_00"] = report.model_copy(
        update={
            "belief": report.belief.model_copy(
                update={
                    "mean": (0.0, 0.0),
                    "covariance": ((1.0, 0.2), (0.0, 1.0)),
                }
            )
        }
    )

    assert region.region_id not in engine._mission_entry_probabilities(0, snapshot)


def test_handoff_evidence_joins_only_current_successor_passive_observations() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(
        scenario_id=config.scenario.scenario_id,
        group_min_size=2,
    )
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    predecessor = RegionMissionState(
        region_id="R1",
        target_id="target_00",
        lifecycle=RegionLifecycle.HANDOFF_PENDING,
        passive_track_uuv_ids=("uuv_00",),
        handoff_to="R2",
    )
    successor = RegionMissionState(
        region_id="R2",
        target_id="target_00",
        lifecycle=RegionLifecycle.PASSIVE_TRACK,
        passive_track_uuv_ids=("uuv_01", "uuv_02"),
        handoff_from="R1",
    )
    snapshot = controller.snapshot().model_copy(
        update={
            "plan_revision": 1,
            "regions": (predecessor, successor),
            "uuv_modes": {
                "uuv_01": UUVMissionMode.PASSIVE_TRACK,
                "uuv_02": UUVMissionMode.PASSIVE_TRACK,
            },
            "uuv_resources": {
                uuv_id: UUVResourceState(
                    uuv_id=uuv_id,
                    carrier_id="carrier_02",
                    mileage_m=100.0,
                    energy_fraction=0.8,
                    healthy=True,
                    capability_active=True,
                    deployment_state="PASSIVE_TRACK",
                )
                for uuv_id in ("uuv_01", "uuv_02")
            },
        }
    )
    for uuv_id in ("uuv_01", "uuv_02"):
        engine._deployment_states[uuv_id] = DeploymentState.DEPLOYED
        engine._waterborne_uuv_ids.add(uuv_id)
    engine.activate_execution_group(
        target_id="target_00",
        region_id="R2",
        member_ids=("uuv_01", "uuv_02"),
    )
    engine._fuse_execution_group_observations(
        60,
        (
            PassiveSonarObservation(
                observation_id="bootstrap-uuv-01",
                scenario_id=config.scenario.scenario_id,
                sim_time_s=60,
                observer_id="uuv_01",
                target_id="target_00",
                azimuth_rad=0.1,
                variance_rad2=0.01,
                detection_confidence=0.9,
                snr_db=8.0,
            ),
        ),
    )
    report = engine._latest_reports["target_00"]
    source_ids = (
        "accepted-uuv-01",
        "accepted-uuv-02",
        "stale-uuv-01",
        "foreign-uuv-03",
        "false-uuv-02",
    )
    engine._latest_reports["target_00"] = report.model_copy(
        update={
            "belief": report.belief.model_copy(
                update={"source_observation_ids": source_ids}
            )
        }
    )
    engine._target_rays["target_00"] = (
        BearingObservation(
            observation_id="accepted-uuv-01",
            scenario_id=config.scenario.scenario_id,
            sim_time_s=60,
            uuv_id="uuv_01",
            target_id="target_00",
            azimuth_rad=0.1,
            variance_rad2=0.01,
            detection_confidence=0.9,
        ),
        BearingObservation(
            observation_id="accepted-uuv-02",
            scenario_id=config.scenario.scenario_id,
            sim_time_s=60,
            uuv_id="uuv_02",
            target_id="target_00",
            azimuth_rad=0.2,
            variance_rad2=0.01,
            detection_confidence=0.9,
        ),
        BearingObservation(
            observation_id="stale-uuv-01",
            scenario_id=config.scenario.scenario_id,
            sim_time_s=30,
            uuv_id="uuv_01",
            target_id="target_00",
            azimuth_rad=0.3,
            variance_rad2=0.01,
            detection_confidence=0.9,
        ),
        BearingObservation(
            observation_id="foreign-uuv-03",
            scenario_id=config.scenario.scenario_id,
            sim_time_s=60,
            uuv_id="uuv_03",
            target_id="target_00",
            azimuth_rad=0.4,
            variance_rad2=0.01,
            detection_confidence=0.9,
        ),
        BearingObservation(
            observation_id="false-uuv-02",
            scenario_id=config.scenario.scenario_id,
            sim_time_s=60,
            uuv_id="uuv_02",
            target_id="target_00",
            azimuth_rad=0.5,
            variance_rad2=0.01,
            detection_confidence=0.9,
            is_false_alarm=True,
        ),
    )

    evidence = engine._mission_handoff_evidence(
        snapshot,
        engine._latest_reports,
        60,
    )["R1"]

    assert tuple(
        observation.observation_id for observation in evidence.accepted_observations
    ) == ("accepted-uuv-01", "accepted-uuv-02")
    assert evidence.required_uuv_ids == ("uuv_01", "uuv_02")
    assert evidence.deployed_uuv_ids == ("uuv_01", "uuv_02")
    assert evidence.healthy_uuv_ids == ("uuv_01", "uuv_02")
    assert evidence.passive_mode_uuv_ids == ("uuv_01", "uuv_02")


def test_handoff_evidence_does_not_expire_an_operational_successor() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(
        scenario_id=config.scenario.scenario_id,
        group_min_size=2,
    )
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    predecessor = RegionMissionState(
        region_id="R1",
        target_id="target_00",
        lifecycle=RegionLifecycle.HANDOFF_PENDING,
        passive_track_uuv_ids=("uuv_00",),
        handoff_to="R2",
    )
    successor = RegionMissionState(
        region_id="R2",
        target_id="target_00",
        lifecycle=RegionLifecycle.PASSIVE_TRACK,
        passive_track_uuv_ids=("uuv_01", "uuv_02"),
        handoff_from="R1",
    )
    snapshot = controller.snapshot().model_copy(
        update={
            "sim_time_s": 60,
            "plan_revision": 1,
            "regions": (predecessor, successor),
            "uuv_modes": {
                "uuv_01": UUVMissionMode.PASSIVE_TRACK,
                "uuv_02": UUVMissionMode.PASSIVE_TRACK,
            },
            "uuv_resources": {
                uuv_id: UUVResourceState(
                    uuv_id=uuv_id,
                    carrier_id="carrier_02",
                    mileage_m=100.0,
                    energy_fraction=0.8,
                    healthy=True,
                    capability_active=True,
                    deployment_state="deployed",
                )
                for uuv_id in ("uuv_01", "uuv_02")
            },
        }
    )
    engine._mission_plan = ExecutableMissionPlan(
        revision=1,
        uuv_batches_by_carrier={
            "carrier_02": (
                UUVMissionBatch(
                    carrier_id="carrier_02",
                    candidate_id="R2",
                    uuv_ids=("uuv_01", "uuv_02"),
                    passive_track_uuv_ids=("uuv_01", "uuv_02"),
                    entry_s=0,
                    exit_s=30,
                ),
            )
        },
    )
    for uuv_id in ("uuv_01", "uuv_02"):
        engine._deployment_states[uuv_id] = DeploymentState.DEPLOYED
        engine._waterborne_uuv_ids.add(uuv_id)
    source_ids = ("successor-uuv-01", "successor-uuv-02")
    engine._latest_reports["target_00"] = GroupReport(
        group_id="G-R2",
        target_id="target_00",
        sim_time_s=60,
        member_ids=("uuv_01", "uuv_02"),
        belief=TargetBelief(
            target_id="target_00",
            sim_time_s=60,
            mean=(0.0, 0.0, 1.0, 0.0, 0.0),
            covariance=(
                (100.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 100.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 1.0),
            ),
            model_probabilities={"cv": 1.0},
            source_observation_ids=source_ids,
            fim_min_eigenvalue=0.1,
            fim_condition=2.0,
        ),
        quality=GroupQuality(
            instant=0.8,
            window_mean=0.8,
            ewma=0.8,
            components={"fim": 0.8},
        ),
        plan_revision=1,
    )
    engine._target_rays["target_00"] = tuple(
        BearingObservation(
            observation_id=observation_id,
            scenario_id=config.scenario.scenario_id,
            sim_time_s=60,
            uuv_id=uuv_id,
            target_id="target_00",
            azimuth_rad=0.1 + index,
            variance_rad2=0.01,
            detection_confidence=0.9,
        )
        for index, (observation_id, uuv_id) in enumerate(
            zip(source_ids, ("uuv_01", "uuv_02"), strict=True)
        )
    )

    evidence = engine._mission_handoff_evidence(
        snapshot,
        engine._latest_reports,
        60,
    )["R1"]

    assert evidence.blocked_reason is None
    assert evidence.is_complete(group_min_size=2)

    engine._carrier_route_status_overrides["carrier_02"] = CarrierRouteStatus.RECOVERING
    engine._latest_reports["target_00"] = engine._latest_reports[
        "target_00"
    ].model_copy(
        update={
            "belief": engine._latest_reports["target_00"].belief.model_copy(
                update={"source_observation_ids": ()}
            )
        }
    )
    waiting_evidence = engine._mission_handoff_evidence(
        snapshot,
        engine._latest_reports,
        60,
    )["R1"]

    assert waiting_evidence.blocked_reason is None

    engine._mission_successor_observation_history.clear()
    unresolved_evidence = engine._mission_handoff_evidence(
        snapshot,
        engine._latest_reports,
        60,
    )["R1"]

    assert unresolved_evidence.blocked_reason is None

    for uuv_id in ("uuv_01", "uuv_02"):
        engine._deployment_states[uuv_id] = DeploymentState.ONBOARD
        engine._waterborne_uuv_ids.discard(uuv_id)
    unavailable_evidence = engine._mission_handoff_evidence(
        snapshot,
        engine._latest_reports,
        60,
    )["R1"]

    assert unavailable_evidence.blocked_reason is None


def test_successor_carrier_recovery_waits_for_pending_predecessor_handoff() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(
        scenario_id=config.scenario.scenario_id,
        group_min_size=2,
    )
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    predecessor = RegionMissionState(
        region_id="R1",
        target_id="target_00",
        lifecycle=RegionLifecycle.HANDOFF_PENDING,
        passive_track_uuv_ids=("uuv_00", "uuv_02"),
        handoff_to="R2",
    )
    successor = RegionMissionState(
        region_id="R2",
        target_id="target_00",
        lifecycle=RegionLifecycle.PASSIVE_TRACK,
        passive_track_uuv_ids=("uuv_04", "uuv_06"),
        handoff_from="R1",
    )
    controller._plan_revision = 1
    controller._regions = {
        predecessor.region_id: predecessor,
        successor.region_id: successor,
    }
    engine._mission_plan = ExecutableMissionPlan(
        revision=1,
        region_assignments=(predecessor, successor),
    )

    assert engine._mission_recovery_waits_for_handoff("R2") is True


def test_completed_predecessor_recovery_waits_for_resource_rotation() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    predecessor = RegionMissionState(
        region_id="R1",
        target_id="target_00",
        lifecycle=RegionLifecycle.TRACKING_COMPLETED,
        passive_track_uuv_ids=("uuv_00", "uuv_02"),
        handoff_to="R2",
    )
    successor = RegionMissionState(
        region_id="R2",
        target_id="target_00",
        lifecycle=RegionLifecycle.PASSIVE_TRACK,
        passive_track_uuv_ids=("uuv_04",),
        handoff_from="R1",
    )
    controller._plan_revision = 1
    controller._regions = {
        predecessor.region_id: predecessor,
        successor.region_id: successor,
    }
    controller._uuv_modes.update(
        {
            "uuv_00": UUVMissionMode.PASSIVE_TRACK,
            "uuv_02": UUVMissionMode.PASSIVE_TRACK,
        }
    )
    engine._mission_plan = ExecutableMissionPlan(
        revision=1,
        uuv_batches_by_carrier={
            "carrier_02": (
                UUVMissionBatch(
                    carrier_id="carrier_02",
                    candidate_id="R1",
                    uuv_ids=("uuv_00", "uuv_02"),
                    active_scan_uuv_ids=(),
                    passive_track_uuv_ids=("uuv_00", "uuv_02"),
                    entry_s=0,
                    exit_s=120,
                ),
            )
        },
        region_assignments=(predecessor, successor),
    )
    engine._mission_batch_by_candidate["carrier_02", "R1"] = ("uuv_00", "uuv_02")
    engine._deployment_states["uuv_00"] = DeploymentState.DEPLOYED
    engine._deployment_states["uuv_02"] = DeploymentState.DEPLOYED

    engine._request_mission_recovery_if_ready(
        "carrier_02",
        "R1",
        "recover:R1",
    )

    assert engine._deployment_states["uuv_00"] is DeploymentState.DEPLOYED
    assert engine._deployment_states["uuv_02"] is DeploymentState.DEPLOYED
    assert engine._mission_recovery_waits_for_handoff("R1") is True

    controller._uuv_modes["uuv_00"] = UUVMissionMode.RETURN_REQUIRED
    controller._regions["R1"] = predecessor.model_copy(
        update={"lifecycle": RegionLifecycle.CARRIER_RECOVERY}
    )
    assert engine._mission_recovery_waits_for_handoff("R1") is True

    controller._uuv_modes["uuv_02"] = UUVMissionMode.RETURN_REQUIRED
    assert engine._mission_recovery_waits_for_handoff("R1") is False
    engine._request_mission_recovery_if_ready(
        "carrier_02",
        "R1",
        "recover:R1",
    )
    assert engine._deployment_states["uuv_00"] is DeploymentState.RETURNING
    assert engine._deployment_states["uuv_02"] is DeploymentState.RETURNING


def test_unconverged_mission_group_restores_auditable_spread() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    for uuv_id in ("uuv_00", "uuv_02"):
        engine._deployment_states[uuv_id] = DeploymentState.DEPLOYED
        engine._waterborne_uuv_ids.add(uuv_id)
        engine._uuvs[uuv_id].position_xy = (0.0, 0.0)
    region = RegionMissionState(
        region_id="R1",
        target_id="target_00",
        lifecycle=RegionLifecycle.ACTIVE_SCAN,
        active_scan_uuv_ids=("uuv_00",),
        passive_track_uuv_ids=("uuv_02",),
        region_polygon=((-2000.0, -2000.0), (2000.0, -2000.0), (2000.0, 2000.0), (-2000.0, 2000.0)),
    )
    snapshot = controller.snapshot().model_copy(
        update={
            "sim_time_s": 0,
            "regions": (region,),
            "uuv_modes": {
                "uuv_00": UUVMissionMode.ACTIVE_SCAN,
                "uuv_02": UUVMissionMode.PASSIVE_TRACK,
            },
        }
    )

    routes = engine._plan_mission_group_waypoints(
        snapshot,
        region,
        ("uuv_00", "uuv_02"),
    )

    assert routes["uuv_00"][-1] == (900.0, 0.0)
    assert abs(routes["uuv_02"][-1][0] + 900.0) < 1e-9
    assert abs(routes["uuv_02"][-1][1]) < 1e-9


def test_transient_resource_episode_change_is_retryable_during_recovery() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    plan = _carrier_plan(config).model_copy(
        update={"resource_episode_by_uuv": {"uuv_00": 0}}
    )

    controller._resource_episode_by_uuv["uuv_00"] = 1
    engine._deployment_states["uuv_00"] = DeploymentState.RETURNING
    engine._last_mission_plan_failure_reason = "resource_episode_stale"

    assert engine.mission_plan_application_is_retryable(plan)

    engine._deployment_states["uuv_00"] = DeploymentState.FAILED
    assert not engine.mission_plan_application_is_retryable(plan)


def test_deferred_resource_plan_application_emits_auditable_replan_event() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    plan = _carrier_plan(config, revision=4)

    engine.defer_mission_plan_application(plan, "resource_episode_stale")
    engine.defer_mission_plan_application(plan, "resource_episode_stale")

    events = [
        event
        for event in engine.events()
        if event.event_type == "carrier_plan_degraded"
    ]
    assert len(events) == 1
    assert events[0].payload["plan_revision"] == 4
    assert events[0].payload["plan_impact"] is True
    assert events[0].payload["reason"] == "resource_state_changed"


def test_normal_passive_tracking_commands_remain_inside_task_region() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    assert config.environment is not None
    mother = next(
        carrier
        for carrier in config.environment.carriers
        if carrier.platform_id == "carrier_02"
    )
    x, y = mother.position_xy
    plan = _carrier_plan(config)
    region = plan.region_assignments[0].model_copy(
        update={
            "lifecycle": RegionLifecycle.PASSIVE_TRACK,
            "region_polygon": (
                (x + 90.0, y - 50.0),
                (x + 250.0, y - 50.0),
                (x + 250.0, y + 50.0),
                (x + 90.0, y + 50.0),
            ),
            "passive_track_uuv_ids": ("uuv_00",),
            "active_scan_uuv_ids": (),
        }
    )
    plan = plan.model_copy(update={"region_assignments": (region,)})
    assert engine.apply_verified_mission_plan(plan) is True
    engine._deployment_states["uuv_00"] = DeploymentState.DEPLOYED
    snapshot = controller.snapshot().model_copy(
        update={
            "regions": (region,),
            "uuv_modes": {"uuv_00": UUVMissionMode.PASSIVE_TRACK},
        }
    )

    engine._plan_mission_waypoints(snapshot)

    waypoint = engine._uuvs["uuv_00"].waypoints[-1]
    assert 90.0 <= waypoint[0] - x <= 250.0
    assert -50.0 <= waypoint[1] - y <= 50.0


def test_engine_dedicated_uuv_returns_to_region_without_carrier_recovery() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(
        scenario_id=config.scenario.scenario_id,
        max_uuv_mileage_m=1_000.0,
    )
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    assert config.environment is not None
    mother = next(
        carrier
        for carrier in config.environment.carriers
        if carrier.platform_id == "carrier_02"
    )
    x, y = mother.position_xy
    plan = _carrier_plan(config)
    region = plan.region_assignments[0].model_copy(
        update={
            "region_polygon": ((x + 90.0, y - 50.0), (x + 250.0, y - 50.0),
                               (x + 250.0, y + 50.0), (x + 90.0, y + 50.0)),
            "scan_waypoints": ((x + 100.0, y - 40.0), (x + 240.0, y - 40.0)),
            "scan_waypoints_by_uuv": {
                "uuv_00": ((x + 100.0, y - 40.0), (x + 240.0, y - 40.0)),
            },
        }
    )
    plan = plan.model_copy(update={"region_assignments": (region,)})
    assert engine.apply_verified_mission_plan(plan) is True
    engine.set_reservations({"target_00": ("uuv_00",)})
    engine._deployment_states["uuv_00"] = DeploymentState.DEPLOYED
    engine._uuvs["uuv_00"].position_xy = (x + 100.0, y - 40.0)
    controller.advance(
        0,
        {"deployed_uuv_ids": {region.region_id: ("uuv_00",)}},
    )
    engine._reconcile_uuv_mission_state()

    engine._mission_distance_m["uuv_00"] = 1_001.0
    engine._advance_mission_controller(30)
    assert controller.snapshot().uuv_modes["uuv_00"] is UUVMissionMode.RETURN_TO_REGION
    engine._plan_mission_waypoints(controller.snapshot())
    assert tuple(engine._uuvs["uuv_00"].waypoints) == region.scan_waypoints_by_uuv["uuv_00"]

    engine._uuvs["uuv_00"].position_xy = (x + 120.0, y - 20.0)
    engine._advance_mission_controller(60)
    snapshot = controller.snapshot()
    assert snapshot.uuv_modes["uuv_00"] is UUVMissionMode.ACTIVE_SCAN
    assert "uuv_00" not in snapshot.dedicated_target_by_uuv
    assert engine._mission_distance_m["uuv_00"] == 0.0
    assert snapshot.carrier_missions["carrier_02"].recoverable_uuv_ids == ()
