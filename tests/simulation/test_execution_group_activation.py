from __future__ import annotations

from math import atan2

import pytest
from tests.public_contract_fixtures import prior_bearing

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import PlanCommand, Waypoint
from underwater_tracking.domain.models import (
    DeploymentState,
    GroupQuality,
    GroupReport,
    TargetBelief,
)
from underwater_tracking.domain.observations import PassiveSonarObservation
from underwater_tracking.domain.execution_models import (
    GroupSensorMode,
    TaskGroupInstance,
    TaskGroupLifecycle,
    TrackingControlState,
)
from underwater_tracking.domain.mission_models import RegionLifecycle, RegionMissionState
from underwater_tracking.config.models import TrackingPolicyConfig
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.runtime.mission_controller import (
    execution_snapshot_to_mission_plan,
)
from tests.domain.test_execution_models import (
    _instance as execution_group_instance,
    _snapshot as execution_snapshot,
)
from underwater_tracking.simulation.engine import SimulationEngine


CONFIG_PATH = "configs/scenario/uuv_only_single_target.yaml"


def _runtime_execution_snapshot():
    base = execution_snapshot()
    groups = tuple(
        TaskGroupInstance(
            group_instance_id=(f"{base.scenario_id}:{region.region_id}:deploy:000009"),
            target_id=base.target_id,
            region_id=region.region_id,
            deployment_revision=9,
            member_uuv_ids=tuple(f"uuv_{(index * 3) + member:02d}" for member in range(3)),
            lifecycle=TaskGroupLifecycle.ENTERING,
            sensor_mode=GroupSensorMode.ACTIVE,
            ownership_status="candidate",
            reason="initial_deployment",
            evidence_ids=(f"runtime-group:{index + 1}",),
        )
        for index, region in enumerate(base.regions)
    )
    regions = tuple(
        region.model_copy(update={"task_group_id": group.group_instance_id})
        for region, group in zip(base.regions, groups, strict=True)
    )
    return base.model_copy(
        deep=True,
        update={
            "scenario_id": "uuv-only-single-target",
            "regions": regions,
            "task_groups": groups,
            "reserve_uuvs": (),
            "tracking_policy": TrackingPolicyConfig(),
            "tracking_control": TrackingControlState(mode="regional"),
        },
    )


def test_runtime_passive_routes_are_scoped_to_each_execution_region() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=42)
    base = _runtime_execution_snapshot()
    groups = tuple(
        group.model_copy(
            update={
                "lifecycle": TaskGroupLifecycle.PASSIVE_TRACK,
                "sensor_mode": GroupSensorMode.PASSIVE,
            }
        )
        for group in base.task_groups[:2]
    )
    report = GroupReport(
        group_id="target_00:execution:owner",
        target_id="target_00",
        sim_time_s=120,
        member_ids=groups[0].member_uuv_ids,
        belief=TargetBelief(
            target_id="target_00",
            sim_time_s=120,
            mean=(-5_000.0, -5_000.0, 0.0, 0.0, 0.0),
            covariance=tuple(
                tuple(100.0 if row == column else 0.0 for column in range(5))
                for row in range(5)
            ),
            model_probabilities={"CV": 1.0},
            source_observation_ids=("passive:target_00:120",),
            track_revision=3,
        ),
        quality=GroupQuality(
            instant=1.0,
            window_mean=1.0,
            ewma=1.0,
            components={"bearing": 1.0},
        ),
        plan_revision=1,
    )
    for group in groups:
        for index, member_id in enumerate(group.member_uuv_ids):
            engine._uuvs[member_id].position_xy = (800.0, (index - 1) * 350.0)

    def region_for(
        group: TaskGroupInstance,
        x_min: float,
        x_max: float,
    ) -> RegionMissionState:
        return RegionMissionState(
            region_id=group.region_id,
            target_id=group.target_id,
            task_group_id=group.group_instance_id,
            lifecycle=RegionLifecycle.PASSIVE_TRACK,
            passive_track_uuv_ids=group.member_uuv_ids,
            tracking_quality=1.0,
            region_polygon=(
                (x_min, -1_000.0),
                (x_max, -1_000.0),
                (x_max, 1_000.0),
                (x_min, 1_000.0),
            ),
        )

    first_region = region_for(groups[0], -1_000.0, 1_000.0)
    second_region = region_for(groups[1], 900.0, 2_900.0)
    first_routes = engine._plan_runtime_passive_routes(
        groups[0], groups[0].member_uuv_ids, report, region=first_region
    )
    second_routes = engine._plan_runtime_passive_routes(
        groups[1], groups[1].member_uuv_ids, report, region=second_region
    )

    assert first_routes
    assert second_routes
    assert all(-1_000.0 <= point[0] <= 1_000.0 for route in first_routes.values() for point in route)
    assert all(900.0 <= point[0] <= 2_900.0 for route in second_routes.values() for point in route)
    assert min(
        point[0] for route in second_routes.values() for point in route
    ) > min(point[0] for route in first_routes.values() for point in route)


def test_uuv_only_initialization_has_public_prior_but_no_execution_group() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)

    snapshot = engine.publication_situation()

    assert snapshot.group_reports == ()
    assert snapshot.execution_groups == ()
    assert len(snapshot.target_search_priors) == 1
    assert snapshot.target_search_priors[0].target_id == "target_00"
    assert engine._assignments == {}
    assert engine._latest_reports == {}
    assert engine.build_slave_contexts(snapshot) == ()
    adversary_inputs = engine.build_adversary_inputs(snapshot)
    assert len(adversary_inputs) == 1
    assert adversary_inputs[0].local_contacts == ()
    assert adversary_inputs[0].platform_threats == ()


def test_runtime_snapshot_materializes_three_uuv_groups_and_boundary_exit() -> None:
    config = load_app_config(CONFIG_PATH)
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    snapshot = _runtime_execution_snapshot()
    engine._clock.sim_time_s = int(snapshot.valid_from_s)

    assert engine.apply_verified_execution_snapshot(snapshot) is True
    assert len(engine._execution_groups) == 4
    assert all(len(group.member_ids) == 3 for group in engine._execution_groups.values())
    assert len(engine._waterborne_uuv_ids) == 12
    assert all(engine._sensor_modes[uuv_id] == "active" for uuv_id in engine._waterborne_uuv_ids)

    outgoing = next(
        group for group in controller.snapshot().task_groups if group.region_id.endswith(":01")
    )
    region = controller.snapshot().regions[0]
    for member in outgoing.member_uuv_ids:
        engine._begin_uuv_boundary_exit(member, region, reason="test_exit")
        assert engine._uuv_is_physically_exposed(member) is True
        assert engine._deployment_states[member] is DeploymentState.RETURNING
        exit_point = engine._boundary_exit_points[member]
        engine._uuvs[member].position_xy = exit_point
        engine._complete_uuv_boundary_exit(member, sim_time_s=30)

    assert all(not engine._uuv_is_physically_exposed(member) for member in outgoing.member_uuv_ids)
    assert all(
        engine._deployment_states[member] is DeploymentState.ONBOARD
        for member in outgoing.member_uuv_ids
    )


def test_runtime_snapshot_retains_returning_members_of_an_exiting_group() -> None:
    config = load_app_config(CONFIG_PATH)
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    groups = tuple(
        execution_group_instance(
            slot=slot,
            deployment_revision=2 if phase == "entering" else 1,
            lifecycle=(
                TaskGroupLifecycle.PASSIVE_TRACK
                if slot == 1 and phase == "entering"
                else TaskGroupLifecycle.ENTERING
                if phase == "entering"
                else TaskGroupLifecycle.EXITING
            ),
            sensor_mode=(
                GroupSensorMode.PASSIVE
                if slot == 1 and phase == "entering"
                else GroupSensorMode.ACTIVE
            ),
            ownership_status=("owner" if slot == 1 and phase == "entering" else "candidate"),
            source_group_instance_id=(
                f"target_00:task:{slot:02d}:deploy:000001" if phase == "entering" else None
            ),
        )
        for slot in range(1, 5)
        for phase in ("entering", "exiting")
    )
    base = execution_snapshot()
    first = type(base).model_validate(
        base.model_dump(mode="python")
        | {
            "scenario_id": config.scenario.scenario_id,
            "execution_revision": 1,
            "base_execution_revision": None,
            "regions": tuple(
                item.model_copy(update={"execution_revision": 1}) for item in base.regions
            ),
            "task_groups": groups,
            "tracking_control": TrackingControlState(
                mode="regional",
                tracking_owner_group_id=groups[0].group_instance_id,
            ),
        }
    )
    engine._clock.sim_time_s = int(first.valid_from_s)
    assert engine.apply_verified_execution_snapshot(first)

    outgoing = groups[1]
    engine._ensure_runtime_group_entities(outgoing)
    for member_id in outgoing.member_uuv_ids:
        engine._deployment_states[member_id] = DeploymentState.RETURNING

    candidate = first.model_copy(
        deep=True,
        update={
            "execution_revision": 2,
            "base_execution_revision": 1,
            "regions": tuple(
                item.model_copy(update={"execution_revision": 2}) for item in first.regions
            ),
        },
    )

    assert engine.apply_verified_execution_snapshot(candidate) is True, (
        engine._last_mission_plan_failure_reason
    )
    assert all(
        engine._deployment_states[member_id] is DeploymentState.RETURNING
        for member_id in outgoing.member_uuv_ids
    )


def test_engine_rolling_snapshot_preserves_owner_and_creates_replacement_pairs() -> None:
    config = load_app_config(CONFIG_PATH)
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    first = _runtime_execution_snapshot()
    engine._clock.sim_time_s = int(first.valid_from_s)
    assert engine.apply_verified_execution_snapshot(first)

    owner_region_id = first.regions[0].region_id
    controller.advance(
        125,
        {
            "deployed_uuv_ids": {
                group.region_id: group.member_uuv_ids for group in first.task_groups
            },
            "evaluate_entry_observation": False,
        },
    )
    for cycle_s in (130, 140):
        controller.advance(
            cycle_s,
            {"region_entry_probabilities": {owner_region_id: 0.90}},
        )
    previous_owner_id = controller.snapshot().tracking_control.tracking_owner_group_id
    assert previous_owner_id is not None

    next_revision = first.execution_revision + 1
    next_groups = tuple(
        group.model_copy(
            update={
                "group_instance_id": group.group_instance_id.replace(
                    "deploy:000009", "deploy:000010"
                ),
                "deployment_revision": group.deployment_revision + 1,
                "lifecycle": TaskGroupLifecycle.ENTERING,
                "sensor_mode": GroupSensorMode.ACTIVE,
                "ownership_status": "candidate",
            }
        )
        for group in first.task_groups
    )
    next_group_by_region = {group.region_id: group for group in next_groups}
    next_regions = tuple(
        region.model_copy(
            update={
                "execution_revision": next_revision,
                "geometry": tuple((point[0] + 100.0, point[1]) for point in region.geometry),
                "center": (region.center[0] + 100.0, region.center[1]),
                "geometry_revision": region.geometry_revision + 1,
                "task_group_id": next_group_by_region[region.region_id].group_instance_id,
            }
        )
        for region in first.regions
    )
    second = first.model_copy(
        deep=True,
        update={
            "execution_revision": next_revision,
            "base_execution_revision": first.execution_revision,
            "source_sim_time_s": 140,
            "generated_at_s": 140.0,
            "valid_from_s": 140.0,
            "valid_until_s": 1940.0,
            "regions": next_regions,
            "task_groups": next_groups,
        },
    )
    engine._clock.sim_time_s = 140

    assert engine.apply_verified_execution_snapshot(second)
    rolled = controller.snapshot()
    member_ids = tuple(
        member_id for group in rolled.task_groups for member_id in group.member_uuv_ids
    )
    assert rolled.tracking_control.tracking_owner_group_id == previous_owner_id
    assert len(rolled.task_groups) == 8
    assert len(member_ids) == 24
    assert len(set(member_ids)) == 24
    assert len(rolled.replacement_states) == 4


def test_runtime_plan_rejects_a_missing_member_resource_episode() -> None:
    config = load_app_config(CONFIG_PATH)
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    snapshot = _runtime_execution_snapshot()
    engine._clock.sim_time_s = int(snapshot.valid_from_s)
    plan = execution_snapshot_to_mission_plan(snapshot)
    missing_member_id = snapshot.task_groups[0].member_uuv_ids[0]
    incomplete = plan.model_copy(
        update={
            "resource_episode_by_uuv": {
                member_id: episode
                for member_id, episode in plan.resource_episode_by_uuv.items()
                if member_id != missing_member_id
            }
        }
    )

    assert engine.apply_verified_mission_plan(incomplete) is False
    assert engine._last_mission_plan_failure_reason == "resource_episode_missing_uuv"
    assert controller.snapshot().plan_revision == 0


def test_runtime_plan_rolls_back_engine_and_controller_after_late_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config(CONFIG_PATH)
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    first = _runtime_execution_snapshot()
    engine._clock.sim_time_s = int(first.valid_from_s)
    assert engine.apply_verified_execution_snapshot(first)
    before_controller = controller.snapshot()
    before_plan = engine._mission_plan
    before_uuv_ids = set(engine._uuvs)
    before_group_instances = dict(engine._uuv_group_instances)
    engine._verification_monitor = engine._build_verification_monitor()
    before_monitor_ids = set(engine._verification_monitor.limits())

    next_revision = first.execution_revision + 1
    next_groups = tuple(
        group.model_copy(
            update={
                "group_instance_id": group.group_instance_id.replace(
                    "deploy:000009", "deploy:000010"
                ),
                "deployment_revision": group.deployment_revision + 1,
                "member_uuv_ids": tuple(
                    f"{member_id}:generation:10" for member_id in group.member_uuv_ids
                ),
            }
        )
        for group in first.task_groups
    )
    next_groups_by_region = {group.region_id: group for group in next_groups}
    candidate = first.model_copy(
        deep=True,
        update={
            "execution_revision": next_revision,
            "base_execution_revision": first.execution_revision,
            "regions": tuple(
                region.model_copy(
                    update={
                        "execution_revision": next_revision,
                        "geometry_revision": region.geometry_revision + 1,
                        "center": (region.center[0] + 100.0, region.center[1]),
                        "geometry": tuple((x + 100.0, y) for x, y in region.geometry),
                        "task_group_id": next_groups_by_region[region.region_id].group_instance_id,
                    }
                )
                for region in first.regions
            ),
            "task_groups": next_groups,
        },
    )
    engine._clock.sim_time_s = int(candidate.valid_from_s)

    def fail_late() -> None:
        raise RuntimeError("late waypoint failure")

    monkeypatch.setattr(engine, "_plan_waypoints", fail_late)
    with pytest.raises(RuntimeError, match="late waypoint failure"):
        engine.apply_verified_execution_snapshot(candidate)

    assert controller.snapshot() == before_controller
    assert engine._mission_plan == before_plan
    assert set(engine._uuvs) == before_uuv_ids
    assert engine._uuv_group_instances == before_group_instances
    assert set(engine._verification_monitor.limits()) == before_monitor_ids


def test_execution_group_requires_physical_exposure_and_does_not_create_belief() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)

    with pytest.raises(ValueError, match="not physically exposed"):
        engine.activate_execution_group(
            target_id="target_00",
            region_id="region-00",
            member_ids=("uuv_00", "uuv_01"),
        )

    engine.request_uuv_deployment("uuv_00", reason="test")
    engine.request_uuv_deployment("uuv_01", reason="test")
    group = engine.activate_execution_group(
        target_id="target_00",
        region_id="region-00",
        member_ids=("uuv_00", "uuv_01"),
    )

    assert group.member_ids == ("uuv_00", "uuv_01")
    assert group.mode == "active_scan"
    assert engine._latest_reports == {}
    assert engine.publication_situation().group_reports == ()
    assert engine.publication_situation().execution_groups == (group,)

    with pytest.raises(ValueError, match="already belongs"):
        engine.activate_execution_group(
            target_id="target_00",
            region_id="region-01",
            member_ids=("uuv_00", "uuv_01"),
        )


def test_public_prior_expires_without_revealing_known_submarine_position() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    engine._clock.sim_time_s = 1800

    snapshot = engine.publication_situation()
    assert snapshot.target_search_priors == ()
    contact = next(item for item in snapshot.contacts if item.contact_id == "target_00")
    assert contact.classification.value == "submarine"
    assert contact.estimated_position_xy is None


def test_engine_does_not_expose_global_target_history_to_operational_callers() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    assert not hasattr(engine, "global_target_history")
    assert not hasattr(engine, "_global_target_histories")
    initial_contact = next(
        item for item in engine.publication_situation().contacts if item.contact_id == "target_00"
    )
    assert initial_contact.estimated_position_xy is None

    engine.step()

    contact = next(
        item for item in engine.publication_situation().contacts if item.contact_id == "target_00"
    )
    assert contact.estimated_position_xy is None


def test_missing_group_creation_rejects_contact_without_public_position() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    command = PlanCommand(
        command_id="command-no-public-position",
        plan_id="plan-no-public-position",
        plan_revision=1,
        scenario_id=engine._scenario_id,
        group_id="G-target_00",
        target_id="target_00",
        sim_time_s=0,
        member_ids=("uuv_00", "uuv_01"),
    )

    assert engine._contact_state["target_00"]["position_xy"] is None
    assert engine._create_missing_group(command) is None
    assert engine._latest_reports == {}
    assert engine._assignments == {}


def test_plan_command_without_public_group_position_is_side_effect_free() -> None:
    engine = SimulationEngine(
        load_app_config("configs/scenario/segmented_single_target.yaml"),
        seed=7,
    )
    target_id = "target_00"
    uuv_id = "uuv_00"
    engine._manager.complete(target_id)
    engine._latest_reports.pop(target_id)
    engine._assignments.pop(target_id)
    assert engine._contact_state[target_id]["position_xy"] is None
    assert engine._deployment_states[uuv_id] is DeploymentState.ONBOARD
    before = {
        "deployment_states": dict(engine._deployment_states),
        "waterborne": set(engine._waterborne_uuv_ids),
        "uuv_groups": dict(engine._uuv_groups),
        "waypoints": tuple(engine._uuvs[uuv_id].waypoints),
        "pending_events": tuple(engine._pending_runtime_events),
        "pending_commands": dict(engine._pending_group_commands),
        "applied_revisions": dict(engine._applied_plan_revisions),
        "recovery_waypoints": dict(engine._recovery_waypoints),
    }
    command = PlanCommand(
        command_id="command-no-public-position",
        plan_id="plan-no-public-position",
        plan_revision=1,
        scenario_id=engine._scenario_id,
        group_id="G-target_00",
        target_id=target_id,
        sim_time_s=0,
        member_ids=(uuv_id,),
        waypoints_by_member={uuv_id: (Waypoint(x=1200.0, y=300.0),)},
        actions={uuv_id: "track"},
    )

    engine.apply_plan_command(command)

    assert engine._deployment_states == before["deployment_states"]
    assert engine._waterborne_uuv_ids == before["waterborne"]
    assert engine._uuv_groups == before["uuv_groups"]
    assert tuple(engine._uuvs[uuv_id].waypoints) == before["waypoints"]
    assert tuple(engine._pending_runtime_events) == before["pending_events"]
    assert engine._pending_group_commands == before["pending_commands"]
    assert engine._applied_plan_revisions == before["applied_revisions"]
    assert engine._recovery_waypoints == before["recovery_waypoints"]


def test_failed_uuv_cannot_join_execution_group() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    engine.request_uuv_deployment("uuv_00", reason="test")
    engine.request_uuv_deployment("uuv_01", reason="test")
    engine.fail_uuv("uuv_01")

    assert engine._deployment_states["uuv_01"] is DeploymentState.FAILED
    with pytest.raises(ValueError, match="not deployable"):
        engine.activate_execution_group(
            target_id="target_00",
            region_id="region-00",
            member_ids=("uuv_00", "uuv_01"),
        )


def test_real_fused_bearings_create_the_first_tracking_report() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    engine.request_uuv_deployment("uuv_00", reason="test")
    engine.request_uuv_deployment("uuv_01", reason="test")
    group = engine.activate_execution_group(
        target_id="target_00",
        region_id="region-00",
        member_ids=("uuv_00", "uuv_01"),
    )

    observations = tuple(
        PassiveSonarObservation(
            observation_id=f"passive:{uuv_id}:target_00:30",
            scenario_id=engine._scenario_id,
            sim_time_s=30,
            observer_id=uuv_id,
            target_id="target_00",
            azimuth_rad=prior_bearing(engine, uuv_id, 30),
            variance_rad2=0.01,
            detection_confidence=0.9,
            snr_db=8.0,
        )
        for uuv_id in group.member_ids
    )
    engine._fuse_execution_group_observations(30, observations)

    report = engine._latest_reports["target_00"]
    assert report.belief.source_observation_ids == tuple(
        observation.observation_id for observation in observations
    )
    assert engine.publication_situation().group_reports


def test_reused_target_filter_publishes_the_current_execution_group_members() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    for uuv_id in ("uuv_00", "uuv_01", "uuv_02", "uuv_03"):
        engine.request_uuv_deployment(uuv_id, reason="test")

    first = engine.activate_execution_group(
        target_id="target_00",
        region_id="region-00",
        member_ids=("uuv_00", "uuv_01"),
    )
    first_observations = tuple(
        PassiveSonarObservation(
            observation_id=f"first:{uuv_id}",
            scenario_id=engine._scenario_id,
            sim_time_s=30,
            observer_id=uuv_id,
            target_id="target_00",
            azimuth_rad=prior_bearing(engine, uuv_id, 30),
            variance_rad2=0.01,
            detection_confidence=0.9,
            snr_db=8.0,
        )
        for uuv_id in first.member_ids
    )
    engine._fuse_execution_group_observations(30, first_observations)

    second = engine.activate_execution_group(
        target_id="target_00",
        region_id="region-01",
        member_ids=("uuv_02", "uuv_03"),
    )
    second_observations = (
        first_observations[0].model_copy(
            update={
                "observation_id": "second:uuv_02",
                "observer_id": "uuv_02",
                "azimuth_rad": prior_bearing(engine, "uuv_02", 60),
                "sim_time_s": 60,
            }
        ),
    )
    engine._fuse_execution_group_observations(
        60,
        (
            *first_observations,
            *second_observations,
        ),
    )

    report = engine._latest_reports["target_00"]
    assert report.group_id == second.group_id
    assert report.member_ids == second.member_ids
    assert report.belief.accepted_observation_ids_this_cycle == tuple(
        observation.observation_id for observation in second_observations
    )
    assert set(report.belief.source_observation_ids) >= {o.observation_id for o in first_observations}


def test_group_fusion_prefers_strongest_current_cycle_member_support() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    for uuv_id in ("uuv_00", "uuv_01", "uuv_02", "uuv_03"):
        engine.request_uuv_deployment(uuv_id, reason="test")
    stronger = engine.activate_execution_group(
        target_id="target_00",
        region_id="region-00",
        member_ids=("uuv_00", "uuv_01"),
    )
    weaker = engine.activate_execution_group(
        target_id="target_00",
        region_id="region-99",
        member_ids=("uuv_02", "uuv_03"),
    )
    initial = tuple(
        PassiveSonarObservation(
            observation_id=f"initial:{uuv_id}",
            scenario_id=engine._scenario_id,
            sim_time_s=0,
            observer_id=uuv_id,
            target_id="target_00",
            azimuth_rad=prior_bearing(engine, uuv_id, 0),
            variance_rad2=0.01,
            detection_confidence=0.9,
            snr_db=8.0,
        )
        for uuv_id in stronger.member_ids
    )
    engine._fuse_execution_group_observations(0, initial)
    current = tuple(
        observation.model_copy(
            update={
                "observation_id": f"current:{observation.observer_id}",
                "sim_time_s": 30,
            }
        )
        for observation in initial
    ) + (
        initial[0].model_copy(
            update={
                "observation_id": "current:uuv_02",
                "observer_id": weaker.member_ids[0],
                "sim_time_s": 30,
            }
        ),
    )

    engine._fuse_execution_group_observations(30, current)

    report = engine._latest_reports["target_00"]
    assert report.group_id == stronger.group_id
    assert report.member_ids == stronger.member_ids
    assert report.belief.accepted_observation_ids_this_cycle == (
        "current:uuv_00",
        "current:uuv_01",
    )
    assert set(report.belief.source_observation_ids) >= {
        "current:uuv_00",
        "current:uuv_01",
    }
    assert "current:uuv_02" not in report.belief.source_observation_ids


def test_public_fusion_drives_two_distinct_entry_confirmation_cycles() -> None:
    config = load_app_config(CONFIG_PATH)
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    execution = _runtime_execution_snapshot()
    engine._clock.sim_time_s = int(execution.valid_from_s)
    assert engine.apply_verified_execution_snapshot(execution)
    region = controller.snapshot().regions[0]
    group = next(
        candidate
        for candidate in controller.snapshot().task_groups
        if candidate.region_id == region.region_id
    )
    target_xy = (
        sum(point[0] for point in region.region_polygon) / len(region.region_polygon),
        sum(point[1] for point in region.region_polygon) / len(region.region_polygon),
    )
    member_positions = (
        (target_xy[0] - 300.0, target_xy[1]),
        (target_xy[0], target_xy[1] - 300.0),
        (target_xy[0] + 300.0, target_xy[1]),
    )
    for member_id, position in zip(
        group.member_uuv_ids,
        member_positions,
        strict=True,
    ):
        engine._uuvs[member_id].position_xy = position

    def fuse(cycle_s: int) -> tuple[str, ...]:
        observations = tuple(
            PassiveSonarObservation(
                observation_id=f"public:{member_id}:{cycle_s}",
                scenario_id=engine._scenario_id,
                sim_time_s=cycle_s,
                observer_id=member_id,
                target_id=group.target_id,
                azimuth_rad=atan2(
                    target_xy[1] - position[1],
                    target_xy[0] - position[0],
                ),
                variance_rad2=0.001,
                detection_confidence=0.99,
                snr_db=20.0,
            )
            for member_id, position in zip(
                group.member_uuv_ids,
                member_positions,
                strict=True,
            )
        )
        engine._fuse_execution_group_observations(cycle_s, observations)
        return tuple(observation.observation_id for observation in observations)

    for warmup_cycle_s in range(40, 130, 10):
        fuse(warmup_cycle_s)
    first_ids = fuse(130)
    first_report = engine._latest_reports[group.target_id]
    assert set(first_ids) <= set(first_report.belief.source_observation_ids)
    first_probabilities = engine._mission_entry_probabilities(130, controller.snapshot())
    assert first_probabilities[region.region_id] >= 0.70, (
        first_report.belief.mean,
        first_report.belief.covariance,
        target_xy,
        region.region_polygon,
    )
    engine._advance_mission_controller(130)
    first = next(
        item for item in controller.snapshot().regions if item.region_id == region.region_id
    )
    assert first.entry_confirmations == 1
    assert first.entry_evidence_ids == first_ids

    assert engine._mission_entry_probabilities(160, controller.snapshot()) == {}
    engine._advance_mission_controller(160)
    stale = next(
        item for item in controller.snapshot().regions if item.region_id == region.region_id
    )
    assert stale.entry_confirmations == 0
    assert stale.entry_reset_reason == "missing_probability"

    fuse(190)
    engine._advance_mission_controller(190)
    second_ids = fuse(220)
    engine._advance_mission_controller(220)

    completed = controller.snapshot()
    completed_region = next(
        item for item in completed.regions if item.region_id == region.region_id
    )
    completed_group = next(
        item for item in completed.task_groups if item.group_instance_id == group.group_instance_id
    )
    assert completed_region.entry_confirmations == 2
    assert completed_region.entry_evidence_ids == second_ids
    assert completed_group.lifecycle is TaskGroupLifecycle.PASSIVE_TRACK
    assert completed_group.sensor_mode is GroupSensorMode.PASSIVE
    assert completed.tracking_control.tracking_owner_group_id == group.group_instance_id
