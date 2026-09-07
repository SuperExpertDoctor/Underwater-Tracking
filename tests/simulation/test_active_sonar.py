# tests/simulation/test_active_sonar.py
"""Active-sonar probe model and decoy entities (spec 5.1/11.1 amendment, R5).

The engine emits decoys with the same passive bearing observations as
submarines; classification comes exclusively from active pings. All tests
construct custom configs (decoy behavior is off by default), are
deterministic under the fixed seed, and never touch truth-boundary gates.
"""

from math import hypot

import pytest

from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.models import DeploymentState, EventAudience, EventLevel
from underwater_tracking.planning.coverage import ping_footprint_coverage_fraction
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.simulation.target import HiddenIntent, TRANSITION_PROBABILITIES
from underwater_tracking.runtime.mission_controller import MissionController
from tests.conftest import CONFIG_PATH
from tests.simulation.test_execution_group_activation import (
    _runtime_execution_snapshot,
)


def _decoy_config(**overrides: object) -> object:
    base = load_app_config(CONFIG_PATH)
    tracking = base.tracking.model_copy(update=overrides)
    scenario = base.scenario.model_copy(update={"initial_decoy_count": 1})
    return base.model_copy(update={"tracking": tracking, "scenario": scenario})


def _run(config: object, steps: int, *, tmp_path, sink=None) -> list[dict[str, object]]:
    engine = SimulationEngine(
        config,
        seed=7,
        output_dir=tmp_path,
        evaluation_sink=sink.append if sink is not None else None,
    )
    frames: list[dict[str, object]] = []
    for _ in range(steps):
        frames.append(engine.step())
    return frames


def test_active_sonar_event_types_classify():
    monitor = EventMonitor()
    assert monitor.classify("active_ping") is EventLevel.INFORMATIONAL
    assert monitor.classify("contact_classified") is EventLevel.INFORMATIONAL


def test_decoy_spawns_unverified_contact(tmp_path):
    config = _decoy_config()
    frames = _run(config, 1, tmp_path=tmp_path)
    contacts = {c["contact_id"]: c for c in frames[0]["contacts"]}
    assert "decoy_00" in contacts
    assert contacts["decoy_00"]["classification"] == "unverified"
    assert set(frames[0].keys()) <= {
        "carrier",
        "contacts",
        "reservations",
        "run_id",
        "scenario_id",
        "sim_time_s",
        "step_index",
        "uuvs",
        "group_reports",
        "tracks",
        "quality",
        "assignments",
        "events",
        "waypoint_commands",
        "platform_core",
        "usvs",
        "communication_links",
        "sonar_observations",
        "execution_groups",
        "target_search_priors",
    }


def test_decoy_is_passively_indistinguishable_from_a_submarine(tmp_path):
    config = _decoy_config()
    frames = _run(config, 1, tmp_path=tmp_path)
    contacts = {c["contact_id"]: c for c in frames[0]["contacts"]}
    rays = contacts["decoy_00"]["bearing_rays"]
    assert 0 < len(rays) <= 12  # probabilistic detection, never geometry-only
    assert all(not ray["is_false_alarm"] for ray in rays)


def test_unverified_ping_request_does_not_expose_contact_position(tmp_path):
    config = _decoy_config()
    frames = _run(config, 6, tmp_path=tmp_path)
    requests = [
        event
        for frame in frames
        for event in frame["events"]
        if event["event_type"] == "active_ping"
        and event["entity_id"] == "decoy_00"
        and "emitter_id" not in event["payload"]
    ]
    assert requests
    assert all("position_xy" not in event["payload"] for event in requests)


def test_truth_reports_decoys(tmp_path):
    config = _decoy_config()
    truths: list[dict[str, object]] = []
    _run(config, 1, tmp_path=tmp_path, sink=truths)
    assert truths
    decoys = truths[-1]["decoys"]
    assert [d["decoy_id"] for d in decoys] == ["decoy_00"]


def test_decoy_drift_speed_is_configured(tmp_path):
    config = _decoy_config()
    truths: list[dict[str, object]] = []
    _run(config, 3, tmp_path=tmp_path, sink=truths)
    # One physics step moves the decoy EXACTLY by speed * step (post-noise)
    # heading; a multi-step chord would be bent short by the heading walk.
    first = truths[0]["decoys"][0]["position_xy"]
    last = truths[1]["decoys"][0]["position_xy"]
    delta = hypot(last[0] - first[0], last[1] - first[1])
    assert delta == pytest.approx(0.5 * config.timing.physics_step_s, abs=1e-9)


def test_unheard_ping_still_consumes_active_sonar_energy(tmp_path):
    config = _decoy_config(sensor_ping_heard_probability=0.0)
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    baseline = SimulationEngine(config, seed=7, output_dir=tmp_path / "baseline")
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="decoy_00")

    frame = engine.step()
    baseline_frame = baseline.step()
    ping_energy = config.tracking.sensor_ping_energy_cost
    actual = next(item["energy_fraction"] for item in frame["uuvs"] if item["uuv_id"] == "uuv_00")
    without_ping = next(
        item["energy_fraction"] for item in baseline_frame["uuvs"] if item["uuv_id"] == "uuv_00"
    )

    assert actual == pytest.approx(without_ping - ping_energy, abs=1e-9)
    assert not any(event["event_type"] == "contact_classified" for event in frame["events"])


def test_active_ping_classifies_and_drains_energy(tmp_path):
    config = _decoy_config(
        sensor_ping_heard_probability=1.0,
        sensor_active_classify_decoy_prob=0.0,  # decoys ALWAYS classify submarine
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="decoy_00")
    frame = engine.step()
    contacts = {c["contact_id"]: c for c in frame["contacts"]}
    assert contacts["decoy_00"]["classification"] == "submarine"
    assert contacts["decoy_00"]["estimated_position_xy"] is not None
    uuvs = {u["uuv_id"]: u for u in frame["uuvs"]}
    # The ping drains exactly 2e-4; uuv_00 also burns motion energy in the
    # same step (40 m at 2e-6/m + 10 s at 1e-7/s = 8.1e-5), so the total
    # sits just below 1 - 2e-4.
    assert uuvs["uuv_00"]["energy_fraction"] == pytest.approx(1.0 - 2e-4, abs=1e-4)
    assert any(e["event_type"] == "contact_classified" for e in frame["events"])
    assert any(e["event_type"] == "active_ping" for e in frame["events"])


def test_uuv_only_active_ping_uses_public_prior_but_requires_physical_echo(tmp_path):
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    config = config.model_copy(
        update={
            "tracking": config.tracking.model_copy(update={"sensor_ping_heard_probability": 1.0})
        }
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    target_origin = engine._targets["target_00"].position_xy
    engine._deployment_states["uuv_00"] = DeploymentState.DEPLOYED
    engine._waterborne_uuv_ids.add("uuv_00")
    engine._uuvs["uuv_00"].position_xy = (target_origin[0] + 500.0, target_origin[1])
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="target_00")

    engine._targets["target_00"].position_xy = (target_origin[0] + 5_000.0, target_origin[1])
    engine._process_pings(0)

    assert any(event.event_type == "active_ping" for event in engine._events)
    assert not any(event.event_type == "contact_classified" for event in engine._events)

    engine._targets["target_00"].position_xy = (target_origin[0] + 1_000.0, target_origin[1])
    engine._process_pings(30)

    assert any(event.event_type == "contact_classified" for event in engine._events)
    assert len(engine._active_sonar_observations) == 1
    observation = engine._active_sonar_observations[0]
    assert observation.observer_id == "uuv_00"
    assert observation.target_id == "target_00"
    assert observation.observation_id.startswith("active:")
    noisy_fix = engine._contact_state["target_00"]["position_xy"]
    assert noisy_fix is not None
    assert noisy_fix != engine._targets["target_00"].position_xy


def test_active_transmission_occurs_once_without_public_estimate_or_echo(tmp_path):
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine._target_search_priors = ()
    engine._deployment_states["uuv_00"] = DeploymentState.DEPLOYED
    engine._waterborne_uuv_ids.add("uuv_00")
    engine._uuvs["uuv_00"].position_xy = (0.0, 0.0)
    engine._targets["target_00"].position_xy = (10_000.0, 10_000.0)
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="target_00")
    before_energy = engine._uuvs["uuv_00"].energy_fraction

    engine._process_pings(0)
    engine._process_pings(0)

    transmissions = tuple(
        event
        for event in engine._events
        if event.event_type == "active_ping" and event.payload.get("emitter_id") == "uuv_00"
    )
    assert len(transmissions) == 1
    assert transmissions[0].payload["source_position_xy"] == (0.0, 0.0)
    assert transmissions[0].payload["configured_range_m"] == pytest.approx(
        config.scenario.tracking_policy.uuv_active_detection_radius_m
    )
    assert engine._uuvs["uuv_00"].energy_fraction == pytest.approx(
        before_energy - config.tracking.sensor_ping_energy_cost
    )
    assert engine._active_sonar_observations == []
    assert not any(event.event_type == "active_echo" for event in engine._events)


def test_passive_uuv_never_emits_active_ping(tmp_path):
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine._deployment_states["uuv_00"] = DeploymentState.DEPLOYED
    engine._waterborne_uuv_ids.add("uuv_00")
    engine.set_sensor_mode("uuv_00", "passive", ping_contact_id="target_00")

    engine._process_pings(0)

    assert not any(
        event.event_type == "active_ping" and event.payload.get("emitter_id") == "uuv_00"
        for event in engine._events
    )


def test_four_runtime_regions_accumulate_only_source_backed_ping_coverage(tmp_path):
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    config = config.model_copy(
        update={
            "tracking": config.tracking.model_copy(update={"sensor_ping_heard_probability": 1.0})
        }
    )
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(
        config,
        seed=7,
        output_dir=tmp_path,
        mission_controller=controller,
    )
    snapshot = _runtime_execution_snapshot()
    start_s = int(snapshot.valid_from_s)
    engine._clock.sim_time_s = start_s
    assert engine.apply_verified_execution_snapshot(snapshot)
    engine._uuvs["uuv_01"].position_xy = (-200.0, 20.0)
    engine._uuvs["uuv_01"].set_waypoints([(-100.0, 20.0)])
    engine._targets["target_00"].position_xy = (90.0, 20.0)
    initial_uuv_01_waypoint = engine._uuvs["uuv_01"].waypoints[0]

    engine._process_pings(start_s)
    engine._advance_mission_controller(start_s)

    counts = tuple(region.ping_count for region in controller.snapshot().regions)
    assert counts == (3, 3, 3, 3)
    public_fix = engine._contact_state["target_00"]["position_xy"]
    assert public_fix is not None
    echo_waypoint = engine._uuvs["uuv_01"].waypoints[0]
    assert hypot(
        echo_waypoint[0] - public_fix[0],
        echo_waypoint[1] - public_fix[1],
    ) < hypot(
        initial_uuv_01_waypoint[0] - public_fix[0],
        initial_uuv_01_waypoint[1] - public_fix[1],
    )

    mission = controller.snapshot()
    emitted_ids = {event.event_id for event in engine._events if event.event_type == "active_ping"}
    assert len(mission.regions) == 4
    assert all(region.ping_count == 3 for region in mission.regions)
    assert all(region.coverage > 0.0 for region in mission.regions)
    assert all(
        region.scan_completed == (region.coverage >= region.scan_completion_threshold)
        for region in mission.regions
    )
    assert all(
        set(region.scan_evidence_ids).issubset(emitted_ids) and len(region.scan_evidence_ids) == 3
        for region in mission.regions
    )


def test_runtime_scan_round_resets_ping_coverage_and_evidence(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(
        config,
        seed=7,
        output_dir=tmp_path,
        mission_controller=controller,
    )
    snapshot = _runtime_execution_snapshot()
    engine._clock.sim_time_s = int(snapshot.valid_from_s)
    assert engine.apply_verified_execution_snapshot(snapshot)
    mission = controller.snapshot()
    group = mission.task_groups[0]
    engine._runtime_scan_ping_samples[group.group_instance_id] = [
        ("active_ping:old-round", 0.0, 0.0, 600.0)
    ]
    engine._runtime_scan_ping_counts[group.group_instance_id] = 1
    for member_id in group.member_uuv_ids:
        engine._uuvs[member_id].set_waypoints([])

    completed_state = engine._runtime_scan_state_observations(mission)[group.region_id]
    controller.observe({"runtime_scan_states": {group.region_id: completed_state}})

    assert completed_state["scan_round"] == 0
    assert completed_state["route_progress"] == 1.0
    assert completed_state["ping_count"] == 1
    assert completed_state["evidence_ids"] == ("active_ping:old-round",)
    public_completed = next(
        item for item in controller.snapshot().regions if item.region_id == group.region_id
    )
    assert public_completed.scan_round == 0
    assert public_completed.route_progress == 1.0

    engine._plan_runtime_group_waypoints(mission)
    engine._process_pings(int(snapshot.valid_from_s) + 30)

    restarted_state = engine._runtime_scan_state_observations(mission)[group.region_id]
    controller.observe({"runtime_scan_states": {group.region_id: restarted_state}})

    assert restarted_state["scan_round"] == 1
    assert restarted_state["route_progress"] == 0.0
    assert restarted_state["coverage"] > 0.0
    assert restarted_state["ping_count"] == 3
    assert len(restarted_state["evidence_ids"]) == 3
    assert len(engine._runtime_scan_ping_samples[group.group_instance_id]) == 3
    public_restarted = next(
        item for item in controller.snapshot().regions if item.region_id == group.region_id
    )
    assert public_restarted.scan_round == 1
    assert public_restarted.route_progress == 0.0


def test_runtime_scan_ping_samples_remain_bounded_when_route_stalls(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(
        config,
        seed=7,
        output_dir=tmp_path,
        mission_controller=controller,
    )
    snapshot = _runtime_execution_snapshot()
    start_s = int(snapshot.valid_from_s)
    engine._clock.sim_time_s = start_s
    assert engine.apply_verified_execution_snapshot(snapshot)
    group = controller.snapshot().task_groups[0]
    emitter_id = group.member_uuv_ids[0]
    for uuv_id in engine._sensor_modes:
        engine._sensor_modes[uuv_id] = "passive"
    engine._sensor_modes[emitter_id] = "active"
    engine._ping_targets[emitter_id] = "target_00"

    for ping_index in range(300):
        engine._process_pings(start_s + ping_index * 30)

    state = engine._runtime_scan_state_observations(controller.snapshot())[group.region_id]

    assert state["ping_count"] == 300
    assert len(state["evidence_ids"]) == 256
    assert len(engine._runtime_scan_ping_samples[group.group_instance_id]) == 256


def test_runtime_scan_coverage_retains_evicted_ping_footprints(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(
        config,
        seed=7,
        output_dir=tmp_path,
        mission_controller=controller,
    )
    snapshot = _runtime_execution_snapshot()
    start_s = int(snapshot.valid_from_s)
    engine._clock.sim_time_s = start_s
    assert engine.apply_verified_execution_snapshot(snapshot)
    first_region_id = controller.snapshot().regions[0].region_id
    controller._regions[first_region_id] = controller._regions[first_region_id].model_copy(
        update={
            "region_polygon": (
                (0.0, 0.0),
                (2000.0, 0.0),
                (2000.0, 2000.0),
                (0.0, 2000.0),
            )
        }
    )
    mission = controller.snapshot()
    group = mission.task_groups[0]
    region = next(item for item in mission.regions if item.region_id == group.region_id)
    emitter_id = group.member_uuv_ids[0]
    for uuv_id in engine._sensor_modes:
        engine._sensor_modes[uuv_id] = "passive"
    engine._sensor_modes[emitter_id] = "active"
    engine._ping_targets[emitter_id] = "target_00"
    min_x = min(point[0] for point in region.region_polygon)
    max_x = max(point[0] for point in region.region_polygon)
    center_y = sum(point[1] for point in region.region_polygon) / len(region.region_polygon)
    early_xy = (min_x + 250.0, center_y)
    late_xy = (max_x - 250.0, center_y)
    engine._uuvs[emitter_id].position_xy = early_xy
    engine._process_pings(start_s)
    engine._uuvs[emitter_id].position_xy = late_xy
    for ping_index in range(1, 301):
        engine._process_pings(start_s + ping_index * 30)

    state = engine._runtime_scan_state_observations(mission)[group.region_id]
    late_only_coverage = ping_footprint_coverage_fraction(
        region.region_polygon,
        (late_xy,),
        detection_radius_m=engine._uuvs[emitter_id].capability.active_range_m,
    )

    assert state["ping_count"] == 301
    assert len(state["evidence_ids"]) == 256
    assert state["coverage"] > late_only_coverage


def test_runtime_scan_state_prunes_retired_group_generations(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(
        config,
        seed=7,
        output_dir=tmp_path,
        mission_controller=controller,
    )
    snapshot = _runtime_execution_snapshot()
    engine._clock.sim_time_s = int(snapshot.valid_from_s)
    assert engine.apply_verified_execution_snapshot(snapshot)
    retired_ids = ("retired:deploy:000001", "retired:deploy:000002")
    for group_id in retired_ids:
        engine._runtime_scan_ping_samples[group_id] = [
            (f"active_ping:{group_id}", 0.0, 0.0, 600.0)
        ]
        engine._runtime_scan_ping_counts[group_id] = 1
        engine._runtime_scan_route_state[group_id] = (0, 0.5)
        engine._runtime_scan_coverage_masks[group_id] = 1

    engine._reconcile_runtime_execution_groups(controller.snapshot())

    live_ids = {
        group.group_instance_id
        for group in controller.snapshot().task_groups
        if group.lifecycle.value != "disappeared"
    }
    for state_by_group in (
        engine._runtime_scan_ping_samples,
        engine._runtime_scan_ping_counts,
        engine._runtime_scan_route_state,
        engine._runtime_scan_coverage_masks,
    ):
        assert set(state_by_group) <= live_ids


def test_uuv_only_active_echo_reaches_public_group_report(tmp_path, monkeypatch):
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    config = config.model_copy(
        update={
            "tracking": config.tracking.model_copy(update={"sensor_ping_heard_probability": 1.0})
        }
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    target_origin = engine._targets["target_00"].position_xy
    positions = {
        "uuv_00": (target_origin[0] + 500.0, target_origin[1]),
        "uuv_01": (target_origin[0] - 500.0, target_origin[1]),
    }
    for uuv_id, position in positions.items():
        engine._deployment_states[uuv_id] = DeploymentState.DEPLOYED
        engine._waterborne_uuv_ids.add(uuv_id)
        engine._uuvs[uuv_id].position_xy = position
    engine._targets["target_00"].position_xy = (
        target_origin[0] + 1_000.0,
        target_origin[1],
    )
    group = engine.activate_execution_group(
        target_id="target_00",
        region_id="region-00",
        member_ids=("uuv_00", "uuv_01"),
    )
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="target_00")
    manager_type = type(engine._manager)
    create_positions: list[dict[str, tuple[float, float]]] = []
    original_create = manager_type.create

    def capture_create(manager, *args, **kwargs):
        create_positions.append(dict(kwargs["member_positions"]))
        return original_create(manager, *args, **kwargs)

    monkeypatch.setattr(manager_type, "create", capture_create)

    engine._process_pings(0)
    echo = next(event for event in engine._events if event.event_type == "active_echo")
    assert echo.audiences == frozenset({EventAudience.OPERATOR_AUDIT, EventAudience.MEMORY_SOURCE})
    assert echo not in engine._blue_public_events((echo,))
    assert echo in engine._operator_public_events((echo,))
    assert any(event["event_type"] == "active_echo" for event in engine._build_frame(0)["events"])
    engine._uuvs["uuv_00"].position_xy = (
        positions["uuv_00"][0],
        positions["uuv_00"][1] + 1_000.0,
    )
    situation = engine._platform_core_observation_cycle_locked(30)

    active_observation = next(
        observation
        for observation in situation.platform_observations
        if observation.observation_id.startswith("active:")
    )
    report = next(report for report in situation.group_reports if report.target_id == "target_00")
    assert active_observation.observer_id == "uuv_00"
    assert report.group_id == group.group_id
    assert active_observation.observation_id in report.belief.source_observation_ids
    assert create_positions[0]["uuv_00"] == positions["uuv_00"]
    public_fix = engine._contact_state["target_00"]["position_xy"]
    assert public_fix is not None
    assert hypot(
        report.belief.mean[0] - public_fix[0],
        report.belief.mean[1] - public_fix[1],
    ) < 0.25 * hypot(
        target_origin[0] - public_fix[0],
        target_origin[1] - public_fix[1],
    )

    invoke_count = 0
    original_invoke = manager_type.invoke

    def count_invoke(manager, *args, **kwargs):
        nonlocal invoke_count
        invoke_count += 1
        return original_invoke(manager, *args, **kwargs)

    monkeypatch.setattr(manager_type, "invoke", count_invoke)
    engine._uuvs["uuv_00"].position_xy = positions["uuv_00"]
    engine._process_pings(30)
    engine._uuvs["uuv_00"].position_xy = (
        positions["uuv_00"][0],
        positions["uuv_00"][1] + 1_000.0,
    )
    engine._platform_core_observation_cycle_locked(60)

    assert invoke_count == 1


def test_heard_ping_queues_bounded_evasive_sprint(tmp_path, monkeypatch):
    config = _decoy_config(
        sensor_ping_heard_probability=1.0,
        sensor_active_classify_submarine_prob=1.0,
    )
    truths: list[dict[str, object]] = []
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path, evaluation_sink=truths.append)
    monkeypatch.setitem(
        TRANSITION_PROBABILITIES,
        HiddenIntent.EVADE,
        {HiddenIntent.EVADE: 1.0},
    )
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="target_00")
    engine.step()
    ping_target = truths[-1]["targets"][0]
    ping_vx, ping_vy = ping_target["velocity_xy"]
    ping_speed = hypot(ping_vx, ping_vy)
    assert ping_target["intent_label"] == "evade"
    assert ping_speed < config.tracking.submarine_sprint_speed_mps

    target_entity = engine._targets["target_00"]
    engine.step()
    next_target = truths[-1]["targets"][0]
    next_vx, next_vy = next_target["velocity_xy"]
    next_speed = hypot(next_vx, next_vy)
    max_speed_delta = target_entity.max_acceleration_mps2 * config.timing.physics_step_s

    assert next_speed > ping_speed
    assert next_speed - ping_speed <= max_speed_delta + 1e-9
    assert next_speed <= config.tracking.submarine_sprint_speed_mps + 1e-9


@pytest.mark.parametrize(
    "deployment_state",
    [
        DeploymentState.RETURNING,
        DeploymentState.ONBOARD,
        DeploymentState.FAILED,
    ],
)
def test_active_sonar_does_not_ping_or_classify_non_deployed_uuvs(
    tmp_path, deployment_state: DeploymentState
):
    config = _decoy_config(
        sensor_ping_heard_probability=1.0,
        sensor_active_classify_decoy_prob=1.0,
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    uuv_id = "uuv_00"
    if deployment_state is DeploymentState.RETURNING:
        engine.request_uuv_recovery(uuv_id)
    elif deployment_state is DeploymentState.ONBOARD:
        engine.request_uuv_recovery(uuv_id)
        engine._uuvs[uuv_id].position_xy = (-2950.0, -3000.0)
        engine.step()
    else:
        engine.fail_uuv(uuv_id)

    engine.set_sensor_mode(uuv_id, "active", ping_contact_id="decoy_00")
    frame = engine.step()

    executed_events = [
        event
        for event in frame["events"]
        if event["event_type"] in {"active_ping", "contact_classified"}
        and event["payload"].get("uuv_id") == uuv_id
    ]
    contacts = {contact["contact_id"]: contact for contact in frame["contacts"]}
    assert executed_events == []
    assert contacts["decoy_00"]["classification"] == "unverified"


def test_drop_contact_removes_the_decoy(tmp_path):
    config = _decoy_config()
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.step()
    engine.drop_contact("decoy_00")
    frame = engine.step()
    assert "decoy_00" not in {c["contact_id"] for c in frame["contacts"]}


def test_promote_contact_creates_target_and_group(tmp_path):
    config = _decoy_config(
        sensor_ping_heard_probability=1.0,
        sensor_active_classify_decoy_prob=0.0,  # ALWAYS submarine
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="decoy_00")
    engine.step()  # the commanded ping classifies decoy_00 as submarine
    engine.promote_contact("decoy_00")
    frame = engine.step()  # the group exists from promotion; the frame carries it
    reports = {r["target_id"]: r for r in frame["group_reports"]}
    assert "decoy_00" in reports
    assert 2 <= len(reports["decoy_00"]["member_ids"]) <= 4


def test_promoted_legacy_target_uses_configured_motion_limits(tmp_path):
    config = _decoy_config(
        submarine_sprint_speed_mps=19.0,
        submarine_turn_rate_rad_s=0.17,
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine._contact_state["decoy_00"]["position_xy"] = (100.0, 200.0)

    engine.promote_contact("decoy_00")

    target = engine._targets["decoy_00"]
    assert target.max_speed_mps == config.tracking.submarine_sprint_speed_mps
    assert target.max_turn_rate_rad_s == config.tracking.submarine_turn_rate_rad_s


def test_reserved_uuv_is_skipped_from_decoy_observation(tmp_path):
    config = _decoy_config()
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.set_reservations({"target_00": ("uuv_00",)})
    frame = engine.step()
    contacts = {c["contact_id"]: c for c in frame["contacts"]}
    rays = contacts["decoy_00"]["bearing_rays"]
    assert 0 < len(rays) <= 11
    assert all(not ray["is_false_alarm"] for ray in rays)
    assert "uuv_00" not in {r["uuv_id"] for r in rays}
