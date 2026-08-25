from __future__ import annotations

from math import cos, hypot, sin
from pathlib import Path

from underwater_tracking.cli import _AgentLoop, _mission_controller_for, _step_with_llm_retries
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.simulation.engine import SimulationEngine
from tests.integration.test_uuv_only_production_acceptance import FixedSeedUUVLLM


CONFIG_PATH = "configs/scenario/uuv_only_single_target.yaml"
SEED = 20260820


def _has_event(frame: object, event_type: str) -> bool:
    return any(event.event_type == event_type for event in frame.events)  # type: ignore[attr-defined]


def _exposed_ids(frame: object) -> tuple[str, ...]:
    return tuple(
        uuv.uuv_id
        for uuv in frame.uuvs  # type: ignore[attr-defined]
        if uuv.physically_exposed
    )


def _collect_transition_frames(frames: list[object]) -> dict[str, object]:
    initial = next(frame for frame in frames if frame.plan_version == 0)  # type: ignore[attr-defined]
    pre_deploy = next(
        frame
        for frame in frames
        if frame.planned_assignments  # type: ignore[attr-defined]
        and not _exposed_ids(frame)
    )
    deploy = next(frame for frame in frames if _has_event(frame, "uuv_deployed"))
    post_deploy = next(
        frame
        for frame in frames
        if frame.sim_time_s > deploy.sim_time_s  # type: ignore[attr-defined]
        and frame.execution_groups  # type: ignore[attr-defined]
        and _exposed_ids(frame)
    )
    return {
        "initial": initial,
        "pre_deploy": pre_deploy,
        "deploy": deploy,
        "post_deploy": post_deploy,
    }


def test_default_entry_publishes_truthful_bootstrap_and_deployment_frames(
    tmp_path: Path,
) -> None:
    config = load_app_config(CONFIG_PATH)
    assert config.environment is not None
    accelerated_environment = config.environment.model_copy(
        update={
            "carrier": config.environment.carrier.model_copy(
                update={"speed_mps": 20.0}
            ),
            "carriers": tuple(
                carrier.model_copy(update={"speed_mps": 40.0})
                for carrier in config.environment.carriers
            ),
        }
    )
    config = config.model_copy(update={"environment": accelerated_environment})
    controller = _mission_controller_for(config)
    assert controller is not None
    llm = FixedSeedUUVLLM()
    loop = _AgentLoop(
        config,
        database_path=tmp_path / "agent.db",
        llm={"master": llm},
        run_id="truthful-bootstrap-deployment",
        steps=160,
        seed=SEED,
    )
    engine = SimulationEngine(
        config,
        seed=SEED,
        output_dir=tmp_path / "frames",
        carrier=loop.on_situation,
        mission_controller=controller,
    )
    frames: list[object] = []
    try:
        loop.attach(engine)
        bootstrap = loop.hub.snapshot()
        assert bootstrap is not None
        frames.append(bootstrap)
        for _ in range(160):
            assert _step_with_llm_retries(engine, loop, config) is True
            frame = loop.hub.snapshot()
            assert frame is not None
            frames.append(frame)
            if frame.planned_assignments and not _exposed_ids(frame):
                assert engine._manager.list_groups() == ()
                assert engine._latest_reports == {}
            if frame.sim_time_s >= 510 and frame.execution_groups:  # type: ignore[attr-defined]
                break
    finally:
        assert loop.close() is True
        engine.logger.close()

    checkpoints = _collect_transition_frames(frames)
    initial = checkpoints["initial"]
    pre_deploy = checkpoints["pre_deploy"]
    deploy = checkpoints["deploy"]
    post_deploy = checkpoints["post_deploy"]

    assert "target_priors" not in initial.model_dump()  # type: ignore[attr-defined]
    assert len(initial.target_estimates) == 1  # type: ignore[attr-defined]
    assert initial.target_estimates[0].classification == "submarine"  # type: ignore[attr-defined]
    assert initial.groups == ()  # type: ignore[attr-defined]
    assert initial.execution_groups == ()  # type: ignore[attr-defined]
    assert initial.planned_assignments == ()  # type: ignore[attr-defined]
    assert len(initial.uuv_resources) == 12  # type: ignore[attr-defined]
    assert all(
        resource.deployment_state == "onboard" and resource.carrier_id is not None
        for resource in initial.uuv_resources  # type: ignore[attr-defined]
    )
    assert not any(
        uuv.physically_exposed for uuv in initial.uuvs  # type: ignore[attr-defined]
    )

    deployment_group = deploy.execution_groups[0]  # type: ignore[attr-defined]
    assignment = next(
        candidate
        for candidate in pre_deploy.planned_assignments  # type: ignore[attr-defined]
        if candidate.target_id == deployment_group.target_id
        and candidate.region_id == deployment_group.region_id
    )
    assert assignment.status in {"planned", "transporting", "ready_to_deploy"}
    assert not set(assignment.uuv_ids) & set(_exposed_ids(pre_deploy))

    assert _has_event(deploy, "uuv_deployed")
    assert deploy.execution_groups  # type: ignore[attr-defined]
    assert set(deployment_group.member_ids) == set(assignment.uuv_ids)
    assert all(group.target_id == "target_00" for group in deploy.groups)  # type: ignore[attr-defined]
    assert len(deploy.target_estimates) == 1  # type: ignore[attr-defined]
    assert deploy.target_estimates[0].classification == "submarine"  # type: ignore[attr-defined]
    assert set(assignment.uuv_ids) <= set(_exposed_ids(deploy))
    assert post_deploy.sim_time_s > deploy.sim_time_s  # type: ignore[attr-defined]
    assert post_deploy.execution_groups  # type: ignore[attr-defined]
    assert set(post_deploy.execution_groups[0].member_ids) <= set(_exposed_ids(post_deploy))  # type: ignore[attr-defined]
    assert "usv" not in post_deploy.model_dump_json().casefold()  # type: ignore[attr-defined]

    current_situation = engine.publication_situation()
    slave_contexts = engine.build_slave_contexts(current_situation)
    execution_members = {
        member
        for group in current_situation.execution_groups
        for member in group.member_ids
    }
    assert all(
        {platform.platform_id for platform in context.platforms} <= execution_members
        for context in slave_contexts
    )

    first = SimulationEngine(config, seed=SEED)
    second = SimulationEngine(config, seed=SEED)
    assert first.publication_situation().model_dump_json() == second.publication_situation().model_dump_json()
    target = config.environment.submarines[0]  # type: ignore[union-attr]
    mothers = config.environment.carriers  # type: ignore[union-attr]
    nearest_distance = min(
        hypot(
            target.position_xy[0] - carrier.position_xy[0],
            target.position_xy[1] - carrier.position_xy[1],
        )
        for carrier in mothers
    )
    assert nearest_distance > target.detection_range_m

    initial_carriers = {
        carrier.carrier_id: carrier
        for carrier in first.publication_situation().carriers
    }
    for _ in range(20):
        first.step()
    advanced_carriers = {
        carrier.carrier_id: carrier
        for carrier in first.publication_situation().carriers
    }
    assert advanced_carriers["carrier_01"].position_xy[0] > initial_carriers["carrier_01"].position_xy[0]
    leader = advanced_carriers["carrier_01"]
    reference_heading = initial_carriers["carrier_01"].heading_rad
    turn_rad = leader.heading_rad - reference_heading
    for carrier_id, initial_carrier in initial_carriers.items():
        if carrier_id == "carrier_01":
            continue
        initial_offset = (
            initial_carrier.position_xy[0] - initial_carriers["carrier_01"].position_xy[0],
            initial_carrier.position_xy[1] - initial_carriers["carrier_01"].position_xy[1],
        )
        offset = (
            cos(turn_rad) * initial_offset[0] - sin(turn_rad) * initial_offset[1],
            sin(turn_rad) * initial_offset[0] + cos(turn_rad) * initial_offset[1],
        )
        carrier = advanced_carriers[carrier_id]
        assert abs((carrier.position_xy[0] - leader.position_xy[0]) - offset[0]) <= 1.0
        assert abs((carrier.position_xy[1] - leader.position_xy[1]) - offset[1]) <= 1.0
        assert carrier.heading_rad == leader.heading_rad
