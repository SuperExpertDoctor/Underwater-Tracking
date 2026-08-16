from hashlib import sha256
from pathlib import Path
from math import atan2, cos, hypot, pi, sin
import random
import re
from uuid import UUID

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AppConfig
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.simulation.kinematics import (
    MotionCommand,
    MotionState,
    advance_motion,
    wrap_angle,
)


SCENARIO = Path("configs/scenario/segmented_single_target.yaml")


def _stable_int(text: str) -> int:
    return int.from_bytes(sha256(text.encode("utf-8")).digest()[:8], "big")


def _public_quality_reconstruction(
    run_id: str,
    observers: dict[str, dict[str, object]],
    observations: list[dict[str, object]],
) -> tuple[float, float] | None:
    """Reproduce the pre-fix public run-id attack when a seed is exposed."""
    match = re.fullmatch(r"run-(\d+)-[0-9a-f]+", run_id)
    if match is None:
        return None
    seed = int(match.group(1))
    recovered: list[tuple[tuple[float, float], float]] = []
    for observation in observations[:3]:
        observer_id = str(observation["observer_id"])
        target_id = str(observation["target_id"])
        observer = observers[observer_id]
        capability = observer["capability"]
        assert isinstance(capability, dict)
        sonar = capability["sonar"]
        assert isinstance(sonar, dict)
        quality_rng = random.Random(seed ^ _stable_int(f"platform:{target_id}:{observer_id}"))
        quality_noise = quality_rng.gauss(0.0, 0.075)
        confidence = float(observation["detection_confidence"])
        range_fraction = (0.9 + quality_noise - confidence) / 0.7
        position_xy = observer["position_xy"]
        assert isinstance(position_xy, tuple)
        recovered.append((position_xy, float(sonar["passive_range_m"]) * range_fraction))

    (x0, y0), r0 = recovered[0]
    (x1, y1), r1 = recovered[1]
    (x2, y2), r2 = recovered[2]
    a = 2.0 * (x1 - x0)
    b = 2.0 * (y1 - y0)
    c = 2.0 * (x2 - x0)
    d = 2.0 * (y2 - y0)
    e = r0**2 - r1**2 + x1**2 + y1**2 - x0**2 - y0**2
    f = r0**2 - r2**2 + x2**2 + y2**2 - x0**2 - y0**2
    determinant = a * d - b * c
    assert determinant != 0.0
    return ((e * d - b * f) / determinant, (a * f - e * c) / determinant)


def _old_quality_formula_reconstruction(
    observers: dict[str, dict[str, object]],
    observations: list[dict[str, object]],
) -> tuple[float, float]:
    recovered: list[tuple[tuple[float, float], float]] = []
    for observation in observations[:3]:
        observer = observers[str(observation["observer_id"])]
        capability = observer["capability"]
        assert isinstance(capability, dict)
        sonar = capability["sonar"]
        assert isinstance(sonar, dict)
        position_xy = observer["position_xy"]
        assert isinstance(position_xy, tuple)
        recovered.append(
            (
                position_xy,
                float(sonar["passive_range_m"])
                * (1.0 - float(observation["detection_confidence"])),
            )
        )

    (x0, y0), r0 = recovered[0]
    (x1, y1), r1 = recovered[1]
    (x2, y2), r2 = recovered[2]
    a = 2.0 * (x1 - x0)
    b = 2.0 * (y1 - y0)
    c = 2.0 * (x2 - x0)
    d = 2.0 * (y2 - y0)
    e = r0**2 - r1**2 + x1**2 + y1**2 - x0**2 - y0**2
    f = r0**2 - r2**2 + x2**2 + y2**2 - x0**2 - y0**2
    determinant = a * d - b * c
    assert determinant != 0.0
    return ((e * d - b * f) / determinant, (a * f - e * c) / determinant)


def _normalized_frame(frame: dict[str, object]) -> dict[str, object]:
    return {**frame, "run_id": "RUN"}


def _validated_platform_core_config(
    *,
    carrier_updates: dict[str, object],
    motion_updates: dict[str, object] | None = None,
    deployment_state: str | None = None,
) -> AppConfig:
    data = load_app_config(SCENARIO).model_dump()
    data["environment"]["carrier"].update(carrier_updates)
    if deployment_state is not None:
        for usv in data["environment"]["usvs"]:
            usv["deployment_state"] = deployment_state
    if motion_updates is not None:
        data["platforms"]["motion_profiles"]["usv_standard"].update(motion_updates)
    return AppConfig.model_validate(data)


def _small_support_fast_carrier_config() -> AppConfig:
    return _validated_platform_core_config(
        carrier_updates={"speed_mps": 100.0, "support_radius_m": 650.0},
        deployment_state="onboard",
    )


def test_explicit_platform_core_world_spawns_from_yaml(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    snapshot = engine.platform_snapshot()

    assert snapshot.scenario_id == "segmented-single-target"
    assert snapshot.carrier.carrier_id == "carrier_01"
    assert [usv.platform_id for usv in snapshot.roster.usvs] == [
        "usv_00", "usv_01", "usv_02", "usv_03"
    ]
    assert [uuv.platform_id for uuv in snapshot.roster.uuvs] == [
        f"uuv_{index:02d}" for index in range(12)
    ]
    assert snapshot.carrier.onboard_platform_ids == tuple(
        f"uuv_{index:02d}" for index in range(12)
    )


def test_explicit_initial_snapshot_normalizes_onboard_uuvs_only(tmp_path: Path) -> None:
    data = load_app_config(SCENARIO).model_dump()
    data["environment"]["uuvs"][0].update(
        {"position_xy": [-7000.0, -7000.0], "heading_rad": 1.2}
    )
    data["environment"]["uuvs"][1].update(
        {
            "deployment_state": "deployed",
            "position_xy": [-6000.0, -6000.0],
            "heading_rad": 0.7,
        }
    )
    config = AppConfig.model_validate(data)
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)

    snapshot = engine.platform_snapshot()
    states = {state.platform_id: state for state in snapshot.roster.uuvs}

    onboard = states["uuv_00"]
    assert onboard.deployment_state == "onboard"
    assert onboard.position_xy == snapshot.carrier.position_xy
    assert onboard.heading_rad == snapshot.carrier.heading_rad
    assert onboard.speed_mps == 0.0
    deployed = states["uuv_01"]
    assert deployed.deployment_state == "deployed"
    assert deployed.position_xy == (-6000.0, -6000.0)
    assert deployed.heading_rad == pytest.approx(0.7)


def test_explicit_frame_exposes_usvs_and_distance_links(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    frame = engine.step()
    for _ in range(2):
        frame = engine.step()

    assert frame["platform_core"] is True
    assert len(frame["usvs"]) == 4
    assert frame["uuvs"][0]["deployment_state"] == "onboard"
    assert any(link["medium"] == "surface" for link in frame["communication_links"])
    assert frame["sonar_observations"]


def test_public_platform_frame_hides_seed_and_blocks_exact_quality_reconstruction(
    tmp_path: Path,
) -> None:
    seed = 42
    first_engine = SimulationEngine(load_app_config(SCENARIO), seed=seed, output_dir=tmp_path / "first")
    second_engine = SimulationEngine(load_app_config(SCENARIO), seed=seed, output_dir=tmp_path / "second")
    for _ in range(3):
        first_frame = first_engine.step()
        second_frame = second_engine.step()

    run_id = str(first_frame["run_id"])
    assert UUID(hex=run_id.removeprefix("run-")).hex == run_id.removeprefix("run-")
    assert _normalized_frame(first_frame) == _normalized_frame(second_frame)
    assert first_frame["run_id"] != second_frame["run_id"]

    public_observers = {
        str(platform["platform_id"]): platform
        for platform in [*first_frame["usvs"], *first_frame["uuvs"]]
    }
    public_observations = [
        observation
        for observation in first_frame["sonar_observations"]
        if observation["target_id"] == "target_00"
    ]
    assert len(public_observations) >= 3
    assert _public_quality_reconstruction(run_id, public_observers, public_observations) is None

    reconstructed_xy = _old_quality_formula_reconstruction(public_observers, public_observations)
    truth_xy = first_engine._targets["target_00"].position_xy
    assert hypot(reconstructed_xy[0] - truth_xy[0], reconstructed_xy[1] - truth_xy[1]) > 1.0


def test_explicit_uuv_energy_uses_yaml_motion_profile(tmp_path: Path) -> None:
    data = load_app_config(SCENARIO).model_dump()
    data["environment"]["uuvs"][0]["deployment_state"] = "deployed"
    profile = data["platforms"]["motion_profiles"]["uuv_standard"]
    profile["transit_energy_per_m"] = 0.001
    profile["hotel_energy_per_s"] = 0.002
    config = AppConfig.model_validate(data)
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    uuv = engine._uuvs["uuv_00"]
    uuv.set_waypoints([(uuv.position_xy[0] + 1_000.0, uuv.position_xy[1])])
    before_position = uuv.position_xy
    before_energy = uuv.energy_fraction

    engine.step()

    displacement_m = hypot(
        uuv.position_xy[0] - before_position[0],
        uuv.position_xy[1] - before_position[1],
    )
    assert uuv.energy_fraction == pytest.approx(
        before_energy
        - displacement_m * profile["transit_energy_per_m"]
        - config.timing.physics_step_s * profile["hotel_energy_per_s"]
    )


def test_explicit_onboard_uuvs_match_carrier_in_frame_and_snapshot(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)
    engine._uuvs["uuv_00"].heading_rad = 1.2
    engine._uuvs["uuv_00"].speed_mps = 3.0

    frame = engine.step()
    snapshot = engine.platform_snapshot()
    frame_uuv = next(state for state in frame["uuvs"] if state["platform_id"] == "uuv_00")
    snapshot_uuv = next(state for state in snapshot.roster.uuvs if state.platform_id == "uuv_00")

    for state in (frame_uuv, snapshot_uuv.model_dump()):
        assert state["deployment_state"] == "onboard"
        assert state["position_xy"] == snapshot.carrier.position_xy
        assert state["heading_rad"] == snapshot.carrier.heading_rad
        assert state["speed_mps"] == 0.0


def test_explicit_recovered_uuv_matches_carrier_heading_and_speed(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)
    uuv_id = "uuv_00"
    engine.request_uuv_deployment(uuv_id)
    engine._uuvs[uuv_id].heading_rad = 1.2
    engine._uuvs[uuv_id].speed_mps = 3.0
    engine.request_uuv_recovery(uuv_id)

    frame = engine.step()
    snapshot = engine.platform_snapshot()
    frame_uuv = next(state for state in frame["uuvs"] if state["platform_id"] == uuv_id)
    snapshot_uuv = next(state for state in snapshot.roster.uuvs if state.platform_id == uuv_id)

    for state in (frame_uuv, snapshot_uuv.model_dump()):
        assert state["deployment_state"] == "onboard"
        assert state["position_xy"] == snapshot.carrier.position_xy
        assert state["heading_rad"] == snapshot.carrier.heading_rad
        assert state["speed_mps"] == 0.0


def test_explicit_step_restores_runtime_and_log_after_sink_failure(tmp_path: Path) -> None:
    base = load_app_config(SCENARIO)
    config = base.model_copy(
        update={"timing": base.timing.model_copy(update={"physics_step_s": 30})}
    )
    should_fail = True

    def fail_once(_: dict[str, object]) -> None:
        nonlocal should_fail
        if should_fail:
            should_fail = False
            raise RuntimeError("sink failed")

    engine = SimulationEngine(
        config,
        seed=42,
        output_dir=tmp_path / "failing",
        evaluation_sink=fail_once,
    )
    reference = SimulationEngine(config, seed=42, output_dir=tmp_path / "reference")
    before_carrier_entity = engine._carrier_entity
    before_usvs = engine._usvs
    before_usv = engine._usvs["usv_00"]
    before_uuvs = engine._uuvs
    before_uuv = engine._uuvs["uuv_00"]
    before_targets = engine._targets
    before_snapshot = engine.platform_snapshot()
    before_target = engine._targets["target_00"]
    before_target_state = (
        before_target.position_xy,
        before_target.velocity_xy,
        before_target.intent,
        before_target._desired_heading_rad,
        before_target._desired_speed_mps,
    )
    before_master_rng = engine._master_rng.getstate()
    before_entity_rngs = {key: rng.getstate() for key, rng in engine._entity_rngs.items()}
    before_observer_rngs = {key: rng.getstate() for key, rng in engine._observer_rngs.items()}
    before_quality_rngs = {key: rng.getstate() for key, rng in engine._quality_rngs.items()}
    before_log = engine.logger.path.read_bytes()

    with pytest.raises(RuntimeError, match="sink failed"):
        engine.step()

    target = engine._targets["target_00"]
    assert engine._carrier_entity is before_carrier_entity
    assert engine._usvs is before_usvs
    assert engine._usvs["usv_00"] is before_usv
    assert engine._uuvs is before_uuvs
    assert engine._uuvs["uuv_00"] is before_uuv
    assert engine._targets is before_targets
    assert target is before_target
    assert engine.platform_snapshot() == before_snapshot
    assert (
        target.position_xy,
        target.velocity_xy,
        target.intent,
        target._desired_heading_rad,
        target._desired_speed_mps,
    ) == before_target_state
    assert engine._master_rng.getstate() == before_master_rng
    assert {key: rng.getstate() for key, rng in engine._entity_rngs.items()} == before_entity_rngs
    assert {key: rng.getstate() for key, rng in engine._observer_rngs.items()} == before_observer_rngs
    assert {key: rng.getstate() for key, rng in engine._quality_rngs.items()} == before_quality_rngs
    assert engine._clock.sim_time_s == 0
    assert engine._step_index == 0
    assert engine.platform_snapshot().communication_links == before_snapshot.communication_links
    assert engine.logger.count == 0
    assert engine.logger.path.read_bytes() == before_log

    recovered_frame = engine.step()
    reference_frame = reference.step()
    assert {key: value for key, value in recovered_frame.items() if key != "run_id"} == {
        key: value for key, value in reference_frame.items() if key != "run_id"
    }


def test_explicit_step_keeps_write_error_primary_when_restore_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = load_app_config(SCENARIO)
    config = base.model_copy(
        update={"timing": base.timing.model_copy(update={"physics_step_s": 30})}
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)

    def fail_write(_: dict[str, object]) -> None:
        raise RuntimeError("write failed")

    def fail_restore(_: object) -> None:
        raise RuntimeError("restore failed")

    monkeypatch.setattr(engine.logger, "write", fail_write)
    monkeypatch.setattr(engine.logger, "restore", fail_restore)

    with pytest.raises(RuntimeError, match="write failed") as exc_info:
        engine.step()

    assert isinstance(exc_info.value.__context__, RuntimeError)
    assert str(exc_info.value.__context__) == "restore failed"


def test_explicit_step_propagates_checkpoint_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    def fail_checkpoint() -> object:
        raise RuntimeError("checkpoint failed")

    monkeypatch.setattr(engine.logger, "checkpoint", fail_checkpoint)

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        engine.step()
    assert engine._clock.sim_time_s == 0
    assert engine._step_index == 0


def test_platform_snapshot_never_contains_target_truth(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    payload = engine.platform_snapshot().model_dump()
    frame = engine.step()

    snapshot_rendered = repr(payload).lower()
    frame_rendered = repr(frame).lower()
    assert "target_00" not in snapshot_rendered
    assert "truth" not in snapshot_rendered
    assert "true_position" not in snapshot_rendered
    assert "true_position" not in frame_rendered
    assert "target_truth" not in frame_rendered
    assert "ground_truth" not in frame_rendered


def test_usvs_remain_inside_carrier_support_radius_during_smoke_run(
    tmp_path: Path,
) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    for _ in range(12):
        engine.step()
        snapshot = engine.platform_snapshot()
        assert all(
            usv.distance_to_carrier_m <= snapshot.carrier.support_radius_m
            for usv in snapshot.roster.usvs
        )


def test_onboard_usvs_follow_fast_carrier_without_transit_energy(
    tmp_path: Path,
) -> None:
    config = _small_support_fast_carrier_config()
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    assert config.platforms is not None
    hotel_energy_per_s = config.platforms.motion_profiles["usv_standard"].hotel_energy_per_s
    physics_step_s = config.timing.physics_step_s

    for tick in range(1, 5):
        engine.step()
        snapshot = engine.platform_snapshot()
        for usv in snapshot.roster.usvs:
            assert usv.deployment_state == "onboard"
            assert usv.position_xy == snapshot.carrier.position_xy
            assert usv.heading_rad == snapshot.carrier.heading_rad
            assert usv.speed_mps == 0.0
            assert usv.distance_to_carrier_m == 0.0
            assert usv.energy_fraction == pytest.approx(
                1.0 - tick * physics_step_s * hotel_energy_per_s
            )


def test_deployed_usv_boundary_uses_limited_motion_and_energy(
    tmp_path: Path,
) -> None:
    config = _validated_platform_core_config(
        carrier_updates={
            "speed_mps": 0.5,
            "support_radius_m": 650.0,
            "patrol_route_xy": [
                [-8000.0, -8000.0],
                [-10000.0, -8000.0],
                [-10000.0, -6000.0],
            ],
        },
        motion_updates={"max_speed_mps": 0.5, "max_acceleration_mps2": 0.02},
        deployment_state="deployed",
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    carrier_xy = engine._carrier_entity.position_xy
    engine._usvs["usv_00"].motion = MotionState(
        position_xy=(carrier_xy[0] + 650.0, carrier_xy[1]),
        heading_rad=pi,
        speed_mps=0.3,
    )
    before = engine.platform_snapshot().roster.usvs[0]

    engine.step()

    after = next(
        usv for usv in engine.platform_snapshot().roster.usvs if usv.platform_id == "usv_00"
    )
    actual_displacement_m = hypot(
        after.position_xy[0] - before.position_xy[0],
        after.position_xy[1] - before.position_xy[1],
    )
    actual_heading_rad = atan2(
        after.position_xy[1] - before.position_xy[1],
        after.position_xy[0] - before.position_xy[0],
    )
    assert after.distance_to_carrier_m <= 650.0 + 1e-9
    assert after.distance_to_carrier_m == pytest.approx(650.0, abs=1e-6)
    assert actual_displacement_m <= 0.5 * config.timing.physics_step_s + 1e-9
    assert after.speed_mps == pytest.approx(
        actual_displacement_m / config.timing.physics_step_s
    )
    assert after.speed_mps <= 0.5 + 1e-9
    assert wrap_angle(after.heading_rad - actual_heading_rad) == pytest.approx(0.0)
    assert config.platforms is not None
    motion = config.platforms.motion_profiles["usv_standard"]
    assert (
        abs(after.speed_mps - before.speed_mps)
        <= motion.max_acceleration_mps2 * config.timing.physics_step_s + 1e-9
    )
    assert abs(wrap_angle(after.heading_rad - before.heading_rad)) <= (
        motion.max_turn_rate_rad_s * config.timing.physics_step_s + 1e-9
    )
    assert after.energy_fraction == pytest.approx(
        before.energy_fraction
        - actual_displacement_m * motion.transit_energy_per_m
        - config.timing.physics_step_s * motion.hotel_energy_per_s
    )


def test_deployed_usv_boundary_uses_actual_displacement_heading(tmp_path: Path) -> None:
    config = _validated_platform_core_config(
        carrier_updates={"speed_mps": 0.0, "support_radius_m": 650.0},
        motion_updates={"max_turn_rate_rad_s": 0.1},
        deployment_state="deployed",
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    usv = engine._usvs["usv_00"]
    carrier_xy = engine._carrier_entity.position_xy
    previous_motion = MotionState(
        position_xy=(carrier_xy[0] + 649.0, carrier_xy[1]),
        heading_rad=1.4,
        speed_mps=0.1,
    )
    candidate_motion = advance_motion(
        previous_motion,
        MotionCommand(
            desired_heading_rad=1.0,
            desired_speed_mps=usv.limits.max_speed_mps,
        ),
        usv.limits,
        config.timing.physics_step_s,
    )
    candidate_distance_to_carrier = hypot(
        candidate_motion.position_xy[0] - carrier_xy[0],
        candidate_motion.position_xy[1] - carrier_xy[1],
    )
    assert candidate_distance_to_carrier > engine._carrier_entity.support_radius_m
    usv.motion = candidate_motion

    engine._constrain_usv_to_carrier_support(
        usv,
        previous_motion=previous_motion,
        previous_energy_fraction=1.0,
        dt_s=config.timing.physics_step_s,
    )

    actual_heading_rad = atan2(
        usv.motion.position_xy[1] - previous_motion.position_xy[1],
        usv.motion.position_xy[0] - previous_motion.position_xy[0],
    )
    assert usv.motion.position_xy != candidate_motion.position_xy
    assert abs(wrap_angle(actual_heading_rad - candidate_motion.heading_rad)) > 1e-3
    assert actual_heading_rad == pytest.approx(1.54, abs=0.05)
    assert usv.motion.heading_rad == pytest.approx(actual_heading_rad)
    assert abs(wrap_angle(usv.motion.heading_rad - previous_motion.heading_rad)) <= (
        usv.limits.max_turn_rate_rad_s * config.timing.physics_step_s + 1e-9
    )


def test_deployed_usv_boundary_rejects_actual_heading_beyond_turn_limit(
    tmp_path: Path,
) -> None:
    config = _validated_platform_core_config(
        carrier_updates={"speed_mps": 0.0, "support_radius_m": 650.0},
        motion_updates={"max_turn_rate_rad_s": 0.1},
        deployment_state="deployed",
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    usv = engine._usvs["usv_00"]
    carrier_xy = engine._carrier_entity.position_xy
    previous_motion = MotionState(
        position_xy=(carrier_xy[0] + 649.0, carrier_xy[1]),
        heading_rad=0.0,
        speed_mps=6.5,
    )
    correction_angle_rad = 0.1
    usv.motion = MotionState(
        position_xy=(
            carrier_xy[0] + 660.0 * cos(correction_angle_rad),
            carrier_xy[1] + 660.0 * sin(correction_angle_rad),
        ),
        heading_rad=correction_angle_rad,
        speed_mps=6.5,
    )

    with pytest.raises(RuntimeError, match="turn-rate limit"):
        engine._constrain_usv_to_carrier_support(
            usv,
            previous_motion=previous_motion,
            previous_energy_fraction=1.0,
            dt_s=config.timing.physics_step_s,
        )

    assert usv.motion == previous_motion
    assert usv.energy_fraction == 1.0


def test_deployed_usv_support_constraint_rejects_infeasible_motion(
    tmp_path: Path,
) -> None:
    config = _validated_platform_core_config(
        carrier_updates={"speed_mps": 100.0, "support_radius_m": 650.0},
        deployment_state="deployed",
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)

    with pytest.raises(
        RuntimeError, match="carrier support constraint is infeasible for USV 'usv_01'"
    ):
        engine.step()


def test_infeasible_usv_tick_restores_platform_core_state(tmp_path: Path) -> None:
    config = _validated_platform_core_config(
        carrier_updates={"speed_mps": 100.0, "support_radius_m": 650.0},
        deployment_state="deployed",
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    before_carrier = (
        engine._carrier_entity.position_xy,
        engine._carrier_entity.heading_rad,
        engine._carrier_entity._next_corner_index,
    )
    before_usvs = {
        usv_id: (usv.motion, usv.energy_fraction, usv.command)
        for usv_id, usv in engine._usvs.items()
    }
    before_sim_time_s = engine._clock.sim_time_s
    before_step_index = engine._step_index
    before_events = list(engine._events)
    before_pending_events = list(engine._pending_runtime_events)

    with pytest.raises(RuntimeError, match="carrier support constraint is infeasible"):
        engine.step()

    assert (
        engine._carrier_entity.position_xy,
        engine._carrier_entity.heading_rad,
        engine._carrier_entity._next_corner_index,
    ) == before_carrier
    assert {
        usv_id: (usv.motion, usv.energy_fraction, usv.command)
        for usv_id, usv in engine._usvs.items()
    } == before_usvs
    assert engine._clock.sim_time_s == before_sim_time_s
    assert engine._step_index == before_step_index
    assert engine._events == before_events
    assert engine._pending_runtime_events == before_pending_events


def test_explicit_snapshot_uses_configured_carrier_heading(tmp_path: Path) -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    configured_heading = 0.73
    carrier = config.environment.carrier.model_copy(
        update={"heading_rad": configured_heading}
    )
    environment = config.environment.model_copy(update={"carrier": carrier})
    config = config.model_copy(update={"environment": environment})

    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)

    assert engine.platform_snapshot().carrier.heading_rad == pytest.approx(configured_heading)
