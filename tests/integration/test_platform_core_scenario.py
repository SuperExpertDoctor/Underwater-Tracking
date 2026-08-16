from pathlib import Path
from math import hypot

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AppConfig
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.simulation.kinematics import MotionState


SCENARIO = Path("configs/scenario/segmented_single_target.yaml")


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
        position_xy=(carrier_xy[0] + 642.0, carrier_xy[1]),
        heading_rad=0.0,
        speed_mps=0.2,
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
    assert after.distance_to_carrier_m <= 650.0 + 1e-9
    assert after.distance_to_carrier_m == pytest.approx(650.0, abs=1e-6)
    assert actual_displacement_m < 0.5 * config.timing.physics_step_s
    assert after.speed_mps == pytest.approx(
        actual_displacement_m / config.timing.physics_step_s
    )
    assert after.speed_mps <= 0.5 + 1e-9
    assert config.platforms is not None
    motion = config.platforms.motion_profiles["usv_standard"]
    assert (
        abs(after.speed_mps - before.speed_mps)
        <= motion.max_acceleration_mps2 * config.timing.physics_step_s + 1e-9
    )
    assert after.energy_fraction == pytest.approx(
        before.energy_fraction
        - actual_displacement_m * motion.transit_energy_per_m
        - config.timing.physics_step_s * motion.hotel_energy_per_s
    )


def test_deployed_usv_support_constraint_rejects_infeasible_motion(
    tmp_path: Path,
) -> None:
    config = _validated_platform_core_config(
        carrier_updates={"speed_mps": 100.0, "support_radius_m": 650.0},
        deployment_state="deployed",
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="carrier support constraint is infeasible"):
        engine.step()


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
