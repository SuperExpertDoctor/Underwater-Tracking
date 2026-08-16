from pathlib import Path

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.simulation.engine import SimulationEngine


SCENARIO = Path("configs/scenario/segmented_single_target.yaml")


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
