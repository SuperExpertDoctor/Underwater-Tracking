from __future__ import annotations

import pytest

from underwater_tracking.cli import _mission_controller_for
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.mission_models import RegionMissionState, UUVMissionMode
from underwater_tracking.domain.models import DeploymentState
from underwater_tracking.simulation.engine import SimulationEngine


def _on_boundary(point: tuple[float, float], half_size: float = 2_000.0) -> bool:
    x, y = point
    return (
        abs(abs(x) - half_size) <= 1e-6 and -half_size <= y <= half_size
    ) or (
        abs(abs(y) - half_size) <= 1e-6 and -half_size <= x <= half_size
    )


def test_uuv_exits_and_replacement_enters_through_region_boundary() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = _mission_controller_for(config)
    assert controller is not None
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    region = RegionMissionState(
        region_id="R1",
        target_id="target_00",
        active_scan_uuv_ids=("uuv_00",),
        passive_track_uuv_ids=("uuv_01",),
        region_polygon=(
            (-2_000.0, -2_000.0),
            (2_000.0, -2_000.0),
            (2_000.0, 2_000.0),
            (-2_000.0, 2_000.0),
        ),
        scan_waypoints=((0.0, 0.0),),
    )

    outgoing = "uuv_00"
    engine._deployment_states[outgoing] = DeploymentState.DEPLOYED
    engine._waterborne_uuv_ids.add(outgoing)
    engine._uuvs[outgoing].position_xy = (0.0, 0.0)
    engine._begin_uuv_boundary_exit(outgoing, region, reason="range_reserve")

    exit_point = engine._uuvs[outgoing].waypoints[0]
    assert _on_boundary(exit_point)
    assert engine._deployment_states[outgoing] is DeploymentState.RETURNING
    assert engine._uuv_display_opacity(outgoing) == pytest.approx(1.0)
    engine._uuvs[outgoing].position_xy = (
        exit_point[0] * 0.5,
        exit_point[1] * 0.5,
    )
    assert engine._uuv_display_opacity(outgoing) == pytest.approx(0.5)
    engine._complete_uuv_boundary_exit(outgoing, sim_time_s=30)
    assert engine._uuv_is_physically_exposed(outgoing) is False

    incoming = "uuv_02"
    engine._deploy_uuv_from_region_boundary(incoming, region)
    assert engine._deployment_states[incoming] is DeploymentState.DEPLOYED
    assert engine._uuv_is_physically_exposed(incoming) is True
    assert _on_boundary(engine._uuvs[incoming].position_xy)
    assert engine._uuv_display_opacity(incoming) == pytest.approx(0.0)
    entry_target = engine._uuvs[incoming].waypoints[0]
    start = engine._uuvs[incoming].position_xy
    engine._uuvs[incoming].position_xy = (
        (start[0] + entry_target[0]) * 0.5,
        (start[1] + entry_target[1]) * 0.5,
    )
    assert engine._uuv_display_opacity(incoming) == pytest.approx(0.5)


def test_controller_rotation_reconciles_both_boundary_transitions() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = _mission_controller_for(config)
    assert controller is not None
    engine = SimulationEngine(config, seed=7, mission_controller=controller)
    region = RegionMissionState(
        region_id="R1",
        target_id="target_00",
        active_scan_uuv_ids=("uuv_00",),
        passive_track_uuv_ids=("uuv_01",),
        reserve_uuv_ids=("uuv_02",),
        region_polygon=(
            (-2_000.0, -2_000.0),
            (2_000.0, -2_000.0),
            (2_000.0, 2_000.0),
            (-2_000.0, 2_000.0),
        ),
        scan_waypoints=((0.0, 0.0),),
    )
    controller._regions = {region.region_id: region}
    controller._uuv_modes["uuv_00"] = UUVMissionMode.ACTIVE_SCAN
    engine._deployment_states["uuv_00"] = DeploymentState.DEPLOYED
    engine._waterborne_uuv_ids.add("uuv_00")
    engine._uuvs["uuv_00"].position_xy = (0.0, 0.0)

    controller._return_uuv("uuv_00", "uuv_range_reserve")
    engine._reconcile_uuv_mission_state()

    updated = controller.snapshot().regions[0]
    assert updated.active_scan_uuv_ids == ("uuv_02",)
    assert len(updated.active_scan_uuv_ids + updated.passive_track_uuv_ids) == 2
    assert engine._deployment_states["uuv_00"] is DeploymentState.RETURNING
    assert engine._deployment_states["uuv_02"] is DeploymentState.DEPLOYED
    assert _on_boundary(engine._uuvs["uuv_00"].waypoints[0])
    assert _on_boundary(engine._uuvs["uuv_02"].position_xy)
