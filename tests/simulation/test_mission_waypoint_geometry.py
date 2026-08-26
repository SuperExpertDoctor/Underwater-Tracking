from __future__ import annotations

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.models import DeploymentState
from underwater_tracking.domain.mission_models import (
    RegionLifecycle,
    RegionMissionState,
    UUVMissionMode,
)
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.simulation.engine import SimulationEngine


def test_unconverged_group_spreads_before_following_nominal_scan_route(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(
        config,
        seed=7,
        mission_controller=controller,
        output_dir=tmp_path,
    )
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
        handoff_from="R0",
        scan_waypoints=((100.0, 100.0), (200.0, 200.0)),
        scan_waypoints_by_uuv={
            "uuv_00": ((100.0, 100.0),),
            "uuv_02": ((100.0, 100.0),),
        },
        region_polygon=(
            (-2000.0, -2000.0),
            (2000.0, -2000.0),
            (2000.0, 2000.0),
            (-2000.0, 2000.0),
        ),
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
    assert routes["uuv_02"][-1] == pytest.approx((-900.0, 0.0))

    second_region = region.model_copy(update={"region_id": "R2"})
    engine._plan_mission_group_waypoints(
        snapshot.model_copy(update={"regions": (region, second_region)}),
        second_region,
        ("uuv_00", "uuv_02"),
    )

    assert set(engine._previous_waypoints) == {"R1", "R2"}
