from __future__ import annotations

import json

from underwater_tracking.api.frame_builder import build_uuv_only_frame
from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    ExecutableMissionPlan,
    RegionMissionState,
    UUVMissionBatch,
)
from underwater_tracking.domain.mission_models import PredictionGrid, PredictionGridCell
from underwater_tracking.runtime.mission_controller import MissionController


def _mission() -> ExecutableMissionPlan:
    return ExecutableMissionPlan(
        revision=3,
        uuv_batches_by_carrier={
            "carrier_01": (
                UUVMissionBatch(
                    carrier_id="carrier_01",
                    candidate_id="T1:r3:cell:0:0",
                    uuv_ids=("U1", "U2", "U3"),
                    active_scan_uuv_ids=("U1",),
                    passive_track_uuv_ids=("U2",),
                    reserve_uuv_ids=("U3",),
                    deployment_point=(100.0, 100.0),
                    recovery_point=(200.0, 100.0),
                    entry_s=30,
                    exit_s=120,
                ),
            )
        },
        reserved_uuv_ids=("U4",),
        region_assignments=(
            RegionMissionState(
                region_id="T1:r3:cell:0:0",
                target_id="T1",
                active_scan_uuv_ids=("U1",),
                passive_track_uuv_ids=("U2",),
                reserve_uuv_ids=("U3",),
                coverage=0.8,
                tracking_quality=0.75,
                handoff_to="T1:r3:cell:1:0",
                carrier_task_id="carrier_01:deploy:0",
                plan_revision=3,
            ),
        ),
        carrier_missions={
            "carrier_01": CarrierMissionModel(
                carrier_id="carrier_01",
                home_battle_group_id="home",
                route_xy=((0.0, 0.0), (100.0, 100.0), (0.0, 0.0)),
                stop_ids=("T1:r3:cell:0:0",),
                ready_uuv_ids=("U1", "U2", "U3", "U4"),
            )
        },
    )


def _snapshot():
    controller = MissionController(scenario_id="S1")
    mission = _mission()
    controller.apply_verified_plan(mission)
    return controller.snapshot(), mission


def test_new_uuv_only_frame_has_no_usv_payload() -> None:
    snapshot, mission = _snapshot()
    grid = PredictionGrid(
        target_id="T1",
        revision=3,
        origin=(0.0, 0.0),
        cell_size_m=100.0,
        cells=(
            PredictionGridCell(
                target_id="T1",
                revision=3,
                grid_x=0,
                grid_y=0,
                min_x=0.0,
                max_x=100.0,
                min_y=0.0,
                max_y=100.0,
                cell_size_m=100.0,
                probability=0.9,
                first_entry_s=30,
                last_exit_s=120,
                covariance_summary=(10.0, 20.0, 0.0),
                intent_label="transit",
                intent_confidence=0.8,
            ),
        ),
        centerline_region_ids=("T1:r3:cell:0:0",),
    )

    frame = build_uuv_only_frame(
        snapshot=snapshot,
        mission=mission,
        prediction_grids=(grid,),
        events=snapshot.events,
    )
    payload = frame.model_dump(mode="json")

    assert frame.uuv_only is True
    assert "usvs" not in payload
    assert "USV" not in json.dumps(payload)
    assert frame.prediction_grids[0].cells[0].probability == 0.9
    assert frame.regional_missions[0].carrier_task_id == "carrier_01:deploy:0"
    assert frame.carrier_missions[0].route[-1] == frame.carrier_missions[0].route[0]
    assert frame.uuv_mission_modes["U1"] == "TRANSIT_TO_REGION"
    assert frame.model_validate_json(frame.model_dump_json()) == frame
