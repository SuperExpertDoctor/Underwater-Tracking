from __future__ import annotations

import json

from underwater_tracking.api.frame_builder import build_operational_frame, build_uuv_only_frame
from underwater_tracking.cli import _mission_controller_for
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    CarrierRouteStatus,
    ExecutableMissionPlan,
    PredictionGrid,
    PredictionGridCell,
    RegionMissionState,
    UUVMissionBatch,
)
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.simulation.engine import SimulationEngine


def test_default_live_initial_frame_exposes_known_submarine_without_prior_or_group() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = _mission_controller_for(config)
    assert controller is not None
    engine = SimulationEngine(config, seed=42, mission_controller=controller)

    frame = build_operational_frame(
        engine.publication_situation(),
        plan=None,
        ledger_tail=(),
        events=(),
        metrics=(),
        mission_snapshot=controller.snapshot(),
        uuv_only=True,
    )

    assert len(frame.target_estimates) == 1
    assert frame.target_estimates[0].target_id == "target_00"
    assert frame.target_estimates[0].classification == "submarine"
    assert frame.target_estimates[0].prediction is not None
    assert frame.target_estimates[0].prediction.centerline_xy[0] == frame.target_estimates[0].mean
    assert frame.target_estimates[0].prediction.centerline_xy[-1] != frame.target_estimates[0].mean
    assert "target_priors" not in frame.model_dump()
    assert frame.groups == ()
    assert frame.execution_groups == ()
    assert frame.planned_assignments == ()
    assert len(frame.uuv_resources) == 12
    assert {brain.role: brain.status for brain in frame.brains} == {
        "master": "ready",
        "slave": "ready",
        "adversary": "ready",
    }
    assert all(not brain.connected_platform_ids for brain in frame.brains)


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
    controller.advance(
        0,
        {
            "mileage_m": {"U1": 123.0},
            "energy_fraction": {"U1": 0.8},
            "uuv_capability_active": {"U1": True},
        },
    )
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
    assert frame.regional_missions[0].carrier_task_id is None
    assert frame.carrier_missions == ()
    assert frame.uuv_mission_modes["U1"] == "TRANSIT_TO_REGION"
    assert frame.uuv_resources[0].uuv_id == "U1"
    assert frame.uuv_resources[0].mileage_m == 123.0
    assert frame.model_validate_json(frame.model_dump_json()) == frame


def test_uuv_only_frame_publishes_situation_scenario_id_when_available() -> None:
    snapshot, mission = _snapshot()
    situation = SituationSnapshot(
        scenario_id="authoritative-scenario",
        snapshot_revision=1,
        sim_time_s=snapshot.sim_time_s,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )

    frame = build_uuv_only_frame(
        snapshot=snapshot,
        mission=mission,
        situation=situation,
    )

    assert frame.scenario_id == "authoritative-scenario"


def test_uuv_only_frame_publishes_mission_scenario_id_without_situation() -> None:
    snapshot, mission = _snapshot()

    frame = build_uuv_only_frame(snapshot=snapshot, mission=mission)

    assert frame.scenario_id == snapshot.scenario_id


def test_live_uuv_only_frame_omits_all_carriers() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    situation = SimulationEngine(config, seed=7).publication_situation()

    frame = build_operational_frame(
        situation,
        plan=None,
        ledger_tail=(),
        events=(),
        metrics=(),
        uuv_only=True,
    )

    assert frame.carrier is None
    assert frame.carriers == ()
    assert frame.carrier_missions == ()


def test_initial_operational_frame_marks_all_uuvs_not_physically_exposed() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    situation = SimulationEngine(config, seed=7).publication_situation()
    frame = build_operational_frame(
        situation,
        plan=None,
        ledger_tail=(),
        events=situation.pending_events,
        metrics=(),
        uuv_only=True,
    )

    assert len(frame.uuvs) == 12
    assert all(uuv.physically_exposed is False for uuv in frame.uuvs)


def test_uuv_only_frame_omits_carrier_route_status() -> None:
    snapshot, mission = _snapshot()
    blocked = mission.carrier_missions["carrier_01"].model_copy(
        update={"route_status": CarrierRouteStatus.RENDEZVOUS_BLOCKED}
    )
    snapshot = snapshot.model_copy(
        update={"carrier_missions": {"carrier_01": blocked}}
    )

    frame = build_uuv_only_frame(snapshot=snapshot, mission=mission)

    assert frame.carrier_missions == ()
