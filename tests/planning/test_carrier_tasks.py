from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    ExecutableMissionPlan,
    UUVMissionBatch,
)
from underwater_tracking.planning.carrier_tasks import (
    CarrierServiceTask,
    CarrierTaskPlanner,
)


def test_carrier_task_planner_uses_region_perimeter_points_for_deploy_and_recovery() -> None:
    batch = UUVMissionBatch(
        carrier_id="carrier_01",
        candidate_id="T1:r1",
        uuv_ids=("U01", "U02"),
        active_scan_uuv_ids=("U01",),
        passive_track_uuv_ids=("U02",),
        deployment_point=(0.0, 100.0),
        recovery_point=(100.0, 100.0),
        entry_s=100,
        exit_s=200,
    )
    plan = ExecutableMissionPlan(
        revision=3,
        uuv_batches_by_carrier={"carrier_01": (batch,)},
        carrier_missions={
            "carrier_01": CarrierMissionModel(
                carrier_id="carrier_01",
                home_battle_group_id="home",
                ready_uuv_ids=("U01", "U02"),
            )
        },
    )

    tasks = CarrierTaskPlanner().build_tasks(
        plan,
        (plan.carrier_missions["carrier_01"],),
    )

    assert tasks == (
        CarrierServiceTask(
            task_id="deploy:T1:r1",
            candidate_id="T1:r1",
            task_type="deploy",
            point=(0.0, 100.0),
            required_uuv_count=2,
            entry_s=100,
            exit_s=200,
        ),
        CarrierServiceTask(
            task_id="recover:T1:r1",
            candidate_id="T1:r1",
            task_type="recover",
            point=(100.0, 100.0),
            required_uuv_count=2,
            entry_s=200,
            exit_s=300,
        ),
    )


def test_carrier_task_planner_rejects_batch_for_unknown_carrier() -> None:
    batch = UUVMissionBatch(
        carrier_id="carrier_99",
        candidate_id="T1:r1",
        uuv_ids=("U01",),
        active_scan_uuv_ids=("U01",),
        deployment_point=(0.0, 100.0),
        recovery_point=(100.0, 100.0),
        entry_s=100,
        exit_s=200,
    )
    plan = ExecutableMissionPlan(
        revision=1,
        uuv_batches_by_carrier={"carrier_99": (batch,)},
    )

    try:
        CarrierTaskPlanner().build_tasks(plan, ())
    except ValueError as exc:
        assert "unknown carrier" in str(exc)
    else:
        raise AssertionError("expected unknown carrier to be rejected")


def test_carrier_task_planner_materializes_complete_routes_for_each_carrier() -> None:
    first_batch = UUVMissionBatch(
        carrier_id="carrier_01",
        candidate_id="T1:r1",
        uuv_ids=("U01",),
        active_scan_uuv_ids=("U01",),
        deployment_point=(1.0, 0.0),
        recovery_point=(2.0, 0.0),
        entry_s=0,
        exit_s=10,
    )
    second_batch = first_batch.model_copy(
        update={
            "carrier_id": "carrier_02",
            "candidate_id": "T1:r2",
            "uuv_ids": ("U02",),
            "active_scan_uuv_ids": ("U02",),
            "deployment_point": (11.0, 0.0),
            "recovery_point": (12.0, 0.0),
        }
    )
    plan = ExecutableMissionPlan(
        revision=3,
        uuv_batches_by_carrier={
            "carrier_01": (first_batch,),
            "carrier_02": (second_batch,),
        },
        carrier_missions={
            "carrier_01": CarrierMissionModel(
                carrier_id="carrier_01",
                home_battle_group_id="home",
                ready_uuv_ids=("U01",),
            ),
            "carrier_02": CarrierMissionModel(
                carrier_id="carrier_02",
                home_battle_group_id="home",
                ready_uuv_ids=("U02",),
            ),
        },
    )

    routes = CarrierTaskPlanner().build_routes(
        plan,
        tuple(plan.carrier_missions.values()),
        current_positions={"carrier_01": (0.0, 0.0), "carrier_02": (10.0, 0.0)},
        home_positions={"carrier_01": (0.0, 0.0), "carrier_02": (10.0, 0.0)},
        map_bounds=(-1.0, 20.0, -1.0, 5.0),
    )

    assert routes["carrier_01"].route_xy[0] == (0.0, 0.0)
    assert routes["carrier_01"].route_xy[-1] == (0.0, 0.0)
    assert routes["carrier_01"].stop_ids == (
        "deploy:T1:r1",
        "recover:T1:r1",
    )
    assert routes["carrier_02"].route_xy[0] == (10.0, 0.0)
    assert routes["carrier_02"].route_xy[-1] == (10.0, 0.0)
