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
