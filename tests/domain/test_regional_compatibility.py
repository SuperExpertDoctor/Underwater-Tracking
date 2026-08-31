from __future__ import annotations

from underwater_tracking.domain.agent_models import derive_legacy_views
from underwater_tracking.domain.regional_models import (
    GridSpec,
    RegionCell,
    RegionTask,
    TargetRegionPlan,
    TimeWindow,
)


def _plan() -> TargetRegionPlan:
    cell = RegionCell(
        region_id="T1:cell:0:0",
        target_id="T1",
        grid_x=0,
        grid_y=0,
        min_x=0.0,
        max_x=100.0,
        min_y=0.0,
        max_y=100.0,
        center_xy=(50.0, 50.0),
        cell_size_m=100.0,
        first_entry_s=10,
        last_exit_s=30,
        visit_windows=(TimeWindow(start_s=10, end_s=30),),
    )
    task = RegionTask(
        region_id=cell.region_id,
        target_id=cell.target_id,
        active_window=TimeWindow(start_s=10, end_s=30),
        required_uuv_count=2,
        uuv_roles=("passive_tracker", "handoff_reserve"),
        assigned_uuv_ids=("uuv-b", "uuv-a"),
        assignment_status="active",
    )
    return TargetRegionPlan(
        target_id="T1",
        grid_spec=GridSpec(),
        cell_size_m=100.0,
        cells=(cell,),
        tasks=(task,),
        prediction_id="prediction-1",
        intent_label="patrol",
        intent_confidence=0.9,
    )


def test_regional_tasks_derive_stable_legacy_views() -> None:
    views = derive_legacy_views({"T1": _plan()})
    assert views["member_ids_by_target"] == {"T1": ("uuv-a", "uuv-b")}
    assert views["roles_by_member"] == {
        "uuv-a": "passive_tracker",
        "uuv-b": "handoff_reserve",
    }
    assert views["active_uuv_ids"] == ("uuv-a", "uuv-b")
    assert views["waypoints_by_member"]["uuv-a"][0].arrive_at_s == 10


def test_degraded_regional_task_remains_in_legacy_target_view() -> None:
    plan = _plan()
    degraded = plan.tasks[0].model_copy(
        update={
            "assigned_uuv_ids": ("uuv-a",),
            "assignment_status": "degraded",
            "degraded_reasons": ("insufficient_uuv",),
        }
    )
    degraded_plan = plan.model_copy(update={"tasks": (degraded,)})
    views = derive_legacy_views({"T1": degraded_plan})
    assert views["member_ids_by_target"] == {"T1": ("uuv-a",)}
    assert views["degraded_regions"] == {"T1:cell:0:0": ("insufficient_uuv",)}
