from __future__ import annotations

from math import hypot

from underwater_tracking.domain.agent_models import IntentHypothesis, PredictedTrackRef
from underwater_tracking.domain.regional_models import GridSpec
from underwater_tracking.planning.regions import generate_target_region_plan
from underwater_tracking.planning.regional_allocation import allocate_regional_tasks
from tests.planning.test_regional_allocation import _plan, _roster


def _plan_with_cell_size(cell_size_m: float):
    base = _plan()
    cell = base.cells[0].model_copy(
        update={
            "max_x": cell_size_m,
            "max_y": cell_size_m,
            "center_xy": (cell_size_m / 2.0, cell_size_m / 2.0),
            "cell_size_m": cell_size_m,
            "predicted_target_xy": (cell_size_m / 2.0, cell_size_m / 2.0),
        }
    )
    return base.model_copy(update={"cells": (cell,)})


def test_regional_waypoint_respects_minimum_target_standoff() -> None:
    result = allocate_regional_tasks(_plan_with_cell_size(600.0), _roster(2))

    waypoint = result.waypoints_by_member["uuv-0"][0]
    distance = hypot(waypoint.x - 300.0, waypoint.y - 300.0)
    assert distance >= 250.0
    assert result.tasks["T1:cell:0:0"].assignment_status == "active"
    assert not any("standoff_infeasible" in issue for issue in result.issues)


def test_small_cell_falls_back_to_center_with_explicit_degradation() -> None:
    result = allocate_regional_tasks(_plan_with_cell_size(100.0), _roster(2))

    task = result.tasks["T1:cell:0:0"]
    waypoint = result.waypoints_by_member["uuv-0"][0]
    assert (waypoint.x, waypoint.y) == (50.0, 50.0)
    assert task.assignment_status == "degraded"
    assert "standoff_infeasible:250m" in task.degraded_reasons
    assert "T1:cell:0:0:standoff_infeasible:250m" in result.issues


def test_region_generation_injects_required_quality() -> None:
    prediction = PredictedTrackRef(
        prediction_id="pred:T1",
        target_id="T1",
        sim_time_s=0,
        horizon_s=100.0,
        sample_step_s=100.0,
        times_s=(0.0,),
        points_xy=((300.0, 300.0),),
        corridor_radius_m=(10.0,),
    )
    intent = IntentHypothesis(
        label="patrol",
        confidence=0.8,
        evidence_ids=("intent:T1",),
        model_id="fake",
        prompt_version="test",
    )

    plan = generate_target_region_plan(
        prediction,
        intent,
        (0.0, 600.0, 0.0, 600.0),
        GridSpec(
            target_grid_cells=1,
            min_cell_size_m=600.0,
            max_cell_size_m=600.0,
            cell_size_rounding_m=50.0,
        ),
        required_quality=0.72,
    )

    assert {task.required_quality for task in plan.tasks} == {0.72}
    assert plan.cells[0].predicted_target_xy == (300.0, 300.0)
