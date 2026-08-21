from __future__ import annotations

from underwater_tracking.api.frame_builder import build_region_timeline
from underwater_tracking.domain.regional_models import (
    CommunicationRequirement,
    GridSpec,
    RegionCell,
    RegionTask,
    TargetRegionPlan,
    TimeWindow,
)
from underwater_tracking.domain.agent_models import TrackingPlan


def _tracking_plan() -> TrackingPlan:
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
        first_entry_s=100,
        last_exit_s=140,
        visit_windows=(TimeWindow(start_s=100, end_s=140),),
        occupancy_likelihood=0.8,
    )
    task = RegionTask(
        region_id=cell.region_id,
        target_id="T1",
        active_window=TimeWindow(start_s=100, end_s=140),
        required_uuv_count=1,
        uuv_roles=("passive_tracker",),
        assigned_uuv_ids=("uuv-1",),
        communication=CommunicationRequirement(),
        communication_links=("carrier-01->uuv-1",),
    )
    regional_plan = TargetRegionPlan(
        target_id="T1",
        grid_spec=GridSpec(),
        cell_size_m=100.0,
        cells=(cell,),
        tasks=(task,),
        prediction_id="prediction-1",
        intent_label="patrol",
        intent_confidence=0.9,
    )
    return TrackingPlan(
        plan_id="scenario-1:plan:1",
        scenario_id="scenario-1",
        revision=1,
        base_snapshot_revision=1,
        valid_from_s=100,
        valid_until_s=200,
        regional_plans={"T1": regional_plan},
    )


def test_region_timeline_acceptance_preserves_live_and_replay_offsets() -> None:
    plan = _tracking_plan()
    frame_at_start = build_region_timeline(plan, sim_time_s=100)[0]
    frame_later = build_region_timeline(plan, sim_time_s=120)[0]

    assert frame_at_start.start_offset_s == 0.0
    assert frame_later.start_offset_s == -20.0
    assert frame_at_start.uuv_assignments[0].platform_id == "uuv-1"
    assert not hasattr(frame_at_start, "usv_assignments")
