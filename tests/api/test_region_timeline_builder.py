from __future__ import annotations

from underwater_tracking.domain.agent_models import TrackingPlan
from underwater_tracking.domain.regional_models import (
    CommunicationRequirement,
    GridSpec,
    RegionCell,
    RegionTask,
    TargetRegionPlan,
    TimeWindow,
)
from underwater_tracking.api.frame_builder import build_region_timeline


def _regional_plan() -> TargetRegionPlan:
    cells = (
        RegionCell(
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
            first_entry_s=80,
            last_exit_s=120,
            visit_windows=(TimeWindow(start_s=80, end_s=120),),
            occupancy_likelihood=0.7,
            evidence_ids=("obs-1",),
        ),
        RegionCell(
            region_id="T1:cell:1:0",
            target_id="T1",
            grid_x=1,
            grid_y=0,
            min_x=100.0,
            max_x=200.0,
            min_y=0.0,
            max_y=100.0,
            center_xy=(150.0, 50.0),
            cell_size_m=100.0,
            first_entry_s=110,
            last_exit_s=160,
            visit_windows=(TimeWindow(start_s=110, end_s=160),),
            occupancy_likelihood=0.5,
            evidence_ids=("obs-2",),
        ),
    )
    tasks = (
        RegionTask(
            region_id="T1:cell:0:0",
            target_id="T1",
            active_window=TimeWindow(start_s=80, end_s=120),
            priority=0.8,
            required_uuv_count=1,
            uuv_roles=("passive_tracker",),
            required_usv_count=1,
            usv_role="surface_relay",
            assigned_uuv_ids=("uuv-1",),
            assigned_usv_ids=("usv-1",),
            assignment_status="degraded",
            degraded_reasons=("insufficient_uuv",),
            successor_region_id="T1:cell:1:0",
            communication=CommunicationRequirement(),
        ),
        RegionTask(
            region_id="T1:cell:1:0",
            target_id="T1",
            active_window=TimeWindow(start_s=110, end_s=160),
            priority=0.6,
            required_uuv_count=1,
            uuv_roles=("handoff_reserve",),
            assigned_uuv_ids=("uuv-2",),
            predecessor_region_id="T1:cell:0:0",
            assignment_status="planned",
        ),
    )
    return TargetRegionPlan(
        target_id="T1",
        grid_spec=GridSpec(),
        cell_size_m=100.0,
        cells=cells,
        tasks=tasks,
        prediction_id="prediction-1",
        intent_label="patrol",
        intent_confidence=0.9,
        evidence_ids=("obs-1", "obs-2"),
    )


def _tracking_plan() -> TrackingPlan:
    return TrackingPlan(
        plan_id="scenario-1:plan:1",
        scenario_id="scenario-1",
        revision=1,
        base_snapshot_revision=1,
        valid_from_s=100,
        valid_until_s=700,
        regional_plans={"T1": _regional_plan()},
    )


def test_build_region_timeline_uses_current_frame_as_t_plus_zero() -> None:
    timeline = build_region_timeline(_tracking_plan(), sim_time_s=100)
    assert [item.region_id for item in timeline] == [
        "T1:cell:0:0",
        "T1:cell:1:0",
    ]
    assert timeline[0].start_offset_s == -20.0
    assert timeline[0].end_offset_s == 20.0
    assert timeline[1].start_offset_s == 10.0


def test_degraded_region_keeps_reason_assignments_and_handoff() -> None:
    timeline = build_region_timeline(_tracking_plan(), sim_time_s=80)
    item = timeline[0]
    assert item.status == "degraded"
    assert item.degraded_reasons == ("insufficient_uuv",)
    assert item.handoff_to == "T1:cell:1:0"
    assert item.uuv_assignments[0].platform_id == "uuv-1"
    assert item.usv_assignments[0].role == "surface_relay"
