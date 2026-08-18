from __future__ import annotations

from underwater_tracking.api.frame_builder import build_operational_frame
from underwater_tracking.domain import (
    GridSpec,
    RegionCell,
    RegionTask,
    TargetRegionPlan,
    TimeWindow,
)

from tests.api.test_frame_pipeline import _plan, _report, _snapshot


def _regional_plan() -> TargetRegionPlan:
    cells = tuple(
        RegionCell(
            region_id=f"T1:cell:{index}:0",
            target_id="T1",
            grid_x=index,
            grid_y=0,
            min_x=index * 100.0,
            max_x=(index + 1) * 100.0,
            min_y=0.0,
            max_y=100.0,
            center_xy=(index * 100.0 + 50.0, 50.0),
            cell_size_m=100.0,
            first_entry_s=100 + index * 10,
            last_exit_s=110 + index * 10,
            predecessor_region_ids=(f"T1:cell:{index - 1}:0",) if index else (),
            successor_region_ids=(f"T1:cell:{index + 1}:0",) if index < 2 else (),
        )
        for index in range(3)
    )
    tasks = tuple(
        RegionTask(
            region_id=cell.region_id,
            target_id="T1",
            active_window=TimeWindow(start_s=cell.first_entry_s, end_s=cell.last_exit_s),
            predecessor_region_id=cell.predecessor_region_ids[0]
            if cell.predecessor_region_ids
            else None,
            successor_region_id=cell.successor_region_ids[0]
            if cell.successor_region_ids
            else None,
            assigned_uuv_ids=("UUV-1", "UUV-2"),
            assigned_usv_ids=("USV-1",),
            usv_role="surface_relay",
            assignment_status="active",
        )
        for cell in cells
    )
    return TargetRegionPlan(
        target_id="T1",
        grid_spec=GridSpec(target_grid_cells=64),
        cell_size_m=100.0,
        cells=cells,
        tasks=tasks,
        prediction_id="prediction-1",
        intent_label="transit",
        intent_confidence=0.9,
        plan_revision=3,
    )


def test_frame_builder_exposes_ordered_regions_members_and_effect_proxy():
    plan = _plan().model_copy(
        update={"regional_plans": {"T1": _regional_plan()}}
    )
    snapshot = _snapshot(reports=(_report("T1", "G1", (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0))),))

    frame = build_operational_frame(snapshot, plan, (), (), ())

    regional = frame.regional_plans["T1"]
    assert [region.display_name for region in regional.regions] == [
        "region_1",
        "region_2",
        "region_3",
    ]
    assert regional.regions[0].successor_region_ids == ("T1:cell:1:0",)
    assert regional.regions[1].predecessor_region_ids == ("T1:cell:0:0",)
    assert regional.regions[0].assigned_uuv_ids == ("UUV-1", "UUV-2")
    assert regional.regions[0].assigned_usv_ids == ("USV-1",)
    assert regional.regions[0].tracking_mode == "uuv_primary_usv_relay"
    assert regional.regions[0].relay_usv_ids == ("USV-1",)
    assert regional.regions[0].effect.quality_source == "group_quality_proxy"
    assert regional.regions[0].effect.quality_score == 0.89


def test_frame_builder_keeps_legacy_frames_without_regional_plans():
    frame = build_operational_frame(_snapshot(), _plan(), (), (), ())

    assert frame.regional_plans == {}
