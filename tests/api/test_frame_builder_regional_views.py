from __future__ import annotations

import pytest

from underwater_tracking.api.frame_builder import build_operational_frame
from underwater_tracking.domain import (
    GridSpec,
    RegionCell,
    RegionTask,
    TargetRegionPlan,
    TimeWindow,
    TrackingEffectView,
)
from underwater_tracking.domain.models import RuntimeEvent

from tests.api.test_frame_pipeline import _plan, _report, _snapshot


def _regional_plan(target_id: str = "T1") -> TargetRegionPlan:
    cells = tuple(
        RegionCell(
            region_id=f"{target_id}:cell:{index}:0",
            target_id=target_id,
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
            visit_windows=(TimeWindow(start_s=101 + index * 10, end_s=108 + index * 10),),
            occupancy_likelihood=0.75,
            evidence_ids=(f"cell-evidence-{index}",),
            predecessor_region_ids=(f"{target_id}:cell:{index - 1}:0",) if index else (),
            successor_region_ids=(f"{target_id}:cell:{index + 1}:0",)
            if index < 2
            else (),
        )
        for index in range(3)
    )
    tasks = tuple(
        RegionTask(
            region_id=cell.region_id,
            target_id=target_id,
            active_window=TimeWindow(start_s=cell.first_entry_s, end_s=cell.last_exit_s),
            predecessor_region_id=cell.predecessor_region_ids[0]
            if cell.predecessor_region_ids
            else None,
            successor_region_id=cell.successor_region_ids[0]
            if cell.successor_region_ids
            else None,
            assigned_uuv_ids=("UUV-1", "UUV-2"),
            assigned_usv_ids=("USV-1",),
            uuv_roles=("passive_tracker", "handoff_reserve"),
            usv_role="surface_relay",
            sonar_policy={"active_allowed": True, "active_mode": "probe"},
            communication={"carrier_to_uuv": True, "usv_relay_required": True},
            communication_links=("UUV-1->USV-1", "USV-1->carrier-01"),
            evidence_ids=(f"task-evidence-{cell.grid_x}",),
            degraded_reasons=("relay_margin_low",) if cell.grid_x == 2 else (),
            plan_revision=3,
            assignment_status="active",
        )
        for cell in cells
    )
    return TargetRegionPlan(
        target_id=target_id,
        grid_spec=GridSpec(target_grid_cells=64),
        cell_size_m=100.0,
        cells=cells,
        tasks=tasks,
        prediction_id="prediction-1",
        intent_label="transit",
        intent_confidence=0.9,
        evidence_ids=("plan-evidence",),
        plan_revision=3,
    )


def test_frame_builder_exposes_ordered_region_details_handoffs_effects_and_causal_events():
    plan = _plan().model_copy(
        update={
            "regional_plans": {"T1": _regional_plan()},
            "trigger_event_ids": ("evt-replan",),
        }
    )
    snapshot = _snapshot(
        sim_time_s=105,
        reports=(_report("T1", "G1", (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0))),),
    )
    event = RuntimeEvent(
        event_id="evt-replan",
        scenario_id=plan.scenario_id,
        sim_time_s=100,
        event_type="regional_replan",
        level="tactical",
        entity_id="T1",
    )

    frame = build_operational_frame(snapshot, plan, (), (event,), ())

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
    assert regional.grid_spec.target_grid_cells == 64
    assert regional.regions[0].grid_x == 0
    assert regional.regions[0].visit_window == TimeWindow(start_s=101, end_s=108)
    assert regional.regions[0].uuv_roles == ("passive_tracker", "handoff_reserve")
    assert regional.regions[0].usv_role == "surface_relay"
    assert regional.regions[0].sonar_policy.active_mode == "probe"
    assert regional.regions[0].communication_links == (
        "USV-1->carrier-01",
        "UUV-1->USV-1",
    )
    assert regional.regions[2].degraded_reasons == ("relay_margin_low",)
    assert regional.regions[0].evidence_ids == (
        "cell-evidence-0",
        "plan-evidence",
        "task-evidence-0",
    )
    assert regional.current_handoff_region_id == "T1:cell:0:0"
    assert regional.next_handoff_region_id == "T1:cell:1:0"
    assert regional.causal_event_ids == ("evt-replan",)
    assert regional.regions[0].effect.quality_source == "group_quality_proxy"
    assert regional.regions[0].effect.quality_score == 0.89


def test_frame_builder_keeps_legacy_frames_without_regional_plans():
    frame = build_operational_frame(_snapshot(), _plan(), (), (), ())

    assert frame.regional_plans == {}


def test_frame_builder_scopes_causal_events_to_target_group_payload_and_plan_window():
    plan = _plan().model_copy(
        update={
            "regional_plans": {"T1": _regional_plan("T1"), "T2": _regional_plan("T2")},
            "trigger_event_ids": (
                "target-t1",
                "group-t1",
                "payload-t1",
                "global",
                "target-t2",
                "out-of-window",
                "missing",
            ),
        }
    )
    events = (
        RuntimeEvent(
            event_id="target-t1",
            scenario_id=plan.scenario_id,
            sim_time_s=100,
            event_type="replan",
            entity_id="T1",
            level="tactical",
        ),
        RuntimeEvent(
            event_id="group-t1",
            scenario_id=plan.scenario_id,
            sim_time_s=101,
            event_type="replan",
            entity_id="G1",
            level="tactical",
        ),
        RuntimeEvent(
            event_id="payload-t1",
            scenario_id=plan.scenario_id,
            sim_time_s=102,
            event_type="replan",
            entity_id="unscoped",
            level="tactical",
            payload={"target_id": "T1"},
        ),
        RuntimeEvent(
            event_id="global",
            scenario_id=plan.scenario_id,
            sim_time_s=103,
            event_type="replan",
            entity_id=None,
            level="tactical",
        ),
        RuntimeEvent(
            event_id="target-t2",
            scenario_id=plan.scenario_id,
            sim_time_s=104,
            event_type="replan",
            entity_id="T2",
            level="tactical",
        ),
        RuntimeEvent(
            event_id="out-of-window",
            scenario_id=plan.scenario_id,
            sim_time_s=241,
            event_type="replan",
            entity_id="T1",
            level="tactical",
        ),
    )

    frame = build_operational_frame(
        _snapshot(
            reports=(
                _report("T1", "G1", (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0))),
                _report("T2", "G2", (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0))),
            )
        ),
        plan,
        (),
        events,
        (),
    )

    assert frame.regional_plans["T1"].causal_event_ids == (
        "target-t1",
        "group-t1",
        "payload-t1",
        "global",
    )
    assert frame.regional_plans["T2"].causal_event_ids == ("global", "target-t2")


def test_regional_view_rejects_unknown_effect_status():
    with pytest.raises(ValueError, match="status"):
        TrackingEffectView(
            status="handed_off",
            coverage_ratio=0.0,
            quality_score=0.0,
            handoff_progress=0.0,
            quality_source="group_quality_proxy",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("coverage_ratio", -0.01), ("quality_score", 1.01), ("handoff_progress", -0.01)),
)
def test_regional_effect_ratios_are_bounded(field: str, value: float):
    payload = {
        "status": "active",
        "coverage_ratio": 0.5,
        "quality_score": 0.5,
        "handoff_progress": 0.5,
        "quality_source": "group_quality_proxy",
    }
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        TrackingEffectView(**payload)
