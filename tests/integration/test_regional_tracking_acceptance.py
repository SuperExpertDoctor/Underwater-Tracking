from __future__ import annotations

import math
import random

from underwater_tracking.api.frame_builder import build_region_timeline
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.adversary_models import AdversaryEscapeDecision
from underwater_tracking.domain.agent_models import PlanCommand, TrackingPlan
from underwater_tracking.domain.regional_models import (
    CommunicationRequirement,
    GridSpec,
    RegionCell,
    RegionTask,
    TargetRegionPlan,
    TimeWindow,
)
from underwater_tracking.simulation.target import HiddenIntent, TargetEntity
from underwater_tracking.simulation.engine import SimulationEngine
from tests.conftest import CONFIG_PATH


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
            first_entry_s=100 + index * 40,
            last_exit_s=140 + index * 40,
            visit_windows=(
                TimeWindow(start_s=100 + index * 40, end_s=140 + index * 40),
            ),
            occupancy_likelihood=0.8 - index * 0.1,
            evidence_ids=(f"T1-evidence-{index}",),
            predecessor_region_ids=("T1:cell:0:0",) if index == 1 else (),
            successor_region_ids=("T1:cell:1:0",) if index == 0 else (),
        )
        for index in range(2)
    )
    tasks = (
        RegionTask(
            region_id="T1:cell:0:0",
            target_id="T1",
            active_window=TimeWindow(start_s=100, end_s=140),
            required_uuv_count=1,
            tracking_mode="heuristic_uuv",
            uuv_roles=("passive_tracker",),
            assigned_uuv_ids=("uuv-1",),
            communication=CommunicationRequirement(),
            communication_links=("carrier-01->uuv-1",),
            predecessor_region_id=None,
            successor_region_id="T1:cell:1:0",
            evidence_ids=("T1-evidence-0",),
            plan_revision=3,
        ),
        RegionTask(
            region_id="T1:cell:1:0",
            target_id="T1",
            active_window=TimeWindow(start_s=140, end_s=180),
            required_uuv_count=1,
            tracking_mode="heuristic_uuv",
            uuv_roles=("handoff_reserve",),
            assigned_uuv_ids=("uuv-2",),
            communication=CommunicationRequirement(),
            predecessor_region_id="T1:cell:0:0",
            successor_region_id=None,
            evidence_ids=("T1-evidence-1",),
            plan_revision=3,
        ),
    )
    return TargetRegionPlan(
        target_id="T1",
        grid_spec=GridSpec(target_grid_cells=64),
        cell_size_m=100.0,
        cells=cells,
        tasks=tasks,
        prediction_id="T1-prediction-3",
        intent_label="transit",
        intent_confidence=0.86,
        evidence_ids=("T1-prediction-evidence",),
        plan_revision=3,
    )


def _tracking_plan() -> TrackingPlan:
    regional = _regional_plan()
    return TrackingPlan(
        plan_id="scenario-1:plan:3",
        scenario_id="scenario-1",
        revision=3,
        base_snapshot_revision=3,
        valid_from_s=100,
        valid_until_s=300,
        regional_plans={"T1": regional},
        regional_llm_hashes={"T1": ("llm-request-t1-v3", "llm-response-t1-v3")},
        region_tasks={task.region_id: task for task in regional.tasks},
    )


def test_single_target_regional_tracking_acceptance_covers_live_and_replay_contracts() -> None:
    plan = _tracking_plan()
    regional = plan.regional_plans["T1"]

    assert [cell.region_id for cell in regional.cells] == [
        "T1:cell:0:0",
        "T1:cell:1:0",
    ]
    assert len(regional.tasks) == len(regional.cells) == 2
    assert {task.region_id for task in regional.tasks} == set(regional.region_ids)
    assert plan.regional_llm_hashes["T1"] == (
        "llm-request-t1-v3",
        "llm-response-t1-v3",
    )
    assert {task.tracking_mode for task in regional.tasks} == {"heuristic_uuv"}
    assert regional.tasks[0].assigned_uuv_ids == ("uuv-1",)
    assert regional.tasks[1].assigned_uuv_ids == ("uuv-2",)

    live_rows = build_region_timeline(plan, sim_time_s=100)
    replay_rows = build_region_timeline(plan, sim_time_s=160)
    assert [row.region_id for row in live_rows] == [
        "T1:cell:0:0",
        "T1:cell:1:0",
    ]
    assert live_rows[0].handoff_to == "T1:cell:1:0"
    assert live_rows[0].uuv_assignments[0].platform_id == "uuv-1"
    assert not hasattr(live_rows[0], "usv_assignments")
    assert replay_rows[0].start_offset_s == -60.0
    assert replay_rows[1].start_offset_s == -20.0


def test_target_maneuver_prediction_and_blue_response_are_observable(tmp_path) -> None:
    target = TargetEntity(
        target_id="T1",
        position_xy=(0.0, 0.0),
        velocity_xy=(8.0, 0.0),
        intent=HiddenIntent.TRANSIT,
        max_acceleration_mps2=0.08,
        max_turn_rate_rad_s=math.pi / 300.0,
    )
    target.apply_adversary_decision(
        AdversaryEscapeDecision(
            target_id="T1",
            maneuver="course_change",
            intent="evade",
            waypoint=(200.0, 200.0),
            segment="target-owned-current",
            speed=14.0,
            heading=math.pi / 2,
            decoy_action="deploy",
            decoy_count=1,
            confidence=0.8,
            rationale="Target-side detection evidence requires a bounded evasive turn.",
            communications_discipline="silent",
        )
    )
    positions = []
    headings = []
    for _ in range(12):
        target.step(1.0, random.Random(42))
        positions.append(target.position_xy)
        headings.append(math.atan2(target.velocity_xy[1], target.velocity_xy[0]))

    assert target.intent is HiddenIntent.EVADE
    assert positions[-1] != positions[0]
    assert all(
        abs(current - previous) <= target.max_turn_rate_rad_s + 1e-9
        for previous, current in zip(headings, headings[1:])
    )
    assert target.adversary_belief(12).intent_hypothesis == "evade"

    config = load_app_config(CONFIG_PATH)
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.apply_adversary_decision(
        AdversaryEscapeDecision(
            target_id="target_00",
            maneuver="course_change",
            intent="evade",
            waypoint=(100.0, 200.0),
            segment="target-owned-current",
            speed=4.0,
            heading=0.2,
            decoy_action="none",
            decoy_count=0,
            confidence=0.8,
            rationale="Change course after target-side detection evidence.",
            communications_discipline="silent",
        )
    )
    # The blue response is only causal after the target command has produced
    # a physical motion effect; a plan submitted at time zero is not enough.
    engine.step()
    engine.apply_plan_command(
        PlanCommand(
            command_id="regional-blue-response",
            plan_id="regional-blue-response",
            plan_revision=3,
            scenario_id=engine._scenario_id,
            group_id="G-target_00",
            region_id="target_00:cell:0:0",
            target_id="target_00",
            sim_time_s=engine._clock.sim_time_s,
            member_ids=("uuv_00", "uuv_01"),
            actions={"uuv_00": "track", "uuv_01": "track"},
        )
    )
    engine.step()
    events = engine.events()
    phases = [event.payload.get("phase") for event in events if event.entity_id == "target_00"]
    assert {
        "target_maneuver",
        "prediction_revision",
        "regional_task_revision",
        "effect_change",
        "blue_response",
    } <= set(phases)
    assert phases.index("target_maneuver") < phases.index("prediction_revision")
    response = next(event for event in events if event.payload.get("phase") == "blue_response")
    assert response.payload["prediction_revision"] >= 1
    assert response.payload["plan_revision"] == 3
    assert response.payload["latency_s"] >= 0
