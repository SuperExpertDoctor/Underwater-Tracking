"""Regional tasks are the authoritative input to carrier plan projections."""

from pathlib import Path

import pytest

from underwater_tracking.agent.nodes.commit import CommitNode, build_commands, validate_plan
from underwater_tracking.agent.nodes.optimize import (
    _attach_regional_metadata,
    _materialize_regional_metadata,
)
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.domain.agent_models import PlanCommand, TrackingPlan, Waypoint
from underwater_tracking.domain.models import (
    GroupQuality,
    GroupReport,
    SituationSnapshot,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.domain.regional_models import (
    GridSpec,
    RegionCell,
    RegionTask,
    TargetRegionPlan,
    TimeWindow,
)
from underwater_tracking.persistence.plans import PlanRepository


def _region_plan() -> TargetRegionPlan:
    cells = tuple(
        RegionCell(
            region_id=f"T1:cell:{index}:0",
            target_id="T1",
            grid_x=index,
            grid_y=0,
            min_x=float(index * 100),
            max_x=float((index + 1) * 100),
            min_y=0.0,
            max_y=100.0,
            center_xy=(float(index * 100 + 50), 50.0),
            cell_size_m=100.0,
            first_entry_s=100 + index * 100,
            last_exit_s=200 + index * 100,
        )
        for index in range(3)
    )
    tasks = tuple(
        RegionTask(
            region_id=cell.region_id,
            target_id="T1",
            active_window=TimeWindow(
                start_s=cell.first_entry_s,
                end_s=cell.last_exit_s,
            ),
        )
        for cell in cells
    )
    return TargetRegionPlan(
        target_id="T1",
        grid_spec=GridSpec(),
        cell_size_m=100.0,
        cells=cells,
        tasks=tasks,
        prediction_id="prediction:T1",
        intent_label="patrol",
        intent_confidence=0.8,
    )


def _single_region_plan() -> TargetRegionPlan:
    regional_plan = _region_plan()
    return regional_plan.model_copy(
        update={
            "cells": (regional_plan.cells[0],),
            "tasks": (regional_plan.tasks[0],),
        }
    )


def test_regional_tasks_override_legacy_projections_and_retain_uncovered_regions() -> None:
    regional_plan = _region_plan()
    active, degraded, uncovered = regional_plan.tasks
    region_tasks = {
        active.region_id: active.model_copy(
            update={
                "assigned_uuv_ids": ("U1",),
                "uuv_roles": ("passive_tracker",),
                "assignment_status": "active",
                "required_quality": 0.8,
            }
        ),
        degraded.region_id: degraded.model_copy(
            update={
                "assigned_uuv_ids": ("U2",),
                "uuv_roles": ("handoff_reserve",),
                "assignment_status": "degraded",
                "degraded_reasons": ("missing_usv_relay",),
                "required_quality": 0.7,
            }
        ),
        uncovered.region_id: uncovered.model_copy(
            update={
                "assignment_status": "uncovered",
                "degraded_reasons": ("missing_uuv_tracking_owner",),
                "required_quality": 0.6,
            }
        ),
    }
    legacy_candidate = TrackingPlan(
        plan_id="S1:plan:1",
        scenario_id="S1",
        revision=1,
        base_snapshot_revision=1,
        member_ids_by_target={"T1": ("LEGACY",)},
        roles_by_member={"LEGACY": "lead"},
        waypoints_by_member={"LEGACY": (Waypoint(x=0.0, y=0.0),)},
    )

    candidate = _attach_regional_metadata(
        legacy_candidate,
        {"T1": regional_plan},
        region_tasks,
    )

    assert candidate.member_ids_by_target == {"T1": ("U1", "U2")}
    assert candidate.roles_by_member == {
        "U1": "passive_tracker",
        "U2": "handoff_reserve",
    }
    assert candidate.waypoints_by_member == {
        "U1": (Waypoint(x=50.0, y=50.0, arrive_at_s=100),),
        "U2": (Waypoint(x=150.0, y=50.0, arrive_at_s=200),),
    }
    assert candidate.region_tasks[uncovered.region_id].assignment_status == "uncovered"
    assert candidate.regional_metrics.uncovered_region_ids == (uncovered.region_id,)
    assert candidate.regional_metrics.degraded_regions == {
        degraded.region_id: ("missing_usv_relay",),
        uncovered.region_id: ("missing_uuv_tracking_owner",),
    }
    assert candidate.regional_metrics.regional_quality_by_region == {
        active.region_id: 0.8,
        degraded.region_id: 0.7,
        uncovered.region_id: 0.6,
    }
    assert candidate.regional_metrics.coverage_rate == 2 / 3
    assert candidate.regional_metrics.relay_links_by_region == {
        active.region_id: (),
        degraded.region_id: (),
        uncovered.region_id: (),
    }
    assert candidate.regional_metrics.metrics_are_planning_proxies is True


def test_plan_command_keeps_optional_region_id_with_legacy_execution_fields() -> None:
    assert "region_id" in PlanCommand.model_fields
    command = PlanCommand(
        command_id="S1:plan:1:region:T1:cell:0:0",
        plan_id="S1:plan:1",
        plan_revision=1,
        scenario_id="S1",
        group_id="G-T1",
        region_id="T1:cell:0:0",
        target_id="T1",
        sim_time_s=100,
        member_ids=("U1",),
        waypoints_by_member={"U1": (Waypoint(x=50.0, y=50.0, arrive_at_s=100),)},
        actions={"U1": "track"},
    )

    assert command.region_id == "T1:cell:0:0"
    assert command.group_id == "G-T1"
    assert command.actions == {"U1": "track"}


def test_plan_validation_surfaces_unknown_regional_tasks() -> None:
    regional_plan = _region_plan()
    candidate = TrackingPlan(
        plan_id="S1:plan:1",
        scenario_id="S1",
        revision=1,
        base_snapshot_revision=1,
        regional_plans={"T1": regional_plan},
        region_tasks={
            "T1:cell:9:0": RegionTask(
                region_id="T1:cell:9:0",
                target_id="T1",
                active_window=TimeWindow(start_s=100, end_s=200),
            )
        },
    )
    snapshot = PlanningSnapshot(
        situation=SituationSnapshot(
            scenario_id="S1",
            snapshot_revision=1,
            sim_time_s=100,
            uuvs=(),
            group_reports=(),
            pending_events=(),
        ),
        active_plan=None,
        applied_directives=(),
    )

    issues = validate_plan(snapshot, candidate)

    assert [(issue.code, issue.field) for issue in issues] == [
        ("regional_unknown_region", "region_tasks[T1:cell:9:0]"),
    ]


def _command_snapshot() -> PlanningSnapshot:
    return PlanningSnapshot(
        situation=SituationSnapshot(
            scenario_id="S1",
            snapshot_revision=1,
            sim_time_s=100,
            uuvs=(),
            group_reports=(
                GroupReport(
                    group_id="G-T1",
                    target_id="T1",
                    sim_time_s=100,
                    member_ids=("U1",),
                    belief=TargetBelief(
                        target_id="T1",
                        sim_time_s=100,
                        mean=(0.0, 0.0),
                        covariance=((1.0, 0.0), (0.0, 1.0)),
                        model_probabilities={"cv": 1.0},
                    ),
                    quality=GroupQuality(
                        instant=0.8,
                        window_mean=0.8,
                        ewma=0.8,
                        components={},
                    ),
                    plan_revision=1,
                ),
            ),
            pending_events=(),
        ),
        active_plan=None,
        applied_directives=(),
    )


def test_uuv_relay_task_keeps_usv_in_projection_and_region_command() -> None:
    assert "usv_ids_by_target" in TrackingPlan.model_fields
    assert "usv_ids" in PlanCommand.model_fields
    regional_plan = _region_plan()
    relay_task = regional_plan.tasks[0].model_copy(
        update={
            "assigned_uuv_ids": ("U1",),
            "uuv_roles": ("passive_tracker",),
            "assigned_usv_ids": ("USV1",),
            "usv_role": "surface_relay",
            "assignment_status": "active",
        }
    )
    candidate = _attach_regional_metadata(
        TrackingPlan(
            plan_id="S1:plan:1",
            scenario_id="S1",
            revision=1,
            base_snapshot_revision=1,
        ),
        {"T1": regional_plan},
        {relay_task.region_id: relay_task},
    )

    command = build_commands(_command_snapshot(), candidate)[0]

    assert candidate.member_ids_by_target == {"T1": ("U1",)}
    assert candidate.usv_ids_by_target == {"T1": ("USV1",)}
    assert candidate.roles_by_member["USV1"] == "surface_relay"
    assert candidate.waypoints_by_member["USV1"] == (
        Waypoint(x=50.0, y=50.0, arrive_at_s=100),
    )
    assert command.region_id == relay_task.region_id
    assert command.usv_ids == ("USV1",)
    assert command.usv_roles_by_member == {"USV1": "surface_relay"}
    assert command.usv_actions == {"USV1": "relay"}
    assert command.waypoints_by_member["USV1"] == (
        Waypoint(x=50.0, y=50.0, arrive_at_s=100),
    )


def test_heuristic_usv_task_builds_executable_command_with_region_id() -> None:
    regional_plan = _region_plan()
    usv_task = regional_plan.tasks[0].model_copy(
        update={
            "tracking_mode": "heuristic_usv",
            "assigned_uuv_ids": (),
            "assigned_usv_ids": ("USV1",),
            "usv_role": "active_tracker",
            "assignment_status": "active",
        }
    )
    candidate = _attach_regional_metadata(
        TrackingPlan(
            plan_id="S1:plan:1",
            scenario_id="S1",
            revision=1,
            base_snapshot_revision=1,
        ),
        {"T1": regional_plan},
        {usv_task.region_id: usv_task},
    )

    commands = build_commands(_command_snapshot(), candidate)

    assert len(commands) == 1
    command = commands[0]
    assert command.member_ids == ()
    assert command.usv_ids == ("USV1",)
    assert command.usv_actions == {"USV1": "track"}
    assert command.waypoints_by_member == {
        "USV1": (Waypoint(x=50.0, y=50.0, arrive_at_s=100),)
    }
    assert command.region_id == usv_task.region_id


def test_commit_accepts_one_uuv_authoritative_regional_task(tmp_path: Path) -> None:
    regional_plan = _single_region_plan()
    task = regional_plan.tasks[0].model_copy(
        update={
            "tracking_mode": "heuristic_uuv",
            "assigned_uuv_ids": ("U1",),
            "uuv_roles": ("passive_tracker",),
            "assignment_status": "active",
        }
    )
    candidate = _attach_regional_metadata(
        TrackingPlan(
            plan_id="S1:plan:one-uuv",
            scenario_id="S1",
            revision=1,
            base_snapshot_revision=1,
        ),
        {"T1": regional_plan},
        {task.region_id: task},
    )
    snapshot = _command_snapshot().situation.model_copy(
        update={
            "uuvs": (
                UUVState(
                    uuv_id="U1",
                    position_xy=(50.0, 50.0),
                    heading_rad=0.0,
                    speed_mps=4.0,
                    energy_fraction=0.8,
                    status=UUVStatus.TRACKING,
                ),
            )
        }
    )
    repository = PlanRepository(tmp_path / "one-uuv.db")
    repository.set_snapshot_revision("S1", 1)
    try:
        result = CommitNode(
            repository=repository,
            snapshot_provider=lambda _: PlanningSnapshot(snapshot, None, ()),
        )({"snapshot_ref": "regional"}, candidate)

        assert result["commit_status"] == "committed"
        assert repository.list_commands(candidate.plan_id)[0].member_ids == ("U1",)
    finally:
        repository.close()


def test_commit_accepts_usv_only_heuristic_regional_task(tmp_path: Path) -> None:
    regional_plan = _single_region_plan()
    task = regional_plan.tasks[0].model_copy(
        update={
            "tracking_mode": "heuristic_usv",
            "assigned_uuv_ids": (),
            "assigned_usv_ids": ("USV1",),
            "usv_role": "active_tracker",
            "assignment_status": "active",
        }
    )
    candidate = _attach_regional_metadata(
        TrackingPlan(
            plan_id="S1:plan:usv-only",
            scenario_id="S1",
            revision=1,
            base_snapshot_revision=1,
        ),
        {"T1": regional_plan},
        {task.region_id: task},
    )
    repository = PlanRepository(tmp_path / "usv-only.db")
    repository.set_snapshot_revision("S1", 1)
    try:
        result = CommitNode(
            repository=repository,
            snapshot_provider=lambda _: _command_snapshot(),
        )({"snapshot_ref": "regional"}, candidate)

        assert result["commit_status"] == "committed"
        command = repository.list_commands(candidate.plan_id)[0]
        assert command.member_ids == ()
        assert command.usv_ids == ("USV1",)
        assert command.region_id == task.region_id
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("strategy", "reason"),
    [(None, "regional_policy_missing"), ("malformed", "regional_policy_invalid")],
)
def test_missing_or_malformed_regional_policy_preserves_uncovered_tasks(
    strategy: object,
    reason: str,
) -> None:
    regional_plan = _region_plan()
    snapshot = PlanningSnapshot(
        situation=SituationSnapshot(
            scenario_id="S1",
            snapshot_revision=1,
            sim_time_s=100,
            uuvs=(),
            group_reports=(),
            pending_events=(),
        ),
        active_plan=None,
        applied_directives=(),
    )

    materialized, tasks = _materialize_regional_metadata(
        snapshot,
        {
            "regional_plans": {"T1": regional_plan},
            "regional_policies": ({"T1": strategy} if strategy is not None else {}),
        },
    )

    assert tuple(materialized) == ("T1",)
    assert tuple(tasks) == regional_plan.region_ids
    assert all(task.assignment_status == "uncovered" for task in tasks.values())
    assert all(task.degraded_reasons == (reason,) for task in tasks.values())
