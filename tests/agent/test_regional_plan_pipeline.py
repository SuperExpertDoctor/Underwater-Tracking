"""Regional tasks are the authoritative input to carrier plan projections."""

from pathlib import Path

import pytest

from underwater_tracking.agent.nodes.commit import CommitNode, build_commands, validate_plan
from underwater_tracking.agent.nodes.optimize import (
    OptimizeNode,
    _attach_regional_metadata,
    _materialize_regional_metadata,
)
from underwater_tracking.agent.llm import LLMCallMetadata
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.domain.agent_models import PlanCommand, TrackingPlan, Waypoint
from underwater_tracking.domain.models import (
    DeploymentState,
    GroupQuality,
    GroupReport,
    SituationSnapshot,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    CommunicationCapability,
    MotionLimits,
    PlatformCapability,
    PlatformKind,
    PlatformRoster,
    PlatformSnapshot,
    SonarCapability,
    UUVPlatformState,
    USVPlatformState,
)
from underwater_tracking.domain.regional_models import (
    GridSpec,
    RegionCell,
    RegionTask,
    RegionalPolicy,
    RegionalStrategySet,
    TargetRegionPlan,
    TimeWindow,
    UUVRegionalPolicy,
    UUVRegionalStrategySet,
)
from underwater_tracking.agent.nodes.regions import regional_plan_to_mission_candidates
from underwater_tracking.domain.agent_models import StrategyProposal, StrategySet
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


def _platform_snapshot() -> PlatformSnapshot:
    def capability(kind: PlatformKind) -> PlatformCapability:
        return PlatformCapability(
            kind=kind,
            motion=MotionLimits(
                max_speed_mps=10.0,
                max_acceleration_mps2=1.0,
                max_turn_rate_rad_s=1.0,
            ),
            sonar=SonarCapability(
                passive_range_m=2_000.0,
                passive_bearing_variance_rad2=0.1,
                active_source_range_m=1_000.0,
                active_receive_range_m=1_000.0,
                active_range_sigma_m=5.0,
                active_bearing_sigma_rad=0.1,
                active_capable=True,
                ping_cooldown_s=10,
                ping_energy_cost_fraction=0.1,
                clutter_sensitivity=0.1,
                exposure_cost=0.1,
            ),
            communications=CommunicationCapability(
                surface_range_m=2_000.0,
                acoustic_range_m=1_000.0,
            ),
        )

    return PlatformSnapshot(
        scenario_id="S1",
        sim_time_s=100,
        carrier=CarrierPlatformState(
            carrier_id="carrier-1",
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=0.0,
            support_radius_m=2_000.0,
            onboard_platform_ids=(),
            deployed_platform_ids=("U1", "USV1"),
            returning_platform_ids=(),
        ),
        roster=PlatformRoster(
            uuvs=(
                UUVPlatformState(
                    platform_id="U1",
                    platform_index=0,
                    position_xy=(50.0, 50.0),
                    heading_rad=0.0,
                    speed_mps=4.0,
                    energy_fraction=0.8,
                    deployment_state="deployed",
                    capability=capability(PlatformKind.UUV),
                ),
            ),
            usvs=(
                USVPlatformState(
                    platform_id="USV1",
                    platform_index=0,
                    position_xy=(50.0, 50.0),
                    heading_rad=0.0,
                    speed_mps=4.0,
                    energy_fraction=0.8,
                    deployment_state="deployed",
                    capability=capability(PlatformKind.USV),
                    distance_to_carrier_m=70.0,
                ),
            ),
        ),
        communication_links=(),
    )


def test_optimize_node_uses_authoritative_single_uuv_relay_policy() -> None:
    regional_plan = _single_region_plan()
    policy = RegionalPolicy(
        region_id=regional_plan.tasks[0].region_id,
        coverage_mode="required",
        priority=1.0,
        required_quality=0.8,
        required_uuv_count=1,
        required_usv_count=1,
        uuv_roles=("passive_tracker",),
        usv_role="surface_relay",
        sonar_policy=regional_plan.tasks[0].sonar_policy,
        communication=regional_plan.tasks[0].communication,
        tracking_mode="uuv_primary_usv_relay",
        assigned_uuv_ids=("U1",),
        assigned_usv_ids=("USV1",),
        rationale="one UUV tracks while the selected USV relays",
        evidence_ids=("B:T1:100",),
    )
    situation = _command_snapshot().situation.model_copy(
        update={
            "uuvs": (
                UUVState(
                    uuv_id="U1",
                    position_xy=(50.0, 50.0),
                    heading_rad=0.0,
                    speed_mps=4.0,
                    energy_fraction=0.8,
                    status=UUVStatus.TRACKING,
                    deployment_state=DeploymentState.DEPLOYED,
                ),
            ),
            "platform_snapshot": _platform_snapshot(),
        }
    )
    snapshot = PlanningSnapshot(situation, None, ())
    snapshots = {"regional": snapshot}
    candidates: dict[str, TrackingPlan] = {}
    optimizer = OptimizeNode(
        snapshot_provider=lambda ref: snapshots[ref], store=candidates
    )

    result = optimizer(
        {
            "snapshot_ref": "regional",
            "strategy_set": StrategySet(
                trigger_event_ids=("evt-regional-replan",),
                proposals=(
                    StrategyProposal(
                        concept="balanced",
                        target_priorities={"T1": 1.0},
                        required_quality={"T1": 0.8},
                        reinforcement_policy={"T1": "hold"},
                        releasable_soft_constraints=(),
                        evidence_ids=("B:T1:100",),
                        rationale="compatibility proposal",
                    ),
                )
            ),
            "regional_plans": {"T1": regional_plan},
            "regional_policies": {"T1": RegionalStrategySet(policies=(policy,))},
            "llm_provenance": {
                "regional_strategy:T1": LLMCallMetadata(
                    operation="regional_strategy",
                    model="test-model",
                    prompt_version="regional-v1",
                    request_hash="request-hash",
                    response_hash="response-hash",
                    sim_time_s=100,
                    scenario_id="S1",
                )
            },
        }
    )

    candidate = candidates[result["selected_plan_ref"]]
    task = candidate.region_tasks[regional_plan.tasks[0].region_id]
    assert candidate.status == "degraded"
    assert candidate.member_ids_by_target == {"T1": ("U1",)}
    assert candidate.usv_ids_by_target == {"T1": ("USV1",)}
    assert candidate.predicted_active_count == 1
    assert task.tracking_mode == "uuv_primary_usv_relay"
    assert task.assigned_uuv_ids == ("U1",)
    assert task.assigned_usv_ids == ("USV1",)
    assert "standoff_infeasible:250m" in task.degraded_reasons
    assert candidate.regional_llm_hashes == {
        "T1": ("request-hash", "response-hash")
    }
    assert candidate.trigger_event_ids == ("evt-regional-replan",)


def test_optimize_node_projects_uuv_only_batch_without_usv_members() -> None:
    regional_plan = _single_region_plan()
    candidate_id = regional_plan.cells[0].region_id
    policy = UUVRegionalPolicy(
        candidate_id=candidate_id,
        coverage_mode="required",
        tracking_mode="active_scan",
        priority=1.0,
        required_quality=0.8,
        assigned_uuv_ids=("U1",),
        rationale="the only available UUV covers the candidate",
        evidence_ids=("B:T1:100",),
    )
    platform_snapshot = _platform_snapshot().model_copy(
        update={
            "carrier": _platform_snapshot().carrier.model_copy(
                update={
                    "onboard_platform_ids": (),
                    "deployed_platform_ids": ("U1",),
                }
            ),
            "roster": _platform_snapshot().roster.model_copy(update={"usvs": ()}),
        }
    )
    situation = _command_snapshot().situation.model_copy(
        update={"platform_snapshot": platform_snapshot}
    )
    snapshot = PlanningSnapshot(situation, None, ())
    candidates: dict[str, TrackingPlan] = {}
    optimizer = OptimizeNode(
        snapshot_provider=lambda ref: {"regional": snapshot}[ref], store=candidates
    )

    result = optimizer(
        {
            "snapshot_ref": "regional",
            "strategy_set": StrategySet(
                proposals=(
                    StrategyProposal(
                        concept="balanced",
                        target_priorities={"T1": 1.0},
                        required_quality={"T1": 0.8},
                        reinforcement_policy={"T1": "hold"},
                        releasable_soft_constraints=(),
                        evidence_ids=("B:T1:100",),
                        rationale="uuv-only regional proposal",
                    ),
                )
            ),
            "regional_plans": {"T1": regional_plan},
            "regional_candidates": {
                "T1": regional_plan_to_mission_candidates(regional_plan)
            },
            "regional_policies": {
                "T1": UUVRegionalStrategySet(policies=(policy,))
            },
        }
    )

    executable = result["executable_mission_plan"]
    assert executable.all_uuv_ids == ("U1",)
    assert all(not task.assigned_usv_ids for task in result["region_tasks"].values())


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


def test_plan_validation_accepts_materialized_regional_evidence() -> None:
    regional_plan = _single_region_plan().model_copy(
        update={
            "evidence_ids": ("prediction:T1",),
            "cells": (
                _single_region_plan().cells[0].model_copy(
                    update={"evidence_ids": ("prediction:T1",)}
                ),
            ),
            "tasks": (
                _single_region_plan().tasks[0].model_copy(
                    update={"evidence_ids": ("prediction:T1",)}
                ),
            ),
        }
    )
    candidate = TrackingPlan(
        plan_id="S1:plan:1",
        scenario_id="S1",
        revision=1,
        base_snapshot_revision=1,
        target_priorities={"T1": 1.0},
        required_quality={"T1": 0.0},
        regional_plans={"T1": regional_plan},
        region_tasks={regional_plan.tasks[0].region_id: regional_plan.tasks[0]},
        evidence_ids=("prediction:T1",),
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

    assert not any(issue.code == "evidence_unresolved" for issue in issues)


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
