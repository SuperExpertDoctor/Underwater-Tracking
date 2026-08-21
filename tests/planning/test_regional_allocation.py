from __future__ import annotations

import pytest

from underwater_tracking.domain.platforms import (
    CommunicationCapability,
    MotionLimits,
    PlatformCapability,
    PlatformKind,
    PlatformRoster,
    SonarCapability,
    UUVPlatformState,
)
from underwater_tracking.domain.regional_models import (
    CommunicationRequirement,
    GridSpec,
    RegionCell,
    RegionTask,
    RegionalPolicy,
    RegionalStrategySet,
    SonarPolicy,
    TargetRegionPlan,
    TimeWindow,
)
from underwater_tracking.planning.regional_allocation import (
    allocate_regional_tasks,
    materialize_regional_plan,
)


def _capability(kind: PlatformKind) -> PlatformCapability:
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


def _roster(uuv_count: int) -> PlatformRoster:
    uuvs = tuple(
        UUVPlatformState(
            platform_id=f"uuv-{index}",
            platform_index=index,
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=2.0,
            energy_fraction=0.9,
            deployment_state="deployed",
            capability=_capability(PlatformKind.UUV),
            master_connected=True,
        )
        for index in range(uuv_count)
    )
    return PlatformRoster(usvs=(), uuvs=uuvs)


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
        first_entry_s=0,
        last_exit_s=100,
        visit_windows=(TimeWindow(start_s=0, end_s=100),),
    )
    task = RegionTask(
        region_id=cell.region_id,
        target_id="T1",
        active_window=TimeWindow(start_s=0, end_s=100),
        required_uuv_count=2,
        uuv_roles=("passive_tracker", "handoff_reserve"),
        assigned_uuv_ids=("uuv-0", "uuv-1"),
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


def test_missing_platform_degrades_region_instead_of_dropping_it() -> None:
    result = allocate_regional_tasks(_plan(), _roster(uuv_count=1))
    task = result.tasks["T1:cell:0:0"]
    assert task.assignment_status == "degraded"
    assert task.assigned_uuv_ids == ("uuv-0",)
    assert task.degraded_reasons


def test_allocation_is_deterministic_and_assigns_roles() -> None:
    first = allocate_regional_tasks(_plan(), _roster(uuv_count=2))
    second = allocate_regional_tasks(_plan(), _roster(uuv_count=2))
    assert first.tasks == second.tasks
    assert first.tasks["T1:cell:0:0"].assignment_status == "degraded"
    assert "standoff_infeasible:250m" in first.tasks["T1:cell:0:0"].degraded_reasons
    assert first.tasks["T1:cell:0:0"].assigned_uuv_ids == ("uuv-0", "uuv-1")
    assert first.waypoints_by_member["uuv-0"]


def test_allocation_preserves_llm_members_without_filling_advisory_counts() -> None:
    task = _plan().tasks[0].model_copy(
        update={
            "required_uuv_count": 0,
            "uuv_roles": (),
            "assigned_uuv_ids": ("uuv-0",),
            "assignment_status": "active",
        }
    )
    plan = _plan().model_copy(update={"tasks": (task,)})

    result = allocate_regional_tasks(plan, _roster(uuv_count=2))

    allocated = result.tasks[task.region_id]
    assert allocated.assigned_uuv_ids == ("uuv-0",)
    assert "insufficient_uuv" not in result.issues


def test_empty_llm_membership_is_uncovered_without_automatic_selection() -> None:
    task = _plan().tasks[0].model_copy(
        update={
            "tracking_mode": "heuristic_uuv",
            "required_uuv_count": 0,
            "uuv_roles": (),
            "assigned_uuv_ids": (),
            "assignment_status": "planned",
        }
    )

    result = allocate_regional_tasks(
        _plan().model_copy(update={"tasks": (task,)}), _roster(uuv_count=2)
    )

    assert result.tasks[task.region_id].assignment_status == "uncovered"
    assert result.tasks[task.region_id].assigned_uuv_ids == ()


def test_llm_uuv_mode_preserves_only_explicit_uuv_members() -> None:
    policy = RegionalPolicy(
        region_id="T1:cell:0:0",
        coverage_mode="required",
        priority=1.0,
        required_quality=0.7,
        required_uuv_count=2,
        uuv_roles=("passive_tracker", "passive_tracker"),
        sonar_policy=SonarPolicy(passive_required=True),
        communication=CommunicationRequirement(),
        rationale="underwater passive coverage",
        evidence_ids=("prediction-1",),
        tracking_mode="heuristic_uuv",
        assigned_uuv_ids=("uuv-0", "uuv-1"),
    )
    result = materialize_regional_plan(
        _plan(), RegionalStrategySet(policies=(policy,)), _roster(uuv_count=2)
    )

    task = result.tasks["T1:cell:0:0"]
    assert task.tracking_mode == "heuristic_uuv"
    assert task.assigned_uuv_ids == ("uuv-0", "uuv-1")


def test_empty_uuv_policy_remains_uncovered_without_automatic_selection() -> None:
    policy = RegionalPolicy(
        region_id="T1:cell:0:0",
        coverage_mode="required",
        priority=1.0,
        required_quality=0.7,
        required_uuv_count=0,
        sonar_policy=SonarPolicy(passive_required=True),
        communication=CommunicationRequirement(),
        rationale="no UUV was selected for this region",
        evidence_ids=("prediction-1",),
        tracking_mode="heuristic_uuv",
        assigned_uuv_ids=(),
    )
    result = materialize_regional_plan(
        _plan(), RegionalStrategySet(policies=(policy,)), _roster(uuv_count=2)
    )

    task = result.tasks["T1:cell:0:0"]
    assert task.tracking_mode == "heuristic_uuv"
    assert task.assigned_uuv_ids == ()
    assert task.assignment_status == "uncovered"


def test_materializer_preserves_explicit_llm_members_and_empty_membership() -> None:
    selected = RegionalPolicy(
        region_id="T1:cell:0:0",
        coverage_mode="required",
        priority=1.0,
        required_quality=0.7,
        required_uuv_count=4,
        uuv_roles=("passive_tracker",),
        sonar_policy=SonarPolicy(passive_required=True),
        communication=CommunicationRequirement(),
        rationale="one selected UUV covers the predicted corridor",
        evidence_ids=("prediction-1",),
        tracking_mode="heuristic_uuv",
        assigned_uuv_ids=("uuv-0",),
    )
    empty = selected.model_copy(
        update={
            "tracking_mode": "heuristic_uuv",
            "required_uuv_count": 4,
            "communication": CommunicationRequirement(),
            "assigned_uuv_ids": (),
        }
    )

    selected_result = materialize_regional_plan(
        _plan(), RegionalStrategySet(policies=(selected,)), _roster(uuv_count=2)
    )
    empty_result = materialize_regional_plan(
        _plan(), RegionalStrategySet(policies=(empty,)), _roster(uuv_count=2)
    )

    selected_task = selected_result.tasks[selected.region_id]
    assert selected_task.assigned_uuv_ids == ("uuv-0",)
    assert selected_task.required_uuv_count == 4
    assert empty_result.tasks[empty.region_id].assignment_status == "uncovered"
    assert empty_result.tasks[empty.region_id].assigned_uuv_ids == ()


def test_materializer_degrades_unknown_duplicate_and_unavailable_llm_members() -> None:
    unavailable_roster = _roster(uuv_count=2).model_copy(
        update={
            "uuvs": (
                _roster(uuv_count=2).uuvs[0],
                _roster(uuv_count=2).uuvs[1].model_copy(
                    update={"deployment_state": "returning"}
                ),
            )
        }
    )
    policy = RegionalPolicy(
        region_id="T1:cell:0:0",
        coverage_mode="required",
        priority=1.0,
        required_quality=0.7,
        required_uuv_count=0,
        sonar_policy=SonarPolicy(passive_required=True),
        communication=CommunicationRequirement(),
        rationale="these are the explicitly selected platforms",
        evidence_ids=("prediction-1",),
        tracking_mode="heuristic_uuv",
        assigned_uuv_ids=("uuv-0", "uuv-0", "uuv-1", "uuv-unknown"),
    )

    result = materialize_regional_plan(
        _plan(), RegionalStrategySet(policies=(policy,)), unavailable_roster
    )

    task = result.tasks[policy.region_id]
    assert task.assignment_status == "degraded"
    assert task.assigned_uuv_ids == ("uuv-0",)
    assert set(task.degraded_reasons) >= {
        "duplicate_uuv:uuv-0",
        "uuv_unavailable:uuv-1",
        "unknown_uuv:uuv-unknown",
    }


def test_regional_policy_requires_explicit_member_lists() -> None:
    common = {
        "region_id": "T1:cell:0:0",
        "coverage_mode": "required",
        "priority": 1.0,
        "required_quality": 0.7,
        "required_uuv_count": 0,
        "sonar_policy": SonarPolicy(passive_required=True),
        "communication": CommunicationRequirement(),
        "rationale": "selection is deliberately empty",
        "evidence_ids": ("prediction-1",),
        "tracking_mode": "heuristic_uuv",
    }

    with pytest.raises(ValueError, match="assigned_uuv_ids"):
        RegionalPolicy(**common)

    policy = RegionalPolicy(
        **common,
        assigned_uuv_ids=(),
    )

    assert policy.assigned_uuv_ids == ()
