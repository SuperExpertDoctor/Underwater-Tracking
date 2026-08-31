from __future__ import annotations

from types import SimpleNamespace

import pytest

from underwater_tracking.planning.dynamic_regions import build_dynamic_region_chain
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.planning.task_groups import (
    ReplacementQueue,
    TaskGroupAllocator,
    allocate_four_task_groups,
)
from underwater_tracking.planning.mission_optimizer import MissionOptimizer


def _chain():
    points = tuple((100.0 + time, 300.0) for time in range(0, 1801, 100))
    prediction = SimpleNamespace(
        prediction_id="pred:T1:1",
        target_id="T1",
        origin_sim_time_s=0.0,
        times_s=tuple(float(time) for time in range(0, 1801, 100)),
        centerline_xy=points,
        covariance_xy=tuple((10.0, 0.0, 0.0, 10.0) for _ in points),
        corridor_radius_m=tuple(5.0 for _ in points),
    )
    return build_dynamic_region_chain(
        prediction,
        execution_revision=4,
        map_bounds_xy=(-500.0, 3_000.0, -500.0, 1_000.0),
    )


def test_allocator_assigns_four_two_uuv_groups_and_four_reserves() -> None:
    allocation = allocate_four_task_groups(
        _chain(),
        tuple(f"UUV-{index:02d}" for index in range(1, 13)),
        execution_revision=4,
    )

    assert len(allocation.assignments) == 4
    assert [assignment.region_id for assignment in allocation.assignments] == [
        "T1:task:01",
        "T1:task:02",
        "T1:task:03",
        "T1:task:04",
    ]
    assert all(len(assignment.member_uuv_ids) == 2 for assignment in allocation.assignments)
    assert all(
        assignment.active_verifier_uuv_id != assignment.passive_tracker_uuv_id
        for assignment in allocation.assignments
    )
    assert len(allocation.reserve_uuvs) == 4
    assert len(set(allocation.assigned_uuv_ids) | set(allocation.reserve_uuv_ids)) == 12
    assert all(region.task_group_id for region in allocation.bound_regions)
    assert allocation.degraded is False


def test_allocator_preserves_current_group_members_when_regions_roll() -> None:
    allocator = TaskGroupAllocator()
    first = allocator.allocate(_chain(), [f"UUV-{index:02d}" for index in range(1, 13)])
    rolled = allocator.allocate(
        _chain(),
        [f"UUV-{index:02d}" for index in range(1, 13)],
        previous_assignments=first.assignments,
    )

    assert rolled.assigned_uuv_ids == first.assigned_uuv_ids
    assert tuple(
        assignment.member_uuv_ids for assignment in rolled.assignments
    ) == tuple(assignment.member_uuv_ids for assignment in first.assignments)


def test_insufficient_resources_are_degraded_without_inventing_members() -> None:
    allocation = allocate_four_task_groups(
        _chain(),
        [f"UUV-{index:02d}" for index in range(1, 11)],
        execution_revision=4,
    )

    assert len(allocation.assignments) == 4
    assert len(allocation.assigned_uuv_ids) == 8
    assert allocation.reserve_uuv_ids == ()
    assert allocation.degraded is True
    assert allocation.degradation_reasons
    assert all(
        member not in {"UUV-11", "UUV-12"}
        for assignment in allocation.assignments
        for member in assignment.member_uuv_ids
    )


def test_allocator_rejects_duplicate_resource_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        allocate_four_task_groups(_chain(), ("UUV-01", "UUV-01"), execution_revision=4)


def test_replacement_queue_uses_deterministic_reserve_priority_and_episode() -> None:
    allocation = allocate_four_task_groups(
        _chain(),
        [f"UUV-{index:02d}" for index in range(1, 13)],
        execution_revision=4,
    )
    queue = ReplacementQueue(allocation.reserve_uuvs)

    replacement = queue.acquire(
        region_id="T1:task:02",
        failed_uuv_id=allocation.assignments[1].active_verifier_uuv_id,
        execution_revision=5,
    )

    assert replacement is not None
    assert replacement.status == "entering"
    assert replacement.resource_episode == 1
    assert queue.acquire(
        region_id="T1:task:02",
        failed_uuv_id="missing",
        execution_revision=5,
    ) is not None
    assert queue.acquire(
        region_id="T1:task:02",
        failed_uuv_id="missing",
        execution_revision=5,
    ) is not None
    assert queue.acquire(
        region_id="T1:task:02",
        failed_uuv_id="missing",
        execution_revision=5,
    ) is not None
    assert queue.acquire(
        region_id="T1:task:02",
        failed_uuv_id="missing",
        execution_revision=5,
    ) is None


def test_mission_optimizer_exposes_the_regional_allocator_without_old_candidates() -> None:
    allocation = MissionOptimizer().allocate_four_task_groups(
        _chain(),
        [f"UUV-{index:02d}" for index in range(1, 13)],
        execution_revision=4,
    )

    assert len(allocation.assignments) == 4
    assert len(allocation.reserve_uuvs) == 4


def test_executable_mission_plan_can_carry_complete_task_group_projection() -> None:
    allocation = allocate_four_task_groups(
        _chain(),
        [f"UUV-{index:02d}" for index in range(1, 13)],
        execution_revision=4,
    )

    plan = ExecutableMissionPlan(
        revision=4,
        task_groups=allocation.assignments,
        reserve_uuvs=allocation.reserve_uuvs,
    )

    assert len(plan.task_groups) == 4
    assert tuple(reserve.uuv_id for reserve in plan.reserve_uuvs) == (
        "UUV-09",
        "UUV-10",
        "UUV-11",
        "UUV-12",
    )
