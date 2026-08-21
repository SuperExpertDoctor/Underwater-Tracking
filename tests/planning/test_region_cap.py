from underwater_tracking.domain.mission_models import MissionCandidate
from underwater_tracking.domain.regional_models import (
    GridSpec,
    RegionCell,
    RegionTask,
    TargetRegionPlan,
    TimeWindow,
)
from underwater_tracking.planning.region_cap import (
    cap_candidate_regions,
    cap_target_region_plan,
)


def _candidate(index: int, *, priority: float, probability: float) -> MissionCandidate:
    return MissionCandidate(
        candidate_id=f"T1:r{index}",
        target_id="T1",
        entry_s=index * 100,
        exit_s=index * 100 + 90,
        probability=probability,
        priority=priority,
        perimeter_points=((0.0, 0.0), (0.0, 100.0), (100.0, 0.0), (100.0, 100.0)),
    )


def test_candidate_cap_keeps_four_highest_value_regions() -> None:
    candidates = tuple(
        _candidate(index, priority=1.0 if index < 2 else 0.0, probability=index / 10)
        for index in range(6)
    )

    selected, excluded = cap_candidate_regions(candidates)

    assert tuple(candidate.candidate_id for candidate in selected) == (
        "T1:r0",
        "T1:r1",
        "T1:r5",
        "T1:r4",
    )
    assert tuple(candidate.candidate_id for candidate in excluded) == (
        "T1:r3",
        "T1:r2",
    )


def test_region_plan_cap_retains_audit_tasks_but_returns_only_executable_tasks() -> None:
    cells = tuple(
        RegionCell(
            region_id=f"T1:cell:{index}:0",
            target_id="T1",
            grid_x=index,
            grid_y=0,
            min_x=float(index * 100),
            max_x=float(index * 100 + 100),
            min_y=0.0,
            max_y=100.0,
            center_xy=(float(index * 100 + 50), 50.0),
            cell_size_m=100.0,
            first_entry_s=index * 100,
            last_exit_s=index * 100 + 90,
        )
        for index in range(5)
    )
    plan = TargetRegionPlan(
        target_id="T1",
        grid_spec=GridSpec(),
        cell_size_m=100.0,
        cells=cells,
        tasks=tuple(
            RegionTask(
                region_id=cell.region_id,
                target_id="T1",
                active_window=TimeWindow(
                    start_s=cell.first_entry_s,
                    end_s=cell.last_exit_s,
                ),
                priority=1.0 if cell.grid_x == 0 else 0.0,
            )
            for cell in cells
        ),
        prediction_id="prediction:T1",
        intent_label="patrol",
        intent_confidence=0.8,
    )

    capped, executable_tasks = cap_target_region_plan(plan)

    assert len(executable_tasks) == 4
    assert len(capped.tasks) == 5
    excluded = [
        task
        for task in capped.tasks
        if "region_cap_not_selected" in task.degraded_reasons
    ]
    assert len(excluded) == 1
    assert excluded[0].assignment_status == "uncovered"
    assert excluded[0].assigned_uuv_ids == ()
