"""Bound the executable regional mission while preserving planning evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from math import hypot
from typing import TypeVar

from underwater_tracking.domain.mission_models import MissionCandidate
from underwater_tracking.domain.regional_models import RegionCell, RegionTask, TargetRegionPlan


MAX_EXECUTABLE_REGIONS_PER_TARGET = 4

_CandidateT = TypeVar("_CandidateT", bound=MissionCandidate)


def cap_candidate_regions(
    candidates: Sequence[_CandidateT],
    *,
    max_regions: int = MAX_EXECUTABLE_REGIONS_PER_TARGET,
    protected_ids: Iterable[str] = (),
) -> tuple[tuple[_CandidateT, ...], tuple[_CandidateT, ...]]:
    """Select a stable, bounded executable subset for every target.

    ``protected_ids`` is used for an already active region or an explicitly
    locked successor. Protected candidates consume the same bounded slots and
    therefore cannot silently make the cap ineffective.
    """
    _validate_max_regions(max_regions)
    protected = frozenset(protected_ids)
    by_target: dict[str, list[_CandidateT]] = {}
    for candidate in candidates:
        by_target.setdefault(candidate.target_id, []).append(candidate)

    selected: list[_CandidateT] = []
    excluded: list[_CandidateT] = []
    for target_id in sorted(by_target):
        ranked = sorted(by_target[target_id], key=_candidate_rank)
        protected_candidates = [
            candidate for candidate in ranked if candidate.candidate_id in protected
        ]
        if len(protected_candidates) > max_regions:
            raise ValueError(
                f"protected regions for target {target_id!r} exceed the executable cap"
            )
        selected_ids = {
            candidate.candidate_id for candidate in protected_candidates
        }
        for candidate in ranked:
            if len(selected_ids) >= max_regions:
                break
            selected_ids.add(candidate.candidate_id)
        selected.extend(
            candidate for candidate in ranked if candidate.candidate_id in selected_ids
        )
        excluded.extend(
            candidate for candidate in ranked if candidate.candidate_id not in selected_ids
        )
    return tuple(selected), tuple(excluded)


def cap_target_region_plan(
    plan: TargetRegionPlan,
    *,
    max_regions: int = MAX_EXECUTABLE_REGIONS_PER_TARGET,
    protected_region_ids: Iterable[str] = (),
) -> tuple[TargetRegionPlan, Mapping[str, RegionTask]]:
    """Keep all region tasks for audit, returning only executable tasks.

    The target plan schema requires one task per generated cell. Unselected
    tasks remain in that full plan with cleared assignments and an explicit
    degradation reason, while the returned mapping is the only execution view.
    """
    _validate_max_regions(max_regions)
    protected = frozenset(protected_region_ids)
    cells_by_id = {cell.region_id: cell for cell in plan.cells}
    ranked = sorted(plan.tasks, key=_task_rank)
    protected_tasks = [task for task in ranked if task.region_id in protected]
    if len(protected_tasks) > max_regions:
        raise ValueError(
            f"protected regions for target {plan.target_id!r} exceed the executable cap"
        )
    selected_ids = {task.region_id for task in protected_tasks}
    by_prediction_point: dict[tuple[float, float], list[RegionTask]] = defaultdict(list)
    fallback_tasks: list[RegionTask] = []
    for task in ranked:
        if task.region_id in selected_ids:
            continue
        cell = cells_by_id.get(task.region_id)
        if cell is None or cell.predicted_target_xy is None:
            fallback_tasks.append(task)
            continue
        by_prediction_point[cell.predicted_target_xy].append(task)

    trajectory_tasks = sorted(
        (
            min(
                tasks,
                key=lambda task: (
                    _prediction_distance_to_cell(cells_by_id[task.region_id]),
                    _task_rank(task),
                ),
            )
            for tasks in by_prediction_point.values()
        ),
        key=_task_rank,
    )
    for task in trajectory_tasks:
        if len(selected_ids) >= max_regions:
            break
        selected_ids.add(task.region_id)
    for task in fallback_tasks:
        if len(selected_ids) >= max_regions:
            break
        selected_ids.add(task.region_id)

    selected_tasks = sorted(
        (task for task in plan.tasks if task.region_id in selected_ids),
        key=lambda task: (
            task.active_window.start_s,
            task.active_window.end_s,
            task.region_id,
        ),
    )
    relinked: dict[str, RegionTask] = {}
    for index, task in enumerate(selected_tasks):
        predecessor = (
            selected_tasks[index - 1]
            if index > 0
            and selected_tasks[index - 1].active_window.start_s < task.active_window.start_s
            else None
        )
        successor = (
            selected_tasks[index + 1]
            if index + 1 < len(selected_tasks)
            and task.active_window.start_s < selected_tasks[index + 1].active_window.start_s
            else None
        )
        relinked[task.region_id] = task.model_copy(
            update={
                "predecessor_region_id": (
                    predecessor.region_id if predecessor is not None else None
                ),
                "successor_region_id": successor.region_id if successor is not None else None,
            }
        )

    updated_tasks: list[RegionTask] = []
    executable: dict[str, RegionTask] = {}
    for task in plan.tasks:
        if task.region_id in selected_ids:
            selected_task = relinked[task.region_id]
            updated_tasks.append(selected_task)
            executable[task.region_id] = selected_task
            continue
        updated = task.model_copy(
            update={
                "required_uuv_count": 0,
                "uuv_roles": (),
                "assigned_uuv_ids": (),
                "assignment_status": "uncovered",
                "communication_links": (),
                "current_sonar_mode": "passive",
                "degraded_reasons": tuple(
                    sorted({*task.degraded_reasons, "region_cap_not_selected"})
                ),
            }
        )
        updated_tasks.append(updated)
    return plan.model_copy(update={"tasks": tuple(updated_tasks)}), executable


def _prediction_distance_to_cell(cell: RegionCell) -> float:
    """Return the distance from a public predicted point to a region cell."""
    predicted = cell.predicted_target_xy
    if predicted is None:
        return float("inf")
    x, y = predicted
    delta_x = max(float(cell.min_x) - x, 0.0, x - float(cell.max_x))
    delta_y = max(float(cell.min_y) - y, 0.0, y - float(cell.max_y))
    return hypot(delta_x, delta_y)


def _candidate_rank(candidate: MissionCandidate) -> tuple[float, int, float, int, str]:
    # Preserve chronological continuity among explicitly prioritized regions;
    # use probability to rank the remaining candidates.
    priority_tie_break = candidate.entry_s if candidate.priority > 0.0 else 0
    return (
        -candidate.priority,
        priority_tie_break,
        -candidate.probability,
        candidate.exit_s,
        candidate.candidate_id,
    )


def _task_rank(task: RegionTask) -> tuple[float, float, int, int, str]:
    return (
        -task.priority,
        -task.required_quality,
        task.active_window.start_s,
        task.active_window.end_s,
        task.region_id,
    )


def _validate_max_regions(max_regions: int) -> None:
    if not isinstance(max_regions, int) or isinstance(max_regions, bool) or max_regions < 1:
        raise ValueError("max_regions must be a positive integer")
