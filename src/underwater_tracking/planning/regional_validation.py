from __future__ import annotations

from collections.abc import Iterable, Mapping

from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    PlatformRoster,
)
from underwater_tracking.domain.regional_models import RegionTask, TargetRegionPlan


def validate_regional_plan(
    plan: TargetRegionPlan,
    roster: PlatformRoster,
    *,
    carrier: CarrierPlatformState | None = None,
    map_bounds_xy: tuple[float, float, float, float] | None = None,
    reserved_uuv_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return deterministic hard-constraint issues for a regional plan.

    The regional contracts catch malformed geometry and time windows during
    construction. This validator covers UUV lifecycle/capability,
    reservations, and overlapping assignments.
    """
    cells = {cell.region_id: cell for cell in plan.cells}
    uuvs = {platform.platform_id: platform for platform in roster.uuvs}
    reserved = frozenset(reserved_uuv_ids)
    issues: set[str] = set()
    assignment_windows: dict[str, list[tuple[int, int, str]]] = {}

    for task in plan.tasks:
        cell = cells.get(task.region_id)
        if cell is None:
            issues.add(f"unknown_region:{task.region_id}")
            continue
        if not cell.first_entry_s <= task.active_window.start_s:
            issues.add(f"task_before_region_window:{task.region_id}")
        if task.active_window.end_s > cell.last_exit_s:
            issues.add(f"task_after_region_window:{task.region_id}")
        if map_bounds_xy is not None and not _cell_inside_bounds(cell, map_bounds_xy):
            issues.add(f"region_outside_map:{task.region_id}")
        if not task.sonar_policy.passive_required:
            issues.add(f"passive_sonar_required:{task.region_id}")
        if len(set(task.assigned_uuv_ids)) != len(task.assigned_uuv_ids):
            issues.add(f"duplicate_uuv_assignment:{task.region_id}")
        _validate_tracking_mode(task, issues)

        for uuv_id in task.assigned_uuv_ids:
            platform = uuvs.get(uuv_id)
            if platform is None:
                issues.add(f"unknown_uuv:{uuv_id}")
                continue
            if uuv_id in reserved:
                issues.add(f"reserved_uuv_assigned:{uuv_id}")
            if platform.deployment_state in {"returning", "failed", "onboard"}:
                issues.add(f"uuv_unavailable:{uuv_id}")
            if task.sonar_policy.active_allowed and not platform.capability.sonar.active_capable:
                issues.add("active_sonar_not_supported")
            _record_window(assignment_windows, uuv_id, task)

    _validate_overlapping_assignments(plan, assignment_windows, issues)
    return tuple(sorted(issues))


def _validate_tracking_mode(task: RegionTask, issues: set[str]) -> None:
    if task.assignment_status != "uncovered" and not task.assigned_uuv_ids:
        issues.add(f"missing_uuv_tracking_owner:{task.region_id}")


def _record_window(
    windows: dict[str, list[tuple[int, int, str]]],
    platform_id: str,
    task: RegionTask,
) -> None:
    windows.setdefault(platform_id, []).append(
        (task.active_window.start_s, task.active_window.end_s, task.region_id)
    )


def _validate_overlapping_assignments(
    plan: TargetRegionPlan,
    windows: Mapping[str, list[tuple[int, int, str]]],
    issues: set[str],
) -> None:
    for platform_id, entries in windows.items():
        ordered = sorted(entries)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if not _windows_overlap(left, right):
                    continue
                issues.add("uuv_double_booked")


def _cell_inside_bounds(
    cell: object,
    bounds: tuple[float, float, float, float],
) -> bool:
    min_x, max_x, min_y, max_y = bounds
    return (
        cell.min_x >= min_x
        and cell.max_x <= max_x
        and cell.min_y >= min_y
        and cell.max_y <= max_y
    )


def _windows_overlap(left: tuple[int, int, str], right: tuple[int, int, str]) -> bool:
    return left[0] < right[1] and right[0] < left[1]
