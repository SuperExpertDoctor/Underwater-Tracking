from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import hypot

from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    PlatformRoster,
    UUVPlatformState,
    USVPlatformState,
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
    construction. This validator covers the runtime-dependent constraints:
    platform lifecycle and capability, carrier support radius, relay paths,
    reservations, and overlapping assignments.
    """
    cells = {cell.region_id: cell for cell in plan.cells}
    uuvs = {platform.platform_id: platform for platform in roster.uuvs}
    usvs = {platform.platform_id: platform for platform in roster.usvs}
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
        if len(set(task.assigned_usv_ids)) != len(task.assigned_usv_ids):
            issues.add(f"duplicate_usv_assignment:{task.region_id}")

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

        for usv_id in task.assigned_usv_ids:
            platform = usvs.get(usv_id)
            if platform is None:
                issues.add(f"unknown_usv:{usv_id}")
                continue
            if platform.deployment_state in {"returning", "failed", "onboard"}:
                issues.add(f"usv_unavailable:{usv_id}")
            if carrier is not None:
                distance = _distance(platform.position_xy, carrier.position_xy)
                if distance > carrier.support_radius_m:
                    issues.add("usv_outside_carrier_radius")
            _record_window(assignment_windows, usv_id, task)

        if task.communication.usv_relay_required:
            if not task.assigned_usv_ids:
                issues.add(f"missing_usv_relay:{task.region_id}")
            else:
                _validate_relay_paths(task, uuvs, usvs, issues)

    _validate_overlapping_assignments(plan, assignment_windows, issues)
    return tuple(sorted(issues))


def _validate_tracking_mode(task: RegionTask, issues: set[str]) -> None:
    if task.tracking_mode == "heuristic_uuv":
        if task.assigned_usv_ids:
            issues.add(f"mixed_tracking_domains:{task.region_id}")
        if not task.assigned_uuv_ids:
            issues.add(f"missing_uuv_tracking_owner:{task.region_id}")
    elif task.tracking_mode == "heuristic_usv":
        if task.assigned_uuv_ids:
            issues.add(f"mixed_tracking_domains:{task.region_id}")
        if not task.assigned_usv_ids:
            issues.add(f"missing_usv_tracking_owner:{task.region_id}")
    else:
        if not task.assigned_uuv_ids:
            issues.add(f"missing_uuv_tracking_owner:{task.region_id}")
        if not task.assigned_usv_ids:
            issues.add(f"missing_usv_relay:{task.region_id}")
        if task.assigned_usv_ids and task.usv_role not in {
            "surface_relay",
            "handoff_reserve",
        }:
            issues.add(f"usv_not_relay:{task.region_id}")


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
    cells = {cell.region_id: cell for cell in plan.cells}
    task_by_region = {task.region_id: task for task in plan.tasks}
    for platform_id, entries in windows.items():
        ordered = sorted(entries)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if not _windows_overlap(left, right):
                    continue
                left_task = task_by_region[left[2]]
                right_task = task_by_region[right[2]]
                if (
                    platform_id in left_task.assigned_usv_ids
                    and _relay_overlap_allowed(plan, cells, left[2], right[2])
                ):
                    continue
                prefix = "usv" if platform_id in _assigned_usvs(plan) else "uuv"
                issues.add(f"{prefix}_double_booked")


def _assigned_usvs(plan: TargetRegionPlan) -> set[str]:
    return {platform_id for task in plan.tasks for platform_id in task.assigned_usv_ids}


def _relay_overlap_allowed(
    plan: TargetRegionPlan,
    cells: Mapping[str, object],
    left_region: str,
    right_region: str,
) -> bool:
    if plan.grid_spec.relay_overlap_policy != "adjacent_connected":
        return False
    left = cells.get(left_region)
    right = cells.get(right_region)
    if left is None or right is None:
        return False
    return abs(left.grid_x - right.grid_x) + abs(left.grid_y - right.grid_y) == 1


def _validate_relay_paths(
    task: RegionTask,
    uuvs: Mapping[str, UUVPlatformState],
    usvs: Mapping[str, USVPlatformState],
    issues: set[str],
) -> None:
    relay = usvs[task.assigned_usv_ids[0]]
    for uuv_id in task.assigned_uuv_ids:
        uuv = uuvs.get(uuv_id)
        if uuv is None:
            continue
        if (
            _distance(uuv.position_xy, relay.position_xy)
            > relay.capability.communications.acoustic_range_m
        ):
            issues.add(f"communication_path_missing:{task.region_id}")


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


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])
