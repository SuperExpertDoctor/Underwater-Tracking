from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import hypot

from underwater_tracking.domain.agent_models import Waypoint
from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    PlatformRoster,
    UUVPlatformState,
    USVPlatformState,
)
from underwater_tracking.domain.regional_models import RegionCell, RegionTask, TargetRegionPlan
from underwater_tracking.planning.regional_validation import validate_regional_plan


@dataclass(frozen=True, slots=True)
class RegionalAllocationResult:
    tasks: Mapping[str, RegionTask]
    waypoints_by_member: Mapping[str, tuple[Waypoint, ...]]
    issues: tuple[str, ...] = ()


def allocate_regional_tasks(
    plan: TargetRegionPlan,
    roster: PlatformRoster,
    *,
    carrier: CarrierPlatformState | None = None,
    reserved_uuv_ids: Iterable[str] = (),
) -> RegionalAllocationResult:
    """Assign regional roles with stable scores and explicit degradation."""
    cells = {cell.region_id: cell for cell in plan.cells}
    uuvs = {platform.platform_id: platform for platform in roster.uuvs}
    usvs = {platform.platform_id: platform for platform in roster.usvs}
    reserved = frozenset(reserved_uuv_ids)
    used_uuvs: set[str] = set()
    used_usvs: set[str] = set()
    result_tasks: dict[str, RegionTask] = {}
    waypoints: dict[str, tuple[Waypoint, ...]] = {}
    allocation_issues: list[str] = []

    ordered_tasks = sorted(
        plan.tasks,
        key=lambda task: (
            task.active_window.start_s,
            task.region_id,
        ),
    )
    for task in ordered_tasks:
        cell = cells[task.region_id]
        assigned_uuvs = _reuse_valid_uuvs(task, uuvs, reserved, used_uuvs)
        missing_uuv_count = max(0, task.required_uuv_count - len(assigned_uuvs))
        if missing_uuv_count:
            candidates = _uuv_candidates(
                task,
                cell,
                uuvs.values(),
                reserved=reserved,
                used=used_uuvs,
            )
            assigned_uuvs.extend(candidates[:missing_uuv_count])
        used_uuvs.update(assigned_uuvs)

        assigned_usvs = _reuse_valid_usvs(task, usvs, used_usvs)
        missing_usv_count = max(0, task.required_usv_count - len(assigned_usvs))
        if missing_usv_count:
            candidates = _usv_candidates(
                cell,
                usvs.values(),
                carrier=carrier,
                used=used_usvs,
                allow_adjacent_overlap=plan.grid_spec.relay_overlap_policy == "adjacent_connected",
            )
            assigned_usvs.extend(candidates[:missing_usv_count])
        used_usvs.update(assigned_usvs)

        reasons = []
        if len(assigned_uuvs) < task.required_uuv_count:
            reasons.append("insufficient_uuv")
        if len(assigned_usvs) < task.required_usv_count:
            reasons.append("insufficient_usv")
        if any(uuv_id in reserved for uuv_id in assigned_uuvs):
            reasons.append("reserved_uuv_unavailable")
        if task.sonar_policy.active_allowed:
            active_ids = {
                uuv_id
                for uuv_id in assigned_uuvs
                if uuvs[uuv_id].capability.sonar.active_capable
            }
            if len(active_ids) < sum(role == "active_verifier" for role in task.uuv_roles):
                reasons.append("active_sonar_not_supported")

        status = "active" if not reasons else "degraded"
        links = _communication_links(assigned_uuvs, assigned_usvs)
        updated = task.model_copy(
            update={
                "assigned_uuv_ids": tuple(sorted(assigned_uuvs)),
                "assigned_usv_ids": tuple(sorted(assigned_usvs)),
                "assignment_status": status,
                "communication_links": links,
                "degraded_reasons": tuple(sorted(set(reasons))),
                "current_sonar_mode": "passive",
            }
        )
        result_tasks[task.region_id] = updated
        for platform_id in (*assigned_uuvs, *assigned_usvs):
            waypoints[platform_id] = (
                Waypoint(
                    x=cell.center_xy[0],
                    y=cell.center_xy[1],
                    arrive_at_s=task.active_window.start_s,
                ),
            )
        allocation_issues.extend(
            f"{task.region_id}:{reason}" for reason in reasons
        )

    allocated_plan = plan.model_copy(update={"tasks": tuple(result_tasks.values())})
    validation_issues = validate_regional_plan(
        allocated_plan,
        roster,
        carrier=carrier,
        reserved_uuv_ids=reserved,
    )
    return RegionalAllocationResult(
        tasks=result_tasks,
        waypoints_by_member=waypoints,
        issues=tuple(sorted(set((*allocation_issues, *validation_issues)))),
    )


def _reuse_valid_uuvs(
    task: RegionTask,
    uuvs: Mapping[str, UUVPlatformState],
    reserved: frozenset[str],
    used: set[str],
) -> list[str]:
    return [
        uuv_id
        for uuv_id in task.assigned_uuv_ids
        if uuv_id in uuvs
        and uuv_id not in reserved
        and uuv_id not in used
        and uuvs[uuv_id].deployment_state == "deployed"
    ]


def _uuv_candidates(
    task: RegionTask,
    cell: RegionCell,
    platforms: Iterable[UUVPlatformState],
    *,
    reserved: frozenset[str],
    used: set[str],
) -> list[str]:
    required_active = sum(role == "active_verifier" for role in task.uuv_roles)
    candidates = []
    for platform in platforms:
        if platform.platform_id in reserved or platform.platform_id in used:
            continue
        if platform.deployment_state != "deployed":
            continue
        if not task.sonar_policy.passive_required:
            continue
        if (
            task.sonar_policy.active_allowed
            and required_active
            and not platform.capability.sonar.active_capable
        ):
            continue
        distance = _distance(platform.position_xy, cell.center_xy)
        if distance > platform.capability.sonar.passive_range_m + cell.cell_size_m:
            continue
        travel = distance / max(platform.capability.motion.max_speed_mps, 1e-6)
        candidates.append(
            (
                -1.0,
                travel,
                1.0 - platform.energy_fraction,
                0.0,
                0.0,
                platform.platform_id,
            )
        )
    candidates.sort()
    return [candidate[-1] for candidate in candidates]


def _reuse_valid_usvs(
    task: RegionTask,
    usvs: Mapping[str, USVPlatformState],
    used: set[str],
) -> list[str]:
    return [
        usv_id
        for usv_id in task.assigned_usv_ids
        if usv_id in usvs
        and usv_id not in used
        and usvs[usv_id].deployment_state == "deployed"
    ]


def _usv_candidates(
    cell: RegionCell,
    platforms: Iterable[USVPlatformState],
    *,
    carrier: CarrierPlatformState | None,
    used: set[str],
    allow_adjacent_overlap: bool,
) -> list[str]:
    candidates = []
    for platform in platforms:
        if platform.platform_id in used or platform.deployment_state != "deployed":
            continue
        if (
            carrier is not None
            and _distance(platform.position_xy, carrier.position_xy)
            > carrier.support_radius_m
        ):
            continue
        distance = _distance(platform.position_xy, cell.center_xy)
        if distance > platform.capability.sonar.passive_range_m + cell.cell_size_m:
            continue
        if (
            not allow_adjacent_overlap
            and platform.distance_to_carrier_m
            > platform.capability.communications.surface_range_m
        ):
            continue
        travel = distance / max(platform.capability.motion.max_speed_mps, 1e-6)
        relay_instability = (
            0.0
            if platform.distance_to_carrier_m
            <= platform.capability.communications.surface_range_m
            else 1.0
        )
        candidates.append(
            (
                -1.0,
                travel,
                1.0 - platform.energy_fraction,
                0.0,
                relay_instability,
                platform.platform_id,
            )
        )
    candidates.sort()
    return [candidate[-1] for candidate in candidates]


def _communication_links(
    uuv_ids: Iterable[str],
    usv_ids: Iterable[str],
) -> tuple[str, ...]:
    links = []
    for usv_id in sorted(usv_ids):
        links.append(f"carrier->{usv_id}")
        links.extend(f"{usv_id}->{uuv_id}" for uuv_id in sorted(uuv_ids))
    return tuple(links)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])
