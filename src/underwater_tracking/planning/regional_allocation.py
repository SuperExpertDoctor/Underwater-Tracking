from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import hypot

from underwater_tracking.domain.agent_models import Waypoint
from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    PlatformRoster,
    UUVPlatformState,
)
from underwater_tracking.domain.regional_models import (
    RegionCell,
    RegionTask,
    RegionalStrategySet,
    TargetRegionPlan,
)
from underwater_tracking.planning.regional_validation import validate_regional_plan

_MIN_REGIONAL_STANDOFF_M = 250.0
_STANDOFF_SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True, slots=True)
class RegionalAllocationResult:
    tasks: Mapping[str, RegionTask]
    waypoints_by_member: Mapping[str, tuple[Waypoint, ...]]
    issues: tuple[str, ...] = ()


def materialize_regional_plan(
    plan: TargetRegionPlan,
    strategy: RegionalStrategySet,
    roster: PlatformRoster,
    *,
    carrier: CarrierPlatformState | None = None,
    reserved_uuv_ids: Iterable[str] = (),
) -> RegionalAllocationResult:
    """Turn LLM regional policies into validated, single-domain task groups.

    Regional policy member IDs are authoritative, including an empty tuple.
    Required counts describe the policy's intended coverage only; they never
    trigger automatic platform selection. Runtime availability and safety
    checks may degrade an explicit selection, but must not replace it.
    """
    policies = {policy.region_id: policy for policy in strategy.policies}
    tasks: list[RegionTask] = []
    reserved = frozenset(reserved_uuv_ids)

    for base_task in sorted(plan.tasks, key=lambda item: (item.active_window.start_s, item.region_id)):
        policy = policies.get(base_task.region_id)
        if policy is None:
            raise ValueError(f"regional strategy omitted {base_task.region_id}")
        uuv_ids = tuple(policy.assigned_uuv_ids)
        uuv_roles = tuple(policy.uuv_roles)
        if uuv_ids and len(uuv_roles) < len(uuv_ids):
            uuv_roles = (*uuv_roles, *("passive_tracker",) * (len(uuv_ids) - len(uuv_roles)))
        task = base_task.model_copy(
            update={
                "tracking_mode": policy.tracking_mode,
                "priority": policy.priority,
                "required_quality": policy.required_quality,
                "coverage_mode": policy.coverage_mode,
                "required_uuv_count": policy.required_uuv_count,
                "uuv_roles": uuv_roles,
                "sonar_policy": policy.sonar_policy,
                "communication": policy.communication,
                "predecessor_region_id": policy.predecessor_region_id,
                "successor_region_id": policy.successor_region_id,
                "assigned_uuv_ids": uuv_ids,
                "evidence_ids": tuple(sorted(policy.evidence_ids)),
            }
        )
        tasks.append(task)

    materialized = plan.model_copy(update={"tasks": tuple(tasks)})
    return allocate_regional_tasks(
        materialized,
        roster,
        carrier=carrier,
        reserved_uuv_ids=reserved,
    )
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
    reserved = frozenset(reserved_uuv_ids)
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
        reasons = []
        requested_uuvs = list(task.assigned_uuv_ids)
        valid_uuvs: list[str] = []
        seen_uuvs: set[str] = set()
        for uuv_id in requested_uuvs:
            if uuv_id in seen_uuvs:
                reasons.append(f"duplicate_uuv:{uuv_id}")
                continue
            seen_uuvs.add(uuv_id)
            if uuv_id not in uuvs:
                reasons.append(f"unknown_uuv:{uuv_id}")
            elif uuv_id in reserved:
                reasons.append("reserved_uuv_unavailable")
            elif uuvs[uuv_id].deployment_state != "deployed":
                reasons.append(f"uuv_unavailable:{uuv_id}")
            else:
                valid_uuvs.append(uuv_id)
        if not requested_uuvs:
            reasons.append("missing_uuv_tracking_owner")
        if task.sonar_policy.active_allowed:
            active_ids = {
                uuv_id
                for uuv_id in valid_uuvs
                if uuvs[uuv_id].capability.sonar.active_capable
            }
            if len(active_ids) < sum(role == "active_verifier" for role in task.uuv_roles):
                reasons.append("active_sonar_not_supported")

        standoff_points: dict[str, tuple[float, float]] = {}
        if valid_uuvs:
            target_xy = cell.predicted_target_xy or cell.center_xy
            for index, platform_id in enumerate(valid_uuvs):
                point = _regional_standoff_point(cell, target_xy, index)
                if point is None:
                    point = cell.center_xy
                    if "standoff_infeasible:250m" not in reasons:
                        reasons.append("standoff_infeasible:250m")
                standoff_points[platform_id] = point

        status = (
            "uncovered"
            if not requested_uuvs
            else "active"
            if not reasons
            else "degraded"
        )
        links = _communication_links(valid_uuvs)
        updated = task.model_copy(
            update={
                "assigned_uuv_ids": tuple(valid_uuvs),
                "assignment_status": status,
                "communication_links": links,
                "degraded_reasons": tuple(sorted(set(reasons))),
                "current_sonar_mode": "passive",
            }
        )
        result_tasks[task.region_id] = updated
        for platform_id in valid_uuvs:
            waypoints[platform_id] = (
                Waypoint(
                    x=standoff_points[platform_id][0],
                    y=standoff_points[platform_id][1],
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


def _regional_standoff_point(
    cell: RegionCell,
    target_xy: tuple[float, float],
    member_index: int,
) -> tuple[float, float] | None:
    """Pick the nearest deterministic in-cell point outside the blind zone."""
    candidates = [
        (cell.min_x + (cell.max_x - cell.min_x) * fraction_x,
         cell.min_y + (cell.max_y - cell.min_y) * fraction_y)
        for fraction_x in _STANDOFF_SAMPLE_FRACTIONS
        for fraction_y in _STANDOFF_SAMPLE_FRACTIONS
    ]
    feasible = [
        point
        for point in candidates
        if _distance(point, target_xy) >= _MIN_REGIONAL_STANDOFF_M
    ]
    if not feasible:
        return None
    feasible.sort(key=lambda point: (_distance(point, target_xy), point[0], point[1]))
    return feasible[member_index % len(feasible)]


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


def _communication_links(
    uuv_ids: Iterable[str],
) -> tuple[str, ...]:
    return tuple(f"carrier->{uuv_id}" for uuv_id in sorted(uuv_ids))


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])
