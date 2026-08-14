"""Independent hard-constraint re-check for elastic group allocations.

``validate_allocation`` re-derives every hard constraint from the
``AllocationInput`` -- the per-target elastic size bounds (growth,
release hysteresis), the one-target-per-UUV rule, UUV availability, and
pair feasibility -- and compares them against a candidate
``AllocationSolution``. It is deliberately solver-agnostic: it shares
the group-size policy with the allocator (the policy defines the hard
constraint rather than being an implementation detail of either module),
but re-derives membership violations from scratch. Violations come back
as a tuple of stable, deterministic messages ordered by target id then
uuv id; an empty tuple means every hard constraint holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from underwater_tracking.planning.allocation import AllocationInput, AllocationSolution


def validate_allocation(
    problem: AllocationInput,
    solution: AllocationSolution,
) -> tuple[str, ...]:
    """Return every hard constraint violated by ``solution``, or ``()``.

    Re-checks, independently of the solver that produced the solution:
    each target's group size against its elastic bounds; at most one
    target per UUV; no assignments for unavailable (returning, failed)
    UUVs; and no assignments on infeasible range/energy pairs.
    """
    # Imported lazily so that this leaf module never triggers a circular
    # import: ``allocation`` imports ``validate_allocation`` at module
    # load, and the policy it needs lives there.
    from underwater_tracking.planning.allocation import _target_bounds

    violations: list[str] = []
    uuvs = sorted(problem.uuv_ids)
    targets = sorted(problem.target_ids)
    members = solution.members_by_target

    for target in targets:
        min_size, max_size = _target_bounds(problem, target)
        group = tuple(members.get(target, ()))
        if not min_size <= len(group) <= max_size:
            violations.append(
                f"target {target}: requires {min_size}..{max_size} members, got {len(group)}"
            )
        if len(set(group)) != len(group):
            violations.append(f"target {target}: duplicate member uuvs in group")
        for uuv in group:
            if uuv not in uuvs:
                violations.append(f"uuv {uuv}: assigned to target {target} but not a known uuv")

    for target in members:
        if target not in targets:
            violations.append(f"target {target}: not a known target")

    assignment: dict[str, str] = {}
    for target in targets:
        for uuv in members.get(target, ()):
            if uuv in assignment:
                violations.append(f"uuv {uuv}: assigned to more than one target")
            else:
                assignment[uuv] = target
    for uuv, target in sorted(assignment.items()):
        if not problem.uuv_available.get(uuv, True):
            violations.append(f"uuv {uuv}: unavailable but assigned to target {target}")
        if problem.feasible_pairs is not None and (uuv, target) not in problem.feasible_pairs:
            violations.append(f"uuv {uuv}: assigned to infeasible target pair ({uuv}, {target})")
    return tuple(violations)
