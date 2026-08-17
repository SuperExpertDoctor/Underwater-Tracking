"""Lexicographic elastic group allocation (spec section 14).

The allocator assigns every target a group of 2-4 UUVs and minimizes,
in lexicographic order: (1) the number of active UUVs, (2) the blended
economic cost (energy + travel + reassignment + rotation), and (3) the
health left idle, preferring healthy UUVs in reserve on ties. Group
sizes are elastic: quality below the warning threshold grows a group to
three members (up to four when the target is explicitly degraded), a
group is kept unchanged while quality sits between the warning and
release thresholds (hysteresis), and a redundant member is released to
the reserve pool only after quality has exceeded the release threshold
for the configured hold time.

Membership is solved with binary variables ``x[u, t]`` (UUV assigned to
target) and ``a[u]`` (UUV active) in three ``scipy.optimize.milp``
passes over the same hard constraints: 2-4 members per target, zero
assignments for unavailable UUVs and for infeasible range/energy pairs,
and at most one target per UUV. When the solver is unavailable or the
model is infeasible, a deterministic bounded branch-and-bound
enumeration over group sizes and member combinations (in stable cost
order, pruning reused UUVs and unfillable remainders) produces the same
lexicographic result. Every returned solution is re-validated by
``validate_allocation``. There is no randomness anywhere: ids are sorted
before any matrix or combination is built, so identical input yields
identical output.
"""

from __future__ import annotations

import itertools
from math import isfinite
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]

from underwater_tracking.planning.validator import validate_allocation

# Integer scaling applied to economic costs so the objective tiers stay
# lexicographic under exact integer arithmetic.
_SCORE_SCALE = 1000.0


def _finite(value: float, label: str) -> float:
    if not isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _finite_at_least(value: float, label: str, minimum: float, *, strict: bool = False) -> None:
    _finite(value, label)
    if value < minimum or (strict and value == minimum):
        comparison = "greater than" if strict else "non-negative"
        raise ValueError(f"{label} must be {comparison} {minimum}")


@dataclass(frozen=True, slots=True)
class AllocationInput:
    """Everything the allocator needs to build one plan.

    ``uuv_ids`` and ``target_ids`` are the canonical ids; the allocator
    sorts both before building any matrix or enumeration.
    ``quality_by_target`` drives the elastic size policy (warning 0.65 /
    release 0.75 / hold 600 s by default). ``uuv_available`` marks
    unavailable (returning, failed, or otherwise unusable) UUVs, which
    can never be assigned. ``feasible_pairs`` is the set of
    (uuv, target) pairs that satisfy range, energy-return, safety,
    boundary and kinematics constraints; ``None`` means every pair is
    feasible. ``prior_members`` and ``assignment_age_s`` feed the release
    hysteresis: a group with three or more members is kept unchanged
    while quality is below the release threshold, and a redundant member
    is released only after ``release_hold_s`` seconds above it.
    ``energy_cost``, ``travel_cost`` and ``rotation_cost`` are per-pair
    economic costs (default 0); the reassignment cost is 0 for pairs in
    ``prior_members`` and ``reassignment_penalty`` otherwise.
    ``uuv_energy_fraction`` scores reserve health for the final
    tie-break. ``target_degraded`` marks targets with degraded
    observability, strong maneuvers, or a member failure, which may grow
    to four members.
    """

    uuv_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    quality_by_target: Mapping[str, float]
    uuv_available: Mapping[str, bool] = field(default_factory=dict)
    reserved_uuv_ids: AbstractSet[str] = frozenset()
    prior_members: Mapping[str, Sequence[str]] = field(default_factory=dict)
    assignment_age_s: Mapping[str, float] = field(default_factory=dict)
    feasible_pairs: AbstractSet[tuple[str, str]] | None = None
    target_degraded: AbstractSet[str] = frozenset()
    energy_cost: Mapping[tuple[str, str], float] = field(default_factory=dict)
    travel_cost: Mapping[tuple[str, str], float] = field(default_factory=dict)
    rotation_cost: Mapping[tuple[str, str], float] = field(default_factory=dict)
    uuv_energy_fraction: Mapping[str, float] = field(default_factory=dict)
    quality_warning: float = 0.65
    quality_release: float = 0.75
    release_hold_s: float = 600.0
    reassignment_penalty: float = 100.0
    required_quality_by_target: Mapping[str, float] = field(default_factory=dict)
    target_priority_by_target: Mapping[str, float] = field(default_factory=dict)
    uuv_passive_range_m: Mapping[str, float] = field(default_factory=dict)
    uuv_bearing_variance_rad2: Mapping[str, float] = field(default_factory=dict)
    uuv_speed_mps: Mapping[str, float] = field(default_factory=dict)
    uuv_max_turn_rate_rad_s: Mapping[str, float] = field(default_factory=dict)
    uuv_passive_sonar_available: Mapping[str, bool] = field(default_factory=dict)
    uuv_endurance_s: Mapping[str, float] = field(default_factory=dict)
    uuv_availability: Mapping[str, float] = field(default_factory=dict)
    plan_horizon_s: float = 600.0
    rotation_threshold: float = 0.3

    def __post_init__(self) -> None:
        _finite_at_least(self.quality_warning, "quality_warning", 0.0)
        _finite_at_least(self.quality_release, "quality_release", 0.0)
        _finite_at_least(self.release_hold_s, "release_hold_s", 0.0)
        _finite_at_least(self.reassignment_penalty, "reassignment_penalty", 0.0)
        _finite_at_least(self.plan_horizon_s, "plan_horizon_s", 0.0, strict=True)
        _finite(self.rotation_threshold, "rotation_threshold")
        if not 0.0 <= self.quality_warning < self.quality_release <= 1.0:
            raise ValueError("need 0 <= quality_warning < quality_release <= 1")
        if not 0.0 <= self.rotation_threshold <= 1.0:
            raise ValueError("rotation_threshold must be in [0, 1]")
        uuvs = frozenset(self.uuv_ids)
        targets = frozenset(self.target_ids)
        for target in self.target_ids:
            if target not in self.quality_by_target:
                raise ValueError(f"quality_by_target is missing target {target!r}")
        for target, quality in self.quality_by_target.items():
            _finite(quality, f"quality of target {target!r}")
            if not 0.0 <= quality <= 1.0:
                raise ValueError(f"quality of target {target!r} must be in [0, 1]")
        for target, quality in self.required_quality_by_target.items():
            if target not in targets:
                raise ValueError(f"required_quality_by_target mentions unknown target {target!r}")
            _finite(quality, f"required quality of target {target!r}")
            if not 0.0 <= quality <= 1.0:
                raise ValueError(f"required quality of target {target!r} must be in [0, 1]")
        for target, priority in self.target_priority_by_target.items():
            if target not in targets:
                raise ValueError(f"target_priority_by_target mentions unknown target {target!r}")
            if not isfinite(priority) or priority < 0.0:
                raise ValueError(f"priority of target {target!r} must be finite and non-negative")
        for uuv in self.uuv_available:
            if uuv not in uuvs:
                raise ValueError(f"uuv_available mentions unknown uuv {uuv!r}")
        for uuv in self.reserved_uuv_ids:
            if uuv not in uuvs:
                raise ValueError(f"reserved_uuv_ids mentions unknown uuv {uuv!r}")
        for target, members in self.prior_members.items():
            if target not in targets:
                raise ValueError(f"prior_members mentions unknown target {target!r}")
            for uuv in members:
                if uuv not in uuvs:
                    raise ValueError(f"prior_members mentions unknown uuv {uuv!r}")
        for target, age in self.assignment_age_s.items():
            if target not in targets:
                raise ValueError(f"assignment_age_s mentions unknown target {target!r}")
            _finite_at_least(age, f"assignment_age_s of target {target!r}", 0.0)
            if age < 0.0:
                raise ValueError(f"assignment_age_s of target {target!r} must be non-negative")
        for target in self.target_degraded:
            if target not in targets:
                raise ValueError(f"target_degraded mentions unknown target {target!r}")
        if self.feasible_pairs is not None:
            for uuv, target in self.feasible_pairs:
                if uuv not in uuvs or target not in targets:
                    raise ValueError(f"feasible_pairs contains unknown pair ({uuv!r}, {target!r})")
        for costs in (self.energy_cost, self.travel_cost, self.rotation_cost):
            for uuv, target in costs:
                if uuv not in uuvs or target not in targets:
                    raise ValueError(f"cost table mentions unknown pair ({uuv!r}, {target!r})")
            for value in costs.values():
                _finite_at_least(value, "economic cost", 0.0)
        for uuv, fraction in self.uuv_energy_fraction.items():
            if uuv not in uuvs:
                raise ValueError(f"uuv_energy_fraction mentions unknown uuv {uuv!r}")
            _finite(fraction, f"energy fraction of uuv {uuv!r}")
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"energy fraction of uuv {uuv!r} must be in [0, 1]")
        for values, label, allow_zero in (
            (self.uuv_passive_range_m, "passive range", False),
            (self.uuv_bearing_variance_rad2, "bearing variance", False),
            (self.uuv_speed_mps, "speed", True),
            (self.uuv_max_turn_rate_rad_s, "turn rate", False),
        ):
            for uuv, value in values.items():
                if uuv not in uuvs:
                    raise ValueError(f"{label} mentions unknown uuv {uuv!r}")
                _finite_at_least(value, f"{label} of uuv {uuv!r}", 0.0, strict=not allow_zero)
                if value < 0.0 or (value == 0.0 and not allow_zero):
                    raise ValueError(f"{label} of uuv {uuv!r} must be positive")
        for uuv in self.uuv_passive_sonar_available:
            if uuv not in uuvs:
                raise ValueError(f"passive sonar table mentions unknown uuv {uuv!r}")
        for values, label, minimum in (
            (self.uuv_endurance_s, "endurance", 0.0),
            (self.uuv_availability, "availability", 0.0),
        ):
            for uuv, value in values.items():
                if uuv not in uuvs:
                    raise ValueError(f"{label} table mentions unknown uuv {uuv!r}")
                _finite_at_least(value, f"{label} of uuv {uuv!r}", minimum)
                if value < minimum:
                    raise ValueError(f"{label} of uuv {uuv!r} must be finite and non-negative")
                if label == "availability" and value > 1.0:
                    raise ValueError(f"availability of uuv {uuv!r} must be in [0, 1]")

    @classmethod
    def synthetic(
        cls,
        uuv_count: int = 6,
        target_count: int = 2,
        feasible_pair_quality: float = 0.8,
        reserved_uuv_ids: AbstractSet[str] = frozenset(),
    ) -> AllocationInput:
        """Build a deterministic problem where every pair is feasible.

        ``feasible_pair_quality`` becomes the quality of every target;
        all UUVs are available with full energy, all economic costs are
        zero, and there is no prior assignment. ``reserved_uuv_ids`` are
        excluded from assignment.
        """
        target_ids = tuple(f"target_{i}" for i in range(target_count))
        return cls(
            uuv_ids=tuple(f"uuv_{i}" for i in range(uuv_count)),
            target_ids=target_ids,
            quality_by_target={target: feasible_pair_quality for target in target_ids},
            reserved_uuv_ids=reserved_uuv_ids,
        )


@dataclass(frozen=True, slots=True)
class AllocationObjective:
    """Economic breakdown of a solved allocation."""

    active_count: int
    energy_cost: float
    travel_cost: float
    reassignment_cost: float
    rotation_cost: float
    reserve_health: float


@dataclass(frozen=True, slots=True)
class AllocationSolution:
    """A validated group allocation.

    ``members_by_target`` maps each target id to its member uuv ids in
    stable sorted order; ``reserve_ids`` are the available, unassigned
    uuvs in stable order. ``solver_status`` is ``"milp"``,
    ``"fallback"``, or ``"infeasible"``. ``hard_violations`` lists every
    hard constraint violated by this solution (empty when it is
    feasible).
    """

    members_by_target: Mapping[str, tuple[str, ...]]
    reserve_ids: tuple[str, ...]
    objective: AllocationObjective
    solver_status: str
    hard_violations: tuple[str, ...]


def _group_size_bounds(
    quality: float,
    prior_size: int,
    assignment_age_s: float,
    degraded: bool,
    quality_warning: float,
    quality_release: float,
    release_hold_s: float,
) -> tuple[int, int]:
    """Return the (minimum, maximum) members allowed for one target.

    Policy, in priority order:

    1. quality below the warning threshold: grow to three members (up to
       four when the target is explicitly degraded);
    2. no redundancy (prior size at most two): a normal 2-3 group, or
       2-4 when degraded;
    3. quality at or above the release threshold for the full hold time:
       release exactly one redundant member to the reserve pool;
    4. otherwise (hysteresis): keep the group exactly as it was.
    """
    if quality < quality_warning:
        return (3, 4 if degraded else 3)
    if prior_size <= 2:
        return (2, 4 if degraded else 3)
    if quality >= quality_release and assignment_age_s >= release_hold_s:
        return (prior_size - 1, prior_size - 1)
    return (prior_size, prior_size)


def _target_bounds(problem: AllocationInput, target: str) -> tuple[int, int]:
    """Elastic (minimum, maximum) group size for one target of ``problem``."""
    minimum, maximum = _group_size_bounds(
        quality=problem.quality_by_target[target],
        prior_size=min(4, len(problem.prior_members.get(target, ()))),
        assignment_age_s=problem.assignment_age_s.get(target, 0.0),
        degraded=target in problem.target_degraded,
        quality_warning=problem.quality_warning,
        quality_release=problem.quality_release,
        release_hold_s=problem.release_hold_s,
    )
    if problem.quality_by_target[target] < problem.required_quality_by_target.get(target, 0.0):
        return max(3, minimum), max(3, maximum)
    return minimum, maximum


def _available(problem: AllocationInput, uuv: str) -> bool:
    if uuv in problem.reserved_uuv_ids:
        return False
    return problem.uuv_available.get(uuv, True)


def _int_pair_cost(problem: AllocationInput, uuv: str, target: str) -> int:
    """Integer-scaled economic cost of assigning ``uuv`` to ``target``."""
    energy = problem.energy_cost.get((uuv, target), 0.0)
    travel = problem.travel_cost.get((uuv, target), 0.0)
    rotation = problem.rotation_cost.get((uuv, target), _rotation_penalty(problem, uuv))
    if uuv in problem.prior_members.get(target, ()):
        reassignment = 0.0
    else:
        reassignment = problem.reassignment_penalty
    capability_loss = 1.0 - _capability_score(problem, uuv)
    priority = problem.target_priority_by_target.get(target, 0.0)
    capability_cost = (
        (1.0 + priority) * capability_loss * max(1.0, problem.reassignment_penalty)
        + problem.required_quality_by_target.get(target, 0.0)
        * priority
        * capability_loss
        * problem.reassignment_penalty
    )
    return round(
        _SCORE_SCALE * (energy + travel + rotation + reassignment + capability_cost)
    )


def _rotation_penalty(problem: AllocationInput, uuv: str) -> float:
    """Make a healthy replacement preferable to retaining a rotating member."""
    prior_member = any(
        uuv in members for members in problem.prior_members.values()
    )
    if (
        prior_member
        and problem.uuv_energy_fraction.get(uuv, 1.0) < problem.rotation_threshold
    ):
        return 10.0 * problem.reassignment_penalty
    return 0.0


def _pair_feasible(problem: AllocationInput, uuv: str, target: str) -> bool:
    """Apply explicit feasibility and optional passive-range bounds."""
    if problem.feasible_pairs is not None and (uuv, target) not in problem.feasible_pairs:
        return False
    if not problem.uuv_passive_sonar_available.get(uuv, True):
        return False
    if problem.uuv_availability.get(uuv, 1.0) <= 0.0:
        return False
    if problem.uuv_endurance_s.get(uuv, problem.plan_horizon_s) < problem.plan_horizon_s:
        return False
    passive_range = problem.uuv_passive_range_m.get(uuv)
    distance = problem.travel_cost.get((uuv, target))
    return passive_range is None or distance is None or distance <= passive_range


def _capability_score(problem: AllocationInput, uuv: str) -> float:
    """Relative sensing and maneuver quality, normalized to fleet defaults."""
    bearing = min(1.0, 0.01 / problem.uuv_bearing_variance_rad2.get(uuv, 0.01))
    speed = min(1.0, problem.uuv_speed_mps.get(uuv, 4.0) / 4.0)
    turn = min(
        1.0,
        problem.uuv_max_turn_rate_rad_s.get(uuv, 0.05235987755982989)
        / 0.05235987755982989,
    )
    return min(bearing, speed, turn)


def projected_tracking_quality(
    problem: AllocationInput,
    target: str,
    members: Sequence[str],
) -> float:
    """Recompute projected tracking quality from raw group capability inputs."""
    if not members:
        return 0.0
    capability = sum(_capability_score(problem, member) for member in members) / len(members)
    quality = (
        problem.quality_by_target[target]
        + 0.1 * (len(members) - 2)
        - 0.2 * (1.0 - capability)
    )
    return min(1.0, max(0.0, quality))


def _milp_assign(
    problem: AllocationInput,
    bounds_by_target: Mapping[str, tuple[int, int]],
    uuvs: list[str],
    targets: list[str],
) -> np.ndarray[Any, Any] | None:
    """Solve the three lexicographic MILP passes; return ``x`` or ``None``.

    Pass 1 finds any hard-feasible solution, pass 2 minimizes the active
    UUV count, and pass 3 fixes that count and minimizes the economic
    cost with a lexicographically subordinate term that prefers healthy
    UUVs idle in reserve. ``None`` signals solver failure or
    infeasibility, in which case the caller falls back to the
    deterministic enumeration.
    """
    uuv_count = len(uuvs)
    target_count = len(targets)

    def x_index(u: int, t: int) -> int:
        return u * target_count + t

    def a_index(u: int) -> int:
        return uuv_count * target_count + u

    var_count = uuv_count * target_count + uuv_count
    integrality = np.ones(var_count)
    bounds = Bounds(np.zeros(var_count), np.ones(var_count))

    rows: list[np.ndarray[Any, Any]] = []
    lower: list[float] = []
    upper: list[float] = []
    for t, target in enumerate(targets):
        row = np.zeros(var_count)
        for u in range(uuv_count):
            row[x_index(u, t)] = 1.0
        rows.append(row)
        min_size, max_size = bounds_by_target[target]
        lower.append(float(min_size))
        upper.append(float(max_size))
    for u in range(uuv_count):
        row = np.zeros(var_count)
        for t in range(target_count):
            row[x_index(u, t)] = 1.0
        row[a_index(u)] = -1.0
        rows.append(row)
        lower.append(0.0)
        upper.append(0.0)
    for u, uuv in enumerate(uuvs):
        if not _available(problem, uuv):
            row = np.zeros(var_count)
            for t in range(target_count):
                row[x_index(u, t)] = 1.0
            rows.append(row)
            lower.append(0.0)
            upper.append(0.0)
    for u, uuv in enumerate(uuvs):
        for t, target in enumerate(targets):
            if not _pair_feasible(problem, uuv, target):
                row = np.zeros(var_count)
                row[x_index(u, t)] = 1.0
                rows.append(row)
                lower.append(0.0)
                upper.append(0.0)
    try:
        hard_constraints = [
            LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper))
        ]
        result = milp(
            c=np.zeros(var_count),
            constraints=hard_constraints,
            integrality=integrality,
            bounds=bounds,
        )
        if not result.success or result.status != 0:
            return None

        active_cost = np.zeros(var_count)
        for u in range(uuv_count):
            active_cost[a_index(u)] = 1.0
        result = milp(
            c=active_cost,
            constraints=hard_constraints,
            integrality=integrality,
            bounds=bounds,
        )
        if not result.success or result.status != 0:
            return None
        min_active = round(float(np.sum(result.x[uuv_count * target_count :])))

        fixed_active = [
            LinearConstraint(active_cost[None, :], [float(min_active)], [float(min_active)])
        ]
        # Pass 3 objective: economic cost at the top, then the reserve
        # health tie-break. The cost coefficient weight makes one
        # economic unit outweigh every possible health difference, so
        # the tiers stay lexicographic.
        economic_cost = np.zeros(var_count)
        cost_weight = (sum(1 for uuv in uuvs if _available(problem, uuv)) + 1) * 1000
        for u, uuv in enumerate(uuvs):
            for t, target in enumerate(targets):
                economic_cost[x_index(u, t)] = float(
                    cost_weight * _int_pair_cost(problem, uuv, target)
                )
            if _available(problem, uuv):
                economic_cost[a_index(u)] = float(
                    round(_SCORE_SCALE * problem.uuv_energy_fraction.get(uuv, 1.0))
                )
        result = milp(
            c=economic_cost,
            constraints=hard_constraints + fixed_active,
            integrality=integrality,
            bounds=bounds,
        )
        if not result.success or result.status != 0:
            return None
        return np.asarray(result.x, dtype=float)
    except Exception:  # noqa: BLE001 - solver failure or infeasibility degrades to the deterministic fallback
        # Solver unavailable or a numerical failure: the caller falls
        # back to the deterministic enumeration.
        return None


def _extract_solution(
    problem: AllocationInput,
    x: np.ndarray[Any, Any],
    uuvs: list[str],
    targets: list[str],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Convert a MILP variable vector into members and reserves."""
    members: dict[str, tuple[str, ...]] = {}
    assigned: set[str] = set()
    for t, target in enumerate(targets):
        group = tuple(
            uuvs[u] for u in range(len(uuvs)) if x[u * len(targets) + t] > 0.5
        )
        members[target] = group
        assigned.update(group)
    reserves = tuple(u for u in uuvs if _available(problem, u) and u not in assigned)
    return members, reserves


def _objective(
    problem: AllocationInput,
    members: Mapping[str, tuple[str, ...]],
    reserves: tuple[str, ...],
) -> AllocationObjective:
    """Economic breakdown recomputed from the members and reserves."""
    energy_cost = 0.0
    travel_cost = 0.0
    rotation_cost = 0.0
    reassignment_cost = 0.0
    for target, group in members.items():
        for uuv in group:
            energy_cost += problem.energy_cost.get((uuv, target), 0.0)
            travel_cost += problem.travel_cost.get((uuv, target), 0.0)
            rotation_cost += problem.rotation_cost.get(
                (uuv, target), _rotation_penalty(problem, uuv)
            )
            if uuv not in problem.prior_members.get(target, ()):
                reassignment_cost += problem.reassignment_penalty
    reserve_health = sum(
        problem.uuv_energy_fraction.get(uuv, 1.0) for uuv in reserves
    )
    return AllocationObjective(
        active_count=sum(len(group) for group in members.values()),
        energy_cost=energy_cost,
        travel_cost=travel_cost,
        reassignment_cost=reassignment_cost,
        rotation_cost=rotation_cost,
        reserve_health=reserve_health,
    )


def _fallback_assign(
    problem: AllocationInput,
    bounds_by_target: Mapping[str, tuple[int, int]],
    uuvs: list[str],
    targets: list[str],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]] | None:
    """Deterministic bounded branch-and-bound enumeration (spec 14.2).

    Tries total active counts from the smallest possible upwards; within
    a count, explores group sizes in ascending order and member
    combinations in stable (cost, id) order. Pruning is threefold: a
    partial that reuses a UUV, one whose size window cannot fill the
    remaining targets, and one whose cost lower bound cannot beat the
    best solution found so far (cost first, then reserve health, per
    spec 14.1(3)-(4)) are all skipped. The first feasible total is
    returned with the best (cost, then reserve health) complete
    assignment. Returns ``(members, reserves)`` or ``None`` when no
    assignment exists.
    """
    target_count = len(targets)
    min_sizes = [bounds_by_target[target][0] for target in targets]
    max_sizes = [bounds_by_target[target][1] for target in targets]
    min_total = sum(min_sizes)
    max_total = min(sum(max_sizes), len(uuvs))
    if min_total > max_total:
        return None

    # Reserve health is the sum of the health of available, unassigned
    # UUVs, i.e. the total available health minus the health of the
    # assigned members. Per-target minima bound how much health the
    # remaining targets must consume, which is what makes the tier-4
    # tie-break prunable.
    health_int = {
        uuv: round(_SCORE_SCALE * problem.uuv_energy_fraction.get(uuv, 1.0))
        for uuv in uuvs
        if _available(problem, uuv)
    }
    total_health = sum(health_int.values())

    # Candidate member combinations per (target index, size), sorted by
    # (cost, member tuple) so the enumeration is stable and cheapest
    # first.
    candidates: list[dict[int, list[tuple[int, tuple[str, ...]]]]] = []
    cheapest_member_cost: list[int] = []
    cheapest_member_health: list[int] = []
    for target in targets:
        eligible = [
            uuv
            for uuv in uuvs
            if _available(problem, uuv)
            and _pair_feasible(problem, uuv, target)
        ]
        cheapest = min(
            (_int_pair_cost(problem, uuv, target) for uuv in eligible),
            default=0,
        )
        cheapest_member_cost.append(cheapest)
        cheapest_member_health.append(
            min((health_int[uuv] for uuv in eligible), default=0)
        )
        per_size: dict[int, list[tuple[int, tuple[str, ...]]]] = {}
        index = len(candidates)
        for size in range(min_sizes[index], max_sizes[index] + 1):
            combos = [
                (
                    sum(_int_pair_cost(problem, uuv, target) for uuv in combo),
                    combo,
                )
                for combo in itertools.combinations(eligible, size)
            ]
            combos.sort(key=lambda item: (item[0], item[1]))
            per_size[size] = combos
        candidates.append(per_size)

    suffix_min_members = [0] * (target_count + 1)
    suffix_max_members = [0] * (target_count + 1)
    suffix_min_cost = [0] * (target_count + 1)
    suffix_min_health = [0] * (target_count + 1)
    for i in range(target_count - 1, -1, -1):
        suffix_min_members[i] = suffix_min_members[i + 1] + min_sizes[i]
        suffix_max_members[i] = suffix_max_members[i + 1] + max_sizes[i]
        suffix_min_cost[i] = (
            suffix_min_cost[i + 1] + min_sizes[i] * cheapest_member_cost[i]
        )
        suffix_min_health[i] = (
            suffix_min_health[i + 1] + min_sizes[i] * cheapest_member_health[i]
        )

    def can_beat_best(
        cost_lower: int,
        health_upper: int,
        best: tuple[int, int],
    ) -> bool:
        """Whether some completion of this partial could beat ``best``.

        ``best`` is ``(cost, -reserve_health)``. A partial can only
        improve it by reaching a strictly lower cost, or the same cost
        with strictly more reserve health; ``cost_lower`` and
        ``health_upper`` bound what any completion can attain.
        """
        best_cost, best_neg_health = best
        if cost_lower > best_cost:
            return False
        if cost_lower < best_cost:
            return True
        return health_upper > -best_neg_health

    best_key: tuple[int, int] | None = None
    best_members: list[tuple[str, ...]] | None = None
    partial_members: list[tuple[str, ...]] = []

    def search(
        i: int,
        budget: int,
        used: frozenset[str],
        cost: int,
        assigned_health: int,
    ) -> None:
        nonlocal best_key, best_members
        if i == target_count:
            if budget == 0:
                key = (cost, -(total_health - assigned_health))
                if best_key is None or key < best_key:
                    best_key = key
                    best_members = list(partial_members)
            return
        if budget < suffix_min_members[i] or budget > suffix_max_members[i]:
            return
        if best_key is not None and not can_beat_best(
            cost + suffix_min_cost[i],
            total_health - assigned_health - suffix_min_health[i],
            best_key,
        ):
            return
        min_size = max(min_sizes[i], budget - suffix_max_members[i + 1])
        max_size = min(max_sizes[i], budget - suffix_min_members[i + 1])
        for size in range(min_size, max_size + 1):
            for combo_cost, combo in candidates[i][size]:
                if any(uuv in used for uuv in combo):
                    continue
                new_used = used | frozenset(combo)
                new_cost = cost + combo_cost
                new_health = assigned_health + sum(health_int[uuv] for uuv in combo)
                if best_key is not None and not can_beat_best(
                    new_cost + suffix_min_cost[i + 1],
                    total_health - new_health - suffix_min_health[i + 1],
                    best_key,
                ):
                    continue
                partial_members.append(combo)
                search(i + 1, budget - size, new_used, new_cost, new_health)
                partial_members.pop()

    for total in range(min_total, max_total + 1):
        best_key = None
        best_members = None
        search(0, total, frozenset(), 0, 0)
        if best_members is not None:
            members = {
                target: combo for target, combo in zip(targets, best_members, strict=True)
            }
            assigned = set(itertools.chain.from_iterable(best_members))
            reserves = tuple(
                uuv for uuv in uuvs if _available(problem, uuv) and uuv not in assigned
            )
            return members, reserves
    return None


def allocate_groups(problem: AllocationInput) -> AllocationSolution:
    """Solve the lexicographic elastic group allocation for ``problem``.

    Prefers the three-pass MILP solve; degrades to the deterministic
    enumeration when the solver is unavailable or infeasible. Every
    result is re-validated against the hard constraints before being
    returned.
    """
    uuvs = sorted(problem.uuv_ids)
    targets = sorted(problem.target_ids)
    bounds_by_target = {target: _target_bounds(problem, target) for target in targets}

    x = _milp_assign(problem, bounds_by_target, uuvs, targets)
    if x is not None:
        members, reserves = _extract_solution(problem, x, uuvs, targets)
        solution = AllocationSolution(
            members_by_target=members,
            reserve_ids=reserves,
            objective=_objective(problem, members, reserves),
            solver_status="milp",
            hard_violations=(),
        )
    else:
        fallback = _fallback_assign(problem, bounds_by_target, uuvs, targets)
        if fallback is not None:
            members, reserves = fallback
            solution = AllocationSolution(
                members_by_target=members,
                reserve_ids=reserves,
                objective=_objective(problem, members, reserves),
                solver_status="fallback",
                hard_violations=(),
            )
        else:
            solution = AllocationSolution(
                members_by_target={target: () for target in targets},
                reserve_ids=tuple(u for u in uuvs if _available(problem, u)),
                objective=_objective(
                    problem, {}, tuple(u for u in uuvs if _available(problem, u))
                ),
                solver_status="infeasible",
                hard_violations=(),
            )
    projected_quality = {
        target: projected_tracking_quality(problem, target, members)
        for target, members in solution.members_by_target.items()
    }
    violations = list(validate_allocation(problem, solution))
    for target in targets:
        required = problem.required_quality_by_target.get(target, 0.0)
        projected = projected_quality.get(target, 0.0)
        if projected < required:
            violations.append(
                f"target {target}: projected quality {projected:.3f} below required {required:.3f}"
            )
    return replace(solution, hard_violations=tuple(violations))
