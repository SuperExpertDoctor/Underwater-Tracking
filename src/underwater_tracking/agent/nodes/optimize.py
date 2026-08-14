# src/underwater_tracking/agent/nodes/optimize.py
"""Deterministic plan optimization over verified strategies (spec 14, 15.2).

For every verified ``StrategyProposal`` of the current ``StrategySet`` the
optimizer adapts the Foundation services: ``allocate_groups`` assigns
2-4 members per target under the elastic size policy and the lexicographic
economic objective, and ``plan_group_waypoints`` plans a robust short
waypoint sequence for every group. Each candidate ``TrackingPlan`` carries
the expected quality/FIM/resources/energy, the churn ``PlanDiff`` versus
the broadcast plan, and a deterministic hard-violation count. Candidates
are sorted lexicographically by (hard violations, active count, economic
cost, stable concept order).

A periodic ``hold_current`` review that does not materially improve the
broadcast plan selects the broadcast plan itself — no new revision. When
the broadcast plan is infeasible (a member failed or is returning, or a
directive disabled it) or the full allocation is infeasible, the emergency
path produces a ``DEGRADED`` plan retaining the highest-priority feasible
targets (spec 15.3, 18). There is no randomness anywhere: ids and concepts
are sorted before any decision, and the Foundation services are pure.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass

import numpy as np

from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import (
    PlanDiff,
    StrategyProposal,
    StrategySet,
    TrackingPlan,
    Waypoint,
)
from underwater_tracking.domain.models import (
    GroupReport,
    TargetBelief,
    UUVStatus,
)
from underwater_tracking.planning.allocation import (
    AllocationInput,
    AllocationSolution,
    allocate_groups,
)
from underwater_tracking.planning.waypoints import plan_group_waypoints

# Default scenario bounds shared with the Foundation planner.
_DEFAULT_BOUNDS: tuple[float, float, float, float] = (
    -5000.0,
    5000.0,
    -5000.0,
    5000.0,
)

# Energy fraction below which a UUV triggers rotation (spec 13: remaining
# energy only enough to return with a 10% safety margin). The group size
# contract (2-4 members) is enforced by the allocator and re-checked by the
# independent commit validation.
_ROTATION_THRESHOLD = 0.3


@dataclass(frozen=True)
class PlanningConfig:
    """Deterministic planning parameters shared by optimize and commit."""

    bounds: tuple[float, float, float, float] = _DEFAULT_BOUNDS
    max_range_m: float = 4000.0
    min_range_m: float = 500.0
    min_separation_m: float = 300.0
    bearing_variance: float = 0.01
    beam_width: int = 8
    range_bins: int = 5
    horizon_steps: int = 3
    replan_period_s: float = 30.0
    return_reserve: float = 0.1
    quality_warning: float = 0.65
    quality_release: float = 0.75
    release_hold_s: float = 600.0
    reassignment_penalty: float = 100.0
    plan_horizon_s: int = 600
    # A candidate only counts as materially better when its economic cost
    # drops by at least this fraction (spec 15.2: no material gain -> hold).
    improvement_margin: float = 0.01


@dataclass(frozen=True)
class CandidateMetrics:
    """Deterministic objective of one candidate plan."""

    hard_violations: tuple[str, ...]
    active_count: int
    economic_cost: float


@dataclass(frozen=True)
class CandidateEvaluation:
    """One candidate plan with its metrics and stable proposal order."""

    plan: TrackingPlan
    metrics: CandidateMetrics
    index: int


# Shared immutable default for node constructors (B008: no call in defaults).
_DEFAULT_CONFIG = PlanningConfig()


def optimize_candidates(
    snapshot: PlanningSnapshot,
    strategy_set: StrategySet,
    config: PlanningConfig = _DEFAULT_CONFIG,
) -> tuple[CandidateEvaluation, ...]:
    """Optimize every verified strategy into a sorted candidate tuple.

    For each proposal the Foundation allocation and waypoint services build
    one candidate ``TrackingPlan``; the emergency path replaces it with a
    ``DEGRADED`` plan over the highest-priority feasible targets whenever
    the broadcast plan is infeasible or the full allocation is infeasible.
    The returned tuple is sorted lexicographically by (hard violations,
    active count, economic cost, stable concept order).
    """
    previous_infeasible = _previous_plan_infeasible(snapshot)
    evaluations: list[CandidateEvaluation] = []
    for index, proposal in enumerate(strategy_set):
        if previous_infeasible:
            evaluations.append(_emergency_evaluation(proposal, snapshot, config, index))
            continue
        problem = _build_problem(
            snapshot, proposal, tuple(sorted(proposal.target_priorities)), config
        )
        solution = allocate_groups(problem)
        if solution.hard_violations:
            evaluations.append(_emergency_evaluation(proposal, snapshot, config, index))
        else:
            evaluations.append(
                _build_evaluation(proposal, snapshot, problem, solution, config, index)
            )
    return tuple(sorted(evaluations, key=lambda evaluation: _sort_key(evaluation)))


def select_candidate(
    snapshot: PlanningSnapshot,
    evaluations: tuple[CandidateEvaluation, ...],
    config: PlanningConfig = _DEFAULT_CONFIG,
) -> TrackingPlan:
    """Pick the best candidate; a hold_current review keeps the broadcast plan.

    The lexicographic best candidate is returned, unless it is a
    ``hold_current`` proposal, the broadcast plan still exists and is
    feasible, and it offers no material objective improvement — in which
    case the broadcast plan itself is selected unchanged (no new revision,
    spec 15.2).
    """
    best = evaluations[0].plan
    active = snapshot.active_plan
    if (
        best.concept == "hold_current"
        and active is not None
        and not _previous_plan_infeasible(snapshot)
        and not material_improvement(best, active, config)
    ):
        return active
    return best


def material_improvement(
    candidate: TrackingPlan,
    active: TrackingPlan,
    config: PlanningConfig = _DEFAULT_CONFIG,
) -> bool:
    """Whether ``candidate`` beats the broadcast plan by a material margin.

    Fewer active UUVs is always material; with equal active counts the
    economic cost must drop by at least ``config.improvement_margin``.
    """
    if candidate.predicted_active_count != active.predicted_active_count:
        return candidate.predicted_active_count < active.predicted_active_count
    return candidate.predicted_energy < active.predicted_energy * (
        1.0 - config.improvement_margin
    )


class OptimizeNode:
    """LangGraph node: optimize the verified strategies and store candidates.

    Loads the immutable ``PlanningSnapshot`` through ``snapshot_provider``,
    runs ``optimize_candidates`` + ``select_candidate``, stores every
    candidate under a deterministic reference in ``store`` (the selected
    plan under ``selected_plan_ref``), and returns the state fragment
    ``{"candidate_plan_refs", "selected_plan_ref"}`` (spec 8.1).
    """

    def __init__(
        self,
        *,
        snapshot_provider: Callable[[str], PlanningSnapshot],
        store: MutableMapping[str, TrackingPlan],
        config: PlanningConfig = _DEFAULT_CONFIG,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        self._store = store
        self._config = config

    def __call__(self, state: CarrierState) -> CarrierState:
        ref = state.get("snapshot_ref")
        if ref is None:
            raise ValueError("OptimizeNode requires snapshot_ref in state")
        snapshot = self._snapshot_provider(ref)
        strategy_set = state.get("strategy_set")
        if strategy_set is None or not strategy_set.proposals:
            raise ValueError("OptimizeNode requires a non-empty strategy_set")
        evaluations = optimize_candidates(snapshot, strategy_set, self._config)
        candidate_refs: list[str] = []
        for evaluation in evaluations:
            candidate_ref = self._ref(snapshot, evaluation.index)
            self._store[candidate_ref] = evaluation.plan
            candidate_refs.append(candidate_ref)
        selected_ref = self._ref(snapshot, len(evaluations))
        self._store[selected_ref] = select_candidate(
            snapshot, evaluations, self._config
        )
        return {
            "candidate_plan_refs": tuple(candidate_refs),
            "selected_plan_ref": selected_ref,
        }

    @staticmethod
    def _ref(snapshot: PlanningSnapshot, index: int) -> str:
        return f"{snapshot.scenario_id}:candidate:{index}:{snapshot.snapshot_revision}"


def _sort_key(evaluation: CandidateEvaluation) -> tuple[int, int, float, int]:
    """Lexicographic candidate order: violations, active count, cost, concept order."""
    return (
        len(evaluation.metrics.hard_violations),
        evaluation.metrics.active_count,
        evaluation.metrics.economic_cost,
        evaluation.index,
    )


def _build_problem(
    snapshot: PlanningSnapshot,
    proposal: StrategyProposal,
    targets: Sequence[str],
    config: PlanningConfig,
) -> AllocationInput:
    """Adapt the immutable snapshot + proposal into an ``AllocationInput``."""
    situation = snapshot.situation
    uuvs = tuple(sorted(uuv.uuv_id for uuv in situation.uuvs))
    status_by_id = {uuv.uuv_id: uuv.status for uuv in situation.uuvs}
    energy_by_id = {uuv.uuv_id: uuv.energy_fraction for uuv in situation.uuvs}
    speed_by_id = {uuv.uuv_id: uuv.speed_mps for uuv in situation.uuvs}
    position_by_id = {uuv.uuv_id: uuv.position_xy for uuv in situation.uuvs}
    disabled = {
        uuv_id
        for directive in snapshot.applied_directives
        for uuv_id in directive.disabled_uuv_ids
    }
    unavailable = {
        uuv_id
        for uuv_id in uuvs
        if status_by_id[uuv_id] in (UUVStatus.FAILED, UUVStatus.RETURNING)
        or uuv_id in disabled
    }
    uuv_available = {uuv_id: uuv_id not in unavailable for uuv_id in uuvs}
    target_mean = {
        target: (_belief(snapshot, target).mean[0], _belief(snapshot, target).mean[1])
        for target in targets
    }
    feasible_pairs = {
        (uuv_id, target)
        for uuv_id in uuvs
        for target in targets
        if uuv_available[uuv_id]
        and _distance(position_by_id[uuv_id], target_mean[target]) <= config.max_range_m
        and energy_by_id[uuv_id] >= config.return_reserve
        and not _locked_to_other(uuv_id, target, snapshot, targets)
    }
    travel_cost = {
        (uuv_id, target): _distance(position_by_id[uuv_id], target_mean[target])
        for uuv_id in uuvs
        for target in targets
    }
    energy_cost = {
        (uuv_id, target): distance / max(speed_by_id[uuv_id], 0.1)
        for (uuv_id, target), distance in travel_cost.items()
    }
    active = snapshot.active_plan
    prior_members: dict[str, Sequence[str]] = {}
    for target in targets:
        if active is not None:
            prior = active.member_ids_by_target.get(target, ())
        else:
            prior = _report(snapshot, target).member_ids
        prior_members[target] = prior
    degraded_targets = {
        target
        for target in targets
        if _report(snapshot, target).quality.instant < config.quality_warning
        or bool(_report(snapshot, target).quality.hard_guard_reasons)
        or any(member in unavailable for member in prior_members[target])
    }
    return AllocationInput(
        uuv_ids=uuvs,
        target_ids=tuple(targets),
        quality_by_target={
            target: _report(snapshot, target).quality.ewma for target in targets
        },
        uuv_available=uuv_available,
        prior_members=prior_members,
        feasible_pairs=feasible_pairs,
        target_degraded=degraded_targets,
        energy_cost=energy_cost,
        travel_cost=travel_cost,
        uuv_energy_fraction=energy_by_id,
        quality_warning=config.quality_warning,
        quality_release=config.quality_release,
        release_hold_s=config.release_hold_s,
        reassignment_penalty=config.reassignment_penalty,
    )


def _build_evaluation(
    proposal: StrategyProposal,
    snapshot: PlanningSnapshot,
    problem: AllocationInput,
    solution: AllocationSolution,
    config: PlanningConfig,
    index: int,
    *,
    degraded: bool = False,
) -> CandidateEvaluation:
    """Construct the candidate ``TrackingPlan`` and its deterministic metrics."""
    members_by_target = {
        target: solution.members_by_target.get(target, ()) for target in problem.target_ids
    }
    waypoints: dict[str, tuple[Waypoint, ...]] = {}
    previous_by_member: Mapping[str, tuple[Waypoint, ...]] | None = None
    active = snapshot.active_plan
    if active is not None:
        previous_by_member = active.waypoints_by_member
    for target in sorted(members_by_target):
        members = members_by_target[target]
        if not members:
            continue
        waypoints.update(_plan_waypoints(snapshot, target, members, config, previous_by_member))

    revision = active.revision + 1 if active is not None else 1
    plan_id = f"{snapshot.scenario_id}:plan:{revision}"
    active_members = tuple(
        sorted(member for group in members_by_target.values() for member in group)
    )
    standby_ids = solution.reserve_ids
    roles: dict[str, str] = {}
    for members in members_by_target.values():
        roles.update(
            {
                member: "lead" if rank == 0 else "wing"
                for rank, member in enumerate(sorted(members))
            }
        )
    rotation_conditions = {
        target: f"energy_reserve_{config.return_reserve}"
        for target, members in members_by_target.items()
        if any(
            _energy(snapshot, member) < _ROTATION_THRESHOLD for member in members
        )
    }
    predicted_quality = {
        target: _report(snapshot, target).quality.ewma for target in problem.target_ids
    }
    predicted_fim = {
        target: _belief(snapshot, target).fim_min_eigenvalue
        for target in problem.target_ids
    }
    active_count = len(active_members)
    economic_cost = (
        solution.objective.energy_cost
        + solution.objective.travel_cost
        + solution.objective.reassignment_cost
        + solution.objective.rotation_cost
    )
    predicted_risk = (
        sum(
            1.0
            for target in problem.target_ids
            if _report(snapshot, target).quality.ewma < config.quality_warning
        )
        / len(problem.target_ids)
        if problem.target_ids
        else 0.0
    )
    violations = tuple(solution.hard_violations)
    if not degraded:
        tracked = {
            report.target_id for report in snapshot.situation.group_reports
        }
        missing = sorted(tracked - set(problem.target_ids))
        violations = violations + tuple(
            f"target {target}: no plan group" for target in missing
        )
    plan = TrackingPlan(
        plan_id=plan_id,
        scenario_id=snapshot.scenario_id,
        revision=revision,
        base_snapshot_revision=snapshot.snapshot_revision,
        status="degraded" if degraded else "draft",
        valid_from_s=snapshot.sim_time_s,
        valid_until_s=snapshot.sim_time_s + config.plan_horizon_s,
        concept=proposal.concept,
        target_priorities={
            target: proposal.target_priorities[target] for target in problem.target_ids
        },
        required_quality={
            target: proposal.required_quality[target] for target in problem.target_ids
        },
        member_ids_by_target=members_by_target,
        roles_by_member=roles,
        waypoints_by_member=waypoints,
        rotation_conditions=rotation_conditions,
        release_actions={
            target: proposal.reinforcement_policy[target]
            for target in problem.target_ids
        },
        active_uuv_ids=active_members,
        standby_uuv_ids=standby_ids,
        returning_uuv_ids=_by_status(snapshot, UUVStatus.RETURNING),
        failed_uuv_ids=_by_status(snapshot, UUVStatus.FAILED),
        predicted_quality=predicted_quality,
        predicted_fim=predicted_fim,
        predicted_active_count=active_count,
        predicted_energy=economic_cost,
        predicted_risk=predicted_risk,
        diff=_plan_diff(active, plan_id, revision, members_by_target, waypoints, proposal),
        evidence_ids=proposal.evidence_ids,
    )
    return CandidateEvaluation(
        plan=plan,
        metrics=CandidateMetrics(
            hard_violations=violations,
            active_count=active_count,
            economic_cost=economic_cost,
        ),
        index=index,
    )


def _emergency_evaluation(
    proposal: StrategyProposal,
    snapshot: PlanningSnapshot,
    config: PlanningConfig,
    index: int,
) -> CandidateEvaluation:
    """DEGRADED emergency plan retaining the highest-priority feasible targets.

    Targets are ordered by descending proposal priority (stable target-id
    tie-break); prefixes from the full set down to the single highest-priority
    target are tried, and the first feasible prefix becomes the degraded
    plan. When no prefix is feasible the best-effort single-target plan is
    returned and the independent commit validation decides (spec 18: hard
    constraints are never relaxed).
    """
    ordered = sorted(
        proposal.target_priorities,
        key=lambda target: (-proposal.target_priorities[target], target),
    )
    for size in range(len(ordered), 1, -1):
        prefix = ordered[:size]
        problem = _build_problem(snapshot, proposal, prefix, config)
        solution = allocate_groups(problem)
        if not solution.hard_violations:
            return _build_evaluation(
                proposal, snapshot, problem, solution, config, index, degraded=True
            )
    prefix = ordered[:1]
    problem = _build_problem(snapshot, proposal, prefix, config)
    solution = allocate_groups(problem)
    return _build_evaluation(
        proposal, snapshot, problem, solution, config, index, degraded=True
    )


def _previous_plan_infeasible(snapshot: PlanningSnapshot) -> bool:
    """Whether any member of the broadcast plan is no longer usable.

    A member that failed, is returning, or was disabled by an applied
    directive makes the broadcast plan infeasible (spec 18: UUV failure ->
    drop the member and re-evaluate; the previous plan cannot be held).
    """
    active = snapshot.active_plan
    if active is None:
        return False
    status_by_id = {uuv.uuv_id: uuv.status for uuv in snapshot.situation.uuvs}
    disabled = {
        uuv_id
        for directive in snapshot.applied_directives
        for uuv_id in directive.disabled_uuv_ids
    }
    return any(
        status_by_id.get(member) in (UUVStatus.FAILED, UUVStatus.RETURNING)
        or member in disabled
        for members in active.member_ids_by_target.values()
        for member in members
    )


def _plan_waypoints(
    snapshot: PlanningSnapshot,
    target: str,
    members: Sequence[str],
    config: PlanningConfig,
    previous_by_member: Mapping[str, tuple[Waypoint, ...]] | None,
) -> dict[str, tuple[Waypoint, ...]]:
    """Plan one robust short waypoint sequence per group member (spec 14.3)."""
    situation = snapshot.situation
    uuvs_by_id = {uuv.uuv_id: uuv for uuv in situation.uuvs}
    positions = np.asarray(
        [uuvs_by_id[member].position_xy for member in members], dtype=float
    )
    # The planner enforces one scalar step bound for the whole group; using
    # the slowest member keeps every first waypoint within ITS OWN
    # ``speed_mps * replan_period_s``, which is what the independent commit
    # validation re-checks per UUV. (``max`` would let the planner place a
    # slow UUV beyond its own kinematic bound and reject the plan every
    # cycle; ``min`` is conservative for faster members but always valid.)
    max_step = min(uuvs_by_id[member].speed_mps for member in members) * (
        config.replan_period_s
    )
    previous = None
    if previous_by_member is not None and all(
        member in previous_by_member for member in members
    ):
        previous = np.asarray(
            [
                (previous_by_member[member][0].x, previous_by_member[member][0].y)
                for member in members
            ],
            dtype=float,
        )
    result = plan_group_waypoints(
        positions,
        np.asarray(_belief_sigma_points(_belief(snapshot, target)), dtype=float),
        previous,
        max_step,
        config.min_separation_m,
        config.bearing_variance,
        config.beam_width,
        uuv_ids=tuple(members),
        min_range_m=config.min_range_m,
        max_range_m=config.max_range_m,
        range_bins=config.range_bins,
        horizon_steps=config.horizon_steps,
        bounds=config.bounds,
    )
    waypoints: dict[str, tuple[Waypoint, ...]] = {}
    step_s = int(config.replan_period_s)
    for rank, member in enumerate(members):
        rows = result.sequence_xy[rank]
        waypoints[member] = tuple(
            Waypoint(
                x=float(rows[step][0]),
                y=float(rows[step][1]),
                arrive_at_s=snapshot.sim_time_s + step * step_s,
            )
            for step in range(config.horizon_steps)
        )
    return waypoints


def _plan_diff(
    active: TrackingPlan | None,
    plan_id: str,
    revision: int,
    members_by_target: Mapping[str, tuple[str, ...]],
    waypoints: Mapping[str, tuple[Waypoint, ...]],
    proposal: StrategyProposal,
) -> PlanDiff | None:
    """Churn versus the broadcast plan: added/removed members and changed waypoints."""
    if active is None:
        return None
    added: dict[str, tuple[str, ...]] = {}
    removed: dict[str, tuple[str, ...]] = {}
    for target in sorted(members_by_target):
        new_members = set(members_by_target[target])
        old_members = set(active.member_ids_by_target.get(target, ()))
        added_target = tuple(sorted(new_members - old_members))
        removed_target = tuple(sorted(old_members - new_members))
        if added_target:
            added[target] = added_target
        if removed_target:
            removed[target] = removed_target
    changed: list[str] = []
    for member in sorted(
        {uuv for group in members_by_target.values() for uuv in group}
    ):
        new_waypoint = waypoints.get(member)
        old_waypoint = active.waypoints_by_member.get(member)
        if (
            new_waypoint
            and old_waypoint
            and _distance(
                (new_waypoint[0].x, new_waypoint[0].y),
                (old_waypoint[0].x, old_waypoint[0].y),
            )
            > 1e-6
        ):
            changed.append(member)
    return PlanDiff(
        from_plan_id=active.plan_id,
        from_revision=active.revision,
        to_plan_id=plan_id,
        to_revision=revision,
        members_added=added,
        members_removed=removed,
        waypoints_changed=tuple(sorted(changed)),
        summary=f"{proposal.concept} plan revision {revision}",
    )


def _belief_sigma_points(belief: TargetBelief) -> tuple[tuple[float, float], ...]:
    """2D unscented sigma set (kappa=0) from the position marginal (spec 14.3)."""
    mean_x, mean_y = belief.mean[0], belief.mean[1]
    chol11, chol21, chol22 = _chol2(belief.covariance)
    scale = math.sqrt(2.0)
    points: list[tuple[float, float]] = [(mean_x, mean_y)]
    for sign in (1.0, -1.0):
        points.append((mean_x + sign * scale * chol11, mean_y))
        points.append(
            (mean_x + sign * scale * chol21, mean_y + sign * scale * chol22)
        )
    return tuple(points)


def _chol2(
    covariance: tuple[tuple[float, ...], ...],
) -> tuple[float, float, float]:
    """Lower-triangular Cholesky entries ``L11, L21, L22`` of the 2x2 marginal."""
    a = covariance[0][0]
    b = covariance[0][1]
    c = covariance[1][1]
    l11 = math.sqrt(a) if a > 0.0 else 0.0
    l21 = b / l11 if l11 > 0.0 else 0.0
    remainder = c - l21 * l21
    l22 = math.sqrt(remainder) if remainder > 0.0 else 0.0
    return l11, l21, l22


def _locked_to_other(
    uuv_id: str,
    target: str,
    snapshot: PlanningSnapshot,
    targets: Sequence[str],
) -> bool:
    """Whether an applied directive locks ``uuv_id`` to another target in scope."""
    for directive in snapshot.applied_directives:
        for locked_target, locked_members in directive.locked_members.items():
            if (
                locked_target != target
                and locked_target in targets
                and uuv_id in locked_members
            ):
                return True
    return False


def _report(snapshot: PlanningSnapshot, target: str) -> GroupReport:
    for report in snapshot.situation.group_reports:
        if report.target_id == target:
            return report
    raise ValueError(f"no group report for target {target!r}")


def _belief(snapshot: PlanningSnapshot, target: str) -> TargetBelief:
    return _report(snapshot, target).belief


def _energy(snapshot: PlanningSnapshot, uuv_id: str) -> float:
    for uuv in snapshot.situation.uuvs:
        if uuv.uuv_id == uuv_id:
            return uuv.energy_fraction
    raise ValueError(f"unknown uuv {uuv_id!r}")


def _by_status(snapshot: PlanningSnapshot, status: UUVStatus) -> tuple[str, ...]:
    return tuple(
        sorted(uuv.uuv_id for uuv in snapshot.situation.uuvs if uuv.status == status)
    )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
