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
from underwater_tracking.config.models import (
    DEFAULT_QUALITY_RELEASE,
    DEFAULT_QUALITY_WARNING,
    DEFAULT_RELEASE_HOLD_S,
)
from underwater_tracking.domain.availability import is_deployable
from underwater_tracking.domain.agent_models import (
    PlanDiff,
    PredictedTrackRef,
    Segment,
    SegmentPlan,
    StrategyProposal,
    StrategySet,
    TrackingPlan,
    Waypoint,
    derive_legacy_views,
)
from underwater_tracking.domain.regional_models import (
    CommunicationRequirement,
    RegionTask,
    RegionalStrategySet,
    TargetRegionPlan,
    UUVRegionalStrategySet,
)
from underwater_tracking.domain.mission_models import (
    ExecutableMissionPlan,
    MissionCandidate,
    RegionLifecycle,
)
from underwater_tracking.planning.regional_allocation import materialize_regional_plan
from underwater_tracking.planning.mission_optimizer import MissionOptimizer
from underwater_tracking.domain.models import (
    DeploymentState,
    GroupReport,
    OperationalScheme,
    TargetBelief,
    UUVStatus,
    UUVState,
)
from underwater_tracking.planning.allocation import (
    AllocationInput,
    AllocationSolution,
    allocate_groups,
    projected_tracking_quality,
)
from underwater_tracking.planning.segmentation import (
    default_segment_plan,
    initial_intercept,
)
from underwater_tracking.planning.waypoints import plan_group_waypoints

# Default scenario bounds shared with the Foundation planner.
_DEFAULT_BOUNDS: tuple[float, float, float, float] = (
    -5000.0,
    5000.0,
    -5000.0,
    5000.0,
)

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
    quality_warning: float = DEFAULT_QUALITY_WARNING
    quality_release: float = DEFAULT_QUALITY_RELEASE
    release_hold_s: float = float(DEFAULT_RELEASE_HOLD_S)
    reassignment_penalty: float = 100.0
    rotation_threshold: float = 0.3
    plan_horizon_s: int = 600
    # A candidate only counts as materially better when its economic cost
    # drops by at least this fraction (spec 15.2: no material gain -> hold).
    improvement_margin: float = 0.01

    def __post_init__(self) -> None:
        float_values = (
            ("max_range_m", self.max_range_m),
            ("min_range_m", self.min_range_m),
            ("min_separation_m", self.min_separation_m),
            ("bearing_variance", self.bearing_variance),
            ("replan_period_s", self.replan_period_s),
            ("return_reserve", self.return_reserve),
            ("quality_warning", self.quality_warning),
            ("quality_release", self.quality_release),
            ("release_hold_s", self.release_hold_s),
            ("reassignment_penalty", self.reassignment_penalty),
            ("rotation_threshold", self.rotation_threshold),
            ("improvement_margin", self.improvement_margin),
            ("plan_horizon_s", self.plan_horizon_s),
        )
        for name, value in float_values:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if len(self.bounds) != 4 or not all(math.isfinite(value) for value in self.bounds):
            raise ValueError("bounds must contain four finite values")
        x_min, x_max, y_min, y_max = self.bounds
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("bounds must have increasing limits")
        if self.max_range_m <= 0.0:
            raise ValueError("max_range_m must be positive")
        if not 0.0 <= self.min_range_m <= self.max_range_m:
            raise ValueError("min_range_m must be in [0, max_range_m]")
        if self.min_separation_m < 0.0:
            raise ValueError("min_separation_m must be non-negative")
        if self.bearing_variance <= 0.0:
            raise ValueError("bearing_variance must be positive")
        if self.beam_width < 1 or self.range_bins < 1 or self.horizon_steps < 1:
            raise ValueError("planning discretization counts must be positive")
        if self.replan_period_s <= 0.0:
            raise ValueError("replan_period_s must be positive")
        if not 0.0 <= self.return_reserve <= 1.0:
            raise ValueError("return_reserve must be in [0, 1]")
        if not 0.0 <= self.quality_warning < self.quality_release <= 1.0:
            raise ValueError("need 0 <= quality_warning < quality_release <= 1")
        if self.release_hold_s < 0.0:
            raise ValueError("release_hold_s must be non-negative")
        if self.reassignment_penalty < 0.0:
            raise ValueError("reassignment_penalty must be non-negative")
        if not 0.0 <= self.rotation_threshold <= 1.0:
            raise ValueError("rotation_threshold must be in [0, 1]")
        if self.plan_horizon_s <= 0:
            raise ValueError("plan_horizon_s must be positive")
        if not 0.0 <= self.improvement_margin <= 1.0:
            raise ValueError("improvement_margin must be in [0, 1]")


@dataclass(frozen=True)
class CandidateMetrics:
    """Deterministic objective of one candidate plan."""

    hard_violations: tuple[str, ...]
    active_count: int
    economic_cost: float
    quality_deficit: float = 0.0
    priority_loss: float = 0.0


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
    predictions: Mapping[str, PredictedTrackRef] | None = None,
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
            evaluations.append(
                _emergency_evaluation(
                    proposal, snapshot, config, index, predictions=predictions
                )
            )
            continue
        problem = _build_problem(
            snapshot, proposal, tuple(sorted(proposal.target_priorities)), config
        )
        solution = allocate_groups(problem)
        if solution.hard_violations:
            evaluations.append(
                _emergency_evaluation(
                    proposal, snapshot, config, index, predictions=predictions
                )
            )
        else:
            evaluations.append(
                _build_evaluation(
                    proposal, snapshot, problem, solution, config, index,
                    predictions=predictions,
                )
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
        if state.get("regional_plans"):
            if _is_uuv_only_regional_state(snapshot, state):
                return self._optimize_uuv_only(snapshot, strategy_set, state)
            return self._optimize_regional(snapshot, strategy_set, state)
        evaluations = optimize_candidates(
            snapshot,
            strategy_set,
            self._config,
            predictions=state.get("predictions"),
        )
        candidate_refs: list[str] = []
        regional_plans, region_tasks = _materialize_regional_metadata(snapshot, state)
        for evaluation in evaluations:
            candidate_ref = self._ref(snapshot, evaluation.index)
            self._store[candidate_ref] = _attach_regional_metadata(
                evaluation.plan, regional_plans, region_tasks
            )
            candidate_refs.append(candidate_ref)
        selected_ref = self._ref(snapshot, len(evaluations))
        self._store[selected_ref] = _attach_regional_metadata(
            select_candidate(snapshot, evaluations, self._config),
            regional_plans,
            region_tasks,
        )
        selected = self._store[selected_ref]
        return {
            "candidate_plan_refs": tuple(candidate_refs),
            "selected_plan_ref": selected_ref,
            "region_tasks": dict(selected.region_tasks),
            "regional_metrics": selected.regional_metrics,
        }

    def _optimize_uuv_only(
        self,
        snapshot: PlanningSnapshot,
        strategy_set: StrategySet,
        state: CarrierState,
    ) -> CarrierState:
        """Run the rolling UUV optimizer and retain the legacy plan projection."""
        candidates_by_target = state.get("regional_candidates") or {}
        if not candidates_by_target:
            from underwater_tracking.agent.nodes.regions import (
                regional_plan_to_mission_candidates,
            )

            candidates_by_target = {
                target_id: regional_plan_to_mission_candidates(plan)
                for target_id, plan in sorted((state.get("regional_plans") or {}).items())
            }
        mission_candidates: list[MissionCandidate] = []
        locked: dict[str, tuple[str, ...]] = {}
        plans = state.get("regional_plans") or {}
        policies = state.get("regional_policies") or {}
        for target_id, candidates in sorted(candidates_by_target.items()):
            plan = plans.get(target_id)
            cells = {cell.region_id: cell for cell in plan.cells} if plan else {}
            policy_set = policies.get(target_id)
            policy_by_id = (
                {policy.candidate_id: policy for policy in policy_set.policies}
                if isinstance(policy_set, UUVRegionalStrategySet)
                else {}
            )
            for candidate in candidates:
                if isinstance(candidate, MissionCandidate):
                    normalized = candidate
                else:
                    cell = cells.get(candidate.candidate_id)
                    policy = policy_by_id.get(candidate.candidate_id)
                    normalized = MissionCandidate(
                        candidate_id=candidate.candidate_id,
                        target_id=target_id,
                        entry_s=candidate.time_window.start_s,
                        exit_s=candidate.time_window.end_s,
                        probability=(
                            max(0.01, cell.occupancy_likelihood)
                            if cell is not None
                            else 0.5
                        ),
                        perimeter_points=candidate.perimeter_points,
                        active_scan_uuv_count=int(
                            getattr(policy, "active_scan_uuv_count", 1)
                        ),
                        passive_track_uuv_count=int(
                            getattr(policy, "passive_track_uuv_count", 1)
                        ),
                        reserve_uuv_count=int(
                            getattr(policy, "reserve_uuv_count", 0)
                        ),
                        optional_uuv_count=int(
                            getattr(policy, "optional_uuv_count", 0)
                        ),
                        predecessor_candidate_ids=candidate.predecessor_candidate_ids,
                        successor_candidate_ids=candidate.successor_candidate_ids,
                    )
                mission_candidates.append(normalized)
                policy = policy_by_id.get(candidate.candidate_id)
                if policy is not None and policy.assigned_uuv_ids:
                    locked[candidate.candidate_id] = tuple(policy.assigned_uuv_ids)

        executable = MissionOptimizer().optimize(
            snapshot,
            tuple(mission_candidates),
            locked_uuv_ids_by_candidate=locked,
        )
        regional_plans, region_tasks = _materialize_uuv_only_metadata(
            state,
            executable,
        )
        candidate = _regional_candidate(
            snapshot,
            strategy_set.proposals[0],
            regional_plans,
            region_tasks,
            _regional_llm_hashes(state, regional_plans),
            strategy_set.trigger_event_ids,
            self._config,
        )
        candidate_ref = self._ref(snapshot, 0)
        selected_ref = self._ref(snapshot, 1)
        self._store[candidate_ref] = candidate
        self._store[selected_ref] = candidate
        return {
            "candidate_plan_refs": (candidate_ref,),
            "selected_plan_ref": selected_ref,
            "region_tasks": dict(candidate.region_tasks),
            "regional_metrics": candidate.regional_metrics,
            "executable_mission_plan": executable,
        }

    def _optimize_regional(
        self,
        snapshot: PlanningSnapshot,
        strategy_set: StrategySet,
        state: CarrierState,
    ) -> CarrierState:
        """Build a candidate directly from approved regional assignments.

        Regional policies carry explicit platform membership and topology.
        Running their compatibility proposal through ``allocate_groups`` first
        would silently reinstate the legacy 2--4 UUV group contract, so this
        path materializes and validates those explicit assignments directly.
        """
        regional_plans, region_tasks = _materialize_regional_metadata(snapshot, state)
        candidate = _regional_candidate(
            snapshot,
            strategy_set.proposals[0],
            regional_plans,
            region_tasks,
            _regional_llm_hashes(state, regional_plans),
            strategy_set.trigger_event_ids,
            self._config,
        )
        candidate_ref = self._ref(snapshot, 0)
        selected_ref = self._ref(snapshot, 1)
        self._store[candidate_ref] = candidate
        self._store[selected_ref] = candidate
        return {
            "candidate_plan_refs": (candidate_ref,),
            "selected_plan_ref": selected_ref,
            "region_tasks": dict(candidate.region_tasks),
            "regional_metrics": candidate.regional_metrics,
        }

    @staticmethod
    def _ref(snapshot: PlanningSnapshot, index: int) -> str:
        return f"{snapshot.scenario_id}:candidate:{index}:{snapshot.snapshot_revision}"


def _materialize_regional_metadata(
    snapshot: PlanningSnapshot,
    state: CarrierState,
) -> tuple[dict[str, TargetRegionPlan], dict[str, RegionTask]]:
    regional_plans = state.get("regional_plans", {})
    policies = state.get("regional_policies", {})
    platform_snapshot = getattr(snapshot.situation, "platform_snapshot", None)
    if not regional_plans:
        return {}, {}
    materialized: dict[str, TargetRegionPlan] = {}
    tasks: dict[str, RegionTask] = {}
    for target_id, target_plan in sorted(regional_plans.items()):
        strategy = policies.get(target_id)
        if strategy is None:
            updated = _uncovered_regional_plan(target_plan, "regional_policy_missing")
        elif not isinstance(strategy, RegionalStrategySet):
            updated = _uncovered_regional_plan(target_plan, "regional_policy_invalid")
        elif platform_snapshot is None:
            updated = _uncovered_regional_plan(target_plan, "platform_snapshot_missing")
        else:
            try:
                allocation = materialize_regional_plan(
                    target_plan,
                    strategy,
                    platform_snapshot.roster,
                    carrier=platform_snapshot.carrier,
                )
            except ValueError:
                updated = _uncovered_regional_plan(target_plan, "regional_policy_invalid")
            else:
                updated = target_plan.model_copy(
                    update={"tasks": tuple(allocation.tasks.values())}
                )
        materialized[target_id] = updated
        tasks.update({task.region_id: task for task in updated.tasks})
    return materialized, tasks


def _is_uuv_only_regional_state(
    snapshot: PlanningSnapshot,
    state: CarrierState,
) -> bool:
    if state.get("uuv_only"):
        return True
    platform_snapshot = getattr(snapshot.situation, "platform_snapshot", None)
    return bool(
        state.get("regional_candidates")
        and platform_snapshot is not None
        and not platform_snapshot.roster.usvs
    )


def _materialize_uuv_only_metadata(
    state: CarrierState,
    executable: ExecutableMissionPlan,
) -> tuple[dict[str, TargetRegionPlan], dict[str, RegionTask]]:
    """Project executable UUV assignments to the legacy task view."""
    assignments = executable.assignments_by_candidate
    materialized: dict[str, TargetRegionPlan] = {}
    tasks: dict[str, RegionTask] = {}
    for target_id, plan in sorted((state.get("regional_plans") or {}).items()):
        updated_tasks: list[RegionTask] = []
        for base_task in plan.tasks:
            assignment = assignments.get(base_task.region_id)
            if assignment is None:
                updated = base_task.model_copy(
                    update={
                        "tracking_mode": "heuristic_uuv",
                        "required_usv_count": 0,
                        "usv_role": None,
                        "assigned_uuv_ids": (),
                        "assigned_usv_ids": (),
                        "assignment_status": "uncovered",
                        "communication": CommunicationRequirement(
                            usv_relay_required=False
                        ),
                        "degraded_reasons": ("candidate_assignment_missing",),
                    }
                )
            else:
                active_ids = tuple(assignment.active_scan_uuv_ids)
                passive_ids = tuple(assignment.passive_track_uuv_ids)
                assigned_ids = (*active_ids, *passive_ids)
                status = {
                    RegionLifecycle.UNCOVERED: "uncovered",
                    RegionLifecycle.DEGRADED: "degraded",
                }.get(assignment.lifecycle, "planned")
                updated = base_task.model_copy(
                    update={
                        "tracking_mode": "heuristic_uuv",
                        "required_uuv_count": len(assigned_ids),
                        "required_usv_count": 0,
                        "uuv_roles": (
                            ("active_verifier",) * len(active_ids)
                            + ("passive_tracker",) * len(passive_ids)
                        ),
                        "usv_role": None,
                        "assigned_uuv_ids": assigned_ids,
                        "assigned_usv_ids": (),
                        "assignment_status": status,
                        "communication": CommunicationRequirement(
                            usv_relay_required=False
                        ),
                        "degraded_reasons": assignment.degraded_reasons,
                        "plan_revision": executable.revision,
                    }
                )
            updated_tasks.append(updated)
            tasks[updated.region_id] = updated
        materialized[target_id] = plan.model_copy(update={"tasks": tuple(updated_tasks)})
    return materialized, tasks


def _regional_candidate(
    snapshot: PlanningSnapshot,
    proposal: StrategyProposal,
    regional_plans: Mapping[str, TargetRegionPlan],
    region_tasks: Mapping[str, RegionTask],
    regional_llm_hashes: Mapping[str, tuple[str, str]],
    trigger_event_ids: tuple[str, ...],
    config: PlanningConfig,
) -> TrackingPlan:
    """Project materialized regional tasks into a deterministic plan.

    The legacy fields are compatibility views only.  Every member, tracking
    mode, relay, waypoint, and degraded/uncovered state originates from the
    materialized regional task set.
    """
    active = snapshot.active_plan
    revision = active.revision + 1 if active is not None else 1
    plan_id = f"{snapshot.scenario_id}:plan:{revision}"
    target_tasks: dict[str, list[RegionTask]] = {}
    for task in region_tasks.values():
        target_tasks.setdefault(task.target_id, []).append(task)
    target_priorities = {
        target_id: max(task.priority for task in tasks)
        for target_id, tasks in sorted(target_tasks.items())
    }
    required_quality = {
        target_id: max(task.required_quality for task in tasks)
        for target_id, tasks in sorted(target_tasks.items())
    }
    degraded = tuple(
        task
        for task in region_tasks.values()
        if task.assignment_status in {"degraded", "uncovered"}
    )
    legacy_views = derive_legacy_views(dict(regional_plans), dict(region_tasks))
    members_by_target = legacy_views["member_ids_by_target"]
    waypoints_by_member = legacy_views["waypoints_by_member"]
    assert isinstance(members_by_target, dict)
    assert isinstance(waypoints_by_member, dict)
    # ``degraded_regions`` is retained as a top-level legacy-view helper for
    # callers that consume ``derive_legacy_views`` directly, but the strict
    # TrackingPlan schema carries it inside ``regional_metrics``.
    plan_legacy_views = {
        key: value for key, value in legacy_views.items() if key != "degraded_regions"
    }
    return TrackingPlan(
        plan_id=plan_id,
        scenario_id=snapshot.scenario_id,
        revision=revision,
        base_snapshot_revision=snapshot.snapshot_revision,
        status="degraded" if degraded else "draft",
        valid_from_s=snapshot.sim_time_s,
        valid_until_s=snapshot.sim_time_s + config.plan_horizon_s,
        concept=proposal.concept,
        target_priorities=target_priorities,
        required_quality=required_quality,
        predicted_quality=required_quality,
        predicted_fim={
            target_id: _belief(snapshot, target_id).fim_min_eigenvalue
            for target_id in target_tasks
            if any(
                report.target_id == target_id
                for report in snapshot.situation.group_reports
            )
        },
        predicted_active_count=len(legacy_views["active_uuv_ids"]),
        predicted_risk=(len(degraded) / len(region_tasks)) if region_tasks else 1.0,
        diff=_plan_diff(
            active,
            plan_id,
            revision,
            members_by_target,
            waypoints_by_member,
            proposal,
        ),
        evidence_ids=proposal.evidence_ids,
        trigger_event_ids=trigger_event_ids,
        regional_plans=dict(regional_plans),
        regional_llm_hashes=dict(regional_llm_hashes),
        region_tasks=dict(region_tasks),
        **plan_legacy_views,
    )


def _uncovered_regional_plan(
    target_plan: TargetRegionPlan,
    reason: str,
) -> TargetRegionPlan:
    """Preserve every region when regional planning cannot be materialized."""
    tasks = tuple(
        task.model_copy(
            update={
                "assigned_uuv_ids": (),
                "assigned_usv_ids": (),
                "assignment_status": "uncovered",
                "communication_links": (),
                "degraded_reasons": (reason,),
            }
        )
        for task in target_plan.tasks
    )
    return target_plan.model_copy(update={"tasks": tasks})


def _attach_regional_metadata(
    plan: TrackingPlan,
    regional_plans: Mapping[str, TargetRegionPlan],
    region_tasks: Mapping[str, RegionTask],
    regional_llm_hashes: Mapping[str, tuple[str, str]] | None = None,
) -> TrackingPlan:
    if not regional_plans:
        return plan
    legacy_views = derive_legacy_views(dict(regional_plans), dict(region_tasks))
    return plan.model_copy(
        update={
            "regional_plans": dict(regional_plans),
            "regional_llm_hashes": dict(regional_llm_hashes or {}),
            "region_tasks": dict(region_tasks),
            **legacy_views,
        }
    )


def _regional_llm_hashes(
    state: CarrierState,
    regional_plans: Mapping[str, TargetRegionPlan],
) -> dict[str, tuple[str, str]]:
    """Project target-scoped regional strategy provenance into a plan payload."""
    provenance = state.get("llm_provenance", {})
    hashes: dict[str, tuple[str, str]] = {}
    for target_id in regional_plans:
        metadata = provenance.get(f"regional_strategy:{target_id}")
        if metadata is not None:
            hashes[target_id] = (metadata.request_hash, metadata.response_hash)
    return hashes


def _sort_key(evaluation: CandidateEvaluation) -> tuple[int, float, float, int, float, int]:
    """Rank hard constraints, quality, priority, cost, then stable proposal order."""
    return (
        len(evaluation.metrics.hard_violations),
        evaluation.metrics.quality_deficit,
        evaluation.metrics.priority_loss,
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
    uuvs_by_id = {uuv.uuv_id: uuv for uuv in situation.uuvs}
    energy_by_id = {uuv.uuv_id: uuv.energy_fraction for uuv in situation.uuvs}
    speed_by_id = {
        uuv.uuv_id: _effective_speed_mps(uuv)
        for uuv in situation.uuvs
    }
    position_by_id = {uuv.uuv_id: uuv.position_xy for uuv in situation.uuvs}
    disabled = {
        uuv_id
        for directive in snapshot.applied_directives
        for uuv_id in (*directive.disabled_uuv_ids, *directive.return_uuv_ids)
    }
    reserved = {
        uuv_id
        for directive in snapshot.applied_directives
        if directive.directive_type == "assignment"
        for uuv_id in directive.assignment_uuv_ids
    }
    # In the explicit platform-core scenario a committed ``track`` command
    # is also the carrier launch order.  Onboard UUVs are therefore eligible
    # dispatch resources; after launch, active-plan members remain transit
    # eligible until their planned standoff is reached.  Failed/returning
    # resources never enter this set.
    platform_core_dispatch = situation.platform_snapshot is not None
    transit_ids = {
        uuv_id
        for uuv_id, uuv in uuvs_by_id.items()
        if platform_core_dispatch and uuv.deployment_state is DeploymentState.ONBOARD
    }
    active = snapshot.active_plan
    if platform_core_dispatch and active is not None:
        transit_ids.update(
            member
            for members in active.member_ids_by_target.values()
            for member in members
        )
    unavailable = {
        uuv_id
        for uuv_id in uuvs
        if (
            (
                not is_deployable(uuvs_by_id[uuv_id])
                and uuv_id not in transit_ids
            )
            or uuvs_by_id[uuv_id].status in {UUVStatus.FAILED, UUVStatus.RETURNING}
        )
        or uuv_id in disabled
        or uuv_id in reserved
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
        and _capability_feasible(
            uuvs_by_id[uuv_id],
            target_mean[target],
            config,
            allow_transit=uuv_id in transit_ids,
        )
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
    required_quality = {
        target: _effective_required_quality(snapshot, proposal, target) for target in targets
    }
    target_priorities = {
        target: _effective_target_priority(snapshot, proposal, target) for target in targets
    }
    return AllocationInput(
        uuv_ids=uuvs,
        target_ids=tuple(targets),
        quality_by_target={target: _report(snapshot, target).quality.ewma for target in targets},
        uuv_available=uuv_available,
        reserved_uuv_ids=frozenset(reserved),
        uuv_transit_ids=frozenset(transit_ids),
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
        required_quality_by_target=required_quality,
        target_priority_by_target=target_priorities,
        uuv_passive_range_m={
            uuv_id: uuvs_by_id[uuv_id].capability.passive_range_m for uuv_id in uuvs
        },
        uuv_bearing_variance_rad2={
            uuv_id: uuvs_by_id[uuv_id].capability.bearing_variance_rad2 for uuv_id in uuvs
        },
        uuv_speed_mps=speed_by_id,
        uuv_max_turn_rate_rad_s={
            uuv_id: uuvs_by_id[uuv_id].capability.max_turn_rate_rad_s for uuv_id in uuvs
        },
        uuv_passive_sonar_available={
            uuv_id: uuvs_by_id[uuv_id].capability.passive_sonar_available for uuv_id in uuvs
        },
        uuv_endurance_s={
            uuv_id: uuvs_by_id[uuv_id].capability.endurance_s for uuv_id in uuvs
        },
        uuv_availability={
            uuv_id: uuvs_by_id[uuv_id].capability.availability for uuv_id in uuvs
        },
        plan_horizon_s=float(config.plan_horizon_s),
        rotation_threshold=config.rotation_threshold,
    )


def _capability_feasible(
    uuv: UUVState,
    target_xy: tuple[float, float],
    config: PlanningConfig,
    *,
    allow_transit: bool = False,
) -> bool:
    """Check passive sensing range and bounded maneuver time for one pair."""
    if (
        not uuv.capability.passive_sonar_available
        or uuv.capability.availability <= 0.0
        or uuv.capability.endurance_s < config.plan_horizon_s
    ):
        return False
    # A transit member is being dispatched toward the predicted tracking
    # sector, not claimed as an already-on-station passive fix.  Its current
    # standoff may therefore exceed both the sensing radius and one planning
    # window; the waypoint/kinematic validator governs the actual movement.
    if allow_transit:
        return True
    distance = _distance(uuv.position_xy, target_xy)
    if not allow_transit and distance > min(config.max_range_m, uuv.capability.passive_range_m):
        return False
    speed = _effective_speed_mps(uuv)
    if speed <= 0.0:
        return distance == 0.0
    bearing = math.atan2(target_xy[1] - uuv.position_xy[1], target_xy[0] - uuv.position_xy[0])
    heading_change = abs((bearing - uuv.heading_rad + math.pi) % (2.0 * math.pi) - math.pi)
    maneuver_s = heading_change / uuv.capability.max_turn_rate_rad_s
    return distance / speed + maneuver_s <= config.plan_horizon_s


def _effective_speed_mps(uuv: UUVState) -> float:
    """Actual speed available to planning, capped by the platform capability."""
    return min(uuv.speed_mps, uuv.capability.max_speed_mps)


def _active_scheme(snapshot: PlanningSnapshot) -> OperationalScheme | None:
    scheme = snapshot.situation.operational_scheme
    if scheme is None or not (
        scheme.valid_from_s <= snapshot.sim_time_s < scheme.valid_until_s
    ):
        return None
    return scheme


def _effective_required_quality(
    snapshot: PlanningSnapshot,
    proposal: StrategyProposal,
    target: str,
) -> float:
    floor = proposal.required_quality.get(target, 0.0)
    scheme = _active_scheme(snapshot)
    if scheme is not None:
        floor = max(floor, scheme.minimum_quality.get(target, 0.0))
    directive_floor = max(
        (
            directive.minimum_quality.get(target, 0.0)
            for directive in snapshot.applied_directives
        ),
        default=0.0,
    )
    return max(floor, directive_floor)


def _effective_target_priority(
    snapshot: PlanningSnapshot,
    proposal: StrategyProposal,
    target: str,
) -> float:
    priority = proposal.target_priorities.get(target, 0.0)
    scheme = _active_scheme(snapshot)
    if scheme is not None:
        priority = max(priority, scheme.target_priorities.get(target, 0.0))
    directive_priority = max(
        (
            directive.target_priorities.get(target, 0.0)
            for directive in snapshot.applied_directives
        ),
        default=0.0,
    )
    return max(priority, directive_priority)


def _usable_segment_plan(
    proposal_plan: SegmentPlan | None,
    snapshot: PlanningSnapshot,
    members_by_target: Mapping[str, Sequence[str]],
    predictions: Mapping[str, PredictedTrackRef] | None,
    bounds: tuple[float, float, float, float] = _DEFAULT_BOUNDS,
) -> SegmentPlan | None:
    """The proposal's segment plan while it still covers the planning window.

    A checkpointed strategy set can carry a segment plan with absolute
    times from an earlier cycle (spec 8.2 continuation); once the head of
    the plan has passed the current simulation time, commit would reject
    every segment (``segment_past``), so the plan is re-based onto the
    current predictions: one deterministic uniform split per covered
    target, indices re-numbered contiguously from 0. Re-built intercepts
    are points on the predicted track, which a long-horizon extrapolation
    can place outside the scenario box; they are clamped into ``bounds``
    so the deterministic re-base can never fail commit's
    ``segment_out_of_bounds`` check on its own output.
    """
    if (
        proposal_plan is not None
        and proposal_plan.segments
        and all(
            segment.start_s >= snapshot.sim_time_s
            for segment in proposal_plan.segments
        )
    ):
        return proposal_plan
    if predictions is None:
        return None
    xmin, xmax, ymin, ymax = bounds
    rebuilt: list[Segment] = []
    for target in sorted(members_by_target):
        if not members_by_target[target]:
            continue
        prediction = predictions.get(target)
        if prediction is None:
            continue
        for segment in default_segment_plan(prediction, (f"G-{target}",)).segments:
            x, y = segment.intercept_xy
            rebuilt.append(
                Segment(
                    index=segment.index,
                    start_s=segment.start_s,
                    end_s=segment.end_s,
                    group_id=segment.group_id,
                    intercept_xy=(min(max(x, xmin), xmax), min(max(y, ymin), ymax)),
                )
            )
    if not rebuilt:
        return None
    return SegmentPlan(
        segments=tuple(
            Segment(
                index=index,
                start_s=segment.start_s,
                end_s=segment.end_s,
                group_id=segment.group_id,
                intercept_xy=segment.intercept_xy,
            )
            for index, segment in enumerate(rebuilt)
        )
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
    predictions: Mapping[str, PredictedTrackRef] | None = None,
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
    segment_plan = _usable_segment_plan(
        proposal.segment_plan,
        snapshot,
        members_by_target,
        predictions,
        config.bounds,
    )
    for target in sorted(members_by_target):
        members = members_by_target[target]
        if not members:
            continue
        waypoints.update(
            _plan_waypoints(
                snapshot,
                target,
                members,
                config,
                previous_by_member,
                intercept=initial_intercept(segment_plan, target),
            )
        )

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
    rotation_uuv_ids = tuple(
        sorted(
            member
            for member in active_members
            if _energy(snapshot, member) < config.rotation_threshold
        )
    )
    rotation_conditions = {
        target: f"energy_reserve_{config.return_reserve}"
        for target, members in members_by_target.items()
        if any(member in rotation_uuv_ids for member in members)
    }
    predicted_quality = {
        target: projected_tracking_quality(problem, target, members_by_target[target])
        for target in problem.target_ids
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
        target_priorities=dict(problem.target_priority_by_target),
        required_quality=dict(problem.required_quality_by_target),
        member_ids_by_target=members_by_target,
        roles_by_member=roles,
        waypoints_by_member=waypoints,
        rotation_conditions=rotation_conditions,
        rotation_uuv_ids=rotation_uuv_ids,
        release_actions={
            # The verify subgraph enforces full target coverage per policy
            # dict, but a model dict may still omit a target after a bounded
            # repair; defaults keep the deterministic optimizer from
            # crashing on partial output (missing policy = no requirement /
            # hold posture).
            target: proposal.reinforcement_policy.get(target, "hold")
            for target in problem.target_ids
        },
        active_uuv_ids=active_members,
        standby_uuv_ids=standby_ids,
        returning_uuv_ids=_requested_return_ids(snapshot),
        return_actions={
            uuv_id: "return" for uuv_id in _requested_return_ids(snapshot)
        },
        failed_uuv_ids=_by_deployment_state(snapshot, DeploymentState.FAILED),
        predicted_quality=predicted_quality,
        predicted_fim=predicted_fim,
        predicted_active_count=active_count,
        predicted_energy=economic_cost,
        predicted_risk=predicted_risk,
        diff=_plan_diff(active, plan_id, revision, members_by_target, waypoints, proposal),
        evidence_ids=proposal.evidence_ids,
        segment_plan=segment_plan,
    )
    quality_deficit = sum(
        max(0.0, problem.required_quality_by_target.get(target, 0.0) - predicted_quality[target])
        for target in problem.target_ids
    )
    priority_loss = sum(
        problem.target_priority_by_target.get(target, 0.0)
        * max(0.0, problem.required_quality_by_target.get(target, 0.0) - predicted_quality[target])
        for target in problem.target_ids
    )
    return CandidateEvaluation(
        plan=plan,
        metrics=CandidateMetrics(
            hard_violations=violations,
            quality_deficit=quality_deficit,
            priority_loss=priority_loss,
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
    *,
    predictions: Mapping[str, PredictedTrackRef] | None = None,
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
                proposal, snapshot, problem, solution, config, index, degraded=True,
                predictions=predictions,
            )
    prefix = ordered[:1]
    problem = _build_problem(snapshot, proposal, prefix, config)
    solution = allocate_groups(problem)
    return _build_evaluation(
        proposal, snapshot, problem, solution, config, index, degraded=True,
        predictions=predictions,
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
    uuvs_by_id = {uuv.uuv_id: uuv for uuv in snapshot.situation.uuvs}
    disabled = {
        uuv_id
        for directive in snapshot.applied_directives
        for uuv_id in (*directive.disabled_uuv_ids, *directive.return_uuv_ids)
    }
    reserved = {
        uuv_id
        for directive in snapshot.applied_directives
        if directive.directive_type == "assignment"
        for uuv_id in directive.assignment_uuv_ids
    }
    return any(
        member not in uuvs_by_id
        or not is_deployable(uuvs_by_id[member])
        or member in disabled
        or member in reserved
        for members in active.member_ids_by_target.values()
        for member in members
    )


def _plan_waypoints(
    snapshot: PlanningSnapshot,
    target: str,
    members: Sequence[str],
    config: PlanningConfig,
    previous_by_member: Mapping[str, tuple[Waypoint, ...]] | None,
    intercept: tuple[float, float] | None = None,
) -> dict[str, tuple[Waypoint, ...]]:
    """Plan one robust short waypoint sequence per group member (spec 14.3).

    When a relay segment plan assigns ``intercept`` to this group, the
    sigma-point lattice is recentered there, so the standoff converges on
    the predicted intercept instead of the current belief mean (R3).
    """
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
    max_step = min(_effective_speed_mps(uuvs_by_id[member]) for member in members) * (
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
    if _is_initial_explicit_dispatch(snapshot, members):
        return _plan_dispatch_waypoints(
            snapshot,
            target,
            members,
            config,
            max_step,
            step_s=int(config.replan_period_s),
        )
    sigma_points = np.asarray(
        _belief_sigma_points(_belief(snapshot, target)), dtype=float
    )
    if intercept is not None:
        sigma_points = sigma_points + (
            np.asarray(intercept) - sigma_points.mean(axis=0)
        )
    result = plan_group_waypoints(
        positions,
        sigma_points,
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


def _is_initial_explicit_dispatch(
    snapshot: PlanningSnapshot, members: Sequence[str]
) -> bool:
    """Whether a group is still co-located onboard and needs to fan out."""
    if snapshot.situation.platform_snapshot is None:
        return False
    onboard = {
        uuv.uuv_id
        for uuv in snapshot.situation.uuvs
        if uuv.deployment_state is DeploymentState.ONBOARD
    }
    return bool(members) and set(members) <= onboard


def _plan_dispatch_waypoints(
    snapshot: PlanningSnapshot,
    target: str,
    members: Sequence[str],
    config: PlanningConfig,
    max_step: float,
    *,
    step_s: int,
) -> dict[str, tuple[Waypoint, ...]]:
    """Create a bounded fan-out path for a newly launched onboard group.

    The first few observations are a launch transient: all vehicles start at
    the carrier and cannot satisfy the final standoff separation in one
    physics window.  This path preserves each vehicle's speed bound while
    increasing lateral separation toward the 300 m formation requirement.
    """
    origin_by_member = {
        member: next(
            uuv.position_xy
            for uuv in snapshot.situation.uuvs
            if uuv.uuv_id == member
        )
        for member in members
    }
    mean = _belief(snapshot, target).mean
    target_xy = (float(mean[0]), float(mean[1]))
    origin = origin_by_member[members[0]]
    dx = target_xy[0] - origin[0]
    dy = target_xy[1] - origin[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        forward = (1.0, 0.0)
    else:
        forward = (dx / distance, dy / distance)
    lateral = (-forward[1], forward[0])
    center = (len(members) - 1) / 2.0
    lateral_bases = (40.0, 100.0, 150.0)
    waypoints: dict[str, tuple[Waypoint, ...]] = {}
    xmin, xmax, ymin, ymax = config.bounds
    for rank, member in enumerate(sorted(members)):
        offset_scale = rank - center
        previous_lateral = 0.0
        forward_distance = 0.0
        rows: list[Waypoint] = []
        for step_index in range(config.horizon_steps):
            lateral_offset = lateral_bases[min(step_index, len(lateral_bases) - 1)] * offset_scale
            delta_lateral = lateral_offset - previous_lateral
            forward_increment = math.sqrt(
                max(0.0, max_step * max_step - delta_lateral * delta_lateral)
            )
            forward_distance = min(distance, forward_distance + forward_increment)
            x = origin[0] + forward[0] * forward_distance + lateral[0] * lateral_offset
            y = origin[1] + forward[1] * forward_distance + lateral[1] * lateral_offset
            rows.append(
                Waypoint(
                    x=min(xmax, max(xmin, x)),
                    y=min(ymax, max(ymin, y)),
                    arrive_at_s=snapshot.sim_time_s + (step_index + 1) * step_s,
                )
            )
            previous_lateral = lateral_offset
        waypoints[member] = tuple(rows)
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


def _by_deployment_state(
    snapshot: PlanningSnapshot, deployment_state: DeploymentState
) -> tuple[str, ...]:
    return tuple(
        sorted(
            uuv.uuv_id
            for uuv in snapshot.situation.uuvs
            if uuv.deployment_state is deployment_state
        )
    )


def _requested_return_ids(snapshot: PlanningSnapshot) -> tuple[str, ...]:
    """Return deployed UUVs requested by an applied directive plus active returns."""
    requested = {
        uuv_id
        for directive in snapshot.applied_directives
        for uuv_id in directive.return_uuv_ids
    }
    eligible = {
        uuv.uuv_id
        for uuv in snapshot.situation.uuvs
        if uuv.deployment_state in {DeploymentState.DEPLOYED, DeploymentState.RETURNING}
    }
    return tuple(sorted(requested & eligible | set(_by_deployment_state(snapshot, DeploymentState.RETURNING))))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
