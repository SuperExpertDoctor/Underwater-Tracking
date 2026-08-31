# src/underwater_tracking/agent/counterfactual.py
"""Isolated counterfactual dry-runs over a planning snapshot (spec 10.2, plan Task 11).

An expert question may carry a counterfactual — "what would the plan look
like if this constraint were different?" — which is answered without
touching the online plan: the immutable planning snapshot is deep-cloned,
only the allowed override keys are applied, and the deterministic optimizer
(``optimize_candidates`` + ``select_candidate``) is re-run over the clone.
The resulting plan carries an isolated run id (``dry-run:<uuid>``) and a
plan id under that namespace; it is never persisted anywhere.

Isolation is by construction: this module never constructs a repository —
no ``PlanRepository``, no event publisher — so the online
``PlanRepository.commit`` method and the event publisher are unreachable
from the dry-run path (pre-flight ruling #7). The uuid in ``dry-run:<uuid>``
is the only nondeterministic bit in the whole question branch; tests may
pin an explicit ``run_id`` for exact determinism.

Allowed override keys (whitelist, deterministic validation):

- ``"{target}.min_quality"`` — minimum required quality in [0, 1]. A value
  above the target's measured EWMA marks the group as a quality risk
  (deterministic margin below the warning threshold), which the elastic
  group policy answers with reinforcement — e.g. "增派 UUV" for T2.
- ``"{target}.priority"`` — target priority weight in [0, 1].
- ``"{uuv}.disabled"`` — ``True`` only: the UUV is treated as unavailable
  (carried via an override directive so the problem builder honors it).

Any other key, unknown entity, or invalid value raises ``ValueError``.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from underwater_tracking.agent.nodes.optimize import (
    PlanningConfig,
    optimize_candidates,
    select_candidate,
)
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    PlanDiff,
    StrategyProposal,
    StrategySet,
    TrackingPlan,
)
from underwater_tracking.domain.models import SituationSnapshot

# Shared immutable default for callers (B008: no call in defaults).
_DEFAULT_CONFIG = PlanningConfig()

_ALLOWED_SUFFIXES = frozenset({"min_quality", "priority", "disabled"})

_DEFAULT_REQUIRED_QUALITY = 0.7


@dataclass(frozen=True)
class CounterfactualOverrides:
    """Validated override sets for one dry-run (all keys optional)."""

    target_priorities: dict[str, float]
    minimum_quality: dict[str, float]
    disabled_uuv_ids: tuple[str, ...]


def parse_counterfactual_overrides(
    overrides: Mapping[str, object], situation: SituationSnapshot
) -> CounterfactualOverrides:
    """Validate the override mapping against the situation (deterministic).

    Keys are ``{entity}.{suffix}`` with the suffix whitelisted above; the
    entity must be a tracked target (for ``min_quality``/``priority``) or a
    known UUV (for ``disabled``); numeric values must be finite and in
    [0, 1]. Raises ``ValueError`` with a deterministic message otherwise.
    """
    known_targets = frozenset(report.target_id for report in situation.group_reports)
    known_uuvs = frozenset(uuv.uuv_id for uuv in situation.uuvs)
    priorities: dict[str, float] = {}
    minimum_quality: dict[str, float] = {}
    disabled: list[str] = []
    for key, value in sorted(overrides.items()):
        entity, separator, suffix = key.rpartition(".")
        if not separator or suffix not in _ALLOWED_SUFFIXES:
            raise ValueError(f"unknown counterfactual override key {key!r}")
        if suffix == "disabled":
            if entity not in known_uuvs:
                raise ValueError(
                    f"unknown counterfactual uuv {entity!r}: no resource state for it"
                )
            if value is not True:
                raise ValueError(
                    f"invalid counterfactual value {key!r}: disabled expects True"
                )
            disabled.append(entity)
            continue
        if entity not in known_targets:
            raise ValueError(
                f"unknown counterfactual target {entity!r}: no group report for it"
            )
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(
                f"invalid counterfactual value {key!r}: {value!r} must be a finite number"
            )
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ValueError(
                f"invalid counterfactual value {key!r}: {value!r} outside [0, 1]"
            )
        if suffix == "min_quality":
            minimum_quality[entity] = number
        else:
            priorities[entity] = number
    return CounterfactualOverrides(
        target_priorities=priorities,
        minimum_quality=minimum_quality,
        disabled_uuv_ids=tuple(sorted(disabled)),
    )


@dataclass(frozen=True)
class ObjectiveChange:
    """Lexicographic objective delta of the dry-run plan versus the online plan."""

    active_count_before: int
    active_count_after: int
    economic_cost_before: float
    economic_cost_after: float


@dataclass(frozen=True)
class CounterfactualResult:
    """One dry-run: isolated plan, its churn, and the objective delta."""

    run_id: str
    plan_id: str
    plan: TrackingPlan
    diff: PlanDiff | None
    objective: ObjectiveChange
    summary: str


def run_counterfactual_dry_run(
    snapshot: PlanningSnapshot,
    overrides: Mapping[str, object],
    *,
    config: PlanningConfig | None = None,
    run_id: str | None = None,
) -> CounterfactualResult:
    """Clone the snapshot, apply the overrides, and re-run the optimizer.

    The optimizer is a pure function pair (``optimize_candidates`` +
    ``select_candidate``): no repository and no event publisher participate,
    so the online ``PlanRepository.commit`` and the event publisher are
    unreachable from this path by construction. The dry-run plan keeps its
    own ``dry-run:<uuid>`` namespace and is never persisted.
    """
    plan_config = config if config is not None else _DEFAULT_CONFIG
    resolved_run_id = run_id if run_id is not None else f"dry-run:{uuid.uuid4().hex}"
    parsed = parse_counterfactual_overrides(overrides, snapshot.situation)
    cloned = _clone_with_overrides(snapshot, parsed, resolved_run_id, plan_config)
    targets = tuple(
        sorted({report.target_id for report in cloned.situation.group_reports})
    )
    proposal = _dry_run_proposal(cloned, targets, parsed)
    evaluations = optimize_candidates(
        cloned, StrategySet(trigger_event_ids=(), proposals=(proposal,)), plan_config
    )
    selected = select_candidate(cloned, evaluations, plan_config)
    plan, plan_id = _isolate_plan(selected, resolved_run_id)
    active = snapshot.active_plan
    objective = ObjectiveChange(
        active_count_before=(
            active.predicted_active_count if active is not None else 0
        ),
        active_count_after=plan.predicted_active_count,
        economic_cost_before=active.predicted_energy if active is not None else 0.0,
        economic_cost_after=plan.predicted_energy,
    )
    return CounterfactualResult(
        run_id=resolved_run_id,
        plan_id=plan_id,
        plan=plan,
        diff=plan.diff,
        objective=objective,
        summary=_summary(resolved_run_id, plan_id, plan, plan.diff, objective),
    )


def _clone_with_overrides(
    snapshot: PlanningSnapshot,
    overrides: CounterfactualOverrides,
    run_id: str,
    config: PlanningConfig,
) -> PlanningSnapshot:
    """Deep-clone the snapshot and apply the override effects to the clone."""
    situation = snapshot.situation.model_copy(deep=True)
    del config
    # A counterfactual changes the requested quality floor, not the observed
    # quality. The allocator already grows a prior group when its measured
    # quality is below that floor; lowering the observation as well would
    # create an artificial infeasibility.
    updated_reports = list(situation.group_reports)
    situation = situation.model_copy(update={"group_reports": tuple(updated_reports)})
    scope = tuple(
        sorted(set(overrides.target_priorities) | set(overrides.minimum_quality))
    )
    override_directive = ExpertDirective(
        directive_id=f"{run_id}:override",
        raw_text=f"counterfactual overrides for {run_id}",
        target_scope=scope,
        target_priorities=overrides.target_priorities,
        minimum_quality=overrides.minimum_quality,
        disabled_uuv_ids=overrides.disabled_uuv_ids,
        confidence=1.0,
        status="applied",
    )
    return PlanningSnapshot(
        situation=situation,
        active_plan=(
            snapshot.active_plan.model_copy(deep=True)
            if snapshot.active_plan is not None
            else None
        ),
        applied_directives=(*snapshot.applied_directives, override_directive),
    )


def _dry_run_proposal(
    snapshot: PlanningSnapshot,
    targets: Sequence[str],
    overrides: CounterfactualOverrides,
) -> StrategyProposal:
    """One deterministic quality_first proposal honoring the overrides."""
    return StrategyProposal(
        concept="quality_first",
        target_priorities={
            target: overrides.target_priorities.get(target, 1.0) for target in targets
        },
        required_quality={
            target: overrides.minimum_quality.get(target, _DEFAULT_REQUIRED_QUALITY)
            for target in targets
        },
        reinforcement_policy={target: "release_when_stable" for target in targets},
        releasable_soft_constraints=("energy_reserve_0.1",),
        evidence_ids=_evidence_ids(snapshot),
        rationale="counterfactual dry-run quality_first",
    )


def _evidence_ids(snapshot: PlanningSnapshot) -> tuple[str, ...]:
    """The dry-run proposal's evidence: the cloned groups' observation ids."""
    return tuple(
        sorted(
            {
                observation_id
                for report in snapshot.situation.group_reports
                for observation_id in report.belief.source_observation_ids
            }
        )
    )


def _isolate_plan(selected: TrackingPlan, run_id: str) -> tuple[TrackingPlan, str]:
    """Rewrite the selected plan into the isolated ``dry-run:<uuid>`` namespace."""
    plan_id = f"{run_id}:plan:{selected.scenario_id}:{selected.revision}"
    diff = selected.diff
    if diff is not None:
        diff = diff.model_copy(update={"to_plan_id": plan_id})
    return selected.model_copy(update={"plan_id": plan_id, "diff": diff}), plan_id


def _summary(
    run_id: str,
    plan_id: str,
    plan: TrackingPlan,
    diff: PlanDiff | None,
    objective: ObjectiveChange,
) -> str:
    """Deterministic one-line summary of the dry-run outcome."""
    return (
        f"dry-run {run_id}: {plan.concept} plan {plan_id} "
        f"active {objective.active_count_before}->{objective.active_count_after} "
        f"economic {objective.economic_cost_before:.1f}->{objective.economic_cost_after:.1f}; "
        f"{_diff_line(diff)}"
    )


def _diff_line(diff: PlanDiff | None) -> str:
    """Render the member churn of the dry-run plan deterministically."""
    if diff is None:
        return "no member change"
    parts: list[str] = []
    for target, members in sorted(diff.members_added.items()):
        parts.append(f"added {','.join(members)} to {target}")
    for target, members in sorted(diff.members_removed.items()):
        parts.append(f"removed {','.join(members)} from {target}")
    if diff.waypoints_changed:
        parts.append(
            f"waypoints changed for {','.join(sorted(diff.waypoints_changed))}"
        )
    return "; ".join(parts) or f"no member change ({diff.to_plan_id})"
