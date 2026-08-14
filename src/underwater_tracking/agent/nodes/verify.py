# src/underwater_tracking/agent/nodes/verify.py
"""Bounded semantic verification of candidate strategies (spec 8.3, plan Task 6).

The Verify subgraph runs the content validation chain — provider structured
output, Pydantic schema, ID/evidence/business semantics, strategy
constraints — and on failure re-injects the machine-readable issues into the
model for at most ``max_repairs`` rounds. ``validate_strategy`` is the pure
per-round validator: strict Pydantic validity, known target ids, finite
priorities/quality, complete target coverage, evidence-id existence,
allowed soft constraints, the member/waypoint prohibition (spec 6.8), and
consistency with applied expert hard constraints. It returns a sorted tuple
of ``ValidationIssue(code, field, message, observed, expected)`` inside a
``ValidationReport``.

``ValidateNode`` performs one validation round; ``RepairNode`` re-invokes the
LLM with the ORIGINAL candidate, the machine-readable issues, and the
UNCHANGED schema — the same ``StrategyProposal`` response model and the
immutable strategy system prompt; ``FallbackNode`` keeps the last valid
strategy while it is still feasible and otherwise builds a deterministic
emergency strategy that prioritizes every already-tracked target. The graph
wiring lives in ``underwater_tracking.agent.graphs.verify``.

Transport retries are independent from semantic repairs: transient and
config errors propagate out of ``RepairNode`` untouched (the LLM port
retries them internally against its own counter), while schema/content
failures and provider exhaustion consume one semantic attempt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypedDict

from pydantic import ValidationError

from underwater_tracking.agent.llm import (
    LLMConfigError,
    LLMError,
    StructuredLLM,
    TransientLLMError,
)
from underwater_tracking.agent.prompts import (
    STRATEGY_PROMPT_VERSION,
    STRATEGY_SYSTEM_PROMPT,
)
from underwater_tracking.domain.agent_models import (
    Concept,
    ExpertDirective,
    StrategyProposal,
    ValidationIssue,
    ValidationReport,
)

# Spec 8.3: bounded content re-injection, at most two semantic repairs.
_MAX_REPAIRS_DEFAULT = 2

# Deterministic emergency strategy constants (spec 8.3: when the last valid
# strategy is no longer feasible, run the deterministic emergency optimizer,
# prioritizing already-tracked high-priority targets).
_EMERGENCY_CONCEPT: Concept = "quality_first"
_EMERGENCY_QUALITY = 0.7
_EMERGENCY_REINFORCEMENT_POLICY = "release_when_stable"
_EMERGENCY_RATIONALE = (
    "Deterministic emergency fallback: no proposal survived repair, so every "
    "already-tracked target keeps full priority with stable release."
)

# StrategyProposal must never carry final group members or waypoints (spec
# 6.8); they live only in TrackingPlan. The free-text ``rationale`` is
# exempt from the scan because prose may legitimately discuss members.
_FORBIDDEN_MARKERS = ("waypoint", "member", "uuv")


class VerifyState(TypedDict, total=False):
    """State of the bounded semantic Verify subgraph (spec 8.3).

    ``candidate`` is the raw provider output under scrutiny — a validated
    ``StrategyProposal`` (the pipeline case) or a raw dict (schema checks
    still apply). ``attempt`` counts semantic repair rounds so far;
    ``max_repairs`` bounds them (default 2, spec 8.3). On success
    ``verified_strategy`` carries the final proposal and ``degraded``
    records whether the fallback path was taken. The optional context
    fields override the graph-level defaults per invoke.
    """

    candidate: dict[str, object] | StrategyProposal | None
    attempt: int
    max_repairs: int
    last_valid_strategy: StrategyProposal | None
    verified_strategy: StrategyProposal | None
    repair_attempts: int
    degraded: bool
    validation_report: ValidationReport | None
    target_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    allowed_soft_constraints: tuple[str, ...]
    expert_directive: ExpertDirective | None
    scenario_id: str
    sim_time_s: int


@dataclass(frozen=True)
class VerifyContext:
    """Static semantic validation context for one Verify subgraph instance."""

    target_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    allowed_soft_constraints: tuple[str, ...] = ()
    expert_directive: ExpertDirective | None = None


# Shared immutable default for node constructors (B008: no call in defaults).
_DEFAULT_CONTEXT = VerifyContext()


def resolve_context(state: VerifyState, defaults: VerifyContext) -> VerifyContext:
    """Per-invoke context: state values win over the constructed defaults."""
    return VerifyContext(
        target_ids=state.get("target_ids", defaults.target_ids),
        evidence_ids=state.get("evidence_ids", defaults.evidence_ids),
        allowed_soft_constraints=state.get(
            "allowed_soft_constraints", defaults.allowed_soft_constraints
        ),
        expert_directive=state.get("expert_directive", defaults.expert_directive),
    )


def parse_strategy(
    candidate: dict[str, object] | StrategyProposal | None,
) -> StrategyProposal | None:
    """Strict Pydantic parse of the candidate; None for absent or invalid input."""
    if isinstance(candidate, StrategyProposal):
        return candidate
    if candidate is None:
        return None
    try:
        return StrategyProposal.model_validate(candidate)
    except ValidationError:
        return None


def validate_strategy(
    candidate: dict[str, object] | StrategyProposal | None,
    *,
    target_ids: tuple[str, ...] = (),
    evidence_ids: tuple[str, ...] = (),
    allowed_soft_constraints: tuple[str, ...] = (),
    expert_directive: ExpertDirective | None = None,
) -> ValidationReport:
    """Validate one candidate through the schema and semantic checks.

    The checks are: strict Pydantic validity, known target ids, finite
    priorities/quality, complete target coverage, evidence-id existence,
    allowed soft constraints, no member IDs/waypoints, and consistency with
    applied expert hard constraints. Issues are sorted deterministically by
    ``(code, field, message, observed)``.
    """
    if candidate is None:
        return ValidationReport(
            valid=False,
            issues=(
                ValidationIssue(
                    code="no_candidate",
                    field="candidate",
                    message="no candidate strategy was produced",
                ),
            ),
        )
    if not isinstance(candidate, StrategyProposal):
        try:
            proposal = StrategyProposal.model_validate(candidate)
        except ValidationError as exc:
            issues = tuple(
                ValidationIssue(
                    code="schema_invalid",
                    field=".".join(str(part) for part in error.get("loc", ())),
                    message=error.get("msg", "invalid StrategyProposal"),
                    observed=str(error.get("input")),
                    expected="valid StrategyProposal",
                )
                for error in exc.errors()
            )
            return ValidationReport(valid=False, issues=_sorted(issues))
    else:
        proposal = candidate
    issues = _semantic_issues(
        proposal,
        target_ids=target_ids,
        evidence_ids=evidence_ids,
        allowed_soft_constraints=allowed_soft_constraints,
        expert_directive=expert_directive,
    )
    return ValidationReport(valid=not issues, issues=_sorted(issues))


def route_validity(state: VerifyState) -> Literal["end", "repair", "fallback"]:
    """Route after one validation round (spec 8.3).

    A valid candidate ends; an invalid one goes back to ``repair`` while
    semantic attempts remain, and to ``fallback`` once the budget is
    exhausted.
    """
    report = state.get("validation_report")
    if report is not None and report.valid:
        return "end"
    if state.get("attempt", 0) < state.get("max_repairs", _MAX_REPAIRS_DEFAULT):
        return "repair"
    return "fallback"


class ValidateNode:
    """One validation round; sets the report and, on success, the verified strategy."""

    def __init__(self, context: VerifyContext = _DEFAULT_CONTEXT) -> None:
        self._defaults = context

    def __call__(self, state: VerifyState) -> VerifyState:
        context = resolve_context(state, self._defaults)
        report = validate_strategy(
            state.get("candidate"),
            target_ids=context.target_ids,
            evidence_ids=context.evidence_ids,
            allowed_soft_constraints=context.allowed_soft_constraints,
            expert_directive=context.expert_directive,
        )
        result: VerifyState = {
            "validation_report": report,
            "repair_attempts": state.get("attempt", 0),
            "degraded": False,
        }
        parsed = parse_strategy(state.get("candidate"))
        if report.valid and parsed is not None:
            result["verified_strategy"] = parsed
        return result


class RepairNode:
    """Re-invoke the LLM with the original candidate, issues, and unchanged schema.

    Transient and config errors propagate untouched — transport retries run
    inside the LLM port against their own independent counter and never
    consume a semantic attempt. Schema/content failures and provider
    exhaustion keep the original candidate and consume one attempt.
    """

    def __init__(
        self,
        llm: StructuredLLM[StrategyProposal],
        *,
        model_id: str = "underwater-assistant-model",
        prompt_version: str = STRATEGY_PROMPT_VERSION,
        temperature: float = 0.2,
        context: VerifyContext = _DEFAULT_CONTEXT,
    ) -> None:
        self._llm = llm
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._temperature = temperature
        self._defaults = context

    def __call__(self, state: VerifyState) -> VerifyState:
        context = resolve_context(state, self._defaults)
        report = state.get("validation_report")
        issues = report.issues if report is not None else ()
        payload = self.build_payload(state, context, issues)
        try:
            repaired = self._llm.invoke_structured(
                "strategy",
                payload,
                StrategyProposal,
                prompt_version=self._prompt_version,
            )
        except (TransientLLMError, LLMConfigError):
            raise
        except LLMError:
            # Schema/content failure or provider exhaustion: the attempt is
            # consumed and the original candidate stays under scrutiny.
            repaired = None
        attempt = state.get("attempt", 0) + 1
        return {
            "candidate": repaired if repaired is not None else state.get("candidate"),
            "attempt": attempt,
            "repair_attempts": attempt,
        }

    def build_payload(
        self,
        state: VerifyState,
        context: VerifyContext,
        issues: tuple[ValidationIssue, ...],
    ) -> dict[str, object]:
        """Machine-readable repair payload: original candidate, issues, schema.

        The response schema (``StrategyProposal``) and the system prompt are
        the unchanged ones from the original strategy call — the repair
        never loosens the contract.
        """
        original = state.get("candidate")
        return {
            "model": self._model_id,
            "temperature": self._temperature,
            "system_prompt": STRATEGY_SYSTEM_PROMPT,
            "response_schema": "StrategyProposal",
            "operation": "strategy_repair",
            "scenario_id": state.get("scenario_id", ""),
            "sim_time_s": state.get("sim_time_s", 0),
            "requested_concept": _requested_concept(original),
            "original_candidate": _serialize(original),
            "validation_issues": [issue.model_dump(mode="json") for issue in issues],
            "target_ids": sorted(context.target_ids),
            "evidence_ids": sorted(context.evidence_ids),
            "allowed_soft_constraints": sorted(context.allowed_soft_constraints),
            "expert_directive": _serialize(context.expert_directive),
        }


class FallbackNode:
    """Degrade to the last valid strategy or a deterministic emergency one."""

    def __init__(self, context: VerifyContext = _DEFAULT_CONTEXT) -> None:
        self._defaults = context

    def __call__(self, state: VerifyState) -> VerifyState:
        context = resolve_context(state, self._defaults)
        attempt = state.get("attempt", 0)
        last_valid = state.get("last_valid_strategy")
        if last_valid is not None:
            report = validate_strategy(
                last_valid,
                target_ids=context.target_ids,
                evidence_ids=context.evidence_ids,
                allowed_soft_constraints=context.allowed_soft_constraints,
                expert_directive=context.expert_directive,
            )
            if report.valid:
                return {
                    "verified_strategy": last_valid,
                    "repair_attempts": attempt,
                    "degraded": True,
                }
        return {
            "verified_strategy": _emergency_strategy(context),
            "repair_attempts": attempt,
            "degraded": True,
        }


def _semantic_issues(
    proposal: StrategyProposal,
    *,
    target_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
    allowed_soft_constraints: tuple[str, ...],
    expert_directive: ExpertDirective | None,
) -> tuple[ValidationIssue, ...]:
    """ID/evidence/business-semantics and strategy-constraint checks."""
    known = frozenset(target_ids)
    allowed = frozenset(allowed_soft_constraints)
    evidence = frozenset(evidence_ids)
    issues: list[ValidationIssue] = []
    for target in sorted(set(proposal.target_priorities) - known):
        issues.append(
            _issue("unknown_target", "target_priorities",
                   f"priority set for unknown target {target!r}", target,
                   "known target id")
        )
    for target in sorted(set(proposal.required_quality) - known):
        issues.append(
            _issue("unknown_target", "required_quality",
                   f"required quality set for unknown target {target!r}", target,
                   "known target id")
        )
    for target in sorted(set(proposal.reinforcement_policy) - known):
        issues.append(
            _issue("unknown_target", "reinforcement_policy",
                   f"reinforcement policy set for unknown target {target!r}",
                   target, "known target id")
        )
    for target, value in sorted(proposal.target_priorities.items()):
        if not math.isfinite(value):
            issues.append(
                _issue("non_finite", f"target_priorities[{target}]",
                       "priority must be a finite float", value, "finite float")
            )
    for target, value in sorted(proposal.required_quality.items()):
        if not math.isfinite(value):
            issues.append(
                _issue("non_finite", f"required_quality[{target}]",
                       "required quality must be a finite float", value,
                       "finite float")
            )
    for target in sorted(known):
        if target not in proposal.target_priorities:
            issues.append(
                _issue("missing_coverage", "target_priorities",
                       f"missing priority for tracked target {target!r}", None,
                       "every tracked target")
            )
        if target not in proposal.required_quality:
            issues.append(
                _issue("missing_coverage", "required_quality",
                       f"missing required quality for tracked target {target!r}",
                       None, "every tracked target")
            )
    for evidence_id in sorted(set(proposal.evidence_ids) - evidence):
        issues.append(
            _issue("unknown_evidence", "evidence_ids",
                   f"evidence id {evidence_id!r} does not exist", evidence_id,
                   "known evidence id")
        )
    for constraint in sorted(set(proposal.releasable_soft_constraints) - allowed):
        issues.append(
            _issue("disallowed_soft_constraint", "releasable_soft_constraints",
                   f"soft constraint {constraint!r} is not allowed", constraint,
                   "allowed soft constraint")
        )
    marker = _find_forbidden_marker(proposal.model_dump(mode="json"))
    if marker is not None:
        marker_path, marker_value = marker
        issues.append(
            _issue("member_or_waypoint", marker_path,
                   "candidate references final members or waypoints", marker_value,
                   "no member ids or waypoints")
        )
    if expert_directive is not None and expert_directive.status == "applied":
        for target, minimum in sorted(expert_directive.minimum_quality.items()):
            actual = proposal.required_quality.get(target)
            if actual is not None and actual < minimum:
                issues.append(
                    _issue("expert_constraint_violation", f"required_quality[{target}]",
                           "quality below the applied expert minimum", actual,
                           f">= {minimum}")
                )
        for target, priority in sorted(expert_directive.target_priorities.items()):
            actual = proposal.target_priorities.get(target)
            if actual is not None and actual != priority:
                issues.append(
                    _issue("expert_constraint_violation", f"target_priorities[{target}]",
                           "priority contradicts the applied expert directive",
                           actual, priority)
                )
    return tuple(issues)


def _emergency_strategy(context: VerifyContext) -> StrategyProposal | None:
    """Deterministic emergency strategy: prioritize every tracked target.

    Targets are sorted for stability; the proposal reuses the available
    evidence. Returns None only when no evidence exists to ground an
    evidence-backed proposal on.
    """
    evidence = sorted(context.evidence_ids)
    if not evidence:
        return None
    targets = sorted(context.target_ids)
    return StrategyProposal(
        concept=_EMERGENCY_CONCEPT,
        target_priorities={target: 1.0 for target in targets},
        required_quality={target: _EMERGENCY_QUALITY for target in targets},
        reinforcement_policy={
            target: _EMERGENCY_REINFORCEMENT_POLICY for target in targets
        },
        releasable_soft_constraints=(),
        evidence_ids=tuple(evidence),
        rationale=_EMERGENCY_RATIONALE,
    )


def _find_forbidden_marker(value: object, path: str = "proposal") -> tuple[str, str] | None:
    """First (path, value) whose key or string value names members/waypoints.

    The free-text ``rationale`` is exempt: prose may legitimately discuss
    members, while the structural fields must never reference final group
    members, UUV ids, or waypoints (spec 6.8).
    """
    if isinstance(value, str):
        if any(marker in value.lower() for marker in _FORBIDDEN_MARKERS):
            return (path, value)
        return None
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            if key == "rationale":
                continue
            if any(marker in str(key).lower() for marker in _FORBIDDEN_MARKERS):
                return (f"{path}.{key}", str(key))
            found = _find_forbidden_marker(child, f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _find_forbidden_marker(child, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    return None


def _requested_concept(candidate: object) -> str | None:
    """The concept the original candidate was requested for, when determinable."""
    if isinstance(candidate, StrategyProposal):
        return candidate.concept
    if isinstance(candidate, dict):
        value = candidate.get("concept")
        if isinstance(value, str):
            return value
    return None


def _serialize(value: object) -> object:
    """JSON-serializable form of a candidate, directive, or raw payload."""
    if isinstance(value, StrategyProposal):
        return value.model_dump(mode="json")
    if isinstance(value, ExpertDirective):
        return value.model_dump(mode="json")
    return value


def _issue(
    code: str,
    field: str,
    message: str,
    observed: object,
    expected: object,
) -> ValidationIssue:
    """One machine-readable ValidationIssue with normalized display values."""
    return ValidationIssue(
        code=code,
        field=field,
        message=message,
        observed=str(observed) if observed is not None else None,
        expected=str(expected),
    )


def _sorted(issues: tuple[ValidationIssue, ...]) -> tuple[ValidationIssue, ...]:
    """Deterministic issue ordering by (code, field, message, observed)."""
    return tuple(sorted(issues, key=_issue_sort_key))


def _issue_sort_key(issue: ValidationIssue) -> tuple[str, str, str, str]:
    return (issue.code, issue.field, issue.message, str(issue.observed))
