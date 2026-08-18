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

``ValidateNode`` performs one validation round and pins the round-0
candidate into ``original_candidate`` before any repair replaces it;
``RepairNode`` re-invokes the LLM with the pinned ORIGINAL candidate, the
current candidate under repair, the machine-readable issues, and the
UNCHANGED schema — the same ``StrategyProposal`` response model and the
immutable strategy system prompt. If the bounded semantic budget is
exhausted, ``ContentFailureNode`` raises ``LLMContentError`` so the runtime
pauses and can retry the real provider; no deterministic strategy replaces
the LLM. The graph wiring lives in ``underwater_tracking.agent.graphs.verify``.

Transport retries are independent from semantic repairs: transient and
config errors propagate out of ``RepairNode`` untouched (the LLM port
retries them internally against its own counter), while schema/content
failures and provider exhaustion consume one semantic attempt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

from pydantic import ValidationError

from underwater_tracking.agent.llm import (
    LLMConfigError,
    LLMContentError,
    LLMError,
    StructuredLLM,
    TransientLLMError,
)
from underwater_tracking.agent.prompts import (
    STRATEGY_PROMPT_VERSION,
    STRATEGY_SYSTEM_PROMPT,
)
from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    StrategyProposal,
    TrackingPlan,
    ValidationIssue,
    ValidationReport,
)
from underwater_tracking.domain.regional_models import RegionTask
from underwater_tracking.planning.regional_validation import validate_regional_plan

if TYPE_CHECKING:
    from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot

# Spec 8.3: bounded content re-injection, at most two semantic repairs.
_MAX_REPAIRS_DEFAULT = 2

# StrategyProposal must never carry final group members or waypoints (spec
# 6.8); they live only in TrackingPlan. The scan covers only the structural
# fields where smuggled members or waypoints could appear; ``evidence_ids``
# legitimately embed producing UUV ids (e.g. ``B:T1:uuv_00:900``) and the
# free-text ``rationale`` may discuss members, so both are exempt.
_FORBIDDEN_MARKERS = ("waypoint", "member", "uuv")
_SCANNED_STRUCTURAL_FIELDS = (
    "concept",
    "target_priorities",
    "required_quality",
    "reinforcement_policy",
    "releasable_soft_constraints",
    "segment_plan",
)

# ``group_id`` inside a segment plan is an identifier the LLM writes for
# the group that takes the segment; exempting that key (never any other
# segment field) keeps legitimately relay-named groups from tripping the
# member/waypoint scan.
_SEGMENT_MARKER_EXEMPT_KEYS = frozenset({"group_id"})


def validate_regional_tasks(
    snapshot: PlanningSnapshot,
    plan: TrackingPlan,
) -> tuple[ValidationIssue, ...]:
    """Validate authoritative regional tasks before legacy plan projections.

    Regional quality and coverage are planning proxies, not sensor truth. A
    degraded or uncovered task remains present in the validation result and
    plan metrics; it is never silently removed to satisfy target-level views.
    """
    if not plan.regional_plans and not plan.region_tasks:
        return ()

    issues: list[ValidationIssue] = []
    known_regions = {
        cell.region_id
        for regional_plan in plan.regional_plans.values()
        for cell in regional_plan.cells
    }
    for region_id in sorted(set(plan.region_tasks) - known_regions):
        issues.append(
            ValidationIssue(
                code="regional_unknown_region",
                field=f"region_tasks[{region_id}]",
                message=f"regional task {region_id} has no matching region cell",
            )
        )

    platform_snapshot = snapshot.situation.platform_snapshot
    for target_id, regional_plan in sorted(plan.regional_plans.items()):
        tasks = tuple(
            plan.region_tasks.get(task.region_id, task)
            for task in regional_plan.tasks
        )
        issues.extend(_regional_role_and_handoff_issues(tasks, regional_plan.region_ids))
        if platform_snapshot is None:
            continue
        effective_plan = regional_plan.model_copy(update={"tasks": tasks})
        for issue in validate_regional_plan(
            effective_plan,
            platform_snapshot.roster,
            carrier=platform_snapshot.carrier,
            map_bounds_xy=snapshot.situation.map_bounds_xy,
        ):
            issues.append(
                ValidationIssue(
                    code=f"regional_{issue.split(':', 1)[0]}",
                    field=_regional_issue_field(issue, target_id),
                    message=issue,
                )
            )
    return tuple(sorted(issues, key=lambda issue: (issue.code, issue.field, issue.message)))


def _regional_role_and_handoff_issues(
    tasks: tuple[RegionTask, ...],
    region_ids: tuple[str, ...],
) -> tuple[ValidationIssue, ...]:
    known_regions = frozenset(region_ids)
    issues: list[ValidationIssue] = []
    for task in tasks:
        field = f"region_tasks[{task.region_id}]"
        if len(task.uuv_roles) > len(task.assigned_uuv_ids):
            issues.append(
                ValidationIssue(
                    code="regional_role_assignment",
                    field=field,
                    message="regional UUV roles exceed assigned UUV members",
                )
            )
        if task.assigned_usv_ids and task.usv_role is None:
            issues.append(
                ValidationIssue(
                    code="regional_role_assignment",
                    field=field,
                    message="regional USV members require a USV role",
                )
            )
        for linked_region, relationship in (
            (task.predecessor_region_id, "predecessor"),
            (task.successor_region_id, "successor"),
        ):
            if linked_region is not None and linked_region not in known_regions:
                issues.append(
                    ValidationIssue(
                        code="regional_handoff_unknown_region",
                        field=field,
                        message=f"{relationship} region {linked_region} is not in the target plan",
                    )
                )
    return tuple(issues)


def _regional_issue_field(issue: str, target_id: str) -> str:
    _, separator, region_id = issue.partition(":")
    if separator and ":cell:" in region_id:
        return f"region_tasks[{region_id}]"
    return f"regional_plans[{target_id}]"


class VerifyState(TypedDict, total=False):
    """State of the bounded semantic Verify subgraph (spec 8.3).

    ``candidate`` is the raw provider output under scrutiny — a validated
    ``StrategyProposal`` (the pipeline case) or a raw dict (schema checks
    still apply). ``attempt`` counts semantic repair rounds so far;
    ``max_repairs`` bounds them (default 2, spec 8.3). On success
    ``verified_strategy`` carries the final proposal. ``degraded`` is retained
    as a compatibility field and is always false for this graph: semantic
    failure raises instead of silently degrading. The optional context fields
    override the graph-level defaults per invoke.
    """

    candidate: dict[str, object] | StrategyProposal | None
    # The TRUE original candidate, pinned by the first validation round
    # before any repair replaces ``candidate``; repair payloads always ship
    # this value under ``original_candidate``.
    original_candidate: dict[str, object] | StrategyProposal | None
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


def route_validity(state: VerifyState) -> Literal["end", "repair", "failure"]:
    """Route after one validation round (spec 8.3).

    A valid candidate ends; an invalid one goes back to ``repair`` while
    semantic attempts remain, and to ``failure`` once the budget is
    exhausted. The failure route is deliberately explicit so a content
    failure cannot be mistaken for a usable degraded strategy.
    """
    report = state.get("validation_report")
    if report is not None and report.valid:
        return "end"
    if state.get("attempt", 0) < state.get("max_repairs", _MAX_REPAIRS_DEFAULT):
        return "repair"
    return "failure"


class ValidateNode:
    """One validation round; sets the report and, on success, the verified strategy.

    The first round also pins the TRUE original candidate into
    ``original_candidate`` before any repair replaces ``candidate``, so
    every later repair payload can still reference the round-0 input.
    """

    def __init__(self, context: VerifyContext = _DEFAULT_CONTEXT) -> None:
        self._defaults = context

    def __call__(self, state: VerifyState) -> VerifyState:
        context = resolve_context(state, self._defaults)
        original = state.get("original_candidate", state.get("candidate"))
        report = validate_strategy(
            state.get("candidate"),
            target_ids=context.target_ids,
            evidence_ids=context.evidence_ids,
            allowed_soft_constraints=context.allowed_soft_constraints,
            expert_directive=context.expert_directive,
        )
        result: VerifyState = {
            "original_candidate": original,
            "validation_report": report,
            "repair_attempts": state.get("attempt", 0),
            "degraded": False,
        }
        parsed = parse_strategy(state.get("candidate"))
        if report.valid and parsed is not None:
            result["verified_strategy"] = parsed
        return result


class RepairNode:
    """Re-invoke the LLM with the pinned original candidate, issues, and unchanged schema.

    The payload carries both the pinned round-0 ``original_candidate`` and
    the current ``candidate`` under repair, which may differ on repair
    rounds after the first. Transient and config errors propagate untouched
    — transport retries run inside the LLM port against their own
    independent counter and never consume a semantic attempt. Schema/content
    failures and provider exhaustion keep the current candidate and consume
    one attempt.
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
        """Machine-readable repair payload: original + current candidate, issues, schema.

        ``original_candidate`` is the pinned round-0 candidate (never a
        repair output); ``candidate`` is the proposal currently under
        repair. The response schema (``StrategyProposal``) and the system
        prompt are the unchanged ones from the original strategy call — the
        repair never loosens the contract.
        """
        original = state.get("original_candidate")
        current = state.get("candidate")
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
            "candidate": _serialize(current),
            "validation_issues": [issue.model_dump(mode="json") for issue in issues],
            "target_ids": sorted(context.target_ids),
            "evidence_ids": sorted(context.evidence_ids),
            "allowed_soft_constraints": sorted(context.allowed_soft_constraints),
            "expert_directive": _serialize(context.expert_directive),
        }


class ContentFailureNode:
    """Stop the cycle after bounded semantic repair is exhausted.

    This node intentionally raises instead of returning a last-valid or
    deterministic emergency proposal. The simulation transaction rolls back
    the current tick and the caller can reconnect/retry the real LLM at the
    same situation time.
    """

    def __call__(self, state: VerifyState) -> VerifyState:
        report = state.get("validation_report")
        details = "; ".join(
            f"{issue.code}:{issue.field}" for issue in (report.issues if report else ())
        )
        suffix = f" ({details})" if details else ""
        raise LLMContentError(
            "strategy verification exhausted semantic repairs; "
            f"real LLM response remains invalid{suffix}"
        )


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
        if target not in proposal.reinforcement_policy:
            issues.append(
                _issue("missing_coverage", "reinforcement_policy",
                       f"missing reinforcement policy for tracked target {target!r}",
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
    segments = proposal.segment_plan
    if segments is not None:
        for index, segment in enumerate(segments.segments):
            if segment.index != index:
                issues.append(
                    _issue("segment_index_gap", f"segment_plan.segments[{index}]",
                           "segment indices must be contiguous from 0",
                           segment.index, index)
                )
            if segment.end_s <= segment.start_s:
                issues.append(
                    _issue("segment_time_invalid", f"segment_plan.segments[{index}]",
                           "segment end must follow its start",
                           segment.end_s, f"> {segment.start_s}")
                )
            if not (
                math.isfinite(segment.intercept_xy[0])
                and math.isfinite(segment.intercept_xy[1])
            ):
                issues.append(
                    _issue("non_finite", f"segment_plan.segments[{index}].intercept_xy",
                           "segment intercept must be finite",
                           segment.intercept_xy, "finite floats")
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


def _find_forbidden_marker(dump: dict[str, object]) -> tuple[str, str] | None:
    """First (path, value) in a structural field naming members/waypoints.

    Only ``_SCANNED_STRUCTURAL_FIELDS`` are scanned — the concept,
    priorities, quality, reinforcement policies, soft constraints, and the
    segment plan — where final members or waypoints would have to appear
    if smuggled (spec 6.8). Citation fields like ``evidence_ids``
    legitimately embed producing UUV ids (e.g. ``B:T1:uuv_00:900``) and
    the free-text ``rationale`` may discuss members; both are exempt.
    """
    for field in _SCANNED_STRUCTURAL_FIELDS:
        value = dump.get(field)
        if value is not None:
            skip_keys = (
                _SEGMENT_MARKER_EXEMPT_KEYS if field == "segment_plan" else frozenset()
            )
            found = _scan_value(value, path=field, skip_keys=skip_keys)
            if found is not None:
                return found
    return None


def _scan_value(
    value: object, path: str, skip_keys: frozenset[str] = frozenset()
) -> tuple[str, str] | None:
    """First (path, value) under ``value`` whose key or string names a marker."""
    if isinstance(value, str):
        if any(marker in value.lower() for marker in _FORBIDDEN_MARKERS):
            return (path, value)
        return None
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            if key in skip_keys:
                continue
            if any(marker in str(key).lower() for marker in _FORBIDDEN_MARKERS):
                return (f"{path}.{key}", str(key))
            found = _scan_value(child, f"{path}.{key}", skip_keys)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _scan_value(child, f"{path}[{index}]", skip_keys)
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
