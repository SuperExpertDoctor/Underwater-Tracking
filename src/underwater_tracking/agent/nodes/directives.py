# src/underwater_tracking/agent/nodes/directives.py
"""Non-blocking expert directives (spec 10.1, plan Task 10).

Annotations enter an independent parse branch:

    raw text -> LLM parse ExpertDirective -> schema/conflict validation
    -> structured preview -> expert clicks apply -> strategic event

Parsing and confirmation never interrupt the running plan:
``preview_directive``/``apply_directive`` are synchronous runtime
operations over the injected LLM and ledger, and the graph's
``directive_branch`` node only surfaces the latest applied directive onto
the checkpointed state. Low-confidence, ambiguous, or hard-constraint
conflicting directives resolve to ``needs_clarification`` and are never
auto-applied; regular annotations never use a blocking ``interrupt()``
(reserved for a future optional approval mode).

``validate_directive`` is the single deterministic validator shared by the
LLM parse path and the typed shortcut helpers (``lock_group_members``,
``set_target_priority``, ``set_minimum_quality``, ``disable_uuv``): it
checks the named IDs against the live situation, the resource bounds, and
hard-constraint conflicts against the applied directives, then resolves
the directive's status.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from underwater_tracking.agent.prompts import DIRECTIVE_SYSTEM_PROMPT
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import ExpertDirective
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.persistence.ledger import DecisionLedger

# The directive parsing operation key (Mock LLM queue key, spec 22).
DIRECTIVE_OPERATION = "directive"
# Strategic event emitted when an expert applies a directive (spec 8.2).
DIRECTIVE_APPLIED_EVENT_TYPE = "directive_applied"
# Matches the ExpertDirective validator: below this confidence, applied is
# impossible and the directive requests clarification instead.
_APPLY_CONFIDENCE_MIN = 0.70
_DEFAULT_MODEL_ID = "underwater-assistant-model"
_DEFAULT_TEMPERATURE = 0.2


class DirectiveNotApplicableError(ValueError):
    """Raised when applying a preview that is not cleanly applicable."""


class _DirectiveState(CarrierState, total=False):
    """Branch state: the carrier channels plus the deferred error marker.

    Mirrors ``CentralState`` in the carrier graph; the marker is defined
    locally so the node module never imports the graph (no circularity).
    """

    node_error: str | None


class DirectiveNode:
    """Carrier branch node surfacing the latest applied directive (spec 10.1).

    Runs between ``build_snapshot`` and the three-tier routing: each
    ``directive_applied`` event in the cycle carries only its directive id,
    which is resolved against the ledger's applied directives, and the
    latest applied directive is set on the state. An event referencing an
    unknown id defers a node error so the cycle completes via
    ``handle_error`` instead of crashing.
    """

    def __init__(self, ledger: DecisionLedger) -> None:
        self._ledger = ledger

    def __call__(self, state: _DirectiveState) -> _DirectiveState:
        scenario_id = state.get("scenario_id")
        if not scenario_id:
            return {"node_error": "directive_branch requires scenario_id in state"}
        applied = {
            directive.directive_id: directive
            for directive in self._ledger.list_directives(
                scenario_id, status="applied"
            )
        }
        latest: ExpertDirective | None = None
        for event in state.get("coalesced_events") or ():
            if event.event_type != DIRECTIVE_APPLIED_EVENT_TYPE:
                continue
            directive_id = str(event.payload.get("directive_id", ""))
            directive = applied.get(directive_id)
            if directive is None:
                return {
                    "node_error": (
                        f"directive_branch: no applied directive {directive_id!r}"
                    )
                }
            latest = directive
        if latest is None:
            return {}
        return {"latest_directive": latest}


def build_directive_payload(
    raw_text: str,
    directive_id: str,
    situation: SituationSnapshot,
    applied_directives: Sequence[ExpertDirective],
    *,
    model_id: str = _DEFAULT_MODEL_ID,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> dict[str, object]:
    """Curated directive parse payload: the raw text and known identifiers.

    IDs are sorted; only the fields the prompt may use are serialized — the
    expert instruction text, the deterministic directive id, the scenario's
    known target/group/UUV ids, and the already-applied directives — never
    the raw snapshot or hidden ground reality.
    """
    return {
        "model": model_id,
        "temperature": temperature,
        "system_prompt": DIRECTIVE_SYSTEM_PROMPT,
        "response_schema": "ExpertDirective",
        "scenario_id": situation.scenario_id,
        "sim_time_s": situation.sim_time_s,
        "directive_id": directive_id,
        "raw_text": raw_text,
        "known_target_ids": sorted(
            {report.target_id for report in situation.group_reports}
        ),
        "known_group_ids": sorted(
            {report.group_id for report in situation.group_reports}
        ),
        "known_uuv_ids": sorted({uuv.uuv_id for uuv in situation.uuvs}),
        "applied_directives": [
            directive.model_dump(mode="json") for directive in applied_directives
        ],
    }


def validate_directive(
    directive: ExpertDirective,
    *,
    situation: SituationSnapshot,
    applied_directives: Sequence[ExpertDirective] = (),
) -> ExpertDirective:
    """Resolve one directive's status deterministically (spec 10.1).

    Checks the named IDs against the live situation (targets, groups,
    UUVs), the resource bounds (finite priorities and quality minimums in
    [0, 1]), and hard-constraint conflicts against the applied directives.
    Any issue — or low confidence — resolves the directive to
    ``needs_clarification`` with the conflict list attached; a clean
    directive resolves to ``preview``. Never auto-applies.
    """
    issues = list(directive.conflicts)
    issues.extend(_id_and_resource_issues(directive, situation))
    issues.extend(_conflict_issues(directive, applied_directives))
    if not _has_any_constraint(directive):
        issues.append("ambiguous_scope: directive names no target or constraint")
    applicable = directive.confidence >= _APPLY_CONFIDENCE_MIN and not issues
    status = "preview" if applicable else "needs_clarification"
    return directive.model_copy(
        update={"status": status, "conflicts": tuple(issues)}
    )


def lock_group_members(
    *,
    directive_id: str,
    raw_text: str,
    target_scope: Sequence[str],
    target_id: str,
    member_ids: Sequence[str],
    confidence: float,
    situation: SituationSnapshot,
    applied_directives: Sequence[ExpertDirective] = (),
) -> ExpertDirective:
    """Typed shortcut: lock the members of the group tracking ``target_id``."""
    return validate_directive(
        ExpertDirective(
            directive_id=directive_id,
            raw_text=raw_text,
            target_scope=tuple(target_scope),
            locked_members={target_id: tuple(member_ids)},
            confidence=confidence,
            status="preview",
        ),
        situation=situation,
        applied_directives=applied_directives,
    )


def set_target_priority(
    *,
    directive_id: str,
    raw_text: str,
    target_scope: Sequence[str],
    target_id: str,
    priority: float,
    confidence: float,
    situation: SituationSnapshot,
    applied_directives: Sequence[ExpertDirective] = (),
) -> ExpertDirective:
    """Typed shortcut: pin the planning priority of ``target_id``."""
    return validate_directive(
        ExpertDirective(
            directive_id=directive_id,
            raw_text=raw_text,
            target_scope=tuple(target_scope),
            target_priorities={target_id: priority},
            confidence=confidence,
            status="preview",
        ),
        situation=situation,
        applied_directives=applied_directives,
    )


def set_minimum_quality(
    *,
    directive_id: str,
    raw_text: str,
    target_scope: Sequence[str],
    target_id: str,
    quality: float,
    confidence: float,
    situation: SituationSnapshot,
    applied_directives: Sequence[ExpertDirective] = (),
) -> ExpertDirective:
    """Typed shortcut: require at least ``quality`` tracking quality."""
    return validate_directive(
        ExpertDirective(
            directive_id=directive_id,
            raw_text=raw_text,
            target_scope=tuple(target_scope),
            minimum_quality={target_id: quality},
            confidence=confidence,
            status="preview",
        ),
        situation=situation,
        applied_directives=applied_directives,
    )


def disable_uuv(
    *,
    directive_id: str,
    raw_text: str,
    uuv_id: str,
    confidence: float,
    situation: SituationSnapshot,
    target_scope: Sequence[str] = (),
    applied_directives: Sequence[ExpertDirective] = (),
) -> ExpertDirective:
    """Typed shortcut: exclude one UUV from future assignments."""
    return validate_directive(
        ExpertDirective(
            directive_id=directive_id,
            raw_text=raw_text,
            target_scope=tuple(target_scope),
            disabled_uuv_ids=(uuv_id,),
            confidence=confidence,
            status="preview",
        ),
        situation=situation,
        applied_directives=applied_directives,
    )


def _has_any_constraint(directive: ExpertDirective) -> bool:
    return bool(
        directive.target_scope
        or directive.locked_members
        or directive.target_priorities
        or directive.minimum_quality
        or directive.disabled_uuv_ids
    )


def _id_and_resource_issues(
    directive: ExpertDirective, situation: SituationSnapshot
) -> list[str]:
    """Unknown IDs and out-of-bounds resources as deterministic issue strings."""
    issues: list[str] = []
    known_targets = {report.target_id for report in situation.group_reports}
    known_uuvs = {uuv.uuv_id for uuv in situation.uuvs}
    for target_id in sorted(set(directive.target_scope) - known_targets):
        issues.append(f"unknown_target {target_id!r}: no group report for it")
    for target_id, members in sorted(directive.locked_members.items()):
        if target_id not in known_targets:
            issues.append(f"unknown_target {target_id!r}: locked members group missing")
        for member_id in sorted(set(members) - known_uuvs):
            issues.append(f"unknown_member {member_id!r}: no resource state for it")
    for uuv_id in sorted(set(directive.disabled_uuv_ids) - known_uuvs):
        issues.append(f"unknown_uuv {uuv_id!r}: no resource state for it")
    for target_id, priority in sorted(directive.target_priorities.items()):
        if not math.isfinite(priority) or not 0.0 <= priority <= 1.0:
            issues.append(
                f"invalid_priority {target_id!r}: {priority!r} outside [0, 1]"
            )
    for target_id, quality in sorted(directive.minimum_quality.items()):
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            issues.append(
                f"invalid_quality {target_id!r}: {quality!r} outside [0, 1]"
            )
    locked = {
        member
        for members in directive.locked_members.values()
        for member in members
    }
    for uuv_id in sorted(set(directive.disabled_uuv_ids) & locked):
        issues.append(f"internal_conflict: uuv {uuv_id!r} is both locked and disabled")
    return issues


def _conflict_issues(
    directive: ExpertDirective, applied_directives: Sequence[ExpertDirective]
) -> list[str]:
    """Hard-constraint conflicts against the applied directives."""
    issues: list[str] = []
    for other in applied_directives:
        for target_id, members in sorted(directive.locked_members.items()):
            other_members = other.locked_members.get(target_id)
            if other_members is not None and set(other_members) != set(members):
                issues.append(
                    f"conflicts with applied {other.directive_id}: locked members"
                    f" of {target_id!r} differ"
                )
        for target_id, priority in sorted(directive.target_priorities.items()):
            if (
                target_id in other.target_priorities
                and other.target_priorities[target_id] != priority
            ):
                issues.append(
                    f"conflicts with applied {other.directive_id}: priority of"
                    f" {target_id!r} differs"
                )
        for target_id, quality in sorted(directive.minimum_quality.items()):
            if (
                target_id in other.minimum_quality
                and other.minimum_quality[target_id] != quality
            ):
                issues.append(
                    f"conflicts with applied {other.directive_id}: minimum quality"
                    f" of {target_id!r} differs"
                )
        locked_elsewhere = {
            member
            for members in other.locked_members.values()
            for member in members
        }
        for uuv_id in sorted(set(directive.disabled_uuv_ids) & locked_elsewhere):
            issues.append(
                f"conflicts with applied {other.directive_id}: uuv {uuv_id!r}"
                " is locked by it"
            )
    return issues
