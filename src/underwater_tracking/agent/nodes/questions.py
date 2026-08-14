# src/underwater_tracking/agent/nodes/questions.py
"""Evidence-backed expert questions with isolated counterfactuals (spec 10.2, plan Task 11).

The question branch is READ-ONLY: an expert question is answered from the
immutable planning snapshot — the DecisionLedger, the plan diffs, the
validation issues, and the observations resolved by evidence id — never by
invoking the carrier graph. The LLM receives a bounded curated payload of
structured reasons and evidence (never the model's hidden chain of thought),
and the answer is REJECTED when it cites evidence ids absent from the
payload (``QuestionEvidenceError``). An optional counterfactual is solved as
an isolated dry-run (``counterfactual.py``) whose ``dry-run:`` plan id is
stamped onto the answer.

``runtime.ask`` (spec 10.2's synchronous expert operation) calls
``answer_question`` and persists one ``question`` run under a deterministic
run id (canonical digest of the text and overrides, so re-asking the same
question dedupes against the ``question_runs`` PRIMARY KEY). The graph's
``question_branch`` node surfaces the latest question run id on the
``latest_question`` channel so conversation summaries have evidence
(plan Task 9 note: directive/question events feed the summary renderer).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from underwater_tracking.agent.counterfactual import (
    CounterfactualResult,
    run_counterfactual_dry_run,
)
from underwater_tracking.agent.llm import StructuredLLM
from underwater_tracking.agent.nodes.optimize import PlanningConfig
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.agent.prompts import (
    EXPLANATION_PROMPT_VERSION,
    EXPLANATION_SYSTEM_PROMPT,
    canonical_digest,
)
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import (
    DecisionRecord,
    PlanDiff,
    TrackingPlan,
    ValidationIssue,
)
from underwater_tracking.domain.models import SituationSnapshot, StrictModel
from underwater_tracking.persistence.events import EventRepository, StoredEvent
from underwater_tracking.persistence.ledger import DecisionLedger

# The question answering operation key (Mock LLM queue key, spec 22).
QUESTION_OPERATION = "question"
# Informational event emitted when an expert question is answered (spec 10.2).
QUESTION_EVENT_TYPE = "question"
_DEFAULT_MODEL_ID = "underwater-assistant-model"
_DEFAULT_TEMPERATURE = 0.2

# Bounded payload sizes (spec 10.2: structured reasons + evidence only).
DECISION_LIMIT = 10
EVIDENCE_ID_LIMIT = 50
TRIGGER_ID_LIMIT = 20
PLAN_ID_LIMIT = 10
REJECTION_LIMIT = 10
OBSERVATION_LIMIT = 100
ISSUE_LIMIT = 50
DIFF_LIMIT = 10

# Deterministic entity-matching tokens: T1..Tn targets, U1..Un UUVs, and
# ``{scenario}:plan:{revision}`` plan ids. ``UUV`` itself carries no digits,
# so the UUV pattern cannot false-positive on it.
_TARGET_TOKEN = re.compile(r"\bT[0-9]+\b")
_UUV_TOKEN = re.compile(r"\bU[0-9]+\b")
_PLAN_ID_TOKEN = re.compile(r"[A-Za-z0-9_-]+:plan:[0-9]+")


class QuestionAnswer(StrictModel):
    """One evidence-backed question answer (spec 10.2).

    ``evidence_ids`` is the subset of the payload's citable evidence ids the
    answer relies on; ``counterfactual_plan_id``/``counterfactual_summary``
    are stamped from the isolated dry-run when the question carries one.
    """

    answer: str
    evidence_ids: tuple[str, ...] = ()
    counterfactual_plan_id: str | None = None
    counterfactual_summary: str | None = None


class QuestionEvidenceError(ValueError):
    """Raised when an answer cites evidence ids absent from the payload."""


@dataclass(frozen=True)
class QuestionEntities:
    """The entities deterministically matched in the question text."""

    target_ids: tuple[str, ...]
    uuv_ids: tuple[str, ...]
    plan_ids: tuple[str, ...]


def match_question_entities(
    raw_text: str, situation: SituationSnapshot
) -> QuestionEntities:
    """Match ``Tn``/``Un``/``plan`` tokens in the text to known ids.

    Deterministic by construction: the text is scanned with fixed patterns
    and every match is validated against the live situation's known ids;
    unknown ids are dropped and the rest are sorted. An LLM is never asked
    to extract entities, so identical text always yields identical entities.
    """
    known_targets = frozenset(report.target_id for report in situation.group_reports)
    known_uuvs = frozenset(uuv.uuv_id for uuv in situation.uuvs)
    targets = tuple(
        sorted(
            {
                token
                for token in _TARGET_TOKEN.findall(raw_text)
                if token in known_targets
            }
        )
    )
    uuvs = tuple(
        sorted({token for token in _UUV_TOKEN.findall(raw_text) if token in known_uuvs})
    )
    plans = tuple(sorted(_PLAN_ID_TOKEN.findall(raw_text)))
    return QuestionEntities(target_ids=targets, uuv_ids=uuvs, plan_ids=plans)


@dataclass(frozen=True)
class QuestionEvidence:
    """The bounded evidence payload for one question.

    ``known_evidence_ids`` is the full set of ids the answer may cite: the
    decision records' input evidence ids and trigger event ids, plus the
    active plan's evidence ids. Any id outside this namespace in an answer
    is rejected.
    """

    known_evidence_ids: tuple[str, ...]
    observations: tuple[StoredEvent, ...]
    decisions: tuple[DecisionRecord, ...]
    plan_diffs: tuple[PlanDiff, ...]
    validation_issues: tuple[ValidationIssue, ...]


def retrieve_question_evidence(
    snapshot: PlanningSnapshot,
    ledger: DecisionLedger,
    events: EventRepository,
) -> QuestionEvidence:
    """Query the ledger, plan diffs, validation issues, and observations.

    The DecisionLedger is queried newest-first within a bounded window;
    plan diffs come from the recorded decisions' final diffs and the active
    plan's churn; validation issues are collected from the recorded
    verification reports; observations are resolved by evidence id from the
    event repository (the spec 9 evidence-id retrieval path).
    """
    scenario_id = snapshot.scenario_id
    decisions = tuple(ledger.list_decisions(scenario_id, limit=DECISION_LIMIT))
    active = snapshot.active_plan
    referenced: set[str] = set()
    for decision in decisions:
        referenced.update(decision.input_evidence_ids)
        referenced.update(decision.trigger_event_ids)
    if active is not None:
        referenced.update(active.evidence_ids)
    known = tuple(sorted(referenced))
    observations: list[StoredEvent] = []
    for event_id in known:
        observation = events.get(event_id)
        if observation is not None:
            observations.append(observation)
    issues: list[ValidationIssue] = []
    for decision in decisions:
        for verification in decision.verification_records:
            issues.extend(verification.issues)
    diffs: list[PlanDiff] = []
    for decision in decisions:
        if decision.final_plan_diff is not None:
            diffs.append(decision.final_plan_diff)
    if active is not None and active.diff is not None:
        diffs.append(active.diff)
    return QuestionEvidence(
        known_evidence_ids=known,
        observations=tuple(observations[:OBSERVATION_LIMIT]),
        decisions=decisions[:DECISION_LIMIT],
        plan_diffs=tuple(diffs[:DIFF_LIMIT]),
        validation_issues=tuple(issues[:ISSUE_LIMIT]),
    )


def build_question_payload(
    raw_text: str,
    entities: QuestionEntities,
    snapshot: PlanningSnapshot,
    evidence: QuestionEvidence,
    counterfactual: CounterfactualResult | None,
    *,
    model_id: str = _DEFAULT_MODEL_ID,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> dict[str, object]:
    """Curated question payload: structured reasons and evidence only.

    The payload is bounded: decisions, diffs, issues, and observations are
    truncated to the module limits, and id lists inside each decision are
    truncated as well. Raw snapshots, candidate internals, and any hidden
    model reasoning are never included (spec 10.2: the UI shows structured
    reasons + evidence, never the model's chain of thought).
    """
    return {
        "model": model_id,
        "temperature": temperature,
        "system_prompt": EXPLANATION_SYSTEM_PROMPT,
        "response_schema": "QuestionAnswer",
        "prompt_version": EXPLANATION_PROMPT_VERSION,
        "scenario_id": snapshot.scenario_id,
        "sim_time_s": snapshot.sim_time_s,
        "question_text": raw_text,
        "matched_entities": {
            "target_ids": list(entities.target_ids),
            "uuv_ids": list(entities.uuv_ids),
            "plan_ids": list(entities.plan_ids),
        },
        "active_plan": _render_plan(snapshot.active_plan),
        "decisions": [_render_decision(decision) for decision in evidence.decisions],
        "plan_diffs": [_render_diff(diff) for diff in evidence.plan_diffs],
        "validation_issues": [
            _render_issue(issue) for issue in evidence.validation_issues
        ],
        "observations": [_render_observation(obs) for obs in evidence.observations],
        "evidence_ids": list(evidence.known_evidence_ids),
        "counterfactual": (
            _render_counterfactual(counterfactual)
            if counterfactual is not None
            else None
        ),
    }


def validate_question_answer(
    answer: QuestionAnswer, known_evidence_ids: Sequence[str]
) -> None:
    """Reject an answer that cites evidence ids absent from the payload.

    ``QuestionEvidenceError`` is raised when the answer cites no evidence
    at all or cites any id outside the payload's ``evidence_ids`` namespace
    (the only citable ids), so the UI can never display an unsupported
    claim (spec 10.2).
    """
    known = frozenset(known_evidence_ids)
    cited = frozenset(answer.evidence_ids)
    missing = sorted(cited - known)
    if not cited or missing:
        raise QuestionEvidenceError(
            "answer cites evidence ids absent from the payload: "
            + (", ".join(missing) if missing else "no evidence cited")
        )


def question_run_id(
    scenario_id: str,
    raw_text: str,
    counterfactual: Mapping[str, object] | None = None,
) -> str:
    """Deterministic question-run id: canonical digest of text and overrides.

    The same question with the same overrides always maps to the same run
    id (``question_runs.run_id`` is a PRIMARY KEY), so re-asking dedupes.
    """
    digest = canonical_digest(
        {"text": raw_text, "counterfactual": dict(counterfactual or {})}
    )
    return f"{scenario_id}:question:{digest[:12]}"


def answer_question(
    *,
    raw_text: str,
    snapshot: PlanningSnapshot,
    ledger: DecisionLedger,
    events: EventRepository,
    llm: StructuredLLM[Any],
    counterfactual: Mapping[str, object] | None = None,
    model_id: str = _DEFAULT_MODEL_ID,
    temperature: float = _DEFAULT_TEMPERATURE,
    planning_config: PlanningConfig | None = None,
) -> QuestionAnswer:
    """Answer one expert question with evidence and an optional dry-run.

    The answer is served from the immutable planning snapshot and the
    bounded evidence payload; a counterfactual, when given, is solved as an
    isolated dry-run whose plan id is stamped onto the answer. The carrier
    graph is never invoked: the question branch is read-only (spec 10.2).
    """
    entities = match_question_entities(raw_text, snapshot.situation)
    evidence = retrieve_question_evidence(snapshot, ledger, events)
    dry_run = None
    if counterfactual is not None:
        dry_run = run_counterfactual_dry_run(
            snapshot, counterfactual, config=planning_config
        )
    payload = build_question_payload(
        raw_text,
        entities,
        snapshot,
        evidence,
        dry_run,
        model_id=model_id,
        temperature=temperature,
    )
    answer: QuestionAnswer = llm.invoke_structured(
        QUESTION_OPERATION,
        payload,
        QuestionAnswer,
        prompt_version=EXPLANATION_PROMPT_VERSION,
    )
    validate_question_answer(answer, evidence.known_evidence_ids)
    if dry_run is not None:
        answer = answer.model_copy(
            update={
                "counterfactual_plan_id": dry_run.plan_id,
                "counterfactual_summary": dry_run.summary,
            }
        )
    return answer


class _QuestionState(CarrierState, total=False):
    """Branch state: the carrier channels plus the deferred error marker.

    Mirrors ``CentralState`` in the carrier graph; the marker is defined
    locally so the node module never imports the graph (no circularity).
    """

    node_error: str | None


class QuestionBranchNode:
    """Carrier branch node surfacing the latest question run (spec 10.2).

    Runs between the directive branch and the three-tier routing: each
    ``question`` event in the cycle carries its run id, which is resolved
    against the ledger's question runs, and the latest resolved run id is
    set on the ``latest_question`` channel. An event referencing an unknown
    run defers a node error so the cycle completes via ``handle_error``
    instead of crashing.
    """

    def __init__(self, ledger: DecisionLedger) -> None:
        self._ledger = ledger

    def __call__(self, state: _QuestionState) -> _QuestionState:
        scenario_id = state.get("scenario_id")
        if not scenario_id:
            return {"node_error": "question_branch requires scenario_id in state"}
        known = {
            run.run_id for run in self._ledger.list_questions(scenario_id)
        }
        latest: str | None = None
        for event in state.get("coalesced_events") or ():
            if event.event_type != QUESTION_EVENT_TYPE:
                continue
            run_id = str(event.payload.get("run_id", ""))
            if run_id not in known:
                return {"node_error": f"question_branch: no question run {run_id!r}"}
            latest = run_id
        if latest is None:
            return {}
        return {"latest_question": latest}


def _render_plan(plan: TrackingPlan | None) -> dict[str, object] | None:
    """The active plan's stable identity, members, and expected outcomes."""
    if plan is None:
        return None
    return {
        "plan_id": plan.plan_id,
        "revision": plan.revision,
        "status": plan.status,
        "concept": plan.concept,
        "member_ids_by_target": {
            target: list(members)
            for target, members in sorted(plan.member_ids_by_target.items())
        },
        "predicted_quality": plan.predicted_quality,
        "predicted_active_count": plan.predicted_active_count,
        "predicted_energy": plan.predicted_energy,
        "diff": plan.diff.model_dump(mode="json") if plan.diff is not None else None,
    }


def _render_decision(decision: DecisionRecord) -> dict[str, object]:
    """One ledger decision, bounded to the fields the prompt may use."""
    return {
        "decision_id": decision.decision_id,
        "sim_time_s": decision.sim_time_s,
        "snapshot_revision": decision.snapshot_revision,
        "trigger_event_ids": _bounded(decision.trigger_event_ids, TRIGGER_ID_LIMIT),
        "input_evidence_ids": _bounded(decision.input_evidence_ids, EVIDENCE_ID_LIMIT),
        "candidates": [proposal.concept for proposal in decision.candidates],
        "candidate_plan_ids": _bounded(decision.candidate_plan_ids, PLAN_ID_LIMIT),
        "rejected_candidates": {
            key: value
            for key, value in list(decision.rejected_candidates.items())[
                :REJECTION_LIMIT
            ]
        },
        "solver_metrics": (
            decision.solver_metrics.model_dump(mode="json")
            if decision.solver_metrics is not None
            else None
        ),
        "final_plan_id": decision.final_plan_id,
        "final_plan_diff": (
            decision.final_plan_diff.model_dump(mode="json")
            if decision.final_plan_diff is not None
            else None
        ),
    }


def _render_diff(diff: PlanDiff) -> dict[str, object]:
    """One plan churn (added/removed members and waypoint changes)."""
    return {
        "from_plan_id": diff.from_plan_id,
        "from_revision": diff.from_revision,
        "to_plan_id": diff.to_plan_id,
        "to_revision": diff.to_revision,
        "members_added": {
            target: list(members)
            for target, members in sorted(diff.members_added.items())
        },
        "members_removed": {
            target: list(members)
            for target, members in sorted(diff.members_removed.items())
        },
        "waypoints_changed": list(diff.waypoints_changed),
        "summary": diff.summary,
    }


def _render_issue(issue: ValidationIssue) -> dict[str, object]:
    """One validation issue (code, field, message)."""
    return {
        "code": issue.code,
        "field": issue.field,
        "message": issue.message,
        "observed": issue.observed,
        "expected": issue.expected,
    }


def _render_observation(observation: StoredEvent) -> dict[str, object]:
    """One stored event resolved by evidence id."""
    return {
        "event_id": observation.event_id,
        "event_type": observation.event_type,
        "sim_time_s": observation.sim_time_s,
        "target_id": observation.target_id,
        "severity": observation.severity,
        "payload": observation.payload,
    }


def _render_counterfactual(result: CounterfactualResult) -> dict[str, object]:
    """The isolated dry-run: its plan, churn, objective, and summary."""
    objective = result.objective
    return {
        "run_id": result.run_id,
        "plan_id": result.plan_id,
        "summary": result.summary,
        "diff": result.diff.model_dump(mode="json") if result.diff is not None else None,
        "objective": {
            "active_count_before": objective.active_count_before,
            "active_count_after": objective.active_count_after,
            "economic_cost_before": objective.economic_cost_before,
            "economic_cost_after": objective.economic_cost_after,
        },
    }


def _bounded(values: Sequence[str], limit: int) -> tuple[str, ...]:
    """Truncate an id list to the payload limit (deterministic prefix)."""
    return tuple(values[:limit])
