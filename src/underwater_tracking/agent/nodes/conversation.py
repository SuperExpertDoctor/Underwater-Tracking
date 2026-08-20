"""Routing for the unified expert conversation client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from underwater_tracking.agent.llm import StructuredLLM
from underwater_tracking.agent.nodes.directives import (
    directive_preview_diff,
    validate_directive,
)
from underwater_tracking.agent.nodes.questions import (
    QuestionAnswer,
    answer_question,
    retrieve_question_evidence,
    validate_conversation_evidence_ids,
)
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot, build_planning_snapshot
from underwater_tracking.domain.agent_models import ExpertDirective, TrackingPlan
from underwater_tracking.domain.conversation_models import (
    ConversationAnswer,
    ConversationClassification,
    ConversationMessage,
    ConversationProposal,
    ConversationTurnResult,
)
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger

CONVERSATION_OPERATION = "conversation_classification"
CONVERSATION_PROMPT_VERSION = "conversation-v1"


@dataclass(frozen=True)
class ConversationContext:
    """Dependencies needed by a single conversation turn."""

    scenario_id: str
    situation: SituationSnapshot
    active_plan: TrackingPlan | None
    ledger: DecisionLedger
    events: EventRepository
    llm: StructuredLLM[Any]
    model_id: str = "underwater-assistant-model"
    planning_config: Any | None = None


def build_classification_payload(
    message: ConversationMessage,
    context: ConversationContext,
) -> dict[str, object]:
    return {
        "conversation_id": message.conversation_id,
        "message_id": message.message_id,
        "text": message.text,
        "expected_plan_version": message.expected_plan_version,
        "target_scope": list(message.target_scope),
        "region_scope": list(message.region_scope),
        "classifications": ["plan_revision", "evidence_query", "mixed", "clarification"],
        "routing_rules": {
            "plan_revision": "return a preview only; never apply",
            "evidence_query": "read-only answer; do not emit an event",
            "mixed": "return independent preview and evidence results; never apply",
            "clarification": "return exactly one clarification question",
        },
        "scenario_id": context.scenario_id,
        "sim_time_s": context.situation.sim_time_s,
    }


def process_conversation_message(
    message: ConversationMessage,
    context: ConversationContext,
) -> ConversationTurnResult:
    """Classify and route one message without applying a plan proposal."""
    current_version = context.active_plan.revision if context.active_plan else 0
    if message.expected_plan_version != current_version:
        raise ValueError(
            "conversation plan version mismatch: "
            f"expected {message.expected_plan_version}, current {current_version}"
        )
    classification = context.llm.invoke_structured(
        CONVERSATION_OPERATION,
        build_classification_payload(message, context),
        ConversationClassification,
        prompt_version=CONVERSATION_PROMPT_VERSION,
    )
    if (
        classification.expected_plan_version is not None
        and classification.expected_plan_version != message.expected_plan_version
    ):
        raise ValueError(
            "conversation classifier returned a different expected plan version"
        )

    evidence = retrieve_question_evidence(
        build_planning_snapshot(context.situation, active_plan=context.active_plan),
        context.ledger,
        context.events,
    )
    if classification.evidence_ids:
        validate_conversation_evidence_ids(
            classification.evidence_ids, evidence.known_evidence_ids
        )
    if classification.classification in {"plan_revision", "mixed"}:
        proposal = _preview_proposal(classification.proposal, context)
    else:
        proposal = None

    if classification.classification in {"evidence_query", "mixed"}:
        answer = _answer_read_only(message.text, context, evidence)
    else:
        answer = None

    turn_id = f"{message.conversation_id}:turn:{message.message_id}"
    expert_message = message.model_copy(update={"turn_id": turn_id})
    messages = [expert_message]
    if classification.classification == "clarification":
        follow_up = classification.clarification_question or "请说明需要调整的目标或区域。"
        messages.append(
            _assistant_message(
                message,
                turn_id,
                follow_up,
                classification="clarification",
            )
        )
        return ConversationTurnResult(
            conversation_id=message.conversation_id,
            turn_id=turn_id,
            user_id=message.user_id,
            assistant_mode=message.assistant_mode,
            classification=classification,
            messages=tuple(messages),
            target_scope=classification.target_scope,
            region_scope=classification.region_scope,
            clarification_question=follow_up,
            expected_plan_version=message.expected_plan_version,
        )

    if proposal is not None:
        messages.append(
            _assistant_message(
                message,
                turn_id,
                "已生成方案预览，请确认后应用。",
                classification="plan_revision",
                proposal=proposal,
            )
        )
    if answer is not None:
        messages.append(
            _assistant_message(
                message,
                turn_id,
                answer.answer,
                classification="evidence_query",
                evidence_ids=answer.evidence_ids,
            )
        )
    return ConversationTurnResult(
        conversation_id=message.conversation_id,
        turn_id=turn_id,
        user_id=message.user_id,
        assistant_mode=message.assistant_mode,
        classification=classification,
        messages=tuple(messages),
        target_scope=classification.target_scope,
        region_scope=classification.region_scope,
        evidence_ids=classification.evidence_ids,
        proposal=proposal,
            answer=ConversationAnswer.model_validate(answer.model_dump(mode="json"))
            if answer
            else None,
            expected_plan_version=message.expected_plan_version,
        )


def _preview_proposal(
    proposal: ExpertDirective | None,
    context: ConversationContext,
) -> ConversationProposal:
    if proposal is None:
        raise ValueError("plan revision classification did not include a proposal")
    applied = context.ledger.list_directives(context.scenario_id, status="applied")
    directive = validate_directive(
        proposal,
        situation=context.situation,
        applied_directives=tuple(applied),
    )
    return ConversationProposal(
        proposal_id=directive.directive_id,
        directive=directive,
        expected_plan_version=context.active_plan.revision if context.active_plan else 0,
        summary=directive.raw_text,
        diff=directive_preview_diff(directive),
        status=directive.status,
    )


def _answer_read_only(
    raw_text: str,
    context: ConversationContext,
    evidence: Any,
) -> QuestionAnswer:
    del evidence
    snapshot: PlanningSnapshot = build_planning_snapshot(
        context.situation,
        active_plan=context.active_plan,
    )
    # answer_question performs evidence validation but does not persist a run
    # or enqueue an event. The runtime owns persistence for the legacy /api/questions
    # path; the unified conversation path remains read-only.
    return answer_question(
        raw_text=raw_text,
        snapshot=snapshot,
        ledger=context.ledger,
        events=context.events,
        llm=context.llm,
        model_id=context.model_id,
        planning_config=context.planning_config,
    )


def _assistant_message(
    source: ConversationMessage,
    turn_id: str,
    text: str,
    *,
    classification: str,
    evidence_ids: tuple[str, ...] = (),
    proposal: ConversationProposal | None = None,
) -> ConversationMessage:
    return ConversationMessage(
        conversation_id=source.conversation_id,
        message_id=f"{turn_id}:{classification}",
        turn_id=turn_id,
        user_id=source.user_id,
        assistant_mode=source.assistant_mode,
        role="assistant",
        text=text,
        classification=classification,  # type: ignore[arg-type]
        target_scope=source.target_scope,
        region_scope=source.region_scope,
        evidence_ids=evidence_ids,
        proposal=proposal,
        expected_plan_version=source.expected_plan_version,
    )
