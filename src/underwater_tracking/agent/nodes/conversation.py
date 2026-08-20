"""Routing for the unified expert conversation client."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

from underwater_tracking.agent.llm import LLMConfigError, StructuredLLM
from underwater_tracking.agent.nodes.directives import (
    DIRECTIVE_OPERATION,
    directive_preview_diff,
    build_directive_payload,
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
    AssistantMode,
    ConversationAnswer,
    ConversationClassification,
    ConversationMessage,
    ConversationProposal,
    ConversationTurnResult,
)
from underwater_tracking.domain.memory_models import (
    MemoryContext,
    MemoryEvidenceTrace,
    MemoryStreamStatus,
)
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.memory.service import MemoryService
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.memory import ShortTermContextRepository
from underwater_tracking.persistence.plans import PlanRepository

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
    memory_service: MemoryService | None = None
    memory_context: MemoryContext | None = None
    user_id: str = "operator"
    assistant_mode: AssistantMode = "auto"
    plans: PlanRepository | None = None
    short_term_repository: ShortTermContextRepository | None = None
    conversation_id: str | None = None
    model_id: str = "underwater-assistant-model"
    planning_config: Any | None = None


def build_classification_payload(
    message: ConversationMessage,
    context: ConversationContext,
) -> dict[str, object]:
    memory_context = context.memory_context
    short_term = (
        memory_context.short_term_context.model_dump(mode="json")
        if memory_context is not None and memory_context.short_term_context is not None
        else None
    )
    long_term = (
        [
            {
                "memory_id": hit.memory.memory_id,
                "memory_type": hit.memory.memory_type.value,
                "summary": hit.memory.summary,
                "similarity_score": hit.similarity_score,
                "rerank_score": hit.rerank_score,
                "retrieval_reason": hit.retrieval_reason,
            }
            for hit in (memory_context.long_term_material[:8] if memory_context else ())
        ]
    )
    return {
        "conversation_id": message.conversation_id,
        "message_id": message.message_id,
        "text": message.text,
        "expected_plan_version": message.expected_plan_version,
        "target_scope": list(message.target_scope),
        "region_scope": list(message.region_scope),
        "user_id": context.user_id,
        "assistant_mode": context.assistant_mode,
        "classifications": ["plan_revision", "evidence_query", "mixed", "clarification"],
        "routing_rules": {
            "plan_revision": "return a preview only; never apply",
            "evidence_query": "read-only answer; do not emit an event",
            "mixed": "return independent preview and evidence results; never apply",
            "clarification": "return exactly one clarification question",
        },
        "scenario_id": context.scenario_id,
        "sim_time_s": context.situation.sim_time_s,
        "current_situation": context.situation.model_dump(mode="json"),
        "current_plan": (
            context.active_plan.model_dump(mode="json")
            if context.active_plan is not None
            else None
        ),
        "short_term_context": short_term,
        "long_term_material": long_term,
        "long_term_material_is_not_fact": True,
        "memory_status": (
            memory_context.memory_status.value
            if memory_context is not None
            else MemoryStreamStatus.DEGRADED.value
        ),
    }


def process_conversation_message(
    message: ConversationMessage,
    context: ConversationContext,
) -> ConversationTurnResult:
    """Classify and route one message without applying a plan proposal."""
    memory_context = _prepare_context(message, context)
    context = replace(
        context,
        memory_context=memory_context,
        user_id=message.user_id,
        assistant_mode=message.assistant_mode,
    )
    current_version = context.active_plan.revision if context.active_plan else 0
    if message.expected_plan_version != current_version:
        raise ValueError(
            "conversation plan version mismatch: "
            f"expected {message.expected_plan_version}, current {current_version}"
        )
    try:
        classification = context.llm.invoke_structured(
            CONVERSATION_OPERATION,
            build_classification_payload(message, context),
            ConversationClassification,
            prompt_version=CONVERSATION_PROMPT_VERSION,
        )
    except LLMConfigError as exc:
        return _degraded_turn(context, message, memory_context, str(exc))
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
    memory_context = _verify_memory_sources(context, memory_context)
    context = replace_context(context, memory_context=memory_context)
    candidate_evidence_ids = tuple(
        dict.fromkeys(
            (*evidence.known_evidence_ids, *_verified_source_ids(memory_context))
        )
    )
    if classification.evidence_ids:
        validate_conversation_evidence_ids(
            classification.evidence_ids, candidate_evidence_ids
        )
    if classification.memory_ids:
        unknown_memory_ids = set(classification.memory_ids) - set(
            memory_context.retrieved_memory_ids
        )
        if unknown_memory_ids:
            raise ValueError(
                "unknown memory id(s): " + ", ".join(sorted(unknown_memory_ids))
            )
    if classification.classification in {"plan_revision", "mixed"}:
        proposal = _preview_proposal(_think_about_plan(message, context), context)
    else:
        proposal = None

    if classification.classification in {"evidence_query", "mixed"}:
        answer = _answer_read_only(
            message.text,
            context,
            evidence,
            allowed_evidence_ids=_verified_source_ids(memory_context)
            if memory_context.retrieved_memory_ids
            else None,
        )
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
        result = ConversationTurnResult(
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
            memory_context=memory_context,
        )
        return _accept_turn(context, message, result)

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
    result = ConversationTurnResult(
        conversation_id=message.conversation_id,
        turn_id=turn_id,
        user_id=message.user_id,
        assistant_mode=message.assistant_mode,
        classification=classification,
        messages=tuple(messages),
        target_scope=classification.target_scope,
        region_scope=classification.region_scope,
        evidence_ids=(
            tuple(answer.evidence_ids)
            if answer is not None
            else tuple(classification.evidence_ids)
        ),
        proposal=proposal,
        answer=ConversationAnswer.model_validate(answer.model_dump(mode="json"))
        if answer
        else None,
        expected_plan_version=message.expected_plan_version,
        memory_context=memory_context,
    )
    return _accept_turn(context, message, result)


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
    context.ledger.save_directive(directive, context.scenario_id)
    return ConversationProposal(
        proposal_id=directive.directive_id,
        directive=directive,
        expected_plan_version=context.active_plan.revision if context.active_plan else 0,
        summary=directive.raw_text,
        diff=directive_preview_diff(directive),
        status=directive.status,
    )


def replace_context(
    context: ConversationContext, *, memory_context: MemoryContext
) -> ConversationContext:
    """Copy request-scoped memory state without mutating shared dependencies."""
    return replace(context, memory_context=memory_context)


def _prepare_context(
    message: ConversationMessage, context: ConversationContext
) -> MemoryContext:
    if context.memory_service is None:
        if (
            context.memory_context is not None
            and context.memory_context.user_id != message.user_id
        ):
            return MemoryContext(
                user_id=message.user_id,
                memory_status=MemoryStreamStatus.DEGRADED,
            )
        return context.memory_context or MemoryContext(user_id=message.user_id)
    try:
        prepared = context.memory_service.prepare_context(
            message.user_id,
            message.conversation_id,
            message.text,
            filters={"assistant_mode": message.assistant_mode},
            scenario_id=context.scenario_id,
        )
        if prepared.user_id != message.user_id:
            return MemoryContext(
                user_id=message.user_id,
                memory_status=MemoryStreamStatus.DEGRADED,
            )
        return prepared
    except Exception:
        # Memory is explicitly non-blocking for the foreground plan/evidence path.
        return MemoryContext(user_id=message.user_id, memory_status=MemoryStreamStatus.DEGRADED)


def _degraded_turn(
    context: ConversationContext,
    message: ConversationMessage,
    memory_context: MemoryContext,
    reason: str,
) -> ConversationTurnResult:
    """Persist the original turn without inventing a chat response."""
    expected_plan_version = message.expected_plan_version
    assert expected_plan_version is not None
    turn_id = f"{message.conversation_id}:turn:{message.message_id}"
    degraded_context = memory_context.model_copy(
        update={
            "memory_status": MemoryStreamStatus.DEGRADED,
            "degraded_reason": reason,
        }
    )
    result = ConversationTurnResult(
        conversation_id=message.conversation_id,
        turn_id=turn_id,
        user_id=message.user_id,
        assistant_mode=message.assistant_mode,
        classification=ConversationClassification(
            classification="clarification",
            confidence=0.0,
            expected_plan_version=expected_plan_version,
        ),
        messages=(message.model_copy(update={"turn_id": turn_id}),),
        expected_plan_version=expected_plan_version,
        memory_context=degraded_context,
    )
    return _accept_turn(context, message, result)


def _think_about_plan(
    message: ConversationMessage, context: ConversationContext
) -> ExpertDirective:
    applied = context.ledger.list_directives(context.scenario_id, status="applied")
    payload = build_directive_payload(
        message.text,
        f"{message.conversation_id}:directive:{message.message_id}",
        context.situation,
        applied,
        model_id=context.model_id,
    )
    classification_payload = build_classification_payload(message, context)
    for key in (
        "user_id",
        "assistant_mode",
        "current_plan",
        "short_term_context",
        "long_term_material",
        "long_term_material_is_not_fact",
        "memory_status",
    ):
        payload[key] = classification_payload[key]
    return cast(
        ExpertDirective,
        context.llm.invoke_structured(
            DIRECTIVE_OPERATION,
            payload,
            ExpertDirective,
            prompt_version="directive-v1",
        ),
    )


def _verify_memory_sources(
    context: ConversationContext, memory_context: MemoryContext
) -> MemoryContext:
    scoped_hits = tuple(
        hit
        for hit in memory_context.long_term_material
        if hit.memory.scenario_id == context.scenario_id
    )
    candidate_memory_ids = tuple(hit.memory.memory_id for hit in scoped_hits)
    traces: list[MemoryEvidenceTrace] = []
    knowledge_runs = {
        run.query_id: run
        for run in context.ledger.list_knowledge_queries(context.scenario_id)
    }
    for hit in scoped_hits:
        memory = hit.memory
        source_event_ids: list[str] = []
        source_decision_ids: list[str] = []
        source_knowledge_ids: list[str] = []
        source_plan_ids: list[str] = []
        source_message_ids: list[str] = []
        if context.short_term_repository is not None and context.conversation_id is not None:
            source_message_ids.extend(
                message.message_id
                for message in context.short_term_repository.get_messages(
                    memory_context.user_id,
                    context.conversation_id,
                    memory.source_message_ids,
                    scenario_id=context.scenario_id,
                )
            )
        for source_id in memory.source_event_ids:
            source = context.events.get(source_id)
            if source is not None and source.scenario_id == context.scenario_id:
                source_event_ids.append(source_id)
        for source_id in memory.source_decision_ids:
            decision = context.ledger.get(source_id)
            if decision is not None and decision.scenario_id == context.scenario_id:
                source_decision_ids.append(source_id)
        for source_id in memory.source_knowledge_ids:
            query = knowledge_runs.get(source_id)
            response = query.response if query is not None else None
            answer = response.get("answer") if isinstance(response, dict) else None
            if (
                query is not None
                and query.status == "completed"
                and isinstance(answer, str)
                and bool(answer.strip())
            ):
                source_knowledge_ids.append(source_id)
        if context.plans is not None:
            for source_id in memory.source_plan_ids:
                plan = context.plans.get_plan(source_id)
                if (
                    plan is not None
                    and plan.scenario_id == context.scenario_id
                    and plan.status in {"active", "degraded"}
                ):
                    source_plan_ids.append(source_id)
        supplied_source_count = sum(
            len(source_ids)
            for source_ids in (
                memory.source_message_ids,
                memory.source_event_ids,
                memory.source_decision_ids,
                memory.source_knowledge_ids,
                memory.source_plan_ids,
            )
        )
        verified_source_count = sum(
            len(source_ids)
            for source_ids in (
                source_message_ids,
                source_event_ids,
                source_decision_ids,
                source_knowledge_ids,
                source_plan_ids,
            )
        )
        status = (
            MemoryStreamStatus.COMPLETED
            if supplied_source_count > 0 and verified_source_count == supplied_source_count
            else MemoryStreamStatus.DEGRADED
        )
        traces.append(
            MemoryEvidenceTrace(
                trace_id=f"{memory.memory_id}:evidence",
                user_id=memory_context.user_id,
                status=status,
                memory_ids=(memory.memory_id,),
                source_message_ids=tuple(source_message_ids),
                source_event_ids=tuple(source_event_ids),
                source_decision_ids=tuple(source_decision_ids),
                source_knowledge_ids=tuple(source_knowledge_ids),
                source_plan_ids=tuple(source_plan_ids),
            )
        )
    overall_status = (
        MemoryStreamStatus.DEGRADED
        if any(trace.status is MemoryStreamStatus.DEGRADED for trace in traces)
        else memory_context.memory_status
    )
    return memory_context.model_copy(
        update={
            "retrieved_memory_ids": candidate_memory_ids,
            "evidence_trace": tuple(traces),
            "memory_status": overall_status,
        }
    )


def _verified_source_ids(memory_context: MemoryContext) -> tuple[str, ...]:
    source_ids: list[str] = []
    for trace in memory_context.evidence_trace:
        if trace.status is not MemoryStreamStatus.COMPLETED:
            continue
        source_ids.extend(
            (
                *trace.source_event_ids,
                *trace.source_decision_ids,
                *trace.source_knowledge_ids,
                *trace.source_plan_ids,
            )
        )
    return tuple(dict.fromkeys(source_ids))


def _accept_turn(
    context: ConversationContext,
    message: ConversationMessage,
    result: ConversationTurnResult,
) -> ConversationTurnResult:
    if context.memory_service is None:
        return result
    turn_payload = message.model_dump(mode="json")
    turn_payload["scenario_id"] = context.scenario_id
    outcome = context.memory_service.accept_turn(
        turn_payload,
        tuple(item.model_dump(mode="json") for item in result.messages),
        source_refs=result.evidence_ids,
    )
    cursor = outcome.get("stream_cursor")
    return result.model_copy(
        update={
            "queued_memory_work_id": outcome.get("work_id"),
            "memory_stream_cursor": cursor if isinstance(cursor, int) else None,
        }
    )


def _answer_read_only(
    raw_text: str,
    context: ConversationContext,
    evidence: Any,
    *,
    allowed_evidence_ids: tuple[str, ...] | None = None,
) -> QuestionAnswer:
    snapshot: PlanningSnapshot = build_planning_snapshot(
        context.situation,
        active_plan=context.active_plan,
    )
    # answer_question performs evidence validation but does not persist a run
    # or enqueue an event. The runtime owns persistence for the legacy /api/questions
    # path; the unified conversation path remains read-only.
    answer = answer_question(
        raw_text=raw_text,
        snapshot=snapshot,
        ledger=context.ledger,
        events=context.events,
        llm=context.llm,
        model_id=context.model_id,
        planning_config=context.planning_config,
        memory_context=context.memory_context,
        allowed_evidence_ids=allowed_evidence_ids,
    )
    if context.memory_context is not None and context.memory_context.retrieved_memory_ids:
        answer = answer.model_copy(
            update={"memory_ids": context.memory_context.retrieved_memory_ids}
        )
        if any(
            trace.status is MemoryStreamStatus.DEGRADED
            for trace in context.memory_context.evidence_trace
        ):
            answer = answer.model_copy(
                update={
                    "answer": answer.answer.rstrip()
                    + "（记忆线索存在、原始证据不足）"
                }
            )
    del evidence
    return answer


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
