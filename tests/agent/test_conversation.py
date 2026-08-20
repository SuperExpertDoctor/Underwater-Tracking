"""Conversation routing tests for the unified expert assistant."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from underwater_tracking.agent.llm import StructuredLLM
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.agent.nodes.conversation import (
    ConversationContext,
    _verify_memory_sources,
    process_conversation_message,
)
from underwater_tracking.agent.nodes.questions import QuestionAnswer
from underwater_tracking.domain.agent_models import ExpertDirective, TrackingPlan
from underwater_tracking.domain.conversation_models import (
    ConversationClassification,
    ConversationMessage,
    ConversationTurnResult,
)
from underwater_tracking.domain.memory_models import (
    MemoryContext,
    MemoryRetrievalHit,
    MemoryStreamStatus,
    MemoryVersion,
    MemoryType,
    ShortTermMessage,
)
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.memory.service import MemoryService
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.memory import LongTermMemoryRepository, ShortTermContextRepository


class RecordingLLM(StructuredLLM[Any]):
    def __init__(
        self,
        classification: ConversationClassification,
        answer: QuestionAnswer | None = None,
    ) -> None:
        self.classification = classification
        self.answer = answer
        self.calls: list[str] = []
        self.payloads: list[dict[str, object]] = []

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        del prompt_version
        self.calls.append(operation)
        self.payloads.append(payload)
        if response_model is ConversationClassification:
            return self.classification
        if response_model is QuestionAnswer:
            return self.answer or QuestionAnswer(answer="当前证据不足以支持进一步判断。")
        if response_model is ExpertDirective:
            assert self.classification.proposal is not None
            return self.classification.proposal
        raise AssertionError(f"unexpected response model {response_model!r}")


class RecordingMemoryService:
    def __init__(self, context: MemoryContext) -> None:
        self.context = context
        self.calls: list[tuple[str, str, str]] = []
        self.accepted: list[tuple[object, object, tuple[str, ...]]] = []

    def prepare_context(
        self,
        user_id: str,
        conversation_id: str,
        query: str,
        filters: object | None = None,
        scenario_id: str | None = None,
    ) -> MemoryContext:
        del filters, scenario_id
        self.calls.append((user_id, conversation_id, query))
        return self.context

    def accept_turn(
        self,
        turn: object,
        result: object,
        source_refs: tuple[str, ...] = (),
    ) -> dict[str, object]:
        self.accepted.append((turn, result, source_refs))
        return {"status": "queued", "work_id": "memory-work-1", "stream_cursor": 7}


@dataclass
class ConversationRig:
    context: ConversationContext
    llm: RecordingLLM
    events: EventRepository

    def close(self) -> None:
        self.events.close()
        self.context.ledger.close()


def make_rig(
    tmp_path: Path,
    classification: ConversationClassification,
    answer: QuestionAnswer | None = None,
) -> ConversationRig:
    database_path = tmp_path / "conversation.db"
    events = EventRepository(database_path)
    ledger = DecisionLedger(database_path)
    llm = RecordingLLM(classification, answer=answer)
    situation = SituationSnapshot.model_construct(
        scenario_id="S1",
        snapshot_revision=3,
        sim_time_s=900,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )
    return ConversationRig(
        context=ConversationContext(
            scenario_id="S1",
            situation=situation,
            active_plan=None,
            ledger=ledger,
            events=events,
            llm=llm,
        ),
        llm=llm,
        events=events,
    )


def message(text: str, *, expected_plan_version: int = 0) -> ConversationMessage:
    return ConversationMessage(
        conversation_id="conversation-1",
        message_id="message-1",
        role="expert",
        text=text,
        expected_plan_version=expected_plan_version,
        target_scope=("T1",),
        region_scope=("region_1",),
    )


def feedback_proposal() -> ExpertDirective:
    return ExpertDirective(
        directive_id="classifier-proposal",
        raw_text="",
        target_scope=(),
        directive_type="feedback",
        feedback_region_ids=("region_1",),
        feedback_text="增加 region_1 的接力余量",
        confidence=0.95,
        status="preview",
    )


def classification(
    kind: str,
    *,
    proposal: ExpertDirective | None = None,
    evidence_ids: tuple[str, ...] = (),
) -> ConversationClassification:
    return ConversationClassification(
        classification=kind,
        confidence=0.95,
        target_scope=(),
        region_scope=("region_1",),
        evidence_ids=evidence_ids,
        proposal=proposal,
        expected_plan_version=0,
        clarification_question="请说明需要调整的目标或区域。" if kind == "clarification" else None,
    )


def test_plan_revision_only_returns_a_preview_without_auto_apply(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, classification("plan_revision", proposal=feedback_proposal()))
    try:
        result = process_conversation_message(message("增加 region_1 的接力余量"), rig.context)

        assert result.classification.classification == "plan_revision"
        assert result.proposal is not None
        assert result.proposal.status == "preview"
        assert result.applied is False
        assert rig.events.list_events(scenario_id="S1", event_type="directive_applied") == []
        assert rig.context.ledger.list_directives("S1", status="preview")
        assert rig.llm.calls == ["conversation_classification", "directive"]
    finally:
        rig.close()


def test_evidence_query_is_read_only_and_emits_no_event(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, classification("evidence_query"))
    try:
        result = process_conversation_message(message("为什么保持当前方案？"), rig.context)

        assert result.classification.classification == "evidence_query"
        assert result.proposal is None
        assert len(result.messages) == 2
        assert result.messages[-1].role == "assistant"
        assert rig.events.list_events(scenario_id="S1") == []
        assert rig.llm.calls == ["conversation_classification", "question"]
    finally:
        rig.close()


def test_conversation_prepares_memory_before_classification_and_queues_after_response(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path, classification("plan_revision", proposal=feedback_proposal()))
    memory = RecordingMemoryService(
        MemoryContext(user_id="operator", memory_status=MemoryStreamStatus.COMPLETED)
    )
    rig.context = rig.context.__class__(
        **{
            **rig.context.__dict__,
            "memory_service": memory,
        }
    )
    try:
        result = process_conversation_message(message("增加 region_1 的接力余量"), rig.context)

        assert memory.calls == [("operator", "conversation-1", "增加 region_1 的接力余量")]
        assert len(memory.accepted) == 1
        assert result.queued_memory_work_id == "memory-work-1"
        assert result.memory_stream_cursor == 7
        assert result.memory_context is not None
        assert result.memory_context.memory_status is MemoryStreamStatus.COMPLETED
    finally:
        rig.close()


def test_classification_payload_marks_long_term_material_as_non_factual(tmp_path: Path) -> None:
    from underwater_tracking.agent.nodes.conversation import build_classification_payload

    rig = make_rig(tmp_path, classification("plan_revision", proposal=feedback_proposal()))
    memory = MemoryVersion(
        memory_id="memory-1",
        memory_family_id="family-1",
        version=1,
        user_id="operator",
        memory_type=MemoryType.SEMANTIC,
        summary="historical preference",
        importance_score=0.8,
        embedding=(1.0,),
    )
    context = rig.context.__class__(
        **{
            **rig.context.__dict__,
            "memory_context": MemoryContext(
                user_id="operator",
                long_term_material=(
                    MemoryRetrievalHit(
                        memory=memory,
                        similarity_score=0.9,
                        rerank_score=0.8,
                        retrieval_reason="semantic match",
                    ),
                ),
                retrieved_memory_ids=("memory-1",),
                memory_status=MemoryStreamStatus.COMPLETED,
            ),
        }
    )
    try:
        payload = build_classification_payload(message("请基于当前方案调整"), context)
        assert payload["assistant_mode"] == "auto"
        assert payload["current_situation"]
        assert payload["short_term_context"] is None
        assert payload["long_term_material"][0]["memory_id"] == "memory-1"  # type: ignore[index]
        assert payload["long_term_material_is_not_fact"] is True
    finally:
        rig.close()


@pytest.mark.parametrize("source_exists", [True, False])
def test_evidence_query_returns_only_verified_memory_sources(
    tmp_path: Path, source_exists: bool
) -> None:
    source_id = "source-event" if source_exists else "missing-event"
    classification_result = classification(
        "evidence_query"
    ).model_copy(update={"expected_plan_version": 1})
    answer = QuestionAnswer(
        answer="基于已验证来源回答。" if source_exists else "无法确认。",
        evidence_ids=(source_id,) if source_exists else (),
    )
    rig = make_rig(tmp_path, classification_result, answer=answer)
    rig.context = replace(
        rig.context,
        active_plan=TrackingPlan(
            plan_id="S1:plan:1",
            scenario_id="S1",
            revision=1,
            base_snapshot_revision=3,
            status="active",
            evidence_ids=(),
        ),
        memory_service=RecordingMemoryService(
            MemoryContext(
                user_id="operator",
                long_term_material=(
                    MemoryRetrievalHit(
                        memory=MemoryVersion(
                            memory_id="memory-1",
                            memory_family_id="family-1",
                            version=1,
                            user_id="operator",
                            scenario_id="S1",
                            memory_type=MemoryType.EPISODIC,
                            summary="historical source summary",
                            importance_score=0.8,
                            embedding=(1.0,),
                            source_event_ids=(source_id,),
                        ),
                        similarity_score=0.9,
                        rerank_score=0.8,
                        retrieval_reason="semantic match",
                    ),
                ),
                retrieved_memory_ids=("memory-1",),
                memory_status=MemoryStreamStatus.COMPLETED,
            )
        ),
    )
    if source_exists:
        rig.events.append(
            event_id=source_id,
            event_type="target_added",
            scenario_id="S1",
            sim_time_s=900,
            payload={},
        )
    try:
        result = process_conversation_message(
            message("为什么？", expected_plan_version=1), rig.context
        )
        assert result.answer is not None
        assert result.answer.memory_ids == ("memory-1",)
        if source_exists:
            assert result.answer.evidence_ids == (source_id,)
            assert "原始证据不足" not in result.answer.answer
        else:
            assert result.answer.evidence_ids == ()
            assert "记忆线索存在、原始证据不足" in result.answer.answer
    finally:
        rig.close()


def test_evidence_query_keeps_current_evidence_when_memory_source_is_verified(
    tmp_path: Path,
) -> None:
    current_id = "current-snapshot-evidence"
    memory_id = "memory-source-event"
    rig = make_rig(
        tmp_path,
        classification("evidence_query").model_copy(update={"expected_plan_version": 1}),
        answer=QuestionAnswer(
            answer="当前快照和记忆回溯来源均可核验。",
            evidence_ids=(current_id, memory_id),
        ),
    )
    rig.context = replace(
        rig.context,
        active_plan=TrackingPlan(
            plan_id="S1:plan:1",
            scenario_id="S1",
            revision=1,
            base_snapshot_revision=3,
            status="active",
            evidence_ids=(current_id,),
        ),
        memory_service=RecordingMemoryService(
            MemoryContext(
                user_id="operator",
                long_term_material=(
                    MemoryRetrievalHit(
                        memory=MemoryVersion(
                            memory_id="memory-1",
                            memory_family_id="family-1",
                            version=1,
                            user_id="operator",
                            scenario_id="S1",
                            memory_type=MemoryType.EPISODIC,
                            summary="historical source summary",
                            importance_score=0.8,
                            embedding=(1.0,),
                            source_event_ids=(memory_id,),
                        ),
                        similarity_score=0.9,
                        rerank_score=0.8,
                        retrieval_reason="semantic match",
                    ),
                ),
                retrieved_memory_ids=("memory-1",),
                memory_status=MemoryStreamStatus.COMPLETED,
            )
        ),
    )
    rig.events.append(
        event_id=memory_id,
        event_type="target_added",
        scenario_id="S1",
        sim_time_s=900,
        payload={},
    )
    rig.events.append(
        event_id=current_id,
        event_type="bearing",
        scenario_id="S1",
        sim_time_s=900,
        payload={},
    )
    try:
        result = process_conversation_message(message("为什么？", expected_plan_version=1), rig.context)

        assert result.answer is not None
        assert set(result.answer.evidence_ids) == {current_id, memory_id}
        assert set(rig.llm.payloads[-1]["evidence_ids"]) == {current_id, memory_id}
    finally:
        rig.close()


def test_real_memory_service_receives_only_the_completed_turn_and_queues_it(tmp_path: Path) -> None:
    database = tmp_path / "conversation-memory.db"
    events = EventRepository(database)
    ledger = DecisionLedger(database)
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    class Retriever:
        def retrieve(self, **kwargs: object) -> MemoryContext:
            del kwargs
            return MemoryContext(user_id="operator", memory_status=MemoryStreamStatus.COMPLETED)

    memory = MemoryService(
        short_term,
        long_term,
        Retriever(),
    )
    llm = RecordingLLM(classification("plan_revision", proposal=feedback_proposal()))
    situation = SituationSnapshot.model_construct(
        scenario_id="S1",
        snapshot_revision=3,
        sim_time_s=900,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )
    context = ConversationContext(
        scenario_id="S1",
        situation=situation,
        active_plan=None,
        ledger=ledger,
        events=events,
        llm=llm,
        memory_service=memory,
    )
    try:
        result = process_conversation_message(message("增加 region_1 的接力余量"), context)
        assert result.queued_memory_work_id is not None
        stored = short_term.get_short_term("operator", "conversation-1", "S1")
        assert stored is not None
        assert [item.role for item in stored.recent_messages] == ["expert", "assistant"]
        assert long_term._conn.execute(
            "SELECT status FROM memory_work_items"
        ).fetchone()["status"] == "pending"
    finally:
        long_term.close()
        short_term.close()
        events.close()
        ledger.close()


def test_failed_knowledge_source_is_degraded_and_never_cited_as_fact(tmp_path: Path) -> None:
    rig = make_rig(
        tmp_path, classification("evidence_query"), answer=QuestionAnswer(answer="无法确认。")
    )
    memory = MemoryVersion(
        memory_id="memory-knowledge",
        memory_family_id="family-knowledge",
        version=1,
        user_id="operator",
        scenario_id="S1",
        memory_type=MemoryType.SEMANTIC,
        summary="ontology summary that is not evidence",
        importance_score=0.8,
        embedding=(1.0,),
        source_knowledge_ids=("knowledge-failed",),
    )
    rig.context.ledger.save_knowledge_query(
        query_id="knowledge-failed",
        scenario_id="S1",
        sim_time_s=900,
        query_text="failed knowledge lookup",
        mode="mix",
        status="failed",
        response={"error": "service unavailable"},
    )
    rig.context = replace(
        rig.context,
        memory_service=RecordingMemoryService(
            MemoryContext(
                user_id="operator",
                long_term_material=(
                    MemoryRetrievalHit(
                        memory=memory,
                        similarity_score=0.9,
                        rerank_score=0.8,
                        retrieval_reason="semantic match",
                    ),
                ),
                retrieved_memory_ids=("memory-knowledge",),
                memory_status=MemoryStreamStatus.COMPLETED,
            )
        ),
    )
    try:
        result = process_conversation_message(message("这个知识结论可靠吗？"), rig.context)

        assert result.answer is not None
        assert result.answer.evidence_ids == ()
        assert "记忆线索存在、原始证据不足" in result.answer.answer
    finally:
        rig.close()


def test_mixed_memory_sources_remain_degraded_and_answer_discloses_missing_source(
    tmp_path: Path,
) -> None:
    rig = make_rig(
        tmp_path,
        classification("evidence_query"),
        answer=QuestionAnswer(answer="无法确认。"),
    )
    valid_event = "event-valid"
    missing_event = "event-missing"
    rig.events.append(
        event_id=valid_event,
        event_type="target_added",
        scenario_id="S1",
        sim_time_s=900,
        payload={},
    )
    memory = MemoryVersion(
        memory_id="memory-mixed",
        memory_family_id="family-mixed",
        version=1,
        user_id="operator",
        scenario_id="S1",
        memory_type=MemoryType.EPISODIC,
        summary="mixed source summary",
        importance_score=0.8,
        embedding=(1.0,),
        source_event_ids=(valid_event, missing_event),
    )
    rig.context = replace(
        rig.context,
        memory_service=RecordingMemoryService(
            MemoryContext(
                user_id="operator",
                long_term_material=(
                    MemoryRetrievalHit(
                        memory=memory,
                        similarity_score=0.9,
                        rerank_score=0.8,
                        retrieval_reason="semantic match",
                    ),
                ),
                retrieved_memory_ids=("memory-mixed",),
                memory_status=MemoryStreamStatus.COMPLETED,
            )
        ),
    )
    try:
        result = process_conversation_message(message("为什么？"), rig.context)

        assert result.memory_context is not None
        assert result.memory_context.evidence_trace[0].status is MemoryStreamStatus.DEGRADED
        assert result.answer is not None
        assert result.answer.evidence_ids == ()
        assert "记忆线索存在、原始证据不足" in result.answer.answer
    finally:
        rig.close()


def test_verified_knowledge_query_is_in_question_evidence_namespace(tmp_path: Path) -> None:
    query_id = "S1:knowledge:900"
    answer = QuestionAnswer(answer="知识服务已返回该结论。", evidence_ids=(query_id,))
    rig = make_rig(tmp_path, classification("evidence_query"), answer=answer)
    rig.context.ledger.save_knowledge_query(
        query_id=query_id,
        scenario_id="S1",
        sim_time_s=900,
        query_text="verified knowledge lookup",
        mode="mix",
        status="completed",
        response={"answer": "verified knowledge answer"},
    )
    memory = MemoryVersion(
        memory_id="memory-knowledge-valid",
        memory_family_id="family-knowledge-valid",
        version=1,
        user_id="operator",
        memory_type=MemoryType.SEMANTIC,
        summary="knowledge summary",
        importance_score=0.8,
        embedding=(1.0,),
        source_knowledge_ids=(query_id,),
    )
    rig.context = replace(
        rig.context,
        memory_service=RecordingMemoryService(
            MemoryContext(
                user_id="operator",
                long_term_material=(
                    MemoryRetrievalHit(
                        memory=memory,
                        similarity_score=0.9,
                        rerank_score=0.8,
                        retrieval_reason="semantic match",
                    ),
                ),
                retrieved_memory_ids=(memory.memory_id,),
                memory_status=MemoryStreamStatus.COMPLETED,
            )
        ),
    )
    try:
        result = process_conversation_message(message("这个知识结论可靠吗？"), rig.context)

        assert result.answer is not None
        assert result.answer.evidence_ids == (query_id,)
        question_payload = next(
            payload for operation, payload in zip(rig.llm.calls, rig.llm.payloads)
            if operation == "question"
        )
        assert query_id in question_payload["evidence_ids"]
        assert question_payload["knowledge_queries"][0]["query_id"] == query_id
    finally:
        rig.close()


def test_memory_context_without_service_cannot_cross_user_into_llm(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, classification("clarification"))
    rig.context = replace(
        rig.context,
        memory_context=MemoryContext(
            user_id="different-user",
            long_term_material=(),
            memory_status=MemoryStreamStatus.COMPLETED,
        ),
    )
    incoming = message("请说明当前情况").model_copy(update={"user_id": "request-user"})
    try:
        result = process_conversation_message(incoming, rig.context)

        assert result.memory_context is not None
        assert result.memory_context.user_id == "request-user"
        assert result.memory_context.memory_status is MemoryStreamStatus.DEGRADED
        assert rig.llm.payloads[0]["long_term_material"] == []
    finally:
        rig.close()


def test_memory_message_provenance_requires_current_user_conversation_and_scenario(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path, classification("clarification"))
    short_term = ShortTermContextRepository(tmp_path / "conversation.db")
    short_term.append_messages(
        "request-user",
        "conversation-1",
        (
            ShortTermMessage(
                message_id="message-valid",
                role="user",
                text="valid source",
                scenario_id="S1",
            ),
        ),
        scenario_id="S1",
    )
    short_term._conn.execute(
        "UPDATE short_term_contexts SET recent_messages = ?"
        " WHERE user_id = ? AND scenario_id = ? AND conversation_id = ?",
        (
            json.dumps(
                [
                    {
                        "message_id": "message-valid",
                        "role": "user",
                        "text": "valid source",
                        "scenario_id": "S1",
                    },
                    {
                        "message_id": "message-wrong-scenario",
                        "role": "user",
                        "text": "wrong scenario source",
                        "scenario_id": "S2",
                    },
                ]
            ),
            "request-user",
            "S1",
            "conversation-1",
        ),
    )
    memory = MemoryVersion(
        memory_id="memory-provenance",
        memory_family_id="family-provenance",
        version=1,
        user_id="request-user",
        scenario_id="S1",
        memory_type=MemoryType.SEMANTIC,
        summary="retrieved summary",
        importance_score=0.8,
        embedding=(1.0,),
        source_message_ids=("message-valid", "message-wrong-scenario", "message-missing"),
    )
    context = replace(
        rig.context,
        user_id="request-user",
        scenario_id="S1",
        short_term_repository=short_term,
        conversation_id="conversation-1",
    )
    memory_context = MemoryContext(
        user_id="request-user",
        long_term_material=(
            MemoryRetrievalHit(
                memory=memory,
                similarity_score=1.0,
                rerank_score=1.0,
                retrieval_reason="semantic match",
            ),
        ),
        retrieved_memory_ids=(memory.memory_id,),
        memory_status=MemoryStreamStatus.COMPLETED,
    )
    try:
        verified = _verify_memory_sources(context, memory_context)
        trace = verified.evidence_trace[0]
        assert trace.source_message_ids == ("message-valid",)
        assert trace.status is MemoryStreamStatus.DEGRADED
        assert _verify_memory_sources(context, memory_context).retrieved_memory_ids == (
            "memory-provenance",
        )
    finally:
        short_term.close()
        rig.close()


def test_apply_conversation_rejects_a_preview_owned_by_another_user() -> None:
    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._lock = __import__("threading").RLock()
    owner_message = message("增加 region_1 的接力余量").model_copy(update={"user_id": "owner"})
    turn = ConversationTurnResult(
        conversation_id=owner_message.conversation_id,
        turn_id="conversation-1:turn:1",
        user_id="owner",
        classification=classification("plan_revision", proposal=feedback_proposal()),
        messages=(owner_message,),
        proposal=feedback_proposal(),
        expected_plan_version=0,
    )
    runtime._conversation_turns = {(turn.conversation_id, turn.turn_id): turn}

    with pytest.raises(ValueError, match="belongs to user"):
        runtime.apply_conversation(
            turn.conversation_id,
            turn.turn_id,
            0,
            user_id="attacker",
        )


def test_mixed_returns_independent_preview_and_evidence_without_applying(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, classification("mixed", proposal=feedback_proposal()))
    try:
        result = process_conversation_message(message("为什么保持方案，并增加 region_1 接力？"), rig.context)

        assert result.classification.classification == "mixed"
        assert result.proposal is not None
        assert result.proposal.status == "preview"
        assert [item.role for item in result.messages] == ["expert", "assistant", "assistant"]
        assert rig.events.list_events(scenario_id="S1") == []
        assert rig.llm.calls == ["conversation_classification", "directive", "question"]
    finally:
        rig.close()


def test_clarification_returns_one_follow_up_message(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, classification("clarification"))
    try:
        result = process_conversation_message(message("请调整一下"), rig.context)

        assert result.proposal is None
        assert len(result.messages) == 2
        assert result.messages[-1].role == "assistant"
        assert result.messages[-1].text == "请说明需要调整的目标或区域。"
        assert rig.llm.calls == ["conversation_classification"]
    finally:
        rig.close()


def test_conversation_rejects_unknown_evidence_and_stale_plan_version(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, classification("evidence_query", evidence_ids=("ghost",)))
    try:
        with pytest.raises(ValueError, match="unknown evidence"):
            process_conversation_message(message("为什么？"), rig.context)
        with pytest.raises(ValueError, match="plan version"):
            process_conversation_message(message("为什么？", expected_plan_version=1), rig.context)
    finally:
        rig.close()


def test_conversation_http_routes_keep_preview_and_apply_explicit() -> None:
    from fastapi.testclient import TestClient

    from underwater_tracking.api.app import create_app
    from underwater_tracking.api.hub import OperationalHub

    class FakeRuntime:
        def __init__(self) -> None:
            self.received: list[ConversationMessage] = []
            self.applied: list[tuple[str, str, int]] = []
            self.last_result: ConversationTurnResult | None = None

        def active_plan(self) -> None:
            return None

        def ask(self, raw_text: str, counterfactual: dict[str, object] | None = None) -> QuestionAnswer:
            del raw_text, counterfactual
            return QuestionAnswer(answer="只读回答")

        def preview_directive(self, raw_text: str) -> ExpertDirective:
            del raw_text
            return feedback_proposal()

        def apply_directive(self, directive_id: str) -> ExpertDirective:
            del directive_id
            return feedback_proposal().model_copy(update={"status": "applied"})

        def preview_assignment(self, *, uuv_ids: tuple[str, ...], target_id: str) -> ExpertDirective:
            del uuv_ids, target_id
            return feedback_proposal()

        def submit_sensor_mode(self, **kwargs: object) -> None:
            del kwargs

        def conversation_message(self, item: ConversationMessage) -> Any:
            self.received.append(item)
            result = process_conversation_message
            del result
            self.last_result = ConversationTurnResult(
                conversation_id=item.conversation_id,
                turn_id="conversation-1:turn:1",
                classification=classification("plan_revision", proposal=feedback_proposal()),
                messages=(
                    item,
                    item.model_copy(
                        update={
                            "message_id": "assistant-1",
                            "role": "assistant",
                            "text": "已生成方案预览，请确认后应用。",
                            "proposal": feedback_proposal(),
                            "turn_id": "conversation-1:turn:1",
                        }
                    ),
                ),
                proposal=feedback_proposal(),
                expected_plan_version=0,
            )
            return self.last_result

        def apply_conversation(
            self,
            conversation_id: str,
            turn_id: str,
            expected_plan_version: int,
            *,
            user_id: str = "operator",
        ) -> Any:
            assert user_id == "operator"
            self.applied.append((conversation_id, turn_id, expected_plan_version))
            assert self.last_result is not None
            return self.last_result.model_copy(update={"applied": True})

    class EmptyReplay:
        def range(self, start_s: float = 0.0, end_s: float | None = None) -> list[Any]:
            del start_s, end_s
            return []

    runtime = FakeRuntime()
    app = create_app(runtime=runtime, replay=EmptyReplay(), hub=OperationalHub())
    client = TestClient(app)

    response = client.post(
        "/api/conversation/messages",
        json={
            "conversation_id": "conversation-1",
            "text": "增加 region_1 的接力余量",
            "expected_plan_version": 0,
            "target_scope": ["T1"],
            "region_scope": ["region_1"],
        },
    )
    assert response.status_code == 200
    assert response.json()["proposal"]["status"] == "preview"
    assert runtime.received[0].role == "expert"

    applied = client.post(
        "/api/conversation/conversation-1/apply",
        json={"turn_id": "conversation-1:turn:1", "expected_plan_version": 0},
    )
    assert applied.status_code == 200
    assert applied.json()["applied"] is True
    assert runtime.applied == [("conversation-1", "conversation-1:turn:1", 0)]
