from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from underwater_tracking.domain.memory_models import (
    MemoryContext,
    MemoryRetrievalHit,
    MemoryStreamEvent,
    MemoryVersion,
    MemoryWorkItem,
    ShortTermContext,
    ShortTermMessage,
)


def _memory_version(**changes: object) -> MemoryVersion:
    values: dict[str, object] = {
        "memory_id": "memory-v1",
        "memory_family_id": "family-1",
        "version": 1,
        "user_id": "operator",
        "memory_type": "episodic",
        "summary": "Operator asked to preserve the target track.",
        "importance_score": 0.8,
        "status": "active",
    }
    values.update(changes)
    return MemoryVersion(**values)


def test_memory_version_enforces_version_and_user_isolation_contract() -> None:
    assert _memory_version().version == 1

    with pytest.raises(ValidationError, match="user_id"):
        _memory_version(user_id=" ")
    with pytest.raises(ValidationError):
        _memory_version(user_id="u" * 121)
    with pytest.raises(ValidationError, match="version"):
        _memory_version(version=0)
    with pytest.raises(ValidationError, match="importance_score"):
        _memory_version(importance_score=1.1)
    with pytest.raises(ValidationError):
        _memory_version(memory_type="working")
    with pytest.raises(ValidationError, match="supersedes_memory_id"):
        _memory_version(version=2)
    with pytest.raises(ValidationError, match="active"):
        _memory_version(status="unknown")


def test_short_term_context_exposes_bounded_context_state() -> None:
    updated_at = datetime(2026, 8, 20, tzinfo=UTC)
    context = ShortTermContext(
        user_id="operator",
        conversation_id="conversation-1",
        summary_text="The operator is reviewing a plan revision.",
        summary_version=2,
        recent_messages=(
            ShortTermMessage(
                message_id="message-1",
                turn_id="turn-1",
                role="expert",
                text="Show the evidence.",
                created_at=updated_at,
            ),
        ),
        message_count=4,
        estimated_tokens=112,
        compression_count=1,
        compression_status="completed",
        updated_at=updated_at,
    )

    assert context.summary_version == 2
    assert context.recent_messages[0].turn_id == "turn-1"
    assert context.estimated_tokens == 112
    assert context.compression_status == "completed"
    assert context.updated_at == updated_at


@pytest.mark.parametrize("work_type", ["observation", "conversation_turn", "maintenance"])
def test_work_item_accepts_only_declared_work_types_and_statuses(work_type: str) -> None:
    item = MemoryWorkItem(
        work_id="work-1",
        user_id="operator",
        work_type=work_type,
        payload={"source_message_ids": ["message-1"]},
    )
    assert item.status == "pending"

    with pytest.raises(ValidationError):
        item.model_copy(update={"status": "queued"}).__class__.model_validate(
            {**item.model_dump(), "status": "queued"}
        )


def test_memory_stream_event_allows_only_structured_safe_payload() -> None:
    event = MemoryStreamEvent(
        cursor=7,
        event_id="stream-7",
        user_id="operator",
        status="completed",
        type="retrieval_completed",
        payload={"hit_count": 2, "memory_ids": ["memory-v1", "memory-v2"]},
    )

    assert event.cursor == 7
    assert event.payload.hit_count == 2
    with pytest.raises(ValidationError, match="Extra inputs"):
        MemoryStreamEvent(
            cursor=8,
            event_id="stream-8",
            user_id="operator",
            status="completed",
            type="retrieval_completed",
            payload={"raw_request": "operator private prompt"},
        )
    with pytest.raises(ValidationError):
        MemoryStreamEvent(
            cursor=9,
            event_id="stream-9",
            user_id="operator",
            status="processing",
            type="llm_thinking",
        )


def test_memory_context_keeps_short_and_long_term_material_separate() -> None:
    short_term = ShortTermContext(
        user_id="operator",
        conversation_id="conversation-1",
        updated_at=datetime(2026, 8, 20, tzinfo=UTC),
    )
    hit = MemoryRetrievalHit(
        memory=_memory_version(),
        similarity_score=0.9,
        rerank_score=0.8,
        retrieval_reason="same target and active plan",
    )
    context = MemoryContext(
        short_term_context=short_term,
        long_term_material=(hit,),
        retrieved_memory_ids=("memory-v1",),
        memory_status="completed",
        evidence_trace=(),
    )

    assert context.short_term_context is short_term
    assert context.long_term_material == (hit,)
    assert context.retrieved_memory_ids == ("memory-v1",)
    assert context.memory_status == "completed"
