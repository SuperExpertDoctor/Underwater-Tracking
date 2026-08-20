from __future__ import annotations

from pathlib import Path

from underwater_tracking.domain.memory_models import (
    MemoryStreamStatus,
    MemoryType,
    MemoryVersion,
    ShortTermMessage,
)
from underwater_tracking.memory.service import MemoryService
from underwater_tracking.persistence.memory import LongTermMemoryRepository, ShortTermContextRepository


class RecordingRetriever:
    def __init__(self, result):
        self.result = result
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def test_prepare_context_keeps_short_and_long_term_separate(tmp_path: Path) -> None:
    short_term = ShortTermContextRepository(tmp_path / "memory.db")
    long_term = LongTermMemoryRepository(tmp_path / "memory.db")
    short_term.append_messages(
        "operator",
        "conversation-1",
        (ShortTermMessage(message_id="message-1", role="user", text="keep this nearby"),),
    )
    memory = MemoryVersion(
        memory_id="memory-1",
        memory_family_id="family-1",
        version=1,
        user_id="operator",
        memory_type=MemoryType.SEMANTIC,
        summary="long-term preference",
        importance_score=0.8,
        embedding=(1.0,),
    )
    long_term.create_memory_version(memory, expected_previous_version=0)
    from underwater_tracking.domain.memory_models import MemoryContext, MemoryRetrievalHit

    retriever = RecordingRetriever(
        MemoryContext(
            user_id="operator",
            long_term_material=(
                MemoryRetrievalHit(
                    memory=memory,
                    similarity_score=1.0,
                    rerank_score=1.0,
                    retrieval_reason="semantic match",
                ),
            ),
            retrieved_memory_ids=("memory-1",),
            memory_status=MemoryStreamStatus.COMPLETED,
        )
    )
    service = MemoryService(short_term, long_term, retriever)

    context = service.prepare_context("operator", "conversation-1", "what do I prefer?", {})

    assert context.short_term_context is not None
    assert context.short_term_context.recent_messages[0].message_id == "message-1"
    assert [hit.memory.memory_id for hit in context.long_term_material] == ["memory-1"]
    assert retriever.calls == [
        {"user_id": "operator", "query": "what do I prefer?", "filters": {}, "now": None}
    ]


def test_accept_turn_persists_messages_then_queues_without_calling_a_reasoner(tmp_path: Path) -> None:
    short_term = ShortTermContextRepository(tmp_path / "memory.db")
    long_term = LongTermMemoryRepository(tmp_path / "memory.db")
    service = MemoryService(short_term, long_term, RecordingRetriever(None))
    turn = {
        "user_id": "operator",
        "conversation_id": "conversation-1",
        "message_id": "message-1",
        "text": "please remember the reporting preference",
    }
    result = {"message_id": "message-2", "text": "I will use concise reports."}

    outcome = service.accept_turn(turn, result, source_refs=("event-1",))

    assert outcome["status"] == "queued"
    context = short_term.get_short_term("operator", "conversation-1")
    assert context is not None
    assert [message.message_id for message in context.recent_messages] == ["message-1", "message-2"]
    row = long_term._conn.execute("SELECT work_type, status FROM memory_work_items").fetchone()
    assert tuple(row) == ("conversation_turn", "pending")
    event = long_term.list_stream_events("operator", "conversation-1", limit=10)[0]
    assert event.status is MemoryStreamStatus.PENDING


def test_enqueue_observation_deduplicates_source_key_without_reasoning(tmp_path: Path) -> None:
    short_term = ShortTermContextRepository(tmp_path / "memory.db")
    long_term = LongTermMemoryRepository(tmp_path / "memory.db")
    service = MemoryService(short_term, long_term, RecordingRetriever(None))

    assert service.enqueue_observation(
        {"source_id": "event-1", "scenario_id": "scenario-1", "user_id": "operator"},
        {"summary": "bearing changed"},
    )["status"] == "queued"
    assert service.enqueue_observation(
        {"source_id": "event-1", "scenario_id": "scenario-1", "user_id": "operator"},
        {"summary": "bearing changed"},
    )["status"] == "duplicate"
