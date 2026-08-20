from __future__ import annotations

import json
from pathlib import Path

from underwater_tracking.domain.memory_models import (
    MemoryStreamEventType,
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
    assert event.type is MemoryStreamEventType.WORK_QUEUED


def test_accept_turn_is_idempotent_for_repeated_turn_and_persists_conversation_cursor(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    service = MemoryService(short_term, long_term, RecordingRetriever(None))
    turn = {
        "user_id": "operator",
        "conversation_id": "conversation-1",
        "scenario_id": "scenario-1",
        "turn_id": "turn-1",
        "message_id": "message-1",
        "text": "remember this turn",
    }
    result = {"message_id": "message-2", "text": "I will remember it."}

    first = service.accept_turn(turn, result)
    second = service.accept_turn(turn, result)

    assert first["status"] == "queued"
    assert second["status"] == "duplicate"
    context = short_term.get_short_term("operator", "conversation-1")
    assert context is not None
    assert context.message_count == 2
    assert [message.message_id for message in context.recent_messages] == ["message-1", "message-2"]
    assert long_term._conn.execute("SELECT COUNT(*) FROM memory_work_items").fetchone()[0] == 1
    assert long_term.get_source_cursor("operator", "scenario-1", "conversation:conversation-1") == 2


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


def test_enqueue_observation_persists_bounded_sanitized_source_projection(
    tmp_path: Path,
) -> None:
    short_term = ShortTermContextRepository(tmp_path / "memory.db")
    long_term = LongTermMemoryRepository(tmp_path / "memory.db")
    service = MemoryService(short_term, long_term, RecordingRetriever(None))

    assert service.enqueue_observation(
        {
            "source_id": "event-bounded",
            "source_type": "runtime_event",
            "scenario_id": "scenario-1",
            "user_id": "operator",
        },
        {
            "event_id": "event-bounded",
            "event_type": "bearing",
            "summary": "keep this source text",
            "private_raw_payload": "do not persist this field",
            "oversized": "x" * 20_000,
        },
    )["status"] == "queued"

    row = long_term._conn.execute(
        "SELECT payload FROM memory_work_items WHERE source_key = ?",
        ("runtime_event:event-bounded",),
    ).fetchone()
    assert row is not None
    persisted = json.loads(row["payload"])
    assert persisted["source_text"] == "keep this source text"
    assert persisted["source_payload"] == {
        "event_id": "event-bounded",
        "event_type": "bearing",
        "summary": "keep this source text",
    }
    assert "private_raw_payload" not in json.dumps(persisted)
    assert len(row["payload"].encode("utf-8")) <= 8192


def test_accept_turn_without_message_ids_is_deterministically_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    service = MemoryService(short_term, long_term, RecordingRetriever(None))
    turn = {
        "user_id": "operator",
        "conversation_id": "conversation-1",
        "turn_id": "turn-without-message-id",
        "text": "remember this stable turn",
    }
    result = {"text": "stable answer"}

    first = service.accept_turn(turn, result)
    second = service.accept_turn(turn, result)

    assert first["status"] == "queued"
    assert second["status"] == "duplicate"
    assert long_term._conn.execute("SELECT COUNT(*) FROM memory_work_items").fetchone()[0] == 1
    context = short_term.get_short_term("operator", "conversation-1")
    assert context is not None
    assert len(context.recent_messages) == 2
    assert len({message.message_id for message in context.recent_messages}) == 2
