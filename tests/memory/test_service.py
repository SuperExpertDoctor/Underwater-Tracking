from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
import pytest

from underwater_tracking.domain.memory_models import (
    MemoryContext,
    MemoryEvidenceTrace,
    MemoryWorkPayload,
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


def test_prepare_context_filters_legacy_wrong_scenario_messages(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    short_term.append_messages(
        "operator",
        "conversation-1",
        (ShortTermMessage(message_id="valid", scenario_id="scenario-a", role="user", text="valid"),),
        scenario_id="scenario-a",
    )
    short_term._conn.execute(
        "UPDATE short_term_contexts SET recent_messages = ?"
        " WHERE user_id = ? AND scenario_id = ? AND conversation_id = ?",
        (
            json.dumps(
                [
                    {"message_id": "valid", "scenario_id": "scenario-a", "role": "user", "text": "valid"},
                    {"message_id": "wrong", "scenario_id": "scenario-b", "role": "user", "text": "wrong"},
                ]
            ),
            "operator",
            "scenario-a",
            "conversation-1",
        ),
    )
    service = MemoryService(
        short_term,
        long_term,
        RecordingRetriever(MemoryContext(user_id="operator")),
    )

    context = service.prepare_context(
        "operator", "conversation-1", "query", scenario_id="scenario-a"
    )

    assert context.short_term_context is not None
    assert [message.message_id for message in context.short_term_context.recent_messages] == ["valid"]


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
    from underwater_tracking.domain.memory_models import MemoryRetrievalHit

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


def test_prepare_context_passes_scenario_filter_to_retriever(tmp_path: Path) -> None:
    short_term = ShortTermContextRepository(tmp_path / "memory.db")
    long_term = LongTermMemoryRepository(tmp_path / "memory.db")
    retriever = RecordingRetriever(MemoryContext(user_id="operator"))
    service = MemoryService(short_term, long_term, retriever)

    service.prepare_context(
        "operator", "conversation-1", "scenario question", filters={}, scenario_id="scenario-a"
    )

    assert retriever.calls == [
        {
            "user_id": "operator",
            "query": "scenario question",
            "filters": {"scenario_id": "scenario-a"},
            "now": None,
        }
    ]


def test_accept_turn_persists_messages_then_queues_without_calling_a_reasoner(tmp_path: Path) -> None:
    short_term = ShortTermContextRepository(tmp_path / "memory.db")
    long_term = LongTermMemoryRepository(tmp_path / "memory.db")
    service = MemoryService(short_term, long_term, RecordingRetriever(None))
    turn = {
        "user_id": "operator",
        "conversation_id": "conversation-1",
        "scenario_id": "scenario-1",
        "message_id": "message-1",
        "text": "please remember the reporting preference",
    }
    result = {"message_id": "message-2", "text": "I will use concise reports."}

    outcome = service.accept_turn(turn, result, source_refs=("event-1",))

    assert outcome["status"] == "queued"
    assert isinstance(outcome["stream_cursor"], int)
    context = short_term.get_short_term("operator", "conversation-1", "scenario-1")
    assert context is not None
    assert [message.message_id for message in context.recent_messages] == ["message-1", "message-2"]
    row = long_term._conn.execute("SELECT work_type, status FROM memory_work_items").fetchone()
    assert tuple(row) == ("conversation_turn", "pending")
    event = long_term.list_stream_events(
        "operator", "conversation-1", scenario_id="scenario-1", limit=10
    )[0]
    assert event.status is MemoryStreamStatus.PENDING
    assert event.type is MemoryStreamEventType.WORK_QUEUED
    assert event.cursor == outcome["stream_cursor"]


def test_evidence_trace_events_are_structured_and_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    service = MemoryService(short_term, long_term, RecordingRetriever(None))
    trace = MemoryEvidenceTrace(
        trace_id="trace:conversation-1:memory-1",
        user_id="operator",
        status=MemoryStreamStatus.COMPLETED,
        memory_ids=("memory-1",),
        source_message_ids=("message-1",),
        source_event_ids=("event-1",),
        source_decision_ids=("decision-1",),
        source_knowledge_ids=("knowledge-1",),
        source_plan_ids=("plan-1",),
    )

    emitted = service.emit_evidence_trace_events(
        user_id="operator",
        conversation_id="conversation-1",
        scenario_id="scenario-1",
        trace=trace,
        plan_version=7,
    )

    assert [event.type for event in emitted] == [
        MemoryStreamEventType.EVIDENCE_TRACE_STARTED,
        MemoryStreamEventType.EVIDENCE_TRACE_COMPLETED,
    ]
    completed = emitted[-1]
    assert completed.payload.source_message_ids == ("message-1",)
    assert completed.payload.source_event_ids == ("event-1",)
    assert completed.payload.source_decision_ids == ("decision-1",)
    assert completed.payload.source_knowledge_ids == ("knowledge-1",)
    assert completed.payload.source_plan_ids == ("plan-1",)
    assert completed.payload.plan_version == 7
    assert service.emit_evidence_trace_events(
        user_id="operator",
        conversation_id="conversation-1",
        scenario_id="scenario-1",
        trace=trace,
        plan_version=7,
    ) == ()


def test_evidence_trace_events_are_atomic_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    service = MemoryService(short_term, long_term, RecordingRetriever(None))
    trace = MemoryEvidenceTrace(
        trace_id="trace:atomic-retry",
        user_id="operator",
        status=MemoryStreamStatus.COMPLETED,
        memory_ids=("memory-atomic",),
        source_event_ids=("event-atomic",),
    )
    original = long_term.append_stream_events

    def fail_once(events):
        monkeypatch.setattr(long_term, "append_stream_events", original)
        raise RuntimeError("temporary stream failure")

    monkeypatch.setattr(long_term, "append_stream_events", fail_once)
    with pytest.raises(RuntimeError, match="temporary stream failure"):
        service.emit_evidence_trace_events(
            user_id="operator",
            conversation_id="conversation-atomic",
            scenario_id="scenario-1",
            trace=trace,
            plan_version=3,
        )

    assert long_term.list_stream_events(
        "operator", "conversation-atomic", scenario_id="scenario-1", limit=10
    ) == []
    emitted = service.emit_evidence_trace_events(
        user_id="operator",
        conversation_id="conversation-atomic",
        scenario_id="scenario-1",
        trace=trace,
        plan_version=3,
    )
    assert [event.type for event in emitted] == [
        MemoryStreamEventType.EVIDENCE_TRACE_STARTED,
        MemoryStreamEventType.EVIDENCE_TRACE_COMPLETED,
    ]


def test_evidence_trace_events_are_idempotent_across_concurrent_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "memory.db"
    bootstrap_short_term = ShortTermContextRepository(database)
    bootstrap_long_term = LongTermMemoryRepository(database)
    bootstrap_short_term.close()
    bootstrap_long_term.close()
    trace = MemoryEvidenceTrace(
        trace_id="trace:concurrent",
        user_id="operator",
        status=MemoryStreamStatus.COMPLETED,
        memory_ids=("memory-concurrent",),
        source_event_ids=("event-concurrent",),
    )
    lookup_barrier = Barrier(2)
    original_lookup = LongTermMemoryRepository.get_stream_event_for_work

    def synchronized_lookup(repository, *args, **kwargs):
        result = original_lookup(repository, *args, **kwargs)
        lookup_barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(
        LongTermMemoryRepository, "get_stream_event_for_work", synchronized_lookup
    )

    def emit_once() -> tuple[MemoryStreamEventType, ...]:
        short_term = ShortTermContextRepository(database)
        long_term = LongTermMemoryRepository(database)
        service = MemoryService(short_term, long_term, RecordingRetriever(None))
        try:
            return tuple(
                event.type
                for event in service.emit_evidence_trace_events(
                    user_id="operator",
                    conversation_id="conversation-concurrent",
                    scenario_id="scenario-1",
                    trace=trace,
                    plan_version=4,
                )
            )
        finally:
            short_term.close()
            long_term.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: emit_once(), range(2)))

    repository = LongTermMemoryRepository(database)
    try:
        events = repository.list_stream_events(
            "operator", "conversation-concurrent", scenario_id="scenario-1", limit=10
        )
    finally:
        repository.close()
    assert [event.type for event in events] == [
        MemoryStreamEventType.EVIDENCE_TRACE_STARTED,
        MemoryStreamEventType.EVIDENCE_TRACE_COMPLETED,
    ]
    assert all(
        set(result) <= {
            MemoryStreamEventType.EVIDENCE_TRACE_STARTED,
            MemoryStreamEventType.EVIDENCE_TRACE_COMPLETED,
        }
        for result in results
    )


def test_conversation_sources_are_conservative_and_plan_versioned(tmp_path: Path) -> None:
    short_term = ShortTermContextRepository(tmp_path / "memory.db")
    long_term = LongTermMemoryRepository(tmp_path / "memory.db")
    service = MemoryService(short_term, long_term, RecordingRetriever(None))
    outcome = service.accept_turn(
        {
            "user_id": "operator",
            "conversation_id": "conversation-typed",
            "scenario_id": "scenario-1",
            "message_id": "message-typed",
            "text": "store this evidence",
        },
        {"message_id": "assistant-typed", "role": "assistant", "text": "stored"},
        source_refs=("untyped-source",),
        source_groups=MemoryWorkPayload(
            source_decision_ids=("decision-typed",),
            source_plan_ids=("plan-typed",),
        ),
        plan_version=9,
    )

    work = long_term.get_work(str(outcome["work_id"]))
    assert work is not None
    assert work.payload.source_event_ids == ("untyped-source",)
    assert work.payload.source_decision_ids == ("decision-typed",)
    assert work.payload.source_plan_ids == ("plan-typed",)
    assert work.payload.source_payload["revision"] == 9


def test_stream_event_source_groups_have_a_total_bound(tmp_path: Path) -> None:
    short_term = ShortTermContextRepository(tmp_path / "memory.db")
    long_term = LongTermMemoryRepository(tmp_path / "memory.db")
    service = MemoryService(short_term, long_term, RecordingRetriever(None))
    event = service.emit_worker_event(
        user_id="operator",
        conversation_id="conversation-bound",
        scenario_id="scenario-1",
        status=MemoryStreamStatus.PROCESSING,
        event_type=MemoryStreamEventType.MEMORY_FILTERED,
        work_id="work-bound",
        source_message_ids=tuple(f"message-{index}" for index in range(64)),
        source_event_ids=tuple(f"event-{index}" for index in range(64)),
        source_decision_ids=tuple(f"decision-{index}" for index in range(64)),
        source_knowledge_ids=tuple(f"knowledge-{index}" for index in range(64)),
        source_plan_ids=tuple(f"plan-{index}" for index in range(64)),
        plan_version=0,
    )

    total = sum(
        len(source_ids)
        for source_ids in (
            event.payload.source_message_ids,
            event.payload.source_event_ids,
            event.payload.source_decision_ids,
            event.payload.source_knowledge_ids,
            event.payload.source_plan_ids,
        )
    )
    assert total <= 64


def test_accept_turn_persists_every_rendered_assistant_message(tmp_path: Path) -> None:
    short_term = ShortTermContextRepository(tmp_path / "memory.db")
    long_term = LongTermMemoryRepository(tmp_path / "memory.db")
    service = MemoryService(short_term, long_term, RecordingRetriever(None))
    turn = {
        "user_id": "operator",
        "conversation_id": "conversation-1",
        "scenario_id": "scenario-1",
        "message_id": "message-1",
        "turn_id": "turn-1",
        "role": "expert",
        "text": "revise and explain",
    }
    result = (
        {"message_id": "message-1", "role": "expert", "text": "revise and explain"},
        {"message_id": "message-2", "role": "assistant", "text": "preview"},
        {"message_id": "message-3", "role": "assistant", "text": "evidence answer"},
    )

    service.accept_turn(turn, result)

    context = short_term.get_short_term("operator", "conversation-1", "scenario-1")
    assert context is not None
    assert [(item.message_id, item.role, item.text) for item in context.recent_messages] == [
        ("message-1", "expert", "revise and explain"),
        ("message-2", "assistant", "preview"),
        ("message-3", "assistant", "evidence answer"),
    ]


def test_accept_turn_rolls_back_messages_and_work_if_queued_event_cannot_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    service = MemoryService(short_term, long_term, RecordingRetriever(None))

    def fail_event(event: object) -> None:
        del event
        raise RuntimeError("stream unavailable")

    monkeypatch.setattr(long_term, "_insert_stream_event", fail_event)
    with pytest.raises(RuntimeError, match="stream unavailable"):
        service.accept_turn(
            {
                "user_id": "operator",
                "conversation_id": "conversation-1",
                "scenario_id": "scenario-1",
                "message_id": "message-1",
                "text": "atomic turn",
            },
            {"message_id": "message-2", "text": "atomic answer"},
        )

    assert short_term.get_short_term("operator", "conversation-1") is None
    assert long_term._conn.execute("SELECT COUNT(*) FROM memory_work_items").fetchone()[0] == 0
    assert long_term.list_stream_events("operator", "conversation-1") == []


def test_accept_turn_requires_scenario_provenance(tmp_path: Path) -> None:
    short_term = ShortTermContextRepository(tmp_path / "memory.db")
    long_term = LongTermMemoryRepository(tmp_path / "memory.db")
    service = MemoryService(short_term, long_term, RecordingRetriever(None))

    with pytest.raises(ValueError, match="scenario_id"):
        service.accept_turn(
            {
                "user_id": "operator",
                "conversation_id": "conversation-1",
                "message_id": "message-1",
                "text": "missing scenario",
            },
            None,
        )


def test_accept_turn_uses_scenario_scoped_source_key(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    service = MemoryService(short_term, long_term, RecordingRetriever(None))

    service.accept_turn(
        {
            "user_id": "operator",
            "conversation_id": "conversation-1",
            "scenario_id": "scenario-1",
            "message_id": "message-1",
            "text": "scenario scoped turn",
        },
        None,
    )

    row = long_term._conn.execute(
        "SELECT source_key, scenario_id FROM memory_work_items"
    ).fetchone()
    assert tuple(row) == (
        "conversation:scenario-1:conversation-1:message-1",
        "scenario-1",
    )


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
    assert second["work_id"] == first["work_id"]
    assert second["stream_cursor"] == first["stream_cursor"]
    context = short_term.get_short_term("operator", "conversation-1", "scenario-1")
    assert context is not None
    assert context.message_count == 2
    assert [message.message_id for message in context.recent_messages] == ["message-1", "message-2"]
    assert long_term._conn.execute("SELECT COUNT(*) FROM memory_work_items").fetchone()[0] == 1
    assert long_term.get_source_cursor(
        "operator", "scenario-1", "conversation:scenario-1:conversation-1"
    ) == 2


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
        ("scenario-1:runtime_event:event-bounded",),
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


def test_enqueue_observation_bounds_total_work_payload_bytes(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    service = MemoryService(short_term, long_term, RecordingRetriever(None))

    payload = {
        "event_id": "event-total-bounded",
        "event_type": "bearing",
        "target_id": "target-1",
        "sim_time_s": 10,
        "severity": "info",
        "summary": "s" * 1000,
        "scenario_id": "scenario-total-bounded",
        "plan_id": "plan-1",
        "revision": 2,
        "status": "active",
        "conversation_id": "conversation-1",
        "text": "t" * 4000,
    }

    assert service.enqueue_observation(
        {
            "source_id": "event-total-bounded",
            "source_type": "runtime_event",
            "scenario_id": "scenario-total-bounded",
            "user_id": "operator",
        },
        payload,
    )["status"] == "queued"

    row = long_term._conn.execute(
        "SELECT payload FROM memory_work_items WHERE source_key = ?",
        ("scenario-total-bounded:runtime_event:event-total-bounded",),
    ).fetchone()
    assert row is not None
    assert len(row["payload"].encode("utf-8")) <= 8192


def test_memory_work_payload_rejects_field_splitting_over_total_json_limit() -> None:
    with pytest.raises(ValueError, match="total JSON"):
        MemoryWorkPayload(
            source_type="runtime_event",
            source_text="x" * 4000,
            source_payload={f"field-{index}": "y" * 1000 for index in range(8)},
        )


def test_accept_turn_without_message_ids_is_deterministically_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    service = MemoryService(short_term, long_term, RecordingRetriever(None))
    turn = {
        "user_id": "operator",
        "conversation_id": "conversation-1",
        "scenario_id": "scenario-1",
        "turn_id": "turn-without-message-id",
        "text": "remember this stable turn",
    }
    result = {"text": "stable answer"}

    first = service.accept_turn(turn, result)
    second = service.accept_turn(turn, result)

    assert first["status"] == "queued"
    assert second["status"] == "duplicate"
    assert long_term._conn.execute("SELECT COUNT(*) FROM memory_work_items").fetchone()[0] == 1
    context = short_term.get_short_term("operator", "conversation-1", "scenario-1")
    assert context is not None
    assert len(context.recent_messages) == 2
    assert len({message.message_id for message in context.recent_messages}) == 2


def test_long_term_memory_retrieval_isolated_by_scenario(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    repository = LongTermMemoryRepository(database)
    for scenario_id, memory_id in (("scenario-a", "memory-a"), ("scenario-b", "memory-b")):
        repository.create_memory_version(
            MemoryVersion(
                memory_id=memory_id,
                memory_family_id=f"family-{scenario_id}",
                version=1,
                user_id="operator",
                scenario_id=scenario_id,
                memory_type=MemoryType.SEMANTIC,
                summary=f"memory from {scenario_id}",
                importance_score=0.8,
                embedding=(1.0,),
            ),
            expected_previous_version=0,
        )

    assert [item.memory_id for item in repository.list_active("operator", {"scenario_id": "scenario-a"})] == [
        "memory-a"
    ]
