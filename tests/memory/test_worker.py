from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from time import monotonic
import sqlite3

from underwater_tracking.agent.llm import TransientLLMError
from underwater_tracking.config.models import MemoryConfig
from underwater_tracking.domain.memory_models import (
    MemoryExtractionResult,
    MemoryFilterDecision,
    MemoryType,
    MemoryWorkItem,
    MemoryWorkPayload,
    MemoryWorkType,
    MemoryStatus,
    MemoryVersion,
    ShortTermCompressionResult,
    ShortTermMessage,
)
from underwater_tracking.domain.models import EventAudience
from underwater_tracking.memory.service import MemoryService
from underwater_tracking.memory.worker import MemoryWorker
from underwater_tracking.memory.embeddings import EmbeddingResult
from underwater_tracking.persistence.memory import (
    LongTermMemoryRepository,
    ShortTermContextRepository,
)
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.memory.source_reader import MemorySource
from underwater_tracking.memory.source_reader import MemorySourceReader


class NoopRetriever:
    def retrieve(self, **kwargs):
        del kwargs
        raise AssertionError("worker must not retrieve context")


class RecordingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> EmbeddingResult:
        self.calls += 1
        assert text == "source text"
        return EmbeddingResult(
            vector=(0.25, 0.75), model="embedding-test-v1", vector_version="test-v2"
        )


class AnyTextEmbedder:
    def embed(self, text: str) -> EmbeddingResult:
        assert text
        return EmbeddingResult(
            vector=(0.25, 0.75), model="embedding-test-v1", vector_version="test-v2"
        )


class RecordingReasoner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def filter(self, **kwargs):
        self.calls.append("filter")
        return MemoryFilterDecision(
            should_store=True,
            memory_type=MemoryType.SEMANTIC,
            operation="create",
            importance_score=0.8,
            reason="real reasoner selected durable material",
        )

    def extract(self, **kwargs):
        self.calls.append("extract")
        return MemoryExtractionResult(
            summary="source text", source_message_ids=("message-1",), change_reason="created"
        )

    def compress_short_term(self, context):
        self.calls.append("compress")
        return ShortTermCompressionResult(
            summary_text=context.recent_messages[-1].text,
            retained_messages=context.recent_messages[-1:],
            source_message_ids=(context.recent_messages[-1].message_id,),
        )


class FilteredReasoner(RecordingReasoner):
    def filter(self, **kwargs):
        self.calls.append("filter")
        return MemoryFilterDecision(
            should_store=False,
            operation="ignore",
            reason="the real memory filter rejected this transient turn",
        )


class TransientFailureReasoner(RecordingReasoner):
    def filter(self, **kwargs):
        self.calls.append("filter")
        raise TransientLLMError("provider temporarily unavailable")


class CompressionFailsOnceReasoner(RecordingReasoner):
    def __init__(self) -> None:
        super().__init__()
        self.compression_attempts = 0

    def compress_short_term(self, context):
        self.compression_attempts += 1
        if self.compression_attempts == 1:
            raise TransientLLMError("compression temporarily unavailable")
        return super().compress_short_term(context)


class PlanRecordingReasoner(RecordingReasoner):
    def __init__(self) -> None:
        super().__init__()
        self.filter_source_texts: tuple[str, ...] = ()
        self.filter_source_knowledge_ids: tuple[str, ...] = ()

    def filter(self, **kwargs):
        self.filter_source_texts = tuple(kwargs["source_texts"])
        self.filter_source_knowledge_ids = tuple(kwargs["source_knowledge_ids"])
        return super().filter(**kwargs)


class ProvenanceRecordingReasoner(RecordingReasoner):
    def __init__(self) -> None:
        super().__init__()
        self.filter_message_ids: tuple[str, ...] = ()
        self.extract_message_ids: tuple[str, ...] = ()

    def filter(self, **kwargs):
        self.filter_message_ids = tuple(kwargs["source_message_ids"])
        return super().filter(**kwargs)

    def extract(self, **kwargs):
        self.extract_message_ids = tuple(kwargs["source_message_ids"])
        return MemoryExtractionResult(
            summary="source text",
            source_message_ids=self.extract_message_ids,
            change_reason="created",
        )


class ObservationRecordingReasoner(RecordingReasoner):
    def __init__(self) -> None:
        super().__init__()
        self.filter_source_texts: tuple[str, ...] = ()
        self.filter_event_ids: tuple[str, ...] = ()

    def filter(self, **kwargs):
        self.filter_source_texts = tuple(kwargs["source_texts"])
        self.filter_event_ids = tuple(kwargs["source_event_ids"])
        return super().filter(**kwargs)

    def extract(self, **kwargs):
        return MemoryExtractionResult(
            summary="keep this source text",
            source_event_ids=tuple(kwargs["source_event_ids"]),
            change_reason="created",
        )


class BlockingReasoner(RecordingReasoner):
    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def filter(self, **kwargs):
        self.entered.set()
        self.release.wait()
        return MemoryFilterDecision(should_store=False, operation="ignore", reason="ignored")


class RecordingSourceReader:
    def __init__(self, source: MemorySource) -> None:
        self.source = source
        self.read_calls = 0

    def read_new(self, user_id: str, scenario_id: str):
        del user_id, scenario_id
        self.read_calls += 1
        return (self.source,)

    def load_work_sources(
        self,
        user_id: str,
        scenario_id: str | None,
        payload: object,
        *,
        conversation_id: str | None = None,
    ):
        del user_id, scenario_id, payload, conversation_id
        return ()


class PagingSourceReader:
    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def read_new(self, user_id: str, scenario_id: str):
        self.seen.append((user_id, scenario_id))
        return ()

    def load_work_sources(
        self,
        user_id: str,
        scenario_id: str | None,
        payload: object,
        *,
        conversation_id: str | None = None,
    ):
        del user_id, scenario_id, payload, conversation_id
        return ()


def _config(**updates: object) -> MemoryConfig:
    values: dict[str, object] = {
        "embedding_base_url": "https://api.example.test/v1",
        "embedding_model": "embedding-test-v1",
        "short_term_message_threshold": 1,
        "short_term_token_threshold": 1,
        "short_term_compress_interval_s": 1.0,
        "poll_interval_s": 0.01,
        "work_lease_timeout_s": 1.0,
    }
    values.update(updates)
    return MemoryConfig(**values)


def test_worker_uses_independent_source_and_maintenance_clocks(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    source_reader = PagingSourceReader()
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        RecordingReasoner(),
        source_reader,
        _config(source_poll_interval_s=2.0, maintenance_interval_s=300.0),
        "worker-independent-clocks",
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)

    assert worker.poll_once(now=start) is False
    long_term.register_source_scope("operator", "scenario-clock")
    assert worker.poll_once(now=start + timedelta(seconds=1)) is False
    assert worker.poll_once(now=start + timedelta(seconds=2)) is False
    assert worker.poll_once(now=start + timedelta(seconds=299)) is False
    assert worker.poll_once(now=start + timedelta(seconds=300)) is True
    maintenance = long_term._conn.execute(
        "SELECT COUNT(*) FROM memory_work_items WHERE work_type = 'maintenance'"
    ).fetchone()[0]
    assert maintenance == 1
    assert len(source_reader.seen) == 2


def test_worker_processes_real_reasoner_steps_and_compresses_after_threshold(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    short_term.append_messages(
        "operator",
        "conversation-1",
        (
            ShortTermMessage(
                message_id="message-1", scenario_id="scenario-1", role="user", text="source text"
            ),
        ),
        scenario_id="scenario-1",
    )
    work = MemoryWorkItem(
        work_id="work-1",
        user_id="operator",
        conversation_id="conversation-1",
        scenario_id="scenario-1",
        work_type=MemoryWorkType.CONVERSATION_TURN,
        payload=MemoryWorkPayload(source_message_ids=("message-1",)),
    )
    assert long_term.enqueue_work(work, "conversation:message-1")
    reasoner = RecordingReasoner()
    service = MemoryService(short_term, long_term, NoopRetriever())
    embedder = RecordingEmbedder()
    worker = MemoryWorker(
        long_term, service, reasoner, None, _config(), "worker-1", embedding_provider=embedder
    )

    assert worker.poll_once(now=datetime.now(UTC)) is True

    assert reasoner.calls == ["filter", "extract", "compress"]
    persisted = long_term.list_active("operator", limit=1)[0]
    assert persisted.summary == "source text"
    assert persisted.embedding == (0.25, 0.75)
    assert persisted.embedding_version == "test-v2"
    compressed = short_term.get_short_term("operator", "conversation-1", "scenario-1")
    assert compressed is not None and compressed.summary_version == 1
    events = long_term.list_stream_events(
        "operator", "conversation-1", scenario_id="scenario-1", limit=20
    )
    assert [event.type.value for event in events] == [
        "work_processing",
        "memory_filtered",
        "memory_extracted",
        "memory_version_created",
        "short_term_compression_started",
        "short_term_compressed",
        "work_completed",
    ]
    filtered = next(event for event in events if event.type.value == "memory_filtered")
    assert filtered.payload.source_message_ids == ("message-1",)
    assert filtered.payload.source_ids == ("message-1",)
    compressed_event = next(
        event for event in events if event.type.value == "short_term_compressed"
    )
    assert compressed_event.payload.source_message_ids == ("message-1",)


def test_worker_starts_a_new_compression_window_after_compression(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    short_term.append_messages(
        "operator",
        "conversation-1",
        tuple(
            ShortTermMessage(
                message_id=f"message-{index}",
                scenario_id="scenario-1",
                role="user",
                text="source text",
            )
            for index in (1, 2)
        ),
        scenario_id="scenario-1",
    )
    reasoner = RecordingReasoner()
    service = MemoryService(short_term, long_term, NoopRetriever())
    worker = MemoryWorker(
        long_term,
        service,
        reasoner,
        None,
        _config(
            short_term_message_threshold=2,
            short_term_token_threshold=999,
            short_term_compress_interval_s=999.0,
        ),
        "worker-compression-window",
        embedding_provider=RecordingEmbedder(),
    )

    first_work = MemoryWorkItem(
        work_id="work-compression-window-1",
        user_id="operator",
        conversation_id="conversation-1",
        scenario_id="scenario-1",
        work_type=MemoryWorkType.CONVERSATION_TURN,
        payload=MemoryWorkPayload(source_message_ids=("message-1", "message-2")),
    )
    assert long_term.enqueue_work(first_work, "conversation:message-1:message-2")
    assert worker.poll_once(now=datetime.now(UTC)) is True

    compressed = short_term.get_short_term("operator", "conversation-1", "scenario-1")
    assert compressed is not None
    assert compressed.message_count == 2
    assert compressed.compressed_message_count == 2

    short_term.append_messages(
        "operator",
        "conversation-1",
        (
            ShortTermMessage(
                message_id="message-3",
                scenario_id="scenario-1",
                role="user",
                text="source text",
            ),
        ),
        scenario_id="scenario-1",
    )
    second_work = first_work.model_copy(
        update={
            "work_id": "work-compression-window-2",
            "payload": MemoryWorkPayload(source_message_ids=("message-1", "message-3")),
        }
    )
    assert long_term.enqueue_work(second_work, "conversation:message-3")
    assert worker.poll_once(now=datetime.now(UTC)) is True

    assert reasoner.calls == ["filter", "extract", "compress", "filter", "extract"]
    updated = short_term.get_short_term("operator", "conversation-1", "scenario-1")
    assert updated is not None
    assert updated.message_count == 3
    assert updated.compressed_message_count == 2


def test_worker_skips_routine_memory_sources_and_advances_runtime_cursor(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    events = EventRepository(database)
    events.append(
        event_id="routine-bearing",
        event_type="bearing",
        scenario_id="scenario-memory-policy",
        sim_time_s=30,
        payload={"summary": "routine bearing"},
    )
    events.append(
        event_id="key-intent",
        event_type="target_intent_changed",
        scenario_id="scenario-memory-policy",
        sim_time_s=60,
        target_id="T1",
        severity="strategic",
        payload={
            "previous_label": "transit",
            "label": "evade",
            "confidence": 0.9,
            "observation_ids": ["observation-1"],
            "evidence_ids": ["evidence-1"],
            "source": "public_estimate",
        },
    )
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        RecordingReasoner(),
        MemorySourceReader(long_term, event_repository=events),
        _config(maintenance_interval_s=300.0),
        "worker-memory-policy",
    )

    assert worker.poll_once(now=datetime.now(UTC)) is True

    assert long_term.get_source_cursor("operator", "scenario-memory-policy", "runtime_event") == 2
    rows = long_term._conn.execute(
        "SELECT source_key FROM memory_work_items WHERE work_type = 'observation'"
    ).fetchall()
    assert [row[0] for row in rows] == ["runtime_event:scenario-memory-policy:key-intent"]


def test_worker_skips_non_memory_audience_events_without_starving_later_key_events(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    events = EventRepository(database)
    for index in (1, 2):
        events.append(
            event_id=f"audit-only-{index}",
            event_type="audit_only_event",
            scenario_id="scenario-memory-policy",
            sim_time_s=index,
            audiences=frozenset({EventAudience.OPERATOR_AUDIT}),
            payload={"summary": "operator audit"},
        )
    events.append(
        event_id="intent-key",
        event_type="target_intent_changed",
        scenario_id="scenario-memory-policy",
        sim_time_s=3,
        target_id="T1",
        payload={"label": "evade", "confidence": 0.9},
    )
    reasoner = ObservationRecordingReasoner()
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        reasoner,
        MemorySourceReader(long_term, event_repository=events, batch_limit=2),
        _config(maintenance_interval_s=300.0),
        "worker-audience-cursor",
    )

    now = datetime.now(UTC)
    assert worker.poll_once(now=now) is False
    assert long_term.get_source_cursor("operator", "scenario-memory-policy", "runtime_event") == 2
    assert worker.poll_once(now=now + timedelta(seconds=3)) is True
    assert worker.poll_once(now=now + timedelta(seconds=4)) is True
    assert reasoner.filter_event_ids == ("intent-key",)


def test_worker_retries_without_processing_when_any_conversation_source_is_missing(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    short_term.append_messages(
        "operator",
        "conversation-1",
        tuple(
            ShortTermMessage(
                message_id=f"message-{index}",
                scenario_id="scenario-1",
                role="user",
                text="source text",
            )
            for index in range(128)
        ),
        scenario_id="scenario-1",
    )
    work = MemoryWorkItem(
        work_id="work-evicted-provenance",
        user_id="operator",
        conversation_id="conversation-1",
        scenario_id="scenario-1",
        work_type=MemoryWorkType.CONVERSATION_TURN,
        payload=MemoryWorkPayload(source_message_ids=("message-evicted", "message-127")),
    )
    assert long_term.enqueue_work(work, "conversation:evicted-provenance")
    reasoner = ProvenanceRecordingReasoner()
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        reasoner,
        MemorySourceReader(long_term, short_term_repository=short_term),
        _config(short_term_message_threshold=999, short_term_token_threshold=999),
        "worker-evicted-provenance",
        embedding_provider=RecordingEmbedder(),
    )

    assert worker.poll_once(now=datetime.now(UTC)) is True

    assert reasoner.calls == []
    assert reasoner.filter_message_ids == ()
    assert reasoner.extract_message_ids == ()
    assert long_term.list_active("operator", limit=1) == []
    work_row = long_term._conn.execute(
        "SELECT status, last_error FROM memory_work_items WHERE work_id = ?",
        ("work-evicted-provenance",),
    ).fetchone()
    assert work_row["status"] == "pending"
    assert "source_message_ids" in work_row["last_error"]
    degraded = long_term.list_stream_events(
        "operator", "conversation-1", scenario_id="scenario-1", limit=20
    )
    assert any(
        event.type.value == "source_read_degraded"
        and event.status.value == "degraded"
        and "message-evicted" in event.payload.source_ids
        for event in degraded
    )
    assert any(event.type.value == "work_retry_scheduled" for event in degraded)


def test_worker_rejects_message_provenance_without_conversation_scope_before_fallback(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    work = MemoryWorkItem(
        work_id="work-unscoped-message",
        user_id="operator",
        scenario_id="scenario-1",
        work_type=MemoryWorkType.CONVERSATION_TURN,
        payload=MemoryWorkPayload(
            source_message_ids=("message-1",),
            source_text="must not be treated as authoritative conversation evidence",
        ),
    )
    assert long_term.enqueue_work(work, "conversation:unscoped-message")
    reasoner = RecordingReasoner()
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        reasoner,
        None,
        _config(short_term_message_threshold=999, short_term_token_threshold=999),
        "worker-unscoped-message",
        embedding_provider=RecordingEmbedder(),
    )

    assert worker.poll_once(now=datetime.now(UTC)) is True

    assert reasoner.calls == []
    assert long_term.list_active("operator", limit=1) == []
    assert any(
        row[0] == "source_read_degraded"
        for row in long_term._conn.execute(
            "SELECT type FROM memory_stream_events WHERE user_id = ? AND scenario_id = ?",
            ("operator", "scenario-1"),
        )
    )


def test_worker_consumes_persisted_observation_projection(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    service = MemoryService(short_term, long_term, NoopRetriever())
    assert (
        service.enqueue_observation(
            {
                "source_id": "event-observation-work",
                "source_type": "runtime_event",
                "scenario_id": "scenario-1",
                "user_id": "operator",
            },
            {"event_id": "event-observation-work", "summary": "keep this source text"},
        )["status"]
        == "queued"
    )
    reasoner = ObservationRecordingReasoner()
    worker = MemoryWorker(
        long_term,
        service,
        reasoner,
        MemorySourceReader(long_term),
        _config(short_term_message_threshold=999, short_term_token_threshold=999),
        "worker-observation-work",
        embedding_provider=AnyTextEmbedder(),
    )

    assert worker.poll_once(now=datetime.now(UTC)) is True

    assert reasoner.filter_source_texts == ("keep this source text",)
    assert reasoner.filter_event_ids == ("event-observation-work",)
    assert long_term.list_active("operator", limit=1)[0].source_event_ids == (
        "event-observation-work",
    )


def test_worker_stop_is_idempotent_and_thread_exits_within_timeout(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        RecordingReasoner(),
        None,
        _config(),
        "worker-stop",
        stop_event=Event(),
    )

    worker.start()
    worker.stop(timeout=1.0)
    worker.stop(timeout=1.0)

    assert worker.is_running is False


def test_worker_uses_reasoner_filter_result_without_keyword_rules(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    short_term.append_messages(
        "operator",
        "conversation-1",
        (
            ShortTermMessage(
                message_id="message-1", scenario_id="scenario-1", role="user", text="thank you"
            ),
        ),
        scenario_id="scenario-1",
    )
    work = MemoryWorkItem(
        work_id="work-filtered",
        user_id="operator",
        conversation_id="conversation-1",
        scenario_id="scenario-1",
        work_type=MemoryWorkType.CONVERSATION_TURN,
        payload=MemoryWorkPayload(source_message_ids=("message-1",)),
    )
    assert long_term.enqueue_work(work, "conversation:filtered")
    reasoner = FilteredReasoner()
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        reasoner,
        None,
        _config(),
        "worker-filter",
    )

    assert worker.poll_once(now=datetime.now(UTC)) is True

    assert reasoner.calls == ["filter", "compress"]
    assert long_term.list_active("operator", limit=10) == []


def test_worker_retries_transient_failure_then_degrades_at_attempt_bound(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    work = MemoryWorkItem(
        work_id="work-retry",
        user_id="operator",
        work_type=MemoryWorkType.OBSERVATION,
    )
    assert long_term.enqueue_work(work, "event:retry")
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        TransientFailureReasoner(),
        None,
        _config(max_attempts=2, retry_backoff_s=1.0),
        "worker-retry",
    )
    now = datetime.now(UTC)

    assert worker.poll_once(now=now) is True
    first = long_term._conn.execute(
        "SELECT status, attempts, last_error FROM memory_work_items WHERE work_id = ?",
        ("work-retry",),
    ).fetchone()
    assert tuple(first) == ("pending", 1, "provider temporarily unavailable")
    assert worker.poll_once(now=now) is False
    assert worker.poll_once(now=now.replace(year=now.year + 1)) is True
    final = long_term._conn.execute(
        "SELECT status, attempts, last_error FROM memory_work_items WHERE work_id = ?",
        ("work-retry",),
    ).fetchone()
    assert tuple(final) == ("degraded", 2, "provider temporarily unavailable")


def test_filter_rejection_does_not_emit_extracted_event(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    short_term.append_messages(
        "operator",
        "conversation-1",
        (
            ShortTermMessage(
                message_id="message-1", scenario_id="scenario-1", role="user", text="thank you"
            ),
        ),
        scenario_id="scenario-1",
    )
    work = MemoryWorkItem(
        work_id="work-filtered-event",
        user_id="operator",
        conversation_id="conversation-1",
        scenario_id="scenario-1",
        work_type=MemoryWorkType.CONVERSATION_TURN,
        payload=MemoryWorkPayload(source_message_ids=("message-1",)),
    )
    assert long_term.enqueue_work(work, "conversation:filtered-event")
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        FilteredReasoner(),
        None,
        _config(),
        "worker-filtered-event",
    )

    assert worker.poll_once(now=datetime.now(UTC)) is True

    events = long_term.list_stream_events(
        "operator", "conversation-1", scenario_id="scenario-1", limit=20
    )
    assert all(event.type.value != "memory_extracted" for event in events)
    assert any(event.type.value == "memory_filtered" for event in events)


def test_plan_source_is_resolved_and_passed_to_reasoner(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    plans = PlanRepository(database)
    plans.set_snapshot_revision("scenario-1", 1)
    from underwater_tracking.domain.agent_models import TrackingPlan

    plan = TrackingPlan(
        plan_id="plan-1",
        scenario_id="scenario-1",
        revision=1,
        base_snapshot_revision=1,
        status="active",
        concept="hold_current",
        member_ids_by_target={"T1": ("U1", "U2")},
        roles_by_member={"U1": "passive_track", "U2": "active_scan"},
        active_uuv_ids=("U2",),
        standby_uuv_ids=("U1",),
    )
    plans.commit(plan)
    work = MemoryWorkItem(
        work_id="work-plan-source",
        user_id="operator",
        scenario_id="scenario-1",
        work_type=MemoryWorkType.OBSERVATION,
        payload=MemoryWorkPayload(source_knowledge_ids=("plan-1",)),
    )
    assert long_term.enqueue_work(work, "plan:plan-1:1")
    reasoner = PlanRecordingReasoner()
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        reasoner,
        MemorySourceReader(long_term, plan_repository=plans),
        _config(),
        "worker-plan-source",
        embedding_provider=RecordingEmbedder(),
    )

    assert worker.poll_once(now=datetime.now(UTC)) is True

    assert reasoner.filter_source_knowledge_ids == ("plan-1",)
    assert reasoner.filter_source_texts
    assert "hold_current" in reasoner.filter_source_texts[0]
    assert "member_ids_by_target" in reasoner.filter_source_texts[0]
    assert "U1" in reasoner.filter_source_texts[0]


def test_plan_source_must_be_current_active_plan_in_same_scenario(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    long_term = LongTermMemoryRepository(database)
    plans = PlanRepository(database)
    plans.set_snapshot_revision("scenario-1", 1)
    from underwater_tracking.domain.agent_models import TrackingPlan

    old_plan = TrackingPlan(
        plan_id="plan-old",
        scenario_id="scenario-1",
        revision=1,
        base_snapshot_revision=1,
        status="active",
        concept="hold_current",
    )
    current_plan = TrackingPlan(
        plan_id="plan-current",
        scenario_id="scenario-1",
        revision=2,
        base_snapshot_revision=1,
        status="active",
        concept="quality_first",
    )
    plans.commit(old_plan)
    plans.commit(current_plan)
    reader = MemorySourceReader(long_term, plan_repository=plans)
    payload = MemoryWorkPayload(source_knowledge_ids=("plan-old",))

    assert reader.load_work_sources("operator", "scenario-1", payload) == ()


def test_maintenance_decays_and_archives_low_weight_memory_persistently(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    created_at = datetime.now(UTC) - timedelta(seconds=100)
    memory = MemoryVersion(
        memory_id="memory-old",
        memory_family_id="family-old",
        version=1,
        user_id="operator",
        memory_type=MemoryType.SEMANTIC,
        summary="old memory",
        importance_score=0.8,
        created_at=created_at,
    )
    long_term.create_memory_version(memory, expected_previous_version=0)
    maintenance_at = created_at + timedelta(seconds=100)
    work = MemoryWorkItem(
        work_id="work-maintenance",
        user_id="operator",
        work_type=MemoryWorkType.MAINTENANCE,
        available_at=maintenance_at,
    )
    assert long_term.enqueue_work(work, "maintenance:operator:1")
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        RecordingReasoner(),
        None,
        _config(decay_half_life_s=10.0, archive_threshold=0.1),
        "worker-maintenance",
    )

    assert worker.poll_once(now=maintenance_at) is True

    row = long_term._conn.execute(
        "SELECT status, importance_score FROM long_term_memories WHERE memory_id = ?",
        ("memory-old",),
    ).fetchone()
    assert row["status"] == MemoryStatus.ARCHIVED.value
    assert row["importance_score"] < 0.1


def test_compression_retry_after_version_creation_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    short_term.append_messages(
        "operator",
        "conversation-1",
        (
            ShortTermMessage(
                message_id="message-1", scenario_id="scenario-1", role="user", text="source text"
            ),
        ),
        scenario_id="scenario-1",
    )
    work = MemoryWorkItem(
        work_id="work-compression-retry",
        user_id="operator",
        conversation_id="conversation-1",
        scenario_id="scenario-1",
        work_type=MemoryWorkType.CONVERSATION_TURN,
        payload=MemoryWorkPayload(source_message_ids=("message-1",)),
    )
    assert long_term.enqueue_work(work, "conversation:compression-retry")
    reasoner = CompressionFailsOnceReasoner()
    embedder = RecordingEmbedder()
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        reasoner,
        None,
        _config(max_attempts=2, retry_backoff_s=0.01),
        "worker-compression-retry",
        embedding_provider=embedder,
    )
    now = datetime.now(UTC)

    assert worker.poll_once(now=now) is True
    assert len(long_term.list_versions("operator", "family:work-compression-retry")) == 1
    assert worker.poll_once(now=now + timedelta(seconds=1)) is True

    context = short_term.get_short_term("operator", "conversation-1", "scenario-1")
    assert context is not None and context.summary_version == 1
    assert len(long_term.list_versions("operator", "family:work-compression-retry")) == 1
    assert reasoner.calls == ["filter", "extract", "compress"]
    assert reasoner.compression_attempts == 2
    assert embedder.calls == 1
    events = long_term.list_stream_events(
        "operator", "conversation-1", scenario_id="scenario-1", limit=20
    )
    assert [event.type.value for event in events].count("compression_degraded") == 1
    assert [event.type.value for event in events].count("work_completed") == 1


def test_worker_claims_work_before_reading_new_sources(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    work = MemoryWorkItem(
        work_id="work-before-source",
        user_id="operator",
        scenario_id="scenario-1",
        work_type=MemoryWorkType.MAINTENANCE,
    )
    assert long_term.enqueue_work(work, "maintenance:before-source")
    source_reader = RecordingSourceReader(
        MemorySource(
            source_key="runtime_event:event-1",
            source_type="runtime_event",
            cursor=1,
            payload={"event_id": "event-1"},
            text="event source",
            source_event_ids=("event-1",),
        )
    )
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        RecordingReasoner(),
        source_reader,
        _config(maintenance_interval_s=0.01),
        "worker-before-source",
    )

    assert worker.poll_once(now=datetime.now(UTC)) is True
    assert source_reader.read_calls == 0
    assert worker.poll_once(now=datetime.now(UTC)) is True
    assert source_reader.read_calls == 1


def test_worker_cold_start_discovers_existing_event_without_claimed_work(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    events = EventRepository(database)
    events.append(
        event_id="event-cold-start",
        event_type="bearing",
        scenario_id="scenario-cold-start",
        sim_time_s=1,
        payload={"summary": "cold start event"},
    )
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        RecordingReasoner(),
        MemorySourceReader(long_term, event_repository=events),
        _config(maintenance_interval_s=0.001),
        "worker-cold-start",
    )

    assert worker.poll_once(now=datetime.now(UTC)) is False
    assert long_term.get_source_cursor("operator", "scenario-cold-start", "runtime_event") == 1
    assert (
        long_term._conn.execute(
            "SELECT COUNT(*) FROM memory_work_items WHERE work_type = 'observation'"
        ).fetchone()[0]
        == 0
    )


def test_worker_consumes_periodic_summary_text_without_reconstructing_payload(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    events = EventRepository(database)
    events.append(
        event_id="periodic_situation_summary:scenario-summary:600",
        event_type="periodic_situation_summary",
        scenario_id="scenario-summary",
        sim_time_s=600,
        payload={
            "summary": "time=600; plan=4; regions=R1:ACTIVE_SCAN:0.80",
            "source_event_ids": ["raw-1", "raw-2"],
        },
    )
    reasoner = ObservationRecordingReasoner()
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        reasoner,
        MemorySourceReader(long_term, event_repository=events),
        _config(maintenance_interval_s=0.001),
        "worker-periodic-summary",
    )

    now = datetime.now(UTC)
    assert worker.poll_once(now=now) is True
    assert worker.poll_once(now=now + timedelta(seconds=1)) is True

    assert reasoner.filter_source_texts == ("time=600; plan=4; regions=R1:ACTIVE_SCAN:0.80",)
    assert reasoner.filter_event_ids == ("periodic_situation_summary:scenario-summary:600",)
    assert long_term.get_source_cursor("operator", "scenario-summary", "runtime_event") == 1
    assert long_term._conn.execute("SELECT COUNT(*) FROM memory_work_items").fetchone()[0] == 1

    restarted_worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        ObservationRecordingReasoner(),
        MemorySourceReader(long_term, event_repository=events),
        _config(maintenance_interval_s=0.001),
        "worker-periodic-summary-restarted",
    )
    assert restarted_worker.poll_once(now=now + timedelta(seconds=2)) is False
    assert long_term._conn.execute("SELECT COUNT(*) FROM memory_work_items").fetchone()[0] == 1


def test_worker_reads_a_bounded_persistent_scope_page_across_rounds(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    for index in range(65):
        long_term.register_source_scope("operator", f"scenario-page-{index:02d}")
    source_reader = PagingSourceReader()
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        RecordingReasoner(),
        source_reader,
        _config(source_poll_interval_s=0.001, maintenance_interval_s=300.0),
        "worker-bounded-scope-page",
    )

    now = datetime.now(UTC)
    assert worker.poll_once(now=now) is False
    first_page = tuple(source_reader.seen)
    assert len(first_page) <= 32
    assert len(first_page) == 32
    assert worker.poll_once(now=now + timedelta(seconds=1)) is False
    second_page = tuple(source_reader.seen[len(first_page) :])

    assert len(second_page) == 32
    assert set(first_page).isdisjoint(second_page)
    assert not hasattr(worker, "_source_scopes")


def test_source_read_failure_is_degraded_without_cursor_advance_and_retries(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    events = EventRepository(database)
    events.append(
        event_id="event-retry-source",
        event_type="bearing",
        scenario_id="scenario-retry-source",
        sim_time_s=1,
        payload={"summary": "retry source"},
    )
    reader = MemorySourceReader(long_term, event_repository=events)
    long_term.register_source_scope("operator", "scenario-retry-source")
    original = events.list_events

    def fail_once(*args, **kwargs):
        monkeypatch.setattr(events, "list_events", original)
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(events, "list_events", fail_once)
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        RecordingReasoner(),
        reader,
        _config(maintenance_interval_s=0.001),
        "worker-source-retry",
    )

    assert worker.poll_once(now=datetime.now(UTC)) is False
    assert long_term.get_source_cursor("operator", "scenario-retry-source", "runtime_event") == 0
    degraded = long_term._conn.execute(
        "SELECT type, status FROM memory_stream_events WHERE type = 'source_read_degraded'"
    ).fetchall()
    assert degraded and tuple(degraded[0]) == ("source_read_degraded", "degraded")
    assert worker.poll_once(now=datetime.now(UTC)) is False
    assert long_term.get_source_cursor("operator", "scenario-retry-source", "runtime_event") == 1


def test_worker_does_not_emit_completed_when_lease_is_lost(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    work = MemoryWorkItem(
        work_id="work-lease-lost", user_id="operator", work_type=MemoryWorkType.OBSERVATION
    )
    assert long_term.enqueue_work(work, "event:lease-lost")
    monkeypatch.setattr(long_term, "complete_work", lambda work_id, worker_id: False)
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        FilteredReasoner(),
        None,
        _config(),
        "worker-lease-lost",
    )

    assert worker.poll_once(now=datetime.now(UTC)) is True
    events = long_term._conn.execute(
        "SELECT type, status FROM memory_stream_events WHERE payload LIKE '%work-lease-lost%'"
    ).fetchall()
    assert all(row[0] != "work_completed" for row in events)
    assert any(row[0] == "work_degraded" and row[1] == "degraded" for row in events)


def test_stop_returns_within_timeout_while_sync_reasoner_is_blocked(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    work = MemoryWorkItem(
        work_id="work-blocking-stop", user_id="operator", work_type=MemoryWorkType.OBSERVATION
    )
    assert long_term.enqueue_work(work, "event:blocking-stop")
    reasoner = BlockingReasoner()
    worker = MemoryWorker(
        long_term,
        MemoryService(short_term, long_term, NoopRetriever()),
        reasoner,
        None,
        _config(),
        "worker-blocking-stop",
    )

    worker.start()
    assert reasoner.entered.wait(timeout=1.0)
    started = monotonic()
    stopped = worker.stop(timeout=0.01)
    elapsed = monotonic() - started
    assert stopped is False
    assert elapsed < 0.2
    reasoner.release.set()
    assert worker.stop(timeout=1.0) is True
