from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from threading import Event

from underwater_tracking.agent.llm import TransientLLMError
from underwater_tracking.config.models import MemoryConfig
from underwater_tracking.domain.memory_models import (
    MemoryExtractionResult,
    MemoryFilterDecision,
    MemoryStreamStatus,
    MemoryType,
    MemoryWorkItem,
    MemoryWorkPayload,
    MemoryWorkType,
    ShortTermCompressionResult,
    ShortTermMessage,
)
from underwater_tracking.memory.service import MemoryService
from underwater_tracking.memory.worker import MemoryWorker
from underwater_tracking.memory.embeddings import EmbeddingResult
from underwater_tracking.persistence.memory import LongTermMemoryRepository, ShortTermContextRepository


class NoopRetriever:
    def retrieve(self, **kwargs):
        del kwargs
        raise AssertionError("worker must not retrieve context")


class RecordingEmbedder:
    def embed(self, text: str) -> EmbeddingResult:
        assert text == "source text"
        return EmbeddingResult(vector=(0.25, 0.75), model="embedding-test-v1", vector_version="test-v2")


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


def test_worker_processes_real_reasoner_steps_and_compresses_after_threshold(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    short_term.append_messages(
        "operator", "conversation-1", (ShortTermMessage(message_id="message-1", role="user", text="source text"),)
    )
    work = MemoryWorkItem(
        work_id="work-1",
        user_id="operator",
        conversation_id="conversation-1",
        work_type=MemoryWorkType.CONVERSATION_TURN,
        payload=MemoryWorkPayload(source_message_ids=("message-1",)),
    )
    assert long_term.enqueue_work(work, "conversation:message-1")
    reasoner = RecordingReasoner()
    service = MemoryService(short_term, long_term, NoopRetriever())
    worker = MemoryWorker(
        long_term, service, reasoner, None, _config(), "worker-1", embedding_provider=RecordingEmbedder()
    )

    assert worker.poll_once(now=datetime.now(UTC)) is True

    assert reasoner.calls == ["filter", "extract", "compress"]
    persisted = long_term.list_active("operator", limit=1)[0]
    assert persisted.summary == "source text"
    assert persisted.embedding == (0.25, 0.75)
    assert persisted.embedding_version == "test-v2"
    compressed = short_term.get_short_term("operator", "conversation-1")
    assert compressed is not None and compressed.summary_version == 1
    events = long_term.list_stream_events("operator", "conversation-1", limit=20)
    assert [event.status for event in events][-2:] == [
        MemoryStreamStatus.PROCESSING,
        MemoryStreamStatus.COMPLETED,
    ]


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
        "operator", "conversation-1", (ShortTermMessage(message_id="message-1", role="user", text="thank you"),)
    )
    work = MemoryWorkItem(
        work_id="work-filtered",
        user_id="operator",
        conversation_id="conversation-1",
        work_type=MemoryWorkType.CONVERSATION_TURN,
        payload=MemoryWorkPayload(source_message_ids=("message-1",)),
    )
    assert long_term.enqueue_work(work, "conversation:filtered")
    reasoner = FilteredReasoner()
    worker = MemoryWorker(long_term, MemoryService(short_term, long_term, NoopRetriever()), reasoner, None, _config(), "worker-filter")

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
        long_term, MemoryService(short_term, long_term, NoopRetriever()), TransientFailureReasoner(), None,
        _config(max_attempts=2, retry_backoff_s=1.0), "worker-retry"
    )
    now = datetime.now(UTC)

    assert worker.poll_once(now=now) is True
    first = long_term._conn.execute("SELECT status, attempts, last_error FROM memory_work_items WHERE work_id = ?", ("work-retry",)).fetchone()
    assert tuple(first) == ("pending", 1, "provider temporarily unavailable")
    assert worker.poll_once(now=now) is False
    assert worker.poll_once(now=now.replace(year=now.year + 1)) is True
    final = long_term._conn.execute("SELECT status, attempts, last_error FROM memory_work_items WHERE work_id = ?", ("work-retry",)).fetchone()
    assert tuple(final) == ("degraded", 2, "provider temporarily unavailable")
