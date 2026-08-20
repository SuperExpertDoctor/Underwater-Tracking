"""Durable, stoppable background worker for semantic memory processing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Event, Thread
from typing import Protocol, Sequence
from uuid import uuid4

from underwater_tracking.agent.llm import LLMConfigError, LLMError
from underwater_tracking.config.models import MemoryConfig
from underwater_tracking.domain.memory_models import (
    MemoryExtractionResult,
    MemoryFilterDecision,
    MemoryStatus,
    MemoryStreamEventType,
    MemoryStreamStatus,
    MemoryType,
    MemoryVersion,
    MemoryWorkItem,
    MemoryWorkStatus,
    MemoryWorkType,
    ShortTermCompressionResult,
    ShortTermContext,
)
from underwater_tracking.memory.service import MemoryService
from underwater_tracking.memory.embeddings import EmbeddingProvider
from underwater_tracking.memory.source_reader import MemorySource, MemorySourceReader
from underwater_tracking.persistence.memory import LongTermMemoryRepository, VersionConflictError


class MemoryReasonerPort(Protocol):
    def filter(self, **kwargs: object) -> MemoryFilterDecision: ...

    def extract(self, **kwargs: object) -> MemoryExtractionResult: ...

    def compress_short_term(self, context: ShortTermContext) -> ShortTermCompressionResult: ...


@dataclass(frozen=True)
class MemoryWorkerMetrics:
    queue_backlog: int
    oldest_item_age_s: float | None
    last_success_at: datetime | None
    degraded_reason: str | None


class MemoryWorker:
    """Owns one daemon thread and never calls runtime or simulation locks."""

    def __init__(
        self,
        repository: LongTermMemoryRepository,
        service: MemoryService,
        reasoner: MemoryReasonerPort,
        source_reader: MemorySourceReader | None,
        config: MemoryConfig,
        worker_id: str,
        *,
        stop_event: Event | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._repository = repository
        self._service = service
        self._reasoner = reasoner
        self._source_reader = source_reader
        self._config = config
        self._worker_id = worker_id
        self._stop_event = stop_event or Event()
        self._embedding_provider = embedding_provider
        self._thread: Thread | None = None
        self._last_source_poll = datetime.min.replace(tzinfo=UTC)
        self._last_success_at: datetime | None = None
        self._degraded_reason: str | None = None
        self._source_scopes: set[tuple[str, str]] = set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def metrics(self) -> MemoryWorkerMetrics:
        row = self._repository._conn.execute(
            "SELECT COUNT(*), MIN(created_at) FROM memory_work_items WHERE status IN ('pending', 'processing')"
        ).fetchone()
        oldest = None
        if row is not None and row[1] is not None:
            oldest = max(0.0, datetime.now(UTC).timestamp() - int(row[1]) / 1000)
        return MemoryWorkerMetrics(int(row[0]) if row is not None else 0, oldest, self._last_success_at, self._degraded_reason)

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = Thread(target=self.run_forever, name="underwater-memory-worker", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> bool:
        self._stop_event.set()
        if self._thread is None:
            return True
        self._thread.join(timeout=max(0.0, timeout))
        return not self._thread.is_alive()

    def run_forever(self) -> None:
        while not self._stop_event.is_set():
            handled = self.poll_once()
            if not handled:
                self._stop_event.wait(self._config.poll_interval_s)

    def poll_once(self, *, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        work = self._repository.claim_work(
            self._worker_id,
            now,
            self._config.work_lease_timeout_s,
            max_attempts=self._config.max_attempts,
        )
        if work is None:
            if self._source_reader is not None and now - self._last_source_poll >= timedelta(
                seconds=self._config.maintenance_interval_s
            ):
                read, succeeded = self._read_sources()
                if succeeded:
                    self._last_source_poll = now
                return read
            return False
        if work.scenario_id is not None:
            self._source_scopes.add((work.user_id, work.scenario_id))
        self._service.emit_worker_event(
            user_id=work.user_id,
            conversation_id=work.conversation_id,
            status=MemoryStreamStatus.PROCESSING,
            event_type=MemoryStreamEventType.WORK_PROCESSING,
            work_id=work.work_id,
            source_ids=_work_source_ids(work),
        )
        try:
            self._process(work, now)
        except LLMError as error:
            self._retry_or_degrade(work, error, now)
        except (VersionConflictError, RuntimeError, ValueError) as error:
            self._retry_or_degrade(work, error, now)
        else:
            self._repository.complete_work(work.work_id, self._worker_id)
            self._last_success_at = now
            self._degraded_reason = None
            self._service.emit_worker_event(
                user_id=work.user_id,
                conversation_id=work.conversation_id,
                status=MemoryStreamStatus.COMPLETED,
                event_type=MemoryStreamEventType.WORK_COMPLETED,
                work_id=work.work_id,
                source_ids=_work_source_ids(work),
            )
        return True

    def _process(self, work: MemoryWorkItem, now: datetime) -> None:
        if work.work_type is MemoryWorkType.MAINTENANCE:
            self._maintenance(work, now)
            return
        sources, short_term = self._sources_for_work(work)
        source_texts = tuple(source.text for source in sources)
        if not source_texts and short_term is not None:
            source_texts = tuple(message.text for message in short_term.recent_messages)
        decision = self._reasoner.filter(
            user_id=work.user_id,
            source_texts=source_texts,
            source_message_ids=work.payload.source_message_ids,
            source_event_ids=work.payload.source_event_ids,
            source_decision_ids=work.payload.source_decision_ids,
            source_knowledge_ids=work.payload.source_knowledge_ids,
            short_term_context=short_term,
        )
        self._service.emit_worker_event(
            user_id=work.user_id,
            conversation_id=work.conversation_id,
            status=MemoryStreamStatus.PROCESSING,
            event_type=MemoryStreamEventType.MEMORY_FILTERED,
            work_id=work.work_id,
            source_ids=_work_source_ids(work),
        )
        if decision.should_store:
            extraction = self._reasoner.extract(
                user_id=work.user_id,
                source_texts=source_texts,
                source_message_ids=work.payload.source_message_ids,
                source_event_ids=work.payload.source_event_ids,
                source_decision_ids=work.payload.source_decision_ids,
                source_knowledge_ids=work.payload.source_knowledge_ids,
            )
            self._create_version(work, decision, extraction)
        if short_term is not None and self._should_compress(short_term, now):
            self._compress(work, short_term)

    def _sources_for_work(
        self, work: MemoryWorkItem
    ) -> tuple[tuple[MemorySource, ...], ShortTermContext | None]:
        sources: tuple[MemorySource, ...] = ()
        if self._source_reader is not None:
            sources = self._source_reader.load_work_sources(work.user_id, work.scenario_id, work.payload)
        short_term = (
            self._service.snapshot(work.user_id, work.conversation_id)
            if work.conversation_id is not None
            else None
        )
        return sources, short_term

    def _create_version(
        self, work: MemoryWorkItem, decision: MemoryFilterDecision, extraction: MemoryExtractionResult
    ) -> None:
        current: MemoryVersion | None = None
        if decision.operation == "update" and decision.candidate_memory_id is not None:
            current = next(
                (
                    memory
                    for memory in self._repository.list_active(work.user_id, limit=100)
                    if memory.memory_id == decision.candidate_memory_id
                ),
                None,
            )
        family = decision.family_key or (current.memory_family_id if current is not None else f"family:{work.work_id}")
        previous = current.version if current is not None else 0
        memory_id = f"memory:{uuid4().hex}"
        if self._embedding_provider is None:
            raise LLMConfigError("memory worker requires a configured real embedding provider")
        embedding = self._embedding_provider.embed(extraction.summary)
        memory = MemoryVersion(
            memory_id=memory_id,
            memory_family_id=family,
            version=previous + 1,
            user_id=work.user_id,
            memory_type=decision.memory_type or MemoryType.SEMANTIC,
            summary=extraction.summary,
            importance_score=decision.importance_score,
            embedding=embedding.vector,
            embedding_version=embedding.vector_version,
            status=MemoryStatus.ACTIVE,
            supersedes_memory_id=current.memory_id if current is not None else None,
            source_message_ids=extraction.source_message_ids,
            source_event_ids=extraction.source_event_ids,
            source_decision_ids=extraction.source_decision_ids,
            source_knowledge_ids=extraction.source_knowledge_ids,
            change_reason=extraction.change_reason,
        )
        persisted = self._repository.create_memory_version(memory, previous, work_id=work.work_id)
        self._service.emit_worker_event(
            user_id=work.user_id,
            conversation_id=work.conversation_id,
            status=MemoryStreamStatus.PROCESSING,
            event_type=MemoryStreamEventType.MEMORY_VERSION_CREATED,
            work_id=work.work_id,
            source_ids=_work_source_ids(work),
            memory_id=persisted.memory_id,
            memory_family_id=persisted.memory_family_id,
            version=persisted.version,
        )

    def _compress(self, work: MemoryWorkItem, context: ShortTermContext) -> None:
        result = self._reasoner.compress_short_term(context)
        expected = getattr(context, "summary_version")
        compressed = self._service._short_term.save_compressed_context(
            getattr(context, "user_id"),
            getattr(context, "conversation_id"),
            expected,
            result.summary_text,
            result.retained_messages[-self._config.recent_message_limit :],
            operation_id=work.work_id,
        )
        self._service.emit_worker_event(
            user_id=work.user_id,
            conversation_id=work.conversation_id,
            status=MemoryStreamStatus.PROCESSING,
            event_type=MemoryStreamEventType.SHORT_TERM_COMPRESSED,
            work_id=work.work_id,
            source_ids=result.source_message_ids,
            version=compressed.summary_version,
        )

    def _should_compress(self, context: ShortTermContext, now: datetime) -> bool:
        message_count = context.message_count
        tokens = context.estimated_tokens
        last = context.last_compressed_at
        return (
            message_count >= self._config.short_term_message_threshold
            or tokens >= self._config.short_term_token_threshold
            or (last is not None and now - last >= timedelta(seconds=self._config.short_term_compress_interval_s))
        )

    def _maintenance(self, work: MemoryWorkItem, now: datetime) -> None:
        updates = self._repository.maintain_active(
            work.user_id,
            now,
            decay_half_life_s=self._config.decay_half_life_s,
            archive_threshold=self._config.archive_threshold,
            limit=32,
        )
        for memory_id, status, score in updates:
            if status is MemoryStatus.ARCHIVED:
                self._service.emit_worker_event(
                    user_id=work.user_id,
                    conversation_id=work.conversation_id,
                    status=MemoryStreamStatus.COMPLETED,
                    event_type=MemoryStreamEventType.MEMORY_ARCHIVED,
                    work_id=work.work_id,
                    memory_id=memory_id,
                    version=None,
                )

    def _read_sources(self) -> tuple[bool, bool]:
        assert self._source_reader is not None
        queued = False
        succeeded = True
        for user_id, scenario_id in tuple(self._source_scopes):
            try:
                for source in self._source_reader.read_new(user_id, scenario_id):
                    source_id = _source_id(source)
                    outcome = self._service.enqueue_observation(
                        {
                            "source_id": source_id,
                            "source_type": source.source_type,
                            "scenario_id": scenario_id,
                            "user_id": user_id,
                            "source_key": source.source_key,
                            "source_cursor": source.cursor,
                            "source_cursor_type": source.source_cursor_type or source.source_type,
                        },
                        source.payload,
                    )
                    queued = queued or outcome["status"] == "queued"
            except (OSError, RuntimeError, ValueError) as error:
                succeeded = False
                self._degraded_reason = type(error).__name__
                self._service.emit_worker_event(
                    user_id=user_id,
                    conversation_id=None,
                    status=MemoryStreamStatus.DEGRADED,
                    event_type=MemoryStreamEventType.SOURCE_READ_DEGRADED,
                    work_id=f"source-read:{scenario_id}",
                )
        return queued, succeeded

    def _retry_or_degrade(self, work: MemoryWorkItem, error: Exception, now: datetime) -> None:
        retry_at = now + timedelta(seconds=self._config.retry_backoff_s * (2 ** max(0, work.attempts - 1)))
        self._repository.fail_work(
            work.work_id,
            self._worker_id,
            MemoryWorkStatus.PENDING,
            str(error)[:1000] or type(error).__name__,
            retry_at,
            max_attempts=self._config.max_attempts,
        )
        self._degraded_reason = type(error).__name__
        self._service.emit_worker_event(
            user_id=work.user_id,
            conversation_id=work.conversation_id,
            status=MemoryStreamStatus.DEGRADED,
            event_type=MemoryStreamEventType.WORK_DEGRADED,
            work_id=work.work_id,
            source_ids=_work_source_ids(work),
        )


def _work_source_ids(work: MemoryWorkItem) -> Sequence[str]:
    return (
        work.payload.source_message_ids
        + work.payload.source_event_ids
        + work.payload.source_decision_ids
        + work.payload.source_knowledge_ids
    )


def _source_id(source: MemorySource) -> str:
    for source_ids in (
        source.source_event_ids,
        source.source_decision_ids,
        source.source_knowledge_ids,
        source.source_message_ids,
    ):
        if source_ids:
            return source_ids[0]
    raise ValueError("memory source must carry one stable source ID")
