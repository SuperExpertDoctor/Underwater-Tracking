from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from underwater_tracking.api.app import create_app
from underwater_tracking.api.dependencies import MemoryServiceAdapter
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.config.models import MemoryConfig
from underwater_tracking.domain.memory_models import (
    MemoryEvidenceTrace,
    MemoryExtractionResult,
    MemoryFilterDecision,
    MemoryStreamStatus,
    MemoryType,
    ShortTermCompressionResult,
)
from underwater_tracking.memory.embeddings import EmbeddingResult
from underwater_tracking.memory.retriever import MemoryRetriever
from underwater_tracking.memory.service import MemoryService
from underwater_tracking.memory.worker import MemoryWorker
from underwater_tracking.persistence.memory import LongTermMemoryRepository, ShortTermContextRepository


class _Runtime:
    def active_plan(self) -> None:
        return None


class _Replay:
    def range(self, *args: object, **kwargs: object) -> list[object]:
        del args, kwargs
        return []


class _Embedder:
    def embed(self, text: str) -> EmbeddingResult:
        assert text.strip()
        return EmbeddingResult(
            vector=(1.0, 0.0),
            model="test-embedding",
            vector_version="test-v1",
        )


class _LifecycleReasoner:
    def __init__(self) -> None:
        self.candidate_memory_id: str | None = None
        self.filter_calls = 0

    def filter(self, **kwargs: object) -> MemoryFilterDecision:
        self.filter_calls += 1
        if self.candidate_memory_id is None:
            return MemoryFilterDecision(
                should_store=True,
                explicit_remember=True,
                memory_type=MemoryType.SEMANTIC,
                operation="create",
                family_key="tracking-doctrine-preference",
                importance_score=0.9,
                reason="explicit durable operator preference",
            )
        return MemoryFilterDecision(
            should_store=True,
            explicit_remember=True,
            memory_type=MemoryType.SEMANTIC,
            operation="update",
            family_key="tracking-doctrine-preference",
            candidate_memory_id=self.candidate_memory_id,
            importance_score=0.9,
            reason="explicit revised durable operator preference",
        )

    def extract(self, **kwargs: object) -> MemoryExtractionResult:
        source_texts = tuple(kwargs["source_texts"])
        return MemoryExtractionResult(
            summary=source_texts[-1],
            source_message_ids=tuple(kwargs["source_message_ids"]),
            change_reason="explicit operator preference",
        )

    def compress_short_term(self, context):
        retained = context.recent_messages[-1:]
        return ShortTermCompressionResult(
            summary_text=retained[-1].text if retained else context.summary_text,
            retained_messages=retained,
            source_message_ids=tuple(message.message_id for message in retained),
        )


def _config() -> MemoryConfig:
    return MemoryConfig(
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding-test-v1",
        embedding_vector_version="test-v1",
        recent_message_limit=1,
        short_term_message_threshold=1,
        short_term_token_threshold=1,
        context_token_budget=200,
        source_poll_interval_s=2.0,
        maintenance_interval_s=300.0,
    )


def test_memory_full_lifecycle_keeps_versions_evidence_and_sources(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    config = _config()
    service = MemoryService(
        short_term,
        long_term,
        MemoryRetriever(embedding_provider=_Embedder(), repository=long_term, config=config),
    )
    reasoner = _LifecycleReasoner()
    worker = MemoryWorker(
        long_term,
        service,
        reasoner,
        None,
        config,
        "lifecycle-worker",
        embedding_provider=_Embedder(),
    )

    first = service.accept_turn(
        {
            "user_id": "operator",
            "conversation_id": "conversation-1",
            "scenario_id": "scenario-1",
            "message_id": "preference-1",
            "text": "目标接触后优先维持被动协同跟踪，只有丢失接触时才启用主动扫描。",
        },
        None,
    )
    assert first["status"] == "queued"
    assert worker.poll_once() is True
    first_memory = long_term.list_active("operator", filters={"scenario_id": "scenario-1"})[0]
    assert first_memory.version == 1
    assert first_memory.memory_family_id == "tracking-doctrine-preference"

    first_cursor = max(
        event.cursor
        for event in long_term.list_stream_events(
            "operator", "conversation-1", scenario_id="scenario-1", limit=128
        )
        if event.type.value == "memory_version_created"
    )
    assert short_term._conn.execute(
        "SELECT COUNT(*) FROM short_term_messages WHERE user_id = ? AND scenario_id = ?",
        ("operator", "scenario-1"),
    ).fetchone()[0] == 1
    compressed = short_term.get_short_term("operator", "conversation-1", "scenario-1")
    assert compressed is not None
    assert compressed.summary_version == 1
    assert len(compressed.recent_messages) <= 1

    reasoner.candidate_memory_id = first_memory.memory_id
    second = service.accept_turn(
        {
            "user_id": "operator",
            "conversation_id": "conversation-1",
            "scenario_id": "scenario-1",
            "message_id": "preference-2",
            "text": "修订偏好：保持被动协同跟踪，只有连续丢失接触后才启用主动扫描。",
        },
        None,
    )
    assert second["status"] == "queued"
    assert worker.poll_once() is True
    active = long_term.list_active("operator", filters={"scenario_id": "scenario-1"})
    assert len(active) == 1
    second_memory = active[0]
    assert second_memory.version == 2
    assert second_memory.supersedes_memory_id == first_memory.memory_id
    versions = long_term.list_versions(
        "operator", "tracking-doctrine-preference", "scenario-1"
    )
    assert [memory.version for memory in versions] == [1, 2]
    assert short_term.get_short_term("operator", "conversation-1", "scenario-1").summary_version == 2

    context = service.prepare_context(
        "operator",
        "conversation-1",
        "被动协同跟踪",
        scenario_id="scenario-1",
    )
    assert context.retrieved_memory_ids == (second_memory.memory_id,)
    assert first_memory.memory_id not in context.retrieved_memory_ids
    trace = MemoryEvidenceTrace(
        trace_id="trace-doctrine-v2",
        user_id="operator",
        status=MemoryStreamStatus.COMPLETED,
        memory_ids=(second_memory.memory_id,),
        source_message_ids=second_memory.source_message_ids,
        source_plan_ids=("plan-version-2",),
    )
    evidence_events = service.emit_evidence_trace_events(
        user_id="operator",
        conversation_id="conversation-1",
        scenario_id="scenario-1",
        trace=trace,
        plan_version=2,
    )
    assert [event.type.value for event in evidence_events] == [
        "evidence_trace_started",
        "evidence_trace_completed",
    ]
    assert evidence_events[-1].payload.source_message_ids == second_memory.source_message_ids
    assert evidence_events[-1].payload.plan_version == 2

    all_events = long_term.list_stream_events(
        "operator", "conversation-1", scenario_id="scenario-1", limit=128
    )
    incremental = [event for event in all_events if event.cursor > first_cursor]
    assert sum(event.type.value == "memory_version_created" for event in all_events) == 2
    assert sum(event.type.value == "memory_version_created" for event in incremental) == 1
    assert sum(event.type.value == "memory_version_superseded" for event in incremental) == 1

    adapter = MemoryServiceAdapter(service)
    app = create_app(runtime=_Runtime(), replay=_Replay(), memory_port=adapter, hub=OperationalHub())
    with TestClient(app) as client:
        stream = client.get(
            "/api/assistant/memory/stream",
            params={
                "user_id": "operator",
                "conversation_id": "conversation-1",
                "scenario_id": "scenario-1",
                "include_scenario_events": True,
            },
        )
        assert stream.status_code == 200
        assert any(item["type"] == "evidence_trace_completed" for item in stream.json()["events"])
        deleted = client.request(
            "DELETE",
            "/api/assistant/memory/" + second_memory.memory_id,
            json={
                "user_id": "operator",
                "scenario_id": "scenario-1",
                "conversation_id": "conversation-1",
            },
        )
        assert deleted.status_code == 200
        snapshot = client.get(
            "/api/assistant/memory",
            params={
                "user_id": "operator",
                "conversation_id": "conversation-1",
                "scenario_id": "scenario-1",
            },
        )
        assert snapshot.status_code == 200
        assert snapshot.json()["semantic"] == []

    assert short_term.get_messages(
        "operator", "conversation-1", ("preference-1", "preference-2"), scenario_id="scenario-1"
    )
    assert any(
        event.type.value == "memory_deleted"
        and event.payload.source_message_ids == second_memory.source_message_ids
        for event in long_term.list_stream_events(
            "operator", "conversation-1", scenario_id="scenario-1", limit=256
        )
    )
    short_term.close()
    long_term.close()
