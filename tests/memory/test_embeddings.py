"""Contracts for the real OpenAI-compatible embedding provider."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from underwater_tracking.agent.llm import LLMConfigError, LLMContentError
from underwater_tracking.config.models import MemoryConfig
from underwater_tracking.domain.memory_models import MemoryType, MemoryVersion
from underwater_tracking.memory.embeddings import (
    HTTPEmbeddingProvider,
    parse_embedding_response,
)
from underwater_tracking.memory.retriever import rank_memories
from underwater_tracking.memory.retriever import MemoryRetriever
from underwater_tracking.persistence.memory import LongTermMemoryRepository


def _config(**changes: object) -> MemoryConfig:
    values: dict[str, object] = {
        "embedding_base_url": "https://api.example.test/v1",
        "embedding_model": "embedding-test-v1",
        "embedding_api_key_env": "UNDERWATER_TRACKING_EMBEDDING_TEST_KEY",
        "embedding_vector_version": "embedding-test-2026-08",
    }
    values.update(changes)
    return MemoryConfig(**values)


def test_missing_embedding_key_raises_typed_config_error_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    monkeypatch.delenv(config.embedding_api_key_env, raising=False)
    provider = HTTPEmbeddingProvider(config)
    try:
        with pytest.raises(LLMConfigError, match=config.embedding_api_key_env):
            provider.embed("do not synthesize an embedding")
    finally:
        provider.close()


def test_embedding_response_validates_vector_model_and_persisted_version(tmp_path) -> None:
    config = _config()
    result = parse_embedding_response(
        {
            "object": "list",
            "model": config.embedding_model,
            "data": [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5]}],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        },
        model=config.embedding_model,
        vector_version=config.embedding_vector_version,
    )
    memory = MemoryVersion(
        memory_id="memory-embedding-1",
        memory_family_id="family-embedding-1",
        version=1,
        user_id="operator",
        memory_type=MemoryType.SEMANTIC,
        summary="The reporting format is concise.",
        importance_score=0.8,
        embedding=result.vector,
        embedding_version=result.vector_version,
    )
    repository = LongTermMemoryRepository(tmp_path / "memory.db")
    try:
        repository.create_memory_version(memory, expected_previous_version=0)
        stored = repository.list_active("operator", limit=1)[0]
    finally:
        repository.close()

    assert result.model == config.embedding_model
    assert result.dimensions == 2
    assert result.token_count == 3
    assert stored.embedding == (0.25, -0.5)
    assert stored.embedding_version == config.embedding_vector_version


@pytest.mark.parametrize(
    "response",
    [
        {"model": "other-model", "data": [{"embedding": [0.1]}]},
        {"model": "embedding-test-v1", "data": [{"embedding": []}]},
        {"model": "embedding-test-v1", "data": [{"embedding": [float("nan")]}]},
        {"model": "embedding-test-v1", "data": []},
    ],
)
def test_embedding_response_rejects_invalid_provider_payload(response: object) -> None:
    config = _config()
    with pytest.raises(LLMContentError):
        parse_embedding_response(
            response,
            model=config.embedding_model,
            vector_version=config.embedding_vector_version,
        )


def test_ranking_keeps_user_scope_and_hard_context_budget() -> None:
    matching = MemoryVersion(
        memory_id="memory-matching",
        memory_family_id="family-matching",
        version=1,
        user_id="operator",
        memory_type=MemoryType.SEMANTIC,
        summary="concise evidence reports",
        importance_score=0.8,
        embedding=(1.0, 0.0),
        created_at=datetime(2026, 8, 20, tzinfo=UTC),
    )
    other_user = matching.model_copy(
        update={"memory_id": "memory-other-user", "user_id": "other-user"}
    )
    hits = rank_memories(
        memories=(matching, other_user),
        user_id="operator",
        query_vector=(1.0, 0.0),
        top_k=8,
        token_budget=8,
        decay_half_life_s=3600.0,
        now=datetime(2026, 8, 20, tzinfo=UTC),
    )

    assert [hit.memory.memory_id for hit in hits] == ["memory-matching"]
    assert hits[0].rerank_score >= hits[0].similarity_score * 0.5


def test_retriever_returns_degraded_empty_context_when_real_embedding_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config = _config()
    monkeypatch.delenv(config.embedding_api_key_env, raising=False)
    provider = HTTPEmbeddingProvider(config)
    repository = LongTermMemoryRepository(tmp_path / "memory.db")
    retriever = MemoryRetriever(
        embedding_provider=provider,
        repository=repository,
        config=config,
    )
    try:
        result = retriever.retrieve(user_id="operator", query="report preference")
    finally:
        provider.close()
        repository.close()

    assert result.memory_status == "degraded"
    assert result.long_term_material == ()
    assert result.retrieved_memory_ids == ()
