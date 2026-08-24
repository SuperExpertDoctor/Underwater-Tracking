"""Contracts for the real local and OpenAI-compatible embedding providers."""

from __future__ import annotations

from datetime import UTC, datetime
import sys
from types import SimpleNamespace

import pytest

from underwater_tracking.agent.llm import LLMConfigError, LLMContentError
from underwater_tracking.config.models import MemoryConfig
from underwater_tracking.domain.memory_models import MemoryType, MemoryVersion
from underwater_tracking.memory.embeddings import (
    EmbeddingResult,
    HTTPEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    parse_embedding_response,
)
from underwater_tracking.memory.retriever import rank_memories
from underwater_tracking.memory.retriever import MemoryRetriever
from underwater_tracking.persistence.memory import LongTermMemoryRepository


def _config(**changes: object) -> MemoryConfig:
    values: dict[str, object] = {
        "embedding_provider": "http",
        "embedding_base_url": "https://api.example.test/v1",
        "embedding_model": "embedding-test-v1",
        "embedding_api_key_env": "UNDERWATER_TRACKING_EMBEDDING_TEST_KEY",
        "embedding_vector_version": "embedding-test-2026-08",
    }
    values.update(changes)
    return MemoryConfig(**values)


def _local_config(**changes: object) -> MemoryConfig:
    values: dict[str, object] = {
        "embedding_provider": "sentence_transformers",
        "embedding_model": "local-test-model",
        "embedding_vector_version": "st-local-test-2026-08",
        "embedding_local_files_only": True,
        "embedding_device": "cpu",
        "embedding_normalize": True,
    }
    values.update(changes)
    return MemoryConfig(**values)


def test_sentence_transformer_provider_uses_local_model_and_real_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            calls["model_name"] = model_name
            calls["constructor"] = kwargs

        def encode(self, text: str, **kwargs: object) -> list[float]:
            calls["text"] = text
            calls["encode"] = kwargs
            return [0.25, -0.5, 0.75]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    provider = SentenceTransformerEmbeddingProvider(_local_config())

    result = provider.embed("concise evidence")

    assert result.vector == (0.25, -0.5, 0.75)
    assert result.model == "local-test-model"
    assert result.vector_version == "st-local-test-2026-08"
    assert calls["model_name"] == "local-test-model"
    constructor = calls["constructor"]
    assert isinstance(constructor, dict)
    assert constructor["local_files_only"] is True
    assert constructor["device"] == "cpu"
    encode = calls["encode"]
    assert isinstance(encode, dict)
    assert encode["normalize_embeddings"] is True
    assert encode["show_progress_bar"] is False


def test_sentence_transformer_provider_verifies_local_model_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            del model_name, kwargs

        def encode(self, text: str, **kwargs: object) -> list[float]:
            del kwargs
            calls.append(text)
            return [0.25, -0.5, 0.75]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    provider = SentenceTransformerEmbeddingProvider(_local_config())

    provider.verify_ready()

    assert calls == ["memory readiness probe"]


def test_sentence_transformer_provider_rejects_non_finite_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            del model_name, kwargs

        def encode(self, text: str, **kwargs: object) -> list[float]:
            del text, kwargs
            return [0.1, float("nan")]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    provider = SentenceTransformerEmbeddingProvider(_local_config())

    with pytest.raises(LLMContentError, match="non-finite"):
        provider.embed("invalid vector")


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


def test_retriever_never_returns_a_memory_from_another_scenario(tmp_path) -> None:
    class FixedEmbedder:
        def embed(self, text: str) -> EmbeddingResult:
            assert text
            return EmbeddingResult(
                vector=(1.0, 0.0), model="embedding-test-v1", vector_version="embedding-test-2026-08"
            )

    repository = LongTermMemoryRepository(tmp_path / "memory.db")
    for scenario_id, memory_id in (("scenario-a", "memory-a"), ("scenario-b", "memory-b")):
        repository.create_memory_version(
            MemoryVersion(
                memory_id=memory_id,
                memory_family_id="shared-family",
                version=1,
                user_id="operator",
                scenario_id=scenario_id,
                memory_type=MemoryType.SEMANTIC,
                summary=f"memory {scenario_id}",
                importance_score=0.9,
                embedding=(1.0, 0.0),
                embedding_version="embedding-test-2026-08",
            ),
            expected_previous_version=0,
        )
    retriever = MemoryRetriever(
        embedding_provider=FixedEmbedder(), repository=repository, config=_config()
    )

    result = retriever.retrieve(
        user_id="operator", query="memory", scenario_id="scenario-a"
    )

    assert result.scenario_id == "scenario-a"
    assert result.retrieved_memory_ids == ("memory-a",)
