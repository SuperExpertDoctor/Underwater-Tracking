"""Credential-gated, real-provider checks for the memory LLM pipeline."""

from __future__ import annotations

import os

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.memory_models import (
    MemoryType,
    MemoryVersion,
    ShortTermContext,
    ShortTermMessage,
)
from underwater_tracking.memory.embeddings import SentenceTransformerEmbeddingProvider
from underwater_tracking.memory.reasoner import MemoryReasoner
from underwater_tracking.memory.retriever import MemoryRetriever
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.memory import LongTermMemoryRepository
from tests.conftest import (
    CONFIG_PATH,
    REAL_LLM_SKIP_REASON,
    make_live_llm,
    real_provider_enabled,
)


def _has_real_memory_credentials() -> bool:
    config = load_app_config(CONFIG_PATH)
    if not (
        os.environ.get("UNDERWATER_TRACKING_API_KEY")
        and config.memory is not None
        and config.memory.enabled
        and config.memory.embedding_provider == "sentence_transformers"
        and config.memory.embedding_model
    ):
        return False
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(
            config.memory.embedding_model,
            device=config.memory.embedding_device,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not real_provider_enabled() or not _has_real_memory_credentials(),
    reason=(
        "UNDERWATER_TRACKING_RUN_REAL_LLM=1, UNDERWATER_TRACKING_API_KEY, and locally cached sentence-transformers weights "
        "are required; "
        + REAL_LLM_SKIP_REASON
    ),
)


@pytest.mark.real_llm
def test_real_memory_embedding_reasoning_and_audit(tmp_path) -> None:
    config = load_app_config(CONFIG_PATH)
    assert config.memory is not None
    ledger = DecisionLedger(tmp_path / "memory.db")
    repository = LongTermMemoryRepository(tmp_path / "memory.db")
    embedding_provider = SentenceTransformerEmbeddingProvider(config.memory, ledger=ledger)
    llm = make_live_llm(ledger=ledger)
    reasoner = MemoryReasoner(llm=llm, repository=repository, config=config.memory)
    context = ShortTermContext(
        user_id="operator",
        conversation_id="conversation-real-1",
        recent_messages=(
            ShortTermMessage(
                message_id="message-real-1",
                role="user",
                text="Remember that I prefer concise evidence-backed reporting.",
            ),
            ShortTermMessage(
                message_id="message-real-2",
                role="assistant",
                text="I will retain that reporting preference.",
            ),
        ),
    )
    try:
        embedding = embedding_provider.embed("concise evidence-backed reporting")
        repository.create_memory_version(
            MemoryVersion(
                memory_id="memory-real-1",
                memory_family_id="family-real-1",
                version=1,
                user_id="operator",
                memory_type=MemoryType.SEMANTIC,
                summary="Remember that I prefer concise evidence-backed reporting.",
                importance_score=0.9,
                embedding=embedding.vector,
                embedding_version=embedding.vector_version,
            ),
            expected_previous_version=0,
        )
        retrieved = MemoryRetriever(
            embedding_provider=embedding_provider,
            repository=repository,
            config=config.memory,
        ).retrieve(user_id="operator", query="concise evidence-backed reporting")
        access_count = repository.list_active("operator", limit=1)[0].access_count
        decision = reasoner.filter(
            user_id="operator",
            source_texts=("Remember that I prefer concise evidence-backed reporting.",),
            source_message_ids=("message-real-1",),
            short_term_context=context,
        )
        extraction = reasoner.extract(
            user_id="operator",
            source_texts=("Remember that I prefer concise evidence-backed reporting.",),
            source_message_ids=("message-real-1",),
        )
        compression = reasoner.compress_short_term(context)
        operations = {call.operation for call in ledger.list_llm_calls(limit=8)}
    finally:
        llm.close()
        embedding_provider.close()
        repository.close()
        ledger.close()
    assert embedding.vector and embedding.model == config.memory.embedding_model
    assert retrieved.retrieved_memory_ids == ("memory-real-1",)
    assert access_count == 1
    assert decision.reason
    assert extraction.summary
    assert compression.summary_text
    assert {
        "memory_embedding",
        "memory_filter",
        "memory_extract",
        "short_term_compress",
    } <= operations
