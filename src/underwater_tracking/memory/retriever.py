"""Bounded, user-scoped long-term memory retrieval over real embeddings."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from underwater_tracking.agent.llm import LLMError
from underwater_tracking.config.models import MemoryConfig
from underwater_tracking.domain.memory_models import (
    MemoryContext,
    MemoryRetrievalHit,
    MemoryStreamStatus,
    MemoryVersion,
)
from underwater_tracking.memory.embeddings import EmbeddingProvider
from underwater_tracking.persistence.memory import LongTermMemoryRepository


class MemoryRetriever:
    """Retrieves only active, version-compatible memory for one user."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        repository: LongTermMemoryRepository,
        config: MemoryConfig,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._repository = repository
        self._config = config

    def retrieve(
        self,
        *,
        user_id: str,
        query: str,
        filters: Mapping[str, object] | None = None,
        now: datetime | None = None,
    ) -> MemoryContext:
        """Return bounded long-term material or an explicit degraded empty context."""
        try:
            embedding = self._embedding_provider.embed(query)
        except LLMError:
            return MemoryContext(user_id=user_id, memory_status=MemoryStreamStatus.DEGRADED)
        selected_filters = dict(filters or {})
        configured_minimum = self._config.min_importance_score
        requested_minimum = selected_filters.get("min_importance_score", configured_minimum)
        if isinstance(requested_minimum, (int, float)) and not isinstance(requested_minimum, bool):
            selected_filters["min_importance_score"] = max(configured_minimum, float(requested_minimum))
        else:
            selected_filters["min_importance_score"] = configured_minimum
        candidates = self._repository.list_active(
            user_id,
            filters=selected_filters,
            limit=self._config.retrieval_candidate_limit,
        )
        compatible = tuple(
            memory
            for memory in candidates
            if memory.embedding_version == embedding.vector_version
        )
        hits = rank_memories(
            memories=compatible,
            user_id=user_id,
            query_vector=embedding.vector,
            top_k=self._config.retrieval_top_k,
            token_budget=self._config.context_token_budget,
            decay_half_life_s=self._config.decay_half_life_s,
            now=now or datetime.now(UTC),
        )
        if hits:
            self._repository.record_access(user_id, tuple(hit.memory.memory_id for hit in hits))
        return MemoryContext(
            user_id=user_id,
            long_term_material=hits,
            retrieved_memory_ids=tuple(hit.memory.memory_id for hit in hits),
            memory_status=MemoryStreamStatus.COMPLETED,
        )


class DegradedMemoryRetriever:
    """Explicit no-provider port used when memory wiring is unavailable."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def retrieve(
        self,
        *,
        user_id: str,
        query: str,
        filters: Mapping[str, object] | None = None,
        now: datetime | None = None,
    ) -> MemoryContext:
        del query, filters, now
        return MemoryContext(user_id=user_id, memory_status=MemoryStreamStatus.DEGRADED)


def rank_memories(
    *,
    memories: Sequence[MemoryVersion],
    user_id: str,
    query_vector: Sequence[float],
    top_k: int,
    token_budget: int,
    decay_half_life_s: float,
    now: datetime,
) -> tuple[MemoryRetrievalHit, ...]:
    """Pure, deterministic score and budget enforcement after repository filtering."""
    if top_k <= 0 or token_budget <= 0 or decay_half_life_s <= 0:
        return ()
    scored: list[MemoryRetrievalHit] = []
    for memory in memories:
        if memory.user_id != user_id or not memory.embedding:
            continue
        similarity = _similarity(query_vector, memory.embedding)
        if similarity is None:
            continue
        recency = _recency_score(memory.created_at, now, decay_half_life_s)
        frequency = min(math.log1p(memory.access_count) / math.log(11.0), 1.0)
        rerank = 0.65 * similarity + 0.2 * memory.importance_score + 0.1 * recency + 0.05 * frequency
        scored.append(
            MemoryRetrievalHit(
                memory=memory,
                similarity_score=similarity,
                rerank_score=min(max(rerank, 0.0), 1.0),
                retrieval_reason="semantic similarity, importance, recency, and access frequency",
            )
        )
    scored.sort(
        key=lambda hit: (hit.rerank_score, hit.similarity_score, hit.memory.created_at, hit.memory.memory_id),
        reverse=True,
    )
    selected: list[MemoryRetrievalHit] = []
    used_tokens = 0
    for hit in scored:
        if len(selected) >= top_k:
            break
        estimated_tokens = (len(hit.memory.summary) + 3) // 4
        if used_tokens + estimated_tokens > token_budget:
            continue
        selected.append(hit)
        used_tokens += estimated_tokens
    return tuple(selected)


def _similarity(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    cosine = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return (cosine + 1.0) / 2.0


def _recency_score(created_at: datetime, now: datetime, half_life_s: float) -> float:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    age_s = max(0.0, (now - created_at).total_seconds())
    return float(0.5 ** (age_s / half_life_s))
