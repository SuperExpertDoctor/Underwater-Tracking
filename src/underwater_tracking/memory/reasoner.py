"""Structured, bounded memory operations with server-side result validation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from underwater_tracking.agent.llm import LLMContentError, StructuredLLM
from underwater_tracking.config.models import MemoryConfig
from underwater_tracking.domain.memory_models import (
    MemoryExtractionResult,
    MemoryFilterDecision,
    MemoryVersion,
    ShortTermCompressionResult,
    ShortTermContext,
    ShortTermMessage,
)
from underwater_tracking.persistence.memory import LongTermMemoryRepository

MEMORY_FILTER_PROMPT_VERSION = "memory-filter-v1"
MEMORY_EXTRACT_PROMPT_VERSION = "memory-extract-v1"
SHORT_TERM_COMPRESS_PROMPT_VERSION = "short-term-compress-v1"


class MemoryReasonerValidationError(LLMContentError):
    """A schema-valid LLM result violated a memory ownership constraint."""


class MemoryReasoner:
    """LLM-only filtering, extraction, and compression for memory workers."""

    def __init__(
        self,
        *,
        llm: StructuredLLM[Any],
        repository: LongTermMemoryRepository,
        config: MemoryConfig,
    ) -> None:
        self._llm = llm
        self._repository = repository
        self._config = config

    def filter(
        self,
        *,
        user_id: str,
        source_texts: Sequence[str],
        source_message_ids: Sequence[str] = (),
        source_event_ids: Sequence[str] = (),
        source_decision_ids: Sequence[str] = (),
        source_knowledge_ids: Sequence[str] = (),
        short_term_context: ShortTermContext | None = None,
    ) -> MemoryFilterDecision:
        """Ask the configured LLM whether bounded source material is durable."""
        candidates = self._repository.list_active(
            user_id, limit=self._config.retrieval_candidate_limit
        )
        payload: dict[str, object] = {
            "instruction": (
                "Decide whether the supplied source material should become durable memory. "
                "Use only the listed candidate IDs. If no candidate applies, return create or "
                "ignore; do not invent IDs. Do not use keyword rules."
            ),
            "user_id": user_id,
            "source": build_bounded_source_payload(
                source_texts,
                source_message_ids,
                source_event_ids,
                source_decision_ids,
                source_knowledge_ids,
                self._config,
            ),
            "short_term": _short_term_payload(short_term_context, self._config),
            "candidates": [_candidate_payload(candidate) for candidate in candidates],
        }
        decision = self._llm.invoke_structured(
            "memory_filter",
            payload,
            MemoryFilterDecision,
            prompt_version=MEMORY_FILTER_PROMPT_VERSION,
        )
        return validate_filter_decision(decision, candidates=candidates, user_id=user_id)

    def extract(
        self,
        *,
        user_id: str,
        source_texts: Sequence[str],
        source_message_ids: Sequence[str] = (),
        source_event_ids: Sequence[str] = (),
        source_decision_ids: Sequence[str] = (),
        source_knowledge_ids: Sequence[str] = (),
    ) -> MemoryExtractionResult:
        """Extract a lexically grounded summary from the supplied sources only."""
        payload: dict[str, object] = {
            "instruction": (
                "Create one concise durable-memory summary using only facts in source. "
                "Copy the summary verbatim from source text so it can be server-verified. "
                "Return only source IDs supplied in source."
            ),
            "user_id": user_id,
            "source": build_bounded_source_payload(
                source_texts,
                source_message_ids,
                source_event_ids,
                source_decision_ids,
                source_knowledge_ids,
                self._config,
            ),
        }
        result = self._llm.invoke_structured(
            "memory_extract",
            payload,
            MemoryExtractionResult,
            prompt_version=MEMORY_EXTRACT_PROMPT_VERSION,
        )
        return validate_extraction_result(
            result,
            source_texts=source_texts,
            source_message_ids=source_message_ids,
            source_event_ids=source_event_ids,
            source_decision_ids=source_decision_ids,
            source_knowledge_ids=source_knowledge_ids,
            max_summary_chars=min(4000, self._config.context_token_budget * 4),
        )

    def compress_short_term(self, context: ShortTermContext) -> ShortTermCompressionResult:
        """Compress only the configured bounded short-term window through the LLM."""
        messages = context.recent_messages[-self._config.recent_message_limit :]
        payload: dict[str, object] = {
            "instruction": (
                "Compress the bounded conversation without adding facts. Copy summary text "
                "verbatim from the existing summary or messages. Retain only complete supplied "
                "messages and cite only supplied message IDs."
            ),
            "user_id": context.user_id,
            "conversation_id": context.conversation_id,
            "existing_summary": context.summary_text,
            "messages": [_message_payload(message) for message in messages],
        }
        result = self._llm.invoke_structured(
            "short_term_compress",
            payload,
            ShortTermCompressionResult,
            prompt_version=SHORT_TERM_COMPRESS_PROMPT_VERSION,
        )
        bounded_context = context.model_copy(update={"recent_messages": messages})
        return validate_compression_result(result, context=bounded_context, config=self._config)


def validate_filter_decision(
    decision: MemoryFilterDecision,
    *,
    candidates: Sequence[MemoryVersion],
    user_id: str,
) -> MemoryFilterDecision:
    """Reject update targets that the prompt did not supply for this user."""
    if any(candidate.user_id != user_id for candidate in candidates):
        raise MemoryReasonerValidationError("candidate user_id does not match memory user_id")
    candidate_ids = {candidate.memory_id for candidate in candidates}
    if decision.should_store:
        if decision.operation == "ignore" or decision.memory_type is None:
            raise MemoryReasonerValidationError("stored memory requires an operation and memory_type")
    elif decision.operation != "ignore":
        raise MemoryReasonerValidationError("should_store false requires ignore operation")
    if decision.operation == "update":
        if decision.candidate_memory_id not in candidate_ids:
            raise MemoryReasonerValidationError(
                "candidate_memory_id must reference a supplied user-scoped candidate"
            )
    elif decision.candidate_memory_id is not None:
        raise MemoryReasonerValidationError("candidate_memory_id is only valid for update")
    return decision


def validate_extraction_result(
    result: MemoryExtractionResult,
    *,
    source_texts: Sequence[str],
    source_message_ids: Sequence[str] = (),
    source_event_ids: Sequence[str] = (),
    source_decision_ids: Sequence[str] = (),
    source_knowledge_ids: Sequence[str] = (),
    max_summary_chars: int = 4000,
) -> MemoryExtractionResult:
    """Keep source references and summary facts within the supplied source scope."""
    _validate_sources(result, source_message_ids, source_event_ids, source_decision_ids, source_knowledge_ids)
    if len(result.summary) > max_summary_chars:
        raise MemoryReasonerValidationError("summary exceeds the configured context budget")
    if not _is_grounded(result.summary, source_texts):
        raise MemoryReasonerValidationError("summary introduces facts outside the supplied source text")
    return result


def validate_compression_result(
    result: ShortTermCompressionResult,
    *,
    context: ShortTermContext,
    config: MemoryConfig,
) -> ShortTermCompressionResult:
    """Ensure a compression keeps only supplied messages and grounded text."""
    available = {message.message_id: message for message in context.recent_messages}
    for message in result.retained_messages:
        if available.get(message.message_id) != message:
            raise MemoryReasonerValidationError(
                "retained_messages must exactly match supplied short-term messages"
            )
    if not set(result.source_message_ids).issubset(available):
        raise MemoryReasonerValidationError("source_message_ids must belong to supplied short-term messages")
    source_texts = (context.summary_text,) + tuple(message.text for message in context.recent_messages)
    if not _is_grounded(result.summary_text, source_texts):
        raise MemoryReasonerValidationError("summary_text introduces facts outside the short-term context")
    if _estimate_tokens(result.summary_text) > config.context_token_budget:
        raise MemoryReasonerValidationError("summary_text exceeds the configured context token budget")
    return result


def _validate_sources(
    result: MemoryExtractionResult,
    source_message_ids: Sequence[str],
    source_event_ids: Sequence[str],
    source_decision_ids: Sequence[str],
    source_knowledge_ids: Sequence[str],
) -> None:
    for field, actual, allowed in (
        ("source_message_ids", result.source_message_ids, source_message_ids),
        ("source_event_ids", result.source_event_ids, source_event_ids),
        ("source_decision_ids", result.source_decision_ids, source_decision_ids),
        ("source_knowledge_ids", result.source_knowledge_ids, source_knowledge_ids),
    ):
        if not set(actual).issubset(allowed):
            raise MemoryReasonerValidationError(f"{field} must belong to supplied source IDs")


def _is_grounded(summary: str, source_texts: Sequence[str]) -> bool:
    normalized_summary = _normalize(summary)
    normalized_source = " ".join(_normalize(text) for text in source_texts)
    return bool(normalized_summary and normalized_summary in normalized_source)


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def build_bounded_source_payload(
    source_texts: Sequence[str],
    source_message_ids: Sequence[str],
    source_event_ids: Sequence[str],
    source_decision_ids: Sequence[str],
    source_knowledge_ids: Sequence[str],
    config: MemoryConfig,
) -> dict[str, object]:
    if not source_texts or any(not isinstance(text, str) or not text.strip() for text in source_texts):
        raise ValueError("source_texts must contain non-blank text")
    remaining_chars = config.context_token_budget * 4
    bounded_texts: list[str] = []
    for text in source_texts[-config.recent_message_limit :]:
        if remaining_chars <= 0:
            break
        bounded_texts.append(text[:remaining_chars])
        remaining_chars -= len(bounded_texts[-1])
    return {
        "texts": bounded_texts,
        "source_message_ids": list(source_message_ids),
        "source_event_ids": list(source_event_ids),
        "source_decision_ids": list(source_decision_ids),
        "source_knowledge_ids": list(source_knowledge_ids),
    }


def _short_term_payload(
    context: ShortTermContext | None, config: MemoryConfig
) -> dict[str, object] | None:
    if context is None:
        return None
    messages = context.recent_messages[-config.recent_message_limit :]
    return {
        "summary_text": context.summary_text[: config.context_token_budget * 4],
        "recent_messages": [_message_payload(message) for message in messages],
    }


def _candidate_payload(candidate: MemoryVersion) -> dict[str, object]:
    return {
        "memory_id": candidate.memory_id,
        "memory_family_id": candidate.memory_family_id,
        "version": candidate.version,
        "memory_type": candidate.memory_type.value,
        "summary": candidate.summary,
        "importance_score": candidate.importance_score,
    }


def _message_payload(message: ShortTermMessage) -> dict[str, object]:
    return message.model_dump(mode="json")
