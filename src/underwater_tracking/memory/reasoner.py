"""Structured, bounded memory operations with server-side result validation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

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

MEMORY_FILTER_PROMPT_VERSION = "memory-filter-v2"
MEMORY_EXTRACT_PROMPT_VERSION = "memory-extract-v1"
SHORT_TERM_COMPRESS_PROMPT_VERSION = "short-term-compress-v1"


class MemoryReasonerValidationError(LLMContentError):
    """A schema-valid LLM result violated a memory ownership constraint."""


class _SourcePayload(TypedDict):
    texts: list[str]
    source_message_ids: list[str]
    source_event_ids: list[str]
    source_decision_ids: list[str]
    source_plan_ids: list[str]


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
        source_plan_ids: Sequence[str] = (),
        short_term_context: ShortTermContext | None = None,
        scenario_id: str | None = None,
    ) -> MemoryFilterDecision:
        """Ask the configured LLM whether bounded source material is durable."""
        candidates = self._repository.list_active(
            user_id,
            filters={"scenario_id": scenario_id} if scenario_id is not None else None,
            limit=self._config.retrieval_candidate_limit,
        )
        budget = _ContextTokenBudget(self._config.context_token_budget)
        source_payload = _build_bounded_source_payload(
            source_texts,
            source_message_ids,
            source_event_ids,
            source_decision_ids,
            source_plan_ids,
            self._config,
            budget,
        )
        short_term_payload, _ = _build_short_term_payload(short_term_context, self._config, budget)
        candidate_payloads, supplied_candidates = _build_bounded_candidate_payloads(candidates, budget)
        payload: dict[str, object] = {
            "instruction": (
                "Decide whether the supplied source material should become durable memory. "
                "Use only the listed candidate IDs and obey these output invariants: "
                "when should_store=true, operation must be create or update and memory_type "
                "must be episodic, semantic, or procedural; when should_store=false, "
                "operation=ignore and memory_type must be null. If unsure, choose "
                "should_store=false with operation=ignore. For update, candidate_memory_id "
                "must be one of the listed IDs; for create, leave it null. Do not invent "
                "IDs or use keyword rules."
            ),
            "user_id": user_id,
            "scenario_id": scenario_id,
            "source": source_payload,
            "short_term": short_term_payload,
            "candidates": candidate_payloads,
            "output_token_budget": 1024,
            "thinking_mode": "disabled",
        }
        decision = self._llm.invoke_structured(
            "memory_filter",
            payload,
            MemoryFilterDecision,
            prompt_version=MEMORY_FILTER_PROMPT_VERSION,
        )
        return validate_filter_decision(decision, candidates=supplied_candidates, user_id=user_id)

    def extract(
        self,
        *,
        user_id: str,
        source_texts: Sequence[str],
        source_message_ids: Sequence[str] = (),
        source_event_ids: Sequence[str] = (),
        source_decision_ids: Sequence[str] = (),
        source_plan_ids: Sequence[str] = (),
    ) -> MemoryExtractionResult:
        """Extract a lexically grounded summary from the supplied sources only."""
        source_payload = build_bounded_source_payload(
            source_texts,
            source_message_ids,
            source_event_ids,
            source_decision_ids,
            self._config,
            source_plan_ids=source_plan_ids,
        )
        payload: dict[str, object] = {
            "instruction": (
                "Create one concise durable-memory summary using only facts in source. "
                "Copy the summary verbatim from source text so it can be server-verified. "
                "Return only source IDs supplied in source."
            ),
            "user_id": user_id,
            "source": source_payload,
            "output_token_budget": 1024,
            "thinking_mode": "disabled",
        }
        result = self._llm.invoke_structured(
            "memory_extract",
            payload,
            MemoryExtractionResult,
            prompt_version=MEMORY_EXTRACT_PROMPT_VERSION,
        )
        return validate_extraction_result(
            result,
            source_texts=source_payload["texts"],
            source_message_ids=source_payload["source_message_ids"],
            source_event_ids=source_payload["source_event_ids"],
            source_decision_ids=source_payload["source_decision_ids"],
            source_plan_ids=source_payload["source_plan_ids"],
            max_summary_chars=min(4000, self._config.context_token_budget * 4),
        )

    def compress_short_term(self, context: ShortTermContext) -> ShortTermCompressionResult:
        """Compress only the configured bounded short-term window through the LLM."""
        payload_context, messages = _build_short_term_payload(
            context,
            self._config,
            _ContextTokenBudget(self._config.context_token_budget),
        )
        assert payload_context is not None
        payload: dict[str, object] = {
            "instruction": (
                "Compress the bounded conversation without adding facts. Copy summary text "
                "verbatim from the existing summary or messages. Retain only complete supplied "
                "messages and cite only supplied message IDs."
            ),
            "user_id": context.user_id,
            "conversation_id": context.conversation_id,
            "existing_summary": payload_context["summary_text"],
            "messages": payload_context["recent_messages"],
            "output_token_budget": 1024,
            "thinking_mode": "disabled",
        }
        result = self._llm.invoke_structured(
            "short_term_compress",
            payload,
            ShortTermCompressionResult,
            prompt_version=SHORT_TERM_COMPRESS_PROMPT_VERSION,
        )
        bounded_context = context.model_copy(
            update={"summary_text": payload_context["summary_text"], "recent_messages": messages}
        )
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
    source_plan_ids: Sequence[str] = (),
    max_summary_chars: int = 4000,
) -> MemoryExtractionResult:
    """Keep source references and summary facts within the supplied source scope."""
    _validate_sources(
        result,
        source_message_ids,
        source_event_ids,
        source_decision_ids,
        source_plan_ids,
    )
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
    result_tokens = _estimate_tokens(result.summary_text) + sum(
        _estimate_payload_tokens(_message_payload(message)) for message in result.retained_messages
    )
    if result_tokens > config.context_token_budget:
        raise MemoryReasonerValidationError(
            "summary_text and retained_messages exceed the configured context token budget"
        )
    return result


def _validate_sources(
    result: MemoryExtractionResult,
    source_message_ids: Sequence[str],
    source_event_ids: Sequence[str],
    source_decision_ids: Sequence[str],
    source_plan_ids: Sequence[str],
) -> None:
    for field, actual, allowed in (
        ("source_message_ids", result.source_message_ids, source_message_ids),
        ("source_event_ids", result.source_event_ids, source_event_ids),
        ("source_decision_ids", result.source_decision_ids, source_decision_ids),
        ("source_plan_ids", result.source_plan_ids, source_plan_ids),
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


def estimate_memory_payload_tokens(payload: Mapping[str, object]) -> int:
    """Estimate the dynamic memory context carried by a structured LLM payload."""
    return sum(
        _estimate_payload_tokens(value)
        for field, value in payload.items()
        if field
        not in {
            "instruction",
            "user_id",
            "conversation_id",
            # These are HTTP client controls, not dynamic memory context.
            "output_token_budget",
            "thinking_mode",
        }
    )


def _estimate_payload_tokens(value: object) -> int:
    if isinstance(value, str):
        return _estimate_tokens(value)
    if isinstance(value, Mapping):
        return sum(_estimate_payload_tokens(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(_estimate_payload_tokens(item) for item in value)
    return 0


class _ContextTokenBudget:
    """Accept complete context values while enforcing one invocation-wide budget."""

    def __init__(self, token_limit: int) -> None:
        self._remaining = token_limit

    def include(self, value: object) -> bool:
        token_count = _estimate_payload_tokens(value)
        if token_count > self._remaining:
            return False
        self._remaining -= token_count
        return True


def build_bounded_source_payload(
    source_texts: Sequence[str],
    source_message_ids: Sequence[str],
    source_event_ids: Sequence[str],
    source_decision_ids: Sequence[str],
    config: MemoryConfig,
    source_plan_ids: Sequence[str] = (),
) -> _SourcePayload:
    return _build_bounded_source_payload(
        source_texts,
        source_message_ids,
        source_event_ids,
        source_decision_ids,
        source_plan_ids,
        config,
        _ContextTokenBudget(config.context_token_budget),
    )


def _build_bounded_source_payload(
    source_texts: Sequence[str],
    source_message_ids: Sequence[str],
    source_event_ids: Sequence[str],
    source_decision_ids: Sequence[str],
    source_plan_ids: Sequence[str],
    config: MemoryConfig,
    budget: _ContextTokenBudget,
) -> _SourcePayload:
    if not source_texts or any(not isinstance(text, str) or not text.strip() for text in source_texts):
        raise ValueError("source_texts must contain non-blank text")
    bounded_texts: list[str] = []
    for text in source_texts[-config.recent_message_limit :]:
        if budget.include(text):
            bounded_texts.append(text)
    bounded_ids: list[list[str]] = []
    for source_ids in (
        source_message_ids,
        source_event_ids,
        source_decision_ids,
        source_plan_ids,
    ):
        selected_ids: list[str] = []
        for source_id in source_ids:
            if budget.include(source_id):
                selected_ids.append(source_id)
        bounded_ids.append(selected_ids)
    return {
        "texts": bounded_texts,
        "source_message_ids": bounded_ids[0],
        "source_event_ids": bounded_ids[1],
        "source_decision_ids": bounded_ids[2],
        "source_plan_ids": bounded_ids[3],
    }


def _build_short_term_payload(
    context: ShortTermContext | None,
    config: MemoryConfig,
    budget: _ContextTokenBudget,
) -> tuple[dict[str, object] | None, tuple[ShortTermMessage, ...]]:
    if context is None:
        return None, ()
    summary_text = context.summary_text if budget.include(context.summary_text) else ""
    selected_messages: list[ShortTermMessage] = []
    for message in reversed(context.recent_messages[-config.recent_message_limit :]):
        if budget.include(_message_payload(message)):
            selected_messages.append(message)
    messages = tuple(reversed(selected_messages))
    return (
        {
            "summary_text": summary_text,
            "recent_messages": [_message_payload(message) for message in messages],
        },
        messages,
    )


def _build_bounded_candidate_payloads(
    candidates: Sequence[MemoryVersion], budget: _ContextTokenBudget
) -> tuple[list[dict[str, object]], tuple[MemoryVersion, ...]]:
    candidate_payloads: list[dict[str, object]] = []
    supplied_candidates: list[MemoryVersion] = []
    for candidate in candidates:
        candidate_payload = _candidate_payload(candidate)
        if budget.include(candidate_payload):
            candidate_payloads.append(candidate_payload)
            supplied_candidates.append(candidate)
    return candidate_payloads, tuple(supplied_candidates)


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
