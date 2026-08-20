"""Pure server-side validation for structured memory LLM results."""

from __future__ import annotations

import pytest

from underwater_tracking.config.models import MemoryConfig
from underwater_tracking.domain.memory_models import (
    MemoryExtractionResult,
    MemoryFilterDecision,
    MemoryType,
    MemoryVersion,
    ShortTermCompressionResult,
    ShortTermContext,
    ShortTermMessage,
)
from underwater_tracking.memory.reasoner import (
    MemoryReasonerValidationError,
    build_bounded_source_payload,
    validate_compression_result,
    validate_extraction_result,
    validate_filter_decision,
)


def _config() -> MemoryConfig:
    return MemoryConfig(
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding-test-v1",
        context_token_budget=40,
    )


def _candidate(*, user_id: str = "operator") -> MemoryVersion:
    return MemoryVersion(
        memory_id="memory-1",
        memory_family_id="family-1",
        version=1,
        user_id=user_id,
        memory_type=MemoryType.SEMANTIC,
        summary="Operator prefers concise evidence-backed reports.",
        importance_score=0.7,
    )


def test_filter_rejects_update_target_outside_user_scoped_candidates() -> None:
    decision = MemoryFilterDecision(
        should_store=True,
        memory_type=MemoryType.SEMANTIC,
        operation="update",
        candidate_memory_id="memory-not-provided",
        importance_score=0.8,
        reason="The preference is durable.",
    )

    with pytest.raises(MemoryReasonerValidationError, match="candidate_memory_id"):
        validate_filter_decision(decision, candidates=(_candidate(),), user_id="operator")


def test_filter_rejects_cross_user_candidate_even_when_id_matches() -> None:
    decision = MemoryFilterDecision(
        should_store=True,
        memory_type=MemoryType.SEMANTIC,
        operation="update",
        candidate_memory_id="memory-1",
        importance_score=0.8,
        reason="The preference is durable.",
    )

    with pytest.raises(MemoryReasonerValidationError, match="user_id"):
        validate_filter_decision(decision, candidates=(_candidate(user_id="other-user"),), user_id="operator")


def test_extraction_rejects_unprovided_source_ids_and_ungrounded_summary() -> None:
    unprovided_source = MemoryExtractionResult(
        summary="Operator prefers concise evidence-backed reports.",
        source_message_ids=("message-not-provided",),
        change_reason="A durable reporting preference was stated.",
    )
    with pytest.raises(MemoryReasonerValidationError, match="source_message_ids"):
        validate_extraction_result(
            unprovided_source,
            source_texts=("Operator prefers concise evidence-backed reports.",),
            source_message_ids=("message-1",),
        )

    invented_fact = MemoryExtractionResult(
        summary="Operator prefers classified satellite reports.",
        source_message_ids=("message-1",),
        change_reason="A durable reporting preference was stated.",
    )
    with pytest.raises(MemoryReasonerValidationError, match="summary"):
        validate_extraction_result(
            invented_fact,
            source_texts=("Operator prefers concise evidence-backed reports.",),
            source_message_ids=("message-1",),
        )


def test_compression_rejects_messages_not_present_in_bounded_context() -> None:
    context = ShortTermContext(
        user_id="operator",
        conversation_id="conversation-1",
        recent_messages=(
            ShortTermMessage(
                message_id="message-1",
                role="user",
                text="Keep the report concise and cite evidence.",
            ),
        ),
    )
    result = ShortTermCompressionResult(
        summary_text="Keep the report concise and cite evidence.",
        retained_messages=(
            ShortTermMessage(
                message_id="message-unknown",
                role="assistant",
                text="Invented retained message.",
            ),
        ),
        source_message_ids=("message-1",),
    )

    with pytest.raises(MemoryReasonerValidationError, match="retained_messages"):
        validate_compression_result(result, context=context, config=_config())


def test_extraction_accepts_grounded_non_latin_source_text() -> None:
    result = MemoryExtractionResult(
        summary="操作员偏好简洁并附带证据的报告。",
        source_message_ids=("message-1",),
        change_reason="用户明确表达报告偏好。",
    )

    assert validate_extraction_result(
        result,
        source_texts=("操作员偏好简洁并附带证据的报告。",),
        source_message_ids=("message-1",),
    ) == result


def test_source_payload_limits_current_source_text_to_memory_config() -> None:
    payload = build_bounded_source_payload(
        ("older source", "latest source exceeds the configured prompt budget"),
        ("message-1", "message-2"),
        (),
        (),
        (),
        MemoryConfig(
            embedding_base_url="https://api.example.test/v1",
            embedding_model="embedding-test-v1",
            recent_message_limit=1,
            context_token_budget=5,
        ),
    )

    assert payload["texts"] == ["latest source exceed"]
    assert payload["source_message_ids"] == ["message-1", "message-2"]
