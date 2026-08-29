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
    MemoryReasoner,
    MemoryReasonerValidationError,
    build_bounded_source_payload,
    estimate_memory_payload_tokens,
    validate_compression_result,
    validate_extraction_result,
    validate_filter_decision,
)
from underwater_tracking.persistence.memory import LongTermMemoryRepository


def _config(**changes: object) -> MemoryConfig:
    values: dict[str, object] = {
        "embedding_base_url": "https://api.example.test/v1",
        "embedding_model": "embedding-test-v1",
        "context_token_budget": 40,
    }
    values.update(changes)
    return MemoryConfig(**values)


class RecordingStructuredLLM:
    """Records a real structured-invocation boundary without a transport."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.operation = ""
        self.payload: dict[str, object] = {}

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: object,
        *,
        prompt_version: str = "",
    ) -> object:
        self.operation = operation
        self.payload = payload
        return self.response


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
        MemoryConfig(
            embedding_base_url="https://api.example.test/v1",
            embedding_model="embedding-test-v1",
            recent_message_limit=1,
            context_token_budget=5,
        ),
    )

    assert payload["texts"] == []
    assert payload["source_message_ids"] == ["message-1"]


def test_source_payload_bounds_complete_texts_and_reference_ids_together() -> None:
    config = _config(context_token_budget=8, recent_message_limit=2)
    source_texts = ("first complete source", "latest complete source")
    payload = build_bounded_source_payload(
        source_texts,
        tuple(f"message-{index}" for index in range(10)),
        tuple(f"event-{index}" for index in range(10)),
        (),
        config,
    )

    assert estimate_memory_payload_tokens({"source": payload}) <= config.context_token_budget
    assert all(text in source_texts for text in payload["texts"])
    assert len(payload["source_message_ids"]) < 10
    assert len(payload["source_event_ids"]) < 10


def test_filter_payload_bounds_source_short_term_and_candidates_together(tmp_path) -> None:
    config = _config(
        context_token_budget=35,
        recent_message_limit=2,
        retrieval_top_k=3,
        retrieval_candidate_limit=3,
    )
    repository = LongTermMemoryRepository(tmp_path / "memory.db")
    for index in range(3):
        repository.create_memory_version(
            MemoryVersion(
                memory_id=f"memory-{index}",
                memory_family_id=f"family-{index}",
                version=1,
                user_id="operator",
                memory_type=MemoryType.SEMANTIC,
                summary=f"candidate summary {index} " + "detail " * 4,
                importance_score=0.7,
            ),
            expected_previous_version=0,
        )
    llm = RecordingStructuredLLM(
        MemoryFilterDecision(
            should_store=True,
            memory_type=MemoryType.SEMANTIC,
            operation="create",
            importance_score=0.8,
            reason="The supplied preference is durable.",
        )
    )
    context = ShortTermContext(
        user_id="operator",
        conversation_id="conversation-1",
        summary_text="existing summary " + "detail " * 3,
        recent_messages=(
            ShortTermMessage(message_id="message-1", role="user", text="message " + "detail " * 3),
            ShortTermMessage(message_id="message-2", role="assistant", text="reply " + "detail " * 3),
        ),
    )
    reasoner = MemoryReasoner(llm=llm, repository=repository, config=config)

    try:
        reasoner.filter(
            user_id="operator",
            source_texts=("source " + "detail " * 4,),
            source_message_ids=("source-message-1",),
            short_term_context=context,
        )
    finally:
        repository.close()

    assert llm.operation == "memory_filter"
    assert llm.payload["output_token_budget"] == 1024
    assert llm.payload["thinking_mode"] == "disabled"
    instruction = str(llm.payload["instruction"])
    assert "should_store=true" in instruction
    assert "memory_type" in instruction
    assert "should_store=false" in instruction
    assert "operation=ignore" in instruction
    assert estimate_memory_payload_tokens(llm.payload) <= config.context_token_budget
    assert len(llm.payload["candidates"]) < config.retrieval_candidate_limit


def test_filter_payload_keeps_only_complete_candidate_items_that_fit(tmp_path) -> None:
    config = _config(
        context_token_budget=26,
        retrieval_top_k=3,
        retrieval_candidate_limit=3,
    )
    repository = LongTermMemoryRepository(tmp_path / "memory.db")
    for index in range(3):
        repository.create_memory_version(
            MemoryVersion(
                memory_id=f"memory-{index}",
                memory_family_id=f"family-{index}",
                version=1,
                user_id="operator",
                memory_type=MemoryType.SEMANTIC,
                summary="complete candidate " + "detail " * 2,
                importance_score=0.7,
            ),
            expected_previous_version=0,
        )
    llm = RecordingStructuredLLM(
        MemoryFilterDecision(
            should_store=False,
            operation="ignore",
            reason="The supplied source is not durable.",
        )
    )
    reasoner = MemoryReasoner(llm=llm, repository=repository, config=config)

    try:
        reasoner.filter(user_id="operator", source_texts=("a",))
    finally:
        repository.close()

    candidates = llm.payload["candidates"]
    assert estimate_memory_payload_tokens(llm.payload) <= config.context_token_budget
    assert len(candidates) == 1
    assert candidates[0]["summary"] == "complete candidate detail detail "


def test_compression_uses_bounded_messages_and_validates_retained_budget(tmp_path) -> None:
    config = _config(context_token_budget=1, recent_message_limit=2)
    first_message = ShortTermMessage(message_id="message-1", role="user", text="x" * 4000)
    second_message = ShortTermMessage(message_id="message-2", role="assistant", text="y" * 4000)
    context = ShortTermContext(
        user_id="operator",
        conversation_id="conversation-1",
        summary_text="a",
        recent_messages=(first_message, second_message),
    )
    llm = RecordingStructuredLLM(
        ShortTermCompressionResult(summary_text="a", source_message_ids=())
    )
    repository = LongTermMemoryRepository(tmp_path / "memory.db")
    reasoner = MemoryReasoner(llm=llm, repository=repository, config=config)

    try:
        reasoner.compress_short_term(context)
    finally:
        repository.close()

    assert llm.operation == "short_term_compress"
    assert estimate_memory_payload_tokens(llm.payload) <= config.context_token_budget
    assert llm.payload["messages"] == []
    oversized_result = ShortTermCompressionResult(
        summary_text="a",
        retained_messages=(first_message, second_message),
        source_message_ids=(),
    )
    with pytest.raises(MemoryReasonerValidationError, match="retained_messages"):
        validate_compression_result(oversized_result, context=context, config=config)
