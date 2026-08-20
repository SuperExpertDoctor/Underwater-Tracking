import pytest
from pydantic import ValidationError

from underwater_tracking.domain.conversation_models import (
    ConversationClassification,
    ConversationMessage,
    ConversationTurnResult,
)
from underwater_tracking.domain.memory_models import MemoryContext


def _message(**changes: object) -> ConversationMessage:
    values: dict[str, object] = {
        "message_id": "message-1",
        "conversation_id": "conversation-1",
        "role": "expert",
        "text": "Show the evidence for this proposal.",
    }
    values.update(changes)
    return ConversationMessage(**values)


def test_conversation_message_defaults_user_and_restricts_assistant_mode() -> None:
    assert _message().user_id == "operator"
    assert _message().assistant_mode == "auto"

    with pytest.raises(ValidationError, match="user_id"):
        _message(user_id="")
    with pytest.raises(ValidationError):
        _message(user_id="u" * 121)
    with pytest.raises(ValidationError):
        _message(assistant_mode="freeform")


def test_turn_result_exposes_memory_response_contract() -> None:
    fields = ConversationTurnResult.model_fields

    assert {"user_id", "assistant_mode", "memory_context", "memory_stream_cursor", "queued_memory_work_id"} <= set(fields)


def test_turn_result_rejects_cross_user_messages_and_memory_context() -> None:
    classification = ConversationClassification(classification="evidence_query")
    with pytest.raises(ValidationError, match="messages"):
        ConversationTurnResult(
            conversation_id="conversation-1",
            turn_id="turn-1",
            user_id="operator",
            classification=classification,
            messages=(_message(user_id="other"),),
            expected_plan_version=0,
        )
    with pytest.raises(ValidationError, match="memory_context"):
        ConversationTurnResult(
            conversation_id="conversation-1",
            turn_id="turn-1",
            user_id="operator",
            classification=classification,
            messages=(_message(),),
            expected_plan_version=0,
            memory_context=MemoryContext(user_id="other"),
        )
