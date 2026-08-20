from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from underwater_tracking.api.app import create_app
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.domain.conversation_models import ConversationMessage, ConversationTurnResult
from underwater_tracking.domain.memory_models import MemoryContext, MemoryStreamStatus


class _Runtime:
    def __init__(self) -> None:
        self.received: list[ConversationMessage] = []

    def active_plan(self) -> None:
        return None

    def conversation_message(self, message: ConversationMessage) -> ConversationTurnResult:
        self.received.append(message)
        return ConversationTurnResult(
            conversation_id=message.conversation_id,
            turn_id="conversation-1:turn:1",
            user_id=message.user_id,
            assistant_mode=message.assistant_mode,
            classification={"classification": "clarification"},
            messages=(
                message,
                message.model_copy(
                    update={
                        "message_id": "assistant-1",
                        "role": "assistant",
                        "text": "请提供更多当前态势信息。",
                        "classification": "clarification",
                        "turn_id": "conversation-1:turn:1",
                    }
                ),
            ),
            expected_plan_version=0,
            memory_context=MemoryContext(
                user_id=message.user_id,
                memory_status=MemoryStreamStatus.PENDING,
            ),
            queued_memory_work_id="work-1",
        )


class _Replay:
    def range(self, start_s: float = 0.0, end_s: float | None = None) -> list[Any]:
        del start_s, end_s
        return []


def test_conversation_http_accepts_user_and_assistant_mode_and_returns_memory_status() -> None:
    runtime = _Runtime()
    app = create_app(runtime=runtime, replay=_Replay(), hub=OperationalHub())

    response = TestClient(app).post(
        "/api/conversation/messages",
        json={
            "conversation_id": "conversation-1",
            "text": "请说明当前证据",
            "expected_plan_version": 0,
            "user_id": "analyst-1",
            "assistant_mode": "evidence_query",
        },
    )

    assert response.status_code == 200
    assert runtime.received[0].user_id == "analyst-1"
    assert runtime.received[0].assistant_mode == "evidence_query"
    assert response.json()["memory_context"]["memory_status"] == "pending"
    assert response.json()["queued_memory_work_id"] == "work-1"
