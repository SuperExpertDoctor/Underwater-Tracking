from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi.testclient import TestClient

from underwater_tracking.api.app import create_app
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.domain.memory_models import (
    MemoryStreamEvent,
    MemoryStreamEventType,
    MemoryStreamPayload,
    MemoryStreamStatus,
    MemoryType,
    MemoryVersion,
    ShortTermContext,
)


class _Replay:
    def range(self, *args: Any, **kwargs: Any) -> list[Any]:
        del args, kwargs
        return []


@dataclass
class _MemoryPort:
    snapshot_value: dict[str, object]
    versions_value: list[MemoryVersion] = field(default_factory=list)
    stream_value: list[MemoryStreamEvent] = field(default_factory=list)
    deleted: list[tuple[str, str, str | None]] = field(default_factory=list)

    def snapshot(self, **kwargs: object) -> dict[str, object]:
        self.last_snapshot = kwargs
        return self.snapshot_value

    def versions(self, **kwargs: object) -> list[MemoryVersion]:
        self.last_versions = kwargs
        return self.versions_value

    def delete(self, **kwargs: object) -> bool:
        self.deleted.append((str(kwargs["user_id"]), str(kwargs["memory_id"]), kwargs.get("scenario_id") if isinstance(kwargs.get("scenario_id"), str) else None))
        return True

    def stream(self, **kwargs: object) -> list[MemoryStreamEvent]:
        self.last_stream = kwargs
        after_cursor = int(kwargs.get("after_cursor", 0))
        return [event for event in self.stream_value if event.cursor > after_cursor]


@dataclass
class _Runtime:
    memory_port: _MemoryPort

    def active_plan(self) -> None:
        return None


def _memory(memory_id: str, family_id: str, user_id: str = "analyst-1") -> MemoryVersion:
    return MemoryVersion(
        memory_id=memory_id,
        memory_family_id=family_id,
        version=1,
        user_id=user_id,
        scenario_id="scenario-1",
        memory_type=MemoryType.SEMANTIC,
        summary="真实长期记忆",
        importance_score=0.8,
        embedding=(1.0, 0.0),
    )


def _app(port: _MemoryPort):
    return create_app(
        runtime=_Runtime(port),
        replay=_Replay(),
        memory_port=port,
        hub=OperationalHub(),
    )


def test_memory_snapshot_exposes_short_term_three_families_hits_versions_and_status() -> None:
    memory = _memory("memory-1", "family-1")
    short_term = ShortTermContext(
        user_id="analyst-1",
        scenario_id="scenario-1",
        conversation_id="conversation-1",
        summary_text="已确认的上下文",
    )
    port = _MemoryPort(
        snapshot_value={
            "user_id": "analyst-1",
            "scenario_id": "scenario-1",
            "conversation_id": "conversation-1",
            "short_term": short_term,
            "episodic": [],
            "semantic": [memory],
            "procedural": [],
            "retrieved_hits": [],
            "versions": [memory],
            "memory_status": "completed",
            "degraded_reason": None,
        }
    )

    with TestClient(_app(port)) as client:
        response = client.get(
            "/api/assistant/memory",
            params={"user_id": "analyst-1", "conversation_id": "conversation-1", "scenario_id": "scenario-1"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["short_term"]["summary_text"] == "已确认的上下文"
    assert payload["semantic"][0]["memory_id"] == "memory-1"
    assert payload["episodic"] == []
    assert payload["procedural"] == []
    assert payload["versions"][0]["version"] == 1
    assert payload["memory_status"] == "completed"
    assert port.last_snapshot["user_id"] == "analyst-1"


def test_memory_routes_reject_invalid_scope_pagination_and_cross_user_results() -> None:
    foreign = _memory("memory-foreign", "family-foreign", user_id="other-user")
    port = _MemoryPort(
        snapshot_value={
            "user_id": "other-user",
            "scenario_id": "scenario-1",
            "conversation_id": "conversation-1",
            "short_term": None,
            "episodic": [],
            "semantic": [],
            "procedural": [],
            "retrieved_hits": [],
            "versions": [],
            "memory_status": "completed",
            "degraded_reason": None,
        },
        versions_value=[foreign],
    )

    with TestClient(_app(port)) as client:
        assert client.get("/api/assistant/memory", params={"conversation_id": "bad id"}).status_code == 422
        assert client.get("/api/assistant/memory/stream", params={"conversation_id": "c", "limit": 0}).status_code == 422
        assert client.get("/api/assistant/memory/stream", params={"conversation_id": "c", "after_cursor": -1}).status_code == 422
        response = client.get(
            "/api/assistant/memory/family-foreign/versions",
            params={"user_id": "analyst-1"},
        )

    assert response.status_code == 404


def test_memory_versions_delete_and_stream_are_user_scoped_and_incremental() -> None:
    memory = _memory("memory-1", "family-1")
    event = MemoryStreamEvent(
        cursor=4,
        event_id="event-4",
        user_id="analyst-1",
        scenario_id="scenario-1",
        conversation_id="conversation-1",
        status=MemoryStreamStatus.PENDING,
        type=MemoryStreamEventType.WORK_QUEUED,
        payload=MemoryStreamPayload(work_id="work-1"),
    )
    port = _MemoryPort(
        snapshot_value={
            "user_id": "analyst-1",
            "scenario_id": "scenario-1",
            "conversation_id": "conversation-1",
            "short_term": None,
            "episodic": [],
            "semantic": [memory],
            "procedural": [],
            "retrieved_hits": [],
            "versions": [memory],
            "memory_status": "degraded",
            "degraded_reason": "embedding credentials are unavailable",
        },
        versions_value=[memory],
        stream_value=[event],
    )

    with TestClient(_app(port)) as client:
        versions = client.get(
            "/api/assistant/memory/family-1/versions",
            params={"user_id": "analyst-1", "scenario_id": "scenario-1"},
        )
        deleted = client.request(
            "DELETE",
            "/api/assistant/memory/memory-1",
            json={"user_id": "analyst-1", "scenario_id": "scenario-1"},
        )
        first = client.get(
            "/api/assistant/memory/stream",
            params={"user_id": "analyst-1", "conversation_id": "conversation-1", "after_cursor": 0, "limit": 10},
        )
        second = client.get(
            "/api/assistant/memory/stream",
            params={"user_id": "analyst-1", "conversation_id": "conversation-1", "after_cursor": 4, "limit": 10},
        )

    assert versions.status_code == 200
    assert versions.json()["versions"][0]["memory_id"] == "memory-1"
    assert deleted.status_code == 200
    assert deleted.json() == {"status": "deleted", "memory_id": "memory-1", "user_id": "analyst-1"}
    assert first.status_code == 200
    assert [item["cursor"] for item in first.json()["events"]] == [4]
    assert second.status_code == 200
    assert second.json()["events"] == []
    assert port.deleted == [("analyst-1", "memory-1", "scenario-1")]


def test_memory_snapshot_rejects_backend_scenario_scope_mismatch() -> None:
    port = _MemoryPort(
        snapshot_value={
            "user_id": "analyst-1",
            "scenario_id": "scenario-2",
            "conversation_id": "conversation-1",
            "short_term": None,
            "episodic": [],
            "semantic": [],
            "procedural": [],
            "retrieved_hits": [],
            "versions": [],
            "memory_status": "completed",
            "degraded_reason": None,
        },
    )

    with TestClient(_app(port)) as client:
        response = client.get(
            "/api/assistant/memory",
            params={
                "user_id": "analyst-1",
                "conversation_id": "conversation-1",
                "scenario_id": "scenario-1",
            },
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "memory snapshot scenario scope mismatch"


def test_memory_stream_publishes_requested_scenario_scope() -> None:
    event = MemoryStreamEvent(
        cursor=4,
        event_id="event-4",
        user_id="analyst-1",
        scenario_id="scenario-1",
        conversation_id="conversation-1",
        status=MemoryStreamStatus.COMPLETED,
        type=MemoryStreamEventType.WORK_QUEUED,
        payload=MemoryStreamPayload(work_id="work-1"),
    )
    port = _MemoryPort(
        snapshot_value={},
        stream_value=[event],
    )

    with TestClient(_app(port)) as client:
        response = client.get(
            "/api/assistant/memory/stream",
            params={
                "user_id": "analyst-1",
                "conversation_id": "conversation-1",
                "scenario_id": "scenario-1",
            },
        )

    assert response.status_code == 200
    assert response.json()["scenario_id"] == "scenario-1"


def test_memory_stream_accepts_scenario_events_without_cross_conversation_events() -> None:
    scenario_event = MemoryStreamEvent(
        cursor=3,
        event_id="scenario-event-3",
        user_id="analyst-1",
        scenario_id="scenario-1",
        conversation_id=None,
        status=MemoryStreamStatus.COMPLETED,
        type=MemoryStreamEventType.CONTEXT_LOADED,
    )
    conversation_event = MemoryStreamEvent(
        cursor=4,
        event_id="conversation-event-4",
        user_id="analyst-1",
        scenario_id="scenario-1",
        conversation_id="conversation-1",
        status=MemoryStreamStatus.COMPLETED,
        type=MemoryStreamEventType.MEMORY_EXTRACTED,
    )
    foreign_conversation = MemoryStreamEvent(
        cursor=5,
        event_id="foreign-conversation-5",
        user_id="analyst-1",
        scenario_id="scenario-1",
        conversation_id="conversation-2",
        status=MemoryStreamStatus.COMPLETED,
        type=MemoryStreamEventType.MEMORY_ACCESSED,
    )
    port = _MemoryPort(snapshot_value={}, stream_value=[scenario_event, conversation_event, foreign_conversation])

    with TestClient(_app(port)) as client:
        response = client.get(
            "/api/assistant/memory/stream",
            params={
                "user_id": "analyst-1",
                "conversation_id": "conversation-1",
                "scenario_id": "scenario-1",
                "include_scenario_events": True,
            },
        )

    assert response.status_code == 403

    port.stream_value = [scenario_event, conversation_event]
    with TestClient(_app(port)) as client:
        response = client.get(
            "/api/assistant/memory/stream",
            params={
                "user_id": "analyst-1",
                "conversation_id": "conversation-1",
                "scenario_id": "scenario-1",
                "include_scenario_events": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["include_scenario_events"] is True
    assert [item["type"] for item in payload["events"]] == [
        "context_loaded",
        "memory_extracted",
    ]


def test_memory_stream_reports_adapter_degraded_state_when_empty() -> None:
    port = _MemoryPort(snapshot_value={}, stream_value=[])
    port.degraded_reason = "Embedding credentials are unavailable"

    with TestClient(_app(port)) as client:
        response = client.get(
            "/api/assistant/memory/stream",
            params={
                "user_id": "analyst-1",
                "conversation_id": "conversation-1",
                "scenario_id": "scenario-1",
            },
        )

    assert response.status_code == 200
    assert response.json()["memory_status"] == "degraded"
    assert response.json()["degraded_reason"] == "Embedding credentials are unavailable"
