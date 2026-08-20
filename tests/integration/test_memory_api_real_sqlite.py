from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from underwater_tracking.api.app import create_app
from underwater_tracking.api.dependencies import MemoryServiceAdapter
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.domain.memory_models import MemoryType, MemoryVersion
from underwater_tracking.memory.retriever import DegradedMemoryRetriever
from underwater_tracking.memory.service import MemoryService
from underwater_tracking.persistence.memory import LongTermMemoryRepository, ShortTermContextRepository


class _Runtime:
    def active_plan(self) -> None:
        return None


class _Replay:
    def range(self, *args: object, **kwargs: object) -> list[object]:
        del args, kwargs
        return []


def _build_app(database: Path):
    short_term = ShortTermContextRepository(database)
    long_term = LongTermMemoryRepository(database)
    service = MemoryService(
        short_term,
        long_term,
        DegradedMemoryRetriever("embedding provider is unavailable"),
        degraded_reason="embedding provider is unavailable",
    )
    adapter = MemoryServiceAdapter(service, scenario_id="scenario-1")
    return create_app(
        runtime=_Runtime(),
        replay=_Replay(),
        memory_port=adapter,
        hub=OperationalHub(),
    ), short_term, long_term


def test_real_sqlite_memory_survives_app_rebuild_and_delete_keeps_audit_rows(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    app, short_term, long_term = _build_app(database)
    outcome = MemoryService(
        short_term,
        long_term,
        DegradedMemoryRetriever("embedding provider is unavailable"),
        degraded_reason="embedding provider is unavailable",
    ).accept_turn(
        {
            "user_id": "analyst-1",
            "conversation_id": "conversation-1",
            "scenario_id": "scenario-1",
            "message_id": "message-1",
            "text": "keep the acoustic report concise",
        },
        None,
    )
    memory = MemoryVersion(
        memory_id="memory-1",
        memory_family_id="family-1",
        version=1,
        user_id="analyst-1",
        scenario_id="scenario-1",
        memory_type=MemoryType.PROCEDURAL,
        summary="保持声学报告简洁",
        importance_score=0.9,
        embedding=(1.0,),
    )
    long_term.create_memory_version(memory, expected_previous_version=0)
    with TestClient(app) as client:
        stream = client.get("/api/assistant/memory/stream", params={"user_id": "analyst-1", "conversation_id": "conversation-1", "scenario_id": "scenario-1"})
        assert stream.status_code == 200
        assert stream.json()["events"][0]["cursor"] == outcome["stream_cursor"]

    short_term.close()
    long_term.close()

    rebuilt, rebuilt_short, rebuilt_long = _build_app(database)
    with TestClient(rebuilt) as client:
        snapshot = client.get("/api/assistant/memory", params={"user_id": "analyst-1", "conversation_id": "conversation-1", "scenario_id": "scenario-1"})
        assert snapshot.status_code == 200
        assert snapshot.json()["procedural"][0]["memory_id"] == "memory-1"
        cross_user_versions = client.get(
            "/api/assistant/memory/family-1/versions",
            params={"user_id": "other-user", "scenario_id": "scenario-1"},
        )
        assert cross_user_versions.status_code in {403, 404}
        deleted = client.request("DELETE", "/api/assistant/memory/memory-1", json={"user_id": "analyst-1", "scenario_id": "scenario-1", "conversation_id": "conversation-1"})
        assert deleted.status_code == 200
        after_delete = client.get("/api/assistant/memory", params={"user_id": "analyst-1", "conversation_id": "conversation-1", "scenario_id": "scenario-1"})
        assert after_delete.status_code == 200
        assert after_delete.json()["procedural"] == []

    delete_events = rebuilt_long.list_stream_events(
        "analyst-1", "conversation-1", scenario_id="scenario-1", limit=10
    )
    assert any(event.type.value == "memory_deleted" for event in delete_events)
    assert rebuilt_long._conn.execute("SELECT COUNT(*) FROM memory_stream_events").fetchone()[0] >= 1
    assert rebuilt_short.get_short_term("analyst-1", "conversation-1", "scenario-1") is not None
    rebuilt_short.close()
    rebuilt_long.close()
