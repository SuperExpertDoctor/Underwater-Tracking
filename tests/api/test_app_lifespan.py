from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from underwater_tracking.api.app import create_app
from underwater_tracking.api.hub import OperationalHub


class _Replay:
    def range(self, *args: Any, **kwargs: Any) -> list[Any]:
        del args, kwargs
        return []


class _MemoryPort:
    def snapshot(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {"user_id": "operator", "short_term": None, "episodic": [], "semantic": [], "procedural": [], "retrieved_hits": [], "versions": [], "memory_status": "degraded", "degraded_reason": "test"}

    def versions(self, **kwargs: object) -> list[Any]:
        del kwargs
        return []

    def delete(self, **kwargs: object) -> bool:
        del kwargs
        return False

    def stream(self, **kwargs: object) -> list[Any]:
        del kwargs
        return []


@dataclass
class _Runtime:
    memory_port: _MemoryPort

    def active_plan(self) -> None:
        return None


class _Controller:
    def __init__(self) -> None:
        self.runtime = _Runtime(_MemoryPort())
        self.replay = _Replay()
        self.hub = OperationalHub()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_lifespan_does_not_start_another_worker_and_closes_controller_once() -> None:
    controller = _Controller()
    app = create_app(controller=controller)

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert not hasattr(app.state, "memory_worker_started")

    assert controller.close_calls == 1


def test_injected_runtime_lifespan_only_closes_request_queue() -> None:
    runtime = _Runtime(_MemoryPort())
    app = create_app(runtime=runtime, replay=_Replay(), hub=OperationalHub())

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
