from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from fastapi.testclient import TestClient

from underwater_tracking.api.app import create_app
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.ui_models import PlanningHealthView
from underwater_tracking.runtime.run_controller import RunController


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/scenario/segmented_single_target.yaml"


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
        self.abort_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    def abort(self) -> None:
        self.abort_calls += 1


def test_lifespan_does_not_start_another_worker_and_closes_controller_once() -> None:
    controller = _Controller()
    app = create_app(controller=controller)

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert not hasattr(app.state, "memory_worker_started")

    assert controller.close_calls == 1


def test_lifespan_aborts_controller_when_asyncio_cancels_shutdown() -> None:
    controller = _Controller()
    app = create_app(controller=controller)

    async def cancelled_lifespan() -> None:
        async with app.router.lifespan_context(app):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancelled_lifespan())

    assert controller.abort_calls == 1
    assert controller.close_calls == 0


def test_injected_runtime_lifespan_only_closes_request_queue() -> None:
    runtime = _Runtime(_MemoryPort())
    app = create_app(runtime=runtime, replay=_Replay(), hub=OperationalHub())

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200


def test_lifespan_closes_controller_even_when_queue_close_raises() -> None:
    class BrokenQueue:
        def close(self) -> None:
            raise RuntimeError("queue close failed")

    controller = _Controller()
    app = create_app(controller=controller, directive_queue=BrokenQueue())

    with pytest.raises(RuntimeError, match="queue close failed"):
        with TestClient(app) as client:
            assert client.get("/api/health").status_code == 200

    assert controller.close_calls == 1


def test_health_exposes_explicit_planning_chat_and_memory_degraded_status() -> None:
    runtime = _Runtime(_MemoryPort())
    runtime.llm_paused = True
    runtime.llm_pause_reason = "chat credentials are unavailable"
    runtime.llm_reconnectable = False
    runtime.memory_port.degraded_reason = "memory credentials are unavailable"
    app = create_app(runtime=runtime, replay=_Replay(), hub=OperationalHub())

    payload = TestClient(app).get("/api/health").json()

    assert payload["planning_status"] == "degraded"
    assert payload["chat_status"] == "degraded"
    assert payload["chat_degraded_reason"] == runtime.llm_pause_reason
    assert payload["memory_status"] == "degraded"
    assert payload["memory_degraded_reason"] == runtime.memory_port.degraded_reason


def test_health_exposes_structured_planning_epoch_status() -> None:
    runtime = _Runtime(_MemoryPort())
    runtime.planning_health = lambda: PlanningHealthView(
        status="running",
        epoch_id="epoch:S1:20:event:a1",
        base_physics_revision=20,
        current_physics_revision=24,
        queued_event_count=2,
        last_result_status=None,
    )
    app = create_app(runtime=runtime, replay=_Replay(), hub=OperationalHub())

    payload = TestClient(app).get("/api/health").json()

    assert payload["planning"]["status"] == "running"
    assert payload["planning"]["epoch_id"] == "epoch:S1:20:event:a1"
    assert payload["planning"]["base_physics_revision"] == 20
    assert payload["planning"]["current_physics_revision"] == 24
    assert payload["planning"]["queued_event_count"] == 2


def test_serve_controller_starts_legacy_flat_config_in_degraded_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNDERWATER_TRACKING_API_KEY", "")
    config = load_app_config(CONFIG_PATH)
    assert config.llm is not None
    config = config.model_copy(
        update={
            "llm": config.llm.model_copy(update={"api_key": None, "roles": None}),
        }
    )
    controller = RunController(
        config,
        output_root=tmp_path / "outputs",
        steps=0,
        speed=0.0,
    )
    controller.start_run(1, seed=7)

    try:
        app = create_app(controller=controller)
        with TestClient(app) as client:
            payload = client.get("/api/health").json()
    finally:
        controller.close()

    assert payload["status"] == "paused"
    assert payload["planning_status"] == "degraded"
    assert payload["chat_status"] == "degraded"
    assert "chat" in payload["chat_degraded_reason"]
    assert payload["memory_status"] == "degraded"
    assert payload["memory_degraded_reason"]
