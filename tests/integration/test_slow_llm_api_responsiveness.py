from __future__ import annotations

from threading import Event
from pathlib import Path
from time import monotonic
from typing import Any

from fastapi.testclient import TestClient

from underwater_tracking.agent.llm import LLMError
from underwater_tracking.api.app import create_app
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.runtime.run_controller import RunController


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/scenario/uuv_only_single_target.yaml"
)


class BlockingLLM:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        del operation, payload, response_model, prompt_version
        self.started.set()
        self.release.wait(timeout=10.0)
        raise LLMError("test provider released")


def test_health_remains_responsive_while_provider_is_blocked(tmp_path: Path) -> None:
    provider = BlockingLLM()
    controller = RunController(
        load_app_config(CONFIG_PATH),
        output_root=tmp_path / "outputs",
        llm={"master": provider},
        steps=0,
        speed=0.0,
    )
    controller.start_run(1, seed=7)
    try:
        assert provider.started.wait(timeout=10.0)
        app = create_app(controller=controller)
        with TestClient(app) as client:
            latencies: list[float] = []
            payload: dict[str, Any] = {}
            for _ in range(20):
                started = monotonic()
                response = client.get("/api/health")
                latencies.append(monotonic() - started)
                assert response.status_code == 200
                payload = response.json()
            assert max(latencies) < 0.5
            assert payload["planning"]["status"] == "running"
            assert payload["plan_version"] == 0
    finally:
        provider.release.set()
        controller.close()
