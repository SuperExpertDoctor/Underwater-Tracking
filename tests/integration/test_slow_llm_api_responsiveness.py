from __future__ import annotations

from threading import Event
from pathlib import Path
from time import monotonic
from time import sleep
from typing import Any

import pytest
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
        self.release.wait()
        raise LLMError("test provider released")


def test_health_remains_responsive_while_provider_is_blocked(tmp_path: Path) -> None:
    provider = BlockingLLM()
    controller = RunController(
        load_app_config(CONFIG_PATH),
        output_root=tmp_path / "outputs",
        llm={"master": provider},
        steps=0,
        bootstrap_planning=True,
    )
    controller.start_run(1, seed=7)
    try:
        assert provider.started.wait(timeout=10.0)
        app = create_app(controller=controller)
        with TestClient(app) as client:
            try:
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
                initial_frame = controller.hub.snapshot()
                assert initial_frame is not None
                release_frame = controller.hub.snapshot()
                assert release_frame is not None
                provider.release.set()
                deadline = monotonic() + 15.0
                failed = None
                while monotonic() < deadline:
                    candidate = controller.current()
                    if candidate.status == "failed":
                        failed = candidate
                        break
                    sleep(0.05)

                assert failed is not None
                assert failed.sim_time_s >= release_frame.sim_time_s
                with pytest.raises(LLMError, match="test provider released"):
                    controller.raise_if_failed()
                stable_frame = controller.hub.snapshot()
                assert stable_frame is not None
                sleep(0.2)
                final_frame = controller.hub.snapshot()
                assert final_frame is not None
                assert final_frame.frame_id == stable_frame.frame_id
                assert final_frame.sim_time_s == stable_frame.sim_time_s
            finally:
                provider.release.set()
    finally:
        provider.release.set()
        controller.close()
