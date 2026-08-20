from __future__ import annotations

from pathlib import Path
from threading import Event, RLock
from typing import Any

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.runtime.run_controller import RunController, _RunBundle


CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs/scenario/default.yaml"
)
EXPLICIT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/scenario/segmented_single_target.yaml"
)


class FakeLLM:
    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        del operation, payload, prompt_version
        raise AssertionError(f"unexpected LLM call for {response_model!r}")


def _controller(tmp_path: Path) -> RunController:
    return RunController(
        load_app_config(CONFIG_PATH),
        output_root=tmp_path / "outputs",
        llm={"master": FakeLLM()},
        steps=1,
        speed=0.0,
    )


def test_synthetic_target_counts_create_distinct_run_bundles(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    try:
        summaries = [
            controller.start_run(target_count, seed=target_count + 6)
            for target_count in range(1, 5)
        ]
        first, *_, second = summaries

        assert first.target_count == 1
        assert second.target_count == 4
        assert len({summary.run_id for summary in summaries}) == 4
        assert len({summary.path for summary in summaries}) == 4
        assert first.path.name.startswith("serve-")
        assert second.path.is_dir()
        assert controller.current().run_id == second.run_id
    finally:
        controller.close()


@pytest.mark.parametrize("target_count", [0, 5])
def test_invalid_target_count_preserves_current_bundle(
    tmp_path: Path, target_count: int
) -> None:
    controller = _controller(tmp_path)
    try:
        current = controller.start_run(1, seed=7)

        with pytest.raises(ValueError, match="target_count"):
            controller.start_run(target_count, seed=8)

        assert controller.current() == current
        assert controller.runtime is not None
        assert controller.replay is not None
        assert controller.hub is not None
    finally:
        controller.close()


def test_explicit_roster_does_not_invent_additional_targets(tmp_path: Path) -> None:
    controller = RunController(
        load_app_config(EXPLICIT_CONFIG_PATH),
        output_root=tmp_path / "outputs",
        llm={"master": FakeLLM()},
        steps=1,
        speed=0.0,
    )
    try:
        current = controller.start_run(1, seed=7)

        with pytest.raises(ValueError, match="platform-core target roster"):
            controller.start_run(2, seed=8)

        assert controller.current() == current
    finally:
        controller.close()


def test_cli_speed_zero_remains_an_unthrottled_override_not_a_config_value(
    tmp_path: Path,
) -> None:
    config = load_app_config(CONFIG_PATH)
    controller = RunController(
        config,
        output_root=tmp_path / "outputs",
        llm={"master": FakeLLM()},
        steps=1,
        speed=0.0,
    )

    assert config.timing.demo_time_scale == 60.0
    assert controller._speed == 0.0


def test_close_keeps_bundle_installed_when_agent_loop_reports_incomplete() -> None:
    class Loop:
        def __init__(self) -> None:
            self.close_results = iter((False, True))
            self.close_calls = 0
            self.manifest_calls = 0

        def write_manifest(self, _run_dir: Path) -> None:
            self.manifest_calls += 1

        def close(self) -> bool:
            self.close_calls += 1
            return next(self.close_results)

    loop = Loop()
    bundle = _RunBundle(
        config=Any,
        run_dir=Path("run"),
        loop=loop,
        engine=Any,
        replay=Any,
        hub=Any,
        stop=Event(),
        worker_errors=[],
    )
    controller = RunController.__new__(RunController)
    controller._lock = RLock()
    controller._bundle = bundle

    controller.close()
    assert controller._bundle is bundle
    assert loop.close_calls == 1

    controller.close()
    assert controller._bundle is None
    assert loop.close_calls == 2
    assert loop.manifest_calls == 1


def test_close_keeps_bundle_when_simulation_worker_does_not_stop() -> None:
    class Worker:
        alive = True

        def join(self, timeout: float) -> None:
            del timeout

        def is_alive(self) -> bool:
            return self.alive

    class Loop:
        close_calls = 0
        manifest_calls = 0

        def write_manifest(self, _run_dir: Path) -> None:
            self.manifest_calls += 1

        def close(self) -> bool:
            self.close_calls += 1
            return True

    worker = Worker()
    loop = Loop()
    bundle = _RunBundle(
        config=Any,
        run_dir=Path("run"),
        loop=loop,
        engine=Any,
        replay=Any,
        hub=Any,
        stop=Event(),
        worker_errors=[],
        worker=worker,
    )
    controller = RunController.__new__(RunController)
    controller._lock = RLock()
    controller._bundle = bundle

    controller.close()
    assert controller._bundle is bundle
    assert loop.close_calls == 0
    assert loop.manifest_calls == 0

    worker.alive = False
    controller.close()
    assert controller._bundle is None
    assert loop.close_calls == 1
    assert loop.manifest_calls == 1
