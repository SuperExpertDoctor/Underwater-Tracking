from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.runtime.run_controller import RunController


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
