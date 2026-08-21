# tests/integration/test_headless_loop.py
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, cast

import numpy as np
import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AppConfig
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.persistence.frame_log import FrameLogger
from underwater_tracking.simulation.engine import SimulationEngine


class _FlakyFlushDelegate:
    """Minimal file-like that fails the first flush, then delegates.

    Simulates the transient ``PermissionError`` that surfaces at ``flush``
    on a shared-volume writer: the line is already buffered by the first
    ``write``, and the failure happens when the buffer is pushed to the OS.
    """

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.flush_attempts = 0

    def write(self, text: str) -> int:
        return int(self._handle.write(text))

    def flush(self) -> None:
        self.flush_attempts += 1
        if self.flush_attempts == 1:
            raise PermissionError("simulated transient flush failure")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def test_default_engine_runs_multirate_loop_without_truth_leak(tmp_path: Path) -> None:
    config = load_app_config("configs/scenario/default.yaml")
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    frames = [engine.step() for _ in range(36)]
    assert frames[-1]["sim_time_s"] == 36 * config.timing.physics_step_s
    assert len(cast(list[object], frames[-1]["uuvs"])) == 12
    assert isinstance(frames[0]["carrier"], dict)
    assert all(uuv["deployment_state"] == "deployed" for uuv in frames[0]["uuvs"])
    assert "target_truth" not in frames[-1]
    assert frames[-1]["group_reports"]


def test_engine_exposes_sink_truth_only_through_callback(tmp_path: Path) -> None:
    truth: list[dict[str, object]] = []

    def sink(truth_frame: dict[str, object]) -> None:
        truth.append(truth_frame)

    config = load_app_config("configs/scenario/default.yaml")
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path, evaluation_sink=sink)
    engine.step()
    assert truth
    assert "targets" in truth[-1]
    assert truth[-1]["sim_time_s"] == config.timing.physics_step_s


def test_engine_carrier_callback_receives_snapshot_with_carrier(tmp_path: Path) -> None:
    snapshots: list[SituationSnapshot] = []
    engine = SimulationEngine(
        load_app_config("configs/scenario/default.yaml"),
        seed=42,
        output_dir=tmp_path,
        carrier=snapshots.append,
    )

    config = engine._config
    frames = [
        engine.step()
        for _ in range(config.timing.observation_step_s // config.timing.physics_step_s)
    ]

    assert len(snapshots) == 1
    assert snapshots[0].carrier is not None
    assert snapshots[0].carrier.model_dump() == frames[-1]["carrier"]


def test_lifecycle_frames_and_jsonl_logs_serialize_carrier_relationship_lists(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config("configs/scenario/default.yaml"), seed=42, output_dir=tmp_path)
    uuv_id = "uuv_00"
    engine.request_uuv_recovery(uuv_id, reason="integration")
    engine._uuvs[uuv_id].position_xy = (-2950.0, -3000.0)

    recovered_frame = engine.step()
    engine.request_uuv_deployment(uuv_id, reason="integration")
    deployed_frame = engine.step()
    logged_frames = [json.loads(line) for line in engine.logger.path.read_text(encoding="utf-8").splitlines()]

    assert all(isinstance(recovered_frame["carrier"][key], tuple) for key in (
        "onboard_uuv_ids", "deployed_uuv_ids", "returning_uuv_ids"
    ))
    assert uuv_id in deployed_frame["carrier"]["deployed_uuv_ids"]
    assert uuv_id in logged_frames[0]["carrier"]["onboard_uuv_ids"]
    assert uuv_id in logged_frames[1]["carrier"]["deployed_uuv_ids"]
    assert all(isinstance(logged_frames[0]["carrier"][key], list) for key in (
        "onboard_uuv_ids", "deployed_uuv_ids", "returning_uuv_ids"
    ))


def _run_log(config: AppConfig, seed: int, output_dir: Path) -> str:
    engine = SimulationEngine(config, seed=seed, output_dir=output_dir)
    for _ in range(360):
        engine.step()
    return engine.logger.path.read_text(encoding="utf-8")


def _normalize(text: str, output_root: Path) -> str:
    text = re.sub(r'"run_id":\s*"[^"]*"', '"run_id": "RUN"', text)
    text = text.replace(str(output_root), "<OUTPUT>")
    return text


def test_same_seed_logs_are_byte_identical_and_other_seed_differs(tmp_path: Path) -> None:
    """Step 5: same seed twice into separate dirs -> matching SHA-256 after
    normalizing run ids and output paths; seed 43 differs."""
    config = load_app_config("configs/scenario/default.yaml")
    first = _normalize(_run_log(config, seed=42, output_dir=tmp_path / "run42-a"), tmp_path)
    second = _normalize(_run_log(config, seed=42, output_dir=tmp_path / "run42-b"), tmp_path)
    other = _normalize(_run_log(config, seed=43, output_dir=tmp_path / "run43"), tmp_path)
    assert first == second
    first_hash = sha256(first.encode("utf-8")).hexdigest()
    assert first_hash == sha256(second.encode("utf-8")).hexdigest()
    assert first_hash != sha256(other.encode("utf-8")).hexdigest()
    assert first.count("\n") == 360


def test_seed_43_long_run_keeps_filter_covariances_finite_psd(tmp_path: Path) -> None:
    config = load_app_config("configs/scenario/default.yaml")
    engine = SimulationEngine(config, seed=43, output_dir=tmp_path / "run43")
    for _ in range(360):
        engine.step()

    for thread_id in engine._manager._threads.values():
        state = engine._manager._graph.get_state(
            {"configurable": {"thread_id": thread_id}}
        )
        snapshot = state.values["filter_snapshot"]
        for model_state in snapshot.filters.values():
            covariance = np.asarray(model_state.covariance, dtype=float)
            assert np.all(np.isfinite(covariance))
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-10)
            assert float(np.min(np.linalg.eigvalsh(covariance))) >= -1e-10

        belief_covariance = np.asarray(state.values["belief"].covariance, dtype=float)
        assert np.all(np.isfinite(belief_covariance))
        np.testing.assert_allclose(belief_covariance, belief_covariance.T, atol=1e-10)
        assert float(np.min(np.linalg.eigvalsh(belief_covariance))) >= -1e-10


def test_frame_logger_retries_transient_flush_without_duplication(tmp_path: Path) -> None:
    """A transient PermissionError at flush must not append the frame twice.

    The retry loop flushes the already-buffered line again instead of
    re-writing it, so a recovered flush leaves exactly one copy of each
    frame in the file while ``count`` advances once per frame.
    """
    logger = FrameLogger(tmp_path)
    flaky = _FlakyFlushDelegate(logger._handle)
    logger._handle = cast(Any, flaky)
    frame = {"sim_time_s": 10, "uuvs": []}
    for _ in range(3):
        logger.write(frame)
    assert flaky.flush_attempts == 4  # one failure + one retry, then 2 clean flushes
    assert logger.count == 3
    lines = logger.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert all(json.loads(line) == frame for line in lines)


def test_cli_simulate_writes_nonempty_jsonl_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from underwater_tracking.cli import main

    monkeypatch.chdir(tmp_path)
    config_path = str(
        Path(__file__).parents[2] / "configs/scenario/uuv_only_single_target.yaml"
    )
    exit_code = main(["simulate", "--config", config_path, "--steps", "12", "--seed", "42"])
    assert exit_code == 0
    runs = list((tmp_path / "outputs").glob("run-*"))
    assert len(runs) == 1
    log_path = runs[0] / "frames.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12
