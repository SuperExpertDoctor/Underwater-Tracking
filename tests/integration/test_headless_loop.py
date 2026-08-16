# tests/integration/test_headless_loop.py
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, cast

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
    engine = SimulationEngine(load_app_config("configs/scenario/default.yaml"), seed=42,
                              output_dir=tmp_path)
    frames = [engine.step() for _ in range(36)]
    assert frames[-1]["sim_time_s"] == 360
    assert len(cast(list[object], frames[-1]["uuvs"])) == 12
    assert isinstance(frames[0]["carrier"], dict)
    assert all(uuv["deployment_state"] == "deployed" for uuv in frames[0]["uuvs"])
    assert "target_truth" not in frames[-1]
    assert frames[-1]["group_reports"]


def test_engine_exposes_sink_truth_only_through_callback(tmp_path: Path) -> None:
    truth: list[dict[str, object]] = []

    def sink(truth_frame: dict[str, object]) -> None:
        truth.append(truth_frame)

    engine = SimulationEngine(
        load_app_config("configs/scenario/default.yaml"), seed=42, output_dir=tmp_path,
        evaluation_sink=sink,
    )
    engine.step()
    assert truth
    assert "targets" in truth[-1]
    assert truth[-1]["sim_time_s"] == 10


def test_engine_carrier_callback_receives_snapshot_with_carrier(tmp_path: Path) -> None:
    snapshots: list[SituationSnapshot] = []
    engine = SimulationEngine(
        load_app_config("configs/scenario/default.yaml"),
        seed=42,
        output_dir=tmp_path,
        carrier=snapshots.append,
    )

    frames = [engine.step() for _ in range(3)]

    assert len(snapshots) == 1
    assert snapshots[0].carrier is not None
    assert snapshots[0].carrier.model_dump() == frames[-1]["carrier"]


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
    config_path = str(Path(__file__).parents[2] / "configs/scenario/default.yaml")
    exit_code = main(["simulate", "--config", config_path, "--steps", "12", "--seed", "42"])
    assert exit_code == 0
    runs = list((tmp_path / "outputs").glob("run-*"))
    assert len(runs) == 1
    log_path = runs[0] / "frames.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12
