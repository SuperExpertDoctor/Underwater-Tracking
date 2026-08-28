"""Explicit opt-in integration entry point for the default live acceptance."""

from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import run_default_live_acceptance as driver


pytestmark = pytest.mark.live_acceptance


@pytest.mark.skipif(
    os.environ.get("UNDERWATER_TRACKING_RUN_LIVE_ACCEPTANCE") != "1"
    or os.environ.get("UNDERWATER_TRACKING_RUN_REAL_LLM") != "1",
    reason=(
        "set UNDERWATER_TRACKING_RUN_LIVE_ACCEPTANCE=1 and "
        "UNDERWATER_TRACKING_RUN_REAL_LLM=1 for the owned-process acceptance"
    ),
)
def test_default_main_live_acceptance() -> None:
    """Keep the expensive command discoverable without running it by default."""

    root = Path(__file__).resolve().parents[2]
    output = root / "outputs" / "acceptance-pytest.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/run_default_live_acceptance.py",
            "--config",
            "configs/scenario/uuv_only_single_target.yaml",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
        cwd=root,
        check=False,
    )
    assert result.returncode == 0


def test_strict_acceptance_artifact_bundle_has_the_run_local_schema(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-20260829T000000Z"
    ui_bundle = tmp_path / "ui-dist"
    ui_bundle.mkdir()
    (ui_bundle / "index.html").write_text("<html></html>\n", encoding="utf-8")
    operational_frames = run_dir / "operational_frames.jsonl"
    operational_frames.parent.mkdir(parents=True)
    operational_frames.write_text('{"frame_id": 1}\n', encoding="utf-8")

    screenshot_paths: dict[str, Path] = {}
    for viewport in ("desktop", "mobile"):
        for checkpoint in driver.CHECKPOINTS_S:
            screenshot = run_dir / "acceptance" / "screenshots" / f"{viewport}-{checkpoint}.png"
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            screenshot.write_bytes(b"test-png")
            screenshot_paths[f"{viewport}-{checkpoint}"] = screenshot

    driver.write_acceptance_artifacts(
        run_dir=run_dir,
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=20260828,
        ui_bundle_path=ui_bundle,
        operational_frames_path=operational_frames,
        checkpoint_records=[
            {
                "checkpoint_s": checkpoint,
                "frame": {"frame_id": checkpoint, "sim_time_s": checkpoint},
                "prediction": [],
                "execution": {},
                "regions": [],
                "groups": [],
                "execution_uuv_ids": [],
                "uuvs": [],
                "map_bounds": {},
                "detection": [],
                "event_ids": [],
                "transport_hashes": {},
                "database": {},
            }
            for checkpoint in driver.CHECKPOINTS_S
        ],
        metrics={
            "schema_version": "live-acceptance.metrics.v1",
            "checkpoints": list(driver.CHECKPOINTS_S),
            **{key: 0 for key in driver._CORE_METRIC_KEYS},
        },
        screenshot_paths=screenshot_paths,
        browser_console_records=[],
        backend_error_records=[],
    )

    acceptance_dir = run_dir / "acceptance"
    required_files = [
        acceptance_dir / "manifest.json",
        acceptance_dir / "metrics.json",
        acceptance_dir / "frame-checkpoints.jsonl",
        acceptance_dir / "browser-console.jsonl",
        acceptance_dir / "backend-errors.jsonl",
    ]
    required_files.extend(
        acceptance_dir / "screenshots" / f"{viewport}-{checkpoint}.png"
        for viewport in ("desktop", "mobile")
        for checkpoint in driver.CHECKPOINTS_S
    )
    assert all(path.is_file() for path in required_files)

    manifest = json.loads((acceptance_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["entrypoint"] == "main.py"
    assert manifest["mock_routes"] == []
    assert manifest["fake_websockets"] is False
    assert manifest["viewports"] == [[1600, 1000], [390, 844]]
    assert manifest["checkpoints_s"] == [600, 1800, 3600, 7200, 14400, 21600, 28800]
    assert manifest["ui_bundle_sha256"]
    assert manifest["operational_frames_sha256"]
    metrics = json.loads((acceptance_dir / "metrics.json").read_text(encoding="utf-8"))
    assert all(key in metrics for key in driver._CORE_METRIC_KEYS)
