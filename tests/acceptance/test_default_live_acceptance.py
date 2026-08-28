"""Explicit opt-in integration entry point for the default live acceptance."""

from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import tomllib

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
    config_path = tmp_path / "scenario.yaml"
    config_path.write_text("scenario:\n  scenario_id: test-scenario\n", encoding="utf-8")
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
        config_path=config_path,
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
        scenario_id="test-scenario",
        provider_identity="unconfigured",
        code_revision="test-revision",
        started_at="2026-08-29T00:00:00+00:00",
        ended_at="2026-08-29T00:01:00+00:00",
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
    assert manifest["status"] == "passed"
    assert manifest["provenance"] == {
        "code_revision": "test-revision",
        "config_sha256": manifest["provenance"]["config_sha256"],
        "scenario_id": "test-scenario",
        "provider_identity": "unconfigured",
        "started_at": "2026-08-29T00:00:00+00:00",
        "ended_at": "2026-08-29T00:01:00+00:00",
    }
    assert manifest["termination"] == {"status": "passed", "reason": None}
    metrics = json.loads((acceptance_dir / "metrics.json").read_text(encoding="utf-8"))
    assert all(key in metrics for key in driver._CORE_METRIC_KEYS)


def test_failed_acceptance_writes_complete_partial_artifact_tree(tmp_path: Path) -> None:
    run_dir = tmp_path / "failed-run"
    config_path = tmp_path / "scenario.yaml"
    config_path.write_text("scenario:\n  scenario_id: failed-scenario\n", encoding="utf-8")
    ui_bundle = tmp_path / "ui-dist"
    ui_bundle.mkdir()
    (ui_bundle / "index.html").write_text("<html></html>\n", encoding="utf-8")
    operational_frames = run_dir / "operational_frames.jsonl"
    operational_frames.parent.mkdir(parents=True)
    operational_frames.write_text('{"frame_id": 600}\n', encoding="utf-8")
    screenshot = tmp_path / "desktop-600.png"
    screenshot.write_bytes(b"partial-png")
    metrics = {
        "schema_version": "live-acceptance.metrics.v1",
        "checkpoints": list(driver.CHECKPOINTS_S),
        **{key: 0 for key in driver._CORE_METRIC_KEYS},
    }
    checkpoint = {
        "checkpoint_s": 600,
        "frame": {"frame_id": 600, "sim_time_s": 600},
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

    driver.write_acceptance_artifacts(
        run_dir=run_dir,
        config_path=config_path,
        seed=42,
        ui_bundle_path=ui_bundle,
        operational_frames_path=operational_frames,
        checkpoint_records=[checkpoint],
        metrics=metrics,
        screenshot_paths={"desktop-600": screenshot},
        browser_console_records=[{"type": "error", "message": "browser failed"}],
        backend_error_records=[{"error": "checkpoint failed", "checkpoint_s": 600}],
        status="failed",
        failure="checkpoint 600s failed",
        scenario_id="failed-scenario",
        provider_identity="provider:model@api.example.test",
        code_revision="failed-revision",
        started_at="2026-08-29T00:00:00+00:00",
        ended_at="2026-08-29T00:00:05+00:00",
    )

    acceptance_dir = run_dir / "acceptance"
    assert (acceptance_dir / "manifest.json").is_file()
    assert (acceptance_dir / "metrics.json").is_file()
    assert (acceptance_dir / "frame-checkpoints.jsonl").is_file()
    assert (acceptance_dir / "screenshots").is_dir()
    assert (acceptance_dir / "screenshots" / "desktop-600.png").is_file()
    assert (acceptance_dir / "browser-console.jsonl").is_file()
    assert (acceptance_dir / "backend-errors.jsonl").is_file()
    records = [
        json.loads(line)
        for line in (acceptance_dir / "frame-checkpoints.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records == [checkpoint]

    manifest_text = (acceptance_dir / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["status"] == "failed"
    assert manifest["failure"] == "checkpoint 600s failed"
    assert manifest["termination"] == {
        "status": "failed",
        "reason": "checkpoint 600s failed",
    }
    assert manifest["provenance"]["code_revision"] == "failed-revision"
    assert manifest["provenance"]["config_sha256"]
    assert manifest["provenance"]["scenario_id"] == "failed-scenario"
    assert manifest["provenance"]["provider_identity"] == "provider:model@api.example.test"
    assert "sk-" not in manifest_text
    metrics_payload = json.loads((acceptance_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["status"] == "failed"
    assert metrics_payload["failure"] == "checkpoint 600s failed"


def test_websockets_sync_client_has_a_runtime_dependency() -> None:
    metadata = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = metadata["project"]["dependencies"]
    assert any(str(dependency).lower().startswith("websockets") for dependency in dependencies)


def test_live_visual_spec_has_attributed_geometry_gates_without_injection() -> None:
    spec_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "underwater_tracking"
        / "ui"
        / "e2e"
        / "live-visualization-acceptance.spec.ts"
    )
    source = spec_path.read_text(encoding="utf-8")
    for helper in (
        "canvasProjection",
        "sampleCanvasColorNear",
        "assertDetectionGeometry",
        "assertSonarAttribution",
        "data-visible-bounds",
        "data-current-task-uuv-ids",
        "data-task-group-id",
        "getTotalLength",
        "rectanglesOverlap",
    ):
        assert helper in source
    assert "pixels.red" not in source
    assert "pixels.amber" not in source
    assert "pixels.cyan" not in source
    for forbidden in ("page.route", "route.fulfill", "addInitScript", "new WebSocket"):
        assert forbidden not in source
