"""Explicit opt-in integration entry point for the default live acceptance."""

from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import time
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


def test_run_live_acceptance_writes_artifacts_when_config_loading_fails(tmp_path: Path) -> None:
    output_path = tmp_path / "early-failure.json"
    missing_config = tmp_path / "missing-scenario.yaml"

    result = driver.run_live_acceptance(
        config_path=missing_config,
        seed=42,
        api_port=0,
        output_path=output_path,
    )

    assert result == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    run_dir = Path(report["run_dir"])
    acceptance_dir = run_dir / "acceptance"
    assert run_dir.parent == tmp_path / "early-failure-owned-runs"
    assert report["status"] == "failed"
    assert report["failure"]
    assert report["acceptance_dir"] == str(acceptance_dir)
    assert (acceptance_dir / "manifest.json").is_file()
    assert (acceptance_dir / "metrics.json").is_file()
    assert (acceptance_dir / "frame-checkpoints.jsonl").is_file()
    assert (acceptance_dir / "screenshots").is_dir()
    assert (acceptance_dir / "browser-console.jsonl").is_file()
    assert (acceptance_dir / "backend-errors.jsonl").is_file()
    manifest_text = (acceptance_dir / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["status"] == "failed"
    assert manifest["termination"]["status"] == "failed"
    assert manifest["termination"]["reason"] == manifest["failure"]
    assert set(manifest["provenance"]) == {
        "code_revision",
        "config_sha256",
        "scenario_id",
        "provider_identity",
        "started_at",
        "ended_at",
    }
    assert "sk-" not in manifest_text
    backend_errors = [
        json.loads(line)
        for line in (acceptance_dir / "backend-errors.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(record["error"] == "acceptance_failure" for record in backend_errors)


def test_windows_owned_process_cleanup_uses_process_group_and_taskkill_tree(monkeypatch) -> None:
    class FakeProcess:
        pid = 321

        def __init__(self) -> None:
            self.returncode = None
            self.killed = False

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 1
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = 1

    process = FakeProcess()
    popen_kwargs = {}
    taskkill_calls = []

    def fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    def fake_run(command, **kwargs):
        taskkill_calls.append((tuple(command), kwargs))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(driver.os, "name", "nt")
    monkeypatch.setattr(driver.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(driver.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(driver.subprocess, "run", fake_run)

    spawned = driver._spawn_owned_process(("main.py",))
    driver._terminate_validated_group(spawned, {})

    assert popen_kwargs["creationflags"] & 0x200
    assert taskkill_calls[0][0] == ("taskkill", "/PID", "321", "/T", "/F")
    assert process.killed is False


def test_windows_cleanup_kills_tree_when_wrapper_already_exited(monkeypatch) -> None:
    class ExitedWrapper:
        pid = 654

        def __init__(self) -> None:
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def send_signal(self, signal):
            raise AssertionError("an exited wrapper must not receive another signal")

    process = ExitedWrapper()
    taskkill_calls = []

    def fake_run(command, **kwargs):
        taskkill_calls.append((tuple(command), kwargs))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(driver.os, "name", "nt")
    monkeypatch.setattr(driver.subprocess, "run", fake_run)

    shutdown = {"sigint_sent": False, "sigint_count": 0}
    driver._shutdown_owned_process(process, shutdown)

    assert taskkill_calls[0][0] == ("taskkill", "/PID", "654", "/T", "/F")
    assert shutdown["tree_cleanup_command"] == ["taskkill", "/PID", "654", "/T", "/F"]
    assert shutdown["tree_cleanup_returncode"] == 0


def test_windows_cleanup_records_exited_pid_gone_as_benign(monkeypatch) -> None:
    class ExitedWrapper:
        pid = 656

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_run(command, **kwargs):
        return type("Completed", (), {"returncode": 128})()

    monkeypatch.setattr(driver.os, "name", "nt")
    monkeypatch.setattr(driver.subprocess, "run", fake_run)

    shutdown = {}
    driver._terminate_validated_group(ExitedWrapper(), shutdown)

    assert shutdown["tree_cleanup_returncode"] == 128
    assert shutdown["benign_pid_gone"] is True
    assert shutdown["tree_cleanup_status"] == "benign_pid_gone"


def test_windows_cleanup_surfaces_taskkill_failure(monkeypatch) -> None:
    class ExitedWrapper:
        pid = 655

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_run(command, **kwargs):
        return type("Completed", (), {"returncode": 5})()

    monkeypatch.setattr(driver.os, "name", "nt")
    monkeypatch.setattr(driver.subprocess, "run", fake_run)

    shutdown = {}
    with pytest.raises(driver._AcceptanceFailure, match="taskkill failed"):
        driver._terminate_validated_group(ExitedWrapper(), shutdown)

    assert shutdown["tree_cleanup_command"] == ["taskkill", "/PID", "655", "/T", "/F"]
    assert shutdown["tree_cleanup_returncode"] == 5


def test_windows_cleanup_surfaces_nonzero_taskkill_for_live_wrapper(monkeypatch) -> None:
    class RunningWrapper:
        pid = 657

        def poll(self):
            return None

        def wait(self, timeout=None):
            return None

    def fake_run(command, **kwargs):
        return type("Completed", (), {"returncode": 128})()

    monkeypatch.setattr(driver.os, "name", "nt")
    monkeypatch.setattr(driver.subprocess, "run", fake_run)

    with pytest.raises(driver._AcceptanceFailure, match="taskkill failed"):
        driver._terminate_validated_group(RunningWrapper(), {})


@pytest.mark.skipif(os.name != "nt", reason="Windows taskkill tree integration test")
def test_windows_taskkill_terminates_real_descendant_before_wrapper_exits(tmp_path) -> None:
    child_script = tmp_path / "acceptance_child.py"
    wrapper_script = tmp_path / "acceptance_wrapper.py"
    pid_file = tmp_path / "child.pid"
    child_script.write_text("import time\ntime.sleep(300)\n", encoding="utf-8")
    wrapper_script.write_text(
        "\n".join(
            (
                "from pathlib import Path",
                "import subprocess",
                "import sys",
                "child = subprocess.Popen([sys.executable, sys.argv[1]])",
                "Path(sys.argv[2]).write_text(str(child.pid), encoding='ascii')",
                "raise SystemExit(child.wait())",
            ),
        )
        + "\n",
        encoding="utf-8",
    )

    def process_is_running(pid: int) -> bool:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)

    process = None
    child_pid = None
    try:
        process = driver._spawn_owned_process(
            (sys.executable, str(wrapper_script), str(child_script), str(pid_file)),
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("wrapper exited before its child became observable")
            if pid_file.exists():
                child_pid = int(pid_file.read_text(encoding="ascii").strip())
                if process_is_running(child_pid):
                    break
            time.sleep(0.05)
        if child_pid is None or not process_is_running(child_pid):
            raise AssertionError("real descendant did not become observable")
        assert process.poll() is None

        shutdown = {}
        driver._terminate_validated_group(process, shutdown)

        assert shutdown["tree_cleanup_command"] == [
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
        ]
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and process_is_running(child_pid):
            time.sleep(0.05)
        assert not process_is_running(child_pid)
        assert process.poll() is not None
    finally:
        cleanup_pids = []
        if process is not None:
            cleanup_pids.append(process.pid)
        if child_pid is not None:
            cleanup_pids.append(child_pid)
        for pid in cleanup_pids:
            if process_is_running(pid):
                subprocess.run(
                    ("taskkill", "/PID", str(pid), "/T", "/F"),
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )


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
        "readPaintedVisualLayers",
        "assertPaintedVisualLayerContract",
        "paintedArcPoints",
        "assertRenderedFrameBinding",
        "readRenderedFrameIdentity",
        "corridorPolygonPoints",
        "parseSvgPoints",
        "data-visible-bounds",
        "data-rendered-frame-id",
        "data-rendered-execution-revision",
        "data-rendered-prediction-id",
        "data-current-task-uuv-ids",
        "data-current-task-uuv-telemetry",
        "data-last-painted-visual-layers",
        "data-task-group-id",
        "getTotalLength",
        "rectanglesOverlap",
    ):
        assert helper in source
    assert "pixels.red" not in source
    assert "pixels.amber" not in source
    assert "pixels.cyan" not in source
    assert "candidates.length === 0) continue" not in source
    assert 'const field = mode === "active" ? "active_range_m" : "passive_range_m";' in source
    assert "return finiteNumber(preferred)" not in source
    for forbidden in ("page.route", "route.fulfill", "addInitScript", "new WebSocket"):
        assert forbidden not in source
