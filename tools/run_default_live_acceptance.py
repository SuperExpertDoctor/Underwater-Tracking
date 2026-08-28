"""Run the strict owned-process acceptance for the default entry point.

The driver deliberately speaks only to the public HTTP surface.  It owns the
``main.py`` process group, walks replay with a bounded cursor, and writes a
machine-readable report without prompts, provider responses, or credentials.
The expensive path is opt-in; the module is also imported by offline driver
tests with a small fixture server.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from typing import cast

import httpx


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_HTTP_TIMEOUT_S = 2.0
_HEALTH_INTERVAL_S = 1.0
_MIN_HEALTH_SAMPLES = 60
_GLOBAL_DEADLINE_S = 1_800.0
_PROCESS_SHUTDOWN_TIMEOUT_S = 10.0
_REPLAY_PAGE_SIZE = 250
_MAX_REPLAY_PAGES_PER_POLL = 64
_MAX_REPLAY_FRAMES = 50_000
_POLL_INTERVAL_S = 0.5
CHECKPOINTS_S = (600, 1_800, 3_600, 7_200, 14_400, 21_600, 28_800)
VIEWPORTS = ((1_600, 1_000), (390, 844))

_CORE_METRIC_KEYS = (
    "target_boundary_recovery_count",
    "target_boundary_recovery_max_duration_s",
    "prediction_valid_fraction",
    "prediction_degraded_fraction",
    "prediction_unavailable_fraction",
    "prediction_max_radius_m",
    "prediction_max_clipped_fraction",
    "execution_max_data_age_s",
    "deterministic_baseline_commit_count",
    "llm_optimization_commit_count",
    "region_generation_failure_count",
    "expired_execution_frame_count",
    "frame_channel_mismatch_count",
    "browser_console_error_count",
    "required_layer_missing_count",
)

_CHECKPOINT_EVIDENCE_KEYS = (
    "checkpoint_s",
    "frame",
    "prediction",
    "execution",
    "regions",
    "groups",
    "execution_uuv_ids",
    "uuvs",
    "map_bounds",
    "detection",
    "event_ids",
    "transport_hashes",
    "database",
)


Predicate = Callable[[dict[str, object]], bool]
QueryScalar = str | int | float | bool | None
QueryParams = Mapping[str, QueryScalar | Sequence[QueryScalar]]


@dataclass(frozen=True, slots=True)
class AcceptanceCheckpoint:
    """One ordered semantic condition in the acceptance run."""

    name: str
    event_type: str | None
    predicate: Predicate
    timeout_s: float


class _AcceptanceFailure(RuntimeError):
    """A bounded acceptance failure with a user-facing diagnostic."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    files = sorted(
        path for path in root.rglob("*") if path.is_file()
    ) if root.is_dir() else [root]
    if not files:
        raise _AcceptanceFailure(f"UI bundle is empty: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix() if root.is_dir() else path.name
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    dict(record),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def write_acceptance_artifacts(
    *,
    run_dir: Path,
    config_path: Path,
    seed: int,
    ui_bundle_path: Path,
    operational_frames_path: Path,
    checkpoint_records: Sequence[Mapping[str, object]],
    metrics: Mapping[str, object],
    screenshot_paths: Mapping[str, Path],
    browser_console_records: Sequence[Mapping[str, object]],
    backend_error_records: Sequence[Mapping[str, object]],
) -> Path:
    """Write the immutable run-local artifact contract after all gates pass."""
    run_dir = run_dir.resolve()
    acceptance_dir = run_dir / "acceptance"
    screenshots_dir = acceptance_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    if not ui_bundle_path.exists():
        raise _AcceptanceFailure(f"UI bundle does not exist: {ui_bundle_path}")
    if not operational_frames_path.is_file() or operational_frames_path.stat().st_size == 0:
        raise _AcceptanceFailure("owned operational frame log is missing or empty")

    checkpoint_values = tuple(
        record.get("checkpoint_s")
        for record in checkpoint_records
    )
    if checkpoint_values != CHECKPOINTS_S:
        raise _AcceptanceFailure(
            f"checkpoint evidence must cover {CHECKPOINTS_S}, got {checkpoint_values}"
        )
    for index, record in enumerate(checkpoint_records):
        missing = [key for key in _CHECKPOINT_EVIDENCE_KEYS if key not in record]
        if missing:
            raise _AcceptanceFailure(
                f"checkpoint evidence {index} is missing fields: {', '.join(missing)}"
            )
    missing_metrics = [key for key in _CORE_METRIC_KEYS if key not in metrics]
    if missing_metrics:
        raise _AcceptanceFailure(
            "metrics.json is missing core fields: " + ", ".join(missing_metrics)
        )
    expected_screenshots = {
        f"{viewport}-{checkpoint}"
        for viewport in ("desktop", "mobile")
        for checkpoint in CHECKPOINTS_S
    }
    if set(screenshot_paths) != expected_screenshots:
        missing = sorted(expected_screenshots - set(screenshot_paths))
        extra = sorted(set(screenshot_paths) - expected_screenshots)
        raise _AcceptanceFailure(
            f"screenshot set mismatch; missing={missing}, extra={extra}"
        )
    for key, source in screenshot_paths.items():
        source = Path(source)
        if not source.is_file() or source.stat().st_size == 0:
            raise _AcceptanceFailure(f"screenshot is missing or empty: {source}")
        destination = screenshots_dir / f"{key}.png"
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)

    frame_checkpoints_path = acceptance_dir / "frame-checkpoints.jsonl"
    metrics_path = acceptance_dir / "metrics.json"
    browser_console_path = acceptance_dir / "browser-console.jsonl"
    backend_errors_path = acceptance_dir / "backend-errors.jsonl"
    _write_jsonl(frame_checkpoints_path, checkpoint_records)
    _write_jsonl(browser_console_path, browser_console_records)
    _write_jsonl(backend_errors_path, backend_error_records)
    metrics_path.write_text(
        json.dumps(
            dict(metrics),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    resolved_config = config_path if config_path.is_absolute() else _REPOSITORY_ROOT / config_path
    manifest = {
        "schema_version": "live-acceptance.v1",
        "entrypoint": "main.py",
        "config": str(resolved_config.resolve()),
        "seed": seed,
        "mock_routes": [],
        "fake_websockets": False,
        "viewports": [list(viewport) for viewport in VIEWPORTS],
        "checkpoints_s": list(CHECKPOINTS_S),
        "ui_bundle_sha256": _sha256_tree(ui_bundle_path),
        "operational_frames_sha256": _sha256_file(operational_frames_path),
        "artifacts": {
            "operational_frames": str(operational_frames_path.relative_to(run_dir)),
            "metrics": str(metrics_path.relative_to(run_dir)),
            "frame_checkpoints": str(frame_checkpoints_path.relative_to(run_dir)),
            "screenshots": str(screenshots_dir.relative_to(run_dir)),
            "browser_console": str(browser_console_path.relative_to(run_dir)),
            "backend_errors": str(backend_errors_path.relative_to(run_dir)),
        },
    }
    (acceptance_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return acceptance_dir


class _LiveWebSocketFrameCapture:
    """Capture the owned server's serialized operational frames in one thread."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frames: dict[int, dict[str, object]] = {}
        self.error: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="live-acceptance-websocket",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def frame(self, frame_id: int) -> dict[str, object] | None:
        with self._lock:
            frame = self._frames.get(frame_id)
            return dict(frame) if frame is not None else None

    def _run(self) -> None:
        try:
            from websockets.sync.client import connect

            with connect(self._url, open_timeout=5.0, close_timeout=2.0) as websocket:
                while not self._stop.is_set():
                    try:
                        raw = websocket.recv(timeout=1.0)
                    except TimeoutError:
                        continue
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    if not isinstance(raw, str):
                        continue
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        continue
                    frame_id = payload.get("frame_id")
                    if not isinstance(frame_id, int) or isinstance(frame_id, bool):
                        continue
                    with self._lock:
                        self._frames[frame_id] = cast(dict[str, object], payload)
        except Exception as exc:  # noqa: BLE001 - reported by the strict backend gate
            if not self._stop.is_set():
                self.error = f"{type(exc).__name__}: {exc}"


def _append_jsonl_record(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                dict(record),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(cast(dict[str, object], payload))
    return records


def _backend_error(
    acceptance_dir: Path,
    errors: list[dict[str, object]],
    error: str,
    *,
    checkpoint_s: int | None = None,
    frame: Mapping[str, object] | None = None,
    details: Mapping[str, object] | None = None,
) -> None:
    record: dict[str, object] = {
        "timestamp_utc": _timestamp(),
        "error": error,
    }
    if checkpoint_s is not None:
        record["checkpoint_s"] = checkpoint_s
    if isinstance(frame, Mapping):
        record["frame_id"] = frame.get("frame_id")
        record["sim_time_s"] = frame.get("sim_time_s")
        execution = frame.get("execution")
        if isinstance(execution, Mapping):
            record["execution_revision"] = execution.get("execution_revision")
    if details:
        record["details"] = dict(details)
    errors.append(record)
    _append_jsonl_record(acceptance_dir / "backend-errors.jsonl", record)


def _frame_from_jsonl(path: Path, frame_id: int) -> dict[str, object] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("frame_id") == frame_id:
                return cast(dict[str, object], payload)
    return None


def _wait_for_jsonl_frame(
    path: Path,
    frame_id: int,
    deadline: float,
) -> dict[str, object] | None:
    """Wait for the owned writer to finish persisting the selected frame line."""
    while time.monotonic() < deadline:
        frame = _frame_from_jsonl(path, frame_id)
        if frame is not None:
            return frame
        time.sleep(0.05)
    return None


def _effective_prediction_cap(frame: Mapping[str, object], config: object) -> float:
    tracking = getattr(config, "tracking", None)
    health = getattr(tracking, "prediction_health", None)
    configured_cap = float(getattr(health, "max_corridor_radius_m", float("nan")))
    fraction = float(getattr(health, "max_corridor_map_fraction", float("nan")))
    bounds = frame.get("map_bounds")
    if not isinstance(bounds, Mapping):
        return configured_cap
    values = [bounds.get(key) for key in ("min_x", "min_y", "max_x", "max_y")]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return configured_cap
    min_x, min_y, max_x, max_y = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (min_x, min_y, max_x, max_y)):
        return configured_cap
    map_fraction_cap = min(max_x - min_x, max_y - min_y) * fraction
    return min(configured_cap, map_fraction_cap)


def _wait_for_owned_run(
    client: httpx.Client,
    process: subprocess.Popen[bytes],
    output_root: Path,
    deadline: float,
) -> tuple[str, Path]:
    """Wait until the one owned controller has published its catalog run."""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise _AcceptanceFailure(
                f"owned main.py exited before run publication: {process.returncode}"
            )
        try:
            status, payload = _request_json(
                client,
                "GET",
                "/api/replay",
                params={"limit": 1},
            )
        except Exception:
            time.sleep(_POLL_INTERVAL_S)
            continue
        run_id = payload.get("run_id")
        if status == 200 and isinstance(run_id, str) and run_id:
            run_path = (output_root / run_id).resolve()
            if (
                Path(run_id).name == run_id
                and run_path.parent == output_root.resolve()
                and run_path.is_dir()
                and (run_path / "agent.db").is_file()
            ):
                return run_id, run_path
        time.sleep(_POLL_INTERVAL_S)
    raise _AcceptanceFailure("owned main.py did not publish a catalog run before the deadline")


def _wait_for_checkpoint_frame(
    client: httpx.Client,
    process: subprocess.Popen[bytes],
    checkpoint_s: int,
    deadline: float,
    *,
    previous_frame_id: int | None,
) -> dict[str, object]:
    last_frame: dict[str, object] = {}
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise _AcceptanceFailure(
                f"owned main.py exited before checkpoint {checkpoint_s}: {process.returncode}"
            )
        try:
            status, frame = _request_json(client, "GET", "/api/operational/snapshot")
        except Exception:
            time.sleep(_POLL_INTERVAL_S)
            continue
        if status == 200:
            last_frame = frame
            sim_time = frame.get("sim_time_s")
            frame_id = frame.get("frame_id")
            if (
                isinstance(sim_time, (int, float))
                and not isinstance(sim_time, bool)
                and math.isfinite(float(sim_time))
                and float(sim_time) >= checkpoint_s
                and isinstance(frame_id, int)
                and not isinstance(frame_id, bool)
                and (previous_frame_id is None or frame_id > previous_frame_id)
            ):
                return frame
        time.sleep(_POLL_INTERVAL_S)
    raise _AcceptanceFailure(
        f"checkpoint {checkpoint_s} timed out; latest_frame={last_frame.get('frame_id')!r}; "
        f"latest_sim_time_s={last_frame.get('sim_time_s')!r}"
    )


def _checkpoint_evidence(
    checkpoint_s: int,
    frame: Mapping[str, object],
    transport_hashes: Mapping[str, str],
    database: Mapping[str, object],
) -> dict[str, object]:
    execution = frame.get("execution")
    execution_object = execution if isinstance(execution, Mapping) else {}
    estimates = frame.get("target_estimates")
    estimate_objects = [item for item in estimates if isinstance(item, Mapping)] if isinstance(estimates, list) else []
    predictions: list[dict[str, object]] = []
    detections: list[dict[str, object]] = []
    for estimate in estimate_objects:
        prediction = estimate.get("prediction")
        if isinstance(prediction, Mapping):
            health = prediction.get("health")
            predictions.append(
                {
                    "target_id": estimate.get("target_id"),
                    "prediction_id": prediction.get("prediction_id"),
                    "prediction_revision": prediction.get("prediction_revision"),
                    "health": dict(health) if isinstance(health, Mapping) else health,
                    "maximum_radius_m": (
                        health.get("maximum_radius_m")
                        if isinstance(health, Mapping)
                        else None
                    ),
                    "centerline_count": len(prediction.get("centerline_xy", ()))
                    if isinstance(prediction.get("centerline_xy"), (list, tuple))
                    else 0,
                }
            )
        else:
            predictions.append(
                {
                    "target_id": estimate.get("target_id"),
                    "prediction_id": None,
                    "prediction_revision": None,
                    "health": {"status": "unavailable"},
                    "maximum_radius_m": None,
                    "centerline_count": 0,
                }
            )
        detections.append(
            {
                "target_id": estimate.get("target_id"),
                "mean": estimate.get("mean"),
                "detection_range_m": estimate.get("detection_range_m"),
            }
        )
    regions = execution_object.get("regions")
    region_objects = [item for item in regions if isinstance(item, Mapping)] if isinstance(regions, list) else []
    groups = execution_object.get("task_groups")
    group_objects = [item for item in groups if isinstance(item, Mapping)] if isinstance(groups, list) else []
    uuvs = frame.get("uuvs")
    uuv_objects = [item for item in uuvs if isinstance(item, Mapping)] if isinstance(uuvs, list) else []
    member_ids = [
        member
        for group in group_objects
        for member in group.get("member_uuv_ids", ())
        if isinstance(member, str)
    ]
    return {
        "checkpoint_s": checkpoint_s,
        "frame": {
            "frame_id": frame.get("frame_id"),
            "sim_time_s": frame.get("sim_time_s"),
            "plan_version": frame.get("plan_version"),
            "run_phase": frame.get("run_phase"),
            "schema_version": frame.get("schema_version"),
        },
        "prediction": predictions,
        "execution": {
            "execution_revision": execution_object.get("execution_revision"),
            "source_snapshot_revision": execution_object.get("source_snapshot_revision"),
            "prediction_revision": execution_object.get("prediction_revision"),
            "prediction_id": execution_object.get("prediction_id"),
            "health_status": execution_object.get("health_status"),
            "data_age_s": execution_object.get("data_age_s"),
            "valid_from_s": execution_object.get("valid_from_s"),
            "valid_until_s": execution_object.get("valid_until_s"),
            "plan_source": execution_object.get("plan_source"),
        },
        "regions": [
            {
                "region_id": region.get("region_id"),
                "target_id": region.get("target_id"),
                "status": region.get("status"),
                "geometry": region.get("geometry"),
                "task_group_id": region.get("task_group_id"),
                "execution_revision": region.get("execution_revision"),
            }
            for region in region_objects
        ],
        "groups": [
            {
                "task_group_id": group.get("task_group_id"),
                "region_id": group.get("region_id"),
                "member_uuv_ids": group.get("member_uuv_ids"),
                "status": group.get("status"),
                "execution_revision": group.get("execution_revision"),
            }
            for group in group_objects
        ],
        "execution_uuv_ids": sorted(set(member_ids)),
        "uuvs": [
            {
                "uuv_id": uuv.get("uuv_id"),
                "position": uuv.get("position"),
                "status": uuv.get("status"),
                "sensor_mode": uuv.get("sensor_mode"),
                "physically_exposed": uuv.get("physically_exposed"),
            }
            for uuv in uuv_objects
            if uuv.get("uuv_id") in set(member_ids)
        ],
        "map_bounds": frame.get("map_bounds"),
        "detection": detections,
        "event_ids": [
            event.get("event_id")
            for key in ("events", "mission_events")
            for event in frame.get(key, ())
            if isinstance(event, Mapping) and event.get("event_id")
        ],
        "transport_hashes": dict(transport_hashes),
        "database": dict(database),
    }


def _frame_event_mappings(frame: Mapping[str, object]) -> list[Mapping[str, object]]:
    events: list[Mapping[str, object]] = []
    for key in ("events", "mission_events"):
        raw_events = frame.get(key)
        if isinstance(raw_events, (list, tuple)):
            events.extend(event for event in raw_events if isinstance(event, Mapping))
    return events


def _finite_metric_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_event_key(event: Mapping[str, object], index: int) -> str:
    event_id = event.get("event_id")
    if isinstance(event_id, str) and event_id:
        return event_id
    return f"{event.get('event_type', '')}:{event.get('sim_time_s', '')}:{index}"


def _checkpoint_metric_observation(
    frame: Mapping[str, object],
    *,
    seen_boundary_events: set[str],
    seen_commit_keys: set[str],
    seen_region_failures: set[str],
) -> dict[str, object]:
    prediction_counts = {"valid": 0, "degraded": 0, "unavailable": 0}
    prediction_observations = 0
    prediction_max_radius = 0.0
    prediction_max_clipped = 0.0
    estimates = frame.get("target_estimates")
    if isinstance(estimates, (list, tuple)):
        for estimate in estimates:
            if not isinstance(estimate, Mapping):
                continue
            prediction_observations += 1
            prediction = estimate.get("prediction")
            health = prediction.get("health") if isinstance(prediction, Mapping) else None
            status = health.get("status") if isinstance(health, Mapping) else None
            if not isinstance(status, str) or status not in prediction_counts:
                status = "unavailable"
            prediction_counts[status] += 1
            if isinstance(health, Mapping):
                radius = _finite_metric_number(health.get("maximum_radius_m"))
                clipped = _finite_metric_number(health.get("clipped_point_fraction"))
                if radius is not None:
                    prediction_max_radius = max(prediction_max_radius, radius)
                if clipped is not None:
                    prediction_max_clipped = max(prediction_max_clipped, clipped)

    boundary_count = 0
    boundary_max_duration = 0.0
    region_failure_count = 0
    for index, event in enumerate(_frame_event_mappings(frame)):
        event_type = str(event.get("event_type", "")).lower()
        event_key = _metric_event_key(event, index)
        payload = event.get("payload")
        payload_object = payload if isinstance(payload, Mapping) else {}
        if event_type == "target_boundary_recovery_started" and event_key not in seen_boundary_events:
            seen_boundary_events.add(event_key)
            boundary_count += 1
        if event_type == "target_boundary_recovery_completed":
            duration = _finite_metric_number(event.get("state_age_s"))
            if duration is None:
                duration = _finite_metric_number(payload_object.get("state_age_s"))
            if duration is not None:
                boundary_max_duration = max(boundary_max_duration, duration)
        if "region" in event_type and any(
            token in event_type for token in ("fail", "error", "reject", "invalid")
        ) and event_key not in seen_region_failures:
            seen_region_failures.add(event_key)
            region_failure_count += 1

    execution = frame.get("execution")
    execution_object = execution if isinstance(execution, Mapping) else {}
    execution_age = _finite_metric_number(execution_object.get("data_age_s")) or 0.0
    sim_time = _finite_metric_number(frame.get("sim_time_s"))
    valid_until = _finite_metric_number(execution_object.get("valid_until_s"))
    expired = int(
        execution_object.get("health_status") in {"expired", "failed"}
        or (
            sim_time is not None
            and valid_until is not None
            and sim_time >= valid_until
        )
    )
    execution_revision = execution_object.get("execution_revision")
    plan_source = execution_object.get("plan_source")
    commit_key = (
        f"{plan_source}:{execution_revision}"
        if isinstance(execution_revision, int) and isinstance(plan_source, str)
        else None
    )
    deterministic_commits = 0
    llm_commits = 0
    if commit_key is not None and commit_key not in seen_commit_keys:
        seen_commit_keys.add(commit_key)
        if plan_source == "deterministic":
            deterministic_commits = 1
        elif plan_source == "llm_optimized":
            llm_commits = 1

    for reason_key in ("health_reasons", "degradation_reasons"):
        reasons = execution_object.get(reason_key)
        if isinstance(reasons, (list, tuple)):
            for reason in reasons:
                text = str(reason).lower()
                if "region" in text and any(
                    token in text for token in ("fail", "error", "reject", "invalid")
                ):
                    reason_key_value = f"{reason_key}:{reason}:{execution_revision}"
                    if reason_key_value not in seen_region_failures:
                        seen_region_failures.add(reason_key_value)
                        region_failure_count += 1

    return {
        "prediction_counts": prediction_counts,
        "prediction_observations": prediction_observations,
        "prediction_max_radius_m": prediction_max_radius,
        "prediction_max_clipped_fraction": prediction_max_clipped,
        "target_boundary_recovery_count": boundary_count,
        "target_boundary_recovery_max_duration_s": boundary_max_duration,
        "execution_max_data_age_s": execution_age,
        "deterministic_baseline_commit_count": deterministic_commits,
        "llm_optimization_commit_count": llm_commits,
        "region_generation_failure_count": region_failure_count,
        "expired_execution_frame_count": expired,
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utc_now().isoformat()


def _as_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("HTTP response was not a JSON object")
    return cast(dict[str, object], value)


def _request_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    params: QueryParams | None = None,
    payload: Mapping[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    response = client.request(
        method,
        path,
        params=params,
        json=payload,
        timeout=_HTTP_TIMEOUT_S,
    )
    try:
        body = _as_object(response.json())
    except (ValueError, RuntimeError):
        body = {}
    return response.status_code, body


def _allocate_port(requested: int) -> int:
    if not 0 <= requested <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", requested))
        return int(probe.getsockname()[1])


def _spawn_owned_process(command: tuple[str, ...]) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    if os.name == "nt":
        return subprocess.Popen(
            list(command),
            cwd=str(_REPOSITORY_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    return subprocess.Popen(
        list(command),
        cwd=str(_REPOSITORY_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _owned_process_group_is_valid(process: subprocess.Popen[bytes]) -> bool:
    if process.poll() is not None:
        return False
    if os.name == "nt":
        return True
    try:
        return os.getpgid(process.pid) == process.pid
    except ProcessLookupError:
        return False


def _send_sigint_once(
    process: subprocess.Popen[bytes],
    shutdown: dict[str, object],
) -> None:
    if process.poll() is not None or bool(shutdown.get("sigint_sent")):
        return
    if os.name == "nt":
        control_break = cast(int, getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT))
        process.send_signal(control_break)
    else:
        if not _owned_process_group_is_valid(process):
            raise _AcceptanceFailure("refused SIGINT because the owned process group was invalid")
        os.killpg(process.pid, signal.SIGINT)
    shutdown["sigint_sent"] = True
    shutdown["sigint_sent_at"] = _timestamp()
    previous_count = shutdown.get("sigint_count", 0)
    shutdown["sigint_count"] = (int(previous_count) if isinstance(previous_count, int) else 0) + 1


def _terminate_validated_group(
    process: subprocess.Popen[bytes],
    shutdown: dict[str, object],
) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.kill()
        return
    if not _owned_process_group_is_valid(process):
        shutdown["forced_kill_refused"] = True
        raise _AcceptanceFailure("refused forced cleanup because the process group was invalid")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2.0)
        shutdown["forced_signal"] = "SIGTERM"
        return
    except subprocess.TimeoutExpired:
        pass
    if not _owned_process_group_is_valid(process):
        shutdown["forced_kill_refused"] = True
        raise _AcceptanceFailure("refused SIGKILL because the process group was invalid")
    os.killpg(process.pid, signal.SIGKILL)
    shutdown["forced_signal"] = "SIGKILL"
    process.wait(timeout=2.0)


def _shutdown_owned_process(
    process: subprocess.Popen[bytes],
    shutdown: dict[str, object],
) -> None:
    shutdown["started_at"] = shutdown.get("started_at", _timestamp())
    try:
        _send_sigint_once(process, shutdown)
        process.wait(timeout=_PROCESS_SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        shutdown["sigint_timeout"] = True
        _terminate_validated_group(process, shutdown)
    finally:
        shutdown["completed_at"] = _timestamp()
        shutdown["returncode"] = process.poll()


class _HealthSampler:
    """One bounded health request per wall-clock second for the whole run."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.samples: list[dict[str, object]] = []
        self._thread = threading.Thread(target=self._run, name="acceptance-health", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        next_due = time.monotonic()
        with httpx.Client(base_url=self._base_url) as client:
            while not self._stop.is_set():
                started = time.perf_counter()
                status: int | None = None
                payload: dict[str, object] = {}
                error: str | None = None
                try:
                    status, payload = _request_json(client, "GET", "/api/health")
                except Exception as exc:  # noqa: BLE001 - report the bounded probe failure
                    error = type(exc).__name__
                latency_ms = (time.perf_counter() - started) * 1000.0
                sample = {
                    "wall_time_utc": _timestamp(),
                    "latency_ms": round(latency_ms, 3),
                    "http_status": status,
                    "status": payload.get("status"),
                    "plan_version": payload.get("plan_version"),
                    "error": error,
                }
                with self._lock:
                    self.samples.append(sample)
                next_due += _HEALTH_INTERVAL_S
                wait_s = max(0.0, next_due - time.monotonic())
                self._stop.wait(wait_s)

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [dict(sample) for sample in self.samples]


class _ReplayScanner:
    """Read replay using the API cursor and a bounded page size."""

    def __init__(self) -> None:
        self.offset = 0
        self.frames: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []
        self._event_keys: set[str] = set()
        self._next_event_order = 0

    def poll(self, client: httpx.Client) -> None:
        pages = 0
        while pages < _MAX_REPLAY_PAGES_PER_POLL:
            status, payload = _request_json(
                client,
                "GET",
                "/api/replay",
                params={"start_s": 0, "offset": self.offset, "limit": _REPLAY_PAGE_SIZE},
            )
            if status == 503:
                return
            if status != 200:
                raise RuntimeError(f"replay returned HTTP {status}")
            raw_frames = payload.get("frames", [])
            if not isinstance(raw_frames, list):
                raise RuntimeError("replay frames were not a JSON array")
            frames = [
                cast(dict[str, object], frame)
                for frame in raw_frames
                if isinstance(frame, dict)
            ]
            if not frames:
                return
            self.frames.extend(frames)
            if len(self.frames) > _MAX_REPLAY_FRAMES:
                del self.frames[:-_MAX_REPLAY_FRAMES]
            for frame in frames:
                raw_events = frame.get("events", [])
                if not isinstance(raw_events, list):
                    continue
                for raw_event in raw_events:
                    if not isinstance(raw_event, dict):
                        continue
                    event = cast(dict[str, object], raw_event)
                    key = str(
                        event.get(
                            "event_id",
                            f"{event.get('event_type')}:{event.get('sim_time_s')}:{self._next_event_order}",
                        )
                    )
                    if key in self._event_keys:
                        continue
                    self._event_keys.add(key)
                    event_with_order = dict(event)
                    event_with_order["_acceptance_order"] = self._next_event_order
                    self.events.append(event_with_order)
                    self._next_event_order += 1
            page_count = len(frames)
            self.offset += page_count
            pages += 1
            total_count = payload.get("total_count")
            if page_count < _REPLAY_PAGE_SIZE:
                return
            if isinstance(total_count, int) and self.offset >= total_count:
                return

    def payload(self) -> dict[str, object]:
        return {
            "frames": list(self.frames),
            "count": len(self.frames),
            "offset": self.offset,
            "page_size": _REPLAY_PAGE_SIZE,
            "events_seen": len(self.events),
        }


def _read_state(
    client: httpx.Client,
    scanner: _ReplayScanner,
    operator_state: Mapping[str, object],
) -> dict[str, object]:
    health_status, health = _request_json(client, "GET", "/api/health")
    snapshot_status, snapshot = _request_json(client, "GET", "/api/operational/snapshot")
    if snapshot_status not in {200, 503}:
        raise RuntimeError(f"operational snapshot returned HTTP {snapshot_status}")
    scanner.poll(client)
    latest_sample = operator_state.get("latest_health_latency_ms")
    state: dict[str, object] = {
        "health": health,
        "health_http_status": health_status,
        "snapshot": snapshot if snapshot_status == 200 else {},
        "replay": scanner.payload(),
        "frames": list(scanner.frames),
        "event_records": list(scanner.events),
        "replay_cursor": scanner.offset,
        "operator": dict(operator_state),
    }
    if latest_sample is not None:
        state["health_latency_ms"] = latest_sample
    return state


def _event_records(
    state: Mapping[str, object],
    event_type: str,
    after_order: int,
) -> list[dict[str, object]]:
    raw_records = state.get("event_records", [])
    if not isinstance(raw_records, list):
        return []
    return [
        cast(dict[str, object], record)
        for record in raw_records
        if isinstance(record, dict)
        and record.get("event_type") == event_type
        and int(record.get("_acceptance_order", -1)) > after_order
    ]


def _event_predicate(event_type: str) -> Predicate:
    def predicate(state: dict[str, object]) -> bool:
        return bool(_event_records(state, event_type, -1))

    return predicate


def _snapshot_plan_version(state: Mapping[str, object]) -> int:
    snapshot = state.get("snapshot")
    if not isinstance(snapshot, dict):
        return -1
    value = snapshot.get("plan_version")
    return int(value) if isinstance(value, (int, float)) else -1


def _int_value(value: object, default: int = 0) -> int:
    return int(value) if isinstance(value, (int, float)) else default


def _health_ready(state: Mapping[str, object]) -> bool:
    health = state.get("health")
    return (
        state.get("health_http_status") == 200
        and isinstance(health, dict)
        and health.get("status") in {"ok", "paused"}
    )


def _has_passive_track(state: Mapping[str, object]) -> bool:
    snapshot = state.get("snapshot")
    if not isinstance(snapshot, dict):
        return False
    groups = snapshot.get("groups", [])
    return isinstance(groups, list) and any(
        isinstance(group, dict) and group.get("mode") == "passive_track" for group in groups
    )


def _operator_finished(state: Mapping[str, object]) -> bool:
    operator = state.get("operator")
    if not isinstance(operator, dict):
        return False
    return bool(operator.get("done")) and operator.get("error") is None


def default_acceptance_checkpoints() -> tuple[AcceptanceCheckpoint, ...]:
    """Return the ordered semantic gates used by the CLI wrapper."""

    return (
        AcceptanceCheckpoint(
            "health_ready",
            None,
            _health_ready,
            30.0,
        ),
        AcceptanceCheckpoint("plan_committed", None, lambda state: _snapshot_plan_version(state) > 0, 300.0),
        AcceptanceCheckpoint(
            "uuv_boundary_entry",
            "uuv_boundary_entry_started",
            _event_predicate("uuv_boundary_entry_started"),
            180.0,
        ),
        AcceptanceCheckpoint("active_scan", "active_ping", _event_predicate("active_ping"), 180.0),
        AcceptanceCheckpoint(
            "target_detection_acquired",
            "target_detection_acquired",
            _event_predicate("target_detection_acquired"),
            180.0,
        ),
        AcceptanceCheckpoint(
            "adversary_decision",
            "target_mission_decision",
            _event_predicate("target_mission_decision"),
            300.0,
        ),
        AcceptanceCheckpoint("passive_track", None, _has_passive_track, 180.0),
        AcceptanceCheckpoint("handoff_completed", "handoff_completed", _event_predicate("handoff_completed"), 300.0),
        AcceptanceCheckpoint(
            "resource_threshold",
            None,
            lambda state: any(
                bool(_event_records(state, event_type, -1))
                for event_type in (
                    "endurance_threshold_crossed",
                    "battery_rotation",
                    "uuv_range_exhausted",
                    "uuv_energy_depleted",
                )
            ),
            300.0,
        ),
        AcceptanceCheckpoint(
            "uuv_boundary_exit",
            None,
            lambda state: any(
                bool(_event_records(state, event_type, -1))
                for event_type in (
                    "uuv_boundary_exit_started",
                    "uuv_boundary_exited",
                    "uuv_boundary_exit_completed",
                )
            ),
            300.0,
        ),
        AcceptanceCheckpoint(
            "uuv_boundary_replacement",
            "uuv_boundary_replacement",
            _event_predicate("uuv_boundary_replacement"),
            300.0,
        ),
        AcceptanceCheckpoint(
            "memory_source_processed",
            "memory_version_created",
            _operator_finished,
            180.0,
        ),
    )


def _health_latency(state: Mapping[str, object]) -> float | None:
    value = state.get("health_latency_ms")
    return float(value) if isinstance(value, (int, float)) else None


def _checkpoint_record(
    checkpoint: AcceptanceCheckpoint,
    state: Mapping[str, object],
    event: Mapping[str, object] | None,
) -> dict[str, object]:
    snapshot = state.get("snapshot")
    frame = cast(dict[str, object], snapshot) if isinstance(snapshot, dict) else {}
    event_ids: list[str] = []
    evidence_ids: list[str] = []
    if event is not None:
        event_id = event.get("event_id")
        if isinstance(event_id, str):
            event_ids.append(event_id)
        for field in ("evidence_ids", "source_ids", "source_event_ids", "source_decision_ids"):
            values = event.get(field)
            if isinstance(values, list):
                evidence_ids.extend(str(value) for value in values)
    return {
        "name": checkpoint.name,
        "event_type": checkpoint.event_type,
        "sim_time_s": frame.get("sim_time_s", event.get("sim_time_s") if event else None),
        "wall_time_utc": _timestamp(),
        "wall_time_s": time.time(),
        "frame_id": frame.get("frame_id"),
        "plan_version": frame.get("plan_version"),
        "event_ids": sorted(set(event_ids)),
        "evidence_ids": sorted(set(evidence_ids)),
        "health_latency_ms": _health_latency(state),
    }


def _latest_snapshot(client: httpx.Client) -> dict[str, object]:
    status, snapshot = _request_json(client, "GET", "/api/operational/snapshot")
    if status != 200:
        raise _AcceptanceFailure(f"operational snapshot returned HTTP {status}")
    return snapshot


def _current_scope(snapshot: Mapping[str, object]) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    targets_raw = snapshot.get("target_estimates", [])
    target_items = targets_raw if isinstance(targets_raw, list) else []
    target_ids = tuple(
        str(item.get("target_id"))
        for item in target_items
        if isinstance(item, dict) and item.get("target_id")
    )
    regions_raw = snapshot.get("regions", [])
    region_items = regions_raw if isinstance(regions_raw, list) else []
    region_ids = tuple(
        str(item.get("region_id"))
        for item in region_items
        if isinstance(item, dict) and item.get("region_id")
    )
    scenario_id = str(snapshot.get("scenario_id") or "uuv-only-single-target")
    return target_ids, region_ids, scenario_id


def _memory_stream_event(
    client: httpx.Client,
    *,
    conversation_id: str,
    scenario_id: str,
    after_cursor: int,
    event_type: str,
    deadline: float,
) -> tuple[dict[str, object], int]:
    cursor = after_cursor
    while time.monotonic() < deadline:
        status, payload = _request_json(
            client,
            "GET",
            "/api/assistant/memory/stream",
            params={
                "user_id": "operator",
                "conversation_id": conversation_id,
                "scenario_id": scenario_id,
                "after_cursor": cursor,
                "limit": 128,
                "include_scenario_events": "true",
            },
        )
        if status != 200:
            raise _AcceptanceFailure(f"memory stream returned HTTP {status}")
        events_raw = payload.get("events", [])
        events = [cast(dict[str, object], item) for item in events_raw if isinstance(item, dict)] if isinstance(events_raw, list) else []
        for event in events:
            if event.get("type") == event_type:
                next_cursor = payload.get("next_cursor")
                return event, int(next_cursor) if isinstance(next_cursor, int) else cursor
        next_cursor = payload.get("next_cursor")
        if isinstance(next_cursor, int) and next_cursor > cursor:
            cursor = next_cursor
        time.sleep(_POLL_INTERVAL_S)
    raise _AcceptanceFailure(f"memory event {event_type} timed out")


def _operator_checkpoint(
    records: list[dict[str, object]],
    name: str,
    event_type: str | None,
    snapshot: Mapping[str, object],
    event: Mapping[str, object] | None = None,
) -> None:
    state = {"snapshot": dict(snapshot), "health_latency_ms": None}
    records.append(_checkpoint_record(AcceptanceCheckpoint(name, event_type, lambda _: True, 0), state, event))


def _assert_safe_feedback(
    proposal: Mapping[str, object],
    target_ids: tuple[str, ...],
    region_ids: tuple[str, ...],
) -> None:
    directive_raw = proposal.get("directive")
    directive = cast(dict[str, object], directive_raw) if isinstance(directive_raw, dict) else {}
    locked_members = directive.get("locked_members", {})
    if locked_members not in ({}, None):
        raise _AcceptanceFailure("assistant preview changed locked UUV membership")
    assignment_ids = directive.get("assignment_uuv_ids", [])
    if assignment_ids not in ([], (), None):
        raise _AcceptanceFailure("assistant preview introduced an assignment")
    disabled_ids = directive.get("disabled_uuv_ids", [])
    if disabled_ids not in ([], (), None):
        raise _AcceptanceFailure("assistant preview disabled a UUV")
    feedback_regions = directive.get("feedback_region_ids", [])
    if isinstance(feedback_regions, list) and not set(map(str, feedback_regions)) <= set(region_ids):
        raise _AcceptanceFailure("assistant preview changed the task region")
    preview_targets = directive.get("target_scope", [])
    if isinstance(preview_targets, list) and not set(map(str, preview_targets)) <= set(target_ids):
        raise _AcceptanceFailure("assistant preview changed the target scope")


def _run_operator_lane(
    base_url: str,
    operator_state: dict[str, object],
    global_deadline: float,
) -> None:
    records: list[dict[str, object]] = []
    try:
        with httpx.Client(base_url=base_url) as client:
            snapshot = _latest_snapshot(client)
            target_ids, region_ids, scenario_id = _current_scope(snapshot)
            conversation_id = "acceptance-memory"
            cursor = 0
            first_plan_value = snapshot.get("plan_version", 0)
            first_plan_version = int(first_plan_value) if isinstance(first_plan_value, (int, float)) else 0
            status, first_turn = _request_json(
                client,
                "POST",
                "/api/conversation/messages",
                payload={
                    "conversation_id": conversation_id,
                    "user_id": "operator",
                    "assistant_mode": "auto",
                    "text": "请记住：目标接触后优先维持被动协同跟踪，只有丢失接触时才启用主动扫描。",
                    "expected_plan_version": first_plan_version,
                    "target_scope": target_ids,
                    "region_scope": region_ids,
                },
            )
            if status != 200:
                raise _AcceptanceFailure(f"durable preference submission returned HTTP {status}")
            first_event, cursor = _memory_stream_event(
                client,
                conversation_id=conversation_id,
                scenario_id=scenario_id,
                after_cursor=cursor,
                event_type="memory_version_created",
                deadline=min(global_deadline, time.monotonic() + 180.0),
            )
            memory_one = str(first_event.get("memory_id") or "")
            family_id = str(first_event.get("memory_family_id") or "")
            if not memory_one or not family_id:
                raise _AcceptanceFailure("memory version 1 had no family or memory id")
            _operator_checkpoint(records, "memory_version_1_created", "memory_version_created", snapshot, first_event)

            snapshot = _latest_snapshot(client)
            second_plan_value = snapshot.get("plan_version", 0)
            second_plan_version = int(second_plan_value) if isinstance(second_plan_value, (int, float)) else 0
            status, _ = _request_json(
                client,
                "POST",
                "/api/conversation/messages",
                payload={
                    "conversation_id": conversation_id,
                    "user_id": "operator",
                    "assistant_mode": "auto",
                    "text": "请改为：目标接触后优先主动扫描，只有确认接触稳定后才转被动协同跟踪。",
                    "expected_plan_version": second_plan_version,
                    "target_scope": target_ids,
                    "region_scope": region_ids,
                },
            )
            if status != 200:
                raise _AcceptanceFailure(f"conflicting preference submission returned HTTP {status}")
            second_event, cursor = _memory_stream_event(
                client,
                conversation_id=conversation_id,
                scenario_id=scenario_id,
                after_cursor=cursor,
                event_type="memory_version_created",
                deadline=min(global_deadline, time.monotonic() + 180.0),
            )
            memory_two = str(second_event.get("memory_id") or "")
            if not memory_two or memory_two == memory_one:
                raise _AcceptanceFailure("memory version 2 did not create a new memory id")
            _operator_checkpoint(records, "memory_version_2_created", "memory_version_created", snapshot, second_event)
            status, versions = _request_json(
                client,
                "GET",
                f"/api/assistant/memory/{family_id}/versions",
                params={"user_id": "operator", "scenario_id": scenario_id},
            )
            if status != 200 or not isinstance(versions.get("versions"), list):
                raise _AcceptanceFailure("memory version chain was unavailable")

            snapshot = _latest_snapshot(client)
            evidence_plan_value = snapshot.get("plan_version", 0)
            evidence_plan_version = int(evidence_plan_value) if isinstance(evidence_plan_value, (int, float)) else 0
            question_status, question = _request_json(
                client,
                "POST",
                "/api/questions",
                payload={"text": "为什么跟踪策略发生变化？"},
            )
            if question_status not in {200, 422}:
                raise _AcceptanceFailure(f"evidence question returned HTTP {question_status}")
            evidence_status, _ = _request_json(
                client,
                "POST",
                "/api/conversation/messages",
                payload={
                    "conversation_id": "acceptance-evidence",
                    "user_id": "operator",
                    "assistant_mode": "evidence_query",
                    "text": "请给出跟踪策略变化的可验证证据。",
                    "expected_plan_version": evidence_plan_version,
                    "target_scope": target_ids,
                    "region_scope": region_ids,
                },
            )
            if evidence_status != 200:
                raise _AcceptanceFailure(f"evidence conversation returned HTTP {evidence_status}")
            evidence_event, cursor = _memory_stream_event(
                client,
                conversation_id="acceptance-evidence",
                scenario_id=scenario_id,
                after_cursor=0,
                event_type="evidence_trace_completed",
                deadline=min(global_deadline, time.monotonic() + 180.0),
            )
            if question_status == 200 and not question:
                raise _AcceptanceFailure("evidence question returned an empty payload")
            _operator_checkpoint(records, "evidence_trace_completed", "evidence_trace_completed", snapshot, evidence_event)

            snapshot = _latest_snapshot(client)
            feedback_plan_value = snapshot.get("plan_version", 0)
            feedback_plan_version = int(feedback_plan_value) if isinstance(feedback_plan_value, (int, float)) else 0
            feedback_status, feedback = _request_json(
                client,
                "POST",
                "/api/conversation/messages",
                payload={
                    "conversation_id": "acceptance-feedback",
                    "user_id": "operator",
                    "assistant_mode": "plan_revision",
                    "text": "保持当前任务区域与 UUV 分配，仅重新确认交接窗口",
                    "expected_plan_version": feedback_plan_version,
                    "target_scope": target_ids,
                    "region_scope": region_ids,
                },
            )
            if feedback_status != 200:
                raise _AcceptanceFailure(f"assistant preview returned HTTP {feedback_status}")
            proposal = feedback.get("proposal")
            if not isinstance(proposal, dict):
                raise _AcceptanceFailure("assistant feedback did not return a preview")
            _assert_safe_feedback(cast(dict[str, object], proposal), target_ids, region_ids)
            turn_id = feedback.get("turn_id")
            if not isinstance(turn_id, str):
                raise _AcceptanceFailure("assistant preview did not return a turn id")
            _operator_checkpoint(records, "assistant_preview_created", "conversation_preview_created", snapshot)
            applied_status, applied = _request_json(
                client,
                "POST",
                "/api/conversation/acceptance-feedback/apply",
                payload={
                    "user_id": "operator",
                    "turn_id": turn_id,
                    "expected_plan_version": feedback_plan_version,
                },
            )
            if applied_status != 200:
                raise _AcceptanceFailure(f"assistant plan apply returned HTTP {applied_status}")
            if applied.get("applied") is not True and applied.get("proposal") is None:
                raise _AcceptanceFailure("assistant apply returned no applied result")
            apply_deadline = min(global_deadline, time.monotonic() + 300.0)
            while time.monotonic() < apply_deadline:
                snapshot = _latest_snapshot(client)
                if _int_value(snapshot.get("plan_version")) > feedback_plan_version:
                    break
                time.sleep(_POLL_INTERVAL_S)
            else:
                raise _AcceptanceFailure("assistant apply did not commit a newer plan")
            _operator_checkpoint(records, "assistant_plan_applied", "plan_committed", snapshot)

            delete_status, _ = _request_json(
                client,
                "DELETE",
                f"/api/assistant/memory/{memory_two}",
                params={
                    "user_id": "operator",
                    "scenario_id": scenario_id,
                    "conversation_id": conversation_id,
                },
            )
            if delete_status != 200:
                raise _AcceptanceFailure(f"memory deletion returned HTTP {delete_status}")
            snapshot = _latest_snapshot(client)
            semantic_raw = snapshot.get("semantic", [])
            semantic_items = semantic_raw if isinstance(semantic_raw, list) else []
            active_ids = {
                str(item.get("memory_id")) for item in semantic_items if isinstance(item, dict)
            }
            if memory_two in active_ids:
                raise _AcceptanceFailure("deleted memory version remained active")
            deleted_event, _ = _memory_stream_event(
                client,
                conversation_id=conversation_id,
                scenario_id=scenario_id,
                after_cursor=cursor,
                event_type="memory_deleted",
                deadline=min(global_deadline, time.monotonic() + 180.0),
            )
            if not deleted_event.get("source_message_ids"):
                raise _AcceptanceFailure("memory deletion did not retain source message provenance")
            _operator_checkpoint(records, "memory_version_deleted", "memory_deleted", snapshot, deleted_event)
    except Exception as exc:  # noqa: BLE001 - lane failure belongs in the artifact
        operator_state["error"] = str(exc)[:500]
    finally:
        operator_state["checkpoints"] = records
        operator_state["done"] = True


def _health_p95(samples: Sequence[Mapping[str, object]]) -> float | None:
    values: list[float] = []
    for sample in samples:
        value = sample.get("latency_ms")
        if isinstance(value, (int, float)):
            values.append(float(value))
    values.sort()
    if len(values) < _MIN_HEALTH_SAMPLES:
        return None
    index = min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)
    return round(values[index], 3)


def _output_run_dirs() -> set[Path]:
    output_root = _REPOSITORY_ROOT / "outputs"
    if not output_root.is_dir():
        return set()
    return {
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "agent.db").is_file()
    }


def _final_database_checks(before: set[Path]) -> dict[str, object]:
    candidates = sorted(_output_run_dirs() - before, key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise _AcceptanceFailure("no owned run directory with agent.db was created")
    run_dir = candidates[-1]
    database_path = run_dir / "agent.db"
    checks: dict[str, object] = {"run_dir": str(run_dir), "database": str(database_path)}
    uri = f"file:{database_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        counts: dict[str, int] = {}
        for table in (
            "llm_calls",
            "plans",
            "long_term_memories",
            "memory_stream_events",
            "short_term_messages",
        ):
            if table in tables:
                counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        operations = (
            sorted(
                str(row[0])
                for row in connection.execute("SELECT DISTINCT operation FROM llm_calls")
            )
            if "llm_calls" in tables
            else []
        )
    replay_path = run_dir / "operational_frames.jsonl"
    frame_count = 0
    contains_usv = False
    if replay_path.is_file():
        with replay_path.open(encoding="utf-8") as replay_file:
            for line in replay_file:
                frame_count += 1
                contains_usv = contains_usv or "usv" in line.lower()
    checks.update(
        {
            "table_counts": counts,
            "llm_operations": operations,
            "replay_frame_count": frame_count,
            "contains_usv": contains_usv,
        }
    )
    if contains_usv:
        raise _AcceptanceFailure("persisted operational replay contains a USV entity")
    if counts.get("plans", 0) < 2:
        raise _AcceptanceFailure("persisted run contains fewer than two plan records")
    if counts.get("memory_stream_events", 0) == 0:
        raise _AcceptanceFailure("persisted run contains no memory stream events")
    return checks


def run_acceptance(
    *,
    command: tuple[str, ...],
    api_port: int,
    ui_port: int | None = None,
    output_path: Path,
    checkpoints: tuple[AcceptanceCheckpoint, ...],
    playwright_command: tuple[str, ...] | None = None,
) -> int:
    """Run an owned process and return zero only when every gate passes."""

    output_path = output_path if output_path.is_absolute() else _REPOSITORY_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "status": "failed",
        "started_at": _timestamp(),
        "command": list(command),
        "checkpoints": [],
        "operator_checkpoints": [],
        "failure": None,
    }
    process: subprocess.Popen[bytes] | None = None
    browser: subprocess.Popen[bytes] | None = None
    sampler: _HealthSampler | None = None
    operator_thread: threading.Thread | None = None
    operator_state: dict[str, object] = {"done": False, "error": None, "checkpoints": []}
    shutdown: dict[str, object] = {"sigint_sent": False, "sigint_count": 0}
    process_started = time.monotonic()
    run_dirs_before = _output_run_dirs()
    success = False
    try:
        del ui_port  # The formal command center serves the UI from FastAPI.
        allocated_api_port = _allocate_port(api_port)
        report["ports"] = {"api": allocated_api_port}
        owned_command = command + (
            "--host",
            "127.0.0.1",
            "--port",
            str(allocated_api_port),
        )
        report["owned_command"] = list(owned_command)
        process = _spawn_owned_process(owned_command)
        process_started = time.monotonic()
        shutdown["process_started_at"] = _timestamp()
        report["process"] = {"pid": process.pid, "start_new_session": os.name != "nt"}
        base_url = f"http://127.0.0.1:{allocated_api_port}"
        sampler = _HealthSampler(base_url)
        sampler.start()
        scanner = _ReplayScanner()
        global_deadline = process_started + _GLOBAL_DEADLINE_S
        last_event_order = -1
        operator_enabled = any(cp.name == "memory_source_processed" for cp in checkpoints)
        with httpx.Client(base_url=base_url) as client:
            for checkpoint in checkpoints:
                if checkpoint.timeout_s <= 0:
                    raise ValueError(f"checkpoint {checkpoint.name} has a non-positive timeout")
                checkpoint_deadline = min(global_deadline, time.monotonic() + checkpoint.timeout_s)
                latest_state: dict[str, object] = {}
                selected_event: dict[str, object] | None = None
                predicate_error: str | None = None
                while time.monotonic() < checkpoint_deadline:
                    if process.poll() is not None:
                        raise _AcceptanceFailure(
                            f"owned process exited with code {process.returncode} before {checkpoint.name}"
                        )
                    try:
                        samples = sampler.snapshot()
                        if samples:
                            operator_state["latest_health_latency_ms"] = samples[-1].get("latency_ms")
                        latest_state = _read_state(client, scanner, operator_state)
                        candidates = (
                            _event_records(latest_state, checkpoint.event_type, last_event_order)
                            if checkpoint.event_type is not None
                            else []
                        )
                        selected_event = candidates[0] if candidates else None
                        try:
                            predicate_ok = checkpoint.predicate(latest_state)
                        except Exception as exc:  # noqa: BLE001 - report malformed fixture state
                            predicate_ok = False
                            predicate_error = f"{type(exc).__name__}: {exc}"
                        if predicate_ok and (checkpoint.event_type is None or selected_event is not None):
                            record = _checkpoint_record(checkpoint, latest_state, selected_event)
                            cast(list[object], report["checkpoints"]).append(record)
                            if selected_event is not None:
                                last_event_order = _int_value(
                                    selected_event.get("_acceptance_order"), last_event_order
                                )
                            if checkpoint.name == "health_ready":
                                if playwright_command is not None and browser is None:
                                    browser_env = os.environ.copy()
                                    browser_env["PLAYWRIGHT_BASE_URL"] = base_url
                                    browser = subprocess.Popen(
                                        list(playwright_command),
                                        cwd=str(_REPOSITORY_ROOT),
                                        env=browser_env,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        start_new_session=os.name != "nt",
                                    )
                                    report["playwright"] = {
                                        "command": list(playwright_command),
                                        "base_url": base_url,
                                        "started_at": _timestamp(),
                                        "pid": browser.pid,
                                    }
                            if checkpoint.name == "plan_committed" and operator_enabled and operator_thread is None:
                                operator_thread = threading.Thread(
                                    target=_run_operator_lane,
                                    args=(base_url, operator_state, global_deadline),
                                    name="acceptance-operator-lane",
                                    daemon=True,
                                )
                                operator_thread.start()
                            break
                    except _AcceptanceFailure:
                        raise
                    except Exception as exc:  # noqa: BLE001 - bounded retry until deadline
                        predicate_error = f"{type(exc).__name__}: {exc}"
                    time.sleep(_POLL_INTERVAL_S)
                else:
                    cursor = latest_state.get("replay_cursor")
                    raise _AcceptanceFailure(
                        f"checkpoint {checkpoint.name} timed out; replay_cursor={cursor!r}; "
                        f"plan_version={_snapshot_plan_version(latest_state)}; "
                        f"predicate_error={predicate_error}"
                    )
                if time.monotonic() >= global_deadline:
                    raise _AcceptanceFailure("global acceptance deadline exceeded")

            if operator_thread is not None:
                remaining = max(0.0, global_deadline - time.monotonic())
                operator_thread.join(timeout=remaining)
                if operator_thread.is_alive():
                    raise _AcceptanceFailure("operator and memory lane did not finish before the global deadline")
                if operator_state.get("error") is not None:
                    raise _AcceptanceFailure(str(operator_state["error"]))
            report["operator_checkpoints"] = list(cast(list[object], operator_state.get("checkpoints", [])))

            if browser is not None:
                remaining = max(0.0, global_deadline - time.monotonic())
                try:
                    browser.wait(timeout=remaining)
                except subprocess.TimeoutExpired as exc:
                    raise _AcceptanceFailure("Playwright did not finish before the global deadline") from exc
                playwright_report = cast(dict[str, object], report["playwright"])
                playwright_report["ended_at"] = _timestamp()
                playwright_report["returncode"] = browser.returncode
                if browser.returncode != 0:
                    raise _AcceptanceFailure(f"Playwright exited with code {browser.returncode}")
                process_end = time.monotonic()
                if process_end < process_started:
                    raise _AcceptanceFailure("invalid process interval recorded for Playwright coupling")
                playwright_report["contained_in_owned_process"] = True

            if sampler is not None:
                samples = sampler.snapshot()
                if operator_enabled and len(samples) < _MIN_HEALTH_SAMPLES:
                    raise _AcceptanceFailure(
                        f"health sample count {len(samples)} is below {_MIN_HEALTH_SAMPLES}"
                    )
        success = True
    except Exception as exc:  # noqa: BLE001 - report and clean up all owned resources
        report["failure"] = str(exc)[:1000]
    finally:
        if sampler is not None:
            sampler.stop()
            samples = sampler.snapshot()
            report["health"] = {
                "sample_count": len(samples),
                "p95_latency_ms": _health_p95(samples),
                "samples": samples,
            }
        if browser is not None and browser.poll() is None:
            try:
                if os.name == "nt":
                    browser.kill()
                elif _owned_process_group_is_valid(browser):
                    os.killpg(browser.pid, signal.SIGTERM)
                browser.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if process is not None:
            try:
                _shutdown_owned_process(process, shutdown)
            except Exception as exc:  # noqa: BLE001 - preserve the original failure
                if report.get("failure") is None:
                    report["failure"] = str(exc)[:1000]
                success = False
            report["process"] = {
                **cast(dict[str, object], report.get("process", {})),
                "ended_at": shutdown.get("completed_at"),
                "returncode": process.returncode,
            }
        if success and operator_enabled:
            try:
                report["database_checks"] = _final_database_checks(run_dirs_before)
            except Exception as exc:  # noqa: BLE001 - final persistence is a gate
                report["failure"] = str(exc)[:1000]
                success = False
        report["shutdown"] = shutdown
        report["ended_at"] = _timestamp()
        report["status"] = "passed" if success and report.get("failure") is None else "failed"
        output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


def run_live_acceptance(
    *,
    config_path: Path,
    seed: int,
    api_port: int,
    output_path: Path,
    playwright_command: tuple[str, ...] | None = None,
) -> int:
    """Run the strict real-server acceptance owned by this runner."""
    from underwater_tracking.config.loader import load_app_config
    from underwater_tracking.verification import live_demo

    output_path = output_path if output_path.is_absolute() else _REPOSITORY_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_config = config_path if config_path.is_absolute() else _REPOSITORY_ROOT / config_path
    report: dict[str, object] = {
        "status": "failed",
        "started_at": _timestamp(),
        "entrypoint": "main.py",
        "config": str(resolved_config.resolve()),
        "seed": seed,
        "checkpoints_s": list(CHECKPOINTS_S),
        "failure": None,
    }
    process: subprocess.Popen[bytes] | None = None
    browser: subprocess.Popen[bytes] | None = None
    websocket_capture: _LiveWebSocketFrameCapture | None = None
    run_dir: Path | None = None
    acceptance_dir: Path | None = None
    backend_errors: list[dict[str, object]] = []
    checkpoint_records: list[dict[str, object]] = []
    database_metrics: dict[str, object] = {}
    transport_metrics: dict[str, object] = {}
    metrics: dict[str, object] = {
        "schema_version": "live-acceptance.metrics.v1",
        "status": "running",
        "checkpoints": list(CHECKPOINTS_S),
        "checkpoint_count": 0,
        "database": database_metrics,
        "transport_hashes": transport_metrics,
        "planning_error_count": 0,
        "target_boundary_recovery_count": 0,
        "target_boundary_recovery_max_duration_s": 0.0,
        "prediction_valid_fraction": 0.0,
        "prediction_degraded_fraction": 0.0,
        "prediction_unavailable_fraction": 0.0,
        "prediction_max_radius_m": 0.0,
        "prediction_max_clipped_fraction": 0.0,
        "execution_max_data_age_s": 0.0,
        "deterministic_baseline_commit_count": 0,
        "llm_optimization_commit_count": 0,
        "region_generation_failure_count": 0,
        "expired_execution_frame_count": 0,
        "frame_channel_mismatch_count": 0,
        "browser_console_error_count": 0,
        "required_layer_missing_count": 0,
    }
    shutdown: dict[str, object] = {"sigint_sent": False, "sigint_count": 0}
    prediction_counts = {"valid": 0, "degraded": 0, "unavailable": 0}
    prediction_observations = 0
    seen_boundary_events: set[str] = set()
    seen_commit_keys: set[str] = set()
    seen_region_failures: set[str] = set()
    current_checkpoint_s: int | None = None
    current_frame: Mapping[str, object] | None = None
    success = False
    started_monotonic = time.monotonic()

    def finalize_metric_fractions() -> None:
        total = prediction_observations
        if total <= 0:
            fractions = {"valid": 0.0, "degraded": 0.0, "unavailable": 0.0}
        else:
            fractions = {
                status: round(count / total, 6)
                for status, count in prediction_counts.items()
            }
        metrics["prediction_valid_fraction"] = fractions["valid"]
        metrics["prediction_degraded_fraction"] = fractions["degraded"]
        metrics["prediction_unavailable_fraction"] = fractions["unavailable"]

    def persist_metrics() -> None:
        if acceptance_dir is None:
            return
        finalize_metric_fractions()
        (acceptance_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    try:
        if not resolved_config.is_file():
            raise _AcceptanceFailure(f"scenario config does not exist: {resolved_config}")
        config = load_app_config(resolved_config)
        scenario_id = str(config.scenario.scenario_id)
        output_root = output_path.parent / f"{output_path.stem}-owned-runs"
        output_root.mkdir(parents=True, exist_ok=True)
        run_dirs_before = {
            path.resolve()
            for path in output_root.iterdir()
            if path.is_dir() and (path / "agent.db").is_file()
        }
        allocated_api_port = _allocate_port(api_port)
        ui_root = _REPOSITORY_ROOT / "src" / "underwater_tracking" / "ui"
        main_command = (
            sys.executable,
            str(_REPOSITORY_ROOT / "main.py"),
            "--config",
            str(resolved_config.resolve()),
            "--seed",
            str(seed),
            "--steps",
            "0",
            "--bootstrap-planning",
            "--verification-audit",
            "--output-root",
            str(output_root.resolve()),
            "--host",
            "127.0.0.1",
            "--port",
            str(allocated_api_port),
        )
        report.update(
            {
                "owned_command": list(main_command),
                "ports": {"api": allocated_api_port},
                "output_root": str(output_root.resolve()),
            }
        )
        process = _spawn_owned_process(main_command)
        report["process"] = {
            "pid": process.pid,
            "entrypoint": "main.py",
            "sole_main_process_owner": True,
            "start_new_session": os.name != "nt",
        }
        shutdown["process_started_at"] = _timestamp()
        global_deadline = started_monotonic + _GLOBAL_DEADLINE_S
        base_url = f"http://127.0.0.1:{allocated_api_port}"
        with httpx.Client(base_url=base_url) as client:
            run_id, run_dir = _wait_for_owned_run(
                client,
                process,
                output_root,
                min(global_deadline, time.monotonic() + 120.0),
            )
            if run_dir in run_dirs_before:
                raise _AcceptanceFailure(f"owned run reused a previous run directory: {run_dir}")
            acceptance_dir = run_dir / "acceptance"
            (acceptance_dir / "screenshots").mkdir(parents=True, exist_ok=True)
            for artifact_name in (
                "metrics.json",
                "frame-checkpoints.jsonl",
                "browser-console.jsonl",
                "backend-errors.jsonl",
            ):
                (acceptance_dir / artifact_name).touch()
            report.update(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "acceptance_dir": str(acceptance_dir),
                }
            )
            persist_metrics()

            websocket_url = base_url.replace("http://", "ws://", 1) + "/ws/operational"
            websocket_capture = _LiveWebSocketFrameCapture(websocket_url)
            websocket_capture.start()
            if playwright_command is None:
                npm = shutil.which("npm.cmd") or shutil.which("npm") or "npm"
                playwright_command = (
                    npm,
                    "--prefix",
                    str(ui_root),
                    "run",
                    "test:e2e:live",
                )
            browser_env = os.environ.copy()
            browser_env.update(
                {
                    "PLAYWRIGHT_BASE_URL": base_url,
                    "UNDERWATER_TRACKING_ACCEPTANCE_DIR": str(acceptance_dir),
                    "UNDERWATER_TRACKING_ACCEPTANCE_RUN_DIR": str(run_dir),
                    "UNDERWATER_TRACKING_ACCEPTANCE_CHECKPOINTS": ",".join(
                        str(value) for value in CHECKPOINTS_S
                    ),
                }
            )
            browser = subprocess.Popen(
                list(playwright_command),
                cwd=str(_REPOSITORY_ROOT),
                env=browser_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=os.name != "nt",
            )
            report["playwright"] = {
                "command": list(playwright_command),
                "base_url": base_url,
                "started_at": _timestamp(),
                "pid": browser.pid,
            }
            previous_frame: dict[str, object] | None = None
            previous_frame_id: int | None = None
            frame_log = run_dir / "operational_frames.jsonl"
            for checkpoint_s in CHECKPOINTS_S:
                current_checkpoint_s = checkpoint_s
                frame = _wait_for_checkpoint_frame(
                    client,
                    process,
                    checkpoint_s,
                    global_deadline,
                    previous_frame_id=previous_frame_id,
                )
                current_frame = frame
                observation = _checkpoint_metric_observation(
                    frame,
                    seen_boundary_events=seen_boundary_events,
                    seen_commit_keys=seen_commit_keys,
                    seen_region_failures=seen_region_failures,
                )
                observed_counts = observation.get("prediction_counts")
                if isinstance(observed_counts, Mapping):
                    for status in prediction_counts:
                        count = observed_counts.get(status, 0)
                        if isinstance(count, int):
                            prediction_counts[status] += count
                observed_count = observation.get("prediction_observations")
                if isinstance(observed_count, int):
                    prediction_observations += observed_count
                for key in (
                    "target_boundary_recovery_count",
                    "deterministic_baseline_commit_count",
                    "llm_optimization_commit_count",
                    "region_generation_failure_count",
                    "expired_execution_frame_count",
                ):
                    increment = observation.get(key, 0)
                    if isinstance(increment, int):
                        metrics[key] = int(metrics.get(key, 0)) + increment
                for key in (
                    "target_boundary_recovery_max_duration_s",
                    "prediction_max_radius_m",
                    "prediction_max_clipped_fraction",
                    "execution_max_data_age_s",
                ):
                    observed_value = observation.get(key)
                    if isinstance(observed_value, (int, float)):
                        metrics[key] = max(
                            float(metrics.get(key, 0.0)),
                            float(observed_value),
                        )
                prediction_cap = _effective_prediction_cap(frame, config)
                violations = live_demo.validate_live_checkpoint_frame(
                    frame,
                    prediction_radius_cap_m=prediction_cap,
                    execution_max_age_s=float(config.tracking.prediction_health.hard_stale_s),
                    previous_frame=previous_frame,
                )
                if violations:
                    for violation in violations:
                        _backend_error(
                            acceptance_dir,
                            backend_errors,
                            violation,
                            checkpoint_s=checkpoint_s,
                            frame=frame,
                            details={"prediction_radius_cap_m": prediction_cap},
                        )
                    raise _AcceptanceFailure(
                        f"backend semantic validation failed at {checkpoint_s}s: "
                        + ", ".join(violations)
                    )

                frame_id = frame.get("frame_id")
                if not isinstance(frame_id, int) or isinstance(frame_id, bool):
                    raise _AcceptanceFailure(f"checkpoint {checkpoint_s}s has no valid frame id")
                websocket_frame = websocket_capture.frame(frame_id) if websocket_capture else None
                websocket_deadline = min(global_deadline, time.monotonic() + 5.0)
                while websocket_frame is None and time.monotonic() < websocket_deadline:
                    time.sleep(0.05)
                    websocket_frame = websocket_capture.frame(frame_id) if websocket_capture else None
                jsonl_frame = _wait_for_jsonl_frame(
                    frame_log,
                    frame_id,
                    min(global_deadline, time.monotonic() + 5.0),
                )
                transport_frames: dict[str, Mapping[str, object] | None] = {
                    "http": frame,
                    "websocket": websocket_frame,
                    "jsonl": jsonl_frame,
                }
                transport_hashes, hash_violations = live_demo.validate_transport_payload_hashes(
                    transport_frames
                )
                transport_violations = live_demo.validate_transport_frame_consistency(
                    {
                        key: value
                        for key, value in transport_frames.items()
                        if isinstance(value, Mapping)
                    }
                )
                all_transport_violations = tuple(
                    dict.fromkeys((*hash_violations, *transport_violations))
                )
                if all_transport_violations:
                    metrics["frame_channel_mismatch_count"] = (
                        int(metrics.get("frame_channel_mismatch_count", 0)) + 1
                    )
                    for violation in all_transport_violations:
                        _backend_error(
                            acceptance_dir,
                            backend_errors,
                            violation,
                            checkpoint_s=checkpoint_s,
                            frame=frame,
                            details={"transport_hashes": transport_hashes},
                        )
                    raise _AcceptanceFailure(
                        f"transport consistency failed at {checkpoint_s}s: "
                        + ", ".join(all_transport_violations)
                    )
                database = live_demo.read_latest_execution_database_evidence(
                    run_dir / "agent.db",
                    scenario_id,
                )
                database_violations = live_demo.validate_database_execution_consistency(
                    frame,
                    database,
                )
                if database_violations:
                    for violation in database_violations:
                        _backend_error(
                            acceptance_dir,
                            backend_errors,
                            violation,
                            checkpoint_s=checkpoint_s,
                            frame=frame,
                            details={"database": database},
                        )
                    raise _AcceptanceFailure(
                        f"database consistency failed at {checkpoint_s}s: "
                        + ", ".join(database_violations)
                    )
                record = _checkpoint_evidence(
                    checkpoint_s,
                    frame,
                    transport_hashes,
                    database,
                )
                checkpoint_records.append(record)
                _append_jsonl_record(acceptance_dir / "frame-checkpoints.jsonl", record)
                database_metrics[str(checkpoint_s)] = dict(database)
                transport_metrics[str(checkpoint_s)] = dict(transport_hashes)
                metrics["checkpoint_count"] = len(checkpoint_records)
                metrics["planning_error_count"] = sum(
                    int(item.get("planning_error_count", 0))
                    for item in database_metrics.values()
                    if isinstance(item, Mapping)
                )
                persist_metrics()
                previous_frame = frame
                previous_frame_id = frame_id

            if websocket_capture.error:
                _backend_error(
                    acceptance_dir,
                    backend_errors,
                    "websocket_capture_failed",
                    checkpoint_s=current_checkpoint_s,
                    frame=current_frame,
                    details={"error": websocket_capture.error},
                )
                raise _AcceptanceFailure(websocket_capture.error)
            if browser is not None:
                remaining = max(0.0, global_deadline - time.monotonic())
                try:
                    browser.wait(timeout=remaining)
                except subprocess.TimeoutExpired as exc:
                    raise _AcceptanceFailure("live Playwright did not finish before the deadline") from exc
                playwright_report = cast(dict[str, object], report["playwright"])
                playwright_report.update(
                    {"ended_at": _timestamp(), "returncode": browser.returncode}
                )
                if browser.returncode != 0:
                    raise _AcceptanceFailure(f"live Playwright exited with code {browser.returncode}")
            browser_console_records = _read_jsonl_records(acceptance_dir / "browser-console.jsonl")
            console_errors = [
                record
                for record in browser_console_records
                if record.get("type") in {"error", "pageerror"}
            ]
            metrics["browser_console_error_count"] = len(console_errors)
            if console_errors:
                raise _AcceptanceFailure(
                    f"browser console contains {len(console_errors)} error record(s)"
                )
            if len(checkpoint_records) != len(CHECKPOINTS_S):
                raise _AcceptanceFailure("not all live checkpoints produced evidence")
            metrics["status"] = "passed"
            success = True
    except Exception as exc:  # noqa: BLE001 - the report is the acceptance handoff
        report["failure"] = str(exc)[:2000]
        if acceptance_dir is not None and not any(
            record.get("error") == "acceptance_failure" for record in backend_errors
        ):
            _backend_error(
                acceptance_dir,
                backend_errors,
                "acceptance_failure",
                checkpoint_s=current_checkpoint_s,
                frame=current_frame,
                details={"message": str(exc)[:1000]},
            )
    finally:
        if websocket_capture is not None:
            websocket_capture.stop()
        if browser is not None and browser.poll() is None:
            try:
                if os.name == "nt":
                    browser.kill()
                elif _owned_process_group_is_valid(browser):
                    os.killpg(browser.pid, signal.SIGTERM)
                browser.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if process is not None:
            try:
                _shutdown_owned_process(process, shutdown)
            except Exception as exc:  # noqa: BLE001 - preserve the primary failure
                if report.get("failure") is None:
                    report["failure"] = str(exc)[:2000]
                success = False
            report["process"] = {
                **cast(dict[str, object], report.get("process", {})),
                "ended_at": shutdown.get("completed_at"),
                "returncode": process.returncode,
            }
        metrics["status"] = "passed" if success else "failed"
        metrics["duration_s"] = round(time.monotonic() - started_monotonic, 3)
        if acceptance_dir is not None:
            try:
                persist_metrics()
            except (OSError, TypeError, ValueError) as exc:
                if report.get("failure") is None:
                    report["failure"] = f"failed to persist metrics: {exc}"
                success = False

    if success and run_dir is not None and acceptance_dir is not None:
        try:
            screenshot_paths = {
                f"{viewport}-{checkpoint}": acceptance_dir / "screenshots" / f"{viewport}-{checkpoint}.png"
                for viewport in ("desktop", "mobile")
                for checkpoint in CHECKPOINTS_S
            }
            write_acceptance_artifacts(
                run_dir=run_dir,
                config_path=resolved_config,
                seed=seed,
                ui_bundle_path=_REPOSITORY_ROOT / "src" / "underwater_tracking" / "ui" / "dist",
                operational_frames_path=run_dir / "operational_frames.jsonl",
                checkpoint_records=checkpoint_records,
                metrics=metrics,
                screenshot_paths=screenshot_paths,
                browser_console_records=_read_jsonl_records(
                    acceptance_dir / "browser-console.jsonl"
                ),
                backend_error_records=_read_jsonl_records(
                    acceptance_dir / "backend-errors.jsonl"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - artifact completeness is a gate
            report["failure"] = str(exc)[:2000]
            success = False
    report["shutdown"] = shutdown
    report["metrics"] = metrics
    report["ended_at"] = _timestamp()
    report["status"] = "passed" if success and report.get("failure") is None else "failed"
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if report["status"] == "passed" else 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-port", type=int, default=0)
    parser.add_argument("--ui-port", type=int, default=0)
    parser.add_argument("--playwright-command", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if os.environ.get("UNDERWATER_TRACKING_RUN_LIVE_ACCEPTANCE") != "1":
        print("set UNDERWATER_TRACKING_RUN_LIVE_ACCEPTANCE=1 to run live acceptance", file=sys.stderr)
        return 2
    if os.environ.get("UNDERWATER_TRACKING_RUN_REAL_LLM") != "1":
        print("set UNDERWATER_TRACKING_RUN_REAL_LLM=1 to run live acceptance", file=sys.stderr)
        return 2
    del args.ui_port  # main.py serves the built UI from the owned API process.
    playwright_command = (
        tuple(shlex.split(args.playwright_command))
        if args.playwright_command
        else None
    )
    return run_live_acceptance(
        config_path=args.config,
        seed=args.seed,
        api_port=args.api_port,
        output_path=args.output,
        playwright_command=playwright_command,
    )


if __name__ == "__main__":
    raise SystemExit(main())
