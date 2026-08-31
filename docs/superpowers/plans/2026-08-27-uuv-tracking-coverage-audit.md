# Multi-UUV Tracking and Coverage Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce current, reproducible evidence for repository-native multi-UUV cooperative tracking and serpentine coverage, correct the confirmed active-scan route-selection defect with TDD, render compact headless media, and push only the isolated review branch.

**Architecture:** A deterministic, no-network audit runner installs the repository's existing public-prior baseline, drives the current `SimulationEngine`, and records operational frames plus the separate evaluation sink. Pure verification helpers compute tracking, motion, route, geometry, determinism, and physics metrics from that single trace; a separate renderer consumes the saved trace so plots and videos cannot diverge from the measured run.

**Tech Stack:** Python 3.12, pytest, NumPy/SciPy, Pydantic, Matplotlib Agg, ImageIO with the repository-local `imageio-ffmpeg` wheel, Git.

---

## 1. Preconditions and evidence policy

- Repository: `D:\Air\反Q\Underwater-Tracking`
- Branch: `review/uuv-tracking-coverage-20260827`
- Design commit: `ec2fe3b`
- Baseline commit: `63b13f60f7de639bed4751260c83236c67e9e54c`
- Scenario: `configs/scenario/uuv_only_single_target.yaml`
- Native seed: `42`
- Short audit length: 360 physics steps = 1,800 simulated seconds at the configured 5 s step.
- Repeat count: two complete runs with seed 42.
- No real LLM, HTTP server, Node process, ROS process, or robot is started.
- Target truth enters only through `evaluation_sink` and the verification package. It never enters estimator, planner, mission controller, or waypoint logic.
- Hard pass/fail claims come only from configured limits or mathematical invariants. Error statistics and sampled coverage are reported without invented performance thresholds.
- `outputs/`, `.venv/`, pytest caches, raw frame logs, SQLite files, and temporary render state remain uncommitted.
- Do not delete any output. If a rerun is needed, use a new suffix such as `run-c` or `rerun-01`.

## 2. File map

### Files to create

- `src/underwater_tracking/verification/uuv_tracking_coverage_audit.py` — pure trace projection, metric calculations, deterministic hashing, route progress, and sampled active-sonar footprint coverage.
- `src/underwater_tracking/verification/uuv_tracking_coverage_runner.py` — deterministic engine orchestration, no-network LLM sentinel, separate truth capture, two-run comparison, and JSON artifact writing.
- `src/underwater_tracking/verification/uuv_tracking_coverage_render.py` — Matplotlib Agg rendering from the persisted trace and ImageIO video/GIF export.
- `scripts/run_uuv_tracking_coverage_audit.py` — thin repository-root entry point for the runner.
- `scripts/render_uuv_tracking_coverage_audit.py` — thin repository-root entry point for rendering.
- `tests/verification/test_uuv_tracking_coverage_audit.py` — pure metric and truth-isolation tests.
- `tests/verification/test_uuv_tracking_coverage_runner.py` — no-network sentinel, projection, and two-step runner smoke tests.
- `tests/verification/test_uuv_tracking_coverage_render.py` — renderer/keyframe smoke tests using a tiny synthetic trace.
- `docs/verification/2026-08-27-uuv-tracking-coverage/README.md` — Chinese audit report written from current command output.
- `docs/verification/2026-08-27-uuv-tracking-coverage/metrics.json`
- `docs/verification/2026-08-27-uuv-tracking-coverage/trajectory.json`
- `docs/verification/2026-08-27-uuv-tracking-coverage/tracking-control.mp4`
- `docs/verification/2026-08-27-uuv-tracking-coverage/coverage-search.mp4`
- `docs/verification/2026-08-27-uuv-tracking-coverage/tracking-keyframe.png`
- `docs/verification/2026-08-27-uuv-tracking-coverage/coverage-keyframe.png`

### Files to modify

- `src/underwater_tracking/simulation/engine.py:6114-6256` — preserve assigned serpentine routes during an ordinary pre-entry `ACTIVE_SCAN`.
- `tests/simulation/test_uuv_only_carrier_group.py:1198-1262` — require both active and passive members to execute their assigned scan routes.
- `pyproject.toml:24-25` — add a reproducible `audit` optional dependency group for headless rendering only.
- `src/underwater_tracking/verification/__init__.py` — export only stable pure audit functions.

### Files explicitly not modified

- All files below `D:\Air\反Q\tracking\multi_AUV_pursuit_ros_ws`
- All files below `D:\Air\反Q\tracking\stft_bot_ros_ws`
- Existing Conda environments
- `configs/scenario/uuv_only_single_target.yaml` and physical/sensor thresholds
- Existing media files `test1.webm` and `test1-preview.png`

## 3. Source-confirmed defect addressed by this plan

The ordinary pre-entry mission path currently computes `rolling_routes_by_region` for any deployed two-member region with an active-capable member. In `ACTIVE_SCAN`, that route is preferred over `RegionMissionState.scan_waypoints_by_uuv`. A public-prior FIM route can therefore replace the documented deterministic serpentine coverage before target entry. The existing multi-member test checks only the passive member's first point and does not assert the active member's complete assigned lane.

The minimal correction is deliberately narrow:

- ordinary `ACTIVE_SCAN` with no predecessor handoff uses the assigned scan route for every region member;
- active-capable members assigned the active role continue to ping;
- passive members remain passive but still execute their coverage lane;
- `PASSIVE_TRACK`, dedicated tracking, handoff geometry, return-to-region, and `_plan_mission_group_waypoints` remain unchanged.

This is a source-level mismatch. Runtime acceptance still has to demonstrate physical route execution after the fix.

### Task 1: Preflight and create the isolated Python environment

**Files:**
- Create locally but do not commit: `.venv/`
- Create locally but do not commit: `outputs/audit-20260827/` through later commands

- [ ] **Step 1: Verify branch, clean state, interpreter, and free disk**

Run from PowerShell:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$py312 = 'C:\Users\10160\.conda\envs\llm_il_py312\python.exe'
git -C $repo status --short --branch
git -C $repo rev-parse HEAD
& $py312 --version
Get-PSDrive -Name D | Select-Object Name,Free,Used
```

Expected: the review branch is clean, HEAD is the plan commit, Python reports 3.12.13, and disk space is sufficient. Stop if the branch, worktree ownership, or interpreter differs.

- [ ] **Step 2: Create `.venv` without activating or modifying Conda**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$py312 = 'C:\Users\10160\.conda\envs\llm_il_py312\python.exe'
& $py312 -m venv "$repo\.venv"
& "$repo\.venv\Scripts\python.exe" --version
& "$repo\.venv\Scripts\python.exe" -m pip --version
```

Expected: both commands resolve inside `D:\Air\反Q\Underwater-Tracking\.venv` and Python remains 3.12.x.

- [ ] **Step 3: Install only declared runtime, development, and headless audit packages into `.venv`**

Run:

```powershell
$python = 'D:\Air\反Q\Underwater-Tracking\.venv\Scripts\python.exe'
& $python -m pip install --disable-pip-version-check --no-cache-dir `
  'httpx>=0.28,<1' 'fastapi>=0.115,<1' 'langgraph>=1.1,<2' `
  'langgraph-checkpoint-sqlite>=3.1,<4' 'numpy>=2.1,<3' `
  'pydantic>=2.10,<3' 'PyYAML>=6,<7' 'scipy>=1.14,<2' `
  'uvicorn>=0.34,<1' 'hypothesis>=6,<7' 'mypy>=1.14,<2' `
  'pytest>=8,<9' 'ruff>=0.9,<1' `
  'matplotlib>=3.9,<4' 'imageio>=2.36,<3' 'imageio-ffmpeg>=0.5,<1'
```

Expected: exit code 0; declared packages are written only under `.venv`. `--no-cache-dir` prevents a persistent user-level pip cache write, and no package is installed into the source Conda environment.

- [ ] **Step 4: Verify imports and record versions**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$python = "$repo\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path "$repo\outputs\audit-20260827\logs" | Out-Null
& $python -c "import sys,numpy,scipy,pydantic,yaml,pytest,matplotlib,imageio,imageio_ffmpeg; print(sys.version); print('numpy',numpy.__version__); print('scipy',scipy.__version__); print('pydantic',pydantic.__version__); print('pytest',pytest.__version__); print('matplotlib',matplotlib.__version__); print('imageio',imageio.__version__); print('imageio_ffmpeg',imageio_ffmpeg.__version__)" |
  Tee-Object -FilePath "$repo\outputs\audit-20260827\logs\environment.txt"
git -C $repo status --short
```

Expected: only ignored `.venv/` and `outputs/` content is created; Git reports no tracked change.

### Task 2: Establish the current test baseline

**Files:**
- Read: `tests/planning/test_coverage_paths.py`
- Read: `tests/tracking/test_imm_uif.py`
- Read: `tests/simulation/test_mission_waypoint_geometry.py`
- Read: `tests/simulation/test_uuv_only_carrier_group.py`
- Read: `tests/integration/test_truthful_bootstrap_deployment_frames.py`
- Create locally but do not commit: `outputs/audit-20260827/logs/baseline-selected.txt`
- Create locally but do not commit: `outputs/audit-20260827/logs/baseline-suite.txt`

- [ ] **Step 1: Run focused current tests before changing behavior**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$python = "$repo\.venv\Scripts\python.exe"
Set-Location -LiteralPath $repo
& $python -m pytest -q `
  tests/planning/test_coverage_paths.py `
  tests/tracking/test_imm_uif.py `
  tests/simulation/test_mission_waypoint_geometry.py `
  tests/simulation/test_uuv_only_carrier_group.py::test_normal_mode_routes_active_uuvs_and_emits_region_scan_ping `
  tests/simulation/test_uuv_only_carrier_group.py::test_normal_mode_routes_all_region_members_before_target_entry `
  tests/integration/test_truthful_bootstrap_deployment_frames.py |
  Tee-Object -FilePath 'outputs/audit-20260827/logs/baseline-selected.txt'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected: current focused tests pass. Preserve any failure as baseline evidence and invoke `systematic-debugging` before behavior changes.

- [ ] **Step 2: Run the non-opt-in Python suite**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$python = "$repo\.venv\Scripts\python.exe"
Set-Location -LiteralPath $repo
& $python -m pytest -q -m 'not real_llm and not long_running and not live_acceptance' |
  Tee-Object -FilePath 'outputs/audit-20260827/logs/baseline-suite.txt'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected: exit code 0. Record the actual pass count and duration; do not reuse a historical count.

### Task 3: Preserve normal serpentine coverage with TDD

**Files:**
- Modify: `tests/simulation/test_uuv_only_carrier_group.py:1198-1262`
- Modify: `src/underwater_tracking/simulation/engine.py:6164-6256`

- [ ] **Step 1: Strengthen the existing pre-entry multi-member regression**

Replace the final assertions of `test_normal_mode_routes_all_region_members_before_target_entry` with:

```python
    assert controller.snapshot().uuv_modes["uuv_00"] is UUVMissionMode.ACTIVE_SCAN
    assert controller.snapshot().uuv_modes["uuv_03"] is UUVMissionMode.PASSIVE_TRACK
    assert tuple(engine._uuvs["uuv_00"].waypoints) == (
        region.scan_waypoints_by_uuv["uuv_00"]
    )
    assert tuple(engine._uuvs["uuv_03"].waypoints) == (
        region.scan_waypoints_by_uuv["uuv_03"]
    )
    assert engine._sensor_modes["uuv_00"] == "active"
    assert engine._sensor_modes["uuv_03"] == "passive"
```

- [ ] **Step 2: Run the regression and verify it fails for route override**

Run:

```powershell
$python = 'D:\Air\反Q\Underwater-Tracking\.venv\Scripts\python.exe'
Set-Location -LiteralPath 'D:\Air\反Q\Underwater-Tracking'
& $python -m pytest -q `
  tests/simulation/test_uuv_only_carrier_group.py::test_normal_mode_routes_all_region_members_before_target_entry
```

Expected: FAIL because at least one member receives a rolling or single-point route instead of its complete assigned serpentine lane. If it passes, stop and re-check the inference before modifying `engine.py`.

- [ ] **Step 3: Add the minimal ordinary-coverage branch**

In `SimulationEngine._plan_mission_waypoints`, immediately after `if region is None: continue` and before `RETURN_TO_REGION` handling, insert:

```python
            if (
                region.lifecycle is RegionLifecycle.ACTIVE_SCAN
                and region.handoff_from is None
                and mode in {
                    UUVMissionMode.ACTIVE_SCAN,
                    UUVMissionMode.PASSIVE_TRACK,
                }
            ):
                assigned_route = (
                    region.scan_waypoints_by_uuv.get(uuv_id, ())
                    or region.scan_waypoints
                )
                if (
                    mode is UUVMissionMode.ACTIVE_SCAN
                    and uuv_id in region.active_scan_uuv_ids
                    and self._uuvs[uuv_id].capability.active_sonar_available
                ):
                    self.set_sensor_mode(
                        uuv_id,
                        "active",
                        ping_contact_id=region.target_id,
                    )
                else:
                    self.set_sensor_mode(uuv_id, "passive")
                if assigned_route:
                    self._set_persistent_uuv_route(uuv_id, assigned_route)
                    commands_by_target.setdefault(region.target_id, {})[
                        uuv_id
                    ] = assigned_route[0]
                self._uuv_groups[uuv_id] = region.target_id
                continue
```

Do not change `_plan_mission_group_waypoints`. Its hold-spread and FIM logic remains available for passive tracking, handoff preparation, and dedicated tracking behavior.

- [ ] **Step 4: Run the new regression and neighboring waypoint tests**

Run:

```powershell
$python = 'D:\Air\反Q\Underwater-Tracking\.venv\Scripts\python.exe'
Set-Location -LiteralPath 'D:\Air\反Q\Underwater-Tracking'
& $python -m pytest -q `
  tests/simulation/test_uuv_only_carrier_group.py::test_normal_mode_routes_active_uuvs_and_emits_region_scan_ping `
  tests/simulation/test_uuv_only_carrier_group.py::test_normal_mode_routes_all_region_members_before_target_entry `
  tests/simulation/test_mission_waypoint_geometry.py `
  tests/planning/test_coverage_paths.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the focused fix**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
git -C $repo diff --check
git -C $repo add -- `
  src/underwater_tracking/simulation/engine.py `
  tests/simulation/test_uuv_only_carrier_group.py
git -C $repo diff --cached --check
git -C $repo commit -m 'fix: preserve serpentine routes during active scan'
```

Expected: one focused code/test commit; no generated files are staged.

### Task 4: Add pure audit metrics with TDD

**Files:**
- Create: `tests/verification/test_uuv_tracking_coverage_audit.py`
- Create: `src/underwater_tracking/verification/uuv_tracking_coverage_audit.py`
- Modify: `src/underwater_tracking/verification/__init__.py`

- [ ] **Step 1: Write metric tests against a two-frame synthetic trace**

Create `tests/verification/test_uuv_tracking_coverage_audit.py`:

```python
from __future__ import annotations

from math import sqrt

import pytest

from underwater_tracking.verification.uuv_tracking_coverage_audit import (
    command_motion_counts,
    deterministic_trace_digest,
    minimum_pairwise_separation_m,
    sampled_footprint_fraction,
    target_position_errors_m,
    waypoint_visit_fraction,
)


def _trace_frames() -> tuple[dict[str, object], ...]:
    return (
        {
            "sim_time_s": 0,
            "uuvs": [
                {"platform_id": "uuv_00", "position_xy": [0.0, 0.0], "deployment_state": "deployed"},
                {"platform_id": "uuv_01", "position_xy": [3.0, 4.0], "deployment_state": "deployed"},
            ],
            "tracks": [{"target_id": "target_00", "mean": [1.0, 0.0, 0.0, 0.0]}],
            "target_truth": [{"target_id": "target_00", "position_xy": [0.0, 0.0]}],
            "waypoint_commands": {"target_00": {"uuv_00": [2.0, 0.0]}},
        },
        {
            "sim_time_s": 5,
            "uuvs": [
                {"platform_id": "uuv_00", "position_xy": [1.0, 0.0], "deployment_state": "deployed"},
                {"platform_id": "uuv_01", "position_xy": [3.0, 4.0], "deployment_state": "deployed"},
            ],
            "tracks": [{"target_id": "target_00", "mean": [0.0, 2.0, 0.0, 0.0]}],
            "target_truth": [{"target_id": "target_00", "position_xy": [0.0, 0.0]}],
            "waypoint_commands": {},
        },
    )


def test_tracking_control_and_separation_metrics_use_the_same_frames() -> None:
    frames = _trace_frames()

    assert target_position_errors_m(frames, "target_00") == pytest.approx((1.0, 2.0))
    assert minimum_pairwise_separation_m(frames) == pytest.approx(sqrt(20.0))
    assert command_motion_counts(frames) == {
        "commanded_intervals": 1,
        "moved_intervals": 1,
    }


def test_route_progress_requires_physical_waypoint_visits() -> None:
    trajectory = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    route = ((0.0, 0.0), (1.0, 0.0), (3.0, 0.0))

    assert waypoint_visit_fraction(trajectory, route) == pytest.approx(2.0 / 3.0)


def test_sampled_footprint_is_unavailable_without_emissions_and_complete_for_large_range() -> None:
    polygon = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))

    assert sampled_footprint_fraction(polygon, (), samples_per_axis=11) is None
    assert sampled_footprint_fraction(
        polygon,
        (((5.0, 5.0), 100.0),),
        samples_per_axis=11,
    ) == pytest.approx(1.0)


def test_trace_digest_is_canonical_and_seed_sensitive() -> None:
    first = {"seed": 42, "frames": list(_trace_frames())}
    second = {"frames": list(_trace_frames()), "seed": 42}

    assert deterministic_trace_digest(first) == deterministic_trace_digest(second)
    assert deterministic_trace_digest(first) != deterministic_trace_digest(
        {**first, "seed": 43}
    )
```

- [ ] **Step 2: Run the test and verify the module is absent**

Run:

```powershell
$python = 'D:\Air\反Q\Underwater-Tracking\.venv\Scripts\python.exe'
Set-Location -LiteralPath 'D:\Air\反Q\Underwater-Tracking'
& $python -m pytest -q tests/verification/test_uuv_tracking_coverage_audit.py
```

Expected: collection fails with `ModuleNotFoundError` for `uuv_tracking_coverage_audit`.

- [ ] **Step 3: Implement the pure metric functions**

Create `src/underwater_tracking/verification/uuv_tracking_coverage_audit.py`:

```python
"""Pure metrics for the fixed-seed multi-UUV tracking and coverage audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from math import hypot

import numpy as np

Point = tuple[float, float]


def deterministic_trace_digest(trace: Mapping[str, object]) -> str:
    payload = json.dumps(
        trace,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _points_by_id(items: object, *, id_field: str) -> dict[str, Point]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return {}
    result: dict[str, Point] = {}
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        identifier = raw.get(id_field)
        position = raw.get("position_xy")
        if not isinstance(identifier, str) or not isinstance(position, Sequence):
            continue
        if len(position) < 2:
            continue
        result[identifier] = (float(position[0]), float(position[1]))
    return result


def target_position_errors_m(
    frames: Sequence[Mapping[str, object]],
    target_id: str,
) -> tuple[float, ...]:
    errors: list[float] = []
    for frame in frames:
        truth = _points_by_id(
            frame.get("target_truth"),
            id_field="target_id",
        ).get(target_id)
        if truth is None:
            continue
        tracks = frame.get("tracks")
        if not isinstance(tracks, Sequence) or isinstance(tracks, (str, bytes)):
            continue
        for raw in tracks:
            if not isinstance(raw, Mapping) or raw.get("target_id") != target_id:
                continue
            mean = raw.get("mean")
            if isinstance(mean, Sequence) and len(mean) >= 2:
                errors.append(
                    hypot(
                        float(mean[0]) - truth[0],
                        float(mean[1]) - truth[1],
                    )
                )
            break
    return tuple(errors)


def minimum_pairwise_separation_m(
    frames: Sequence[Mapping[str, object]],
) -> float | None:
    minimum: float | None = None
    for frame in frames:
        uuvs = frame.get("uuvs")
        if not isinstance(uuvs, Sequence) or isinstance(uuvs, (str, bytes)):
            continue
        deployed = [
            point
            for raw in uuvs
            if isinstance(raw, Mapping)
            and raw.get("deployment_state") == "deployed"
            for point in _points_by_id((raw,), id_field="platform_id").values()
        ]
        for index, left in enumerate(deployed):
            for right in deployed[index + 1 :]:
                distance = hypot(left[0] - right[0], left[1] - right[1])
                minimum = distance if minimum is None else min(minimum, distance)
    return minimum


def command_motion_counts(
    frames: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    commanded = 0
    moved = 0
    for current, following in zip(frames, frames[1:]):
        commands = current.get("waypoint_commands")
        if not isinstance(commands, Mapping):
            continue
        commanded_ids = {
            str(uuv_id)
            for by_target in commands.values()
            if isinstance(by_target, Mapping)
            for uuv_id in by_target
        }
        before = _points_by_id(current.get("uuvs"), id_field="platform_id")
        after = _points_by_id(following.get("uuvs"), id_field="platform_id")
        for uuv_id in sorted(commanded_ids & before.keys() & after.keys()):
            commanded += 1
            if hypot(
                after[uuv_id][0] - before[uuv_id][0],
                after[uuv_id][1] - before[uuv_id][1],
            ) > 1.0e-9:
                moved += 1
    return {"commanded_intervals": commanded, "moved_intervals": moved}


def waypoint_visit_fraction(
    trajectory: Sequence[Point],
    route: Sequence[Point],
    *,
    numerical_tolerance_m: float = 1.0e-6,
) -> float | None:
    if not route:
        return None
    visited = sum(
        any(
            hypot(sample[0] - point[0], sample[1] - point[1])
            <= numerical_tolerance_m
            for sample in trajectory
        )
        for point in route
    )
    return visited / len(route)


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        x1, y1 = start
        x2, y2 = end
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x <= crossing_x:
                inside = not inside
    return inside


def sampled_footprint_fraction(
    polygon: Sequence[Point],
    emissions: Sequence[tuple[Point, float]],
    *,
    samples_per_axis: int = 81,
) -> float | None:
    if not emissions:
        return None
    if samples_per_axis < 2:
        raise ValueError("samples_per_axis must be at least two")
    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    candidates = [
        (float(x), float(y))
        for x in np.linspace(min_x, max_x, samples_per_axis)
        for y in np.linspace(min_y, max_y, samples_per_axis)
        if _point_in_polygon((float(x), float(y)), polygon)
    ]
    if not candidates:
        return None
    covered = sum(
        any(
            hypot(point[0] - center[0], point[1] - center[1]) <= radius
            for center, radius in emissions
        )
        for point in candidates
    )
    return covered / len(candidates)


def percentile_summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }
```

- [ ] **Step 4: Export the stable pure functions**

Replace `src/underwater_tracking/verification/__init__.py` with:

```python
"""Release verification contracts and monitors."""

from underwater_tracking.verification.physics_invariants import (
    BattleEvidenceChain,
    EntityMotionAudit,
    EntityMotionLimits,
    FullBattleAcceptance,
    PhysicsInvariantMonitor,
)
from underwater_tracking.verification.uuv_tracking_coverage_audit import (
    command_motion_counts,
    deterministic_trace_digest,
    minimum_pairwise_separation_m,
    percentile_summary,
    sampled_footprint_fraction,
    target_position_errors_m,
    waypoint_visit_fraction,
)

__all__ = [
    "BattleEvidenceChain",
    "EntityMotionAudit",
    "EntityMotionLimits",
    "FullBattleAcceptance",
    "PhysicsInvariantMonitor",
    "command_motion_counts",
    "deterministic_trace_digest",
    "minimum_pairwise_separation_m",
    "percentile_summary",
    "sampled_footprint_fraction",
    "target_position_errors_m",
    "waypoint_visit_fraction",
]
```

- [ ] **Step 5: Run tests and static checks**

Run:

```powershell
$python = 'D:\Air\反Q\Underwater-Tracking\.venv\Scripts\python.exe'
Set-Location -LiteralPath 'D:\Air\反Q\Underwater-Tracking'
& $python -m pytest -q tests/verification/test_uuv_tracking_coverage_audit.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m ruff check `
  src/underwater_tracking/verification/uuv_tracking_coverage_audit.py `
  tests/verification/test_uuv_tracking_coverage_audit.py
```

Expected: all tests pass and Ruff reports no issues.

### Task 5: Add the deterministic no-network audit runner

**Files:**
- Create: `tests/verification/test_uuv_tracking_coverage_runner.py`
- Create: `src/underwater_tracking/verification/uuv_tracking_coverage_runner.py`
- Create: `scripts/run_uuv_tracking_coverage_audit.py`
- Modify: `pyproject.toml:24-25`

- [ ] **Step 1: Add the reproducible audit dependency group**

Extend `[project.optional-dependencies]`:

```toml
dev = ["hypothesis>=6,<7", "mypy>=1.14,<2", "pytest>=8,<9", "ruff>=0.9,<1"]
audit = [
  "matplotlib>=3.9,<4",
  "imageio>=2.36,<3",
  "imageio-ffmpeg>=0.5,<1",
]
```

This records reproducibility only; packages remain installed exclusively in `.venv`.

- [ ] **Step 2: Write no-network and projection tests**

Create `tests/verification/test_uuv_tracking_coverage_runner.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from underwater_tracking.verification.uuv_tracking_coverage_runner import (
    NoNetworkLLM,
    project_audit_frame,
    run_once,
)


def test_no_network_llm_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="network LLM is disabled"):
        NoNetworkLLM().invoke_structured("strategy", {}, dict)


def test_projected_frame_pairs_truth_without_mutating_operational_input() -> None:
    operational = {
        "sim_time_s": 5,
        "uuvs": [],
        "tracks": [],
        "events": [],
        "waypoint_commands": {},
    }
    truth = {
        "sim_time_s": 5,
        "targets": [{"target_id": "target_00", "position_xy": [1.0, 2.0]}],
    }

    projected = project_audit_frame(
        operational,
        truth,
        mission_modes={"uuv_00": "ACTIVE_SCAN"},
        region_lifecycles={"R1": "ACTIVE_SCAN"},
    )

    assert "target_truth" not in operational
    assert projected["target_truth"] == truth["targets"]
    assert projected["mission_modes"] == {"uuv_00": "ACTIVE_SCAN"}


def test_two_step_runner_uses_repository_baseline_without_network(tmp_path: Path) -> None:
    result = run_once(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=2,
        work_dir=tmp_path / "run",
    )

    assert result["seed"] == 42
    assert len(result["frames"]) == 2
    assert result["routes"]
    assert result["regions"]
```

- [ ] **Step 3: Run the new tests and verify the runner is absent**

Run:

```powershell
$python = 'D:\Air\反Q\Underwater-Tracking\.venv\Scripts\python.exe'
Set-Location -LiteralPath 'D:\Air\反Q\Underwater-Tracking'
& $python -m pytest -q tests/verification/test_uuv_tracking_coverage_runner.py
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 4: Implement the runner's stable interfaces**

Create `src/underwater_tracking/verification/uuv_tracking_coverage_runner.py`; this first block defines orchestration and trace capture:

```python
"""Deterministic no-network runner for the multi-UUV audit."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
import json
from math import hypot, isfinite
from pathlib import Path
from typing import Any, cast

from underwater_tracking.cli import _AgentLoop, _mission_controller_for
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.verification.uuv_tracking_coverage_audit import (
    command_motion_counts,
    deterministic_trace_digest,
    minimum_pairwise_separation_m,
    percentile_summary,
    sampled_footprint_fraction,
    target_position_errors_m,
    waypoint_visit_fraction,
)

Point = tuple[float, float]


class NoNetworkLLM:
    """Sentinel client: baseline construction must never invoke an LLM."""

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        raise RuntimeError(
            f"network LLM is disabled for audit operation {operation!r}"
        )

    def cancel(self) -> None:
        return None

    def close(self) -> None:
        return None


def project_audit_frame(
    operational: Mapping[str, object],
    truth: Mapping[str, object],
    *,
    mission_modes: Mapping[str, str],
    region_lifecycles: Mapping[str, str],
) -> dict[str, object]:
    if operational.get("sim_time_s") != truth.get("sim_time_s"):
        raise ValueError("operational and truth frames must share sim_time_s")
    return {
        "sim_time_s": operational.get("sim_time_s"),
        "uuvs": operational.get("uuvs", []),
        "tracks": operational.get("tracks", []),
        "quality": operational.get("quality", []),
        "events": operational.get("events", []),
        "waypoint_commands": operational.get("waypoint_commands", {}),
        "target_truth": truth.get("targets", []),
        "mission_modes": dict(sorted(mission_modes.items())),
        "region_lifecycles": dict(sorted(region_lifecycles.items())),
    }


def _route_projection(
    engine: SimulationEngine,
) -> tuple[dict[str, object], dict[str, object]]:
    snapshot = engine.mission_snapshot()
    if snapshot is None:
        return {}, {}
    routes = {
        region.region_id: {
            uuv_id: [list(point) for point in route]
            for uuv_id, route in sorted(
                region.scan_waypoints_by_uuv.items()
            )
        }
        for region in snapshot.regions
    }
    regions = {
        region.region_id: {
            "target_id": region.target_id,
            "polygon": [list(point) for point in region.region_polygon],
            "active_scan_uuv_ids": list(region.active_scan_uuv_ids),
            "passive_track_uuv_ids": list(region.passive_track_uuv_ids),
        }
        for region in snapshot.regions
    }
    return routes, regions


def run_once(
    *,
    config_path: Path,
    seed: int,
    steps: int,
    work_dir: Path,
) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be positive")
    work_dir.mkdir(parents=True, exist_ok=False)
    config = load_app_config(config_path)
    controller = _mission_controller_for(config)
    if controller is None:
        raise RuntimeError("audit scenario requires a mission controller")
    truth_frames: list[dict[str, object]] = []
    loop = _AgentLoop(
        config,
        database_path=work_dir / "agent.db",
        llm={"master": NoNetworkLLM()},
        run_id=f"audit-seed-{seed}-steps-{steps}",
        steps=steps,
        seed=seed,
    )
    engine: SimulationEngine | None = None
    frames: list[dict[str, object]] = []
    failure: BaseException | None = None
    try:
        engine = SimulationEngine(
            config,
            seed=seed,
            output_dir=work_dir / "frames",
            evaluation_sink=truth_frames.append,
            mission_controller=controller,
            verification_audit=True,
        )
        loop.attach(engine)
        baseline = loop.install_deterministic_baseline(
            engine.publication_situation()
        )
        if baseline is None:
            raise RuntimeError("deterministic baseline was not installed")
        routes, regions = _route_projection(engine)
        active_ranges_m = {
            uuv_id: float(uuv.capability.active_range_m)
            for uuv_id, uuv in sorted(engine._uuvs.items())
        }
        for _ in range(steps):
            operational = engine.step()
            if not truth_frames:
                raise RuntimeError("evaluation sink produced no truth frame")
            snapshot = engine.mission_snapshot()
            modes = (
                {
                    key: value.value
                    for key, value in snapshot.uuv_modes.items()
                }
                if snapshot is not None
                else {}
            )
            lifecycles = (
                {
                    region.region_id: region.lifecycle.value
                    for region in snapshot.regions
                }
                if snapshot is not None
                else {}
            )
            frames.append(
                project_audit_frame(
                    operational,
                    truth_frames[-1],
                    mission_modes=modes,
                    region_lifecycles=lifecycles,
                )
            )
        physics = engine.verification_audit()
        evidence = engine.verification_evidence()
    except BaseException as error:
        failure = error
        raise
    finally:
        if engine is not None:
            engine.logger.close()
        close_ok = loop.close()
        if not close_ok and failure is None:
            raise RuntimeError("agent loop did not close cleanly")
    return {
        "schema_version": 1,
        "scenario": config.scenario.scenario_id,
        "seed": seed,
        "steps": steps,
        "physics_step_s": config.timing.physics_step_s,
        "routes": routes,
        "regions": regions,
        "active_ranges_m": active_ranges_m,
        "frames": frames,
        "physics_audit": physics,
        "verification_evidence": evidence,
    }
```

Continue the same module with the complete aggregation and CLI implementation:

```python
def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return cast(Mapping[str, object], value)


def _as_items(value: object) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(value)


def _point(value: object) -> Point | None:
    values = _as_items(value)
    if len(values) < 2:
        return None
    x, y = values[:2]
    if (
        not isinstance(x, (int, float))
        or isinstance(x, bool)
        or not isinstance(y, (int, float))
        or isinstance(y, bool)
    ):
        return None
    point = (float(x), float(y))
    return point if all(isfinite(component) for component in point) else None


def _frames(trace: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw_frames = _as_items(trace.get("frames"))
    if not raw_frames:
        raise ValueError("audit trace contains no frames")
    frames: list[Mapping[str, object]] = []
    for raw in raw_frames:
        if not isinstance(raw, Mapping):
            raise TypeError("every audit frame must be a mapping")
        frames.append(cast(Mapping[str, object], raw))
    return tuple(frames)


def _uuv_positions(frame: Mapping[str, object]) -> dict[str, Point]:
    positions: dict[str, Point] = {}
    for raw in _as_items(frame.get("uuvs")):
        item = _as_mapping(raw)
        uuv_id = item.get("platform_id")
        point = _point(item.get("position_xy"))
        if (
            isinstance(uuv_id, str)
            and point is not None
            and item.get("deployment_state") == "deployed"
        ):
            positions[uuv_id] = point
    return positions


def _uuv_trajectories(
    frames: Sequence[Mapping[str, object]],
) -> dict[str, tuple[Point, ...]]:
    trajectories: dict[str, list[Point]] = {}
    for frame in frames:
        for uuv_id, point in _uuv_positions(frame).items():
            trajectories.setdefault(uuv_id, []).append(point)
    return {
        uuv_id: tuple(points)
        for uuv_id, points in sorted(trajectories.items())
    }


def _polyline_length_m(points: Sequence[Point]) -> float:
    return sum(
        hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(points, points[1:])
    )


def _point_in_or_on_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        x1, y1 = start
        x2, y2 = end
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        on_segment = (
            abs(cross) <= 1.0e-8
            and min(x1, x2) - 1.0e-8 <= x <= max(x1, x2) + 1.0e-8
            and min(y1, y2) - 1.0e-8 <= y <= max(y1, y2) + 1.0e-8
        )
        if on_segment:
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


def _active_emissions_by_target(
    frames: Sequence[Mapping[str, object]],
    active_ranges_m: Mapping[str, object],
) -> dict[str, tuple[tuple[Point, float], ...]]:
    emissions: dict[str, list[tuple[Point, float]]] = {}
    for frame in frames:
        positions = _uuv_positions(frame)
        for raw_event in _as_items(frame.get("events")):
            event = _as_mapping(raw_event)
            if event.get("event_type") != "active_ping":
                continue
            payload = _as_mapping(event.get("payload"))
            emitter_id = payload.get("emitter_id")
            target_id = event.get("entity_id")
            radius = (
                active_ranges_m.get(emitter_id)
                if isinstance(emitter_id, str)
                else None
            )
            if (
                isinstance(emitter_id, str)
                and isinstance(target_id, str)
                and emitter_id in positions
                and isinstance(radius, (int, float))
                and not isinstance(radius, bool)
                and isfinite(float(radius))
                and float(radius) > 0.0
            ):
                emissions.setdefault(target_id, []).append(
                    (positions[emitter_id], float(radius))
                )
    return {
        target_id: tuple(values)
        for target_id, values in sorted(emissions.items())
    }


def _coverage_metrics(
    trace: Mapping[str, object],
    frames: Sequence[Mapping[str, object]],
    trajectories: Mapping[str, Sequence[Point]],
) -> tuple[dict[str, object], bool, int]:
    routes = _as_mapping(trace.get("routes"))
    regions = _as_mapping(trace.get("regions"))
    active_ranges = _as_mapping(trace.get("active_ranges_m"))
    emissions = _active_emissions_by_target(frames, active_ranges)
    coverage: dict[str, object] = {}
    geometry_valid = True
    route_count = 0
    for region_id, raw_region in sorted(regions.items()):
        region = _as_mapping(raw_region)
        polygon = tuple(
            point
            for raw_point in _as_items(region.get("polygon"))
            if (point := _point(raw_point)) is not None
        )
        by_uuv = _as_mapping(routes.get(region_id))
        route_visitation: dict[str, float | None] = {}
        lengths: dict[str, float] = {}
        for uuv_id, raw_route in sorted(by_uuv.items()):
            route = tuple(
                point
                for raw_point in _as_items(raw_route)
                if (point := _point(raw_point)) is not None
            )
            if not route:
                geometry_valid = False
                continue
            route_count += 1
            geometry_valid = geometry_valid and bool(polygon) and all(
                _point_in_or_on_polygon(point, polygon) for point in route
            )
            lengths[uuv_id] = _polyline_length_m(route)
            route_visitation[uuv_id] = waypoint_visit_fraction(
                trajectories.get(uuv_id, ()),
                route,
            )
        target_id = region.get("target_id")
        target_emissions = (
            emissions.get(target_id, ())
            if isinstance(target_id, str)
            else ()
        )
        footprint = (
            sampled_footprint_fraction(polygon, target_emissions)
            if polygon
            else None
        )
        length_values = tuple(lengths.values())
        coverage[region_id] = {
            "target_id": target_id,
            "assigned_route_count": len(lengths),
            "planned_route_length_m_by_uuv": lengths,
            "planned_route_load_span_m": (
                max(length_values) - min(length_values)
                if length_values
                else None
            ),
            "waypoint_visit_fraction_by_uuv": route_visitation,
            "active_emission_count": len(target_emissions),
            "sampled_active_sonar_footprint_fraction": footprint,
            "sampled_active_sonar_footprint_unavailable_reason": (
                None
                if footprint is not None
                else "no emitted active-sonar footprint was available"
            ),
        }
    if not regions:
        geometry_valid = False
    return coverage, geometry_valid, route_count


def _target_ids(
    frames: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    identifiers = {
        target_id
        for frame in frames
        for raw in _as_items(frame.get("target_truth"))
        if isinstance((target_id := _as_mapping(raw).get("target_id")), str)
    }
    return tuple(sorted(identifiers))


def _physics_violation_count(physics: Mapping[str, object]) -> int:
    count = 0
    for raw in _as_items(physics.get("audits")):
        audit = _as_mapping(raw)
        for field in (
            "limit_violation_count",
            "teleport_count",
            "boundary_violation_count",
        ):
            value = audit.get(field, 0)
            if isinstance(value, int) and not isinstance(value, bool):
                count += value
    return count


def _all_finite(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_finite(child) for child in value)
    return False


def summarize_trace(trace: Mapping[str, object]) -> dict[str, object]:
    frames = _frames(trace)
    target_ids = _target_ids(frames)
    tracking: dict[str, object] = {}
    estimate_available = bool(target_ids)
    for target_id in target_ids:
        errors = target_position_errors_m(frames, target_id)
        summary = percentile_summary(errors)
        estimate_available = estimate_available and summary is not None
        tracking[target_id] = {
            "sample_count": len(errors),
            "position_error_m": summary,
            "unavailable_reason": (
                None
                if summary is not None
                else "the shortened run produced no fused estimate for this target"
            ),
        }
    trajectories = _uuv_trajectories(frames)
    control = command_motion_counts(frames)
    coverage, geometry_valid, route_count = _coverage_metrics(
        trace,
        frames,
        trajectories,
    )
    physics = _as_mapping(trace.get("physics_audit"))
    violation_count = _physics_violation_count(physics)
    verification = _as_mapping(trace.get("verification_evidence"))
    public_observation_count = len(
        _as_items(verification.get("public_observation_ids"))
    )
    descriptive: dict[str, object] = {
        "tracking": tracking,
        "control_and_motion": {
            **control,
            "minimum_pairwise_separation_m": minimum_pairwise_separation_m(
                frames
            ),
            "trajectory_sample_count_by_uuv": {
                uuv_id: len(points)
                for uuv_id, points in trajectories.items()
            },
            "distance_travelled_m_by_uuv": {
                uuv_id: _polyline_length_m(points)
                for uuv_id, points in trajectories.items()
            },
        },
        "coverage": coverage,
        "physics_audit": dict(physics),
        "evidence": {
            "public_observation_count": public_observation_count,
        },
    }
    hard_checks = {
        "truth_targets_present": bool(target_ids),
        "fused_tracking_estimate_available": estimate_available,
        "assigned_routes_present": route_count > 0,
        "assigned_route_geometry_valid": geometry_valid,
        "commands_emitted": control["commanded_intervals"] > 0,
        "commanded_uuv_motion_observed": control["moved_intervals"] > 0,
        "configured_physics_invariants": violation_count == 0,
        "metrics_finite": _all_finite(descriptive),
    }
    return {
        "schema_version": 1,
        "scenario": trace.get("scenario"),
        "seed": trace.get("seed"),
        "steps": trace.get("steps"),
        **descriptive,
        "physics_violation_count": violation_count,
        "hard_checks": hard_checks,
        "status": "PASS" if all(hard_checks.values()) else "FAIL",
    }


def _write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    pretty: bool,
) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        stream.write("\n")


def run_audit(
    *,
    config_path: Path,
    seed: int,
    steps: int,
    repeat: int,
    work_dir: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    if repeat != 2:
        raise ValueError("final audit requires exactly two repeated runs")
    run_dirs = (work_dir / "run-a", work_dir / "run-b")
    for path in (*run_dirs, evidence_dir):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing path: {path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    first = run_once(
        config_path=config_path,
        seed=seed,
        steps=steps,
        work_dir=run_dirs[0],
    )
    second = run_once(
        config_path=config_path,
        seed=seed,
        steps=steps,
        work_dir=run_dirs[1],
    )
    first_digest = deterministic_trace_digest(first)
    second_digest = deterministic_trace_digest(second)
    metrics = summarize_trace(first)
    raw_checks = metrics.get("hard_checks")
    if not isinstance(raw_checks, dict):
        raise TypeError("summary hard_checks must be a mutable dictionary")
    hard_checks = cast(dict[str, bool], raw_checks)
    hard_checks["deterministic_repeat"] = first_digest == second_digest
    metrics["trace_digests"] = {
        "run-a": first_digest,
        "run-b": second_digest,
    }
    metrics["status"] = "PASS" if all(hard_checks.values()) else "FAIL"
    _write_json(evidence_dir / "trajectory.json", first, pretty=False)
    _write_json(evidence_dir / "metrics.json", metrics, pretty=True)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Run the deterministic multi-UUV tracking/coverage audit."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    metrics = run_audit(
        config_path=args.config,
        seed=args.seed,
        steps=args.steps,
        repeat=args.repeat,
        work_dir=args.work_dir,
        evidence_dir=args.evidence_dir,
    )
    status = metrics.get("status")
    print(
        json.dumps(
            {
                "status": status,
                "trace_digests": metrics.get("trace_digests"),
                "evidence_dir": str(args.evidence_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1
```

The hard gate deliberately treats “no fused estimate” as a failure of this acceptance task, but leaves RMSE, percentiles, route visitation, sampled footprint, load span, travelled distance, and separation as descriptive measurements without invented thresholds.

- [ ] **Step 5: Create the thin runner script**

Create `scripts/run_uuv_tracking_coverage_audit.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from underwater_tracking.verification.uuv_tracking_coverage_runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
```

The module `main` accepts exactly:

```text
--config PATH --seed INT --steps INT --repeat INT
--work-dir PATH --evidence-dir PATH
```

Reject final acceptance runs whose `--repeat` is not 2.

- [ ] **Step 6: Run runner tests and static checks**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$python = "$repo\.venv\Scripts\python.exe"
Set-Location -LiteralPath $repo
& $python -m pytest -q `
  tests/verification/test_uuv_tracking_coverage_audit.py `
  tests/verification/test_uuv_tracking_coverage_runner.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m ruff check `
  src/underwater_tracking/verification/uuv_tracking_coverage_audit.py `
  src/underwater_tracking/verification/uuv_tracking_coverage_runner.py `
  scripts/run_uuv_tracking_coverage_audit.py `
  tests/verification/test_uuv_tracking_coverage_audit.py `
  tests/verification/test_uuv_tracking_coverage_runner.py
```

Expected: all tests pass and the smoke run makes no network call.

### Task 6: Add rendering from the measured trace

**Files:**
- Create: `tests/verification/test_uuv_tracking_coverage_render.py`
- Create: `src/underwater_tracking/verification/uuv_tracking_coverage_render.py`
- Create: `scripts/render_uuv_tracking_coverage_audit.py`

- [ ] **Step 1: Write a keyframe smoke test**

Create `tests/verification/test_uuv_tracking_coverage_render.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from underwater_tracking.verification.uuv_tracking_coverage_render import (
    render_keyframes,
)


def test_render_keyframes_uses_saved_trace(tmp_path: Path) -> None:
    trace = {
        "regions": {
            "R1": {
                "polygon": [
                    [-10.0, -10.0],
                    [10.0, -10.0],
                    [10.0, 10.0],
                    [-10.0, 10.0],
                ],
                "target_id": "target_00",
                "active_scan_uuv_ids": ["uuv_00"],
                "passive_track_uuv_ids": ["uuv_01"],
            }
        },
        "routes": {
            "R1": {
                "uuv_00": [[-10.0, -5.0], [10.0, -5.0]],
                "uuv_01": [[10.0, 5.0], [-10.0, 5.0]],
            }
        },
        "frames": [
            {
                "sim_time_s": 0,
                "uuvs": [
                    {
                        "platform_id": "uuv_00",
                        "position_xy": [-10.0, -5.0],
                        "deployment_state": "deployed",
                    },
                    {
                        "platform_id": "uuv_01",
                        "position_xy": [10.0, 5.0],
                        "deployment_state": "deployed",
                    },
                ],
                "tracks": [
                    {
                        "target_id": "target_00",
                        "mean": [0.0, 1.0, 0.0, 0.0],
                    }
                ],
                "target_truth": [
                    {
                        "target_id": "target_00",
                        "position_xy": [0.0, 0.0],
                    }
                ],
                "waypoint_commands": {
                    "target_00": {"uuv_00": [10.0, -5.0]}
                },
                "mission_modes": {
                    "uuv_00": "ACTIVE_SCAN",
                    "uuv_01": "PASSIVE_TRACK",
                },
            }
        ],
    }
    trace_path = tmp_path / "trajectory.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")

    outputs = render_keyframes(trace_path, tmp_path)

    assert outputs["tracking"].is_file()
    assert outputs["coverage"].is_file()
    assert outputs["tracking"].stat().st_size > 0
    assert outputs["coverage"].stat().st_size > 0
```

- [ ] **Step 2: Run the test and verify the renderer is absent**

Run:

```powershell
$python = 'D:\Air\反Q\Underwater-Tracking\.venv\Scripts\python.exe'
Set-Location -LiteralPath 'D:\Air\反Q\Underwater-Tracking'
& $python -m pytest -q tests/verification/test_uuv_tracking_coverage_render.py
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the renderer**

Create `src/underwater_tracking/verification/uuv_tracking_coverage_render.py`:

```python
"""Headless media rendering for a persisted multi-UUV audit trace."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import imageio.v2 as imageio
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np

Point = tuple[float, float]
_PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#7F3C8D",
    "#11A579",
    "#3969AC",
    "#F2B701",
)


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return cast(Mapping[str, object], value)


def _as_items(value: object) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(value)


def _point(value: object) -> Point | None:
    values = _as_items(value)
    if len(values) < 2:
        return None
    x, y = values[:2]
    if (
        not isinstance(x, (int, float))
        or isinstance(x, bool)
        or not isinstance(y, (int, float))
        or isinstance(y, bool)
    ):
        return None
    return float(x), float(y)


def _load_trace(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("trajectory root must be a JSON object")
    trace = cast(dict[str, object], payload)
    frames = _as_items(trace.get("frames"))
    if not frames or any(not isinstance(frame, Mapping) for frame in frames):
        raise ValueError("trajectory must contain mapping frames")
    return trace


def _frames(trace: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        cast(Mapping[str, object], frame)
        for frame in _as_items(trace.get("frames"))
    )


def _uuv_ids(trace: Mapping[str, object]) -> tuple[str, ...]:
    identifiers = {
        uuv_id
        for frame in _frames(trace)
        for raw in _as_items(frame.get("uuvs"))
        if isinstance((uuv_id := _as_mapping(raw).get("platform_id")), str)
    }
    return tuple(sorted(identifiers))


def _colour_by_uuv(trace: Mapping[str, object]) -> dict[str, str]:
    return {
        uuv_id: _PALETTE[index % len(_PALETTE)]
        for index, uuv_id in enumerate(_uuv_ids(trace))
    }


def _positions_by_uuv(frame: Mapping[str, object]) -> dict[str, Point]:
    positions: dict[str, Point] = {}
    for raw in _as_items(frame.get("uuvs")):
        item = _as_mapping(raw)
        uuv_id = item.get("platform_id")
        point = _point(item.get("position_xy"))
        if isinstance(uuv_id, str) and point is not None:
            positions[uuv_id] = point
    return positions


def _trail(
    frames: Sequence[Mapping[str, object]],
    frame_index: int,
    *,
    collection: str,
    identifier_field: str,
    identifier: str,
    point_field: str,
) -> tuple[Point, ...]:
    points: list[Point] = []
    for frame in frames[: frame_index + 1]:
        for raw in _as_items(frame.get(collection)):
            item = _as_mapping(raw)
            if item.get(identifier_field) != identifier:
                continue
            point = _point(item.get(point_field))
            if point is not None:
                points.append(point)
            break
    return tuple(points)


def _estimate_trail(
    frames: Sequence[Mapping[str, object]],
    frame_index: int,
    target_id: str,
) -> tuple[Point, ...]:
    points: list[Point] = []
    for frame in frames[: frame_index + 1]:
        for raw in _as_items(frame.get("tracks")):
            item = _as_mapping(raw)
            if item.get("target_id") != target_id:
                continue
            point = _point(item.get("mean"))
            if point is not None:
                points.append(point)
            break
    return tuple(points)


def _all_plot_points(trace: Mapping[str, object]) -> tuple[Point, ...]:
    points: list[Point] = []
    for raw_region in _as_mapping(trace.get("regions")).values():
        region = _as_mapping(raw_region)
        points.extend(
            point
            for raw in _as_items(region.get("polygon"))
            if (point := _point(raw)) is not None
        )
    for frame in _frames(trace):
        points.extend(_positions_by_uuv(frame).values())
        for collection, field in (
            ("target_truth", "position_xy"),
            ("tracks", "mean"),
        ):
            points.extend(
                point
                for raw in _as_items(frame.get(collection))
                if (point := _point(_as_mapping(raw).get(field))) is not None
            )
    return tuple(points)


def _axis_bounds(trace: Mapping[str, object]) -> tuple[float, float, float, float]:
    points = _all_plot_points(trace)
    if not points:
        return -1.0, 1.0, -1.0, 1.0
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    span = max(max_x - min_x, max_y - min_y, 1.0)
    margin = 0.05 * span
    return min_x - margin, max_x + margin, min_y - margin, max_y + margin


def _target_ids(trace: Mapping[str, object]) -> tuple[str, ...]:
    identifiers = {
        target_id
        for frame in _frames(trace)
        for raw in _as_items(frame.get("target_truth"))
        if isinstance((target_id := _as_mapping(raw).get("target_id")), str)
    }
    return tuple(sorted(identifiers))


def _draw_regions_and_routes(
    figure_axes: Any,
    trace: Mapping[str, object],
    colours: Mapping[str, str],
    *,
    show_routes: bool,
) -> None:
    region_label_used = False
    route_label_used = False
    for region_id, raw_region in sorted(
        _as_mapping(trace.get("regions")).items()
    ):
        region = _as_mapping(raw_region)
        polygon = tuple(
            point
            for raw in _as_items(region.get("polygon"))
            if (point := _point(raw)) is not None
        )
        if polygon:
            closed = (*polygon, polygon[0])
            figure_axes.plot(
                [point[0] for point in closed],
                [point[1] for point in closed],
                color="#666666",
                linewidth=1.5,
                label="task region" if not region_label_used else None,
            )
            region_label_used = True
            centroid = (
                sum(point[0] for point in polygon) / len(polygon),
                sum(point[1] for point in polygon) / len(polygon),
            )
            figure_axes.annotate(region_id, centroid, color="#444444")
        if not show_routes:
            continue
        by_uuv = _as_mapping(
            _as_mapping(trace.get("routes")).get(region_id)
        )
        for uuv_id, raw_route in sorted(by_uuv.items()):
            route = tuple(
                point
                for raw in _as_items(raw_route)
                if (point := _point(raw)) is not None
            )
            if not route:
                continue
            figure_axes.plot(
                [point[0] for point in route],
                [point[1] for point in route],
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
                color=colours.get(uuv_id, "#333333"),
                label="assigned serpentine route" if not route_label_used else None,
            )
            route_label_used = True


def _draw_uuvs(
    figure_axes: Any,
    frames: Sequence[Mapping[str, object]],
    frame_index: int,
    colours: Mapping[str, str],
    *,
    show_modes: bool,
) -> None:
    frame = frames[frame_index]
    current = _positions_by_uuv(frame)
    modes = _as_mapping(frame.get("mission_modes"))
    for uuv_id, colour in colours.items():
        trail = _trail(
            frames,
            frame_index,
            collection="uuvs",
            identifier_field="platform_id",
            identifier=uuv_id,
            point_field="position_xy",
        )
        if trail:
            figure_axes.plot(
                [point[0] for point in trail],
                [point[1] for point in trail],
                color=colour,
                linewidth=1.7,
                label=f"{uuv_id} actual trail",
            )
        point = current.get(uuv_id)
        if point is not None:
            figure_axes.scatter(
                [point[0]],
                [point[1]],
                color=colour,
                marker="o",
                s=35,
                zorder=5,
            )
            if show_modes:
                figure_axes.annotate(
                    f"{uuv_id}\n{modes.get(uuv_id, 'UNKNOWN')}",
                    point,
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=colour,
                )


def _draw_tracking(
    figure_axes: Any,
    trace: Mapping[str, object],
    frames: Sequence[Mapping[str, object]],
    frame_index: int,
) -> None:
    for target_id in _target_ids(trace):
        truth = _trail(
            frames,
            frame_index,
            collection="target_truth",
            identifier_field="target_id",
            identifier=target_id,
            point_field="position_xy",
        )
        estimate = _estimate_trail(frames, frame_index, target_id)
        if truth:
            figure_axes.plot(
                [point[0] for point in truth],
                [point[1] for point in truth],
                color="#000000",
                linewidth=2.2,
                label=f"{target_id} evaluation truth",
            )
            figure_axes.scatter(
                [truth[-1][0]],
                [truth[-1][1]],
                color="#000000",
                marker="*",
                s=90,
                zorder=6,
            )
        if estimate:
            figure_axes.plot(
                [point[0] for point in estimate],
                [point[1] for point in estimate],
                color="#D55E00",
                linewidth=2.0,
                label=f"{target_id} fused estimate",
            )
            figure_axes.scatter(
                [estimate[-1][0]],
                [estimate[-1][1]],
                color="#D55E00",
                marker="x",
                s=55,
                zorder=6,
            )
    positions = _positions_by_uuv(frames[frame_index])
    command_label_used = False
    for raw_by_target in _as_mapping(
        frames[frame_index].get("waypoint_commands")
    ).values():
        for uuv_id, raw_point in _as_mapping(raw_by_target).items():
            start = positions.get(uuv_id)
            destination = _point(raw_point)
            if start is None or destination is None:
                continue
            figure_axes.plot(
                [start[0], destination[0]],
                [start[1], destination[1]],
                color="#7F3C8D",
                linewidth=1.0,
                alpha=0.6,
                label="current waypoint command" if not command_label_used else None,
            )
            command_label_used = True


def _deduplicated_legend(figure_axes: Any) -> None:
    handles, labels = figure_axes.get_legend_handles_labels()
    unique = {
        label: handle
        for handle, label in zip(handles, labels)
        if label
    }
    if unique:
        figure_axes.legend(
            unique.values(),
            unique.keys(),
            loc="best",
            fontsize=7,
            framealpha=0.9,
        )


def _draw_frame(
    trace: Mapping[str, object],
    frame_index: int,
    *,
    view: str,
) -> Figure:
    frames = _frames(trace)
    if not 0 <= frame_index < len(frames):
        raise IndexError("frame index is outside the trace")
    if view not in {"tracking", "coverage"}:
        raise ValueError(f"unsupported render view: {view}")
    colours = _colour_by_uuv(trace)
    figure, figure_axes = plt.subplots(figsize=(9.6, 7.2), dpi=100)
    _draw_regions_and_routes(
        figure_axes,
        trace,
        colours,
        show_routes=view == "coverage",
    )
    _draw_uuvs(
        figure_axes,
        frames,
        frame_index,
        colours,
        show_modes=view == "coverage",
    )
    if view == "tracking":
        _draw_tracking(figure_axes, trace, frames, frame_index)
    min_x, max_x, min_y, max_y = _axis_bounds(trace)
    figure_axes.set_xlim(min_x, max_x)
    figure_axes.set_ylim(min_y, max_y)
    figure_axes.set_aspect("equal", adjustable="box")
    figure_axes.grid(True, alpha=0.25)
    figure_axes.set_xlabel("x (m)")
    figure_axes.set_ylabel("y (m)")
    sim_time_s = frames[frame_index].get("sim_time_s")
    title = (
        "Multi-UUV tracking/control"
        if view == "tracking"
        else "Multi-UUV serpentine coverage"
    )
    figure_axes.set_title(f"{title} — t={sim_time_s} s")
    _deduplicated_legend(figure_axes)
    figure.tight_layout()
    return figure


def _figure_rgb(figure: Figure) -> np.ndarray[Any, Any]:
    figure.canvas.draw()
    canvas = cast(Any, figure.canvas)
    rgba = np.asarray(canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, :3])


def _coverage_frame_index(trace: Mapping[str, object]) -> int:
    frames = _frames(trace)
    candidates = [
        index
        for index, frame in enumerate(frames)
        if "ACTIVE_SCAN" in _as_mapping(frame.get("mission_modes")).values()
    ]
    return candidates[-1] if candidates else 0


def _tracking_frame_index(trace: Mapping[str, object]) -> int:
    frames = _frames(trace)
    candidates = [
        index
        for index, frame in enumerate(frames)
        if _as_items(frame.get("tracks"))
    ]
    return candidates[-1] if candidates else len(frames) - 1


def _write_png(
    trace: Mapping[str, object],
    frame_index: int,
    *,
    view: str,
    path: Path,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing media: {path}")
    figure = _draw_frame(trace, frame_index, view=view)
    try:
        figure.savefig(path, format="png", dpi=150)
    finally:
        plt.close(figure)


def render_keyframes(
    trace_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Render tracking and coverage PNGs from one persisted trace."""
    trace = _load_trace(trace_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tracking": output_dir / "tracking-keyframe.png",
        "coverage": output_dir / "coverage-keyframe.png",
    }
    _write_png(
        trace,
        _tracking_frame_index(trace),
        view="tracking",
        path=outputs["tracking"],
    )
    _write_png(
        trace,
        _coverage_frame_index(trace),
        view="coverage",
        path=outputs["coverage"],
    )
    return outputs


def _animation_indices(frame_count: int, frame_stride: int) -> tuple[int, ...]:
    indices = list(range(0, frame_count, frame_stride))
    if not indices or indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return tuple(indices)


def _write_animation_pair(
    trace: Mapping[str, object],
    outputs: Mapping[str, Path],
    *,
    fps: int,
    frame_stride: int,
    writer_kwargs: Mapping[str, object],
) -> None:
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing media: {path}")
    writers: dict[str, Any] = {}
    try:
        writers = {
            view: imageio.get_writer(path, **writer_kwargs)
            for view, path in outputs.items()
        }
        for frame_index in _animation_indices(
            len(_frames(trace)),
            frame_stride,
        ):
            for view, writer in writers.items():
                figure = _draw_frame(trace, frame_index, view=view)
                try:
                    writer.append_data(_figure_rgb(figure))
                finally:
                    plt.close(figure)
    finally:
        for writer in writers.values():
            writer.close()


def render_videos(
    trace_path: Path,
    output_dir: Path,
    *,
    fps: int = 10,
    frame_stride: int = 3,
) -> dict[str, Path]:
    """Render two MP4 files; create new-name GIFs and raise on encoding failure."""
    if fps < 1 or frame_stride < 1:
        raise ValueError("fps and frame_stride must be positive")
    trace = _load_trace(trace_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tracking": output_dir / "tracking-control.mp4",
        "coverage": output_dir / "coverage-search.mp4",
    }
    try:
        _write_animation_pair(
            trace,
            outputs,
            fps=fps,
            frame_stride=frame_stride,
            writer_kwargs={
                "codec": "libx264",
                "fps": fps,
                "quality": 8,
            },
        )
    except Exception as encoder_error:
        fallback = {
            "tracking": output_dir / "tracking-control-fallback.gif",
            "coverage": output_dir / "coverage-search-fallback.gif",
        }
        try:
            _write_animation_pair(
                trace,
                fallback,
                fps=fps,
                frame_stride=frame_stride,
                writer_kwargs={
                    "mode": "I",
                    "duration": 1000.0 / fps,
                    "loop": 0,
                },
            )
        except Exception as fallback_error:
            raise RuntimeError(
                "MP4 encoding and new-name GIF fallback both failed; "
                "partial files were preserved"
            ) from fallback_error
        raise RuntimeError(
            "MP4 encoding failed; new-name GIF fallbacks were written to "
            f"{fallback['tracking']} and {fallback['coverage']}; "
            "partial MP4 files were preserved"
        ) from encoder_error
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Render media from a persisted multi-UUV audit trace."
    )
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=3)
    args = parser.parse_args(argv)
    keyframes = render_keyframes(args.trace, args.output_dir)
    videos = render_videos(
        args.trace,
        args.output_dir,
        fps=args.fps,
        frame_stride=args.frame_stride,
    )
    print(
        json.dumps(
            {
                "keyframes": {
                    key: str(path) for key, path in keyframes.items()
                },
                "videos": {
                    key: str(path) for key, path in videos.items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0
```

This implementation loads `trajectory.json` once per public call, uses only trace-contained routes/states/commands/truth, fixes map bounds for every frame, keeps evaluation truth explicitly labelled, converts each Agg canvas directly to RGB memory, and never writes raw frame images. Every figure and writer closes in `finally`. A failed MP4 encode preserves partial files, writes differently named GIF fallbacks when possible, and raises a clear error so the report cannot mislabel a fallback as MP4 success.

- [ ] **Step 4: Create the thin renderer script**

Create `scripts/render_uuv_tracking_coverage_audit.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from underwater_tracking.verification.uuv_tracking_coverage_render import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run renderer tests and static checks**

Run:

```powershell
$python = 'D:\Air\反Q\Underwater-Tracking\.venv\Scripts\python.exe'
Set-Location -LiteralPath 'D:\Air\反Q\Underwater-Tracking'
& $python -m pytest -q tests/verification/test_uuv_tracking_coverage_render.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m ruff check `
  src/underwater_tracking/verification/uuv_tracking_coverage_render.py `
  scripts/render_uuv_tracking_coverage_audit.py `
  tests/verification/test_uuv_tracking_coverage_render.py
```

Expected: the PNG smoke test passes and Ruff reports no issues.

- [ ] **Step 6: Commit audit tooling**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
git -C $repo diff --check
git -C $repo add -- `
  pyproject.toml `
  src/underwater_tracking/verification/__init__.py `
  src/underwater_tracking/verification/uuv_tracking_coverage_audit.py `
  src/underwater_tracking/verification/uuv_tracking_coverage_runner.py `
  src/underwater_tracking/verification/uuv_tracking_coverage_render.py `
  scripts/run_uuv_tracking_coverage_audit.py `
  scripts/render_uuv_tracking_coverage_audit.py `
  tests/verification/test_uuv_tracking_coverage_audit.py `
  tests/verification/test_uuv_tracking_coverage_runner.py `
  tests/verification/test_uuv_tracking_coverage_render.py
git -C $repo diff --cached --check
git -C $repo commit -m 'test: add deterministic UUV audit evidence'
```

Expected: only source, tests, scripts, and `pyproject.toml` are committed.

### Task 7: Run the fixed-seed audit and apply the evidence gate

**Files:**
- Create ignored: `outputs/audit-20260827/run-a/`
- Create ignored: `outputs/audit-20260827/run-b/`
- Create: `docs/verification/2026-08-27-uuv-tracking-coverage/metrics.json`
- Create: `docs/verification/2026-08-27-uuv-tracking-coverage/trajectory.json`

- [ ] **Step 1: Run two complete 360-step traces**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$python = "$repo\.venv\Scripts\python.exe"
Set-Location -LiteralPath $repo
& $python scripts/run_uuv_tracking_coverage_audit.py `
  --config configs/scenario/uuv_only_single_target.yaml `
  --seed 42 `
  --steps 360 `
  --repeat 2 `
  --work-dir outputs/audit-20260827 `
  --evidence-dir docs/verification/2026-08-27-uuv-tracking-coverage |
  Tee-Object -FilePath outputs/audit-20260827/logs/audit-run.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected:

- two deterministic run directories exist;
- `metrics.json` reports equal trace digests;
- `trajectory.json` contains exactly 360 projected frames from run A;
- no network LLM method was called;
- zero configured physics-invariant violations;
- nonempty planned routes, UUV commands, and subsequent physical motion.

- [ ] **Step 2: Inspect hard gates and descriptive metrics**

Run:

```powershell
$metrics = Get-Content -LiteralPath 'D:\Air\反Q\Underwater-Tracking\docs\verification\2026-08-27-uuv-tracking-coverage\metrics.json' -Raw |
  ConvertFrom-Json
$metrics | ConvertTo-Json -Depth 12
```

Classify results:

- **PASS:** determinism, finite/PSD state, configured physics limits, route geometry, and command-to-motion evidence all pass.
- **MEASURED:** RMSE/p95/max, route visitation, sampled footprint, load balance, observation count, and movement counts have no invented thresholds.
- **UNAVAILABLE:** a metric lacks a source-backed model or the shortened native run does not expose it; include the exact reason.
- **FAIL:** a configured or mathematical invariant is violated.

If tracking produces no fused estimate, or a tracking invariant fails, stop before rendering final success media. Invoke `systematic-debugging`, reduce the failure to a deterministic test, update this plan with the exact tracking repair, obtain any governance authorization required by the new scope, and only then modify tracking code. Do not report the coverage fix as completion of the tracking requirement.

- [ ] **Step 3: Verify the evaluation boundary statically**

Run:

```powershell
$python = 'D:\Air\反Q\Underwater-Tracking\.venv\Scripts\python.exe'
Set-Location -LiteralPath 'D:\Air\反Q\Underwater-Tracking'
& $python -m pytest -q `
  tests/domain/test_truth_boundary.py `
  tests/property/test_foundation_invariants.py::test_no_operational_package_imports_target_truth `
  tests/integration/test_headless_loop.py::test_engine_exposes_sink_truth_only_through_callback
```

Expected: all three truth-boundary checks pass.

### Task 8: Render current evidence and write the Chinese audit report

**Files:**
- Create: `docs/verification/2026-08-27-uuv-tracking-coverage/tracking-control.mp4`
- Create: `docs/verification/2026-08-27-uuv-tracking-coverage/coverage-search.mp4`
- Create: `docs/verification/2026-08-27-uuv-tracking-coverage/tracking-keyframe.png`
- Create: `docs/verification/2026-08-27-uuv-tracking-coverage/coverage-keyframe.png`
- Create: `docs/verification/2026-08-27-uuv-tracking-coverage/README.md`

- [ ] **Step 1: Render videos and keyframes from `trajectory.json`**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$python = "$repo\.venv\Scripts\python.exe"
Set-Location -LiteralPath $repo
& $python scripts/render_uuv_tracking_coverage_audit.py `
  --trace docs/verification/2026-08-27-uuv-tracking-coverage/trajectory.json `
  --output-dir docs/verification/2026-08-27-uuv-tracking-coverage `
  --fps 10 `
  --frame-stride 3 |
  Tee-Object -FilePath outputs/audit-20260827/logs/render.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected: two nonempty MP4 files and two PNG files. If MP4 encoding fails, create `tracking-control.gif` and `coverage-search.gif` in addition to the PNGs and record the encoder failure; do not claim MP4 success.

- [ ] **Step 2: Inspect media metadata and sizes**

Run:

```powershell
$evidence = 'D:\Air\反Q\Underwater-Tracking\docs\verification\2026-08-27-uuv-tracking-coverage'
Get-ChildItem -LiteralPath $evidence |
  Select-Object Name,Length,LastWriteTime
```

Expected: committed media stays compact. If a video exceeds 25 MiB, rerender to a new filename with a larger frame stride; do not delete the original without user action.

- [ ] **Step 3: Visually inspect both keyframes**

Open:

- `tracking-keyframe.png`
- `coverage-keyframe.png`

Confirm:

- legends distinguish planned routes, actual trajectories, estimate, and evaluation truth;
- labels are readable and not clipped;
- region and map scales are not distorted;
- UUV colors are stable between coverage and tracking views;
- the shown timestamp exists in `trajectory.json`.

- [ ] **Step 4: Write the Chinese report from current evidence**

Create `docs/verification/2026-08-27-uuv-tracking-coverage/README.md` with:

1. 验收范围与版本
2. 当前源码调用链
3. 多 UUV 协同跟踪结论
4. 多 UUV 蛇形覆盖结论
5. 已确认缺陷与最小修复
6. 测试、仿真命令与环境版本
7. 指标表及证据等级
8. 视频与关键截图
9. 未验证项、限制和风险
10. 复现步骤
11. Git 提交与分支状态

Copy numeric values only from `metrics.json` and current command logs. State explicitly that two identical fixed-seed runs demonstrate reproducibility for this scenario but not multi-seed statistical robustness.

### Task 9: Fresh verification, evidence commit, and branch push

**Files:**
- Verify all modified source and tests
- Commit only `docs/verification/2026-08-27-uuv-tracking-coverage/` in the evidence commit

- [ ] **Step 1: Invoke `verification-before-completion` and run focused verification**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$python = "$repo\.venv\Scripts\python.exe"
Set-Location -LiteralPath $repo
& $python -m pytest -q `
  tests/planning/test_coverage_paths.py `
  tests/tracking/test_imm_uif.py `
  tests/simulation/test_mission_waypoint_geometry.py `
  tests/simulation/test_uuv_only_carrier_group.py::test_normal_mode_routes_active_uuvs_and_emits_region_scan_ping `
  tests/simulation/test_uuv_only_carrier_group.py::test_normal_mode_routes_all_region_members_before_target_entry `
  tests/verification/test_uuv_tracking_coverage_audit.py `
  tests/verification/test_uuv_tracking_coverage_runner.py `
  tests/verification/test_uuv_tracking_coverage_render.py `
  tests/domain/test_truth_boundary.py
```

Expected: exit code 0.

- [ ] **Step 2: Run the complete non-opt-in Python suite**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$python = "$repo\.venv\Scripts\python.exe"
Set-Location -LiteralPath $repo
& $python -m pytest -q -m 'not real_llm and not long_running and not live_acceptance' |
  Tee-Object -FilePath outputs/audit-20260827/logs/final-suite.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected: exit code 0 with a current pass count recorded in the report.

- [ ] **Step 3: Run static checks**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$python = "$repo\.venv\Scripts\python.exe"
Set-Location -LiteralPath $repo
& $python -m ruff check `
  src/underwater_tracking/simulation/engine.py `
  src/underwater_tracking/verification `
  scripts/run_uuv_tracking_coverage_audit.py `
  scripts/render_uuv_tracking_coverage_audit.py `
  tests/verification
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m mypy `
  src/underwater_tracking/verification/uuv_tracking_coverage_audit.py `
  src/underwater_tracking/verification/uuv_tracking_coverage_runner.py `
  src/underwater_tracking/verification/uuv_tracking_coverage_render.py
```

Expected: Ruff and MyPy exit 0. Existing unrelated MyPy failures outside these files are not hidden or folded into this result.

- [ ] **Step 4: Validate JSON, media, and Git scope**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$python = "$repo\.venv\Scripts\python.exe"
$evidence = "$repo\docs\verification\2026-08-27-uuv-tracking-coverage"
& $python -m json.tool "$evidence\metrics.json" | Out-Null
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m json.tool "$evidence\trajectory.json" | Out-Null
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Get-ChildItem -LiteralPath $evidence | Select-Object Name,Length
git -C $repo status --short
git -C $repo diff --check
```

Expected: JSON validation succeeds, required media is nonempty, and no `.venv`, `outputs`, cache, raw frames, database, or unrelated file appears in Git status.

- [ ] **Step 5: Commit final evidence**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$evidence = 'docs/verification/2026-08-27-uuv-tracking-coverage'
git -C $repo add -- $evidence
$staged = git -C $repo diff --cached --name-only
$unexpected = $staged | Where-Object { $_ -notlike "$evidence/*" }
if ($unexpected) { $unexpected; exit 5 }
git -C $repo diff --cached --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git -C $repo commit -m 'docs: record UUV tracking and coverage evidence'
```

Expected: one evidence commit containing report, JSON, and compact media only.

- [ ] **Step 6: Verify final branch history before push**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
git -C $repo status --short --branch
git -C $repo log --oneline --decorate origin/master..HEAD
git -C $repo diff --stat origin/master...HEAD
```

Expected: worktree clean, only review-branch commits are listed, and no merge commit exists.

- [ ] **Step 7: Push only the review branch**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
git -C $repo push --set-upstream origin review/uuv-tracking-coverage-20260827
```

If Git requests interactive authentication unavailable through the configured credential helper, stop and ask the user to authenticate. Do not change Git credential configuration.

- [ ] **Step 8: Verify the remote branch without merging**

Run:

```powershell
$repo = 'D:\Air\反Q\Underwater-Tracking'
$local = git -C $repo rev-parse HEAD
$remote = git -C $repo ls-remote --heads origin review/uuv-tracking-coverage-20260827
$local
$remote
git -C $repo status --short --branch
```

Expected: the remote hash equals local `HEAD`, the branch tracks `origin/review/uuv-tracking-coverage-20260827`, and `master` remains untouched.

## 4. Completion report requirements

The final response must state:

- branch name and remote hash;
- every created or modified source/test/evidence file;
- focused and full-suite current results;
- fixed-seed configuration and both deterministic digests;
- tracking measurements and whether control commands produced physical motion;
- coverage geometry, actual route visitation, sonar-footprint availability, and limitations;
- configured physics/safety invariant results;
- links to both videos and both keyframes;
- the confirmed defect and exact minimal correction;
- unverified items and why they remain unverified;
- that the two ROS workspaces, Conda environments, system configuration, `master`, and existing media were not modified.

Do not claim general robustness, superiority over another controller, or statistical significance from this same-seed audit.
