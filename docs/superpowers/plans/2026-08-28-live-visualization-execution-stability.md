# Live Visualization and Execution Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a real `main.py` run produce bounded target motion, honest and usable prediction uncertainty, four executable task regions, visible sensor ranges, and stable desktop/mobile map framing, with semantic evidence proving the result comes from the real backend rather than mocked browser data.

**Architecture:** Insert explicit health gates between target motion, prediction, planning, execution, transport, and rendering. A deterministic four-region baseline is committed before optional LLM optimization; one atomic `OperationalFrame` carries matching prediction and execution revisions to every transport. The UI renders that authoritative state without inventing geometry and uses a map-clamped semantic camera. A real owned-process acceptance runner records backend, transport, browser, and screenshot evidence at fixed simulation checkpoints.

**Tech Stack:** Python 3.11, Pydantic 2, NumPy/SciPy, FastAPI/WebSocket/JSONL, pytest/Hypothesis, React 18, TypeScript 5.9, Canvas 2D, Vitest/Testing Library, Playwright.

## Global Constraints

- The approved design in `docs/superpowers/specs/2026-08-28-live-visualization-execution-stability-design.md` is the source of truth. Stop and amend the design before changing any listed invariant.
- Preserve the estimator-safe boundary: no simulator truth history may enter prediction, planning, `OperationalFrame`, HTTP, WebSocket, JSONL, replay, or browser state.
- A live frame may never combine a prediction from one revision with regions or task groups from another revision.
- Exactly four stable region slots (`<target_id>:task:01` through `:04`) and four two-UUV task groups are required whenever execution health is `current` or `degraded`.
- LLM output may adjust semantic policy only. It may not originate or mutate prediction points, region geometry, time-window topology, physical routes, platform membership, or sensor geometry.
- The frontend may style confidence and health but may not enlarge, clip, smooth, or otherwise rewrite backend prediction geometry.
- Legacy replay remains readable. Missing health fields normalize to `legacy_unknown`; new live frames must contain complete health fields.
- Do not update the historical `outputs/imm-confidence-trajectory-effect.png` artifact. It remains a visual reference, not an acceptance oracle.
- Do not use route interception, fake WebSockets, fixture frames, or hard-coded coordinates in real-run acceptance.
- Run the focused red test before implementation in every task, then the focused green test, then the listed regression set.

---

### Task 1: Add Prediction Health Configuration and Domain Contracts

**Files:**
- Modify: `src/underwater_tracking/config/models.py`
- Modify: `configs/tracking.yaml`
- Modify: `src/underwater_tracking/domain/execution_models.py`
- Create: `src/underwater_tracking/domain/prediction_models.py`
- Test: `tests/config/test_models.py`
- Create: `tests/domain/test_prediction_health.py`

- [ ] **Step 1: Write failing configuration tests**

Add tests proving defaults load from the real YAML and invalid thresholds are rejected:

```python
def test_tracking_config_loads_prediction_health_thresholds() -> None:
    config = load_app_config(CONFIG_PATH)

    health = config.tracking.prediction_health
    assert health.refresh_interval_s == 450
    assert health.hard_stale_s == 900
    assert health.max_clipped_point_fraction == 0.20
    assert health.max_corridor_radius_m == 6_000
    assert health.max_corridor_map_fraction == 0.25
    assert health.minimum_point_confidence == 0.02
    assert health.coordinate_tolerance_m == 0.000001
    assert health.boundary_recovery_timeout_s == 300


def test_prediction_health_rejects_stale_window_before_refresh_window() -> None:
    with pytest.raises(ValueError, match="hard_stale_s"):
        PredictionHealthConfig(refresh_interval_s=450, hard_stale_s=449)
```

- [ ] **Step 2: Run the configuration tests and confirm they fail**

Run:

```powershell
python -m pytest tests/config/test_models.py -q
```

Expected: FAIL because `PredictionHealthConfig` and `tracking.prediction_health` do not exist.

- [ ] **Step 3: Add the strict configuration model and YAML values**

Implement in `config/models.py`:

```python
class PredictionHealthConfig(StrictModel):
    refresh_interval_s: int = Field(default=450, gt=0)
    hard_stale_s: int = Field(default=900, gt=0)
    max_clipped_point_fraction: float = Field(default=0.20, ge=0, le=1)
    max_corridor_radius_m: float = Field(default=6_000.0, gt=0)
    max_corridor_map_fraction: float = Field(default=0.25, gt=0, le=1)
    minimum_point_confidence: float = Field(default=0.02, ge=0, le=1)
    coordinate_tolerance_m: float = Field(default=0.000001, gt=0)
    boundary_recovery_timeout_s: int = Field(default=300, gt=0)

    @model_validator(mode="after")
    def validate_windows(self) -> PredictionHealthConfig:
        if self.hard_stale_s < self.refresh_interval_s:
            raise ValueError("hard_stale_s must be >= refresh_interval_s")
        return self
```

Add `prediction_health: PredictionHealthConfig = Field(default_factory=PredictionHealthConfig)` to `TrackingConfig` and add the approved values under `tracking.prediction_health` in `configs/tracking.yaml`.

- [ ] **Step 4: Write failing domain-contract tests**

Create tests for strict enum values, bounded metrics, and unavailable predictions:

```python
def test_prediction_health_accepts_a_valid_imm_result() -> None:
    health = PredictionHealth(
        status="valid",
        regime="imm",
        reason_codes=(),
        source_track_age_s=10.0,
        clipped_point_fraction=0.0,
        maximum_radius_m=900.0,
        raw_prediction_id="prediction-7",
    )
    assert health.status == "valid"


def test_accepted_prediction_requires_payload_unless_unavailable() -> None:
    health = PredictionHealth(
        status="valid",
        regime="imm",
        reason_codes=(),
        source_track_age_s=10.0,
        clipped_point_fraction=0.0,
        maximum_radius_m=900.0,
        raw_prediction_id="prediction-7",
    )
    with pytest.raises(ValueError, match="valid prediction requires"):
        AcceptedPrediction(prediction=None, health=health)
```

- [ ] **Step 5: Run the domain tests and confirm they fail**

Run:

```powershell
python -m pytest tests/domain/test_prediction_health.py -q
```

Expected: FAIL because the health contracts do not exist.

- [ ] **Step 6: Add prediction health contracts without creating a domain import cycle**

Create `domain/prediction_models.py`. It imports `PredictedTrackRef` from
`agent_models.py` and `StrictModel` from `models.py`; `execution_models.py`
must not import `agent_models.py`, because `agent_models.py` already imports
`IMMModelForecast` in the opposite direction. Add these types in the new file:

```python
PredictionHealthStatus = Literal["valid", "degraded", "unavailable"]
AcceptedPredictionRegime = Literal[
    "imm", "bspline", "short_history", "boundary_recovery"
]


class PredictionHealth(StrictModel):
    status: PredictionHealthStatus
    regime: AcceptedPredictionRegime
    reason_codes: tuple[str, ...] = ()
    source_track_age_s: NonNegativeFloat
    clipped_point_fraction: UnitFloat
    maximum_radius_m: NonNegativeFloat
    raw_prediction_id: str | None = None


class AcceptedPrediction(StrictModel):
    prediction: PredictedTrackRef | None = None
    health: PredictionHealth

    @model_validator(mode="after")
    def validate_payload(self) -> AcceptedPrediction:
        if self.health.status == "unavailable" and self.prediction is not None:
            raise ValueError("unavailable prediction cannot carry a payload")
        if self.health.status != "unavailable" and self.prediction is None:
            raise ValueError("valid prediction requires a payload")
        return self
```

Broaden the existing execution-layer `PredictionRegime` literal to include
`"boundary_recovery"`, and make `_as_imm_prediction` preserve `imm`, `bspline`,
`short_history`, or `boundary_recovery` instead of collapsing every fallback
to `short_history`. Do not move or duplicate `PredictedTrackRef`.

- [ ] **Step 7: Run focused tests and static checks**

Run:

```powershell
python -m pytest tests/config/test_models.py tests/domain/test_prediction_health.py -q
python -m ruff check src/underwater_tracking/config/models.py src/underwater_tracking/domain/execution_models.py src/underwater_tracking/domain/prediction_models.py tests/config/test_models.py tests/domain/test_prediction_health.py
python -m mypy src/underwater_tracking/config/models.py src/underwater_tracking/domain/execution_models.py src/underwater_tracking/domain/prediction_models.py
```

Expected: all commands PASS.

- [ ] **Step 8: Commit Task 1**

```powershell
git add configs/tracking.yaml src/underwater_tracking/config/models.py src/underwater_tracking/domain/execution_models.py src/underwater_tracking/domain/prediction_models.py tests/config/test_models.py tests/domain/test_prediction_health.py
git commit -m "feat: define prediction health contracts"
```

---

### Task 2: Replace Boundary Freezing with a Recoverable Target Navigation State Machine

**Files:**
- Modify: `src/underwater_tracking/simulation/kinematics.py`
- Modify: `src/underwater_tracking/simulation/target.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Create: `tests/simulation/test_target_boundary_recovery.py`
- Modify: `tests/simulation/test_kinematics.py`
- Modify: `tests/simulation/test_engine.py`

- [ ] **Step 1: Write failing recovery-state tests**

Cover deceleration, turning, recovery, legal positions, zero-speed heading retention, and timeout:

```python
def test_target_recovers_from_an_outward_boundary_approach() -> None:
    target = TargetEntity(
        target_id="T1",
        position_xy=(9_700.0, 0.0),
        velocity_xy=(12.0, 0.0),
        intent=HiddenIntent.TRANSIT,
        bounds_xy=(-10_000.0, 10_000.0, -10_000.0, 10_000.0),
        max_acceleration_mps2=0.5,
        max_deceleration_mps2=0.8,
        max_turn_rate_rad_s=0.05,
    )

    states = []
    positions = []
    for _ in range(240):
        target.step(1.0, random.Random(9))
        states.append(target.navigation_state)
        positions.append(target.position_xy)
        if target.navigation_state == "NORMAL" and "BOUNDARY_RECOVERING" in states:
            break

    assert "BOUNDARY_DECELERATING" in states
    assert "BOUNDARY_TURNING" in states
    assert "BOUNDARY_RECOVERING" in states
    assert states[-1] == "NORMAL"
    assert all(-10_000 <= x <= 10_000 and -10_000 <= y <= 10_000 for x, y in positions)
    assert target.last_navigation_error is None


def test_boundary_recovery_times_out_explicitly() -> None:
    target = boundary_target(recovery_timeout_s=2)
    for _ in range(4):
        target.step(1.0, random.Random(1))
    assert target.navigation_state == "FAILED"
    assert target.last_navigation_error == "boundary_recovery_timeout"
```

- [ ] **Step 2: Run the recovery tests and confirm they fail**

Run:

```powershell
python -m pytest tests/simulation/test_target_boundary_recovery.py -q
```

Expected: FAIL because the navigation state and recovery fields are absent.

- [ ] **Step 3: Add guard geometry helpers**

In `kinematics.py`, add pure helpers and test them independently:

```python
def stopping_distance_m(speed_mps: float, deceleration_mps2: float) -> float:
    if deceleration_mps2 <= 0:
        raise ValueError("deceleration_mps2 must be positive")
    return speed_mps * speed_mps / (2.0 * deceleration_mps2)


def minimum_turn_radius_m(speed_mps: float, turn_rate_rad_s: float) -> float:
    if turn_rate_rad_s <= 0:
        raise ValueError("turn_rate_rad_s must be positive")
    return speed_mps / turn_rate_rad_s
```

The navigation guard distance is `stopping_distance + minimum_turn_radius + 50.0`. Retain `navigation_segment_is_legal`, but validate the actual constrained candidate step before committing it.

- [ ] **Step 4: Implement the explicit state machine in `TargetEntity`**

Use these states and public read-only properties:

```python
NavigationState = Literal[
    "NORMAL",
    "BOUNDARY_DECELERATING",
    "BOUNDARY_TURNING",
    "BOUNDARY_RECOVERING",
    "FAILED",
]
```

Add private state backing:

```python
_navigation_state: NavigationState = "NORMAL"
_navigation_state_since_s: float = 0.0
_navigation_recovery_waypoint_xy: tuple[float, float] | None = None
_navigation_guard_distance_m: float = 0.0
_last_navigation_error: str | None = None
```

Transition rules:

- `NORMAL -> BOUNDARY_DECELERATING` when the projected constrained step violates the guard margin.
- Decelerate as far as zero while preserving body heading.
- `BOUNDARY_DECELERATING -> BOUNDARY_TURNING` once a legal inward recovery heading can be acquired.
- Choose a recovery waypoint inside the map by at least the guard margin; rotate under the normal turn-rate limit.
- `BOUNDARY_TURNING -> BOUNDARY_RECOVERING` when the constrained next step points safely inward.
- Return to `NORMAL` only after two consecutive legal steps and restored guard margin.
- Enter `FAILED` after `boundary_recovery_timeout_s`; do not silently reset or teleport.

The legacy `_navigation_guard_failed` flag may remain as a compatibility property derived from `navigation_state == "FAILED"`; it must no longer cause repeated frozen steps during a recoverable condition.

- [ ] **Step 5: Add transition events at the engine boundary**

Record state changes once, using these exact event types:

```text
target_boundary_recovery_started
target_boundary_turn_started
target_boundary_recovery_completed
target_navigation_recovery_failed
```

Each event must include target ID, old/new state, position, guard distance, state age, and error reason where present. Update `tests/simulation/test_engine.py` to assert ordered, non-duplicated events for one recovery episode.

- [ ] **Step 6: Run focused and regression tests**

Run:

```powershell
python -m pytest tests/simulation/test_target_boundary_recovery.py tests/simulation/test_kinematics.py tests/simulation/test_target.py tests/simulation/test_target_guidance.py tests/simulation/test_engine.py -q
python -m ruff check src/underwater_tracking/simulation/kinematics.py src/underwater_tracking/simulation/target.py src/underwater_tracking/simulation/engine.py tests/simulation/test_target_boundary_recovery.py
```

Expected: all tests PASS; no test accepts an out-of-bounds position or a silent permanent freeze.

- [ ] **Step 7: Commit Task 2**

```powershell
git add src/underwater_tracking/simulation/kinematics.py src/underwater_tracking/simulation/target.py src/underwater_tracking/simulation/engine.py tests/simulation/test_target_boundary_recovery.py tests/simulation/test_kinematics.py tests/simulation/test_engine.py
git commit -m "fix: recover target motion at map boundaries"
```

---

### Task 3: Validate Predictions and Execute the Bounded Fallback Chain

**Files:**
- Create: `src/underwater_tracking/prediction/health.py`
- Modify: `src/underwater_tracking/prediction/port.py`
- Modify: `src/underwater_tracking/prediction/imm_forecast.py`
- Create: `tests/prediction/test_health.py`
- Modify: `tests/prediction/test_port.py`
- Modify: `tests/prediction/test_imm_forecast.py`

- [ ] **Step 1: Write table-driven failing health tests**

Use real `PredictedTrackRef` instances and cover every rejection reason:

```python
@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"points_xy": ((float("nan"), 0.0),)}, "non_finite_point"),
        ({"points_xy": ((20_001.0, 0.0),)}, "point_out_of_bounds"),
        ({"corridor_radius_m": (6_001.0,)}, "corridor_radius_exceeded"),
        ({"clipping_records": tuple(str(i) for i in range(3))}, "excessive_clipping"),
        ({"times_s": (60.0, 30.0)}, "non_monotonic_time"),
    ],
)
def test_assess_prediction_reports_machine_readable_reasons(mutation, reason) -> None:
    prediction = valid_prediction().model_copy(update=mutation)
    health = assess_prediction(
        prediction,
        snapshot_sim_time_s=100,
        map_bounds_xy=(-10_000, 10_000, -10_000, 10_000),
        config=health_config(),
        max_speed_mps=15.0,
        max_turn_rate_rad_s=0.05,
        point_confidence=tuple(0.9 for _ in prediction.points_xy),
    )
    assert health.status != "valid"
    assert reason in health.reason_codes
```

Also assert point-array length equality, speed, turn rate, confidence range, monotonic non-increasing confidence, final confidence floor, source age, clip fraction, and the effective radius cap:

```python
assert effective_radius_limit_m(
    (-10_000, 10_000, -8_000, 8_000), health_config()
) == 4_000.0
```

- [ ] **Step 2: Run the health tests and confirm they fail**

Run:

```powershell
python -m pytest tests/prediction/test_health.py -q
```

Expected: FAIL because the assessor does not exist.

- [ ] **Step 3: Implement a pure prediction assessor**

Create `prediction/health.py` with this stable public boundary:

```text
def assess_prediction(
    prediction: PredictedTrackRef,
    *,
    snapshot_sim_time_s: int,
    map_bounds_xy: tuple[float, float, float, float],
    config: PredictionHealthConfig,
    max_speed_mps: float,
    max_turn_rate_rad_s: float,
    point_confidence: Sequence[float],
) -> PredictionHealth


def effective_radius_limit_m(
    map_bounds_xy: tuple[float, float, float, float],
    config: PredictionHealthConfig,
) -> float
```

The block above is the public interface contract, not a partial function body.
Implement `effective_radius_limit_m` as
`min(config.max_corridor_radius_m, min(map_width, map_height) *
config.max_corridor_map_fraction)`. The assessor must return all applicable
reason codes in deterministic sorted order. It must not clip or repair the
input prediction.

- [ ] **Step 4: Write failing fallback-order tests through the public predictor port**

Inject deterministic forecasters so the test can force each stage:

```python
def test_predictor_falls_back_from_invalid_imm_to_bounded_bspline() -> None:
    predictor = make_snapshot_predictor(
        imm_forecaster=invalid_out_of_bounds_imm,
        bspline_forecaster=valid_bspline,
        short_history_forecaster=pytest.fail,
    )
    accepted = predictor(snapshot_with_track_history())
    assert accepted.health.status == "degraded"
    assert accepted.health.regime == "bspline"
    assert accepted.prediction is not None
    assert "imm_point_out_of_bounds" in accepted.health.reason_codes


def test_predictor_returns_unavailable_after_every_bounded_fallback_fails() -> None:
    accepted = all_invalid_predictor()(snapshot_with_track_history())
    assert accepted.prediction is None
    assert accepted.health.status == "unavailable"
    assert accepted.health.regime == "boundary_recovery"
```

- [ ] **Step 5: Change `make_snapshot_predictor` to return `AcceptedPrediction`**

Keep the estimator-safe snapshot input. Execute and assess candidates in this exact order:

```text
IMM -> bounded B-spline -> bounded short history -> boundary recovery -> unavailable
```

Rules:

- Return `valid/imm` only when the IMM candidate passes every check.
- Any accepted fallback is `degraded` and retains the rejected upstream reason codes with a regime prefix.
- The boundary-recovery candidate uses only the current public track state, legal map bounds, and configured motion limits.
- Never clamp a failed candidate and call it valid.
- Preserve the raw candidate ID in `health.raw_prediction_id`, including when the final state is unavailable.
- Generate point confidence once in the prediction layer and assess exactly those values that will later be published.

Update all direct call sites and tests to consume `accepted.prediction` only after checking health.

- [ ] **Step 6: Run prediction regressions and estimator-boundary tests**

Run:

```powershell
python -m pytest tests/prediction/test_health.py tests/prediction/test_port.py tests/prediction/test_imm_forecast.py tests/prediction/test_bspline.py tests/domain/test_truth_boundary.py -q
python -m ruff check src/underwater_tracking/prediction tests/prediction
python -m mypy src/underwater_tracking/prediction
```

Expected: all commands PASS, including the existing test that the predictor exposes no simulator truth-history port.

- [ ] **Step 7: Commit Task 3**

```powershell
git add src/underwater_tracking/prediction/health.py src/underwater_tracking/prediction/port.py src/underwater_tracking/prediction/imm_forecast.py tests/prediction/test_health.py tests/prediction/test_port.py tests/prediction/test_imm_forecast.py
git commit -m "feat: gate predictions through bounded fallbacks"
```

---

### Task 4: Build the Deterministic Four-Region Baseline

**Files:**
- Create: `src/underwater_tracking/planning/region_baseline.py`
- Modify: `src/underwater_tracking/planning/dynamic_regions.py`
- Modify: `src/underwater_tracking/planning/regions.py`
- Create: `tests/planning/test_region_baseline.py`
- Modify: `tests/planning/test_regions.py`

- [ ] **Step 1: Write failing invariant tests for all generation modes**

Create one assertion helper used by valid IMM, degraded prediction, boundary recovery, and previous-plan reprojection cases:

```python
def assert_four_region_invariants(result: FourRegionBaseline) -> None:
    regions = result.regions
    assert tuple(region.region_id for region in regions) == (
        "T1:task:01", "T1:task:02", "T1:task:03", "T1:task:04"
    )
    assert [(region.start_s, region.end_s) for region in regions] == [
        (1_000, 1_540),
        (1_450, 1_990),
        (1_900, 2_440),
        (2_350, 2_800),
    ]
    assert all(polygon_is_inside_map(region.geometry) for region in regions)
    assert only_adjacent_regions_overlap(regions)
    assert all(region.geometry_area_m2 > 0 for region in regions)
```

Add an adversarial corner/stationary case matching the observed failure mode: all predicted x coordinates identical and the largest raw uncertainty radius larger than the map. Assert that deterministic degradation still returns four legal regions instead of raising `map bounds cannot retain minimum dynamic region area`.

- [ ] **Step 2: Run the new planning tests and confirm they fail**

Run:

```powershell
python -m pytest tests/planning/test_region_baseline.py -q
```

Expected: FAIL because `FourRegionBaseline` and `build_four_region_baseline` do not exist.

- [ ] **Step 3: Implement the baseline result and mode types**

Use the existing `ExecutionRegion` contract rather than creating a second region DTO:

```text
RegionGenerationMode = Literal[
    "imm", "degraded_prediction", "boundary_recovery", "reprojected_previous"
]


@dataclass(frozen=True, slots=True)
class FourRegionBaseline:
    regions: tuple[ExecutionRegion, ExecutionRegion, ExecutionRegion, ExecutionRegion]
    mode: RegionGenerationMode
    reason_codes: tuple[str, ...]


def build_four_region_baseline(
    accepted: AcceptedPrediction,
    *,
    target_id: str,
    execution_revision: int,
    origin_sim_time_s: float,
    map_bounds_xy: tuple[float, float, float, float],
    prior_regions: Sequence[ExecutionRegion] = (),
) -> FourRegionBaseline
```

The block above is the public interface contract. The implementation body is
the deterministic generation and reprojection algorithm in Step 4.

- [ ] **Step 4: Implement deterministic generation and reprojection**

Apply this order:

```text
valid IMM geometry
-> accepted degraded geometry
-> current-track boundary-recovery geometry
-> prior four-region reprojection
-> explicit failure
```

Use fixed origin-relative windows:

```python
WINDOW_OFFSETS_S = ((0, 540), (450, 990), (900, 1_440), (1_350, 1_800))
```

Derive geometry from time-segment centerline envelopes, effective bounded radius, and map intersection. When a segment is stationary or too close to an edge, construct a minimum-area in-map rectangle around the accepted segment and shift it inward; do not preserve an impossible raw radius. Only neighboring slots may overlap. Stable IDs and predecessor/successor topology are deterministic from target ID and slot index.

- [ ] **Step 5: Demote LLM planning to semantic selection**

Change `build_llm_task_region_plan` so it receives the immutable baseline candidate set and may return only policy attributes already represented by `UUVRegionalPolicyDecision`: coverage mode, tracking mode, priority, quality, role counts, selected UUV IDs, rationale, and evidence IDs. Reject any response containing coordinates, windows, topology, or unknown candidate IDs.

Retain legacy helpers only for replay/migration call sites and mark them clearly in docstrings. New live execution must not call the geometry-producing LLM route.

- [ ] **Step 6: Run focused and property-style planning tests**

Run:

```powershell
python -m pytest tests/planning/test_region_baseline.py tests/planning/test_regions.py tests/planning/test_dynamic_regions.py -q
python -m ruff check src/underwater_tracking/planning tests/planning
```

Expected: all commands PASS; the stationary/corner/huge-radius case returns four regions without an exception.

- [ ] **Step 7: Commit Task 4**

```powershell
git add src/underwater_tracking/planning/region_baseline.py src/underwater_tracking/planning/dynamic_regions.py src/underwater_tracking/planning/regions.py tests/planning/test_region_baseline.py tests/planning/test_regions.py
git commit -m "feat: guarantee a four-region execution baseline"
```

---

### Task 5: Commit Baseline Execution First and Enforce Snapshot Freshness

**Files:**
- Create: `src/underwater_tracking/runtime/execution_health.py`
- Modify: `src/underwater_tracking/runtime/execution_snapshot_factory.py`
- Modify: `src/underwater_tracking/runtime/execution_coordinator.py`
- Modify: `src/underwater_tracking/runtime/mission_controller.py`
- Modify: `src/underwater_tracking/cli.py`
- Create: `tests/runtime/test_execution_health.py`
- Modify: `tests/runtime/test_execution_snapshot_factory.py`
- Modify: `tests/runtime/test_execution_coordinator.py`
- Modify: `tests/runtime/test_mission_controller.py`
- Modify: `tests/cli/test_cli.py`

- [ ] **Step 1: Write failing freshness tests**

Use boundary values, not approximate ranges:

```python
@pytest.mark.parametrize(
    ("age_s", "expected"),
    [(0, "current"), (450, "current"), (451, "degraded"),
     (900, "degraded"), (901, "expired")],
)
def test_execution_health_age_boundaries(age_s: float, expected: str) -> None:
    assert classify_execution_health(
        snapshot(valid_from_s=1_000, valid_until_s=1_450),
        sim_time_s=1_000 + age_s,
        hard_stale_s=900,
    ).status == expected
```

Also assert `failed` is distinct from `expired`, and that only `current` or `degraded` snapshots are executable.

- [ ] **Step 2: Run the health tests and confirm they fail**

Run:

```powershell
python -m pytest tests/runtime/test_execution_health.py -q
```

Expected: FAIL because the classifier does not exist.

- [ ] **Step 3: Implement execution health as a pure function**

```python
ExecutionHealthStatus = Literal["current", "degraded", "expired", "failed"]


@dataclass(frozen=True, slots=True)
class ExecutionHealth:
    status: ExecutionHealthStatus
    age_s: float
    reason_codes: tuple[str, ...]

    @property
    def executable(self) -> bool:
        return self.status in {"current", "degraded"}
```

Calculate age from `sim_time_s - snapshot.valid_from_s`. Preserve model validation errors as `failed`, not as age-based expiry.

- [ ] **Step 4: Write failing two-stage commit tests**

Prove the baseline is visible before an intentionally blocked optimizer completes, and prove stale compare-and-set output cannot replace it:

```python
def test_baseline_is_committed_before_llm_optimization_returns() -> None:
    optimizer = BlockingOptimizer()
    coordinator = coordinator_with_valid_prediction(optimizer=optimizer)
    coordinator.refresh(sim_time_s=1_000)

    active = coordinator.active_snapshot()
    assert active is not None
    assert active.plan_source == "deterministic"
    assert len(active.regions) == 4
    assert len(active.task_groups) == 4
    assert optimizer.started.wait(timeout=1)


def test_stale_semantic_optimization_cannot_replace_newer_baseline() -> None:
    result = coordinator.commit_semantic_optimization(
        stale_candidate, base_execution_revision=1
    )
    assert result.status == "rejected"
    assert result.reason == "stale_execution_base"
```

- [ ] **Step 5: Make `build_execution_snapshot` consume accepted geometry**

Update the factory boundary to receive `AcceptedPrediction` and `FourRegionBaseline`. It must:

- reject unavailable predictions;
- set `prediction_revision` once and propagate it to snapshot, prediction, regions, groups, and evidence;
- set `valid_from_s = situation.sim_time_s` and `valid_until_s = valid_from_s + 450`;
- allocate exactly four task groups using eight distinct execution UUVs;
- retain extra UUVs only in `reserve_uuvs`;
- attach baseline mode and prediction-health reason codes to degradation evidence.

- [ ] **Step 6: Refactor the coordinator and CLI into two stages**

The synchronous path in `cli.py` must be:

```text
accepted prediction
-> deterministic intent
-> deterministic four-region baseline
-> build snapshot
-> compare-and-set commit
-> publish
```

Only after that commit may the asynchronous LLM path start. Its result may update semantic policy through compare-and-set using the baseline execution revision. Geometry, windows, topology, prediction revision, and member UUV IDs remain byte-for-byte identical.

Remove any live path where planning failure merely preserves an already expired snapshot. Instead publish health `failed` with the reason, and prevent mission dispatch.

- [ ] **Step 7: Gate mission execution by health**

In `mission_controller.py`, reject expired and failed snapshots before translating regions into assignments:

```python
health = classify_execution_health(
    snapshot,
    sim_time_s=current_sim_time_s,
    hard_stale_s=config.tracking.prediction_health.hard_stale_s,
)
if not health.executable:
    return MissionApplyResult.rejected("execution_snapshot_not_executable")
```

The active plan may remain visible for audit but may not continue issuing new mission commands.

- [ ] **Step 8: Run focused runtime and CLI regressions**

Run:

```powershell
python -m pytest tests/runtime/test_execution_health.py tests/runtime/test_execution_snapshot_factory.py tests/runtime/test_execution_coordinator.py tests/runtime/test_mission_controller.py tests/cli/test_cli.py -q
python -m ruff check src/underwater_tracking/runtime src/underwater_tracking/cli.py tests/runtime tests/cli/test_cli.py
```

Expected: all commands PASS; a delayed or failed LLM cannot prevent the deterministic revision from becoming active.

- [ ] **Step 9: Commit Task 5**

```powershell
git add src/underwater_tracking/runtime/execution_health.py src/underwater_tracking/runtime/execution_snapshot_factory.py src/underwater_tracking/runtime/execution_coordinator.py src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/cli.py tests/runtime/test_execution_health.py tests/runtime/test_execution_snapshot_factory.py tests/runtime/test_execution_coordinator.py tests/runtime/test_mission_controller.py tests/cli/test_cli.py
git commit -m "feat: commit fresh deterministic execution snapshots"
```

---

### Task 6: Extend the Atomic Operational Frame and Preserve Replay Compatibility

**Files:**
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/api/live.py`
- Modify: `src/underwater_tracking/api/replay.py`
- Modify: `src/underwater_tracking/api/legacy_frame_adapter.py`
- Modify: `tests/api/test_frame_contracts.py`
- Modify: `tests/api/test_frame_pipeline.py`
- Modify: `tests/api/test_live_publisher.py`
- Modify: `tests/api/test_replay_compatibility.py`

- [ ] **Step 1: Write failing frame-contract tests**

Assert new live fields and cross-revision consistency:

```python
def test_live_frame_carries_prediction_and_execution_health() -> None:
    frame = build_live_frame()
    prediction = frame.estimates[0].prediction
    assert prediction is not None
    assert prediction.prediction_revision == frame.execution.prediction_revision
    assert prediction.origin_sim_time_s == 1_000
    assert prediction.health.status == "valid"
    assert frame.execution.valid_from_s == 1_000
    assert frame.execution.valid_until_s == 1_450
    assert frame.execution.health_status == "current"
    assert frame.execution.region_generation_mode == "imm"


def test_frame_rejects_prediction_execution_revision_mismatch() -> None:
    with pytest.raises(ValueError, match="prediction revision"):
        OperationalFrame.model_validate(mismatched_frame_payload())
```

- [ ] **Step 2: Run frame tests and confirm they fail**

Run:

```powershell
python -m pytest tests/api/test_frame_contracts.py tests/api/test_frame_pipeline.py -q
```

Expected: FAIL because the new view fields and cross-field validation are absent.

- [ ] **Step 3: Extend view contracts**

Add to `PredictionCorridorView`:

```python
prediction_id: str
prediction_revision: int = Field(ge=1)
origin_sim_time_s: float = Field(ge=0)
health: PredictionHealthView
```

Add a transport-specific `PredictionHealthView` with the same fields as
`PredictionHealth`, plus replay-only `legacy_unknown` values for both status
and regime. Add to `ExecutionView`:

```python
valid_from_s: float = Field(ge=0)
valid_until_s: float = Field(gt=0)
health_status: Literal["current", "degraded", "expired", "failed"]
health_reasons: tuple[str, ...] = ()
region_generation_mode: Literal[
    "imm", "degraded_prediction", "boundary_recovery", "reprojected_previous"
]
```

Replace `data_status: current|stale|unavailable` only after adding a `model_validator(mode="before")` that maps legacy `stale -> degraded` and `unavailable -> failed` during replay normalization.

- [ ] **Step 4: Build frame geometry only from accepted authoritative state**

Update `_build_prediction` and `_build_execution_view` to consume the committed execution snapshot and its accepted prediction metadata. Remove new-live use of `_clip_point`: invalid business geometry must have been rejected before frame construction. Keep clipping only inside an explicitly named legacy replay normalizer.

Calculate point confidence in one backend function and publish the exact assessed array. Assert all lengths match before constructing `PredictionCorridorView`.

- [ ] **Step 5: Make publication atomic across transports**

In `OperationalFramePublisher.publish`, validate once, serialize once, then pass the same immutable serialized payload to:

```text
in-memory latest frame -> WebSocket subscribers -> JSONL logger
```

Add a test hashing the payload observed by all three sinks:

```python
assert http_hash == websocket_hash == jsonl_hash
```

No sink may independently rebuild a frame.

- [ ] **Step 6: Normalize old replay frames explicitly**

In `legacy_frame_adapter.py`, when health fields are absent, set prediction
status and regime to `legacy_unknown`, preserve existing geometry, and add
`legacy_health_missing` to reasons. `replay.py` must continue routing every
read through that adapter. Do not apply this default to the live builder. Add
fixture tests for an old JSONL line and a new JSONL line.

- [ ] **Step 7: Run API, publisher, and replay regressions**

Run:

```powershell
python -m pytest tests/api/test_frame_contracts.py tests/api/test_frame_pipeline.py tests/api/test_live_publisher.py tests/api/test_replay_compatibility.py -q
python -m ruff check src/underwater_tracking/api src/underwater_tracking/domain/ui_models.py tests/api
python -m mypy src/underwater_tracking/api src/underwater_tracking/domain/ui_models.py
```

Expected: all commands PASS; old replay fixtures load as `legacy_unknown`, while new live frames reject missing health.

- [ ] **Step 8: Commit Task 6**

```powershell
git add src/underwater_tracking/domain/ui_models.py src/underwater_tracking/api/frame_builder.py src/underwater_tracking/api/live.py src/underwater_tracking/api/replay.py src/underwater_tracking/api/legacy_frame_adapter.py tests/api/test_frame_contracts.py tests/api/test_frame_pipeline.py tests/api/test_live_publisher.py tests/api/test_replay_compatibility.py
git commit -m "feat: publish health-aware atomic operational frames"
```

---

### Task 7: Render Honest Prediction, Region, and Sensor Layers

**Files:**
- Modify: `src/underwater_tracking/ui/src/types/frames.ts`
- Modify: `src/underwater_tracking/ui/src/components/map/PredictionOverlay.tsx`
- Modify: `src/underwater_tracking/ui/src/components/map/PredictionOverlay.test.ts`
- Modify: `src/underwater_tracking/ui/src/components/CanvasMap.tsx`
- Modify: `src/underwater_tracking/ui/src/components/CanvasMap.test.ts`
- Modify: `src/underwater_tracking/ui/src/App.css`

- [ ] **Step 1: Update TypeScript contracts and write failing normalization tests**

Mirror the Python view fields exactly:

```typescript
export type PredictionHealthStatus = "valid" | "degraded" | "unavailable" | "legacy_unknown";

export interface PredictionHealthView {
  status: PredictionHealthStatus;
  regime: "imm" | "bspline" | "short_history" | "boundary_recovery" | "legacy_unknown";
  reason_codes: string[];
  source_track_age_s: number;
  clipped_point_fraction: number;
  maximum_radius_m: number;
  raw_prediction_id: string | null;
}
```

Add corresponding prediction ID/revision/origin fields and execution freshness/mode fields. Update fixture builders explicitly; do not use broad type casts to bypass the new contract.

- [ ] **Step 2: Write failing tests proving the frontend does not rewrite radii**

Replace the current confidence-inflation expectation:

```typescript
it("renders backend radii without confidence inflation", () => {
  const prediction = predictionFixture({
    radius_m: [200, 300, 400],
    point_confidence: [0.9, 0.5, 0.1],
  });
  expect(displayRadii(prediction)).toEqual([200, 300, 400]);
});

it("does not draw a corridor for unavailable prediction health", () => {
  render(<PredictionOverlay prediction={predictionFixture({
    health: healthFixture({ status: "unavailable" }),
  })} />);
  expect(screen.queryByTestId("prediction-corridor")).not.toBeInTheDocument();
});
```

- [ ] **Step 3: Run the overlay tests and confirm they fail**

Run:

```powershell
Set-Location src/underwater_tracking/ui
npx vitest run src/components/map/PredictionOverlay.test.ts
```

Expected: FAIL because `confidenceAdjustedRadii` currently changes backend geometry and health styles are absent.

- [ ] **Step 4: Implement health-aware prediction rendering**

Rename `confidenceAdjustedRadii` to `displayRadii` and return a validated copy of `radius_m`. Render:

- `valid`: solid centerline, normal translucent fill, confidence-marked samples;
- `degraded`: dashed centerline, hatched or patterned fill, visible degraded status affordance;
- `unavailable`: no corridor polygon or invented line, only an operator-visible unavailable status;
- `legacy_unknown`: restrained legacy style distinct from valid live data.

Point confidence controls alpha/marker emphasis only. It never changes radius or centerline coordinates.

- [ ] **Step 5: Write failing canvas draw-order and sensor-style tests**

Mock the canvas context and assert this semantic order:

```text
map/grid
regions/handoffs
prediction corridor
prediction centerline/samples
target detection circle
UUV sonar fans
labels
selection/errors
```

Assert the target circle uses a red dashed stroke, active sonar an amber fan, and passive sonar a cyan fan. Assert region status styles differ for planned, active, handoff, degraded, and uncovered.

- [ ] **Step 6: Implement the ordered map layers**

Split the large canvas drawing body into named internal layer functions without changing component ownership:

```typescript
drawMapAndGrid(ctx, frame, transform, styles);
drawExecutionRegionsAndHandoffs(ctx, frame, transform, styles);
drawPredictionCorridor(ctx, frame, transform, styles);
drawPredictionCenterline(ctx, frame, transform, styles);
drawTargetDetectionZones(ctx, frame, transform, styles);
drawUuvSonarFields(ctx, frame, transform, styles);
drawLabels(ctx, frame, transform, styles);
drawSelectionAndErrors(ctx, frame, transform, styles);
```

Use the backend `detection_range_m`, UUV sensor mode, bearing, and configured range. Keep all visual styling in a small palette keyed by semantic state.

- [ ] **Step 7: Run UI unit tests and build**

Run:

```powershell
Set-Location src/underwater_tracking/ui
npx vitest run src/components/map/PredictionOverlay.test.ts src/components/CanvasMap.test.ts
npm run build
```

Expected: all commands PASS with no TypeScript contract escapes.

- [ ] **Step 8: Commit Task 7**

```powershell
Set-Location ../../..
git add src/underwater_tracking/ui/src/types/frames.ts src/underwater_tracking/ui/src/components/map/PredictionOverlay.tsx src/underwater_tracking/ui/src/components/map/PredictionOverlay.test.ts src/underwater_tracking/ui/src/components/CanvasMap.tsx src/underwater_tracking/ui/src/components/CanvasMap.test.ts src/underwater_tracking/ui/src/App.css
git commit -m "feat: render health-aware tracking layers"
```

---

### Task 8: Add a Map-Clamped Semantic Camera and Stable Labels

**Files:**
- Modify: `src/underwater_tracking/ui/src/components/map/geometry.ts`
- Create: `src/underwater_tracking/ui/src/components/map/camera.ts`
- Create: `src/underwater_tracking/ui/src/components/map/camera.test.ts`
- Modify: `src/underwater_tracking/ui/src/components/CanvasMap.tsx`
- Modify: `src/underwater_tracking/ui/src/components/CanvasMap.test.ts`

- [ ] **Step 1: Write failing camera geometry tests**

Cover both viewports and the observed extreme prediction case:

```typescript
it.each([
  { width: 1600, height: 1000 },
  { width: 390, height: 844 },
])("keeps semantic bounds inside the map at $width x $height", (viewport) => {
  const camera = semanticCameraForFrame(extremeLiveFrame(), viewport);
  expect(boundsContains(mapBounds(), camera.worldBounds)).toBe(true);
  expect(camera.targetDetectionDiameterPx).toBeGreaterThanOrEqual(160);
  expect(camera.minimumRegionDimensionPx).toBeGreaterThanOrEqual(48);
  expect(camera.twoKilometerSegmentPx).toBeGreaterThanOrEqual(120);
});

it("ignores unavailable prediction geometry when fitting", () => {
  const frame = extremeLiveFrame({ predictionStatus: "unavailable" });
  expect(semanticCameraCandidates(frame)).not.toContainEqual([60_000, 60_000]);
});
```

- [ ] **Step 2: Run the camera tests and confirm they fail**

Run:

```powershell
Set-Location src/underwater_tracking/ui
npx vitest run src/components/map/camera.test.ts src/components/CanvasMap.test.ts
```

Expected: FAIL because semantic camera functions do not exist and the current camera can expand around unusable geometry.

- [ ] **Step 3: Implement semantic candidate collection**

`semanticCameraCandidates(frame)` includes:

- current target estimate;
- prediction centerline only when health is `valid` or `degraded`;
- all four execution-region vertices when execution health is `current` or `degraded`;
- all eight assigned UUV positions;
- target detection-circle cardinal points.

It excludes expired/failed execution geometry and unavailable prediction geometry. It intersects every candidate with authoritative map bounds before fitting.

- [ ] **Step 4: Implement deterministic fit and minimum readability constraints**

`semanticCameraForFrame` must:

- compute candidate bounds;
- add 8% padding;
- clamp to map bounds;
- preserve viewport aspect ratio without exceeding map bounds;
- enforce marker size in screen pixels rather than world meters;
- iteratively tighten the fit, within map bounds, until region, line, and detection-range minimums are satisfied when geometrically possible;
- return explicit `readabilityWarnings` when mutually incompatible constraints cannot all be met.

Do not scale font size with viewport width. Use fixed label sizes and stable screen-space marker dimensions.

- [ ] **Step 5: Integrate camera revision behavior**

Auto-fit only when:

- the prediction revision changes;
- execution changes from unusable to usable;
- the operator activates Reset View.

Do not auto-fit on each animation frame or while the operator has panned/zoomed. Track the last fitted prediction revision and a user-camera-dirty flag in `CanvasMap` state.

- [ ] **Step 6: Add deterministic label priority and collision handling**

Priority order:

```text
selected entity -> target -> active region -> current/next handoff -> active UUV -> remaining UUV
```

Try fixed candidate offsets around each anchor, retain the first non-overlapping screen rectangle, and suppress only lower-priority labels. Never move the underlying marker or world geometry.

- [ ] **Step 7: Run camera, canvas, and build verification**

Run:

```powershell
Set-Location src/underwater_tracking/ui
npx vitest run src/components/map/camera.test.ts src/components/CanvasMap.test.ts
npm run build
```

Expected: all commands PASS on desktop and mobile fixture dimensions.

- [ ] **Step 8: Commit Task 8**

```powershell
Set-Location ../../..
git add src/underwater_tracking/ui/src/components/map/geometry.ts src/underwater_tracking/ui/src/components/map/camera.ts src/underwater_tracking/ui/src/components/map/camera.test.ts src/underwater_tracking/ui/src/components/CanvasMap.tsx src/underwater_tracking/ui/src/components/CanvasMap.test.ts
git commit -m "feat: fit tracking views with a semantic camera"
```

---

### Task 9: Add Cross-Layer Integration Tests for Long-Running Health Transitions

**Files:**
- Create: `tests/integration/test_live_tracking_health_pipeline.py`
- Modify: `tests/integration/test_uuv_only_production_acceptance.py`
- Modify: `tests/api/test_frame_pipeline.py`
- Modify: `tests/runtime/test_execution_coordinator.py`

- [ ] **Step 1: Write a deterministic accelerated pipeline test**

Run the real engine/predictor/planner/coordinator/frame builder without HTTP or browser, using a fixed seed and enough accelerated simulation time to cross refresh and stale boundaries:

```python
def test_tracking_pipeline_remains_bounded_and_executable_for_eight_hours() -> None:
    harness = LiveTrackingHarness(seed=20260828)
    checkpoints = (600, 1_800, 3_600, 7_200, 14_400, 21_600, 28_800)

    for sim_time_s, frame in harness.frames_at(checkpoints):
        assert frame.sim_time_s >= sim_time_s
        assert frame.consistency.valid
        assert frame.execution is not None
        assert frame.execution.health_status in {"current", "degraded"}
        assert len(frame.execution.regions) == 4
        assert len(frame.execution.task_groups) == 4
        assert len({uuv for group in frame.execution.task_groups for uuv in group.member_uuv_ids}) == 8
        assert frame.estimates[0].prediction.health.status in {"valid", "degraded"}
        assert_prediction_inside_map(frame)
        assert_execution_inside_map(frame)
```

- [ ] **Step 2: Run the integration test and confirm it fails before wiring is complete**

Run:

```powershell
python -m pytest tests/integration/test_live_tracking_health_pipeline.py -q
```

Expected: FAIL at the first missing cross-layer health or revision invariant.

- [ ] **Step 3: Wire only missing integration boundaries**

Fix discovered adapter gaps in the already modified production files. Do not weaken assertions, widen thresholds, skip checkpoints, or introduce test-only branches. The test uses the same configuration loader and factories as `main.py`.

- [ ] **Step 4: Add explicit fault-transition coverage**

Inject one invalid IMM cycle and one delayed optimizer cycle. Assert:

```text
valid IMM -> degraded bounded fallback -> next valid IMM
deterministic baseline commit -> stale LLM result rejected -> newer baseline remains active
```

Also assert emitted health reasons and execution revision monotonicity.

- [ ] **Step 5: Run integration and core regressions**

Run:

```powershell
python -m pytest tests/integration/test_live_tracking_health_pipeline.py tests/integration/test_uuv_only_production_acceptance.py tests/api/test_frame_pipeline.py tests/runtime/test_execution_coordinator.py -q
```

Expected: all commands PASS.

- [ ] **Step 6: Commit Task 9**

```powershell
git add tests/integration/test_live_tracking_health_pipeline.py tests/integration/test_uuv_only_production_acceptance.py tests/api/test_frame_pipeline.py tests/runtime/test_execution_coordinator.py
git commit -m "test: cover long-running tracking health transitions"
```

Before staging, inspect `git diff --name-only`. If Step 3 changed a production
adapter, commit that adapter with its focused test before the integration-test
commit, using its exact path. Never stage an entire source directory or any
unrelated user change.

---

### Task 10: Turn the Owned-Process Runner into Real Semantic and Visual Acceptance

**Files:**
- Modify: `tools/run_default_live_acceptance.py`
- Modify: `tests/acceptance/test_default_live_acceptance.py`
- Modify: `src/underwater_tracking/verification/live_demo.py`
- Modify: `tests/verification/test_live_demo_monitor.py`
- Create: `src/underwater_tracking/ui/e2e/live-visualization-acceptance.spec.ts`
- Modify: `src/underwater_tracking/ui/playwright.live.config.ts`
- Rename: `src/underwater_tracking/ui/e2e/task-region-effect.spec.ts` to `src/underwater_tracking/ui/e2e/synthetic-task-region-effect.spec.ts`

- [ ] **Step 1: Write failing artifact-schema tests**

Require this exact run-local structure:

```text
<run_dir>/acceptance/manifest.json
<run_dir>/acceptance/metrics.json
<run_dir>/acceptance/frame-checkpoints.jsonl
<run_dir>/acceptance/screenshots/desktop-<checkpoint>.png
<run_dir>/acceptance/screenshots/mobile-<checkpoint>.png
<run_dir>/acceptance/browser-console.jsonl
<run_dir>/acceptance/backend-errors.jsonl
```

Test the manifest fields:

```python
assert manifest["entrypoint"] == "main.py"
assert manifest["mock_routes"] == []
assert manifest["fake_websockets"] is False
assert manifest["viewports"] == [[1600, 1000], [390, 844]]
assert manifest["checkpoints_s"] == [600, 1800, 3600, 7200, 14400, 21600, 28800]
assert manifest["ui_bundle_sha256"]
assert manifest["operational_frames_sha256"]
```

- [ ] **Step 2: Run acceptance unit tests and confirm they fail**

Run:

```powershell
python -m pytest tests/acceptance/test_default_live_acceptance.py tests/verification/test_live_demo_monitor.py -q
```

Expected: FAIL because the strict artifact bundle is not produced.

- [ ] **Step 3: Extend the runner without creating a second server lifecycle**

Continue using `run_default_live_acceptance.py` as the sole owner of the `main.py` child process. Add checkpoint collection at simulation seconds:

```python
CHECKPOINTS_S = (600, 1_800, 3_600, 7_200, 14_400, 21_600, 28_800)
VIEWPORTS = ((1600, 1_000), (390, 844))
```

At every checkpoint, record frame ID, sim time, prediction ID/revision/health, maximum radius, execution revision/health/age, four region summaries, four task groups, eight member IDs, map bounds, target detection range, transport hashes, and relevant event IDs.

Read the owned run's execution database at the same checkpoint and record the
latest committed execution revision, source prediction revision, validity
window, and planning-error count in `metrics.json`. The database query is
read-only and uses the run directory selected by the owned process; it must not
fall back to a different previous run.

- [ ] **Step 4: Add backend semantic failure conditions**

Fail the run immediately on:

- non-finite or out-of-map prediction points;
- prediction radius above the effective configured cap;
- missing/expired/failed execution;
- not exactly four stable regions or four two-UUV groups;
- not exactly eight distinct execution UUVs;
- mismatched prediction/execution revisions;
- planning exception text, navigation recovery timeout, or repeated frozen target position outside a legitimate zero-speed state;
- unequal latest-frame, WebSocket, and JSONL payload hashes.
- database execution/prediction revisions that disagree with the published frame.

Write all fatal and recoverable backend errors to `backend-errors.jsonl` with checkpoint and frame context.

- [ ] **Step 5: Create a Playwright test against the real server**

The new test must not call `page.route`, `route.fulfill`, `addInitScript` for frame injection, or construct a `WebSocket` substitute. It connects to the owned server URL and, at each checkpoint, asserts:

- canvas is nonblank by pixel variance;
- target detection circle has a measurable red dashed perimeter;
- at least one amber active-sonar fan and cyan passive-sonar field are visible when the frame declares those modes;
- four region geometries occupy measurable screen area and use status-distinct colors;
- prediction line length and corridor fill occupy measurable pixels when health is usable;
- unavailable health does not produce a synthetic corridor;
- no rendered label or control overlaps the map bounds or leaves the viewport;
- browser console contains no error.

Capture both 1600x1000 and 390x844 screenshots into the runner-provided acceptance directory.

- [ ] **Step 6: Separate synthetic visual documentation from acceptance**

Rename the existing synthetic test and its test title so its role is explicit. Preserve it as a fast design regression, but ensure `playwright.live.config.ts` selects only `live-visualization-acceptance.spec.ts` for strict live acceptance. The historical output image is not read or compared by the live test.

- [ ] **Step 7: Run runner unit tests and UI test discovery**

Run:

```powershell
python -m pytest tests/acceptance/test_default_live_acceptance.py tests/verification/test_live_demo_monitor.py -q
Set-Location src/underwater_tracking/ui
npx playwright test --config playwright.live.config.ts --list
```

Expected: unit tests PASS and Playwright lists the real live visualization test without starting a mocked fixture.

- [ ] **Step 8: Commit Task 10**

```powershell
Set-Location ../../..
git add tools/run_default_live_acceptance.py tests/acceptance/test_default_live_acceptance.py src/underwater_tracking/verification/live_demo.py tests/verification/test_live_demo_monitor.py src/underwater_tracking/ui/e2e/live-visualization-acceptance.spec.ts src/underwater_tracking/ui/e2e/synthetic-task-region-effect.spec.ts src/underwater_tracking/ui/playwright.live.config.ts
git commit -m "test: verify real live tracking visualization"
```

---

### Task 11: Run the Full Verification Matrix and Record the New Evidence

**Files:**
- Modify only if verification exposes a defect: files already listed in Tasks 1-10
- Generate at runtime: `<run_dir>/acceptance/**`
- Do not modify: `outputs/imm-confidence-trajectory-effect.png`

- [ ] **Step 1: Run Python unit and integration tests**

Run:

```powershell
python -m pytest -q -m "not real_llm and not live_acceptance and not long_running"
```

Expected: PASS with no unexpected skips beyond explicitly marked external/long-running tests.

- [ ] **Step 2: Run Python quality gates**

Run:

```powershell
python -m ruff check src tests tools
python -m mypy src/underwater_tracking
```

Expected: both commands PASS.

- [ ] **Step 3: Run frontend unit, type, and production-build gates**

Run:

```powershell
Set-Location src/underwater_tracking/ui
npm test
npm run build
```

Expected: both commands PASS.

- [ ] **Step 4: Run the real owned-process acceptance**

From the repository root, with real-provider credentials already configured by the operator:

```powershell
$env:UNDERWATER_TRACKING_RUN_LIVE_ACCEPTANCE = "1"
$env:UNDERWATER_TRACKING_RUN_REAL_LLM = "1"
python tools/run_default_live_acceptance.py --playwright-command "npm --prefix src/underwater_tracking/ui run test:e2e:live"
```

Expected: exit code 0 after the 28,800-second simulation checkpoint, with all required artifacts beneath the owned run directory. The runner must print that absolute run directory.

- [ ] **Step 5: Independently inspect semantic evidence**

Verify `metrics.json` and every line of `frame-checkpoints.jsonl` satisfy:

```text
prediction health is valid or degraded
prediction points are finite and inside map bounds
maximum radius is under the effective cap
execution health is current or degraded
execution age never exceeds 900 seconds
four stable regions and four groups exist
eight distinct execution UUVs exist
prediction and execution revisions match
HTTP, WebSocket, and JSONL hashes match
no terminal navigation or planning errors exist
```

- [ ] **Step 6: Inspect screenshots and canvas pixels at both viewports**

For each checkpoint, inspect desktop and mobile screenshots. Confirm the target detection circle, active/passive UUV sensor fields, four task regions, prediction centerline, and uncertainty band are simultaneously readable where semantically applicable. Confirm labels, controls, and overlays do not overlap incoherently, and the camera stays within the map.

Pixel checks must prove nonblank canvas and measurable semantic colors/areas; screenshots alone are supporting evidence, not the only assertion.

- [ ] **Step 7: Check the worktree before the verification commit**

Run:

```powershell
Set-Location ../../..
git status --short
git diff --check
git log --oneline -12
```

Expected: no whitespace errors. Preserve and do not stage any pre-existing user changes in `command-center.spec.ts`, its snapshots, or `uuv-live-timeline.spec.ts` unless the user separately authorizes incorporating them.

- [ ] **Step 8: Commit only necessary verification fixes**

If verification required production or test fixes, commit each cohesive fix separately after rerunning the failing gate. Runtime acceptance artifacts normally remain uncommitted unless repository policy already tracks that run directory.

Use a final documentation-only commit only when a tracked verification index already exists:

```powershell
git commit -m "docs: record live visualization acceptance evidence"
```

Do not create an empty commit and do not claim completion until all commands and real-run evidence have been reviewed.

---

## Implementation Review Checkpoints

Pause for review after each dependency boundary:

1. **After Tasks 1-3:** confirm navigation recovery and prediction health/fallback semantics before planning consumes them.
2. **After Tasks 4-6:** confirm four-region invariants, two-stage execution commit, freshness, and atomic frame compatibility.
3. **After Tasks 7-8:** inspect focused desktop/mobile component screenshots before starting the long real-run acceptance work.
4. **After Tasks 9-10:** review deterministic integration evidence and the acceptance harness for any hidden mock path.
5. **After Task 11:** review the complete evidence bundle before declaring the issue fixed.

## Completion Criteria

Implementation is complete only when all of the following are true:

- A seeded eight-hour accelerated run reaches 28,800 simulation seconds without terminal target navigation recovery failure.
- Every accepted prediction is finite, map-bounded, kinematically plausible, confidence-valid, and under the effective uncertainty-radius cap.
- Every usable execution frame carries four stable regions, four two-UUV task groups, eight distinct execution UUVs, and matching prediction/execution revisions.
- Execution is refreshed within 450 seconds when healthy, visibly degraded through 900 seconds, and never dispatched after expiry or failure.
- The browser renders backend geometry without confidence-based inflation or client-side repair.
- Desktop and mobile semantic camera checks keep the target, usable prediction, four regions, eight UUVs, and target detection range readable inside the map.
- Real HTTP, WebSocket, JSONL, replay, browser, and database evidence agree for the owned `main.py` process.
- The acceptance suite contains no mocked route, fake WebSocket, injected fixture frame, or hard-coded acceptance geometry.
- The full Python and frontend verification matrix passes, and pre-existing unrelated user changes remain untouched.
