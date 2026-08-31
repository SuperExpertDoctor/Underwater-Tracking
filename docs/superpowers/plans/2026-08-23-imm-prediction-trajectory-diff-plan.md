# IMM Prediction Trajectory Diff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect materially divergent consecutive IMM-derived trajectory forecasts, verify suspected semantic intent changes through the real Intent LLM, and expose one auditable diff-to-plan causal chain in live/replay UI and final acceptance reports.

**Architecture:** Add a pure time-aligned trajectory comparator and a pure persistent hysteresis gate under `prediction/`. The central graph owns checkpointed diff/gate state and a dedicated post-prediction Intent LLM branch that cannot loop back into prediction; only confirmed semantic label changes enter regional replanning. Durable events are the single source for ledger, memory, live/replay frames, UI, and acceptance evidence.

**Tech Stack:** Python 3.11, Pydantic v2 strict models, NumPy, LangGraph checkpoints, SQLite event/decision stores, React 18, TypeScript, Vitest/Testing Library, pytest, Playwright, real configured LongCat-compatible LLM clients.

## Global Constraints

- Compare forecasts only at equal absolute simulation times over at least `300s` and three samples.
- Trigger only when `D_norm >= 2.45` and `D_abs >= 250m` for two consecutive observation cycles.
- Clear the latch when `D_norm < 1.75` or `D_abs < 150m`.
- Use `uncertainty_floor_m=1.0` and near-term exponential decay `600s`.
- IMM labels `cv`, `left_turn`, and `right_turn` are motion modes, never semantic intent labels.
- `target_intent_changed` is strategic and may be emitted only after a changed semantic label passes confidence `0.70`, margin `0.15`, two consecutive real-LLM analyses, and provenance checks.
- Public prediction/diff code may use only estimated beliefs, public source IDs, and public runtime state; target truth and adversary-private state are forbidden.
- Provider failure is persisted as DEGRADED/FAIL; no heuristic or fake response may substitute for the real Intent or regional LLM in release acceptance.
- Unit-test doubles are allowed only for deterministic contract/failure tests and are not release evidence.
- Preserve all unrelated dirty-worktree changes, especially user-owned changes in `tests/api/test_app.py` and existing UI screenshots.

---

## File Structure

- Create `src/underwater_tracking/prediction/diff.py`: pure alignment, geometric metrics, Jensen-Shannon distance, and validation.
- Create `src/underwater_tracking/prediction/diff_gate.py`: pure consecutive-cycle and hysteresis state transition.
- Create `tests/prediction/test_diff.py`: comparator examples and invalid-input coverage.
- Create `tests/prediction/test_diff_properties.py`: deterministic invariance/property coverage.
- Create `tests/prediction/test_diff_gate.py`: persistence, reset, and one-shot suspicion tests.
- Create `tests/domain/test_agent_models.py`: strict diff/prediction contract round trips and validators.
- Create `src/underwater_tracking/ui/src/components/PredictionDiffPanel.tsx`: compact operational diff presentation.
- Create `src/underwater_tracking/ui/src/components/PredictionDiffPanel.test.tsx`: UI states and responsive-safe content tests.
- Modify `src/underwater_tracking/config/models.py` and `configs/agent.yaml`: strict resolved thresholds.
- Modify `src/underwater_tracking/domain/agent_models.py`: prediction metadata and diff/gate contracts.
- Modify `src/underwater_tracking/prediction/port.py`: attach public IMM probabilities and explicit predictor regime.
- Modify `src/underwater_tracking/agent/state.py`: checkpointed diff and gate channels.
- Modify `src/underwater_tracking/agent/graphs/central.py`: compare forecasts, emit suspicion, run post-prediction real Intent LLM verification, and route confirmation without a graph loop.
- Modify `src/underwater_tracking/agent/nodes/event_monitor.py` and `src/underwater_tracking/domain/event_registry.py`: correct semantic event names, levels, audiences, and payload validation.
- Modify `src/underwater_tracking/simulation/engine.py`: rename estimator-only semantic overclaim to `imm_motion_mode_changed`.
- Modify `src/underwater_tracking/api/live.py`, `src/underwater_tracking/api/frame_builder.py`, and `src/underwater_tracking/domain/ui_models.py`: project checkpointed diff state into live/replay frames.
- Modify `src/underwater_tracking/ui/src/types/frames.ts`, `src/underwater_tracking/ui/src/App.tsx`, `src/underwater_tracking/ui/src/components/PlaybackBar.tsx`, and sidebar CSS: render prediction-diff and event states.
- Modify `src/underwater_tracking/verification/live_demo.py`, `scripts/monitor_main_battle.py`, and acceptance tests: require the real diff-to-plan evidence chain.

---

### Task 1: Strict Configuration and Cross-Layer Contracts

**Files:**
- Modify: `src/underwater_tracking/config/models.py`
- Modify: `configs/agent.yaml`
- Modify: `src/underwater_tracking/domain/agent_models.py`
- Modify: `src/underwater_tracking/agent/state.py`
- Test: `tests/agent/test_agent_loader.py`
- Create: `tests/domain/test_agent_models.py`

**Interfaces:**
- Produces: `TrajectoryDiffConfig`, `PredictionRegime`, `TrajectoryDiffStatus`, `TrajectoryDiffResult`, `TrajectoryDiffGateState`, and new `CarrierState` channels.
- Consumes: existing `StrictModel`, `PredictedTrackRef`, and `AgentConfig` loader behavior.

- [ ] **Step 1: Write failing strict-config tests**

Append tests that load defaults and reject invalid hysteresis:

```python
def test_agent_config_has_strict_trajectory_diff_defaults() -> None:
    config = AgentConfig()
    assert config.trajectory_diff.normalized_threshold == 2.45
    assert config.trajectory_diff.absolute_floor_m == 250.0
    assert config.trajectory_diff.uncertainty_floor_m == 1.0
    assert config.trajectory_diff.confirmation_cycles == 2


def test_trajectory_diff_reset_thresholds_must_be_lower() -> None:
    with pytest.raises(ValueError, match="reset normalized threshold"):
        TrajectoryDiffConfig(reset_normalized_threshold=2.45)
    with pytest.raises(ValueError, match="reset absolute floor"):
        TrajectoryDiffConfig(reset_absolute_floor_m=250.0)
```

- [ ] **Step 2: Run the config tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/agent/test_agent_loader.py -q
```

Expected: FAIL because `TrajectoryDiffConfig` and `AgentConfig.trajectory_diff` do not exist.

- [ ] **Step 3: Add the strict configuration model**

Add before `AgentConfig`:

```python
class TrajectoryDiffConfig(StrictModel):
    normalized_threshold: float = Field(default=2.45, gt=0, allow_inf_nan=False)
    absolute_floor_m: float = Field(default=250.0, gt=0, allow_inf_nan=False)
    uncertainty_floor_m: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    near_term_decay_s: float = Field(default=600.0, gt=0, allow_inf_nan=False)
    confirmation_cycles: int = Field(default=2, ge=1)
    reset_normalized_threshold: float = Field(default=1.75, ge=0, allow_inf_nan=False)
    reset_absolute_floor_m: float = Field(default=150.0, ge=0, allow_inf_nan=False)
    minimum_overlap_s: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    minimum_samples: int = Field(default=3, ge=2)
    schema_version: str = Field(default="trajectory-diff-v1", min_length=1)

    @model_validator(mode="after")
    def reset_thresholds_are_lower(self) -> "TrajectoryDiffConfig":
        if self.reset_normalized_threshold >= self.normalized_threshold:
            raise ValueError("reset normalized threshold must be below trigger threshold")
        if self.reset_absolute_floor_m >= self.absolute_floor_m:
            raise ValueError("reset absolute floor must be below trigger floor")
        return self
```

Add `trajectory_diff: TrajectoryDiffConfig = Field(default_factory=TrajectoryDiffConfig)` to `AgentConfig`, and add the approved YAML block to `configs/agent.yaml` once, removing the duplicate `retention:` block currently present there while preserving its values.

- [ ] **Step 4: Write failing contract round-trip tests**

```python
def test_trajectory_diff_result_round_trips_with_auditable_thresholds() -> None:
    result = TrajectoryDiffResult(
        diff_id="S1:T1:prediction-diff:30:60",
        target_id="T1",
        previous_prediction_id="P1",
        current_prediction_id="P2",
        previous_sim_time_s=30,
        current_sim_time_s=60,
        status="comparable",
        reason=None,
        overlap_start_s=90.0,
        overlap_end_s=390.0,
        overlap_duration_s=300.0,
        comparison_step_s=30.0,
        sample_count=11,
        absolute_rms_m=300.0,
        normalized_rms=3.0,
        p90_distance_m=350.0,
        max_distance_m=400.0,
        max_distance_time_s=390.0,
        js_distance=0.25,
        previous_leading_model="cv",
        current_leading_model="left_turn",
        leading_model_changed=True,
        previous_evidence_ids=("O1",),
        current_evidence_ids=("O2",),
        normalized_threshold=2.45,
        absolute_floor_m=250.0,
        reset_normalized_threshold=1.75,
        reset_absolute_floor_m=150.0,
        threshold_schema_version="trajectory-diff-v1",
        confirmation_cycles=2,
        exceeded=True,
    )
    assert TrajectoryDiffResult.model_validate_json(result.model_dump_json()) == result
```

- [ ] **Step 5: Add contracts and state channels**

Add these exact aliases and models to `domain/agent_models.py`:

```python
PredictionRegime = Literal["public_prior", "short_history", "bspline"]
TrajectoryDiffStatus = Literal[
    "comparable",
    "first_prediction",
    "no_new_evidence",
    "insufficient_overlap",
    "predictor_regime_reset",
    "target_mismatch",
    "invalid_prediction",
]


class TrajectoryDiffResult(StrictModel):
    diff_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    previous_prediction_id: str | None = None
    current_prediction_id: str = Field(min_length=1)
    previous_sim_time_s: int | None = Field(default=None, ge=0)
    current_sim_time_s: int = Field(ge=0)
    status: TrajectoryDiffStatus
    reason: str | None = None
    overlap_start_s: float | None = None
    overlap_end_s: float | None = None
    overlap_duration_s: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    comparison_step_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    sample_count: int = Field(default=0, ge=0)
    absolute_rms_m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    normalized_rms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    p90_distance_m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_distance_m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_distance_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    js_distance: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    previous_leading_model: str | None = None
    current_leading_model: str | None = None
    leading_model_changed: bool = False
    previous_evidence_ids: tuple[str, ...] = ()
    current_evidence_ids: tuple[str, ...] = ()
    normalized_threshold: float = Field(gt=0, allow_inf_nan=False)
    absolute_floor_m: float = Field(gt=0, allow_inf_nan=False)
    reset_normalized_threshold: float = Field(ge=0, allow_inf_nan=False)
    reset_absolute_floor_m: float = Field(ge=0, allow_inf_nan=False)
    threshold_schema_version: str = Field(min_length=1)
    confirmation_cycles: int = Field(ge=1)
    exceeded: bool = False
    consecutive_count: int = Field(default=0, ge=0)
    latched: bool = False
    gate_transition: Literal[
        "none", "accumulating", "suspected", "verifying", "confirmed", "reset"
    ] = "none"


class TrajectoryDiffGateState(StrictModel):
    target_id: str = Field(min_length=1)
    consecutive_count: int = Field(default=0, ge=0)
    latched: bool = False
    verification_pending: bool = False
    suspicion_event_id: str | None = None
    latest_diff_id: str | None = None
```

Extend `PredictedTrackRef` with default-compatible fields:

```python
prediction_regime: PredictionRegime = "short_history"
imm_model_probabilities: dict[str, float] = Field(default_factory=dict)
```

Extend `CarrierState` with:

```python
prediction_diffs: dict[str, TrajectoryDiffResult]
prediction_diff_gates: dict[str, TrajectoryDiffGateState]
prediction_intent_verification_target_ids: tuple[str, ...]
prediction_intent_confirmed: bool
```

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/agent/test_agent_loader.py tests/domain/test_agent_models.py -q
```

Expected: PASS.

Commit:

```bash
git add configs/agent.yaml src/underwater_tracking/config/models.py src/underwater_tracking/domain/agent_models.py src/underwater_tracking/agent/state.py tests/agent/test_agent_loader.py tests/domain/test_agent_models.py
git commit -m "feat: define trajectory diff contracts"
```

---

### Task 2: Pure Time-Aligned Trajectory Comparator

**Files:**
- Create: `src/underwater_tracking/prediction/diff.py`
- Create: `tests/prediction/test_diff.py`
- Create: `tests/prediction/test_diff_properties.py`
- Modify: `src/underwater_tracking/prediction/__init__.py`

**Interfaces:**
- Consumes: `PredictedTrackRef`, `TrajectoryDiffConfig`.
- Produces: `compare_predicted_tracks(previous: PredictedTrackRef | None, current: PredictedTrackRef, config: TrajectoryDiffConfig) -> TrajectoryDiffResult` and `jensen_shannon_distance(left, right) -> float | None`.

- [ ] **Step 1: Write failing alignment and dual-gate tests**

Create helpers that build straight `PredictedTrackRef` objects, then add:

```python
def test_equal_absolute_path_with_rolled_window_has_zero_diff() -> None:
    old = prediction("P1", 0, (30, 60, 90, 120), lambda t: (2.0 * t, 0.0))
    new = prediction("P2", 30, (60, 90, 120, 150), lambda t: (2.0 * t, 0.0), evidence=("O2",))
    result = compare_predicted_tracks(old, new, config(minimum_overlap_s=60.0))
    assert result.status == "comparable"
    assert result.absolute_rms_m == pytest.approx(0.0)
    assert result.normalized_rms == pytest.approx(0.0)
    assert result.exceeded is False


def test_both_normalized_and_absolute_gates_are_required() -> None:
    old = prediction("P1", 0, TIMES, lambda t: (t, 0.0), radius=1_000.0)
    absolute_only = prediction("P2", 30, TIMES, lambda t: (t + 300.0, 0.0), radius=1_000.0, evidence=("O2",))
    assert not compare_predicted_tracks(old, absolute_only, CONFIG).exceeded

    narrow_old = old.model_copy(update={"corridor_radius_m": (1.0,) * len(TIMES)})
    normalized_only = prediction("P3", 30, TIMES, lambda t: (t + 100.0, 0.0), radius=1.0, evidence=("O3",))
    assert not compare_predicted_tracks(narrow_old, normalized_only, CONFIG).exceeded

    both = prediction("P4", 30, TIMES, lambda t: (t + 300.0, 0.0), radius=1.0, evidence=("O4",))
    assert compare_predicted_tracks(narrow_old, both, CONFIG).exceeded
```

- [ ] **Step 2: Run comparator tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/prediction/test_diff.py -q
```

Expected: collection FAIL because `prediction.diff` does not exist.

- [ ] **Step 3: Implement validation, alignment, metrics, and JSD**

Implement the public entry point with private helpers:

```python
def compare_predicted_tracks(
    previous: PredictedTrackRef | None,
    current: PredictedTrackRef,
    config: TrajectoryDiffConfig,
) -> TrajectoryDiffResult:
    if previous is None:
        return _unavailable(current, config, "first_prediction", "no previous prediction")
    invalid = _validation_reason(previous, current)
    if invalid is not None:
        return _unavailable(current, config, "invalid_prediction", invalid, previous)
    if previous.target_id != current.target_id:
        return _unavailable(current, config, "target_mismatch", "target ids differ", previous)
    if previous.prediction_regime != current.prediction_regime:
        return _unavailable(current, config, "predictor_regime_reset", "prediction regime changed", previous)
    if not set(current.source_belief_history_ids).difference(previous.source_belief_history_ids):
        return _unavailable(current, config, "no_new_evidence", "no new source observation id", previous)

    grid = _comparison_grid(previous, current, config)
    if grid is None:
        return _unavailable(current, config, "insufficient_overlap", "overlap below configured minimum", previous)
    old_xy, old_r = _interpolate(previous, grid)
    new_xy, new_r = _interpolate(current, grid)
    delta = new_xy - old_xy
    distance = np.linalg.norm(delta, axis=1)
    sigma = np.sqrt(old_r**2 + new_r**2 + config.uncertainty_floor_m**2)
    normalized = distance / sigma
    weights = np.exp(-(grid - float(current.sim_time_s)) / config.near_term_decay_s)
    absolute_rms = _weighted_rms(distance, weights)
    normalized_rms = _weighted_rms(normalized, weights)
    exceeded = (
        normalized_rms >= config.normalized_threshold
        and absolute_rms >= config.absolute_floor_m
    )
    max_index = int(np.argmax(distance))
    return TrajectoryDiffResult(
        diff_id=f"{current.prediction_id}:diff-from:{previous.prediction_id}",
        target_id=current.target_id,
        previous_prediction_id=previous.prediction_id,
        current_prediction_id=current.prediction_id,
        previous_sim_time_s=previous.sim_time_s,
        current_sim_time_s=current.sim_time_s,
        status="comparable",
        overlap_start_s=float(grid[0]),
        overlap_end_s=float(grid[-1]),
        overlap_duration_s=float(grid[-1] - grid[0]),
        comparison_step_s=float(grid[1] - grid[0]),
        sample_count=len(grid),
        absolute_rms_m=absolute_rms,
        normalized_rms=normalized_rms,
        p90_distance_m=_weighted_quantile(distance, weights, 0.90),
        max_distance_m=float(distance[max_index]),
        max_distance_time_s=float(grid[max_index]),
        js_distance=jensen_shannon_distance(previous.imm_model_probabilities, current.imm_model_probabilities),
        previous_leading_model=_leading_model(previous.imm_model_probabilities),
        current_leading_model=_leading_model(current.imm_model_probabilities),
        leading_model_changed=_leading_model(previous.imm_model_probabilities) != _leading_model(current.imm_model_probabilities),
        previous_evidence_ids=tuple(sorted(previous.source_belief_history_ids)),
        current_evidence_ids=tuple(sorted(current.source_belief_history_ids)),
        normalized_threshold=config.normalized_threshold,
        absolute_floor_m=config.absolute_floor_m,
        reset_normalized_threshold=config.reset_normalized_threshold,
        reset_absolute_floor_m=config.reset_absolute_floor_m,
        threshold_schema_version=config.schema_version,
        confirmation_cycles=config.confirmation_cycles,
        exceeded=exceeded,
    )
```

Use `numpy.interp` independently for x, y, and radius. Reject empty arrays, unequal lengths, non-increasing times, negative radius, and any non-finite input. Implement JSD with base-2 logs and return `sqrt(divergence)` so the result is symmetric and bounded `[0,1]`; return `None` if either map is empty.

- [ ] **Step 4: Add invalid-input and property tests**

Cover `first_prediction`, `no_new_evidence`, `predictor_regime_reset`, target mismatch, no overlap, NaN, negative radius, non-monotonic times, mixed sample steps, common translation, common rotation, JSD symmetry, and weighted P90.

Use a fixed seed and 100 generated rigid transforms without adding Hypothesis as a dependency:

```python
def test_common_rigid_transforms_preserve_scores() -> None:
    random = np.random.default_rng(42)
    baseline = compare_predicted_tracks(OLD, NEW, CONFIG)
    for _ in range(100):
        angle = random.uniform(-np.pi, np.pi)
        translation = random.uniform(-10_000.0, 10_000.0, size=2)
        transformed_old = rigid_transform(OLD, angle, translation)
        transformed_new = rigid_transform(NEW, angle, translation)
        result = compare_predicted_tracks(transformed_old, transformed_new, CONFIG)
        assert result.absolute_rms_m == pytest.approx(baseline.absolute_rms_m)
        assert result.normalized_rms == pytest.approx(baseline.normalized_rms)
```

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/prediction/test_diff.py tests/prediction/test_diff_properties.py -q
```

Expected: PASS.

Commit:

```bash
git add src/underwater_tracking/prediction/diff.py src/underwater_tracking/prediction/__init__.py tests/prediction/test_diff.py tests/prediction/test_diff_properties.py
git commit -m "feat: compare uncertainty-aware target forecasts"
```

---

### Task 3: Public IMM Metadata and Predictor Regimes

**Files:**
- Modify: `src/underwater_tracking/prediction/port.py`
- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Test: `tests/prediction/test_port.py`
- Test: `tests/agent/test_prior_seeded_planning.py`

**Interfaces:**
- Consumes: `GroupReport.belief.model_probabilities` and `PredictedTrackRef` fields from Task 1.
- Produces: every forecast explicitly identifies `prediction_regime` and sorted public IMM model probabilities.

- [ ] **Step 1: Write failing metadata tests**

```python
def test_short_history_prediction_carries_public_imm_metadata() -> None:
    prediction = predictor(snapshot_with_probabilities({"cv": 0.7, "left_turn": 0.2, "right_turn": 0.1}), "T1")
    assert prediction.prediction_regime == "short_history"
    assert prediction.imm_model_probabilities == {"cv": 0.7, "left_turn": 0.2, "right_turn": 0.1}


def test_public_prior_prediction_has_explicit_prior_regime() -> None:
    prediction = _prior_seeded_planning_inputs(SITUATION)["predictions"]["T1"]
    assert prediction.prediction_regime == "public_prior"
    assert prediction.imm_model_probabilities == {}
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/prediction/test_port.py tests/agent/test_prior_seeded_planning.py -q
```

Expected: FAIL because the producer does not set explicit regimes/probabilities.

- [ ] **Step 3: Attach only public estimator metadata**

In `_ref_from_prediction` and `_short_history_ref`, set:

```python
prediction_regime="bspline" if not fallback_used else "short_history",
imm_model_probabilities=(
    dict(sorted(report.belief.model_probabilities.items())) if report is not None else {}
),
```

In `_prior_seeded_planning_inputs`, set:

```python
prediction_regime="public_prior",
imm_model_probabilities={},
```

Validate probabilities as finite, non-negative, and positive-sum at the `PredictedTrackRef` model boundary. Normalize only inside JSD; do not silently rewrite persisted estimator output.

- [ ] **Step 4: Run prediction tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/prediction tests/agent/test_prior_seeded_planning.py -q
```

Expected: PASS.

Commit:

```bash
git add src/underwater_tracking/prediction/port.py src/underwater_tracking/agent/graphs/central.py tests/prediction/test_port.py tests/agent/test_prior_seeded_planning.py
git commit -m "feat: preserve IMM forecast provenance"
```

---

### Task 4: Persistent Diff Gate and Correct Event Semantics

**Files:**
- Create: `src/underwater_tracking/prediction/diff_gate.py`
- Create: `tests/prediction/test_diff_gate.py`
- Modify: `src/underwater_tracking/domain/event_registry.py`
- Modify: `src/underwater_tracking/agent/nodes/event_monitor.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `tests/agent/test_event_monitor.py`
- Modify: `tests/agent/test_event_policy.py`
- Modify: `tests/simulation/test_engine.py`

**Interfaces:**
- Consumes: `TrajectoryDiffResult`, `TrajectoryDiffGateState`, `TrajectoryDiffConfig`.
- Produces: `advance_diff_gate(...) -> TrajectoryDiffGateDecision`; corrected `imm_motion_mode_changed`, `target_intent_change_suspected`, and semantic `target_intent_changed` events.

- [ ] **Step 1: Write failing gate state-transition tests**

```python
def test_gate_requires_two_exceedances_and_latches_once() -> None:
    first = advance_diff_gate(None, exceeded_diff("D1"), CONFIG)
    assert first.state.consecutive_count == 1
    assert not first.emit_suspicion
    second = advance_diff_gate(first.state, exceeded_diff("D2"), CONFIG)
    assert second.emit_suspicion
    assert second.request_intent_verification
    third = advance_diff_gate(second.state, exceeded_diff("D3"), CONFIG)
    assert not third.emit_suspicion
    assert third.request_intent_verification


def test_gate_releases_on_either_lower_reset_threshold() -> None:
    state = TrajectoryDiffGateState(target_id="T1", latched=True, verification_pending=True)
    decision = advance_diff_gate(state, comparable_diff(normalized=1.7, absolute=500.0), CONFIG)
    assert decision.state.latched is False
    assert decision.state.verification_pending is False
```

- [ ] **Step 2: Run gate tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/prediction/test_diff_gate.py -q
```

Expected: collection FAIL because `diff_gate.py` does not exist.

- [ ] **Step 3: Implement the pure gate**

Define:

```python
@dataclass(frozen=True, slots=True)
class TrajectoryDiffGateDecision:
    state: TrajectoryDiffGateState
    emit_suspicion: bool
    request_intent_verification: bool
    reset: bool


def advance_diff_gate(
    previous: TrajectoryDiffGateState | None,
    diff: TrajectoryDiffResult,
    config: TrajectoryDiffConfig,
) -> TrajectoryDiffGateDecision:
    state = previous or TrajectoryDiffGateState(target_id=diff.target_id)
    if diff.status != "comparable":
        cleared = TrajectoryDiffGateState(target_id=diff.target_id, latest_diff_id=diff.diff_id)
        return TrajectoryDiffGateDecision(cleared, False, False, state.latched or state.consecutive_count > 0)
    normalized = diff.normalized_rms or 0.0
    absolute = diff.absolute_rms_m or 0.0
    if state.latched and (
        normalized < config.reset_normalized_threshold
        or absolute < config.reset_absolute_floor_m
    ):
        cleared = TrajectoryDiffGateState(target_id=diff.target_id, latest_diff_id=diff.diff_id)
        return TrajectoryDiffGateDecision(cleared, False, False, True)
    if state.latched:
        kept = state.model_copy(update={"latest_diff_id": diff.diff_id})
        return TrajectoryDiffGateDecision(kept, False, kept.verification_pending, False)
    if not diff.exceeded:
        cleared = TrajectoryDiffGateState(target_id=diff.target_id, latest_diff_id=diff.diff_id)
        return TrajectoryDiffGateDecision(cleared, False, False, state.consecutive_count > 0)
    count = state.consecutive_count + 1
    emit = count >= config.confirmation_cycles
    updated = TrajectoryDiffGateState(
        target_id=diff.target_id,
        consecutive_count=count,
        latched=emit,
        verification_pending=emit,
        suspicion_event_id=None,
        latest_diff_id=diff.diff_id,
    )
    return TrajectoryDiffGateDecision(updated, emit, emit, False)
```

- [ ] **Step 4: Write failing event-semantics tests**

Update tests to assert:

```python
assert event_definition("target_intent_change_suspected").default_level == EventLevel.TACTICAL
assert event_definition("target_intent_changed").default_level == EventLevel.STRATEGIC
assert event_definition("imm_motion_mode_changed").default_level == EventLevel.INFORMATIONAL
assert EVENT_REGISTRY["target_intent_changed"].audiences == PUBLIC_AUDIENCES
```

Change the engine test so two estimator mode changes produce
`imm_motion_mode_changed` and `imm_confidence_shifted`, never
`target_intent_changed`.

- [ ] **Step 5: Correct event producers and payload validation**

Register the new events with families `prediction_diff`, `intent`, and
`imm_motion`; remove `target_intent_changed` from informational overrides.
Require suspicion payload keys:

```python
{
    "diff_id",
    "previous_prediction_id",
    "current_prediction_id",
    "observation_ids",
    "absolute_rms_m",
    "normalized_rms",
    "absolute_floor_m",
    "normalized_threshold",
    "consecutive_count",
    "source",
}
```

Rename the estimator-only engine event and payload source:

```python
event_type="imm_motion_mode_changed"
payload={**payload, "motion_model": label, "source": "public_imm_belief"}
```

Change `EventMonitor.observe_intent_analysis` to emit
`target_intent_changed` at `EventLevel.STRATEGIC`; retain
`intent_change_confirmed` only as a read-compatible legacy registry event,
never as the new producer.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/prediction/test_diff_gate.py tests/agent/test_event_monitor.py tests/agent/test_event_policy.py tests/simulation/test_engine.py -q
```

Expected: PASS.

Commit:

```bash
git add src/underwater_tracking/prediction/diff_gate.py src/underwater_tracking/domain/event_registry.py src/underwater_tracking/agent/nodes/event_monitor.py src/underwater_tracking/simulation/engine.py tests/prediction/test_diff_gate.py tests/agent/test_event_monitor.py tests/agent/test_event_policy.py tests/simulation/test_engine.py
git commit -m "feat: gate prediction divergence events"
```

---

### Task 5: Central Graph Diff Detection and Real Intent Verification

**Files:**
- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Modify: `src/underwater_tracking/agent/nodes/intent.py`
- Modify: `src/underwater_tracking/agent/prompts.py`
- Modify: `src/underwater_tracking/agent/runtime.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `tests/agent/test_central_graph.py`
- Modify: `tests/agent/test_semantic_nodes.py`
- Modify: `tests/agent/test_background_cycle.py`

**Interfaces:**
- Consumes: comparator/gate interfaces and `CarrierDependencies.trajectory_diff_config`.
- Produces: checkpointed `prediction_diffs`, suspicion events, `prediction_intent_analysis` graph node, and a no-loop confirmed/unconfirmed route.

- [ ] **Step 1: Write failing trajectory-node tests**

Instantiate `TrajectoryPredictionNode` with a deterministic predictor and assert:

```python
first = node({"scenario_id": "S1", "snapshot_ref": REF})
assert first["prediction_diffs"]["T1"].status == "first_prediction"
second = node({**first, "snapshot_ref": NEXT_REF})
assert second["prediction_diff_gates"]["T1"].consecutive_count == 1
third = node({**first, **second, "snapshot_ref": THIRD_REF})
assert third["prediction_intent_verification_target_ids"] == ("T1",)
assert [event.event_type for event in third["coalesced_events"]] == [
    "target_intent_change_suspected"
]
```

Also assert identical/low diffs never request semantic analysis and a regime reset clears the count.

- [ ] **Step 2: Run graph tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/agent/test_central_graph.py -q
```

Expected: FAIL because prediction node returns only `predictions`.

- [ ] **Step 3: Wire comparator and gate into `TrajectoryPredictionNode`**

Add `diff_config` to the constructor. Before replacing predictions, read the checkpointed previous map. For each target, call the comparator then gate. Attach gate state to the persisted result:

```python
annotated = diff.model_copy(
    update={
        "consecutive_count": decision.state.consecutive_count,
        "latched": decision.state.latched,
        "gate_transition": (
            "reset" if decision.reset
            else "suspected" if decision.emit_suspicion
            else "verifying" if decision.request_intent_verification
            else "accumulating" if decision.state.consecutive_count
            else "none"
        ),
    }
)
```

Build the public suspicion event using `event_definition(...)` and the complete metric payload. Append it to current `coalesced_events` without duplicating previous-cycle events. Return all four channels:

The central node, not the pure gate, creates the globally unique event ID
`{scenario_id}:target_intent_change_suspected:{target_id}:{sim_time_s}` and
copies it into `TrajectoryDiffGateState.suspicion_event_id`. It also annotates
the result's `gate_transition` as `accumulating`, `suspected`, `verifying`, or
`reset`; the semantic wrapper later changes it to `confirmed` when applicable.

```python
return {
    "predictions": predictions,
    "prediction_diffs": diffs,
    "prediction_diff_gates": gates,
    "prediction_intent_verification_target_ids": tuple(sorted(pending)),
    "coalesced_events": (*current_events, *emitted),
}
```

- [ ] **Step 4: Write failing no-loop semantic-route tests**

Add an integration rig whose scripted structured LLM returns two high-confidence
`evade` hypotheses and count operations:

```python
assert first_result["prediction_intent_confirmed"] is False
assert llm.operations.count("intent") == 1
assert second_result["prediction_intent_confirmed"] is True
assert llm.operations.count("intent") == 2
assert "regional_strategy" in llm.operations
assert not any(
    edge.source == "prediction_intent_analysis"
    and edge.target == "trajectory_prediction"
    for edge in graph.get_graph().edges
)
```

Add unchanged-label and low-confidence cases that route directly to
`resource_optimizer` with no regional LLM call.

- [ ] **Step 5: Add a dedicated post-prediction intent wrapper**

Create `PredictionIntentWiringNode` around `IntentAnalysisNode` and
`EventMonitor`. It analyzes only `prediction_intent_verification_target_ids`,
adds the current diff ID and suspicion event ID to evidence, and adds matching
`LLMCallMetadata` fields to a confirmed event payload. Its return includes:

```python
{
    "intent_hypotheses": hypotheses,
    "llm_provenance": provenance,
    "confirmed_intent_labels": confirmed,
    "prediction_diff_gates": updated_gates,
    "prediction_intent_verification_target_ids": remaining,
    "prediction_intent_confirmed": bool(confirmed_events),
    "coalesced_events": (*events, *confirmed_events),
}
```

An unchanged label or failed confidence/margin gate sets
`verification_pending=False` while retaining the latch until geometric reset.
A changed high-confidence label that has only one semantic confirmation keeps
`verification_pending=True` for the next observation cycle.

Add an optional `intent_target_ids` state filter to `IntentAnalysisNode`; when
present it intersects the public snapshot targets and analyzes only those IDs.
The existing pre-prediction strategic node leaves the filter absent and keeps
its current all-target behavior.

Extend the intent payload with a bounded `trajectory_diff` object containing
the approved metrics and IDs. Do not add target truth, private adversary events,
or heuristic intent labels.

- [ ] **Step 6: Wire conditional edges without a cycle**

Register a second graph node named `prediction_intent_analysis` using the same
real `dependencies.llm`. Change `_route_after_prediction` to return
`"intent_verification" | "strategic" | "tactical"`. Add:

```python
def _route_after_prediction_intent(
    state: CentralState,
) -> Literal["strategic", "tactical", "error"]:
    if state.get("node_error") is not None:
        return "error"
    return "strategic" if state.get("prediction_intent_confirmed") else "tactical"
```

Wire:

```text
trajectory_prediction --intent_verification--> prediction_intent_analysis
prediction_intent_analysis --strategic--> regional_generation
prediction_intent_analysis --tactical--> resource_optimizer
prediction_intent_analysis --error--> handle_error
```

Never add an edge from `prediction_intent_analysis` back to
`trajectory_prediction`.

- [ ] **Step 7: Inject configured thresholds through runtime/CLI**

Add `trajectory_diff_config: TrajectoryDiffConfig = field(default_factory=TrajectoryDiffConfig)`
to `CarrierDependencies`. At CLI composition, pass
`config.agent.trajectory_diff if config.agent else TrajectoryDiffConfig()`.
Ensure reopen/resume reads gate state from LangGraph checkpoint and does not
reset the streak.

- [ ] **Step 8: Run graph/runtime tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/agent/test_central_graph.py tests/agent/test_semantic_nodes.py tests/agent/test_background_cycle.py -q
```

Expected: PASS with exact real-port invocation counts in scripted tests.

Commit:

```bash
git add src/underwater_tracking/agent/graphs/central.py src/underwater_tracking/agent/nodes/intent.py src/underwater_tracking/agent/prompts.py src/underwater_tracking/agent/runtime.py src/underwater_tracking/cli.py tests/agent/test_central_graph.py tests/agent/test_semantic_nodes.py tests/agent/test_background_cycle.py
git commit -m "feat: verify prediction divergence with intent LLM"
```

---

### Task 6: Durable Evidence, Live/Replay Frame Contract, and Memory Source

**Files:**
- Modify: `src/underwater_tracking/api/live.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/memory/source_reader.py`
- Modify: `tests/api/test_frame_contracts.py`
- Modify: `tests/api/test_frame_pipeline.py`
- Modify: `tests/api/test_live_publisher.py`
- Modify: `tests/api/test_replay_compatibility.py`
- Modify: `tests/memory/test_source_reader.py`

**Interfaces:**
- Consumes: checkpoint `prediction_diffs` and durable suspicion/confirmation events.
- Produces: `PredictionDiffView` inside `PredictionCorridorView`; identical live/replay JSON; memory entries sourced from durable events.

- [ ] **Step 1: Write failing frame round-trip and live projection tests**

```python
def test_prediction_diff_round_trips_in_operational_frame() -> None:
    frame = build_operational_frame(
        SNAPSHOT,
        PLAN,
        (),
        (),
        (),
        predictions={"T1": PREDICTION},
        prediction_diffs={"T1": DIFF},
    )
    view = frame.target_estimates[0].prediction.diff
    assert view.state == "suspected"
    assert view.absolute_rms_m == 300.0
    assert view.normalized_rms == 3.0
    assert OperationalFrame.model_validate_json(frame.model_dump_json()) == frame
```

Assert `LiveFramePublisher` reads `prediction_diffs` from runtime state and
legacy replay rows without the field deserialize with `diff=None`.

- [ ] **Step 2: Add strict UI view contracts**

Add:

```python
class PredictionDiffView(StrictModel):
    diff_id: str
    state: Literal[
        "stable", "accumulating", "suspected", "verifying",
        "confirmed", "reset", "unavailable"
    ]
    status: str
    reason: str | None = None
    absolute_rms_m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    normalized_rms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    absolute_floor_m: float = Field(gt=0, allow_inf_nan=False)
    normalized_threshold: float = Field(gt=0, allow_inf_nan=False)
    consecutive_count: int = Field(ge=0)
    confirmation_cycles: int = Field(ge=1)
    previous_prediction_id: str | None = None
    current_prediction_id: str
    leading_model_changed: bool = False
    js_distance: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    suspicion_event_id: str | None = None
    confirmed_intent: str | None = None
    resulting_plan_revision: int | None = Field(default=None, ge=1)
```

Add `diff: PredictionDiffView | None = None` to `PredictionCorridorView`.

- [ ] **Step 3: Project one authoritative state into frames**

Add `prediction_diffs` to `build_operational_frame` and `_build_estimate`.
Build state from the diff result plus current gate and durable events; do not
recalculate metrics. In `api/live.py` parse strict mappings:

```python
prediction_diffs = _mapping_of(state.get("prediction_diffs"), TrajectoryDiffResult)
prediction_gates = _mapping_of(state.get("prediction_diff_gates"), TrajectoryDiffGateState)
```

Pass them through `build_operational_frame`. Replay uses the persisted frame,
so no replay-only calculation is added.

- [ ] **Step 4: Ensure memory consumes event evidence once**

Add new event types to the source reader's accepted event taxonomy and assert
the source cursor writes one memory work item per durable event ID. The memory
payload must retain `diff_id`, prediction IDs, observation IDs, and confirmed
LLM provenance IDs; it must not copy target truth or raw prompts.

- [ ] **Step 5: Run API/memory tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/api/test_frame_contracts.py tests/api/test_frame_pipeline.py tests/api/test_live_publisher.py tests/api/test_replay_compatibility.py tests/memory/test_source_reader.py -q
```

Expected: PASS.

Commit:

```bash
git add src/underwater_tracking/api/live.py src/underwater_tracking/api/frame_builder.py src/underwater_tracking/domain/ui_models.py src/underwater_tracking/memory/source_reader.py tests/api/test_frame_contracts.py tests/api/test_frame_pipeline.py tests/api/test_live_publisher.py tests/api/test_replay_compatibility.py tests/memory/test_source_reader.py
git commit -m "feat: expose trajectory diff evidence"
```

---

### Task 7: Command-Center Prediction Diff UI

**Files:**
- Create: `src/underwater_tracking/ui/src/components/PredictionDiffPanel.tsx`
- Create: `src/underwater_tracking/ui/src/components/PredictionDiffPanel.test.tsx`
- Modify: `src/underwater_tracking/ui/src/types/frames.ts`
- Modify: `src/underwater_tracking/ui/src/App.tsx`
- Modify: `src/underwater_tracking/ui/src/components/PlaybackBar.tsx`
- Modify: `src/underwater_tracking/ui/src/components/SidebarPanels.css`
- Modify: `src/underwater_tracking/ui/e2e/command-center.spec.ts`

**Interfaces:**
- Consumes: `TargetEstimateView.prediction.diff` only.
- Produces: compact current-state metrics and timeline labels with no frontend recomputation.

- [ ] **Step 1: Add failing component tests**

```tsx
it("distinguishes suspected prediction divergence from confirmed intent", () => {
  render(<PredictionDiffPanel targets={[targetWithDiff({ state: "suspected" })]} />);
  expect(screen.getByText("疑似行为变化")).toBeInTheDocument();
  expect(screen.getByText("300 m")).toBeInTheDocument();
  expect(screen.getByText("3.00 / 2.45")).toBeInTheDocument();
  expect(screen.queryByText("意图已改变")).not.toBeInTheDocument();
});

it("renders unavailable reasons without showing a zero score", () => {
  render(<PredictionDiffPanel targets={[targetWithDiff({ state: "unavailable", reason: "insufficient_overlap", absolute_rms_m: null, normalized_rms: null })]} />);
  expect(screen.getByText("证据不足")).toBeInTheDocument();
  expect(screen.queryByText("0.00")).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run Vitest and verify RED**

Run:

```bash
npm --prefix src/underwater_tracking/ui test -- --run src/components/PredictionDiffPanel.test.tsx
```

Expected: FAIL because the component/types do not exist.

- [ ] **Step 3: Add TypeScript contracts and component**

Mirror the Python fields exactly. Render a stable two-column metric strip with
`absolute_rms_m / absolute_floor_m`, `normalized_rms / normalized_threshold`,
confirmation count, status badge, and previous/current prediction IDs. Use
plain semantic text and existing CSS tokens; do not add gradients, decorative
cards, animations, or nested cards.

Use status labels:

```typescript
const STATE_LABEL: Record<PredictionDiffView["state"], string> = {
  stable: "轨迹稳定",
  accumulating: "变化累积",
  suspected: "疑似行为变化",
  verifying: "意图核验中",
  confirmed: "意图已改变",
  reset: "变化已解除",
  unavailable: "证据不足",
};
```

- [ ] **Step 4: Wire UI and timeline**

Render `PredictionDiffPanel` above `AssignmentPanel` in `App.tsx`. Add
`target_intent_change_suspected`, `imm_motion_mode_changed`, and
`target_intent_changed` to playback color/label maps, with distinct labels
`预测分歧`, `IMM 模式变化`, and `意图确认`.

- [ ] **Step 5: Test desktop/mobile layout and commit**

Run:

```bash
npm --prefix src/underwater_tracking/ui test -- --run
npm --prefix src/underwater_tracking/ui run build
npm --prefix src/underwater_tracking/ui run test:e2e -- command-center.spec.ts
```

Expected: all Vitest tests pass, TypeScript build succeeds, and Playwright
desktop/mobile assertions find no overlap or horizontal overflow.

Commit:

```bash
git add src/underwater_tracking/ui/src/components/PredictionDiffPanel.tsx src/underwater_tracking/ui/src/components/PredictionDiffPanel.test.tsx src/underwater_tracking/ui/src/types/frames.ts src/underwater_tracking/ui/src/App.tsx src/underwater_tracking/ui/src/components/PlaybackBar.tsx src/underwater_tracking/ui/src/components/SidebarPanels.css src/underwater_tracking/ui/e2e/command-center.spec.ts
git commit -m "feat: show prediction divergence evidence"
```

---

### Task 8: Acceptance Monitor and Diff-to-Plan Causal Proof

**Files:**
- Modify: `src/underwater_tracking/verification/live_demo.py`
- Modify: `scripts/monitor_main_battle.py`
- Modify: `tests/verification/test_live_demo_monitor.py`
- Modify: `tests/integration/test_uuv_only_production_acceptance.py`
- Modify: `docs/verification/main-live-battle-acceptance.md`

**Interfaces:**
- Consumes: durable event/LLM/decision/plan records and frame diff views.
- Produces: JSON and Markdown evidence sections for every target prediction-diff chain; FAIL/BLOCKED on missing provider/provenance/link.

- [ ] **Step 1: Write failing report-chain tests**

Build a repository fixture with a complete chain and assert:

```python
chain = report.prediction_intent_chains[0]
assert chain.diff_id == "D1"
assert chain.suspicion_event_id == "E-suspect"
assert chain.intent_llm_call_ids == ("LLM-1", "LLM-2")
assert chain.confirmed_event_id == "E-confirmed"
assert chain.resulting_plan_revision == 3
assert chain.blue_response_event_ids
```

Delete each link in parametrized cases and assert a specific violation:
`missing_prediction_diff`, `missing_real_intent_provider`,
`missing_intent_confirmation`, `missing_regional_replan`,
`missing_committed_plan`, or `missing_blue_response`.

- [ ] **Step 2: Run verification tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/verification/test_live_demo_monitor.py -q
```

Expected: FAIL because the report has no prediction-intent chain section.

- [ ] **Step 3: Add strict acceptance evidence extraction**

Resolve chains by durable IDs, not time proximity alone:

```text
target_intent_change_suspected.payload.diff_id
  -> event observation/prediction IDs
  -> llm_calls request/response hashes and non-empty model
  -> target_intent_changed payload suspicion_event_id/diff_id
  -> decision trigger_event_ids
  -> committed plan revision
  -> later physical blue response event
```

Reject `error_category != ""`, empty provider model, fake/test model IDs, missing
hashes, unknown evidence IDs, and confirmation without two qualifying real
Intent calls.

- [ ] **Step 4: Render JSON and Markdown sections**

Add a table with target, diff metrics/thresholds, comparison window, suspicion
time, Intent provider/model, confirmation time, plan revision, latency, and
blue response. Keep FAIL/BLOCKED visible in both desktop/mobile screenshots and
the report summary.

- [ ] **Step 5: Run monitor/integration tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/verification/test_live_demo_monitor.py tests/integration/test_uuv_only_production_acceptance.py -q
```

Expected: PASS.

Commit:

```bash
git add src/underwater_tracking/verification/live_demo.py scripts/monitor_main_battle.py tests/verification/test_live_demo_monitor.py tests/integration/test_uuv_only_production_acceptance.py docs/verification/main-live-battle-acceptance.md
git commit -m "test: require prediction intent evidence chain"
```

---

### Task 9: Regression, Real Provider Run, and Goal Acceptance Continuation

**Files:**
- Modify only files required by failures proven during this task.
- Generate: `docs/verification/main-live-battle-acceptance.json`
- Generate: `docs/verification/main-live-battle-acceptance.md`
- Generate: desktop/mobile screenshots under the accepted verification run directory.

**Interfaces:**
- Consumes: all prior tasks and the existing full-system goal acceptance monitor.
- Produces: test evidence and a real `main.py` run through `sim_time_s=28800`; does not declare PASS from partial or fake evidence.

- [ ] **Step 1: Run focused backend and static checks**

```bash
PYTHONPATH=src pytest tests/prediction tests/agent/test_central_graph.py tests/agent/test_event_monitor.py tests/simulation/test_engine.py tests/api/test_frame_contracts.py tests/api/test_frame_pipeline.py tests/api/test_live_publisher.py tests/verification/test_live_demo_monitor.py -q
ruff check src tests scripts/monitor_main_battle.py
python -m compileall -q src tests scripts
git diff --check
```

Expected: zero failures and zero formatting/static errors.

- [ ] **Step 2: Run complete backend/frontend suites**

```bash
PYTHONPATH=src pytest -q
npm --prefix src/underwater_tracking/ui test -- --run
npm --prefix src/underwater_tracking/ui run build
npm --prefix src/underwater_tracking/ui run test:e2e
```

Expected: all configured tests pass. Any environment-only skipped test remains
listed and cannot prove the real-provider requirement.

- [ ] **Step 3: Run a bounded real-provider diagnostic**

```bash
PYTHONPATH=src python scripts/monitor_main_battle.py \
  --main main.py \
  --scenario configs/scenario/uuv_only_single_target.yaml \
  --wall-timeout-s 420 \
  --expected-duration-s 1800 \
  --require-real-provider \
  --output-report docs/verification/main-live-battle-diagnostics-imm-diff.json
```

Expected: real provider calls succeed, the run reaches `sim_time_s>=1800`, no
physical/browser/request violations occur, and any maneuver chain has resolvable
diff evidence. If no sensor-visible maneuver occurs in this short window, the
diagnostic may remain incomplete but must not claim the final chain passed.

- [ ] **Step 4: Continue fixing proven full-goal blockers**

Use `superpowers:systematic-debugging` for each observed failure. In particular,
re-run the known v18 resource-lock case and prove that recovery-state UUVs are
not retained as stale hard locks while current valid locks remain enforced.
Add a failing regression test before every code fix.

- [ ] **Step 5: Run the final real eight-hour simulation-time acceptance**

```bash
PYTHONPATH=src python scripts/monitor_main_battle.py \
  --main main.py \
  --scenario configs/scenario/uuv_only_single_target.yaml \
  --wall-timeout-s 900 \
  --expected-duration-s 28800 \
  --require-real-provider \
  --output-report docs/verification/main-live-battle-acceptance.json
```

Expected only for PASS:

- `final_sim_time_s == 28800` and run phase is completed;
- all 17 entities are present in every audited physics frame;
- zero speed, acceleration, turn, boundary, depth, and teleport violations;
- zero browser console errors and zero failed requests;
- complete real adversary decision and blue counter-tracking chains;
- at least one complete prediction-diff-to-intent-to-plan chain when a
  sensor-visible maneuver occurs;
- desktop/mobile UI, Memory Steam, LLM thinking, timeline, ledger, and reports
  agree on IDs, times, metrics, and plan versions;
- real configured provider provenance exists for all required LLM stages.

If any item fails or the provider is unavailable, keep status FAIL/BLOCKED and
continue diagnosis; never generate a synthetic PASS.

- [ ] **Step 6: Review, commit, and merge only after proof**

Invoke `superpowers:requesting-code-review`, address findings, then invoke
`superpowers:verification-before-completion`. Inspect:

```bash
git status --short --branch
git diff --check
git log --oneline --decorate master..HEAD
```

Commit remaining scoped implementation/report changes without adding unrelated
user screenshots. Invoke `superpowers:finishing-a-development-branch`, merge the
feature branch into `master` non-interactively, and rerun the authoritative
tests on `master`. Do not merge while any final acceptance item is unproven.
