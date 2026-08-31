# Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan:** Underwater Tracking Evaluation and Acceptance

**Goal:** Build a reproducible experiment and acceptance system that compares the full assistant against fixed and adaptive baselines, quantifies tracking quality and UUV economy, verifies agent behavior, and prevents simulator truth leakage.

**Architecture:** Experiments are immutable manifests executed as paired-seed batches. Each run writes operational telemetry and a separate evaluation-only truth stream. A metric registry reduces runs into per-target, group, resource, agent, intent, and human-interaction results; paired aggregation produces confidence intervals and machine-readable acceptance verdicts.

**Tech Stack:** Python 3.11, Pydantic 2, NumPy, SciPy, Pandas, PyArrow, Pytest, Matplotlib, JSON, YAML.

---

**Prerequisites:** Complete the foundation, agent, and UI plans. The evaluation plan consumes their stable public contracts and does not patch internal state to obtain measurements.

## File map

- `src/underwater_tracking/evaluation/`: manifests, baselines, runner, metrics, aggregation, acceptance, and reports.
- `configs/scenario/`: canonical small, default, pressure, and event-focused scenarios.
- `configs/experiments/`: pilot and frozen formal experiment matrices.
- `tests/evaluation/`: metric, baseline, runner, determinism, statistics, and acceptance tests.
- `outputs/experiments/`: ignored generated run artifacts.

### Task 1: Define immutable experiment manifests and isolated run artifacts

**Files:**
- Create: `src/underwater_tracking/evaluation/models.py`
- Create: `src/underwater_tracking/evaluation/store.py`
- Create: `tests/evaluation/test_models_and_store.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing manifest and isolation tests**

```python
from underwater_tracking.evaluation.models import ExperimentManifest, PolicyName


def test_manifest_identity_changes_when_policy_changes(manifest_factory):
    first = manifest_factory(policy=PolicyName.FULL)
    second = manifest_factory(policy=PolicyName.B0_3)
    assert first.run_id != second.run_id


def test_truth_and_operational_streams_use_different_paths(tmp_path, artifact_store_factory):
    store = artifact_store_factory(tmp_path)
    assert store.operational_path != store.truth_path
    assert "truth" not in store.operational_path.name.lower()
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/evaluation/test_models_and_store.py -v`

Expected: FAIL importing evaluation models.

- [ ] **Step 3: Implement frozen experiment contracts**

Define `PolicyName` values `b0_2`, `b0_3`, `b1`, `b2`, and `full`. Define `AblationName` values `no_bspline`, `no_fim_waypoint`, `no_elastic_group`, `no_history_compression`, `no_llm_intent`, and `no_expert_feedback`. Define frozen strict models for `ScenarioRef`, `ModelRef`, `ExperimentManifest`, `RunPaths`, `RunStatus`, and `AcceptanceResult`.

Compute `run_id` from canonical JSON containing scenario hash, policy, seed, software commit, config hashes, prompt versions, model IDs, and evaluation schema version. Reject dirty or missing source identifiers in formal mode.

- [ ] **Step 4: Implement append-only artifact storage**

Each run directory contains:

- `manifest.json`;
- `status.json`;
- `operational/frames.jsonl` and `operational/ledger.jsonl`;
- `evaluation/truth.jsonl`;
- `metrics/per_step.parquet` and `metrics/summary.json`;
- `logs/run.log`.

Use exclusive directory creation and atomic status replacement. Never mix evaluation truth into operational files.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/evaluation/test_models_and_store.py -v`

Expected: PASS.

```bash
git add src/underwater_tracking/evaluation tests/evaluation/test_models_and_store.py .gitignore
git commit -m "feat: define reproducible experiment artifacts"
```

### Task 2: Implement comparable tracking policies

**Files:**
- Create: `src/underwater_tracking/evaluation/policies.py`
- Create: `src/underwater_tracking/evaluation/baselines.py`
- Create: `tests/evaluation/test_baselines.py`

- [ ] **Step 1: Write failing policy-invariant tests**

```python
def test_b0_3_assigns_three_when_resources_allow(policy_context):
    decision = policy_context.b0_3.decide(policy_context.snapshot)
    assert all(len(group.member_ids) == 3 for group in decision.groups)


def test_all_policies_obey_safety_and_use_estimated_state_only(policy_context):
    for policy in policy_context.all_policies:
        decision = policy.decide(policy_context.snapshot)
        assert policy_context.guard.validate(decision).is_valid
        assert policy_context.truth_access_count(policy) == 0
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/evaluation/test_baselines.py -v`

Expected: FAIL importing baselines.

- [ ] **Step 3: Define a common policy port**

Every policy receives the same `SituationSnapshot`, estimator outputs, config, previous committed plan, and deterministic random stream. Every policy returns the same `TrackingPlanCandidate`; all candidates pass the same safety guard and commit path.

- [ ] **Step 4: Implement five policies**

- `b0_2`: fixed two-UUV groups, nearest-distance assignment, and no active FIM waypoint optimization.
- `b0_3`: fixed three-UUV groups, nearest-distance assignment, and no active FIM waypoint optimization; this is the primary economic baseline.
- `b1`: IMM-UIF plus FIM waypoint optimization with fixed group membership.
- `b2`: rule-based intent/strategy plus dynamic economic grouping and deterministic waypoint optimization.
- `full`: LLM strategy, all deterministic algorithms, and the human-in-the-loop interfaces.

Implement ablations as explicit Full-policy feature gates, one removed capability per run: B-spline prediction, FIM waypoint optimization, elastic grouping, history compression, LLM intent, or expert feedback. Record the selected ablation in the manifest and report it separately from baseline comparisons.

Apply identical target initiation, loss, energy, safety, and communication assumptions to all policies.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/evaluation/test_baselines.py -v`

Expected: PASS for normal, scarce-resource, low-energy, target-addition, and target-loss fixtures.

```bash
git add src/underwater_tracking/evaluation tests/evaluation/test_baselines.py
git commit -m "feat: add comparable tracking baselines"
```

### Task 3: Build the metric registry

**Files:**
- Create: `src/underwater_tracking/evaluation/metrics/base.py`
- Create: `src/underwater_tracking/evaluation/metrics/tracking.py`
- Create: `src/underwater_tracking/evaluation/metrics/observability.py`
- Create: `src/underwater_tracking/evaluation/metrics/resources.py`
- Create: `src/underwater_tracking/evaluation/metrics/agent.py`
- Create: `src/underwater_tracking/evaluation/metrics/intent.py`
- Create: `src/underwater_tracking/evaluation/metrics/hitl.py`
- Create: `src/underwater_tracking/evaluation/metrics/registry.py`
- Create: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Write failing analytic metric tests**

Use small hand-computed fixtures to verify position and velocity RMSE, 95th-percentile error, NEES, NIS, track availability, loss rate, recovery time, group-quality dwell, FIM minimum eigenvalue and condition number, active UUV-hours per target-hour, active-member count, reserve ratio, energy consumed, distance traveled, assignment and replanning churn, plan latency, invalid-output rate, retry count, fallback rate, LLM calls and tokens, stale-result rejection, intent macro-F1 and calibration error, intent delay, directive parse accuracy, structured-constraint satisfaction, directive latency, evidence coverage, and counterfactual consistency.

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/evaluation/test_metrics.py -v`

Expected: FAIL importing the metric registry.

- [ ] **Step 3: Implement typed streaming metrics**

Each metric declares its input stream, units, aggregation level, required fields, and reduction. Missing required data must produce a typed `metric_unavailable` result with reason; it must not become zero. Tracking metrics are computed per target before macro and duration-weighted aggregation.

- [ ] **Step 4: Freeze exact formulas**

- Track availability: duration with a valid estimate and covariance divided by target-present duration.
- Active UUV-hours: sum of active tracking duration across UUVs divided by 3600.
- Assignment churn: member additions plus removals per simulation hour.
- Evidence coverage: cited valid evidence IDs divided by all claims requiring evidence.
- Economic saving: `(b0_3_uuv_hours - candidate_uuv_hours) / b0_3_uuv_hours`.
- Relative RMSE change: `(candidate_rmse - b0_3_rmse) / b0_3_rmse`.

Document NEES/NIS degrees of freedom and confidence bounds in module docstrings and tests.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/evaluation/test_metrics.py -v`

Expected: PASS with exact values for analytic fixtures and explicit unavailable results.

```bash
git add src/underwater_tracking/evaluation/metrics tests/evaluation/test_metrics.py
git commit -m "feat: add tracking and agent metric registry"
```

### Task 4: Create canonical scenarios and event scripts

**Files:**
- Create: `configs/scenario/small_6uuv_1target.yaml`
- Create: `configs/scenario/default_12uuv_2target.yaml`
- Create: `configs/scenario/pressure_20uuv_6target.yaml`
- Create: `configs/scenario/events/target_appearance.yaml`
- Create: `configs/scenario/events/target_loss.yaml`
- Create: `configs/scenario/events/energy_depletion.yaml`
- Create: `configs/scenario/events/uuv_failure.yaml`
- Create: `configs/scenario/events/quality_collapse.yaml`
- Create: `src/underwater_tracking/evaluation/scenarios.py`
- Create: `tests/evaluation/test_scenarios.py`

- [ ] **Step 1: Write failing scenario-validation tests**

Verify exact counts, unique IDs, bounds, 2D fixed-depth enforcement, event ordering, nominal target-count range 1 through 4, pressure peak of 6 targets, seed reproducibility, and the absence of impossible initial overlaps.

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/evaluation/test_scenarios.py -v`

Expected: FAIL because canonical files are absent.

- [ ] **Step 3: Author the three scale scenarios**

- Small: 6 UUVs, 1 initial target, 45 simulated minutes.
- Default: 12 UUVs, 2 initial targets, 90 simulated minutes.
- Pressure: 20 UUVs, 4 initial targets and 6 targets at peak, 120 simulated minutes.

Give each scenario a bounded operating area, common sensor/noise assumptions, UUV energy and speed limits, target motion phases, and explicit difficulty rationale. Store truth motion separately from operational observations at load time.

- [ ] **Step 4: Author focused event scripts**

Each event scenario isolates one trigger and defines an observation-based expected response window. Include quality threshold dwell, energy threshold, failure time, target appearance confirmation, and target loss confirmation. Do not assert exact group membership unless required by safety.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/evaluation/test_scenarios.py -v`

Expected: PASS.

```bash
git add configs/scenario src/underwater_tracking/evaluation/scenarios.py tests/evaluation/test_scenarios.py
git commit -m "test: add canonical tracking scenarios"
```

### Task 5: Implement deterministic single-run execution

**Files:**
- Create: `src/underwater_tracking/evaluation/runner.py`
- Create: `src/underwater_tracking/evaluation/telemetry.py`
- Create: `src/underwater_tracking/evaluation/cli.py`
- Create: `tests/evaluation/test_runner.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write a failing same-seed replay test**

```python
def test_same_manifest_produces_byte_identical_operational_results(tmp_path, manifest_factory):
    from underwater_tracking.evaluation.runner import run_once

    manifest = manifest_factory(seed=20260814, deterministic_llm=True)
    first = run_once(manifest, tmp_path / "first")
    second = run_once(manifest, tmp_path / "second")
    assert first.operational_digest == second.operational_digest
    assert first.ledger_digest == second.ledger_digest
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/evaluation/test_runner.py -v`

Expected: FAIL importing runner.

- [ ] **Step 3: Implement run lifecycle**

Validate the manifest, create artifacts, seed all random sources, construct the selected policy, run the headless engine, write telemetry, finalize streaming metrics, and atomically mark the run succeeded or failed. Record wall-clock node latencies separately from simulation time.

- [ ] **Step 4: Add a CLI**

Expose:

`python -m underwater_tracking.evaluation.cli run --manifest PATH --output-root PATH`

`python -m underwater_tracking.evaluation.cli inspect --run-dir PATH`

Exit nonzero on invalid config, truth leakage, corrupt artifacts, engine failure, or incomplete metrics.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/evaluation/test_runner.py -v`

Expected: PASS for success, injected failure, resume rejection, and byte-identical deterministic runs.

```bash
git add src/underwater_tracking/evaluation tests/evaluation/test_runner.py pyproject.toml
git commit -m "feat: add deterministic experiment runner"
```

### Task 6: Add paired batch execution and statistical aggregation

**Files:**
- Create: `src/underwater_tracking/evaluation/batch.py`
- Create: `src/underwater_tracking/evaluation/statistics.py`
- Create: `configs/experiments/pilot.yaml`
- Create: `tests/evaluation/test_batch_and_statistics.py`

- [ ] **Step 1: Write failing pairing and confidence-interval tests**

Verify that every scenario-seed pair contains all selected policies, missing pairs fail aggregation, paired differences preserve sign, bootstrap intervals are reproducible, and sample count is reported.

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/evaluation/test_batch_and_statistics.py -v`

Expected: FAIL importing batch support.

- [ ] **Step 3: Implement bounded batch execution**

Expand a matrix of scenarios, policies, and seeds into immutable manifests. Run locally with a configurable worker count, but keep each run isolated. A failed run is recorded and retried only when the failure class is explicitly retryable; never overwrite its first attempt.

- [ ] **Step 4: Implement paired statistics**

Aggregate mean, median, standard deviation, 95% percentile bootstrap interval, and paired difference against `b0_3`. Run a paired permutation test when its assumptions are configured; otherwise run Wilcoxon signed-rank. Report the p-value and matched-pairs rank-biserial effect size. Report both per-scenario and pooled results. Weight pooled tracking quality by target-present duration; report resource metrics as absolute totals and paired percentages.

- [ ] **Step 5: Define the pilot matrix**

Use all three scale scenarios, all five policies, all six Full-policy ablations, and seeds 101, 202, and 303. The pilot exists to test plumbing, estimate variance, and calibrate only noise-, unit-, and sensor-dependent thresholds; it cannot produce a final acceptance verdict.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/evaluation/test_batch_and_statistics.py -v`

Expected: PASS.

```bash
git add src/underwater_tracking/evaluation configs/experiments/pilot.yaml tests/evaluation/test_batch_and_statistics.py
git commit -m "feat: add paired experiment aggregation"
```

### Task 7: Freeze the formal matrix after a pilot

**Files:**
- Create: `src/underwater_tracking/evaluation/freeze.py`
- Create: `configs/experiments/formal.yaml`
- Create: `configs/acceptance.yaml`
- Create: `docs/evaluation/experiment-protocol.md`
- Create: `tests/evaluation/test_freeze.py`

- [ ] **Step 1: Write a failing freeze test**

Test that a formal matrix cannot be frozen without a complete pilot, explicit scenario hashes, fixed policy and ablation configs, fixed model/prompt versions, at least 30 paired seeds, metric schema version, versioned acceptance thresholds, and a source commit.

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/evaluation/test_freeze.py -v`

Expected: FAIL importing freeze support.

- [ ] **Step 3: Implement protocol freezing**

`freeze_formal_protocol` reads pilot summaries, records observed variance, and writes a content-addressed formal matrix. It may increase seed count to achieve a predeclared confidence-interval half-width target, with a minimum of 30 paired seeds and a maximum of 50. Pilot data may calibrate only RMSE, FIM, and group-quality thresholds that directly depend on noise, units, and sensor configuration. Safety, truth isolation, availability, economic improvement, relative-RMSE degradation, reproducibility, provenance, and latency gates are locked before the pilot and cannot be tuned from outcomes.

- [ ] **Step 4: Document the frozen protocol**

Record hypotheses, baselines, ablations, inclusion rules, paired-seed logic, metric formulas, paired significance test, effect size, confidence method, hardware/model identifiers, failure handling, truth isolation, and the exact acceptance gates. Write calibrated RMSE/FIM/quality values plus all locked gates to `configs/acceptance.yaml`. Mark changes after freezing as a new protocol version.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/evaluation/test_freeze.py -v`

Expected: PASS.

```bash
git add src/underwater_tracking/evaluation/freeze.py configs/experiments/formal.yaml configs/acceptance.yaml docs/evaluation/experiment-protocol.md tests/evaluation/test_freeze.py
git commit -m "docs: freeze formal evaluation protocol"
```

### Task 8: Implement machine-verifiable acceptance gates

**Files:**
- Create: `src/underwater_tracking/evaluation/acceptance.py`
- Create: `tests/evaluation/test_acceptance.py`

- [ ] **Step 1: Write failing threshold tests**

Create synthetic summaries immediately below, at, and above every threshold. Assert inclusive boundary behavior and a failed verdict when required data is unavailable.

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/evaluation/test_acceptance.py -v`

Expected: FAIL importing acceptance.

- [ ] **Step 3: Implement primary acceptance gates**

The full system passes only when all are true:

- operational truth-leak count equals 0;
- invalid committed plan count equals 0;
- deterministic replay mismatches equal 0 for deterministic configurations;
- track availability is at least 95% in the nominal default-scenario aggregate; pressure-scenario availability is reported separately;
- paired active UUV-hours improve by at least 15% versus `b0_3`;
- paired position RMSE is no more than 5% worse than `b0_3`;
- tactical adjustment p95 wall latency is at most 1 second;
- strategic LLM adjustment p95 wall latency is at most 30 seconds;
- every critical event receives a guarded response inside its configured deadline;
- every committed strategic decision has valid evidence and ledger lineage.

- [ ] **Step 4: Add diagnostic secondary gates**

Report, without replacing primary gates: NEES/NIS consistency, intent accuracy and calibration, schema retry/fallback rates, plan churn, directive turnaround, question evidence coverage, target-add/loss response, energy-depletion response, and UUV-failure recovery.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/evaluation/test_acceptance.py -v`

Expected: PASS.

```bash
git add src/underwater_tracking/evaluation/acceptance.py tests/evaluation/test_acceptance.py
git commit -m "feat: encode formal acceptance gates"
```

### Task 9: Generate machine-readable and human-readable reports

**Files:**
- Create: `src/underwater_tracking/evaluation/report.py`
- Create: `src/underwater_tracking/evaluation/plots.py`
- Create: `tests/evaluation/test_report.py`
- Modify: `src/underwater_tracking/evaluation/cli.py`

- [ ] **Step 1: Write failing report-content tests**

Verify that reports include protocol identity, pass/fail verdicts, scenario and seed counts, paired tables, confidence intervals, failure inventory, metric units, and direct artifact references. Verify stable ordering and no raw truth samples in the operational appendix.

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/evaluation/test_report.py -v`

Expected: FAIL importing report support.

- [ ] **Step 3: Implement reports and plots**

Write `acceptance.json`, `summary.csv`, and `report.md`. Generate labeled PNG figures for RMSE versus UUV-hours, availability by scenario, error distribution, FIM/quality threshold dwell, plan churn, energy, latency, retry/fallback rates, intent confusion, and paired economic savings. Every plot must state units, aggregation, sample count, and confidence method.

- [ ] **Step 4: Extend the CLI**

Expose:

`python -m underwater_tracking.evaluation.cli batch --experiment PATH --output-root PATH`

`python -m underwater_tracking.evaluation.cli aggregate --experiment-dir PATH`

`python -m underwater_tracking.evaluation.cli accept --experiment-dir PATH`

Return exit code 0 only when artifact integrity is valid; the `accept` command returns a distinct nonzero code for a valid experiment that fails acceptance.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/evaluation/test_report.py -v`

Expected: PASS.

```bash
git add src/underwater_tracking/evaluation tests/evaluation/test_report.py
git commit -m "feat: report paired evaluation results"
```

### Task 10: Run the integrated evaluation exit gate

**Files:**
- Create: `tests/integration/test_evaluation_smoke.py`
- Create: `tests/integration/test_truth_leak_scan.py`
- Create: `tests/integration/test_deterministic_replay.py`
- Create: `docs/evaluation/runbook.md`

- [ ] **Step 1: Add a short integrated smoke experiment**

Run one 10-minute default-scenario seed for `b0_3`, `b2`, and `full` with the deterministic Mock LLM. Assert complete artifacts, all required metrics, valid ledger lineage, zero truth leaks, and a report. This test verifies plumbing, not formal performance.

- [ ] **Step 2: Add byte-level leak and replay scans**

Scan operational JSONL, ledger, API captures, browser traces, and standard UI build artifacts for truth field names, truth coordinates, and evaluation route calls. Replay the same manifest twice and compare normalized operational and ledger digests.

- [ ] **Step 3: Run the complete automated gate**

Run: `python -m pytest tests/evaluation tests/integration/test_evaluation_smoke.py tests/integration/test_truth_leak_scan.py tests/integration/test_deterministic_replay.py -v`

Expected: PASS.

Run: `python -m underwater_tracking.evaluation.cli batch --experiment configs/experiments/pilot.yaml --output-root outputs/experiments/pilot`

Expected: all pilot manifests finish with `status=succeeded`.

Run: `python -m underwater_tracking.evaluation.cli aggregate --experiment-dir outputs/experiments/pilot`

Expected: summary artifacts are generated and pairing validation passes.

- [ ] **Step 4: Document formal execution**

The runbook must list environment creation, dependency lock verification, source cleanliness, protocol hash verification, batch execution, failure triage, aggregation, acceptance, report inspection, and archival. State that the formal matrix is run only after the pilot and freeze tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/integration docs/evaluation/runbook.md
git commit -m "test: add evaluation and reproducibility gate"
```

## Evaluation plan exit criteria

- [ ] All policies use the same estimated-state inputs, safety guard, simulator assumptions, and paired seeds.
- [ ] Truth storage is physically and logically separated from operational telemetry and UI artifacts.
- [ ] Metric formulas, units, aggregation, missing-data behavior, and confidence methods are tested and documented.
- [ ] Pilot results can freeze a versioned formal protocol without changing acceptance thresholds.
- [ ] Acceptance is emitted as a machine-readable verdict covering quality, economy, latency, safety, provenance, and determinism.
- [ ] The integrated smoke experiment, leak scan, and deterministic replay gate pass before formal execution.
