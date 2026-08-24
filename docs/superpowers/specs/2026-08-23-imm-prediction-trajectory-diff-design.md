# IMM Prediction Trajectory Diff Design

**Date:** 2026-08-23
**Status:** Approved design
**Scope:** Deterministic prediction-difference detection, intent-change
verification, audit evidence, and command-center presentation

## 1. Purpose

The system needs an objective way to compare consecutive target trajectory
predictions. A sufficiently large difference is evidence that the target's
motion behavior may have changed, but it is not by itself proof that the
target's semantic mission intent changed.

This design adds an uncertainty-aware trajectory diff and a persistent
confirmation gate. Deterministic mathematics detects a prediction divergence;
the real Intent LLM interprets its semantic meaning. Only an LLM hypothesis
whose label changes and passes the existing confidence, margin, provenance,
and consecutive-analysis gates may become a strategic intent-change event and
drive regional replanning.

The design uses only public estimated beliefs, prediction corridors, IMM model
probabilities, and evidence IDs. It must never read target truth or adversary-
private state.

## 2. Existing Context and Correction

The current tracking chain produces a `TargetBelief` containing a mixed IMM
mean, covariance, and probabilities for `cv`, `left_turn`, and `right_turn`.
The prediction port converts public belief history into a time-sampled
`PredictedTrackRef` with a centerline and uncertainty corridor. The central
graph uses that prediction to generate regions and optimize resources.

Two existing behaviors require correction as part of this work:

1. Consecutive forecasts are not compared, so a meaningful prediction
   divergence has no first-class metric or causal event.
2. The engine currently allows an IMM motion-model label change to surface as
   `target_intent_changed`. The labels `cv`, `left_turn`, and `right_turn`
   describe kinematic modes, not semantic intentions such as transit, patrol,
   evade, approach, or withdraw. Such changes must instead be recorded as
   `imm_motion_mode_changed` and treated only as supporting evidence.

IMM literature supports using innovation significance, model probabilities,
and persistence for maneuver detection rather than treating one position
difference as semantic intent. The implementation therefore keeps the
mathematical detector and semantic classifier as separate stages.

## 3. Considered Approaches

### 3.1 Fixed Distance Only

Compare aligned centerline points in metres and trigger above a fixed value.
This is easy to explain but ignores estimator uncertainty. The same 250 m
difference can be highly significant for a narrow corridor and noise for a
diffuse bearing-only track.

### 3.2 Uncertainty-Normalized Only

Normalize point differences by the combined prediction uncertainty and use a
statistical threshold. This adapts to tracking quality but can flag a small,
statistically significant change that has no regional planning impact.

### 3.3 Hybrid Statistical and Absolute Gate

Require both an uncertainty-normalized threshold and an absolute distance
floor. This is the selected design because it requires the change to be both
statistically surprising and operationally meaningful.

## 4. Time Alignment

Forecasts must be compared at equal absolute simulation times. Comparing the
same array indices would mistake the normal rolling of the prediction window
for target motion change.

For previous prediction `P_old` and current prediction `P_new`:

1. Validate target identity, finite coordinates, finite non-negative corridor
   radii, strictly increasing sample times, and matching point/radius lengths.
2. Compute the intersection of their absolute time ranges.
3. Build a deterministic comparison grid over that intersection using the
   larger of the two sample steps. This prevents a denser forecast from
   receiving more weight solely because it has more samples.
4. Linearly interpolate centerline coordinates and corridor radii onto the
   common grid.
5. Require at least three samples and 300 seconds of overlap.

No comparison is made when the current prediction has no new public evidence.
New evidence means that the current source-observation ID set contains at
least one ID absent from the previous set. The comparator reports a reason
instead of manufacturing a zero score.

## 5. Diff Mathematics

For aligned sample `i`, define the Euclidean displacement:

\[
d_i = \left\|p_i^{new} - p_i^{old}\right\|_2.
\]

The existing forecast contract carries a scalar corridor radius rather than a
full covariance at every future sample. The combined isotropic uncertainty is:

\[
s_i = \sqrt{(r_i^{new})^2 + (r_i^{old})^2 + \sigma_{floor}^2}.
\]

The Mahalanobis-like normalized displacement is:

\[
z_i = \frac{d_i}{s_i}.
\]

This is explicitly called Mahalanobis-like because the current corridor is
isotropic. If per-sample 2-D prediction covariance is later added, the same
contract can use the strict form
`sqrt(delta.T @ inv(Sigma_old + Sigma_new) @ delta)` without changing the
event semantics.

Near-term differences are more actionable than equal differences at the end
of a 30-minute forecast. With current forecast origin `t_now`:

\[
w_i = \exp\left(-\frac{t_i-t_{now}}{600\;s}\right).
\]

The two trigger metrics are:

\[
D_{abs} = \sqrt{\frac{\sum_i w_i d_i^2}{\sum_i w_i}},
\qquad
D_{norm} = \sqrt{\frac{\sum_i w_i z_i^2}{\sum_i w_i}}.
\]

The result also records unweighted maximum displacement, weighted P90
displacement, and the time of maximum displacement for diagnosis. These
diagnostics do not independently trigger intent analysis.

Weighted P90 is the smallest displacement whose cumulative normalized weight
reaches `0.90` after samples are ordered by displacement. The uncertainty
floor is `1m`; it is a numerical guard against a collapsed corridor, not an
additional trigger threshold.

## 6. IMM Probability Evidence

`PredictedTrackRef` gains the public IMM model probabilities used to create the
forecast. The comparator computes Jensen-Shannon distance over the union of
model labels after validating and normalizing both distributions. It records
whether the leading motion-model label changed.

An empty probability map is permitted for a public search-prior envelope. In
that case the Jensen-Shannon metric is unavailable and cannot affect the
geometric comparison; transitioning from that envelope to a sensor-derived
prediction resets the baseline as specified below.

Jensen-Shannon distance is supporting evidence only. Model probabilities can
jitter around a crossover and the model labels are kinematic, so neither a
large probability distance nor a leading-model change can independently assert
semantic intent change.

## 7. Threshold Configuration

The agent configuration gains:

```yaml
trajectory_diff:
  normalized_threshold: 2.45
  absolute_floor_m: 250.0
  uncertainty_floor_m: 1.0
  near_term_decay_s: 600
  confirmation_cycles: 2
  reset_normalized_threshold: 1.75
  reset_absolute_floor_m: 150.0
  minimum_overlap_s: 300
  minimum_samples: 3
```

`2.45` is approximately the 95% radial boundary for a two-dimensional
Gaussian. `250m` matches the configured minimum prediction-grid cell size, so
the absolute gate represents at least one region-scale change. These values
are defaults, not hidden constants: each result and event records the resolved
thresholds and a threshold-schema version.

A forecast comparison exceeds the detector only when both conditions hold:

```text
D_norm >= normalized_threshold
and
D_abs >= absolute_floor_m
```

## 8. Data Contracts

Add a strict `TrajectoryDiffResult` containing:

- target ID and previous/current prediction IDs;
- previous/current prediction simulation times;
- overlap start/end, overlap duration, comparison step, and sample count;
- `D_abs`, `D_norm`, P90 and maximum displacement metrics;
- Jensen-Shannon distance and previous/current leading IMM models;
- resolved thresholds and threshold-schema version;
- previous/current source belief evidence IDs;
- comparison status and explicit reason;
- `exceeded`, consecutive count, and latch state.

Comparison status is one of:

- `comparable`;
- `first_prediction`;
- `no_new_evidence`;
- `insufficient_overlap`;
- `predictor_regime_reset`;
- `target_mismatch`;
- `invalid_prediction`.

The comparator is a pure function in `prediction/diff.py`. It does not mutate
state, publish events, invoke an LLM, inspect truth, or select thresholds.

## 9. Predictor-Regime Baselines

Some large differences reflect a change in evidence or predictor mechanics,
not target behavior. The detector resets its baseline and emits no suspicion
when:

- the target has no previous prediction;
- a public search-prior envelope is replaced by the first sensor-derived IMM
  belief;
- short-history fallback changes to fitted B-spline prediction or vice versa;
- the target is reacquired after the prediction baseline was absent;
- there is no new source observation evidence;
- the overlap requirements are not met.

The reset reason is auditable. The new forecast becomes the baseline for the
next eligible comparison.

## 10. Persistent Confirmation and Hysteresis

The central checkpoint keeps per-target diff-gate state. One comparison above
both thresholds increments the consecutive count. A comparison below either
trigger threshold resets an unconfirmed count.

After two consecutive above-threshold comparisons, the system emits
`target_intent_change_suspected`. This event means only that the estimated
future trajectory diverged materially. It is tactical, public, and visible to
blue planning, operator audit, and memory ingestion.

Once suspected, the gate stays latched while both metrics remain above their
lower reset thresholds. The latch clears when:

```text
D_norm < 1.75
or
D_abs < 150m
```

Separate trigger and reset levels prevent repeated events near the boundary.
After reset, a later independent divergence can start a new confirmation
sequence.

## 11. Real LLM Intent Verification

The prediction-change route invokes the configured real Intent LLM. The diff
result, motion features, public IMM probabilities, belief quality, previous
confirmed intent, and source evidence IDs enter the bounded intent payload.
The deterministic diff is evidence; it never substitutes for the LLM.

The existing semantic confirmation rules remain mandatory:

- the proposed semantic label differs from the last confirmed label;
- confidence is at least `0.70`;
- confidence leads the runner-up by at least `0.15`;
- the same changed label passes two consecutive analyses;
- model ID, prompt version, request/response hashes, and evidence provenance
  are recorded.

While verification is pending, each new qualifying observation cycle supplies
fresh evidence for the next real LLM analysis. A low-confidence result, an
unchanged label, contradictory evidence, or a diff reset ends the pending
verification without strategic replanning.

Only successful semantic confirmation emits strategic
`target_intent_changed`. That event enters the full chain:

```text
intent confirmation
  -> trajectory prediction
  -> regional generation
  -> real regional strategy LLM
  -> semantic verification
  -> resource optimization
  -> plan validation
  -> commit
```

The graph must avoid a prediction-to-intent-to-prediction loop. A dedicated
post-prediction intent-verification route reuses the same Intent LLM contract,
then routes an unconfirmed result to deterministic continuation and a
confirmed result to regional regeneration.

## 12. Event Semantics and Audit Chain

Add or correct these events:

- `imm_motion_mode_changed`: informational evidence that the leading IMM
  kinematic model changed; never a semantic intent assertion.
- `target_intent_change_suspected`: tactical prediction-diff event carrying
  the complete `TrajectoryDiffResult` reference and source evidence.
- `target_intent_changed`: strategic event emitted only by confirmed semantic
  intent analysis with complete LLM provenance.

The ledger causal chain is:

```text
previous prediction
  -> current prediction
  -> trajectory diff
  -> suspicion event
  -> intent LLM analyses
  -> confirmed intent event
  -> regional plan revision
  -> committed executable plan
```

All IDs must be resolvable from durable stores. A report is invalid if it
cannot resolve either prediction, source evidence, LLM call, decision, or
committed plan.

## 13. UI, Memory, and Replay

The live and replay frame contracts expose the same latest diff state. The
prediction panel shows:

- absolute and normalized diff values;
- trigger and reset thresholds;
- stable, accumulating, suspected, verifying, confirmed, reset, or unavailable
  state;
- consecutive count;
- previous/current prediction revisions;
- supporting IMM mode-probability change;
- resulting intent label and plan revision when confirmed.

The event timeline renders suspicion separately from confirmed intent and
prediction/plan revisions. Memory Steam consumes the durable event payload;
it does not recalculate the metric. Desktop and mobile views use the same
contract and must not hide a FAIL, BLOCKED, unavailable-provider, or
insufficient-evidence state.

## 14. Error Handling

Invalid previous data does not poison a valid current prediction: the current
prediction becomes a new baseline and the diff is recorded unavailable.
Invalid current prediction remains a prediction-chain error and may not be
converted to a stable diff.

No overlap, no new evidence, and predictor-regime changes are expected states,
not exceptions. NaN/Inf values, negative corridor radii, target mismatch, and
non-monotonic samples are invalid inputs with explicit reasons.

Provider failure during semantic verification is persisted as DEGRADED or
FAIL according to the existing provider policy. It may not be replaced by a
heuristic label or a fabricated successful intent change.

## 15. Verification

### 15.1 Unit and Property Tests

- Equal physical paths with shifted rolling windows produce a near-zero diff.
- Different sample steps align deterministically and do not change the score
  solely because one forecast is denser.
- A large absolute difference inside a wide uncertainty corridor does not
  trigger.
- A high normalized difference below 250 m does not trigger.
- Both gates above threshold produce `exceeded=true`.
- Translation and rotation of both trajectories together preserve the score.
- Empty, non-finite, mismatched, and insufficient-overlap inputs produce their
  specified status.
- Prior/fallback/B-spline regime changes reset the baseline.
- Jensen-Shannon distance is finite, symmetric, and bounded.

### 15.2 Gate and Graph Tests

- One exceedance does not invoke intent analysis.
- Two consecutive exceedances invoke the real-intent port once per qualifying
  observation cycle while verification is pending.
- An unchanged or low-confidence semantic label does not replan.
- Two qualifying changed-label analyses emit exactly one confirmed event.
- Hysteresis suppresses duplicates until reset.
- Confirmed intent executes the regional chain and records a committed plan
  whose trigger IDs resolve to the diff evidence.
- IMM motion-model changes alone never emit semantic intent change.

Test doubles may verify deterministic contracts and failure paths in unit
tests, but they are not release evidence and cannot replace the real-provider
acceptance run.

### 15.3 Real Acceptance

Run the real `main.py` from bootstrap through `sim_time_s=28800`. At least one
sensor-visible adversary maneuver must produce a resolvable chain:

```text
public observations
  -> IMM belief and forecast revision
  -> thresholded trajectory diff
  -> target_intent_change_suspected
  -> real Intent LLM provenance
  -> target_intent_changed
  -> regional plan and executable-plan revision
  -> observed blue-force response
```

The JSON and Markdown acceptance reports include all diff metrics, thresholds,
evidence IDs, LLM call status, decision latency, and resulting plan version.
The run passes only with real configured providers, zero physical violations,
zero browser errors, zero failed requests, valid desktop/mobile displays, and
complete ledger provenance. Missing provider access or any broken link is
BLOCKED/FAIL, never synthetic PASS.

## 16. References

- T.-J. Ho, "A switched IMM-Extended Viterbi estimator-based algorithm for
  maneuvering target tracking," *Automatica*, vol. 47, no. 1, pp. 92-98,
  2011. DOI: 10.1016/j.automatica.2010.10.005.
- Y. Pan et al., "Bearing-only target tracking for multi-UUV via IMM-UIF with
  initial filter value correction," *Ocean Engineering*, 2026. DOI:
  10.1016/j.oceaneng.2026.125347.

## 17. Non-Goals

- Inferring semantic intent directly from IMM motion-model labels.
- Reading target truth to calibrate or trigger the online detector.
- Replacing the real Intent or regional LLM with threshold heuristics.
- Adding a learned trajectory classifier.
- Replacing the current B-spline prediction algorithm in this change.
