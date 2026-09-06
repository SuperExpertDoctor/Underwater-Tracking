# UUV Tracking Control C-01–C-06 Implementation Plan

> **For Codex:** Execute task by task with test-driven development. Do not modify files
> outside the approved list, do not delete or overwrite existing artifacts, and do not
> merge or force-push.

**Goal:** Make the repository-native UUV-only production path satisfy C-01 through C-06
and publish the completed work only on
`fix/uuv-tracking-control-c01-c06-20260906`.

**Architecture:** Keep the existing coordinator → mission controller → simulation engine
→ frame builder pipeline. Add bounded runtime evidence at its owning layer, make rolling
snapshot application transactional, separate sonar transmission from echo, and project
the resulting state through one canonical `OperationalFrame`.

**Stack:** Python 3.12, Pydantic, NumPy/SciPy, Shapely, pytest, Hypothesis, Ruff, Mypy.

**Design:**
`docs/superpowers/specs/2026-09-06-uuv-tracking-control-remediation-design.md`

## Persistent artifact policy

Use the isolated interpreter and environment under `.python/` and `.venv/`. Disable the
pytest cache provider and use a new, never-reused directory below `.test-artifacts/` for
each test command. Store production audit results below
`outputs/codex-uuv-tracking-control-20260906/`. Never clear or overwrite a failed run;
use the next unused attempt suffix.

## Task 1: Establish the current production baseline

**Read:**

- `configs/scenario/uuv_only_single_target.yaml`
- `src/underwater_tracking/verification/uuv_tracking_coverage_runner.py`
- `src/underwater_tracking/verification/uuv_tracking_coverage_audit.py`

**Steps:**

1. Confirm the branch points at the fetched `origin/master` base plus the design commit.
2. Create the approved isolated Python 3.12 environment and install `.[dev]` without a
   shared cache, PATH change, registry change, or Conda modification.
3. Run the existing relevant tests before changing production code.
4. Run the seed-42, 120-step, two-repeat audit into:
   - `outputs/codex-uuv-tracking-control-20260906/baseline-work`
   - `outputs/codex-uuv-tracking-control-20260906/baseline-evidence`
5. Record command, versions, exit status, digests, key state counts, and any failure in
   the final verification note. Treat the existing documentation as historical evidence,
   not as a substitute for this run.

## Task 2: Make snapshot admission recoverable and resource-safe (C-01)

**Files:**

- Modify: `src/underwater_tracking/domain/execution_models.py`
- Modify: `src/underwater_tracking/planning/execution_snapshot_factory.py`
- Modify: `src/underwater_tracking/planning/task_group_instances.py`
- Modify: `src/underwater_tracking/runtime/execution_coordinator.py`
- Modify: `src/underwater_tracking/runtime/execution_evidence.py`
- Modify: `src/underwater_tracking/runtime/mission_controller.py`
- Test: `tests/domain/test_execution_models.py`
- Test: `tests/runtime/test_execution_snapshot_factory.py`
- Test: `tests/runtime/test_execution_coordinator.py`
- Test: `tests/runtime/test_mission_controller.py`

**Red tests:**

1. A higher revision commits while the previous snapshot is still executable.
2. A candidate rejected as `execution_snapshot_not_executable` leaves all controller,
   owner, resource, confirmation, coverage, and event state unchanged.
3. A later valid candidate commits after that rejection and clears only obsolete terminal
   candidate state.
4. Four simultaneous geometry replacements create eight groups with 24 globally unique
   member IDs and four explicit incoming/outgoing pairs.
5. Unchanged-region progress is preserved and revisions never decrease.

**Implementation:**

1. Validate the complete candidate before mutation.
2. Snapshot and restore all touched controller state on any apply failure.
3. Scope terminal failure/expiry to the rejected revision rather than the coordinator as
   a permanent blocker.
4. Allocate incoming member IDs from a new deployment generation whenever an outgoing
   group still owns the baseline resources.
5. Keep release and event emission idempotent.

**Green command:** run only the four listed test modules with a fresh basetemp.

## Task 3: Drive passive acquisition from public fusion (C-02)

**Files:**

- Modify: `src/underwater_tracking/domain/mission_models.py`
- Modify: `src/underwater_tracking/runtime/mission_controller.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Test: `tests/domain/test_mission_models.py`
- Test: `tests/runtime/test_mission_controller.py`
- Test: `tests/integration/test_uuv_only_production_acceptance.py`

**Red tests:**

1. The group fusion path prefers the valid current-cycle set with the strongest member
   support rather than lexical group order.
2. A public fused belief yields a finite probability for each region.
3. Two consecutive probabilities at or above 0.70 atomically make all three members
   passive and install exactly one owner.
4. Missing, non-finite, below-threshold, and simultaneous-exit inputs reset the counter
   to zero with the correct structured reason.
5. Removing evaluation truth from the harness does not change the operational decision;
   supplying truth without public evidence cannot trigger it.

**Implementation:**

1. Carry source observation IDs from group fusion into the public report and probability
   evidence.
2. Publish current/required confirmation and the most recent reset reason.
3. Commit mode, lifecycle, all three member modes, and initial ownership in one transition.

## Task 4: Make ping and coverage evidence physical (C-03)

**Files:**

- Modify: `src/underwater_tracking/planning/coverage.py`
- Modify: `src/underwater_tracking/planning/task_group_waypoints.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/domain/mission_models.py`
- Test: `tests/planning/test_coverage_paths.py`
- Test: `tests/simulation/test_active_sonar.py`
- Test: `tests/verification/test_uuv_tracking_coverage_audit.py`
- Test: `tests/verification/test_uuv_tracking_coverage_runner.py`

**Red tests:**

1. A deployed active UUV emits a source-backed ping on schedule even when no target echo
   exists; energy/range accounting happens exactly once.
2. Passive groups emit no active ping.
3. Coverage starts at zero, increases only after a physical ping, is monotonic, and
   de-duplicates overlapping footprints.
4. Route completion loops the route and increments scan round without fabricating 100%
   coverage.
5. `scan_completed` becomes true only at the configured coverage threshold.
6. Three-member waypoint generation remains inside the region and respects the 300 m
   separation contract after establishment.

**Implementation:**

1. Split transmission-event emission from echo/contact creation.
2. Maintain bounded per-region/per-generation scan state owned by the engine/controller.
3. Clip and union configured-radius ping footprints against the square region.
4. Derive route progress from physical position and visited waypoints.
5. Replace the verification audit's planned-route coverage gate with source-backed runtime
   evidence while retaining planned geometry checks as a separate static metric.

## Task 5: Enforce safe handoff, replacement, and dedicated restoration (C-04/C-05)

**Files:**

- Modify: `src/underwater_tracking/domain/execution_models.py`
- Modify: `src/underwater_tracking/runtime/mission_controller.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Test: `tests/runtime/test_mission_controller.py`
- Test: `tests/acceptance/test_three_uuv_tracking_modes.py`

**Red tests:**

1. Handoff publishes 0/3, 1/3, 2/3, and 3/3 current-cycle successor states.
2. At 0/3 through 2/3 the old owner remains unchanged with a structured blocking reason.
3. At 3/3 the new owner appears in the same frame in which the old owner becomes
   `EXITING`; no frame has two owners or an owner gap.
4. Missing deployment, health, passive mode, current-cycle observation, or adjacency keeps
   the handoff pending.
5. Old groups remain visible until boundary-exit evidence and release once.
6. Four replacement pairs never exceed eight groups/24 UUVs, ignore stale pending
   revisions, and never transfer resources by teleportation.
7. Dedicated mode has exactly one three-member passive group after exits; the 7,000 m
   threshold creates the latest regional generation and restores regional mode only after
   valid 3/3 takeover.

**Implementation:**

1. Remove any projection that marks an owner exiting during candidate merge.
2. Centralize successor-readiness evaluation and expose its per-member result.
3. Order transfer, exit, boundary disappearance, and release as separate idempotent facts.
4. Apply latest-revision-wins semantics to pending replacement generations.

## Task 6: Publish one authoritative execution frame (C-06)

**Files:**

- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/runtime/execution_evidence.py`
- Test: `tests/api/test_execution_evidence.py`
- Test: `tests/api/test_execution_frame_contract.py`
- Test: `tests/api/test_uuv_only_frame_contract.py`

**Red tests:**

1. One frame contains every C-06 region, group, control, successor, replacement, revision,
   event-time, reason, and evidence-ID field.
2. Frame coverage and ping counts exactly match runtime scan state and are not derived from
   route geometry or visibility.
3. JSONL, hub, REST/WebSocket model serialization, and replay are canonical and equal.
4. Frame construction cannot accept or inspect an evaluation frame/truth target.
5. Fields remain bounded and satisfy the existing operational frame size budget.

**Implementation:** add typed bounded projections with stable defaults for legacy replay;
populate them only from authoritative execution/controller/engine state.

## Task 7: Strengthen repository-native acceptance

**Files:**

- Modify: `tests/acceptance/test_three_uuv_tracking_modes.py`
- Modify: `tests/integration/test_uuv_only_production_acceptance.py`
- Modify: `tests/verification/test_uuv_tracking_coverage_runner.py`
- Modify: `src/underwater_tracking/verification/uuv_tracking_coverage_runner.py`
- Modify: `src/underwater_tracking/verification/uuv_tracking_coverage_audit.py`

**Steps:**

1. Keep focused observation-injection tests for deterministic state-machine edges.
2. Add a production-path acceptance that obtains entry probability from actual public
   fusion and never injects `entry_probability` as the final proof.
3. Make the audit fail if four active regions lack source-backed pings, if confirmation
   never reaches 2/2 when evidence supports it, if owner uniqueness/gap invariants fail,
   or if coverage fields disagree with the runtime frame.
4. Assert revision monotonicity and canonical transport equality throughout the trace.

## Task 8: Full verification, evidence, review, and branch publication

**Create:**

- `docs/verification/2026-09-06-uuv-tracking-control-remediation.md`

**Verification:**

1. Run Ruff with cache disabled.
2. Run Mypy with incremental cache disabled.
3. Run all focused C-01–C-06 tests with fresh basetemps.
4. Run the complete pytest suite with a fresh basetemp.
5. Run two identical seed-42, 120-step production traces into:
   - `outputs/codex-uuv-tracking-control-20260906/final-work`
   - `outputs/codex-uuv-tracking-control-20260906/final-evidence`
6. Confirm equal deterministic digests and record all C-01–C-06 metrics, commands,
   versions, exit codes, output paths, limitations, and unverified items.
7. Review the complete diff against `origin/master`, run `git diff --check`, and confirm
   that no file outside the approved scope changed.
8. Commit the verified implementation and verification note.
9. Push with no force:

   ```powershell
   git -c http.proxy=http://127.0.0.1:7897 `
     -C "D:\Air\反Q\Underwater-Tracking-uuv-tracking-control-c01-c06-20260906" `
     push -u origin fix/uuv-tracking-control-c01-c06-20260906
   ```

Do not merge into `master`, open a merge commit, delete a worktree, delete a branch, or
remove any generated artifact.

