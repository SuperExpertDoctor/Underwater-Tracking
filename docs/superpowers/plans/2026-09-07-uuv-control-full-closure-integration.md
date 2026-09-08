# UUV Control and Full Tracking Closure Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge origin/fix/uuv-tracking-control-c01-c06-20260906 into branch1, reconcile the existing contracts, and make the real seed-42 tracking path satisfy every I-01 through I-06 requirement in docs/three-uuv-tracking-team-remediation-and-acceptance.md.

**Architecture:** Keep SimulationEngine and MissionController authoritative for physical execution and deterministic lifecycle transitions. Keep public fusion/world-model provenance and the LLM recovery loop behind the existing interfaces. Publish all evidence through one bounded OperationalFrame, then consume that frame through the real HTTP, WebSocket, JSONL, Replay, and UI paths.

**Tech Stack:** Python 3.11, Pydantic 2, NumPy/SciPy, Shapely, FastAPI, React 18, TypeScript, Vite, pytest, Ruff, Mypy, and the underwater-tracking Conda environment.

## Global Constraints

- The final target branch is branch1; the UUV source ref is origin/fix/uuv-tracking-control-c01-c06-20260906 at fetched revision 8c5dc2d.
- Work and conflict resolution happen first on feature/uuv-control-full-closure-20260907 in .worktrees/uuv-control-full-closure-20260907.
- User-owned dirty files in the main checkout, including tests/api/test_replay_compatibility.py, docs/llm-tracking-remediation-plan.md, and docs/weeks/, remain untouched.
- Formal operational data must come from the real backend and simulation. No mock frame, mock stream, mock replay, synthetic state-machine result, evaluation truth, or browser-side state inference may be used as production evidence.
- Formal HTTP, WebSocket, JSONL, Replay, world-model, LLM, and UI paths must not contain target truth fields. Truth remains in an explicit evaluation-only channel.
- The acceptance run uses seed 42, reaches at least 28800 simulation seconds, and completes in approximately 480 wall-clock seconds through the repository's accelerated simulation path. The exact command, measured wall time, and acceleration parameters are recorded.
- Every new behavior is developed with a failing focused test, a minimal implementation, a green focused run, and a regression run before the next behavior.
- Existing evidence directories are never deleted or overwritten. New attempts use a new output directory.

---

### Task 1: Merge the UUV Remediation Into an Isolated Integration Branch

**Files:**
- Modify: every file reported by the merge, with special attention to src/underwater_tracking/api/frame_builder.py, src/underwater_tracking/domain/ui_models.py, src/underwater_tracking/runtime/mission_controller.py, src/underwater_tracking/simulation/engine.py, and their tests
- Add: docs/superpowers/plans/2026-09-07-uuv-control-full-closure-integration.md
- Test: the target branch UUV control tests and current branch1 contract tests

**Interfaces:**
- Consumes: branch1 commit f156630 and UUV remediation commit 8c5dc2d.
- Produces: one compilable integration branch containing the target branch C-01 through C-06 implementation and branch1 frontend/LLM/world-model contracts.

- [ ] **Step 1: Confirm the isolated branch and source revision.**

~~~powershell
git status --short --branch
git rev-parse HEAD
git rev-parse origin/fix/uuv-tracking-control-c01-c06-20260906
~~~

Expected: the isolated branch is clean at the current branch1 commit and the source ref is 8c5dc2d.

- [ ] **Step 2: Merge the complete source branch without creating the merge commit yet.**

~~~powershell
git merge --no-commit --no-ff origin/fix/uuv-tracking-control-c01-c06-20260906
git status --short
git diff --name-only --diff-filter=U
~~~

Expected: all unresolved paths are listed explicitly; no main-checkout file is involved.

- [ ] **Step 3: Resolve overlapping contracts according to ownership.**

For frame_builder.py and ui_models.py, retain branch1 events, situation, public estimate, and batch-id inputs while incorporating the UUV branch entry, scan, handoff, replacement, revision, and evidence fields. For engine.py, retain branch1 public-estimate refresh and operator-event flow while incorporating target-independent physical ping emission and runtime scan state. For mission_controller.py, retain branch1 execution-refresh/reconcile behavior while incorporating the UUV branch transactional admission, public-fusion transition, handoff, replacement, release, and dedicated restoration rules. Combine tests so each assertion covers one authoritative behavior.

- [ ] **Step 4: Run syntax and import checks before committing the merge resolution.**

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
conda run --no-capture-output -n underwater-tracking python -m compileall -q src tests tools
~~~

Expected: exit code 0, with no unresolved merge markers.

- [ ] **Step 5: Commit the resolved merge.**

~~~powershell
git add -A
git commit -m "merge: integrate UUV tracking control remediation"
~~~

### Task 2: Close C-01 Snapshot Admission and Recovery

**Files:**
- Modify: src/underwater_tracking/domain/execution_models.py
- Modify: src/underwater_tracking/planning/execution_snapshot_factory.py
- Modify: src/underwater_tracking/planning/task_group_instances.py
- Modify: src/underwater_tracking/runtime/execution_coordinator.py
- Modify: src/underwater_tracking/runtime/execution_evidence.py
- Modify: src/underwater_tracking/runtime/mission_controller.py
- Test: tests/domain/test_execution_models.py
- Test: tests/runtime/test_execution_snapshot_factory.py
- Test: tests/runtime/test_execution_coordinator.py
- Test: tests/runtime/test_mission_controller.py

**Interfaces:**
- Consumes: candidate scenario ID, base execution revision, candidate execution revision, geometry, resource assignments, validity deadline, and evidence IDs.
- Produces: all-or-nothing snapshot admission, monotonic revisions, recoverable rejection state, unique incoming resources, and idempotent release/event facts.

- [ ] **Step 1: Add or preserve failing regression tests for the transaction boundary.**

~~~python
def test_rejected_candidate_restores_controller_state() -> None:
    before = controller.snapshot()
    assert coordinator.apply_verified_execution_snapshot(invalid_candidate) is False
    assert controller.snapshot() == before


def test_valid_revision_recovers_after_non_executable_candidate() -> None:
    assert coordinator.apply_verified_execution_snapshot(invalid_candidate) is False
    assert coordinator.apply_verified_execution_snapshot(valid_higher_revision) is True
    assert controller.snapshot().execution_revision == valid_higher_revision.execution_revision
~~~

- [ ] **Step 2: Run only the C-01 tests and verify the expected failure.**

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
conda run --no-capture-output -n underwater-tracking pytest tests/domain/test_execution_models.py tests/runtime/test_execution_snapshot_factory.py tests/runtime/test_execution_coordinator.py tests/runtime/test_mission_controller.py -q --disable-warnings --basetemp=.pytest_cache/tmp-c01-red
~~~

- [ ] **Step 3: Implement complete pre-validation and rollback.**

Validate scenario, base revision, freshness, geometry, resources, evidence, and replacement identities before mutating controller state. Capture and restore task groups, UUV modes, resource episodes, owner, confirmation counters, scan state, replacement state, and emitted-event dedupe state if application fails. Scope terminal rejection/expiry to the rejected candidate revision. Allocate incoming replacement members from a new deployment generation while outgoing resources remain visible.

- [ ] **Step 4: Run the C-01 tests green and inspect the state assertions.**

~~~powershell
conda run --no-capture-output -n underwater-tracking pytest tests/domain/test_execution_models.py tests/runtime/test_execution_snapshot_factory.py tests/runtime/test_execution_coordinator.py tests/runtime/test_mission_controller.py -q --disable-warnings --basetemp=.pytest_cache/tmp-c01-green
~~~

- [ ] **Step 5: Commit the C-01 change.**

~~~powershell
git add src/underwater_tracking/domain/execution_models.py src/underwater_tracking/planning/execution_snapshot_factory.py src/underwater_tracking/planning/task_group_instances.py src/underwater_tracking/runtime/execution_coordinator.py src/underwater_tracking/runtime/execution_evidence.py src/underwater_tracking/runtime/mission_controller.py tests/domain/test_execution_models.py tests/runtime/test_execution_snapshot_factory.py tests/runtime/test_execution_coordinator.py tests/runtime/test_mission_controller.py
git commit -m "fix: make execution snapshot admission recoverable"
~~~

### Task 3: Close C-02 Public-Fusion Passive Acquisition

**Files:**
- Modify: src/underwater_tracking/domain/mission_models.py
- Modify: src/underwater_tracking/runtime/mission_controller.py
- Modify: src/underwater_tracking/simulation/engine.py
- Test: tests/domain/test_mission_models.py
- Test: tests/runtime/test_mission_controller.py
- Test: tests/integration/test_uuv_only_production_acceptance.py

**Interfaces:**
- Consumes: current-cycle public observations, fused belief mean/covariance, source observation IDs, estimate revision, estimate time, and validity deadline.
- Produces: finite per-region probabilities, structured reset reasons, 0/2 through 2/2 confirmation evidence, atomic three-member passive mode, and one owner.

- [ ] **Step 1: Add failing tests for public-fusion selection and confirmation.**

~~~python
def test_public_fusion_selects_strongest_current_cycle_support() -> None:
    report = fuse_public_group_reports(current_cycle_reports)
    assert set(report.belief.source_observation_ids) == expected_current_cycle_ids


def test_two_public_confirmations_atomically_install_passive_owner() -> None:
    controller.advance(public_observations=first_confirmation)
    controller.advance(public_observations=second_confirmation)
    group = controller.group(group_id)
    assert group.sensor_mode == GroupSensorMode.PASSIVE
    assert group.member_modes == {
        uuv_id: GroupSensorMode.PASSIVE for uuv_id in group.member_uuv_ids
    }
    assert controller.tracking_owner_group_id == group.group_instance_id
~~~

- [ ] **Step 2: Run the C-02 tests and confirm failure is in the missing behavior.**

~~~powershell
conda run --no-capture-output -n underwater-tracking pytest tests/domain/test_mission_models.py tests/runtime/test_mission_controller.py tests/integration/test_uuv_only_production_acceptance.py -q --disable-warnings --basetemp=.pytest_cache/tmp-c02-red
~~~

- [ ] **Step 3: Wire public fusion into region probabilities and atomic transition.**

Select the strongest valid current-cycle member set, carry source IDs into the public report and probability evidence, reject missing/non-finite/expired/low probabilities, and record the reset reason. Keep the 0.70 threshold and two-cycle requirement in deterministic UUV control. Commit lifecycle, all three member modes, and owner together. Do not read EvaluationFrame, target truth, planning windows, or region priors.

- [ ] **Step 4: Run focused C-02 tests green and verify truth isolation.**

~~~powershell
conda run --no-capture-output -n underwater-tracking pytest tests/domain/test_mission_models.py tests/runtime/test_mission_controller.py tests/integration/test_uuv_only_production_acceptance.py tests/world_model/test_truth_transport_contract.py -q --disable-warnings --basetemp=.pytest_cache/tmp-c02-green
~~~

- [ ] **Step 5: Commit the C-02 change.**

~~~powershell
git add src/underwater_tracking/domain/mission_models.py src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/simulation/engine.py tests/domain/test_mission_models.py tests/runtime/test_mission_controller.py tests/integration/test_uuv_only_production_acceptance.py
git commit -m "fix: drive passive acquisition from public fusion"
~~~

### Task 4: Close C-03 Physical Ping and Coverage Evidence

**Files:**
- Modify: src/underwater_tracking/planning/coverage.py
- Modify: src/underwater_tracking/planning/task_group_waypoints.py
- Modify: src/underwater_tracking/simulation/engine.py
- Modify: src/underwater_tracking/domain/mission_models.py
- Modify: src/underwater_tracking/verification/uuv_tracking_coverage_audit.py
- Modify: src/underwater_tracking/verification/uuv_tracking_coverage_runner.py
- Test: tests/planning/test_coverage_paths.py
- Test: tests/simulation/test_active_sonar.py
- Test: tests/verification/test_uuv_tracking_coverage_audit.py
- Test: tests/verification/test_uuv_tracking_coverage_runner.py

**Interfaces:**
- Consumes: deployed healthy active UUV positions, configured ping period/range, route state, region polygon, scan generation, and physical transmission events.
- Produces: source-backed active_ping events, bounded per-generation scan state, deduplicated coverage, route progress, scan round, and threshold-based scan_completed.

- [ ] **Step 1: Add failing physical-evidence tests.**

~~~python
def test_active_uuv_transmits_without_target_echo() -> None:
    events = engine.step_until_ping_without_target_contact()
    assert [event.event_type for event in events] == ["active_ping"]


def test_route_completion_does_not_fabricate_full_coverage() -> None:
    scan = run_route_once_without_physical_ping()
    assert scan.route_progress == 1.0
    assert scan.active_coverage_ratio == 0.0
    assert scan.scan_completed is False
~~~

- [ ] **Step 2: Run planning, sonar, and audit tests and verify the expected failure.**

~~~powershell
conda run --no-capture-output -n underwater-tracking pytest tests/planning/test_coverage_paths.py tests/simulation/test_active_sonar.py tests/verification/test_uuv_tracking_coverage_audit.py tests/verification/test_uuv_tracking_coverage_runner.py -q --disable-warnings --basetemp=.pytest_cache/tmp-c03-red
~~~

- [ ] **Step 3: Separate transmission from echo and accumulate physical coverage.**

Emit a source-backed transmission on schedule for every deployed active scanner even when no echo is returned. Charge energy and range once. Clip configured-radius ping footprints to the region polygon and union them within the current region/generation. Derive route progress from physical positions and visited waypoints. Reset only the current scan-round counters on a new round; never convert route completion or visibility into coverage.

- [ ] **Step 4: Enforce the waypoint and safety contracts.**

Keep all three generated waypoints inside their region and check the minimum established member separation is at least 300m. Passive groups emit no active pings. The audit uses runtime source-backed evidence for coverage while retaining planned geometry as a separate metric.

- [ ] **Step 5: Run C-03 tests green and commit.**

~~~powershell
conda run --no-capture-output -n underwater-tracking pytest tests/planning/test_coverage_paths.py tests/simulation/test_active_sonar.py tests/verification/test_uuv_tracking_coverage_audit.py tests/verification/test_uuv_tracking_coverage_runner.py -q --disable-warnings --basetemp=.pytest_cache/tmp-c03-green
git add src/underwater_tracking/planning/coverage.py src/underwater_tracking/planning/task_group_waypoints.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/domain/mission_models.py src/underwater_tracking/verification/uuv_tracking_coverage_audit.py src/underwater_tracking/verification/uuv_tracking_coverage_runner.py tests/planning/test_coverage_paths.py tests/simulation/test_active_sonar.py tests/verification/test_uuv_tracking_coverage_audit.py tests/verification/test_uuv_tracking_coverage_runner.py
git commit -m "fix: derive scan coverage from physical pings"
~~~

### Task 5: Close C-04 Handoff and C-05 Replacement/Dedicated Recovery

**Files:**
- Modify: src/underwater_tracking/domain/execution_models.py
- Modify: src/underwater_tracking/runtime/mission_controller.py
- Modify: src/underwater_tracking/simulation/engine.py
- Modify: src/underwater_tracking/api/frame_builder.py
- Test: tests/runtime/test_mission_controller.py
- Test: tests/acceptance/test_three_uuv_tracking_modes.py

**Interfaces:**
- Consumes: adjacent successor groups, deployment/health/passive mode, current-cycle per-member observations, boundary-exit evidence, geometry revisions, mileage, and deployment generations.
- Produces: successor readiness 0/3 through 3/3, one owner with no gap, explicit EXITING and DISAPPEARED lifecycle, latest-revision-wins replacement pairs, and dedicated-mode restoration after valid takeover.

- [ ] **Step 1: Add failing handoff and replacement assertions.**

~~~python
def test_handoff_waits_for_three_current_cycle_successor_observations() -> None:
    for expected in (0, 1, 2):
        frame = controller.advance(successor_observations=observations[:expected])
        assert frame.tracking_control.pending_successor_observed_count == expected
        assert frame.tracking_control.tracking_owner_group_id == old_owner_id


def test_three_of_three_transfers_owner_without_gap() -> None:
    frame = controller.advance(successor_observations=all_three_observations)
    assert frame.tracking_control.tracking_owner_group_id == new_owner_id
    assert frame.group(old_owner_id).lifecycle == TaskGroupLifecycle.EXITING
    assert frame.tracking_control.tracking_owner_gap_frames == 0
~~~

- [ ] **Step 2: Run runtime and acceptance tests and confirm the missing lifecycle behavior.**

~~~powershell
conda run --no-capture-output -n underwater-tracking pytest tests/runtime/test_mission_controller.py tests/acceptance/test_three_uuv_tracking_modes.py -q --disable-warnings --basetemp=.pytest_cache/tmp-c045-red
~~~

- [ ] **Step 3: Centralize successor readiness and atomic handoff.**

Require adjacency, deployment, health, passive mode, and same-cycle evidence for all three members. Publish each member result and a structured blocker. Until 3/3, retain the old owner. At 3/3, set the new owner and old EXITING in the same transition. Keep outgoing groups visible until boundary exit, then mark DISAPPEARED and release resources once.

- [ ] **Step 4: Preserve replacement pairs and dedicated restoration.**

Keep one incoming/outgoing pair per region slot, allocate unique member IDs for visible generations, ignore stale pending geometry revisions, and do not teleport resources. Trigger DEDICATED_RELEASE_PENDING at 7000m or less for any dedicated owner member. Create the latest regional generation, require deployed/passive 3/3 takeover, transfer ownership, restore REGIONAL, then let the dedicated group exit.

- [ ] **Step 5: Run C-04/C-05 tests green and commit.**

~~~powershell
conda run --no-capture-output -n underwater-tracking pytest tests/runtime/test_mission_controller.py tests/acceptance/test_three_uuv_tracking_modes.py tests/api/test_execution_frame_contract.py -q --disable-warnings --basetemp=.pytest_cache/tmp-c045-green
git add src/underwater_tracking/domain/execution_models.py src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/api/frame_builder.py tests/runtime/test_mission_controller.py tests/acceptance/test_three_uuv_tracking_modes.py
git commit -m "fix: close tracking handoff and replacement lifecycle"
~~~

### Task 6: Publish and Consume the Complete Authoritative Frame (C-06 and I-04/I-06)

**Files:**
- Modify: src/underwater_tracking/domain/ui_models.py
- Modify: src/underwater_tracking/api/frame_builder.py
- Modify: src/underwater_tracking/runtime/execution_evidence.py
- Modify: src/underwater_tracking/api/live.py, src/underwater_tracking/api/app.py, and replay code only if transport parity requires it
- Modify: src/underwater_tracking/ui/src/types/frames.ts
- Modify: src/underwater_tracking/ui/src/App.tsx and evidence presentation components only where the merged contract requires it
- Modify: src/underwater_tracking/agent/ and src/underwater_tracking/memory/ only where assistant context or audit records lack current frame provenance
- Test: tests/api/test_execution_evidence.py
- Test: tests/api/test_execution_frame_contract.py
- Test: tests/api/test_uuv_only_frame_contract.py
- Test: API transport tests, UI unit tests, and truth-transport tests

**Interfaces:**
- Consumes: authoritative controller/engine evidence and public estimate freshness.
- Produces: bounded typed frame fields for revisions, event time, scan telemetry, entry evidence, group lifecycle, owner/successor, replacement, blockers, source IDs, freshness, and canonical serialization across all transports.

- [ ] **Step 1: Add failing contract and no-inference tests.**

~~~python
def test_operational_frame_contains_all_c06_evidence() -> None:
    frame = build_operational_frame(real_runtime_state)
    assert frame.execution.region_entry_evidence
    assert frame.execution.handoff_evidence
    assert frame.execution.replacements
    assert frame.execution.execution_revision >= 1
    assert frame.execution.event_time_s is not None


def test_transport_payloads_are_canonical_for_one_frame() -> None:
    assert json.loads(jsonl_payload) == websocket_payload == replay_payload == http_payload
~~~

- [ ] **Step 2: Run API, world-model, UI, and truth-boundary tests and verify the expected failure.**

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
conda run --no-capture-output -n underwater-tracking pytest tests/api tests/world_model/test_truth_transport_contract.py -q --disable-warnings --basetemp=.pytest_cache/tmp-c06-red
Set-Location src/underwater_tracking/ui
npm test -- --run
~~~

- [ ] **Step 3: Add bounded typed projections and freshness semantics.**

Populate the frame only from runtime/controller/engine state. Preserve stable defaults for legacy replay as unknown/unavailable. Make valid_until_s determine current/stale/expired status, keep estimate/prediction/world-model provenance on one revision, and expose source evidence IDs. Reject non-finite values at the model boundary.

- [ ] **Step 4: Make all transports and assistant explanations use the same frame.**

Reuse the existing sanitizer and canonical public payload for snapshot HTTP, WebSocket, JSONL, and Replay. Assistant responses reference current frame ID, execution revision, data age, lifecycle stage, and structured blocker without reading truth or inferring from screenshots. Memory records retain scenario, frame, revision, event/evidence IDs, and time but do not become a control input.

- [ ] **Step 5: Remove remaining mock production paths and browser inference.**

~~~powershell
rg -n "mockFrames|useMock|VITE_MOCK|mockEnabled|mock://|MOCK /|MOCK_RUN|target_truth|truth_position|truth_velocity" src/underwater_tracking/ui src/underwater_tracking/api src/underwater_tracking/agent src/underwater_tracking/world_model
~~~

Expected: no mock runtime source and no truth field in formal paths. UI values are read from useWebSocket or useReplay; missing evidence renders unknown/unavailable; display mode changes only presentation.

- [ ] **Step 6: Run C-06 and transport tests green, then build the UI.**

~~~powershell
Set-Location ../..
conda run --no-capture-output -n underwater-tracking pytest tests/api tests/world_model/test_truth_transport_contract.py tests/agent tests/memory -q --disable-warnings --basetemp=.pytest_cache/tmp-c06-green
Set-Location src/underwater_tracking/ui
npm test -- --run
npm run build
~~~

- [ ] **Step 7: Commit the authoritative frame and frontend integration.**

~~~powershell
git add src/underwater_tracking/domain/ui_models.py src/underwater_tracking/api src/underwater_tracking/ui/src src/underwater_tracking/agent src/underwater_tracking/memory tests/api tests/world_model/test_truth_transport_contract.py tests/agent tests/memory
git commit -m "feat: complete authoritative tracking frame integration"
~~~

### Task 7: Strengthen Real Repository-Native Acceptance and Acceleration

**Files:**
- Modify: src/underwater_tracking/verification/uuv_tracking_coverage_runner.py
- Modify: src/underwater_tracking/verification/uuv_tracking_coverage_audit.py
- Modify: tools/run_default_live_acceptance.py
- Modify: tests/integration/test_uuv_only_production_acceptance.py
- Modify: tests/acceptance/test_default_live_acceptance.py and related acceptance tests
- Modify: tests/verification/test_uuv_tracking_coverage_runner.py
- Add: docs/verification/2026-09-07-uuv-control-full-closure.md

**Interfaces:**
- Consumes: a real SimulationEngine, MissionController, public fusion stream, and canonical frames.
- Produces: two reproducible seed-42 traces with a minimum of 28800 simulation seconds, approximately 480 seconds wall time, C-01 through C-06 metrics, and deterministic digests.

- [ ] **Step 1: Add a failing production-path assertion that rejects synthetic proof.**

~~~python
def test_production_acceptance_requires_public_fusion_and_source_backed_ping() -> None:
    result = run_real_acceptance(seed=42, duration_s=480)
    assert result.sim_time_s >= 28800
    assert result.public_fusion_probability_frames > 0
    assert result.source_backed_active_ping_count > 0
    assert result.synthetic_entry_probability_frames == 0
~~~

- [ ] **Step 2: Run the acceptance tests before implementation and record the missing metric or gate.**

~~~powershell
conda run --no-capture-output -n underwater-tracking pytest tests/integration/test_uuv_only_production_acceptance.py tests/acceptance/test_default_live_acceptance.py tests/verification/test_uuv_tracking_coverage_runner.py -q --disable-warnings --basetemp=.pytest_cache/tmp-live-red
~~~

- [ ] **Step 3: Make the runner measure accelerated simulation time separately from wall time.**

Use the configured physical step and accelerated loop to reach sim_time_s >= 28800 while the command duration is capped near 480 seconds. Record wall_started_at, wall_finished_at, sim_time_s, step count, acceleration parameters, and process exit status. A fixture-only frame, injected entry_probability, evaluation truth, or planned-route coverage cannot satisfy the production gate.

- [ ] **Step 4: Enforce all six issue gates in the runtime audit.**

Fail the audit when revisions decrease, an expired snapshot blocks a later valid revision, four active regions lack source-backed pings, coverage disagrees with runtime scan state, public fusion never reaches valid two-cycle confirmation when evidence supports it, an owner gap or duplicate owner appears, old groups disappear before boundary evidence, transports differ, formal payloads contain truth, or UI/assistant data is marked live after expiry.

- [ ] **Step 5: Run the production-path tests green.**

~~~powershell
conda run --no-capture-output -n underwater-tracking pytest tests/integration/test_uuv_only_production_acceptance.py tests/acceptance/test_default_live_acceptance.py tests/verification/test_uuv_tracking_coverage_runner.py -q --disable-warnings --basetemp=.pytest_cache/tmp-live-green
~~~

- [ ] **Step 6: Commit the acceptance implementation and report template.**

~~~powershell
git add src/underwater_tracking/verification tools/run_default_live_acceptance.py tests/integration/test_uuv_only_production_acceptance.py tests/acceptance/test_default_live_acceptance.py tests/verification/test_uuv_tracking_coverage_runner.py docs/verification/2026-09-07-uuv-control-full-closure.md
git commit -m "test: enforce full real tracking acceptance"
~~~

### Task 8: Verify the Complete Integration and Merge Back to branch1

**Files:**
- Modify: docs/verification/2026-09-07-uuv-control-full-closure.md
- Generated: new files below outputs/codex-uuv-control-full-closure-20260907/; never overwrite previous attempts

**Interfaces:**
- Consumes: the fully tested isolated integration branch.
- Produces: fresh static checks, complete test results, UI build results, two equal real production traces, and a local merge into branch1.

- [ ] **Step 1: Run static checks with caches disabled.**

~~~powershell
conda run --no-capture-output -n underwater-tracking ruff check --no-cache src tests tools
conda run --no-capture-output -n underwater-tracking mypy --no-incremental src
conda run --no-capture-output -n underwater-tracking python -m compileall -q src tests tools
git diff --check HEAD~1 HEAD
~~~

- [ ] **Step 2: Run the complete Python and UI suites.**

~~~powershell
conda run --no-capture-output -n underwater-tracking pytest -q --disable-warnings --basetemp=.pytest_cache/tmp-full
Set-Location src/underwater_tracking/ui
npm test -- --run
npm run build
~~~

- [ ] **Step 3: Run two real accelerated seed-42 traces into new directories.**

~~~powershell
Set-Location ../../..
conda run --no-capture-output -n underwater-tracking python tools/run_default_live_acceptance.py --config configs/scenario/uuv_only_single_target.yaml --seed 42 --duration-s 480 --output-dir outputs/codex-uuv-control-full-closure-20260907/run-1
conda run --no-capture-output -n underwater-tracking python tools/run_default_live_acceptance.py --config configs/scenario/uuv_only_single_target.yaml --seed 42 --duration-s 480 --output-dir outputs/codex-uuv-control-full-closure-20260907/run-2
~~~

Expected: each run reaches at least 28800 simulation seconds, contains the real scan/passive/handoff/release evidence, has no formal truth leakage or transport mismatch, and produces the same deterministic digest. If the command's actual options differ after merge, update the command and report the exact supported options instead of substituting a fixture.

- [ ] **Step 4: Fill the verification report from command output.**

Record Python/package versions, branch and commit SHAs, every command and exit code, test counts, UI build result, wall and simulation durations, C-01 through C-06 metrics, trace digests, output paths, limitations, and any criterion that remains unverified. Do not claim full closure for an unverified criterion.

- [ ] **Step 5: Perform the final diff and source-boundary review.**

~~~powershell
git status --short --branch
git diff --stat branch1...HEAD
git diff --check branch1...HEAD
rg -n "mockFrames|useMock|VITE_MOCK|mockEnabled|target_truth|truth_position|truth_velocity|EvaluationFrame" src/underwater_tracking/ui src/underwater_tracking/api src/underwater_tracking/agent src/underwater_tracking/world_model
~~~

Confirm every changed file is in the approved implementation, test, UI, acceptance, or verification scope; confirm no existing evidence was overwritten.

- [ ] **Step 6: Commit the verification report and merge the verified branch into branch1.**

~~~powershell
git add docs/verification/2026-09-07-uuv-control-full-closure.md
git commit -m "docs: record full tracking closure verification"
git checkout branch1
git merge --no-ff feature/uuv-control-full-closure-20260907 -m "merge: integrate full UUV tracking closure"
~~~

- [ ] **Step 7: Run post-merge smoke checks on branch1.**

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
conda run --no-capture-output -n underwater-tracking pytest tests/api/test_execution_frame_contract.py tests/acceptance/test_three_uuv_tracking_modes.py tests/integration/test_uuv_only_production_acceptance.py -q --disable-warnings --basetemp=.pytest_cache/tmp-post-merge
git status --short --branch
~~~

Expected: the post-merge smoke tests pass, branch1 contains the UUV source revision through the integration merge, and the main checkout's pre-existing user modifications remain present and unstaged/uncommitted.
