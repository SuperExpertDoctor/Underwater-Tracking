# Frontend Backend Contract Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Merge \`origin/front\` into \`branch1\` and make the repaired UI consume only authoritative backend frames through live HTTP, WebSocket, JSONL, and Replay paths.

**Architecture:** Keep \`branch1\` as the source of truth for the LLM execution-refresh loop, deterministic UUV state machine, public-payload sanitizer, and evaluation boundary. Import the evidence-backed UI presentation from \`origin/front\`, extend the backend frame builder from live runtime state, and remove every mock frame/stream/replay entry point from the production UI. The same public frame is published to snapshot, WebSocket, JSONL, and Replay.

**Tech Stack:** Python 3.11, Pydantic 2, FastAPI, simulation/runtime state, React 18, TypeScript, Vite, Vitest, Playwright, pytest, Conda environment \`underwater-tracking\`.

## Global Constraints

- Do not add or retain a production UI mock mode, mock frame source, mock replay source, mock stream source, or mock operational-data asset.
- UI state must come from backend-produced \`OperationalFrame\`; missing evidence renders as \`unknown\` or \`unavailable\`.
- Scan, entry, handoff, deployment, replacement, and freshness values must come from runtime state, never from geometry, UUV count, route progress, or browser history.
- HTTP snapshot, WebSocket, JSONL, and Replay must expose the same public field names and semantics for a frame.
- Truth/evaluation data remains outside formal operational frames and ordinary UI transports.
- Run Python commands with \`PYTHONPATH=src\` through \`conda run --no-capture-output -n underwater-tracking\`.
- Preserve the user-owned dirty files on \`branch1\`: \`docs/llm-tracking-remediation-plan.md\` and the extra replay assertion.
- Use the accelerated \`seed=42\` audit with \`--duration-s 480\` for the eight-minute simulation acceptance; do not substitute fixture frames.

---

### Task 1: Merge Frontend Branch Without Mock Data

**Files:**
- Modify: all files affected by \`origin/front\`, resolving conflicts against \`branch1\`
- Modify: \`src/underwater_tracking/ui/src/App.tsx\` and \`src/underwater_tracking/ui/src/types/frames.ts\`
- Delete from final tree: \`src/underwater_tracking/ui/src/mock/mockFrames.ts\`, \`useMockReplay.ts\`, \`useMockStream.ts\`, \`src/underwater_tracking/ui/MOCK_RUN.md\`, and mock-only image assets
- Test: existing UI suite and no-mock source scan

**Interfaces:**
- Consumes: \`origin/front\` commit \`02048705f2f42946d2fdba50e1287ba17c6383b2\` and \`branch1\` commit \`3a99615ed18c4a16ce7d2bf42cb3925e5d31c68c\`.
- Produces: an integration branch whose production UI has only \`useWebSocket\` and \`useReplay\` transport sources.

- [ ] **Step 1: Merge without committing.**

\`\`\`powershell
git merge --no-commit --no-ff origin/front
git status --short
\`\`\`

- [ ] **Step 2: Resolve conflicts.** Preserve all branch1 execution-refresh fields while adding the front evidence types. Remove \`mockEnabled\`, mock hooks, mock banners, and mock-only controls from \`App.tsx\`.

- [ ] **Step 3: Remove mock sources and assets.** No runtime import, build variable, URL switch, or acceptance path may select mock frames.

- [ ] **Step 4: Verify the boundary.**

\`\`\`powershell
rg -n "mockFrames|useMock|VITE_MOCK|mockEnabled|mock://|MOCK /|MOCK_RUN" src/underwater_tracking/ui
\`\`\`

Expected: no output from production or acceptance files.

- [ ] **Step 5: Commit the merge resolution.**

\`\`\`powershell
git add -A
git commit -m "merge: integrate evidence-backed frontend without mocks"
\`\`\`

### Task 2: Define Authoritative Evidence Contracts

**Files:**
- Modify: \`src/underwater_tracking/domain/ui_models.py\`
- Modify: the execution model module used by the public frame
- Test: \`tests/api/test_execution_frame_contract.py\`
- Test: \`tests/domain/test_mission_models.py\`

**Interfaces:**
- Consumes: typed mission/controller state already used by \`frame_builder.py\`.
- Produces: strict models for scan telemetry, region entry evidence, handoff evidence, blockers, estimate freshness, and optional deployed/passive UUV IDs.

- [ ] **Step 1: Write failing assertions** for a real runtime-derived frame:

\`\`\`python
assert region.scan_telemetry.active_coverage_ratio == expected_active_coverage
assert region.scan_telemetry.source_backed_ping_count == expected_ping_count
assert frame.execution.region_entry_evidence[region_id].confirmation_count == expected_count
assert frame.execution.handoff_evidence.required_uuv_ids == expected_successor_ids
assert target.estimate_freshness.track_revision == target_track.track_revision
\`\`\`

- [ ] **Step 2: Run the focused tests and observe the missing-field failure.**

\`\`\`powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
conda run --no-capture-output -n underwater-tracking pytest tests/api/test_execution_frame_contract.py tests/domain/test_mission_models.py -q
\`\`\`

- [ ] **Step 3: Implement strict JSON-safe Pydantic models.** Reject non-finite values, preserve stable IDs, and keep new evidence optional only for legacy frames.

- [ ] **Step 4: Run the focused tests green.**

- [ ] **Step 5: Commit the model contract.**

\`\`\`powershell
git add src/underwater_tracking/domain tests/api/test_execution_frame_contract.py tests/domain/test_mission_models.py
git commit -m "feat: define frontend evidence contracts"
\`\`\`

### Task 3: Publish Live Scan, Entry, Handoff, and Group Evidence

**Files:**
- Modify: \`src/underwater_tracking/runtime/mission_controller.py\`
- Modify: \`src/underwater_tracking/simulation/engine.py\`
- Modify: \`src/underwater_tracking/groups/\`
- Modify: \`src/underwater_tracking/api/frame_builder.py\`
- Modify: \`src/underwater_tracking/runtime/execution_snapshot_factory.py\`
- Test: \`tests/runtime/test_mission_controller.py\`
- Test: \`tests/simulation/test_uuv_only_carrier_group.py\`
- Test: \`tests/api/test_execution_frame_contract.py\`

**Interfaces:**
- Consumes: source-backed observations, current-cycle counters, typed handoff evidence, physical UUV deployment state, and replacement lifecycle state.
- Produces: per-region \`scan_telemetry\`; execution entry evidence and blockers; current-cycle \`handoff_evidence\`; backend-confirmed deployment/passive IDs; replacement \`batch_id\`.

- [ ] **Step 1: Add failing runtime tests** for zero-ping regions, route-complete but scan-incomplete regions, scan-round reset, interrupted entry confirmation, incomplete handoff, exact \`3/3\` handoff, and \`EXITING\` before \`DISAPPEARED\`.

- [ ] **Step 2: Run the runtime tests and verify red.**

\`\`\`powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
conda run --no-capture-output -n underwater-tracking pytest tests/runtime/test_mission_controller.py tests/simulation/test_uuv_only_carrier_group.py tests/api/test_execution_frame_contract.py -q
\`\`\`

- [ ] **Step 3: Wire evidence from authoritative runtime state.** Active coverage and ping counts use this-cycle active observations only. Entry counters reset on a new round or interruption. Handoff validity uses the current observation cycle and required successor IDs. Group lists are intersections with physical group membership.

- [ ] **Step 4: Build one public frame without deriving scan completion from route geometry, route progress, group existence, or legacy \`effect.coverage_ratio\`.

- [ ] **Step 5: Run the runtime tests green and commit.**

\`\`\`powershell
git add src/underwater_tracking/runtime src/underwater_tracking/simulation src/underwater_tracking/groups src/underwater_tracking/api/frame_builder.py tests/runtime tests/simulation tests/api/test_execution_frame_contract.py
git commit -m "feat: publish authoritative tracking evidence"
\`\`\`

### Task 4: Publish Estimate Freshness and Preserve Public Provenance

**Files:**
- Modify: \`src/underwater_tracking/api/frame_builder.py\`
- Modify: the public estimate/target model module
- Modify: \`src/underwater_tracking/world_model/adapter.py\` only where public provenance is assembled
- Test: \`tests/api/test_execution_frame_contract.py\`
- Test: \`tests/world_model/test_truth_transport_contract.py\`

**Interfaces:**
- Consumes: public target belief track revision, estimate time, validity deadline, age, and source observation IDs.
- Produces: \`target_estimates[].estimate_freshness\` with status, time, deadline, age, revision, source IDs, and reason, without truth fields.

- [ ] **Step 1: Add failing tests** for current, stale, expired, unavailable, and legacy-unknown estimates; \`valid_until_s\` must win over a contradictory live label.

- [ ] **Step 2: Implement freshness from estimator state**, retaining flat aliases only for migration compatibility and keeping prediction/world-model provenance on the same revision.

- [ ] **Step 3: Run tests and truth-transport checks.**

\`\`\`powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
conda run --no-capture-output -n underwater-tracking pytest tests/api/test_execution_frame_contract.py tests/world_model/test_truth_transport_contract.py -q
\`\`\`

- [ ] **Step 4: Commit the freshness contract.**

### Task 5: Make All Four Backend Transports Identical

**Files:**
- Modify: \`src/underwater_tracking/api/live.py\`
- Modify: \`src/underwater_tracking/api/app.py\`
- Modify: \`src/underwater_tracking/api/replay.py\` only if required
- Test: \`tests/api/test_live_publisher.py\`
- Test: \`tests/api/test_app.py\`
- Test: \`tests/api/test_replay_compatibility.py\`

**Interfaces:**
- Consumes: one sanitized immutable \`OperationalFrame\`.
- Produces: equivalent snapshot HTTP, WebSocket, JSONL, and Replay payloads, including evidence and LLM refresh fields.

- [ ] **Step 1: Add a failing publisher test** comparing transport payloads by frame ID and execution revision, including legacy replay defaults.

- [ ] **Step 2: Reuse the existing sanitizer and publisher payload; do not create a frontend-specific adapter or second source of truth.

- [ ] **Step 3: Run transport tests green.**

\`\`\`powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
conda run --no-capture-output -n underwater-tracking pytest tests/api/test_live_publisher.py tests/api/test_app.py tests/api/test_replay_compatibility.py -q
\`\`\`

- [ ] **Step 4: Commit the transport integration.**

### Task 6: Integrate the Evidence-Backed UI With Real Hooks

**Files:**
- Modify: \`src/underwater_tracking/ui/src/App.tsx\`
- Modify: \`src/underwater_tracking/ui/src/types/frames.ts\`
- Add/import: front presentation logic, \`TargetStatusBoard.tsx\`, and related styles
- Modify: \`CanvasMap.tsx\`, \`RegionTimelinePanel.tsx\`, \`RegionTimelineRow.tsx\`, \`RightSidebar.tsx\`
- Modify: \`AssignmentPanel.tsx\`, \`PredictionOverlay.tsx\`, \`WorldModelEventOverlay.tsx\`
- Test: corresponding UI unit tests

**Interfaces:**
- Consumes: frames returned by real \`useWebSocket\` and \`useReplay\`.
- Produces: evidence-backed freshness, scan, entry, owner/lifecycle, handoff, replacement, and blocker presentation with no browser inference.

- [ ] **Step 1: Add/update UI tests** using frames generated by the backend contract helper, not \`src/mock\` data. Assert missing values render unknown/unavailable, route completion does not imply scan completion, and expired estimates coexist with advancing simulation time.

- [ ] **Step 2: Implement front presentation logic while preserving branch1 refresh fields.** Every displayed effective value is read from the frame.

- [ ] **Step 3: Run UI tests and build.**

\`\`\`powershell
Set-Location src/underwater_tracking/ui
npm test -- --run
npm run build
\`\`\`

- [ ] **Step 4: Commit the real-data UI integration.**

### Task 7: Replace Mock E2E Paths With Real Backend Runs

**Files:**
- Modify: \`src/underwater_tracking/ui/e2e/command-center.spec.ts\`
- Modify: any changed front E2E spec that fulfills operational responses or WebSocket frames
- Modify: \`src/underwater_tracking/ui/playwright.live.config.ts\` only if server lifecycle requires it
- Test: real-server acceptance coverage

**Interfaces:**
- Consumes: a real FastAPI process and real \`seed=42\` simulation.
- Produces: browser assertions against live snapshot, WebSocket, and Replay payloads with no state response stubs.

- [ ] **Step 1: Remove fake operational response routes** from acceptance specs. Keep only network logging or static asset handling.

- [ ] **Step 2: Add real-server assertions** for the same frame ID and execution revision across snapshot, WebSocket, Replay, and rendered DOM, including evidence and freshness.

- [ ] **Step 3: Run the real E2E configuration.**

\`\`\`powershell
Set-Location src/underwater_tracking/ui
npm run test:e2e:live
\`\`\`

- [ ] **Step 4: Commit the real-data E2E coverage.**

### Task 8: Verify and Integrate Into branch1

**Files:**
- Modify: \`docs/three-uuv-tracking-frontend-cross-team-dependencies.md\` only when implementation changes a documented field or migration note
- Generated audit output: ignored \`outputs/\` only; do not commit it

**Interfaces:**
- Consumes: final integration branch and real backend runner.
- Produces: reproducible proof of mock-free, truthful, transport-consistent frontend/backend flow.

- [ ] **Step 1: Run backend tests, compile checks, UI tests, and build.**

\`\`\`powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
conda run --no-capture-output -n underwater-tracking pytest tests/api tests/runtime tests/simulation tests/world_model tests/verification -q
conda run --no-capture-output -n underwater-tracking python -m compileall -q src tests
Set-Location src/underwater_tracking/ui
npm test -- --run
npm run build
\`\`\`

- [ ] **Step 2: Run the real accelerated \`seed=42\` audit twice with \`--duration-s 480\` and compare trace digests.** Assert zero truth leakage, zero transport mismatch, zero stale-frame acceptance, and nonzero backend-produced evidence.

- [ ] **Step 3: Run focused Ruff on changed Python files and \`git diff --check\`; separate unrelated pre-existing findings.

- [ ] **Step 4: Request code review against the branch1 merge base, resolve critical/important findings, and rerun affected tests.

- [ ] **Step 5: Merge the verified feature branch into \`branch1\`, run post-merge smoke tests, and leave the user-owned dirty files intact.**
