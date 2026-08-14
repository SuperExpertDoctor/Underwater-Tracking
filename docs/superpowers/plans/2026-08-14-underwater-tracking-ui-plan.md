# Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan:** Underwater Tracking Command UI

**Goal:** Deliver a replayable command interface that reuses the proven FastAPI, WebSocket, frame-builder, Canvas, sidebar, drawer, and playback architecture of `E:\项目\创新院\Maritime-Surveillance` while presenting the underwater tracking assistant in a coherent deep-ocean visual system.

**Architecture:** The backend converts committed runtime state into a versioned operational frame, broadcasts it over WebSocket, and serves replay data from JSONL logs. The React client renders estimated tracks, sensor geometry, group assignments, plans, events, and decision evidence. Operational endpoints never expose target truth; a separately enabled evaluation route and panel own all truth-only data.

**Tech Stack:** FastAPI, Pydantic 2, Uvicorn, React 18, Vite, TypeScript, Canvas 2D, Lucide React, Vitest, Testing Library, Playwright.

---

**Prerequisites:** Complete `2026-08-14-underwater-tracking-foundation-plan.md` and `2026-08-14-underwater-tracking-agent-plan.md` with green suites.

## File map

- `src/underwater_tracking/api/`: FastAPI application, runtime adapter, replay service, and WebSocket hub.
- `src/underwater_tracking/ui/`: migrated and adapted React/Vite application.
- `src/underwater_tracking/domain/ui_models.py`: versioned operational and evaluation frame contracts.
- `tests/api/`: endpoint, WebSocket, replay, and truth-isolation tests.
- `tests/ui/`: component tests.
- `tests/e2e/`: Playwright operator workflows.

### Task 1: Migrate the reference application shell without a runtime dependency

**Files:**
- Create: `src/underwater_tracking/ui/package.json`
- Create: `src/underwater_tracking/ui/vite.config.ts`
- Create: `src/underwater_tracking/ui/tsconfig.json`
- Create: `src/underwater_tracking/ui/index.html`
- Create: `src/underwater_tracking/ui/src/main.tsx`
- Create: `src/underwater_tracking/ui/src/App.tsx`
- Create: `src/underwater_tracking/ui/src/test/setup.ts`
- Create: `tests/ui/test_reference_migration.ps1`

- [ ] **Step 1: Write the failing migration-boundary test**

```powershell
$root = Resolve-Path "$PSScriptRoot\..\.."
$ui = Join-Path $root "src\underwater_tracking\ui"
$required = @("package.json", "vite.config.ts", "src\App.tsx", "src\main.tsx")
foreach ($path in $required) {
    if (-not (Test-Path (Join-Path $ui $path))) { throw "missing UI file: $path" }
}
$forbidden = Select-String -Path (Join-Path $ui "**\*") -Pattern "E:\\项目\\创新院\\Maritime-Surveillance" -ErrorAction SilentlyContinue
if ($forbidden) { throw "runtime reference to source project detected" }
```

- [ ] **Step 2: Run and verify failure**

Run: `pwsh -File tests/ui/test_reference_migration.ps1`

Expected: FAIL with `missing UI file`.

- [ ] **Step 3: Copy only the structural seams from the reference**

Use these source files as implementation references, not runtime imports:

- `frontend/src/App.jsx`
- `frontend/src/components/CanvasMap.jsx`
- `frontend/src/components/RightSidebar.jsx`
- `frontend/src/components/BottomDrawer.jsx`
- `frontend/src/components/PlaybackBar.jsx`
- `frontend/src/hooks/useWebSocket.js`
- `frontend/src/hooks/useReplay.js`
- `backend/server.py`
- `backend/frame_builder.py`
- `backend/frame_logger.py`

Create TypeScript equivalents under the new repository. Preserve the component boundaries and data flow, but rename domain fields to the contracts in this plan.

- [ ] **Step 4: Pin scripts and dependencies**

`package.json` must expose `dev`, `build`, `test`, and `test:e2e`; it must use React 18, Vite, TypeScript, Lucide React, Vitest, Testing Library, and Playwright. Commit the generated lock file.

- [ ] **Step 5: Run the boundary test and production build**

Run: `pwsh -File tests/ui/test_reference_migration.ps1`

Expected: PASS.

Run: `npm --prefix src/underwater_tracking/ui run build`

Expected: exit code 0 and `dist/index.html` exists.

- [ ] **Step 6: Commit**

```bash
git add src/underwater_tracking/ui tests/ui/test_reference_migration.ps1
git commit -m "feat: migrate surveillance UI shell"
```

### Task 2: Define operational frames and enforce truth isolation

**Files:**
- Create: `src/underwater_tracking/domain/ui_models.py`
- Create: `tests/api/test_frame_contracts.py`
- Modify: `src/underwater_tracking/domain/__init__.py`

- [ ] **Step 1: Write the failing truth-isolation test**

```python
from underwater_tracking.domain.ui_models import OperationalFrame


def test_operational_frame_schema_contains_no_truth_fields():
    forbidden = {"truth", "true_position", "target_truth", "ground_truth"}
    schema_text = str(OperationalFrame.model_json_schema()).lower()
    assert all(name not in schema_text for name in forbidden)
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/api/test_frame_contracts.py -v`

Expected: FAIL importing `ui_models`.

- [ ] **Step 3: Implement versioned frame contracts**

Define strict models for:

- `MapBounds`, `Point2D`, and `CovarianceEllipse`;
- `UUVView`, including state, energy, group, current waypoint, and breadcrumb;
- `TargetEstimateView`, including mean, ellipse, intent, prediction corridor, and quality;
- `BearingRayView`, `GroupView`, `EventView`, `PlanView`, `LedgerView`, and `MetricView`;
- `OperationalFrame`, including `schema_version`, `frame_id`, `sim_time_s`, `plan_version`, and the operational collections;
- `EvaluationFrame`, holding target truth and paired-run metadata, with no inheritance from `OperationalFrame`.

Use discriminated state literals and reject unknown fields.

- [ ] **Step 4: Add serialization and rejection tests**

Test a valid round trip, unknown-field rejection, a plan-version mismatch, and the absence of truth fields from serialized operational JSON.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/api/test_frame_contracts.py -v`

Expected: PASS.

```bash
git add src/underwater_tracking/domain tests/api/test_frame_contracts.py
git commit -m "feat: define truth-safe UI frame contracts"
```

### Task 3: Build the runtime frame adapter, logger, and replay service

**Files:**
- Create: `src/underwater_tracking/api/frame_builder.py`
- Create: `src/underwater_tracking/api/frame_logger.py`
- Create: `src/underwater_tracking/api/replay.py`
- Create: `tests/api/test_frame_pipeline.py`

- [ ] **Step 1: Write a failing append-and-replay test**

```python
def test_logged_operational_frames_round_trip_in_order(tmp_path, frame_factory):
    from underwater_tracking.api.frame_logger import FrameLogger
    from underwater_tracking.api.replay import ReplayService

    path = tmp_path / "frames.jsonl"
    logger = FrameLogger(path)
    logger.append(frame_factory(frame_id=2, sim_time_s=20.0))
    logger.append(frame_factory(frame_id=3, sim_time_s=30.0))
    frames = ReplayService(path).range(start_s=0.0, end_s=30.0)
    assert [frame.frame_id for frame in frames] == [2, 3]
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/api/test_frame_pipeline.py -v`

Expected: FAIL importing `frame_logger`.

- [ ] **Step 3: Implement pure frame construction**

`build_operational_frame(snapshot, plan, ledger_tail, events, metrics)` must map only estimator-visible state. It must convert covariance matrices into ellipse axes and rotation, clip map geometry, sort all entity lists by stable ID, and attach the committed plan version.

- [ ] **Step 4: Implement durable JSONL logging and indexed replay**

Write one validated frame per line with an immediate flush. Build an in-memory `(sim_time_s, byte_offset)` index on startup. Reject corrupt lines with a typed error containing the line number; never silently skip them.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest tests/api/test_frame_pipeline.py -v`

Expected: PASS for deterministic ordering, covariance conversion, append, reload, and time-range replay.

```bash
git add src/underwater_tracking/api tests/api/test_frame_pipeline.py
git commit -m "feat: add operational frame and replay pipeline"
```

### Task 4: Expose runtime, replay, directive, and question APIs

**Files:**
- Create: `src/underwater_tracking/api/app.py`
- Create: `src/underwater_tracking/api/hub.py`
- Create: `src/underwater_tracking/api/dependencies.py`
- Create: `tests/api/test_app.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing endpoint tests**

```python
def test_health_and_operational_snapshot(client):
    assert client.get("/api/health").json()["status"] == "ok"
    payload = client.get("/api/operational/snapshot").json()
    assert payload["schema_version"] == "1.0"
    assert "target_truth" not in str(payload).lower()


def test_directive_is_accepted_asynchronously(client):
    response = client.post("/api/directives", json={
        "text": "优先保证T1，减少不必要换组",
        "author": "expert-1",
        "expected_plan_version": 4,
    })
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/api/test_app.py -v`

Expected: FAIL importing `api.app`.

- [ ] **Step 3: Implement application factory and dependency ports**

Create `create_app(runtime, replay, directive_queue, question_service, evaluation_enabled=False)`. Add:

- `GET /api/health`;
- `GET /api/operational/snapshot`;
- `GET /api/replay?start_s=&end_s=`;
- `POST /api/directives`, returning 202 without waiting for graph completion;
- `POST /api/questions`, returning a cited answer or a typed insufficient-evidence response;
- `WS /ws/operational`, sending validated frames and heartbeat messages.

Bind interfaces through FastAPI dependencies so tests use in-memory fakes.

- [ ] **Step 4: Enforce optimistic concurrency**

Return HTTP 409 when a directive's `expected_plan_version` is stale. Include `current_plan_version` and a human-readable retry instruction. Do not mutate the active plan in the HTTP handler.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/api/test_app.py -v`

Expected: PASS, including WebSocket disconnect cleanup and stale directive rejection.

```bash
git add src/underwater_tracking/api tests/api/test_app.py pyproject.toml
git commit -m "feat: expose assistant runtime APIs"
```

### Task 5: Establish the deep-ocean command visual system

**Files:**
- Create: `src/underwater_tracking/ui/src/styles/tokens.css`
- Create: `src/underwater_tracking/ui/src/styles/global.css`
- Create: `src/underwater_tracking/ui/src/components/layout/CommandShell.tsx`
- Create: `src/underwater_tracking/ui/src/components/common/StatusPill.tsx`
- Create: `src/underwater_tracking/ui/src/components/common/MetricReadout.tsx`
- Create: `src/underwater_tracking/ui/src/components/common/SectionHeader.tsx`
- Create: `src/underwater_tracking/ui/src/components/layout/CommandShell.test.tsx`
- Modify: `src/underwater_tracking/ui/src/App.tsx`

- [ ] **Step 1: Write the failing shell test**

```tsx
it("keeps the map primary and exposes system state", () => {
  render(<CommandShell map={<div>map</div>} sidebar={<div>side</div>} drawer={<div>drawer</div>} />);
  expect(screen.getByRole("main")).toHaveAttribute("data-region", "tactical-map");
  expect(screen.getByText(/链路状态/)).toBeVisible();
  expect(screen.getByText(/方案版本/)).toBeVisible();
});
```

- [ ] **Step 2: Run and verify failure**

Run: `npm --prefix src/underwater_tracking/ui test -- CommandShell.test.tsx`

Expected: FAIL importing `CommandShell`.

- [ ] **Step 3: Implement tokens and hierarchy**

Use near-black navy surfaces, restrained cyan for sensor/estimate data, amber for warnings, red only for critical loss, green for healthy commitments, and off-white text. Define spacing, typography, border, elevation, chart, and motion tokens. Avoid gradients behind data, decorative glow, dense glass effects, and color-only status signaling.

The default desktop grid is header, large tactical map, 360px right sidebar, 280px collapsible bottom drawer, and compact replay bar. Add a usable 1280px breakpoint and a read-only narrow layout.

- [ ] **Step 4: Run component tests and accessibility checks**

Run: `npm --prefix src/underwater_tracking/ui test -- CommandShell.test.tsx`

Expected: PASS with no missing accessible names.

- [ ] **Step 5: Commit**

```bash
git add src/underwater_tracking/ui/src
git commit -m "feat: add deep ocean command design system"
```

### Task 6: Render the tactical map and scientific overlays

**Files:**
- Create: `src/underwater_tracking/ui/src/components/map/TacticalMap.tsx`
- Create: `src/underwater_tracking/ui/src/components/map/renderer.ts`
- Create: `src/underwater_tracking/ui/src/components/map/geometry.ts`
- Create: `src/underwater_tracking/ui/src/components/map/layers.ts`
- Create: `src/underwater_tracking/ui/src/components/map/legend.tsx`
- Create: `src/underwater_tracking/ui/src/components/map/geometry.test.ts`
- Create: `src/underwater_tracking/ui/src/types/frames.ts`

- [ ] **Step 1: Write deterministic geometry tests**

Test world-to-screen conversion, zoom-around-cursor, covariance ellipse axes, clipped bearing rays, B-spline corridor polygons, and stable hit-testing at device pixel ratios 1 and 2.

- [ ] **Step 2: Run and verify failure**

Run: `npm --prefix src/underwater_tracking/ui test -- geometry.test.ts`

Expected: FAIL importing map geometry.

- [ ] **Step 3: Implement ordered Canvas layers**

Render in this fixed order: bathymetry-neutral grid, prediction corridor, plan routes, UUV breadcrumbs, bearing rays, covariance ellipses, target estimates, UUVs, waypoints, selection halos, and labels. Provide independent toggles for bearings, uncertainty, prediction, routes, breadcrumbs, and quality shading.

Use shapes plus labels for group identity. Display estimator mean and covariance only; operational code cannot import `EvaluationFrame`.

- [ ] **Step 4: Add interaction behavior**

Support wheel zoom, drag pan, fit-all, entity selection, pinned tooltips, and keyboard focus for the entity list that mirrors canvas selection. Throttle redraws with `requestAnimationFrame`; do not update React state per animation frame.

- [ ] **Step 5: Run tests and commit**

Run: `npm --prefix src/underwater_tracking/ui test -- geometry.test.ts`

Expected: PASS.

```bash
git add src/underwater_tracking/ui/src/components/map src/underwater_tracking/ui/src/types
git commit -m "feat: render underwater tactical map"
```

### Task 7: Implement live transport, replay, and timeline controls

**Files:**
- Create: `src/underwater_tracking/ui/src/hooks/useOperationalStream.ts`
- Create: `src/underwater_tracking/ui/src/hooks/useReplay.ts`
- Create: `src/underwater_tracking/ui/src/state/frameStore.ts`
- Create: `src/underwater_tracking/ui/src/components/playback/PlaybackBar.tsx`
- Create: `src/underwater_tracking/ui/src/hooks/useOperationalStream.test.tsx`
- Modify: `src/underwater_tracking/ui/src/App.tsx`

- [ ] **Step 1: Write connection-state and ordering tests**

Verify initial snapshot loading, live-frame monotonicity, stale-frame rejection, reconnect backoff, live/replay mode separation, play/pause, step, speed selection, and jump-to-live.

- [ ] **Step 2: Run and verify failure**

Run: `npm --prefix src/underwater_tracking/ui test -- useOperationalStream.test.tsx`

Expected: FAIL importing stream hooks.

- [ ] **Step 3: Implement a bounded frame store**

Keep the latest live frame and at most 600 replay frames. Compare `(sim_time_s, frame_id)` before accepting a live update. On reconnect, fetch a fresh snapshot before consuming deltas. Replay mode must never overwrite the saved latest-live frame.

- [ ] **Step 4: Implement playback controls**

Expose 0.5x, 1x, 2x, and 4x speed; previous/next event; range loading; keyboard shortcuts; visible LIVE or REPLAY state; and current simulation time. Disable directive submission while inspecting historical frames.

- [ ] **Step 5: Run tests and commit**

Run: `npm --prefix src/underwater_tracking/ui test -- useOperationalStream.test.tsx`

Expected: PASS.

```bash
git add src/underwater_tracking/ui/src
git commit -m "feat: add live and replay frame controls"
```

### Task 8: Build status, plan, event, ledger, and metric workspaces

**Files:**
- Create: `src/underwater_tracking/ui/src/components/sidebar/AssetPanel.tsx`
- Create: `src/underwater_tracking/ui/src/components/sidebar/TargetPanel.tsx`
- Create: `src/underwater_tracking/ui/src/components/sidebar/GroupPanel.tsx`
- Create: `src/underwater_tracking/ui/src/components/drawer/BottomDrawer.tsx`
- Create: `src/underwater_tracking/ui/src/components/drawer/PlanTab.tsx`
- Create: `src/underwater_tracking/ui/src/components/drawer/EventTab.tsx`
- Create: `src/underwater_tracking/ui/src/components/drawer/LedgerTab.tsx`
- Create: `src/underwater_tracking/ui/src/components/drawer/MetricsTab.tsx`
- Create: `src/underwater_tracking/ui/src/components/drawer/BottomDrawer.test.tsx`

- [ ] **Step 1: Write failing operator-content tests**

Verify that the UI shows active/reserve/lost UUV counts, per-target RMSE proxy and quality, group membership, current versus proposed plan, event severity, decision evidence IDs, and resource metrics. Verify that critical events are sorted before informational events at equal simulation time.

- [ ] **Step 2: Run and verify failure**

Run: `npm --prefix src/underwater_tracking/ui test -- BottomDrawer.test.tsx`

Expected: FAIL importing drawer components.

- [ ] **Step 3: Implement information-dense panels**

Keep the right sidebar for current state and the drawer for temporal or explanatory detail. Every plan shows version, status, reason, affected targets, group changes, and effective time. Every ledger row links evidence IDs to the corresponding event, observation, intent result, or constraint.

- [ ] **Step 4: Add scientific metric presentations**

Use compact sparklines and exact numeric readouts for group quality, FIM minimum eigenvalue, estimated RMSE, energy, active UUV-hours, and plan churn. Include units, thresholds, and sample windows; do not use unlabeled decorative charts.

- [ ] **Step 5: Run tests and commit**

Run: `npm --prefix src/underwater_tracking/ui test -- BottomDrawer.test.tsx`

Expected: PASS.

```bash
git add src/underwater_tracking/ui/src/components
git commit -m "feat: add command status and decision workspaces"
```

### Task 9: Close the human-in-the-loop interaction

**Files:**
- Create: `src/underwater_tracking/ui/src/components/assistant/DirectiveComposer.tsx`
- Create: `src/underwater_tracking/ui/src/components/assistant/QuestionPanel.tsx`
- Create: `src/underwater_tracking/ui/src/services/assistantApi.ts`
- Create: `src/underwater_tracking/ui/src/components/assistant/DirectiveComposer.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/drawer/BottomDrawer.tsx`

- [ ] **Step 1: Write failing interaction tests**

Test directive preview, explicit submit, queued state, 409 conflict recovery, evidence-linked answers, insufficient-evidence display, and continuation of live frame updates while requests are pending.

- [ ] **Step 2: Run and verify failure**

Run: `npm --prefix src/underwater_tracking/ui test -- DirectiveComposer.test.tsx`

Expected: FAIL importing assistant components.

- [ ] **Step 3: Implement non-blocking directive flow**

The composer sends expert text, author, active plan version, and optional selected target IDs. Display parsed intent and affected constraints when available, but do not imply that a queued directive is committed. On 409, show the newer plan and require the expert to review before resubmission.

- [ ] **Step 4: Implement cited expert questioning**

Answers must render conclusion, confidence, evidence links, and counterfactual factors. Selecting an evidence link jumps the timeline and highlights the related entity. Insufficient evidence is a first-class result, not a transport error.

- [ ] **Step 5: Run tests and commit**

Run: `npm --prefix src/underwater_tracking/ui test -- DirectiveComposer.test.tsx`

Expected: PASS, including a test proving the WebSocket hook receives frames during a delayed question response.

```bash
git add src/underwater_tracking/ui/src
git commit -m "feat: close nonblocking expert interaction loop"
```

### Task 10: Add the gated evaluation view and end-to-end acceptance

**Files:**
- Create: `src/underwater_tracking/api/evaluation.py`
- Create: `src/underwater_tracking/ui/src/components/evaluation/EvaluationPanel.tsx`
- Create: `src/underwater_tracking/ui/src/components/evaluation/TruthOverlay.tsx`
- Create: `tests/api/test_truth_isolation.py`
- Create: `tests/e2e/command-center.spec.ts`
- Create: `tests/e2e/truth-isolation.spec.ts`
- Create: `src/underwater_tracking/ui/playwright.config.ts`
- Modify: `src/underwater_tracking/api/app.py`
- Modify: `src/underwater_tracking/ui/src/App.tsx`

- [ ] **Step 1: Write failing isolation and operator-journey tests**

Test that evaluation-disabled servers return 404 for `/api/evaluation/*`, operational frames never contain truth, and the standard UI bundle does not request evaluation routes. Test the journey: connect, inspect T1, toggle uncertainty, enter replay, jump live, submit a directive, receive a new committed plan, and open its ledger evidence.

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/api/test_truth_isolation.py -v`

Expected: FAIL because the evaluation gate is absent.

Run: `npm --prefix src/underwater_tracking/ui run test:e2e -- command-center.spec.ts`

Expected: FAIL because the end-to-end fixture is absent.

- [ ] **Step 3: Implement an explicit evaluation gate**

Mount evaluation routes only when `evaluation_enabled=True`. Serve truth frames from the evaluation store, never from the operational frame builder. Require an `evaluation` build-time flag to import `EvaluationPanel` and `TruthOverlay`. Give the page a persistent `EVALUATION / TRUTH ENABLED` banner and distinct truth styling.

- [ ] **Step 4: Implement end-to-end fixtures**

Start a deterministic headless engine, FastAPI server, and Vite preview server from Playwright configuration. Use fixed seed 20260814 and stable data-test IDs. Capture screenshots at 1440x900 for the live command view, replay view, expert directive, and evaluation-only truth comparison.

- [ ] **Step 5: Run the UI exit gate**

Run: `python -m pytest tests/api -v`

Expected: PASS.

Run: `npm --prefix src/underwater_tracking/ui test`

Expected: PASS.

Run: `npm --prefix src/underwater_tracking/ui run build`

Expected: PASS.

Run: `npm --prefix src/underwater_tracking/ui run test:e2e`

Expected: PASS with no console errors, no failed requests, and all screenshots generated.

- [ ] **Step 6: Commit**

```bash
git add src/underwater_tracking/api src/underwater_tracking/ui tests/api tests/e2e
git commit -m "test: verify command UI and truth isolation"
```

## UI plan exit criteria

- [ ] The application runs entirely from `Underwater-Tracking`; no runtime import or asset path points to the reference project.
- [ ] Operational REST, WebSocket, replay, logs, and browser state contain no target truth.
- [ ] Live tracking continues during directives and expert questions.
- [ ] The map renders estimates, uncertainty, bearings, prediction corridors, groups, routes, and waypoints with stable layer semantics.
- [ ] The command interface is usable at 1440x900 and 1280x720, and the evaluation build is visibly distinct.
- [ ] Python API tests, Vitest, production build, and Playwright acceptance all pass.
