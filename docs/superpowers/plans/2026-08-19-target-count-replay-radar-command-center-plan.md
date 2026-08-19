# Target Count, Replay, Radar, and Segmented Command Center Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make the real `main.py` command center default to one target, support target-count-selected new runs and run-specific historical replay, and present compact tactical markers, segmented tracking, and bounded radar sectors clearly.

**Architecture:** Keep one `main.py` backend and one Vite process. A run controller owns the replaceable simulation bundle (`_AgentLoop`, `SimulationEngine`, `OperationalHub`, `ReplayService`, and run directory) behind stable API ports. A run catalog indexes immutable `outputs/serve-*` logs. Operational frames carry estimator-safe target heading; the Canvas map uses the same heading/range predicate for bounded radar sectors and the UI makes the regional timeline the primary task view.

**Tech Stack:** Python 3.10 in conda environment `lang_py310`, FastAPI, Pydantic, SQLite/WAL, LangGraph, React 18, TypeScript, Vite, Vitest, Playwright.

## Global Constraints

- Run every backend command through `conda run --no-capture-output -n lang_py310`.
- Preserve target truth isolation: radar heading must come from estimator-visible belief/prediction or operational adversary summary.
- Target count changes happen only when starting a new run; never mutate target rosters inside an active engine.
- Historical replay must select an explicit `run_id`; never silently use the newest output directory.
- Preserve current `physics_step_s=5`, `observation_step_s=30`, and physical-step playback timing.
- Do not drop or pop existing git stashes.
- Keep unrelated user changes in the worktree and use `apply_patch` for edits.

---

### Task 1: Finish Non-Blocking Carrier Cycles

**Files:**
- Modify: `src/underwater_tracking/cli.py`
- Modify: `src/underwater_tracking/agent/runtime.py`
- Test: `tests/agent/test_runtime_master_slave_adversary.py`

**Interfaces:**
- Produces `_AgentLoop(background_carrier: bool = False)` and `apply_background_cycle() -> None`.
- Produces `CarrierRuntime.get_state()` fallback to the last state cache while `_cycle_running` is true.
- Existing synchronous test construction remains synchronous; `_serve` passes `background_carrier=True`.

- [ ] **Step 1: Add a blocking-provider regression test.**

Add a test helper that blocks one role with `threading.Event`, construct `_AgentLoop(..., background_carrier=True)`, run six 5-second physical steps, wait for the provider to start, run six more steps while the provider remains blocked, and assert:

```python
assert engine._clock.sim_time_s == 60
assert engine._step_index == 12
```

Release the event, poll `loop.apply_background_cycle()` until the worker finishes, then close the loop.

- [ ] **Step 2: Run the new test and confirm the pre-fix failure.**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lang_py310 python -m pytest tests/agent/test_runtime_master_slave_adversary.py -q
```

Expected before the completed fix: the new background-cycle test fails because the worker or runtime lock blocks physical advancement.

- [ ] **Step 3: Complete the implementation boundary.**

Keep provider invocation and graph work in `_run_background_cycle`. Store typed local decisions and graph results in `_BackgroundCarrierCycle`; apply sensor modes, plans, verification commands, reservations, and adversary/slave decisions only from `apply_background_cycle()` immediately before `engine.step()`.

Keep the runtime cache guard exact:

```python
if getattr(self, "_cycle_running", False):
    return dict(getattr(self, "_state_cache", {}))
```

Set `_cycle_running` in `tick()` and `resume()` with `finally` cleanup, and update `_state_cache` after normal `get_state()` reads.

- [ ] **Step 4: Run focused verification.**

Run:

```bash
PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m ruff check src/underwater_tracking/cli.py src/underwater_tracking/agent/runtime.py tests/agent/test_runtime_master_slave_adversary.py
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lang_py310 python -m pytest tests/agent/test_runtime_master_slave_adversary.py tests/api/test_app.py tests/api/test_live_publisher.py -q
```

Expected: all selected tests pass and a blocked provider does not stop physical frames.

- [ ] **Step 5: Commit.**

```bash
git add src/underwater_tracking/cli.py src/underwater_tracking/agent/runtime.py tests/agent/test_runtime_master_slave_adversary.py
git commit -m "fix: keep physics running during carrier llm cycles"
```

### Task 2: Target-Count Run Controller

**Files:**
- Create: `src/underwater_tracking/runtime/run_controller.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `src/underwater_tracking/config/models.py`
- Modify: `configs/scenario/default.yaml`
- Test: `tests/runtime/test_run_controller.py`
- Test: `tests/config/test_loader.py`

**Interfaces:**
- `RunRequest(BaseModel)` fields: `target_count: int = Field(ge=1)`, `seed: int | None = Field(default=None, ge=0)`.
- `RunSummary(BaseModel)` fields: `run_id`, `scenario_id`, `target_count`, `seed`, `sim_time_s`, `frame_count`, `status`, `path`.
- `RunController.start_run(target_count: int, seed: int | None = None) -> RunSummary` validates and atomically installs a candidate bundle.
- `RunController.current() -> RunSummary`, `RunController.runtime`, `RunController.replay`, `RunController.hub`, and `RunController.close() -> None`.

- [ ] **Step 1: Add failing config and controller tests.**

Test that `load_app_config("configs/scenario/default.yaml").scenario.initial_target_count == 1`, that target counts `1..max_target_count` are accepted for the synthetic scenario, and that `0` and `max_target_count + 1` are rejected without replacing the current bundle.

Use a temporary output root and a fake LLM mapping so the controller tests never call the provider.

- [ ] **Step 2: Run the tests to verify failure.**

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lang_py310 python -m pytest tests/runtime/test_run_controller.py tests/config/test_loader.py -q
```

Expected: missing controller/config behavior fails.

- [ ] **Step 3: Implement the bundle and atomic replacement.**

Move the repeated `_serve` construction into `RunController._build_bundle(config, seed)`. Validate target count before creating resources. For synthetic scenarios, clone `config.scenario` with `initial_target_count=target_count` and `max_target_count=target_count`; preserve timing and tracking settings. For explicit platform-core rosters, reject counts larger than the loaded roster instead of silently inventing targets.

Build the candidate directory and resources first. Start its simulation worker only after the candidate is complete. Under the controller lock, stop and close the previous bundle, then install the candidate. On any validation or construction exception before installation, leave the previous bundle untouched.

Set `configs/scenario/default.yaml` to:

```yaml
scenario:
  initial_target_count: 1
  max_target_count: 4
```

If the runtime-selected synthetic scenario supports more targets, keep its capacity in a separate controller validation field rather than contradicting the default startup setting.

- [ ] **Step 4: Run focused tests and verify rollback.**

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lang_py310 python -m pytest tests/runtime/test_run_controller.py tests/config/test_loader.py -q
```

Expected: accepted counts create distinct `serve-*` directories; invalid counts preserve the previous run.

- [ ] **Step 5: Commit.**

```bash
git add src/underwater_tracking/runtime/run_controller.py src/underwater_tracking/cli.py src/underwater_tracking/config/models.py configs/scenario/default.yaml tests/runtime/test_run_controller.py tests/config/test_loader.py
git commit -m "feat: add target-count run controller"
```

### Task 3: Run Catalog and Explicit Historical Replay

**Files:**
- Create: `src/underwater_tracking/runtime/run_catalog.py`
- Modify: `src/underwater_tracking/api/replay.py`
- Modify: `src/underwater_tracking/api/app.py`
- Modify: `src/underwater_tracking/cli.py`
- Test: `tests/runtime/test_run_catalog.py`
- Test: `tests/api/test_app.py`
- Test: `tests/api/test_frame_pipeline.py`

**Interfaces:**
- `RunCatalog(output_root: Path)` with `list_runs() -> tuple[RunSummary, ...]`, `get(run_id: str) -> RunSummary`, and `replay(run_id: str) -> ReplayService`.
- `GET /api/runs` returns `{ "runs": [...] }`.
- `POST /api/runs` accepts `RunRequest` and returns the new `RunSummary` with HTTP 202.
- `GET /api/replay?run_id=<id>&start_s=<number>&end_s=<number>` returns only that run's frames.

- [ ] **Step 1: Write catalog and route tests.**

Create two temporary run directories with distinct frame IDs and manifests. Assert catalog ordering is deterministic, replaying run A never returns run B frames, and an unknown run returns 404. Add a route test for `GET /api/runs` and a 422/404 route test for missing run IDs.

- [ ] **Step 2: Run tests to confirm missing behavior.**

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lang_py310 python -m pytest tests/runtime/test_run_catalog.py tests/api/test_app.py tests/api/test_frame_pipeline.py -q
```

- [ ] **Step 3: Implement catalog and route resolution.**

Scan only `outputs/serve-*`, parse `manifest.json` when present, count validated operational frames, and derive the final simulation time. Never expose API keys or database contents. `RunCatalog.get()` must reject path traversal and IDs not matching a catalog directory.

Keep `ReplayService` as the single JSONL validator/indexer; instantiate it for the selected immutable path and let it refresh while the selected run is live.

Use a controller-backed stable API dependency so `operational_snapshot`, health, WebSocket subscription, and replay resolve the current bundle at request time.

- [ ] **Step 4: Run API tests and check serialized contracts.**

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lang_py310 python -m pytest tests/runtime/test_run_catalog.py tests/api/test_app.py tests/api/test_frame_pipeline.py -q
```

Expected: route payloads contain explicit `run_id`; old `/api/replay` callers without `run_id` use the current run only for backward compatibility.

- [ ] **Step 5: Commit.**

```bash
git add src/underwater_tracking/runtime/run_catalog.py src/underwater_tracking/api/replay.py src/underwater_tracking/api/app.py src/underwater_tracking/cli.py tests/runtime/test_run_catalog.py tests/api/test_app.py tests/api/test_frame_pipeline.py
git commit -m "feat: expose run catalog and isolated replay"
```

### Task 4: Target Control and Historical Replay UI

**Files:**
- Create: `src/underwater_tracking/ui/src/components/RunControl.tsx`
- Create: `src/underwater_tracking/ui/src/components/RunControl.test.tsx`
- Modify: `src/underwater_tracking/ui/src/App.tsx`
- Modify: `src/underwater_tracking/ui/src/hooks/useReplay.ts`
- Modify: `src/underwater_tracking/ui/src/types/frames.ts`
- Create: `src/underwater_tracking/ui/src/services/runApi.ts`
- Modify: `src/underwater_tracking/ui/src/App.css`

**Interfaces:**
- `RunSummary` browser type mirrors backend fields.
- `RunControl` props: `runs: RunSummary[]`, `currentRun: RunSummary | null`, `onStart: (targetCount: number) => Promise<void>`, `onSelectReplay: (runId: string) => void`.
- `useReplay(enabled: boolean, runId: string | null)` clears frames and reloads when `runId` changes; `loadRange(startS, endS)` always includes `run_id`.

- [ ] **Step 1: Add failing component and hook tests.**

Test that selecting target count 1 and clicking “新建直播” calls `POST /api/runs`, that a busy state disables duplicate submissions, and that switching from run A to run B clears A's frame before loading B. Test replay query serialization includes `run_id`, `start_s`, and `end_s`.

- [ ] **Step 2: Run the frontend tests to confirm failure.**

```bash
cd src/underwater_tracking/ui
npm test -- --run src/components/RunControl.test.tsx src/hooks/useReplay.test.tsx
```

- [ ] **Step 3: Implement the run control and replay selector.**

Use a compact control with a numeric/select target-count input, a clear “新建直播” action, current run status, and a historical run select visible in replay mode. Fetch `/api/runs` on mode entry and after a successful new run. Display API validation errors inline without hiding the current frame.

Update `useReplay` fetch construction to:

```ts
const params = new URLSearchParams({ run_id: runId ?? "current", start_s: String(Math.max(0, startS)) });
```

Use the existing `physics_step_s` interval logic and preserve keyboard seek/play controls.

- [ ] **Step 4: Run all frontend tests and build.**

```bash
cd src/underwater_tracking/ui
npm test -- --run
npm run build
```

- [ ] **Step 5: Commit.**

```bash
git add src/underwater_tracking/ui/src/components/RunControl.tsx src/underwater_tracking/ui/src/components/RunControl.test.tsx src/underwater_tracking/ui/src/App.tsx src/underwater_tracking/ui/src/hooks/useReplay.ts src/underwater_tracking/ui/src/hooks/useReplay.test.tsx src/underwater_tracking/ui/src/types/frames.ts src/underwater_tracking/ui/src/services/runApi.ts src/underwater_tracking/ui/src/App.css
git commit -m "feat: add target run controls and historical replay selection"
```

### Task 5: Estimator-Safe Heading and Bounded Radar Sector

**Files:**
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/ui/src/types/frames.ts`
- Modify: `src/underwater_tracking/ui/src/components/CanvasMap.tsx`
- Test: `tests/api/test_frame_pipeline.py`
- Test: `src/underwater_tracking/ui/src/components/CanvasMap.test.ts`

**Interfaces:**
- `TargetEstimateView.heading_rad: float | None` is optional and estimator-safe.
- `radarSectorContains(target: TargetEstimateView, point: Point2D, rangeM: number, halfAngleRad = Math.PI * 70 / 360) -> boolean`.
- `radarSectorPath(center: Point2D, headingRad: number, radiusPx: number, halfAngleRad: number) -> Point2D[]` returns two radial endpoints and arc samples for deterministic rendering.

- [ ] **Step 1: Write failing geometry and frame tests.**

Assert a platform directly ahead of heading 0 is inside a 70-degree sector, a platform at 90 degrees is outside, and a platform beyond the range is outside. Assert the frame builder uses adversary operational heading first, prediction direction second, and `None`/zero fallback without exposing target truth.

- [ ] **Step 2: Run focused tests and confirm failure.**

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lang_py310 python -m pytest tests/api/test_frame_pipeline.py -q
cd src/underwater_tracking/ui && npm test -- --run src/components/CanvasMap.test.ts
```

- [ ] **Step 3: Implement one shared sector predicate.**

Derive heading from `_build_estimate` in the documented priority order. In the map, replace `context.arc(center.x, center.y, radius * scale, 0, Math.PI * 2)` with a path starting at the left radial edge, sampling the bounded arc, closing through the center/right edge, filling with low-opacity red, and stroking only the arc and radial edges.

Use the exact same `radarSectorContains` predicate in `detectedPlatformIds` and exposure badges. Keep the existing `detection_range_m` as the radius.

- [ ] **Step 4: Run backend/frontend focused tests.**

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lang_py310 python -m pytest tests/api/test_frame_pipeline.py -q
cd src/underwater_tracking/ui && npm test -- --run src/components/CanvasMap.test.ts
```

- [ ] **Step 5: Commit.**

```bash
git add src/underwater_tracking/domain/ui_models.py src/underwater_tracking/api/frame_builder.py src/underwater_tracking/ui/src/types/frames.ts src/underwater_tracking/ui/src/components/CanvasMap.tsx src/underwater_tracking/ui/src/components/CanvasMap.test.ts tests/api/test_frame_pipeline.py
git commit -m "feat: render estimator-safe bounded radar sectors"
```

### Task 6: Compact Tactical Map and Segmented-Tracking Priority

**Files:**
- Modify: `src/underwater_tracking/ui/src/components/CanvasMap.tsx`
- Modify: `src/underwater_tracking/ui/src/components/map/RegionOverlay.tsx`
- Modify: `src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.tsx`
- Modify: `src/underwater_tracking/ui/src/components/BottomDrawer.tsx`
- Modify: `src/underwater_tracking/ui/src/types/viewConfig.ts`
- Modify: `src/underwater_tracking/ui/src/App.tsx`
- Modify: `src/underwater_tracking/ui/src/App.css`
- Test: `src/underwater_tracking/ui/src/components/CanvasMap.test.ts`
- Test: `src/underwater_tracking/ui/src/components/AssignmentPanel.test.tsx`
- Test: `src/underwater_tracking/ui/src/components/BottomDrawer.test.tsx`

**Interfaces:**
- Marker size tokens remain stable screen dimensions with reduced defaults and unchanged minimum hit areas.
- `RegionOverlay` accepts a visual emphasis mode that renders active/current and next-handoff entries at high contrast and future entries subdued.
- Bottom drawer selects the segmented timeline tab when `frame.region_timeline` is non-empty.

- [ ] **Step 1: Add failing visual-state tests.**

Assert reduced marker dimensions, active region emphasis, subdued future region opacity, and automatic segmented timeline selection when regional rows exist.

- [ ] **Step 2: Run focused frontend tests to confirm failure.**

```bash
cd src/underwater_tracking/ui
npm test -- --run src/components/CanvasMap.test.ts src/components/assistant/AssignmentPanel.test.tsx src/components/BottomDrawer.test.tsx
```

- [ ] **Step 3: Implement hierarchy and interaction.**

Reduce marker token defaults to approximately 65-75% while leaving `spriteHitAreaContains` tolerance unchanged. In `RegionOverlay`, use current timeline status and successor relation to select dominant entries; apply low-opacity fill/stroke to future entries. In `AssignmentPanel`/`BottomDrawer`, choose timeline as the initial tab only when rows are available, retain graph/list tabs, and preserve region selection callbacks.

Keep the dynamic scale bar in the lower-right map corner and ensure mobile CSS does not place it over the playback bar or drawer.

- [ ] **Step 4: Run all frontend tests and build.**

```bash
cd src/underwater_tracking/ui
npm test -- --run
npm run build
```

- [ ] **Step 5: Commit.**

```bash
git add src/underwater_tracking/ui/src/components/CanvasMap.tsx src/underwater_tracking/ui/src/components/map/RegionOverlay.tsx src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.tsx src/underwater_tracking/ui/src/components/BottomDrawer.tsx src/underwater_tracking/ui/src/types/viewConfig.ts src/underwater_tracking/ui/src/App.tsx src/underwater_tracking/ui/src/App.css src/underwater_tracking/ui/src/components/CanvasMap.test.ts src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.test.tsx src/underwater_tracking/ui/src/components/BottomDrawer.test.tsx
git commit -m "fix: prioritize segmented tracking in tactical map"
```

### Task 7: Real `main.py` Acceptance and Integration

**Files:**
- Modify: `tests/main/test_main.py`
- Create: `tests/integration/test_target_count_replay_acceptance.py`
- Modify: `docs/audit-hyperparameters.md` or the SDD progress ledger if acceptance notes are needed.

**Interfaces:**
- Acceptance uses the real `main.py` process and real Vite page, never fixture-only UI data.
- All backend algorithm runs use `lang_py310`.

- [ ] **Step 1: Add bounded startup/route acceptance tests.**

Start the real service on free API/UI ports, call `/api/runs`, POST a second target-count run, verify two distinct output directories, load the first run through `/api/replay?run_id=...`, and assert frame IDs/times belong only to that run.

- [ ] **Step 2: Run backend and frontend verification.**

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lang_py310 python -m pytest tests -m 'not real_llm' -q
cd src/underwater_tracking/ui && npm test -- --run && npm run build
git diff --check
```

Record exact failures rather than treating known baseline failures as passes.

- [ ] **Step 3: Run the actual interactive system.**

Use free ports:

```bash
conda run --no-capture-output -n lang_py310 python main.py --config configs/scenario/default.yaml --steps 0 --seed 42 --host 127.0.0.1 --port 8030 --ui-port 5230
```

Use Playwright against the printed Web UI URL after `networkidle`. Verify nonblank Canvas pixels, compact marker bounds, bounded radar-sector geometry, segmented timeline visibility, target-count submission, historical run selection, range replay, click selection, drag pan, wheel zoom, and mobile layout. Save desktop/mobile screenshots and inspect `outputs/serve-*` JSONL/manifest files.

- [ ] **Step 4: Review and commit acceptance evidence.**

Update the progress/audit note with run IDs, frame counts, selected target counts, replay isolation result, and any provider outage behavior. Commit only source/tests/docs; do not commit generated `outputs/` logs or frontend `dist/` artifacts.

```bash
git add tests/main/test_main.py tests/integration/test_target_count_replay_acceptance.py docs/audit-hyperparameters.md
git commit -m "test: verify target runs replay and tactical command center"
```

## Self-Review Checklist

- [ ] Every design decision in `docs/superpowers/specs/2026-08-19-target-count-replay-radar-command-center-design.md` maps to at least one task.
- [ ] No task uses `TODO`, `TBD`, or unspecified “add appropriate handling” language.
- [ ] Target count, `run_id`, `physics_step_s`, `heading_rad`, sector predicate, and segmented timeline names are consistent across backend and TypeScript tasks.
- [ ] The first task preserves synchronous unit-test behavior while fixing the real `main.py` background path.
- [ ] Final verification uses `lang_py310` and real browser interaction.
