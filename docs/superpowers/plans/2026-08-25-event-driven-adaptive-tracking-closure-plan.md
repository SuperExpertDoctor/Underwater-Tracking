# Event-Driven Adaptive Tracking Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the adversarial UUV tracking run automatically replan only through valid physical commits, preserve durable multi-layer memory, and render the resulting evidence chain in the command center.

**Architecture:** Keep the existing planning-epoch and semantic-revalidation boundary. Repair the handoff state transition at the mission-controller boundary, make local embedding readiness explicit at the provider boundary, and repair UI test data at the operational-frame validation boundary. The final acceptance run verifies events, plans, validation reports, memory records, and browser-visible state together.

**Tech Stack:** Python 3.13, pytest, Pydantic, SQLite, SentenceTransformers, React 18, TypeScript, Vite, Playwright.

## Global Constraints

- Event-triggered plans submit automatically only after existing semantic and physical revalidation succeeds.
- A rejected candidate keeps the active plan and records a degraded reason; it must never complete the blocked handoff.
- Embeddings remain local-only with `embedding_local_files_only: true`; no HTTP, hash, or mock vector fallback is permitted in production.
- The target local model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` on CPU.
- Runtime and UI expose truthful degraded status instead of fabricated memory or operational data.
- Use `apply_patch` for code and test edits; do not alter unrelated files.

---

### Task 1: Degrade blocked handoffs with stale evidence

**Files:**
- Modify: `src/underwater_tracking/simulation/engine.py:5032-5110`
- Modify: `src/underwater_tracking/runtime/mission_controller.py:560-675`
- Test: `tests/integration/test_uuv_only_physical_execution.py:142-246`
- Test: `tests/simulation/test_uuv_only_carrier_group.py:1242`

**Interfaces:**
- Consumes: `SimulationEngine._mission_handoff_evidence(snapshot, reports, sim_time_s) -> dict[str, HandoffEvidence]`.
- Produces: a `HandoffEvidence` with `blocked_reason` whenever the current cycle lacks complete effective successor observations; `MissionController._block_handoff()` transitions the predecessor to `RegionLifecycle.DEGRADED`.

- [ ] **Step 1: Preserve the failing integration assertion as the red test**

Run: `pytest tests/integration/test_uuv_only_physical_execution.py::test_engine_blocks_handoff_without_current_effective_observations -q`

Expected: FAIL because `R1` remains `HANDOFF_PENDING` instead of `DEGRADED`.

- [ ] **Step 2: Add a focused stale-evidence regression case**

Add a test beside the existing current-cycle handoff test that builds a predecessor in `HANDOFF_PENDING`, supplies an incomplete successor report for the current `sim_time_s`, applies controller observations, and asserts:

```python
assert snapshot.regions_by_id["R1"].lifecycle is RegionLifecycle.DEGRADED
assert any(event.event_type == "handoff_blocked" for event in controller.events())
assert not any(event.event_type == "handoff_completed" for event in controller.events())
```

- [ ] **Step 3: Run the focused test to verify red**

Run: `pytest tests/simulation/test_uuv_only_carrier_group.py -k stale -q`

Expected: FAIL because the incomplete current-cycle evidence is not marked blocked.

- [ ] **Step 4: Mark the missing effective-observation condition as blocked**

In `_mission_handoff_evidence`, derive a deterministic `blocked_reason` when a required successor UUV is absent, not deployed, unhealthy, not in passive tracking, or has no accepted observation from `sim_time_s`. Pass the resulting evidence to the existing controller path so `_block_handoff()` owns the lifecycle transition and durable event.

- [ ] **Step 5: Verify the handoff state machine**

Run: `pytest tests/integration/test_uuv_only_physical_execution.py tests/simulation/test_uuv_only_carrier_group.py -q`

Expected: PASS, including the predecessor `DEGRADED` assertion and no premature handoff completion.

- [ ] **Step 6: Commit the isolated repair**

```bash
git add src/underwater_tracking/simulation/engine.py src/underwater_tracking/runtime/mission_controller.py tests/integration/test_uuv_only_physical_execution.py tests/simulation/test_uuv_only_carrier_group.py
git commit -m "fix: degrade handoffs without effective observations"
```

### Task 2: Make local memory readiness and terminal work status observable

**Files:**
- Modify: `src/underwater_tracking/memory/embeddings.py:203-345`
- Modify: `src/underwater_tracking/cli.py:257-280, 1300-1365`
- Modify: `src/underwater_tracking/memory/worker.py:120-230`
- Test: `tests/memory/test_embeddings.py`
- Test: `tests/memory/test_worker.py`
- Test: `tests/api/test_app_lifespan.py`

**Interfaces:**
- Consumes: `SentenceTransformerEmbeddingProvider.embed(text: str) -> EmbeddingResult`.
- Produces: `SentenceTransformerEmbeddingProvider.verify_ready() -> None`, a truthful `MemoryService.degraded_reason`, and terminal `MemoryWorkStatus.COMPLETED` or `MemoryWorkStatus.DEGRADED` records.

- [ ] **Step 1: Write readiness tests**

Add tests that monkeypatch `SentenceTransformer` to return a 384-dimension vector and assert `verify_ready()` loads once and encodes a fixed non-empty probe. Add a failing-loader test asserting `LLMConfigError` includes the configured model name.

- [ ] **Step 2: Run the readiness tests to verify red**

Run: `pytest tests/memory/test_embeddings.py -k verify_ready -q`

Expected: FAIL because `verify_ready` does not exist.

- [ ] **Step 3: Add provider readiness verification and startup wiring**

Implement:

```python
def verify_ready(self) -> None:
    self.embed("memory readiness probe")
```

Call it immediately after each local provider is created in the live runtime setup. On `LLMConfigError`, preserve the existing degraded memory adapter and include the typed reason in health and memory-stream responses.

- [ ] **Step 4: Add terminal retry regression coverage**

Extend `tests/memory/test_worker.py` with a reasoner or embedding provider that always raises `LLMConfigError`. Poll through `max_attempts` and assert:

```python
assert repository.get_work(work_id).status is MemoryWorkStatus.DEGRADED
assert any(event.type is MemoryStreamEventType.WORK_DEGRADED for event in events)
```

- [ ] **Step 5: Ensure exhausted work cannot remain pending or processing**

Adjust the worker retry path only where needed so the final failed attempt atomically records a degraded stream event and finalizes the row as `DEGRADED`; retain bounded retries for earlier attempts.

- [ ] **Step 6: Verify memory boundaries**

Run: `pytest tests/memory/test_embeddings.py tests/memory/test_worker.py tests/api/test_app_lifespan.py -q`

Expected: PASS with both ready and degraded cases reported truthfully.

- [ ] **Step 7: Commit the memory repair**

```bash
git add src/underwater_tracking/memory/embeddings.py src/underwater_tracking/cli.py src/underwater_tracking/memory/worker.py tests/memory/test_embeddings.py tests/memory/test_worker.py tests/api/test_app_lifespan.py
git commit -m "fix: make local memory readiness and failures explicit"
```

### Task 3: Restore valid operational-frame UI regression coverage

**Files:**
- Modify: `src/underwater_tracking/ui/e2e/command-center.spec.ts:1-230`
- Modify: `src/underwater_tracking/ui/src/hooks/useWebSocket.ts:60-180` only if fixture validation exposes a real guard defect
- Test: `src/underwater_tracking/ui/e2e/command-center.spec.ts`
- Test: `src/underwater_tracking/ui/src/components/map/sceneAssets.test.ts`

**Interfaces:**
- Consumes: `isOperationalFrame(value: unknown): value is OperationalFrame` and `loadSceneAssets()`.
- Produces: a valid mocked `OperationalFrame` accepted by the production hook, observable asset requests, and browser coverage for event evidence, prediction, replay, and fallback rendering.

- [ ] **Step 1: Write the fixture validation test**

Export a narrow validator test helper only if necessary, then assert the e2e fixture satisfies `isOperationalFrame` before browser navigation:

```ts
expect(isOperationalFrame(frame)).toBe(true);
```

- [ ] **Step 2: Run the single command-center test to verify red**

Run: `npm --prefix src/underwater_tracking/ui run test:e2e -- --grep "operator can inspect"`

Expected: FAIL because the existing fixture is rejected and no scene assets are requested.

- [ ] **Step 3: Align the fixture with the current authoritative frame schema**

Add every required current-frame field, including the carrier collection, planning state, and any nullable fields required by `isOperationalFrame`. Keep API routes mocked at `/api/operational/snapshot` and `/api/replay` so the production hook receives the fixture exactly as it would receive the live frame.

- [ ] **Step 4: Verify map interaction and fallback tests**

Run: `npm --prefix src/underwater_tracking/ui run test:e2e`

Expected: PASS for normal scene assets, one missing asset fallback, UUV details, replay, and desktop/mobile prediction evidence without overflow.

- [ ] **Step 5: Commit the UI repair**

```bash
git add src/underwater_tracking/ui/e2e/command-center.spec.ts src/underwater_tracking/ui/src/hooks/useWebSocket.ts
git commit -m "test: restore command center operational frame coverage"
```

### Task 4: Prove the automatic adversarial closure in an eight-minute run

**Files:**
- Modify: `tests/integration/test_uuv_only_production_acceptance.py`
- Modify: `tests/verification/test_live_demo_monitor.py`
- Test: `tests/integration/test_uuv_only_production_acceptance.py`
- Test: `tests/verification/test_live_demo_monitor.py`

**Interfaces:**
- Consumes: the live `RunController` output directory containing `agent.db` and `operational_frames.jsonl`.
- Produces: an acceptance assertion over adversary calls, plan revisions, revalidation reports, terminal memory work, and monotonic 480-second operational frames.

- [ ] **Step 1: Add an eight-minute acceptance assertion**

Use a deterministic local provider and run 96 physical steps. Query the resulting repositories and assert:

```python
assert final_frame.sim_time_s == 480
assert adversary_decision_count >= 1
assert len(committed_plan_versions) >= 2
assert all(report.valid for report in revalidation_reports)
assert terminal_memory_work_count >= 1
```

- [ ] **Step 2: Run the acceptance test to verify red**

Run: `pytest tests/integration/test_uuv_only_production_acceptance.py -k eight_minute -q`

Expected: FAIL until the test's explicit closure assertions are implemented.

- [ ] **Step 3: Add only the repository readers required by the acceptance test**

Expose bounded query helpers on the existing repositories when a required value is not already available. Do not inspect private simulation state; use persisted plans, validation reports, memory work, ledger calls, and operational frames.

- [ ] **Step 4: Run focused and broad verification**

Run:

```bash
pytest tests/verification/test_physics_invariants.py tests/integration/test_uuv_only_physical_execution.py tests/integration/test_uuv_only_production_acceptance.py tests/verification/test_live_demo_monitor.py -q
npm --prefix src/underwater_tracking/ui run test:e2e
python main.py --steps 96 --bootstrap-planning --verification-audit --seed 42 --host 127.0.0.1 --port 18200 --ui-port 15200
```

Expected: all automated checks pass; the run reaches 480 seconds, exits with no owned child process, and its SQLite ledger records automatic adversary, strategy, revalidation, memory, and execution evidence.

- [ ] **Step 5: Commit acceptance coverage**

```bash
git add tests/integration/test_uuv_only_production_acceptance.py tests/verification/test_live_demo_monitor.py
git commit -m "test: verify event driven tracking closure"
```
