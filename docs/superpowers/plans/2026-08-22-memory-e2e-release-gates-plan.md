# Memory, Frontend E2E, and Release Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the asynchronous memory pipeline, Smart Assistant, evidence trace, Memory Steam, full mission UI, real role-specific LLM calls, and default `main.py` lifecycle work together under reproducible release gates.

**Architecture:** Separate memory source polling from maintenance and expose scenario-wide plus conversation-specific stream events through one cursor. Verify memory creation/versioning/evidence/deletion with real persisted sources, keep plan thinking distinct from memory processing, isolate live Playwright from unit UI tests, and add an automated default-entry acceptance driver that records semantic events, API latency, browser assertions, provider ledger calls, and shutdown.

**Tech Stack:** Python 3.11/3.12, SQLite, SentenceTransformers, LongCat structured LLM, FastAPI, React 18, TypeScript, Vitest, Playwright, pytest, Ruff, mypy.

## Global Constraints

- Memory source discovery starts immediately and uses `source_poll_interval_s`, not the maintenance interval.
- Memory processing remains off the physics and API threads.
- Scenario runtime sources and conversation sources share ordered cursors but preserve user/scenario/conversation provenance.
- A memory summary is retrieval material, never direct factual evidence.
- Long-term filtering may ignore ordinary observations; acceptance uses an explicit durable expert preference to require one version chain.
- `llm_thinking` contains bounded operator rationale for planning only and never private chain-of-thought.
- Memory Steam contains memory worker/service events only.
- Unit tests never call real providers because an API key happens to exist.
- Live Playwright uses one controlled `main.py` process and waits for semantic states.
- Final acceptance requires the phase-one, phase-two, and phase-three reports.

---

### Task 1: Separate immediate memory source polling from maintenance

**Files:**
- Modify: `src/underwater_tracking/config/models.py`
- Modify: `configs/memory.yaml`
- Modify: `src/underwater_tracking/memory/worker.py`
- Modify: `tests/config/test_models.py`
- Modify: `tests/memory/test_worker.py`
- Modify: `tests/memory/test_source_reader.py`

**Interfaces:**

```python
class MemoryConfig(StrictModel):
    poll_interval_s: _LLMTimeout = 2.0
    source_poll_interval_s: _LLMTimeout = 2.0
    maintenance_interval_s: _LLMTimeout = 300.0

class MemoryWorker:
    def poll_sources_once(self, *, now: datetime | None = None) -> bool: ...
    def schedule_maintenance_once(self, *, now: datetime | None = None) -> bool: ...
```

- [ ] **Step 1: Add an immediate-discovery regression test.**

```python
def test_first_poll_discovers_sources_without_waiting_for_maintenance(tmp_path: Path) -> None:
    worker, repositories = worker_with_periodic_event(tmp_path, maintenance_interval_s=300.0)
    assert worker.poll_once(now=utc(0)) is True
    assert repositories.long_term.metrics().queue_backlog == 1
```

- [ ] **Step 2: Add independent-clock tests.**

At `t=0`, poll sources; at `t=1`, do neither; at `t=2`, poll sources again; at `t=299`, do not maintain; at `t=300`, enqueue one maintenance work item. A source read failure must not move the source poll timestamp and must emit one degraded event.

- [ ] **Step 3: Run tests and confirm current 300-second source gate fails.**

```bash
PYTHONPATH=src python -m pytest tests/config/test_models.py tests/memory/test_worker.py tests/memory/test_source_reader.py -q
```

- [ ] **Step 4: Implement independent monotonic schedules.**

Initialize `_last_source_poll` and `_last_maintenance` to `None`, making both operations due at startup. `poll_once()` first claims existing work, then polls sources when due, then schedules maintenance when due. It never holds repository transactions across reasoner or embedding calls.

- [ ] **Step 5: Update configuration.**

```yaml
poll_interval_s: 2.0
source_poll_interval_s: 2.0
maintenance_interval_s: 300.0
```

- [ ] **Step 6: Verify and commit.**

```bash
PYTHONPATH=src python -m pytest tests/config/test_models.py tests/memory/test_worker.py tests/memory/test_source_reader.py -q
ruff check src/underwater_tracking/config/models.py src/underwater_tracking/memory/worker.py tests/memory/test_worker.py
mypy src/underwater_tracking/memory/worker.py
git add configs/memory.yaml src/underwater_tracking/config/models.py src/underwater_tracking/memory/worker.py tests/config/test_models.py tests/memory/test_worker.py tests/memory/test_source_reader.py
git commit -m "fix: poll memory sources independently from maintenance"
```

---

### Task 2: Expose complete, provenance-safe Memory Steam events

**Files:**
- Modify: `src/underwater_tracking/domain/memory_models.py`
- Modify: `src/underwater_tracking/memory/service.py`
- Modify: `src/underwater_tracking/memory/worker.py`
- Modify: `src/underwater_tracking/persistence/memory.py`
- Modify: `src/underwater_tracking/api/app.py`
- Modify: `tests/memory/test_service.py`
- Modify: `tests/memory/test_worker.py`
- Modify: `tests/api/test_memory_routes.py`
- Modify: `tests/integration/test_memory_api_real_sqlite.py`

**Interfaces:**

Keep the existing enum values and add aliases only where UI terminology differs. Required persisted types are:

```text
context_loaded
retrieval_started / retrieval_completed
memory_filtered
memory_extracted
memory_version_created / memory_version_superseded
short_term_compression_started / short_term_compressed
memory_accessed / memory_archived / memory_deleted
evidence_trace_started / evidence_trace_completed
source_read_degraded / work_degraded / work_retry_scheduled
worker_recovered
```

Extend the service API:

```python
def MemoryService.stream(
    self,
    user_id: str,
    conversation_id: str,
    *,
    scenario_id: str | None = None,
    after_cursor: int = 0,
    limit: int = 100,
    include_scenario_events: bool = True,
) -> list[MemoryStreamEvent]: ...
```

- [ ] **Step 1: Add a combined-scope cursor test.**

Persist one scenario event with `conversation_id=None`, one event for conversation A, and one for conversation B. Query A with `include_scenario_events=True`; return scenario+A in cursor order, never B. Query after the first cursor; return only newer matching events.

- [ ] **Step 2: Add typed provenance tests.**

Every source-derived event must carry its typed event/decision/plan/message IDs. A version-created event carries `memory_id`, `memory_family_id`, version and operation. Evidence-completed carries all verified source groups and `plan_version`.

- [ ] **Step 3: Add API assertions for non-empty real SQLite stream.**

```python
payload = client.get(
    "/api/assistant/memory/stream",
    params={
        "user_id": "operator",
        "conversation_id": "ops",
        "scenario_id": "S1",
        "after_cursor": 0,
        "include_scenario_events": True,
    },
).json()
assert [event["type"] for event in payload["events"]] == [
    "context_loaded", "memory_filtered", "memory_version_created"
]
assert payload["next_cursor"] > 0
```

- [ ] **Step 4: Run tests and verify current exact-conversation filter fails.**

```bash
PYTHONPATH=src python -m pytest tests/memory/test_service.py tests/memory/test_worker.py tests/api/test_memory_routes.py tests/integration/test_memory_api_real_sqlite.py -q
```

- [ ] **Step 5: Implement a single SQL query for combined scope.**

Use `conversation_id = ? OR conversation_id IS NULL` only when `include_scenario_events` is true. Always filter user and normalized scenario before cursor/limit. Do not merge two independently limited queries in Python.

- [ ] **Step 6: Emit recovery explicitly.**

Add `worker_recovered` after a degraded worker completes its next work/source poll. Emit once per degraded episode. Do not emit normal idle polls.

- [ ] **Step 7: Verify and commit.**

```bash
PYTHONPATH=src python -m pytest tests/memory/test_service.py tests/memory/test_worker.py tests/api/test_memory_routes.py tests/integration/test_memory_api_real_sqlite.py -q
ruff check src/underwater_tracking/domain/memory_models.py src/underwater_tracking/memory src/underwater_tracking/persistence/memory.py src/underwater_tracking/api/app.py
mypy src/underwater_tracking/domain/memory_models.py src/underwater_tracking/memory src/underwater_tracking/persistence/memory.py
git add src/underwater_tracking/domain/memory_models.py src/underwater_tracking/memory/service.py src/underwater_tracking/memory/worker.py src/underwater_tracking/persistence/memory.py src/underwater_tracking/api/app.py tests/memory/test_service.py tests/memory/test_worker.py tests/api/test_memory_routes.py tests/integration/test_memory_api_real_sqlite.py
git commit -m "feat: stream scenario and conversation memory activity"
```

---

### Task 3: Verify short-term compression, long-term versioning, evidence, and deletion

**Files:**
- Modify: `tests/memory/test_worker.py`
- Modify: `tests/memory/test_reasoner.py`
- Modify: `tests/integration/test_memory_api_real_sqlite.py`
- Create: `tests/integration/test_memory_full_lifecycle.py`

**Interfaces:**
- Consumes: `MemoryService.enqueue_conversation_turn`, worker polling, memory snapshot/versions/delete APIs, and evidence trace.
- Produces: one complete deterministic memory lifecycle over real SQLite with fake embedding and structured reasoner ports.

- [ ] **Step 1: Test explicit durable preference creation.**

Submit: `请记住：目标接触后优先维持被动协同跟踪，只有丢失接触时才启用主动扫描。` The fake reasoner returns semantic memory family `tracking-doctrine-preference`, version 1, importance 0.9, with the exact source message ID.

- [ ] **Step 2: Test short-term compression without evidence loss.**

Append enough messages to exceed the configured threshold. Assert summary version increments, only `recent_message_limit` raw messages remain in short-term context, all original messages remain in SQLite message storage, and the compression stream event lists retained/source IDs.

- [ ] **Step 3: Test conflict update and version chain.**

Record the stream cursor after version 1, then submit a revised explicit preference. Assert version 2 becomes active, version 1 becomes superseded, `versions()` returns both in order, and the incremental stream after that cursor contains exactly one created plus one superseded event. The full stream contains two created events total.

- [ ] **Step 4: Test retrieval/evidence trace.**

Ask why the tracking doctrine changed. Assert retrieval returns version 2, then evidence verification resolves its source message and current plan version. The answer must not cite the memory summary as a source ID.

- [ ] **Step 5: Test deletion.**

Delete version 2 through the API. Assert it becomes non-retrievable, the immutable source message remains, and `memory_deleted` appears with the family/version/source IDs.

- [ ] **Step 6: Run lifecycle tests.**

```bash
PYTHONPATH=src python -m pytest tests/memory/test_worker.py tests/memory/test_reasoner.py tests/integration/test_memory_api_real_sqlite.py tests/integration/test_memory_full_lifecycle.py -q
```

- [ ] **Step 7: Commit.**

```bash
git add tests/memory/test_worker.py tests/memory/test_reasoner.py tests/integration/test_memory_api_real_sqlite.py tests/integration/test_memory_full_lifecycle.py
git commit -m "test: verify complete persisted memory lifecycle"
```

---

### Task 4: Keep planning thinking and Memory Steam semantically separate

**Files:**
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/agent/runtime.py`
- Modify: `src/underwater_tracking/ui/src/components/BottomDrawer.tsx`
- Modify: `src/underwater_tracking/ui/src/components/BottomDrawer.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/MemorySteam.tsx`
- Modify: `src/underwater_tracking/ui/src/components/MemorySteam.test.tsx`
- Modify: `src/underwater_tracking/ui/src/hooks/useMemory.ts`
- Modify: `src/underwater_tracking/ui/src/hooks/useMemory.test.ts`
- Modify: `src/underwater_tracking/ui/src/services/memoryApi.ts`
- Modify: `src/underwater_tracking/ui/src/services/memoryApi.test.ts`

**Interfaces:**

```python
class OperationalThinkingSummary(StrictModel):
    epoch_id: str
    plan_version: int
    trigger: Literal["initialization", "critical_event", "expert_feedback"]
    summary: str
    source_event_ids: tuple[str, ...]
```

`OperationalFrame.llm_thinking` remains a string for wire compatibility, but it is generated only from `OperationalThinkingSummary.summary`; `llm_thinking_trigger` and plan version come from the same record.

- [ ] **Step 1: Add backend source-separation tests.**

Memory events alone never change `llm_thinking`. A committed planning epoch updates thinking once. Repeated frames with the same plan version retain or omit the bounded summary according to the existing frame delta contract but never include filter/compression/version text.

- [ ] **Step 2: Add frontend tab tests.**

The LLM tab renders plan version, trigger and summary from the frame. The Memory tab fetches `/api/assistant/memory/stream`, advances its cursor, preserves at most the configured recent event count, and renders event provenance. Switching tabs does not reset either cursor/state.

- [ ] **Step 3: Run tests and verify failures.**

```bash
PYTHONPATH=src python -m pytest tests/api/test_frame_pipeline.py tests/api/test_live_publisher.py -q
npm --prefix src/underwater_tracking/ui test -- --run src/components/BottomDrawer.test.tsx src/components/MemorySteam.test.tsx src/hooks/useMemory.test.ts src/services/memoryApi.test.ts
```

- [ ] **Step 4: Implement the bounded thinking record and combined memory stream query.**

Use `include_scenario_events=true` from the Memory Steam hook. Keep the UI label `Memory Steam` for compatibility with the confirmed design. Do not display raw prompt, response, hidden reasoning, API keys, or full unbounded source payloads.

- [ ] **Step 5: Verify and commit.**

```bash
PYTHONPATH=src python -m pytest tests/api/test_frame_pipeline.py tests/api/test_live_publisher.py -q
npm --prefix src/underwater_tracking/ui test -- --run
npm --prefix src/underwater_tracking/ui run build
git add src/underwater_tracking/api/frame_builder.py src/underwater_tracking/agent/runtime.py src/underwater_tracking/ui/src/components/BottomDrawer.tsx src/underwater_tracking/ui/src/components/BottomDrawer.test.tsx src/underwater_tracking/ui/src/components/MemorySteam.tsx src/underwater_tracking/ui/src/components/MemorySteam.test.tsx src/underwater_tracking/ui/src/hooks/useMemory.ts src/underwater_tracking/ui/src/hooks/useMemory.test.ts src/underwater_tracking/ui/src/services/memoryApi.ts src/underwater_tracking/ui/src/services/memoryApi.test.ts
git commit -m "fix: separate plan thinking from memory processing stream"
```

---

### Task 5: Verify Smart Assistant preview/apply and evidence trace end to end

**Files:**
- Modify: `tests/api/test_conversation.py`
- Create: `tests/integration/test_smart_assistant_lifecycle.py`
- Modify: `src/underwater_tracking/ui/src/components/assistant/SmartAssistantPanel.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/assistant/MemoryWindow.test.tsx`

**Interfaces:**
- Consumes: `/api/conversation/messages`, `/api/conversation/{id}/apply`, memory snapshot/versions/delete routes, and the planning-epoch version guard.
- Produces: one plan-adjustment path and one evidence-query path with verified UI contracts.

- [ ] **Step 1: Add plan-preview integration coverage.**

Submit expert feedback with `expected_plan_version=N`; assert response status is preview and active plan remains N. Apply the exact `turn_id`; assert a new expert planning epoch is queued and only its committed result advances to N+1.

- [ ] **Step 2: Add version conflict coverage.**

Advance the plan between preview and apply. Assert HTTP 409 contains `current_plan_version`, no directive applies, and UI asks for a refreshed preview.

- [ ] **Step 3: Add evidence-query coverage.**

Submit a question against the version-2 memory from Task 3. Assert the returned chain contains memory version, source message/event/decision/knowledge IDs that exist in repositories, and plan version. An unverifiable source produces `insufficient_evidence`, not a fluent unsupported answer.

- [ ] **Step 4: Add UI tests for four memory types, version expansion, and deletion.**

Verify episodic, semantic, procedural, and short-term tabs; expand the full version chain; delete an active memory; and refresh to prove it disappears without deleting source evidence.

- [ ] **Step 5: Run tests.**

```bash
PYTHONPATH=src python -m pytest tests/api/test_conversation.py tests/integration/test_smart_assistant_lifecycle.py -q
npm --prefix src/underwater_tracking/ui test -- --run src/components/assistant/SmartAssistantPanel.test.tsx src/components/assistant/MemoryWindow.test.tsx
```

- [ ] **Step 6: Commit.**

```bash
git add tests/api/test_conversation.py tests/integration/test_smart_assistant_lifecycle.py src/underwater_tracking/ui/src/components/assistant/SmartAssistantPanel.test.tsx src/underwater_tracking/ui/src/components/assistant/MemoryWindow.test.tsx
git commit -m "test: verify smart assistant and evidence lifecycle"
```

---

### Task 6: Isolate local and live Playwright suites

**Files:**
- Modify: `src/underwater_tracking/ui/playwright.config.ts`
- Create: `src/underwater_tracking/ui/playwright.live.config.ts`
- Move: `tests/e2e/command-center.spec.ts` -> `src/underwater_tracking/ui/e2e/command-center.spec.ts`
- Move: `tests/e2e/command-center.spec.ts-snapshots` -> `src/underwater_tracking/ui/e2e/command-center.spec.ts-snapshots`
- Move: `tests/e2e/uuv-live-timeline.spec.ts` -> `src/underwater_tracking/ui/e2e/uuv-live-timeline.spec.ts`
- Delete after merging coverage: `src/underwater_tracking/ui/src/e2e/visualCommandCenterFlow.test.ts`
- Modify: `src/underwater_tracking/ui/src/components/CanvasMap.tsx`
- Modify: `src/underwater_tracking/ui/package.json`

**Interfaces:**

Local config:

```typescript
export default defineConfig({
  testDir: "e2e",
  testMatch: "command-center.spec.ts",
  fullyParallel: true,
});
```

Live config:

```typescript
export default defineConfig({
  testDir: "e2e",
  testMatch: "uuv-live-timeline.spec.ts",
  fullyParallel: false,
  workers: 1,
  use: { baseURL: process.env.PLAYWRIGHT_BASE_URL },
});
```

- [ ] **Step 1: Move tests inside the UI package and import `@playwright/test` normally.**

Remove imports that name `ui/node_modules`. The local config must never traverse repository `.worktrees`.

- [ ] **Step 2: Combine the stateful live timeline into one serial test.**

Merge the operator/API coverage from `src/e2e/visualCommandCenterFlow.test.ts` into the timeline before deleting it. One live test then owns the sequence initial -> plan -> deploy -> scan -> adversary -> passive track -> handoff -> recovery -> returned. It records desktop screenshots at semantic checkpoints, then validates the current final state at 390x844. It does not assert “all onboard” in a later independent test after the shared simulation has advanced.

- [ ] **Step 3: Add semantic canvas attributes.**

```tsx
<canvas
  data-carrier-count={frame.carriers.length}
  data-waterborne-uuv-count={waterborneUuvs(frame).length}
  data-target-estimate-count={frame.target_estimates.length}
  data-plan-version={frame.plan_version}
/>
```

Playwright asserts these values against `/api/operational/snapshot`, then performs a pixel-variance check and screenshot. Non-transparent background pixels alone are insufficient.

- [ ] **Step 4: Add no-overflow/no-error checks at both viewports.**

Assert `document.documentElement.scrollWidth <= innerWidth`, visible sidebar/drawer text fits its bounding box, no console/page errors occur, and each required assistant/memory control is reachable.

- [ ] **Step 5: Add package scripts.**

```json
{
  "test:e2e": "playwright test --config playwright.config.ts",
  "test:e2e:live": "playwright test --config playwright.live.config.ts"
}
```

- [ ] **Step 6: Run local Playwright and build.**

```bash
npm --prefix src/underwater_tracking/ui run build
npm --prefix src/underwater_tracking/ui run test:e2e
```

Expected: no duplicate Playwright installation error and all local projects pass.

- [ ] **Step 7: Commit.**

```bash
git add src/underwater_tracking/ui/playwright.config.ts src/underwater_tracking/ui/playwright.live.config.ts src/underwater_tracking/ui/e2e src/underwater_tracking/ui/src/e2e/visualCommandCenterFlow.test.ts src/underwater_tracking/ui/src/components/CanvasMap.tsx src/underwater_tracking/ui/package.json tests/e2e
git commit -m "test: isolate live command center browser acceptance"
```

---

### Task 7: Make real-provider tests explicit and role-complete

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`
- Modify: `tests/integration/test_llm_real_api.py`
- Modify: `tests/memory/test_real_llm_memory.py`
- Create: `tests/integration/test_real_role_smoke.py`

**Interfaces:**

```python
def real_provider_enabled() -> bool:
    return os.environ.get("UNDERWATER_TRACKING_RUN_REAL_LLM") == "1" and has_live_api_key()
```

- [ ] **Step 1: Add collection-time isolation.**

When `UNDERWATER_TRACKING_RUN_REAL_LLM != 1`, every `real_llm` test skips even if config or `.env` contains a key. Normal `pytest` must make zero outbound provider calls.

- [ ] **Step 2: Normalize structured response contracts.**

Real tests assert typed domain outputs, not provider-specific extra keys such as raw `commit_status`. Graph tests inspect graph state after typed validation; they do not index unvalidated provider JSON.

- [ ] **Step 3: Add one role-complete smoke.**

Invoke master planning, a deployed-group slave decision, target initial/contact decision, memory filter/extract, and short-term compression. Assert ledger operations include:

```python
{
    "intent",
    "regional_strategy",
    "slave_sonar_decision",
    "adversary_mission_decision",
    "memory_filter",
    "memory_extract",
    "memory_compress",
}
```

Validate schemas, evidence IDs, and non-empty latency; never print prompt, response, or credentials.

- [ ] **Step 4: Verify ordinary suite performs no real calls.**

```bash
unset UNDERWATER_TRACKING_RUN_REAL_LLM
PYTHONPATH=src python -m pytest -m real_llm -q
```

Expected: all selected tests skip.

- [ ] **Step 5: Run explicit smoke when authorized.**

```bash
UNDERWATER_TRACKING_RUN_REAL_LLM=1 PYTHONPATH=src python -m pytest tests/integration/test_real_role_smoke.py -q
```

Expected: pass with all required ledger operation names.

- [ ] **Step 6: Commit.**

```bash
git add pyproject.toml tests/conftest.py tests/integration/test_llm_real_api.py tests/memory/test_real_llm_memory.py tests/integration/test_real_role_smoke.py
git commit -m "test: isolate and verify real role providers"
```

---

### Task 8: Add a default-main semantic acceptance driver

**Files:**
- Create: `tools/run_default_live_acceptance.py`
- Create: `tests/acceptance/test_default_live_acceptance_driver.py`
- Create: `tests/acceptance/test_default_live_acceptance.py`
- Modify: `pyproject.toml`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class AcceptanceCheckpoint:
    name: str
    event_type: str | None
    predicate: Callable[[dict[str, object]], bool]
    timeout_s: float

def run_acceptance(
    *,
    command: tuple[str, ...],
    api_port: int,
    ui_port: int,
    output_path: Path,
    checkpoints: tuple[AcceptanceCheckpoint, ...],
    playwright_command: tuple[str, ...] | None = None,
) -> int: ...
```

The CLI wrapper accepts `--config`, `--seed`, and `--output`, then constructs `command=(sys.executable, "main.py", "--config", ..., "--seed", ...)` before calling `run_acceptance()`. Unit tests may pass a fixture-server command directly.

- [ ] **Step 1: Unit-test subprocess ownership, polling, timeout, and cleanup with a local fixture server.**

The driver allocates ports, appends `--host 127.0.0.1 --port <api> --ui-port <ui>` to the `main.py` command, launches it in a new process group, polls health/snapshot/replay with bounded request timeouts, runs the optional Playwright command while that same process is alive with `PLAYWRIGHT_BASE_URL` injected, sends one `SIGINT`, waits ten seconds, and kills only its validated process group on failure.

- [ ] **Step 2: Encode semantic checkpoints.**

Required order:

```text
health_ready
plan_committed
carrier_dispatched
uuv_deployed
active_scan
target_detection_acquired
adversary_decision
passive_track
handoff_completed
uuv_recovered
carrier_returned_to_fleet
memory_source_processed
```

After `plan_committed`, run a second ordered operator/memory lane concurrently with the physical lane:

```text
memory_version_1_created
 -> memory_version_2_created
 -> evidence_trace_completed
 -> assistant_preview_created
 -> assistant_plan_applied
 -> memory_version_deleted
```

The lanes may interleave, but order inside each lane is strict and final shutdown waits for both.

Each checkpoint records sim time, wall time, frame ID, plan version, event/evidence IDs, and health latency. Checkpoint search uses replay pagination/cursors and never requests an unbounded history. Poll `/api/health` once per wall-clock second for the whole run, retain every latency sample, and require at least 60 samples before computing p95.

Use wall-clock deadlines: 30 seconds for `health_ready`; 300 seconds for `plan_committed` and `adversary_decision`; 180 seconds each for dispatch, deployment, scan, detection, tracking and memory processing; and 300 seconds each for handoff, recovery and fleet return. Enforce a 1800-second global deadline. A timeout report names the last frame/event cursor and current epoch/mission state.

- [ ] **Step 3: Drive assistant and memory state through public APIs.**

After the first mission commit, submit one explicit durable preference, wait for memory version 1, submit a conflicting update, and wait for version 2/superseded version 1. Ask an evidence question and verify its source chain. Submit non-disruptive expert feedback “保持当前任务区域与 UUV 分配，仅重新确认交接窗口” with the current version, verify preview causes no execution change, apply it, and wait for the next committed plan version. Reject any preview that changes region/member IDs in this acceptance run. The live Playwright test captures the four memory tabs and version chain before deleting version 2, then performs deletion and verifies source evidence remains.

- [ ] **Step 4: Add final database checks after shutdown.**

Open the run SQLite read-only and assert at least one master, slave, adversary and memory LLM call, at least two committed plan versions with an active/superseded chain, non-empty memory stream, and the expected memory version/evidence/deletion records. Check the loaded default config, runtime platform registry, every persisted operational frame, and paginated replay for zero USV entities/fields; do not inspect only the current frame.

- [ ] **Step 5: Add marker and explicit enable flag.**

Register `live_acceptance` only on `tests/acceptance/test_default_live_acceptance.py`. That live test skips unless both `UNDERWATER_TRACKING_RUN_LIVE_ACCEPTANCE=1` and real-provider enablement are present. `test_default_live_acceptance_driver.py` is an unmarked offline unit suite and must always run without network/provider access.

- [ ] **Step 6: Run driver unit tests.**

```bash
PYTHONPATH=src python -m pytest tests/acceptance/test_default_live_acceptance_driver.py -q
ruff check tools/run_default_live_acceptance.py tests/acceptance/test_default_live_acceptance_driver.py
mypy tools/run_default_live_acceptance.py
```

- [ ] **Step 7: Commit.**

```bash
git add tools/run_default_live_acceptance.py tests/acceptance/test_default_live_acceptance_driver.py tests/acceptance/test_default_live_acceptance.py pyproject.toml
git commit -m "test: add default main semantic acceptance driver"
```

---

### Task 9: Execute the final backend, provider, browser, and shutdown gates

**Files:**
- Create: `docs/superpowers/reports/2026-08-22-end-to-end-adversarial-runtime-acceptance.md`

**Interfaces:**
- Consumes: all four implementation plans and their phase reports.
- Produces: the only report allowed to claim the total design is implemented.

- [ ] **Step 1: Run the complete offline backend suite.**

```bash
unset UNDERWATER_TRACKING_RUN_REAL_LLM UNDERWATER_TRACKING_RUN_LIVE_ACCEPTANCE
PYTHONPATH=src python -m pytest -m "not real_llm and not long_running and not live_acceptance" -q
ruff check main.py src tests tools
mypy src/underwater_tracking tools/run_default_live_acceptance.py
```

Expected: zero failures, zero Ruff violations, zero mypy errors, and zero outbound provider calls.

- [ ] **Step 2: Run all frontend unit/build/local browser gates.**

```bash
npm --prefix src/underwater_tracking/ui test -- --run
npm --prefix src/underwater_tracking/ui run build
npm --prefix src/underwater_tracking/ui run test:e2e
```

Expected: all commands pass.

- [ ] **Step 3: Run explicit real-role smoke.**

```bash
UNDERWATER_TRACKING_RUN_REAL_LLM=1 PYTHONPATH=src python -m pytest tests/integration/test_real_role_smoke.py tests/memory/test_real_llm_memory.py -q
```

Expected: all typed role contracts and ledger assertions pass.

- [ ] **Step 4: Launch the default live acceptance.**

```bash
UNDERWATER_TRACKING_RUN_REAL_LLM=1 \
UNDERWATER_TRACKING_RUN_LIVE_ACCEPTANCE=1 \
PYTHONPATH=src python tools/run_default_live_acceptance.py \
  --config configs/scenario/uuv_only_single_target.yaml \
  --seed 42 \
  --playwright-command "npm --prefix src/underwater_tracking/ui run test:e2e:live" \
  --output outputs/acceptance-2026-08-22.json
```

Expected: every semantic checkpoint and the live Playwright command pass against the same owned process, at least 60 whole-run health samples yield p95 below 500 ms in the acceptance environment, and one `SIGINT` exits with code 130 within 10 seconds.

- [ ] **Step 5: Verify browser-process coupling in the acceptance artifact.**

The driver must record the allocated UI URL, Playwright command, exit code, screenshots, and start/end timestamps in machine-readable JSON. Assert the browser interval is contained within the owned `main.py` process interval; do not use or print a manually substituted shared port.

- [ ] **Step 6: Review screenshots and semantic assertions.**

Required viewports are 1440x900 and 390x844. Verify actual carrier/UUV/target/task state, no overlaps or horizontal overflow, assistant preview/apply, evidence trace, four memory views, version chain/delete, LLM Thinking, and non-empty Memory Steam. Capture the version chain before deletion and the absence of active version 2 after deletion. The user's reference PNGs remain unmodified comparison inputs.

- [ ] **Step 7: Write the final report.**

Include commits under test, config hash, seed, exact commands/exit codes, package versions, all phase report links, checkpoint timing table, provider ledger operation table, trajectory constraint maxima, API latency, memory version/evidence chain, Playwright screenshots, shutdown duration, and known residual risks. Do not include credentials, raw prompts/responses, private target truth, or chain-of-thought.

- [ ] **Step 8: Commit the report only after every gate passes.**

```bash
git add docs/superpowers/reports/2026-08-22-end-to-end-adversarial-runtime-acceptance.md
git commit -m "test: document end to end adversarial runtime acceptance"
```
