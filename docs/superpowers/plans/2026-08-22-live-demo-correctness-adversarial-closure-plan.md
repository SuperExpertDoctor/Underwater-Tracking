# Live Demo Correctness and Adversarial Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the default `main.py` produce a truthful, bounded, end-to-end UUV-versus-submarine demonstration with terminal planning epochs, correct event isolation, controlled runtime output, honest UI state, and constrained three-degree-of-freedom submarine motion.

**Architecture:** Keep `MissionController` as the execution state source and keep LLM/memory work off the physics path. Add strict contracts around regional-policy authority, planning terminal results, event audiences, bootstrap readiness, scenario completion, sampled persistence, and role-level UI status. Extend only submarine motion with depth and vertical limits while preserving the existing x/y map projection.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, LangGraph, SQLite, httpx, FastAPI, deterministic simulation, React 18, TypeScript, Vitest, Playwright, pytest, Ruff, mypy.

## Global Constraints

- Live platforms are exactly one carrier, three mother ships, twelve UUVs, one submarine, and zero USVs.
- All twelve UUVs remain onboard until a committed plan sends a mother ship to a physical deployment point.
- LongCat selects regional policy; deterministic code owns candidate topology, final region cap, transport, handoff, recovery, and return routes.
- A reserved or running planning epoch produces exactly one of `committed`, `invalidated`, `rejected`, or `failed`.
- The initial executable plan must commit before simulation time advances beyond the bootstrap frame.
- The initial planning wall-clock deadline is 180 seconds; failure enters `AWAITING_RETRY` without advancing simulation time.
- Physics runs at `demo_time_scale=60` only after the initial plan commits.
- Target-private decisions never enter blue planning inputs.
- The default run stops physics at `scenario.duration_s=28800`; only explicit `--continuous` may exceed it.
- Persisted operational frames are sampled at 30 simulation seconds plus state/event boundaries, and one default run stays below 250 MiB.
- UI never derives overall planning success from a successful LLM sub-call.
- Submarine depth uses metres positive down and is bounded by configured depth, vertical-speed, vertical-acceleration, and pitch limits.
- Real-provider release acceptance may be skipped or blocked when credentials/provider are unavailable, but may never be replaced by fake-provider success.

---

### Task 1: Separate LLM regional decisions from deterministic handoff topology

**Files:**
- Modify: `src/underwater_tracking/domain/regional_models.py`
- Modify: `src/underwater_tracking/agent/prompts.py`
- Modify: `src/underwater_tracking/agent/nodes/regional_strategy.py`
- Modify: `src/underwater_tracking/planning/regional_plan_validator.py`
- Modify: `src/underwater_tracking/planning/mission_optimizer.py`
- Modify: `tests/planning/test_regional_plan_validator.py`
- Modify: `tests/agent/test_regional_strategy.py`
- Modify: `tests/integration/test_uuv_only_production_acceptance.py`

**Interfaces:**

```python
class UUVRegionalPolicyDecision(StrictModel):
    candidate_id: str
    coverage_mode: RegionCoverageMode
    tracking_mode: UUVRegionalTrackingMode
    priority: float
    required_quality: UnitFloat
    active_scan_uuv_count: int = 1
    passive_track_uuv_count: int = 1
    reserve_uuv_count: int = 0
    optional_uuv_count: int = 0
    assigned_uuv_ids: tuple[str, ...] = ()
    rationale: str
    evidence_ids: tuple[str, ...]


class UUVRegionalStrategyDecisionSet(StrictModel):
    policies: tuple[UUVRegionalPolicyDecision, ...]


def resolve_uuv_strategy(
    candidates: Sequence[RegionalMissionCandidate],
    decisions: UUVRegionalStrategyDecisionSet,
    available_uuvs: AvailableUUVs,
) -> ValidatedRegionalStrategy: ...
```

`UUVRegionalPolicy` remains the internal resolved policy and receives predecessor/successor IDs only from `RegionalMissionCandidate` and the deterministic optimizer.

- [ ] **Step 1: Write a regression test for the audited cross-batch failure.**

```python
def test_cross_batch_predecessor_is_resolved_from_complete_candidate_graph() -> None:
    candidates = make_linear_candidates(8)
    decisions = UUVRegionalStrategyDecisionSet(
        policies=tuple(decision_for(candidate.candidate_id) for candidate in candidates),
    )
    resolved = resolve_uuv_strategy(candidates, decisions, available_uuvs())
    assert resolved.policies[4].predecessor_candidate_id == candidates[3].candidate_id
    assert resolved.policies[3].successor_candidate_id == candidates[4].candidate_id
```

Add a second test proving an LLM response containing `predecessor_candidate_id` is rejected by the strict decision schema instead of being treated as internal failure.

- [ ] **Step 2: Run the regression tests and verify failure.**

Run:

```bash
PYTHONPATH=src pytest -q \
  tests/planning/test_regional_plan_validator.py \
  tests/agent/test_regional_strategy.py
```

Expected: FAIL because the current response model grants topology fields to the LLM and validates relations against each four-candidate batch.

- [ ] **Step 3: Add the topology-free LLM response models and prompt schema.**

Use `UUVRegionalStrategyDecisionSet` as the response model in `_invoke_uuv()`. The prompt must state that the model selects policy only and must not emit predecessor/successor or route geometry. Keep compatibility deserialization for old persisted `UUVRegionalStrategySet` outside the live node.

- [ ] **Step 4: Validate each batch locally, then resolve the merged decision set globally.**

Implement the node flow as:

```python
batch_decisions = validate_uuv_decision_batch(batch, llm_response, resources)
merged.extend(batch_decisions.policies)
resolved = resolve_uuv_strategy(
    normalized_candidates,
    UUVRegionalStrategyDecisionSet(policies=tuple(merged)),
    resources,
)
```

`resolve_uuv_strategy()` must copy only topology declared by each immutable candidate and must reject duplicate candidate policies, missing policies, unknown UUVs, overlapping hard locks, and resource-count violations.

- [ ] **Step 5: Add one bounded semantic correction.**

Catch `RegionalPlanError` around validation, invoke the same batch once with:

```python
correction_feedback = {
    "category": "semantic",
    "message": str(error),
    "allowed_candidate_ids": [candidate.candidate_id for candidate in batch],
}
```

Do not retry a second semantic failure. Raise `RegionalSemanticRejection` so Task 2 can create a terminal rejected epoch.

- [ ] **Step 6: Keep batch result ordering deterministic.**

If batch calls are executed concurrently, cap the executor at three workers, collect by original batch index, cancel outstanding futures on shutdown, and merge only after every batch succeeds. Add a test with deliberately reversed completion order and assert identical request/response hashes and resolved policy order.

- [ ] **Step 7: Verify and commit.**

```bash
PYTHONPATH=src pytest -q \
  tests/planning/test_regional_plan_validator.py \
  tests/agent/test_regional_strategy.py \
  tests/integration/test_uuv_only_production_acceptance.py
ruff check src/underwater_tracking/domain/regional_models.py \
  src/underwater_tracking/agent/nodes/regional_strategy.py \
  src/underwater_tracking/planning/regional_plan_validator.py
mypy src/underwater_tracking/domain/regional_models.py \
  src/underwater_tracking/agent/nodes/regional_strategy.py \
  src/underwater_tracking/planning/regional_plan_validator.py
git add src/underwater_tracking/domain/regional_models.py \
  src/underwater_tracking/agent/prompts.py \
  src/underwater_tracking/agent/nodes/regional_strategy.py \
  src/underwater_tracking/planning/regional_plan_validator.py \
  src/underwater_tracking/planning/mission_optimizer.py \
  tests/planning/test_regional_plan_validator.py \
  tests/agent/test_regional_strategy.py \
  tests/integration/test_uuv_only_production_acceptance.py
git commit -m "fix: keep regional handoff topology deterministic"
```

---

### Task 2: Make every planning epoch terminal and keep latest physics fresh

**Files:**
- Modify: `src/underwater_tracking/agent/state.py`
- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Modify: `src/underwater_tracking/domain/planning_epoch_models.py`
- Modify: `src/underwater_tracking/runtime/planning_epoch.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `tests/agent/test_epoch_commit_graph.py`
- Modify: `tests/agent/test_background_cycle.py`
- Modify: `tests/runtime/test_planning_epoch.py`
- Create: `tests/integration/test_planning_epoch_terminal_contract.py`

**Interfaces:**

```python
EpochFailureCategory = Literal[
    "schema", "content", "semantic", "stale", "timeout",
    "provider", "internal", "cancelled",
]


class FinalizeEpochNode:
    def __call__(self, state: CentralState) -> CentralState:
        """Return exactly one EpochCommitResult for an active epoch."""


class PlanningEpochCoordinator:
    def observe(self, situation: SituationSnapshot) -> None: ...
    def request(self, triggers: tuple[EpochTrigger, ...]) -> None: ...
```

- [ ] **Step 1: Add failing graph-exit tests.**

Parameterize every error-producing node (`regional_strategy`, adapter, semantic verification, optimizer, plan verification) and assert:

```python
result = graph.invoke(state_with_epoch(epoch, forced_error=node_name))
terminal = result["epoch_commit_result"]
assert terminal.epoch_id == epoch.epoch_id
assert terminal.status in {"rejected", "failed"}
```

Add a guard test that graph completion with an active epoch and no result raises `PlanningEpochInvariantError` during tests.

- [ ] **Step 2: Add a failing freshness test.**

Start a blocking background cycle at revision 1, submit revisions 2 through 20, and assert before releasing the provider:

```python
assert loop.planning_health().base_physics_revision == 1
assert loop.planning_health().current_physics_revision == 20
```

Expected current behavior: current revision remains 1.

- [ ] **Step 3: Add `FinalizeEpochNode` and route all graph exits through it.**

Replace `handle_error -> END` with `handle_error -> finalize_epoch -> END`. Successful commit paths also pass through `finalize_epoch`, which preserves an existing authoritative result and rejects a second result. Map `RegionalSemanticRejection` to `rejected/semantic`; map transport exceptions to `failed/timeout|provider`; map invariant errors to `failed/internal`.

- [ ] **Step 4: Stop creating epochs for informational events.**

In `_prepare_epoch()`, call `coordinator.request()` only when the normalized event is strategic and has `plan_impact=True`, or is initialization/expert-confirmation. Informational events still persist and enter MemoryWorker but cannot reserve an epoch.

- [ ] **Step 5: Observe every situation before background-cycle branching.**

At the start of `on_situation()`:

```python
coordinator = self._epoch_coordinator
if coordinator is not None:
    coordinator.observe(situation)
self._submit_due_periodic_summary(situation)
```

Remove the duplicate `observe()` from `_prepare_epoch()`. Mailbox updates remain idempotent and do not replace the captured situation of the active epoch.

- [ ] **Step 6: Make `_finish_epoch()` an assertion boundary.**

It must accept only a matching `EpochCommitResult`. For production invariant failure, create one `failed/internal` result and increment `planning_epoch_invariant_failures`; tests must fail if this metric is nonzero. Do not classify ordinary semantic node errors as internal.

- [ ] **Step 7: Fix retry categories.**

Set transient categories to `timeout` and `provider` only. `internal`, `schema`, `content`, `semantic`, `cancelled`, and stale version errors do not automatically retry the same event. Persist dead-letter reason and expose an explicit expert retry method.

- [ ] **Step 8: Verify and commit.**

```bash
PYTHONPATH=src pytest -q \
  tests/agent/test_epoch_commit_graph.py \
  tests/agent/test_background_cycle.py \
  tests/runtime/test_planning_epoch.py \
  tests/integration/test_planning_epoch_terminal_contract.py
ruff check src/underwater_tracking/agent/graphs/central.py \
  src/underwater_tracking/runtime/planning_epoch.py src/underwater_tracking/cli.py
mypy src/underwater_tracking/agent/graphs/central.py \
  src/underwater_tracking/runtime/planning_epoch.py src/underwater_tracking/cli.py
git add src/underwater_tracking/agent/state.py \
  src/underwater_tracking/agent/graphs/central.py \
  src/underwater_tracking/domain/planning_epoch_models.py \
  src/underwater_tracking/runtime/planning_epoch.py src/underwater_tracking/cli.py \
  tests/agent/test_epoch_commit_graph.py tests/agent/test_background_cycle.py \
  tests/runtime/test_planning_epoch.py \
  tests/integration/test_planning_epoch_terminal_contract.py
git commit -m "fix: guarantee terminal planning epoch results"
```

---

### Task 3: Enforce adversarial event audiences and a single public-event registry

**Files:**
- Create: `src/underwater_tracking/domain/event_registry.py`
- Modify: `src/underwater_tracking/domain/models.py`
- Modify: `src/underwater_tracking/agent/nodes/event_monitor.py`
- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/persistence/events.py`
- Modify: `src/underwater_tracking/memory/source_reader.py`
- Modify: `tests/agent/test_event_monitor.py`
- Modify: `tests/simulation/test_adversary_local_sensing.py`
- Modify: `tests/domain/test_truth_boundary.py`
- Create: `tests/integration/test_adversarial_event_audience.py`

**Interfaces:**

```python
class EventAudience(StrEnum):
    BLUE_PLANNING = "blue_planning"
    ADVERSARY_PRIVATE = "adversary_private"
    OPERATOR_AUDIT = "operator_audit"
    MEMORY_SOURCE = "memory_source"


class RuntimeEvent(StrictModel):
    event_id: str
    scenario_id: str
    sim_time_s: int
    event_type: str
    entity_id: str | None = None
    level: EventLevel
    audiences: frozenset[EventAudience]
    payload: dict[str, Any] = {}


class EventDefinition(StrictModel):
    event_type: str
    default_level: EventLevel
    audiences: frozenset[EventAudience]
    plan_impact_policy: Literal["always", "evidence_required", "never"]
    coalescing_family: str | None = None
```

- [ ] **Step 1: Add a truth-boundary regression test.**

Apply an `AdversaryIntentDecision`, build the next blue `SituationSnapshot`, and assert:

```python
assert all(event.event_type != "target_mission_decision" for event in snapshot.pending_events)
audit = event_repository.get(event_id)
assert EventAudience.OPERATOR_AUDIT in audit.audiences
assert EventAudience.BLUE_PLANNING not in audit.audiences
assert "guidance_waypoint_xy" not in blue_snapshot_json
```

Add a positive test: a sensor-derived `target_maneuver_observed` with observation IDs is visible to blue planning and classified according to the registry.

- [ ] **Step 2: Run tests and confirm private decision leakage.**

```bash
PYTHONPATH=src pytest -q \
  tests/domain/test_truth_boundary.py \
  tests/integration/test_adversarial_event_audience.py
```

- [ ] **Step 3: Define the registry and replace duplicated routing sets.**

Move all public event metadata from `_STRATEGIC_TYPES`, `_TACTICAL_TYPES`, `_INFORMATIONAL_TYPES`, `_ALWAYS_STRATEGIC_EVENT_TYPES`, and coalescing maps into `EVENT_REGISTRY`. `EventMonitor.classify()` must read only this registry and continue to reject unknown public event types.

- [ ] **Step 4: Mark target decisions private.**

Persist `target_mission_decision` with audiences `{ADVERSARY_PRIVATE, OPERATOR_AUDIT, MEMORY_SOURCE}`. The carrier projection filters on `BLUE_PLANNING`. The operator frame may show the adversary decision record by ID, but blue prompts, target estimates, plan evidence and event-monitor inputs cannot contain its private fields.

- [ ] **Step 5: Add observable target events.**

Generate `target_maneuver_observed`, `target_speed_regime_changed`, and `target_depth_regime_changed` only from fused observations/estimates. Require non-empty public observation IDs in payload validation. These events use `{BLUE_PLANNING, OPERATOR_AUDIT, MEMORY_SOURCE}`.

- [ ] **Step 6: Persist and migrate event audiences.**

Add an `audiences_json` column when absent. Existing rows migrate to the legacy public audience set, except known historical `target_mission_decision` rows, which migrate to private/audit/memory. MemorySourceReader may consume `MEMORY_SOURCE` events but must preserve the source audience metadata in evidence chains.

- [ ] **Step 7: Add producer/registry completeness tests.**

Collect literal `event_type=` values from engine/controller/runtime fixtures and assert every event entering blue planning exists in `EVENT_REGISTRY`. Explicitly test `target_mission_decision` never reaches EventMonitor, preventing the audited unknown-event crash without making private truth public.

- [ ] **Step 8: Verify and commit.**

```bash
PYTHONPATH=src pytest -q \
  tests/agent/test_event_monitor.py \
  tests/simulation/test_adversary_local_sensing.py \
  tests/domain/test_truth_boundary.py \
  tests/integration/test_adversarial_event_audience.py
ruff check src/underwater_tracking/domain/event_registry.py \
  src/underwater_tracking/domain/models.py \
  src/underwater_tracking/agent/nodes/event_monitor.py
mypy src/underwater_tracking/domain/event_registry.py \
  src/underwater_tracking/domain/models.py
git add src/underwater_tracking/domain/event_registry.py \
  src/underwater_tracking/domain/models.py \
  src/underwater_tracking/agent/nodes/event_monitor.py \
  src/underwater_tracking/agent/graphs/central.py \
  src/underwater_tracking/simulation/engine.py \
  src/underwater_tracking/persistence/events.py \
  src/underwater_tracking/memory/source_reader.py \
  tests/agent/test_event_monitor.py \
  tests/simulation/test_adversary_local_sensing.py \
  tests/domain/test_truth_boundary.py \
  tests/integration/test_adversarial_event_audience.py
git commit -m "fix: isolate adversary-private runtime events"
```

---

### Task 4: Add bootstrap planning readiness and an explicit retry phase

**Files:**
- Modify: `src/underwater_tracking/config/models.py`
- Modify: `configs/scenario/uuv_only_single_target.yaml`
- Modify: `src/underwater_tracking/runtime/models.py`
- Modify: `src/underwater_tracking/runtime/run_controller.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `src/underwater_tracking/api/app.py`
- Modify: `tests/runtime/test_run_controller.py`
- Modify: `tests/integration/test_slow_llm_api_responsiveness.py`
- Create: `tests/integration/test_bootstrap_planning_gate.py`

**Interfaces:**

```python
class RunPhase(StrEnum):
    CREATED = "created"
    BOOTSTRAP_PLANNING = "bootstrap_planning"
    AWAITING_RETRY = "awaiting_retry"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class PlanningRuntimeConfig(StrictModel):
    initial_plan_timeout_s: float = Field(default=180.0, gt=0)
    regional_batch_size: int = Field(default=4, ge=1, le=4)
    regional_max_concurrency: int = Field(default=3, ge=1, le=3)
    semantic_correction_attempts: int = Field(default=1, ge=0, le=1)


class _AgentLoop:
    def begin_bootstrap_planning(self, situation: SituationSnapshot) -> None: ...
    def bootstrap_result(self) -> EpochCommitResult | None: ...
    def retry_initial_planning(self, *, expected_epoch_id: str | None) -> str: ...
```

- [ ] **Step 1: Add a blocking-provider bootstrap test.**

Start RunController with a provider blocked for one second. During the block assert:

```python
summary = controller.current()
assert summary.phase is RunPhase.BOOTSTRAP_PLANNING
assert summary.sim_time_s == 0
assert controller.mission_controller.snapshot().plan_revision == 0
assert client.get("/api/health").elapsed.total_seconds() < 0.5
```

Release a valid plan and assert phase becomes RUNNING before `sim_time_s` becomes positive.

- [ ] **Step 2: Add timeout and retry tests.**

Use a 50 ms deadline and a blocked provider. Assert `AWAITING_RETRY`, sim time 0, all UUVs onboard, and no automatic new provider call after the deadline. POST the retry endpoint with the current failed epoch ID, release a valid result, and assert exactly one new epoch starts and commits.

- [ ] **Step 3: Publish the bootstrap frame before starting planners.**

Keep the existing initial OperationalFrame. After it is in the hub/logger, call `begin_bootstrap_planning(engine.publication_situation())`. Do not require `engine.step()` to create the initialization epoch.

- [ ] **Step 4: Gate the physics worker on a committed executable plan.**

In `RunController._start_worker()`:

```python
while bundle.phase is RunPhase.BOOTSTRAP_PLANNING:
    outcome = bundle.loop.bootstrap_result()
    if outcome is not None:
        bundle.phase = (
            RunPhase.RUNNING
            if outcome.status == "committed"
            else RunPhase.AWAITING_RETRY
        )
        break
    if bundle.stop.wait(0.05):
        return
```

The timeout owner is AgentLoop/coordinator; it cancels the active provider and emits one terminal failed result. The worker must not poll SQLite while holding the RunController lock.

- [ ] **Step 5: Add the explicit retry API.**

Implement `POST /api/runs/current/planning/retry` with `expected_epoch_id`. Reject while RUNNING/COMPLETED, reject stale IDs with HTTP 409, and return the new epoch ID without waiting for provider completion.

- [ ] **Step 6: Verify and commit.**

```bash
PYTHONPATH=src pytest -q \
  tests/runtime/test_run_controller.py \
  tests/integration/test_slow_llm_api_responsiveness.py \
  tests/integration/test_bootstrap_planning_gate.py
ruff check src/underwater_tracking/runtime/models.py \
  src/underwater_tracking/runtime/run_controller.py src/underwater_tracking/cli.py
mypy src/underwater_tracking/runtime/models.py \
  src/underwater_tracking/runtime/run_controller.py
git add src/underwater_tracking/config/models.py \
  configs/scenario/uuv_only_single_target.yaml \
  src/underwater_tracking/runtime/models.py \
  src/underwater_tracking/runtime/run_controller.py \
  src/underwater_tracking/cli.py src/underwater_tracking/api/app.py \
  tests/runtime/test_run_controller.py \
  tests/integration/test_slow_llm_api_responsiveness.py \
  tests/integration/test_bootstrap_planning_gate.py
git commit -m "fix: gate live physics on initial planning"
```

---

### Task 5: Stop at scenario duration, sample persisted frames, and close owned resources

**Files:**
- Modify: `src/underwater_tracking/config/models.py`
- Modify: `src/underwater_tracking/runtime/models.py`
- Modify: `src/underwater_tracking/runtime/run_controller.py`
- Modify: `src/underwater_tracking/api/frame_logger.py`
- Modify: `src/underwater_tracking/api/live.py`
- Modify: `src/underwater_tracking/api/replay.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `main.py`
- Modify: `tests/runtime/test_run_controller.py`
- Modify: `tests/api/test_replay_compatibility.py`
- Modify: `tests/integration/test_uuv_only_8h_replay_acceptance.py`
- Modify: `tests/integration/test_main_shutdown.py`

**Interfaces:**

```python
class OperationalFrameLogConfig(StrictModel):
    sample_interval_s: int = Field(default=30, ge=5)
    max_run_bytes: int = Field(default=262_144_000, gt=0)


class FramePersistencePolicy:
    def should_persist(
        self,
        frame: OperationalFrame,
        previous: OperationalFrame | None,
    ) -> bool: ...


class ShutdownReport(StrictModel):
    completed: bool
    remaining_resources: tuple[str, ...] = ()
```

- [ ] **Step 1: Add a duration regression test.**

Run a short scenario with `duration_s=20`, `physics_step_s=5`, and a committed bootstrap plan. Assert the last frame is between 20 and 25 seconds, phase is COMPLETED, the worker exits, and API snapshot/replay remain available.

- [ ] **Step 2: Add a persistence-policy test.**

Publish frames every 5 seconds for one simulated hour. Assert persisted timestamps are every 30 seconds, plus injected plan/event/phase transitions. Every persisted frame must parse independently and ReplayService must seek across sparse timestamps.

- [ ] **Step 3: Add the 8-hour size gate before implementation.**

Update `test_uuv_only_8h_replay_acceptance.py` to write real OperationalFrame JSON through FrameLogger and assert:

```python
assert frame_log.stat().st_size < 250 * 1024 * 1024
assert last_frame.sim_time_s <= config.scenario.duration_s + config.timing.physics_step_s
```

- [ ] **Step 4: Stop the worker at duration unless continuous mode is explicit.**

Add `continuous: bool = False` to RunController and `--continuous` to `main.py`. The loop condition must include the duration boundary. Set phase COMPLETED without setting the operator stop flag so summary distinguishes completion from manual stop.

- [ ] **Step 5: Add sampled persistence without changing live publication.**

`OperationalFramePublisher` always publishes to the hub. It invokes FrameLogger only when `FramePersistencePolicy.should_persist()` returns true. State boundaries include run phase, plan version, epoch terminal status, deployment/group state, and critical mission events. Persist the final completed frame even when it is off the regular sample boundary.

- [ ] **Step 6: Enforce the per-run byte ceiling.**

Before append, FrameLogger checks projected bytes. When the ceiling would be exceeded, write one compact `frame_log_limit_reached` terminal record, stop further frame persistence, keep live hub publication running, and expose `log_truncated=true` in the run manifest. Do not delete or rotate historical files automatically.

- [ ] **Step 7: Return a concrete shutdown report.**

Change close paths to cancel role HTTP clients before joining the background planning thread. Record remaining owner names such as `physics-worker`, `carrier-llm`, `memory-worker`, `vite:<pid>`, or `http-client:<role>`. Ensure a second close is a no-op. The CLI may print a timeout only when `remaining_resources` is non-empty.

- [ ] **Step 8: Verify and commit.**

```bash
PYTHONPATH=src pytest -q \
  tests/runtime/test_run_controller.py \
  tests/api/test_replay_compatibility.py \
  tests/integration/test_uuv_only_8h_replay_acceptance.py \
  tests/integration/test_main_shutdown.py
ruff check src/underwater_tracking/runtime/run_controller.py \
  src/underwater_tracking/api/frame_logger.py src/underwater_tracking/api/live.py
mypy src/underwater_tracking/runtime/run_controller.py \
  src/underwater_tracking/api/frame_logger.py
git add src/underwater_tracking/config/models.py \
  src/underwater_tracking/runtime/models.py \
  src/underwater_tracking/runtime/run_controller.py \
  src/underwater_tracking/api/frame_logger.py \
  src/underwater_tracking/api/live.py src/underwater_tracking/api/replay.py \
  src/underwater_tracking/cli.py main.py \
  tests/runtime/test_run_controller.py \
  tests/api/test_replay_compatibility.py \
  tests/integration/test_uuv_only_8h_replay_acceptance.py \
  tests/integration/test_main_shutdown.py
git commit -m "fix: bound live run duration and frame persistence"
```

---

### Task 6: Publish truthful planning, brain, and run-phase UI state

**Files:**
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/persistence/ledger.py`
- Modify: `src/underwater_tracking/api/live.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/ui/src/types/frames.ts`
- Modify: `src/underwater_tracking/ui/src/components/RightSidebar.tsx`
- Modify: `src/underwater_tracking/ui/src/components/RightSidebar.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/BottomDrawer.tsx`
- Modify: `src/underwater_tracking/ui/src/components/BottomDrawer.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/assistant/SmartAssistantPanel.tsx`
- Modify: `src/underwater_tracking/ui/e2e/uuv-live-timeline.spec.ts`

**Interfaces:**

```python
class PlanningHealthView(StrictModel):
    status: Literal[
        "idle", "queued", "running", "committed", "invalidated",
        "rejected", "failed", "awaiting_retry",
    ]
    epoch_id: str | None = None
    base_physics_revision: int | None = None
    current_physics_revision: int | None = None
    base_sim_time_s: int | None = None
    current_sim_time_s: int | None = None
    data_age_s: int | None = None
    node: str | None = None
    attempt: int = 0
    deadline_utc_ms: int | None = None
    last_result_status: str | None = None
    last_error: str | None = None


class OperationalFrame(StrictModel):
    run_phase: RunPhase
    planning: PlanningHealthView
    # existing fields remain
```

- [ ] **Step 1: Add a backend regression test for the misleading master status.**

Persist a successful `regional_strategy` LLM call followed by a rejected epoch. Build a frame and assert:

```python
master = next(brain for brain in frame.brains if brain.role == "master")
assert master.status == "failed"
assert master.operation == "planning_epoch"
assert frame.plan_version == 0
assert frame.planning.status == "rejected"
```

The sub-call remains visible in the LLM call ledger but does not control the role status.

- [ ] **Step 2: Run focused backend/UI tests and verify failure.**

```bash
PYTHONPATH=src pytest -q tests/api/test_live_publisher.py tests/api/test_frame_pipeline.py
npm --prefix src/underwater_tracking/ui test -- --run \
  src/components/RightSidebar.test.tsx \
  src/components/BottomDrawer.test.tsx
```

- [ ] **Step 3: Derive master status from epoch state.**

Keep `latest_role_activity()` for low-level call history, but in the publisher overlay the master record from PlanningHealthView for running and terminal epoch states. Only a committed epoch may produce master `succeeded`; rejected/failed map to `failed`, invalidated maps to `degraded`, awaiting retry maps to `degraded` with the terminal reason.

- [ ] **Step 4: Add explicit run-phase and planning panels.**

In `RightSidebar`, show bootstrap planning, awaiting retry, running, completed, stopped and failed states. When plan version is zero, replace the ambiguous empty counters with a concise status line and a retry command when allowed. Do not display private target waypoint, private depth, raw rationale, prompt or completion.

- [ ] **Step 5: Keep LLM thinking and Memory Steam independent.**

LLM thinking displays epoch trigger, safe summary, base/current revision and terminal result. Memory Steam continues cursor-based reads and may show the source ID of a private/audit event without displaying its private payload. Evidence queries remain available at plan v0; plan adjustment apply uses bootstrap retry semantics when no base plan exists.

- [ ] **Step 6: Add desktop and mobile Playwright coverage.**

Cover:

```text
bootstrap planning -> rejected -> retry -> committed -> running -> completed
```

Assert the map contains no waterborne UUV before deployment, no target marker after public prior expiry without an estimate, no horizontal overflow at 390x844, and no console/page/request errors. Verify Smart Assistant modes and Memory Steam remain usable in rejected and completed phases.

- [ ] **Step 7: Verify and commit.**

```bash
PYTHONPATH=src pytest -q tests/api/test_live_publisher.py tests/api/test_frame_pipeline.py
npm --prefix src/underwater_tracking/ui test -- --run
npm --prefix src/underwater_tracking/ui run build
npm --prefix src/underwater_tracking/ui run test:e2e -- uuv-live-timeline.spec.ts
git add src/underwater_tracking/domain/ui_models.py \
  src/underwater_tracking/persistence/ledger.py \
  src/underwater_tracking/api/live.py \
  src/underwater_tracking/api/frame_builder.py \
  src/underwater_tracking/ui/src/types/frames.ts \
  src/underwater_tracking/ui/src/components/RightSidebar.tsx \
  src/underwater_tracking/ui/src/components/RightSidebar.test.tsx \
  src/underwater_tracking/ui/src/components/BottomDrawer.tsx \
  src/underwater_tracking/ui/src/components/BottomDrawer.test.tsx \
  src/underwater_tracking/ui/src/components/assistant/SmartAssistantPanel.tsx \
  src/underwater_tracking/ui/e2e/uuv-live-timeline.spec.ts
git commit -m "fix: expose truthful live planning state"
```

---

### Task 7: Extend submarine motion with bounded depth dynamics

**Files:**
- Modify: `src/underwater_tracking/config/platform_core.py`
- Modify: `src/underwater_tracking/config/models.py`
- Modify: `configs/environment_uuv_only.yaml`
- Modify: `configs/platforms.yaml`
- Modify: `src/underwater_tracking/domain/platforms.py`
- Modify: `src/underwater_tracking/domain/adversary_models.py`
- Create: `src/underwater_tracking/simulation/submarine_kinematics.py`
- Modify: `src/underwater_tracking/simulation/target.py`
- Modify: `src/underwater_tracking/simulation/target_guidance.py`
- Modify: `src/underwater_tracking/simulation/adversary_sensing.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/ui/src/types/frames.ts`
- Modify: `tests/config/test_platform_core_loader.py`
- Create: `tests/simulation/test_submarine_kinematics.py`
- Modify: `tests/simulation/test_target_guidance.py`
- Modify: `tests/simulation/test_target_adversary_motion.py`
- Modify: `tests/simulation/test_adversary_local_sensing.py`

**Interfaces:**

```python
class SubmarineMotionLimits(StrictModel):
    min_speed_mps: float
    max_speed_mps: float
    max_acceleration_mps2: float
    max_deceleration_mps2: float
    max_turn_rate_rad_s: float
    min_depth_m: float
    max_depth_m: float
    max_vertical_speed_mps: float
    max_vertical_acceleration_mps2: float
    max_pitch_rad: float


class SubmarineMotionState(StrictModel):
    position_xy: tuple[float, float]
    depth_m: float
    heading_rad: float
    speed_mps: float
    vertical_speed_mps: float


class SubmarineMotionCommand(StrictModel):
    desired_heading_rad: float
    desired_speed_mps: float
    desired_depth_m: float


DepthIntent = Literal["maintain_depth", "go_deeper", "go_shallower"]
```

- [ ] **Step 1: Add failing configuration tests.**

Require a default target depth within the configured range. Reject negative depth, `min_depth_m >= max_depth_m`, initial depth outside bounds, non-positive vertical limits, and a route whose configured depth profile violates maximum pitch.

- [ ] **Step 2: Add kinematic invariant tests.**

For every integration step assert:

```python
assert limits.min_depth_m <= next.depth_m <= limits.max_depth_m
assert abs(next.vertical_speed_mps) <= limits.max_vertical_speed_mps
assert abs(next.vertical_speed_mps - state.vertical_speed_mps) <= (
    limits.max_vertical_acceleration_mps2 * dt_s + 1e-9
)
assert abs(wrap_angle(next.heading_rad - state.heading_rad)) <= (
    limits.max_turn_rate_rad_s * dt_s + 1e-9
)
assert abs(math.atan2(next.vertical_speed_mps, next.speed_mps)) <= limits.max_pitch_rad + 1e-9
```

Use property-style parameterization over shallow/deep boundaries, acceleration/deceleration, hard turns and a 5-second step split into 0.5-second substeps.

- [ ] **Step 3: Implement the three-degree-of-freedom integrator.**

Reuse horizontal `advance_motion()` for x/y/heading/speed. Integrate vertical speed with acceleration limits, clamp desired depth, limit pitch by reducing vertical speed, and reject a segment that crosses depth or horizontal navigation boundaries. The function must be deterministic and contain no LLM calls.

- [ ] **Step 4: Add high-level depth intent to adversary decisions.**

`AdversaryIntentDecision` receives `depth_intent` with default `maintain_depth`. The LLM prompt offers only the three enum values. `resolve_target_guidance()` converts it to a safe desired depth step bounded by the operating envelope; the LLM never returns raw desired depth.

- [ ] **Step 5: Update target state and guidance.**

Store depth and vertical speed in TargetEntity, include them in private belief/input, and apply the 3-DOF integrator. Public target identity remains truth-safe. Navigation failure produces safe hold with zero vertical speed and a bounded operator audit event.

- [ ] **Step 6: Use 3-D range for target-local sensing.**

Surface carriers use depth 0. UUVs use explicit platform depth, initially configured to their operating depth when deployed. Compute local range with `sqrt(dx*dx + dy*dy + dz*dz)`. Do not expose true target/UUV depth to blue planning; publish estimated depth only when a sensor observation supports it.

- [ ] **Step 7: Preserve two-dimensional map compatibility.**

Keep `position_xy` in map contracts. Add optional `estimated_depth_m` and `depth_uncertainty_m` to target estimate views and optional waterborne UUV depth to tooltips. Legacy frames without depth remain readable as unknown, never as zero-depth truth.

- [ ] **Step 8: Verify and commit.**

```bash
PYTHONPATH=src pytest -q \
  tests/config/test_platform_core_loader.py \
  tests/simulation/test_submarine_kinematics.py \
  tests/simulation/test_target_guidance.py \
  tests/simulation/test_target_adversary_motion.py \
  tests/simulation/test_adversary_local_sensing.py
ruff check src/underwater_tracking/simulation/submarine_kinematics.py \
  src/underwater_tracking/simulation/target.py \
  src/underwater_tracking/simulation/target_guidance.py
mypy src/underwater_tracking/simulation/submarine_kinematics.py \
  src/underwater_tracking/simulation/target.py
git add src/underwater_tracking/config/platform_core.py \
  src/underwater_tracking/config/models.py \
  configs/environment_uuv_only.yaml configs/platforms.yaml \
  src/underwater_tracking/domain/platforms.py \
  src/underwater_tracking/domain/adversary_models.py \
  src/underwater_tracking/simulation/submarine_kinematics.py \
  src/underwater_tracking/simulation/target.py \
  src/underwater_tracking/simulation/target_guidance.py \
  src/underwater_tracking/simulation/adversary_sensing.py \
  src/underwater_tracking/simulation/engine.py \
  src/underwater_tracking/domain/ui_models.py \
  src/underwater_tracking/ui/src/types/frames.ts \
  tests/config/test_platform_core_loader.py \
  tests/simulation/test_submarine_kinematics.py \
  tests/simulation/test_target_guidance.py \
  tests/simulation/test_target_adversary_motion.py \
  tests/simulation/test_adversary_local_sensing.py
git commit -m "feat: constrain submarine depth dynamics"
```

---

### Task 8: Verify Smart Assistant and MemoryWorker across failed and completed runs

**Files:**
- Modify: `src/underwater_tracking/conversation/service.py`
- Modify: `src/underwater_tracking/api/conversation.py`
- Modify: `src/underwater_tracking/memory/worker.py`
- Modify: `src/underwater_tracking/ui/src/components/assistant/SmartAssistantPanel.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/assistant/MemoryWindow.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/MemorySteam.test.tsx`
- Create: `tests/integration/test_assistant_run_phase_lifecycle.py`
- Modify: `tests/integration/test_memory_full_lifecycle.py`

**Interfaces:**
- Consumes: `RunPhase`, `PlanningHealthView`, event audiences, preview/apply version guards, memory stream cursor.
- Produces: truthful assistant behavior at plan v0, rejected bootstrap, committed plan and completed run.

- [ ] **Step 1: Add a plan-v0 assistant lifecycle test.**

Submit an evidence question during `AWAITING_RETRY`; assert evidence retrieval works. Submit a plan-adjustment message; assert it returns an initialization preview and does not mutate MissionController. Confirm the preview to create a new initialization epoch; only a committed epoch may advance plan version.

- [ ] **Step 2: Add apply conflict and terminal failure tests.**

Use a stale `expected_epoch_id` or plan version and assert HTTP 409. Force the confirmed epoch to reject and assert the assistant turn contains the public terminal reason, no active plan changed, and the retry action is available.

- [ ] **Step 3: Add MemoryWorker audience tests.**

Feed a target-private audit event. Assert the memory stream records source discovery and extraction metadata, but the stored memory text and evidence answer do not expose private waypoint, exact truth depth or raw rationale. Blue-public events remain fully traceable to their observation IDs.

- [ ] **Step 4: Add completed-run behavior.**

At `COMPLETED`, evidence queries and memory deletion/version expansion remain available. Plan apply and sensor mutation endpoints reject with a stable `run_completed` conflict instead of silently changing the final replay state.

- [ ] **Step 5: Verify and commit.**

```bash
PYTHONPATH=src pytest -q \
  tests/integration/test_assistant_run_phase_lifecycle.py \
  tests/integration/test_memory_full_lifecycle.py
npm --prefix src/underwater_tracking/ui test -- --run \
  src/components/assistant/SmartAssistantPanel.test.tsx \
  src/components/assistant/MemoryWindow.test.tsx \
  src/components/MemorySteam.test.tsx
git add src/underwater_tracking/conversation/service.py \
  src/underwater_tracking/api/conversation.py \
  src/underwater_tracking/memory/worker.py \
  src/underwater_tracking/ui/src/components/assistant/SmartAssistantPanel.test.tsx \
  src/underwater_tracking/ui/src/components/assistant/MemoryWindow.test.tsx \
  src/underwater_tracking/ui/src/components/MemorySteam.test.tsx \
  tests/integration/test_assistant_run_phase_lifecycle.py \
  tests/integration/test_memory_full_lifecycle.py
git commit -m "test: cover assistant and memory run phases"
```

---

### Task 9: Add deterministic and real-provider release gates for `main.py`

**Files:**
- Create: `src/underwater_tracking/verification/live_demo.py`
- Create: `scripts/verify_live_demo.py`
- Create: `tests/integration/test_live_demo_release_contract.py`
- Create: `tests/integration/test_live_demo_real_provider.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**

```python
class LiveDemoAcceptanceResult(StrictModel):
    first_plan_wall_s: float | None
    final_sim_time_s: int
    final_plan_version: int
    observed_stage_ids: frozenset[str]
    adversary_llm_decision_count: int
    memory_event_count: int
    api_p95_ms: float
    output_bytes: int
    shutdown_s: float
    violations: tuple[str, ...]


def verify_live_demo(
    *,
    base_url: str,
    output_dir: Path,
    require_real_provider: bool,
) -> LiveDemoAcceptanceResult: ...
```

- [ ] **Step 1: Add a deterministic release-contract test.**

Run the real RunController/API/engine with a recorded structured provider that reproduces the audited cross-batch predecessor relation and delayed responses. Assert bootstrap remains at sim 0, first plan commits, all mission stages occur, duration stops, output stays under the limit, and shutdown reports no remaining resources.

- [ ] **Step 2: Add a real-provider pytest marker.**

`test_live_demo_real_provider.py` requires the configured role credentials and endpoint. If absent, use `pytest.skip` with an explicit reason. It launches `main.py` on free ports and calls the same verifier with `require_real_provider=True`; it must not inject `FixedSeedUUVLLM` or recorded responses.

- [ ] **Step 3: Implement the verifier.**

Poll `/api/health`, `/api/operational/snapshot`, WebSocket frames and SQLite ledgers without reading private engine state. Record:

```text
bootstrap onboard inventory
first committed plan and wall time
transport/deploy/ACTIVE_SCAN/PASSIVE_TRACK/handoff/resource/recovery/return stages
adversary role LLM calls
Memory Steam cursor growth
API latency samples
duration boundary
output bytes
shutdown duration and remaining resources
```

Fail on plan v0 at deadline, private event evidence in blue decisions, any USV, UUV exposure before deployment, sim time beyond duration tolerance, missing stages, output above 250 MiB, p95 above 200 ms, or shutdown above 10 seconds.

- [ ] **Step 4: Add the operator command.**

```bash
PYTHONPATH=src python scripts/verify_live_demo.py \
  --main main.py \
  --timeout-s 1200 \
  --require-real-provider
```

The command prints one JSON result and exits nonzero on violations. It redacts endpoints, headers, API keys, prompts and raw model responses.

- [ ] **Step 5: Document local and CI gates.**

Add pytest markers `live_provider` and `release_gate`. Normal CI runs deterministic gates; a credentialed protected job runs the real-provider gate. A skipped real-provider job is not a release approval.

- [ ] **Step 6: Run the deterministic gate.**

```bash
PYTHONPATH=src pytest -q \
  tests/integration/test_live_demo_release_contract.py \
  -m "not live_provider"
```

Expected: PASS with zero violations.

- [ ] **Step 7: Run the real-provider gate when credentials are available.**

```bash
PYTHONPATH=src pytest -q \
  tests/integration/test_live_demo_real_provider.py \
  -m live_provider -rs
```

Expected for release approval: PASS, not SKIPPED.

- [ ] **Step 8: Commit.**

```bash
git add src/underwater_tracking/verification/live_demo.py \
  scripts/verify_live_demo.py \
  tests/integration/test_live_demo_release_contract.py \
  tests/integration/test_live_demo_real_provider.py \
  pyproject.toml README.md
git commit -m "test: gate the real live demonstration"
```

---

### Task 10: Run the full release verification and record evidence

**Files:**
- Create: `docs/verification/live-demo-correctness-release.md`
- Modify only if a verification failure requires a scoped fix: files owned by Tasks 1-9 and their matching tests.

**Interfaces:**
- Consumes: every implementation and acceptance interface from Tasks 1-9.
- Produces: one reproducible release evidence record with commands, commit, configuration digest and results.

- [ ] **Step 1: Run Python unit and integration tests.**

```bash
PYTHONPATH=src pytest -q
```

Expected: all non-live-provider tests pass with no unexpected skips.

- [ ] **Step 2: Run static checks.**

```bash
ruff check src tests scripts
mypy src/underwater_tracking
```

Expected: zero errors.

- [ ] **Step 3: Run frontend verification.**

```bash
npm --prefix src/underwater_tracking/ui test -- --run
npm --prefix src/underwater_tracking/ui run build
npm --prefix src/underwater_tracking/ui run test:e2e
```

Expected: all Vitest and Playwright tests pass, TypeScript and Vite build succeed.

- [ ] **Step 4: Run deterministic 8-hour and release gates.**

```bash
PYTHONPATH=src pytest -q \
  tests/integration/test_uuv_only_8h_replay_acceptance.py \
  tests/integration/test_live_demo_release_contract.py
```

Expected: complete stage sequence, bounded sim time, output below 250 MiB, no truth-boundary violation.

- [ ] **Step 5: Run the real-provider gate.**

```bash
PYTHONPATH=src python scripts/verify_live_demo.py \
  --main main.py \
  --timeout-s 1200 \
  --require-real-provider
```

Expected: exit 0, first plan within 180 seconds, every required stage observed, API p95 below 200 ms, shutdown below 10 seconds.

- [ ] **Step 6: Write the evidence document.**

Record the exact Git commit, config file SHA-256 digests, provider model IDs without credentials, command outputs, first-plan wall time, stage timestamps, plan versions, epoch terminal counts, target motion maxima, API p95, output bytes and shutdown report. Do not paste prompts, completions, private target truth or secrets.

- [ ] **Step 7: Verify no services or generated outputs remain unintentionally active.**

```bash
pgrep -af 'python main.py|vite' || true
git status --short
```

Only explicitly retained verification artifacts may remain. Stop owned services; do not terminate unrelated user processes.

- [ ] **Step 8: Commit release evidence.**

```bash
git add docs/verification/live-demo-correctness-release.md
git commit -m "docs: record live demo release evidence"
```

---

### Task 11: Run `main.py` through the full battle and audit every entity trajectory

**Files:**
- Create: `src/underwater_tracking/verification/physics_invariants.py`
- Create: `scripts/monitor_main_battle.py`
- Create: `tests/verification/test_physics_invariants.py`
- Create: `docs/verification/main-live-battle-acceptance.json`
- Create: `docs/verification/main-live-battle-acceptance.md`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/runtime/run_controller.py`
- Modify: `src/underwater_tracking/api/app.py`
- Modify: `src/underwater_tracking/ui/playwright.live.config.ts`

**Interfaces:**

```python
class EntityMotionLimits(StrictModel):
    max_speed_mps: float
    max_acceleration_mps2: float
    max_deceleration_mps2: float
    max_turn_rate_rad_s: float
    min_depth_m: float | None = None
    max_depth_m: float | None = None
    max_vertical_speed_mps: float | None = None
    max_vertical_acceleration_mps2: float | None = None
    max_pitch_rad: float | None = None


class EntityMotionAudit(StrictModel):
    entity_id: str
    entity_kind: Literal["carrier", "mother_ship", "uuv", "submarine"]
    observed_steps: int
    max_speed_mps: float
    max_acceleration_mps2: float
    max_deceleration_mps2: float
    max_turn_rate_rad_s: float
    min_depth_m: float | None = None
    max_depth_m: float | None = None
    max_vertical_speed_mps: float | None = None
    max_vertical_acceleration_mps2: float | None = None
    max_pitch_rad: float | None = None
    teleport_count: int = 0
    boundary_violation_count: int = 0
    limit_violation_count: int = 0
    violating_frame_ids: tuple[int, ...] = ()


class BattleEvidenceChain(StrictModel):
    target_detection_event_id: str
    adversary_decision_id: str
    adversary_source_event_ids: tuple[str, ...]
    resulting_public_observation_ids: tuple[str, ...]
    blue_estimate_ids: tuple[str, ...]
    blue_epoch_id: str | None
    blue_plan_version: int | None


class FullBattleAcceptance(StrictModel):
    completed: bool
    final_sim_time_s: int
    final_plan_version: int
    required_stage_ids: frozenset[str]
    battle_evidence_chains: tuple[BattleEvidenceChain, ...]
    motion_audits: tuple[EntityMotionAudit, ...]
    browser_error_count: int
    failed_request_count: int
    output_bytes: int
    shutdown_s: float
    violations: tuple[str, ...]
```

- [ ] **Step 1: Write failing unit tests for the trajectory auditor.**

Cover every entity kind and every invariant. Include a legal curved route and fixtures that independently trigger speed, acceleration, deceleration, turn-rate, horizontal-boundary, depth, vertical-speed, vertical-acceleration, pitch and unexplained-teleport violations.

```python
def test_unexplained_uuv_position_jump_is_a_teleport() -> None:
    audit = PhysicsInvariantMonitor(limits_by_entity())
    audit.observe(frame(0, uuv(onboard=True, x=0)), events=())
    audit.observe(frame(5, uuv(deployed=True, x=5000)), events=())
    assert audit.result("uuv_00").teleport_count == 1


def test_deployment_event_explains_only_a_continuous_launch_transition() -> None:
    audit = PhysicsInvariantMonitor(limits_by_entity())
    audit.observe(frame(0, uuv(onboard=True, x=0)), events=())
    audit.observe(
        frame(5, uuv(deployed=True, x=20)),
        events=(deployment_event("uuv_00", launch_xy=(0, 0)),),
    )
    assert audit.result("uuv_00").teleport_count == 0
```

- [ ] **Step 2: Run the auditor tests and verify failure.**

```bash
PYTHONPATH=src pytest -q tests/verification/test_physics_invariants.py
```

Expected: FAIL because `PhysicsInvariantMonitor` does not exist.

- [ ] **Step 3: Implement continuous per-step physics auditing.**

`PhysicsInvariantMonitor.observe()` receives an internal truth-state projection before public-frame redaction. It computes finite differences using the actual step duration, unwraps heading across ±π, validates configured navigation bounds, and distinguishes physical motion from state transitions. Deployment/recovery events may explain a lifecycle change but never excuse displacement beyond the launch/rendezvous tolerance.

For mother ships, validate formation error only while their MissionController state is `fleet_standby` or `return_completed`; during transport/recovery, validate the committed carrier route instead. For onboard UUVs, require their internal position to remain colocated with their owner mother ship and exclude them from waterborne mileage.

The monitor stores aggregates and violating frame IDs only. Raw target coordinates and truth depth remain in process memory and are not returned by public API. Add `/api/verification/physics` only when `--verification-audit` is enabled; otherwise return 404.

- [ ] **Step 4: Implement the battle monitor orchestration script.**

`scripts/monitor_main_battle.py` must:

1. resolve free API/UI ports;
2. start `python main.py --verification-audit` with the default scenario and real configured role providers;
3. wait through bootstrap planning without advancing sim time;
4. poll health and snapshot, subscribe to WebSocket, and incrementally read memory/events/decisions;
5. launch Playwright at 1440x900 and 390x844, collect stage screenshots and browser errors;
6. monitor until run phase COMPLETED at 28800 seconds or the 1200-second wall deadline;
7. request aggregate physics audit results;
8. send SIGINT, verify shutdown within 10 seconds, and confirm owned ports/processes are gone;
9. write one redacted JSON result and return nonzero on any violation.

It must never submit expert directives, fabricate observations, force sensor contacts, alter positions, or call internal mutation endpoints.

- [ ] **Step 5: Require the complete blue tracking chain.**

Assert ordered durable evidence for:

```text
committed initial plan
< carrier dispatch completed
< UUV deployed
< ACTIVE_SCAN
< public target estimate with source observations
< PASSIVE_TRACK
< HANDOFF_PENDING
< handoff completed
< resource threshold crossed
< recovery requested
< UUV recovered
< carrier returned to fleet
```

At every point, assert UUV IDs, carrier owners, region IDs, plan versions and source IDs match across MissionController, events, operational frames and SQLite.

- [ ] **Step 6: Require a real counter-tracking evidence chain.**

Find at least one target-side local detection or audible active-sonar event followed by a real `adversary_mission_decision` LLM role call, its private/audit `target_mission_decision` event, and a bounded guidance change. Then require a later blue sensor observation/estimate that reflects the resulting maneuver and either updates the existing tracking solution or triggers a blue planning epoch.

Reject a chain when the blue epoch cites `target_mission_decision`, private waypoint, exact target truth or private depth. The adversary decision must cite only target-local event IDs, and the blue response must cite only public observation/estimate/event IDs.

- [ ] **Step 7: Require zero motion violations for every configured entity.**

The final audit must contain exactly:

```text
carrier_01
carrier_02, carrier_03, carrier_04
uuv_00 .. uuv_11
target_00
```

For each entity assert `observed_steps > 0`, `teleport_count == 0`, `boundary_violation_count == 0`, and `limit_violation_count == 0`. Also compare every observed maximum with its configured limit plus a numeric tolerance of `1e-6`.

- [ ] **Step 8: Require truthful frontend behavior throughout the run.**

Capture bootstrap, initial-plan committed, first deployment, active scan, passive track, handoff, adversary maneuver response, recovery and completed states. Assert no onboard UUV is drawn as waterborne; no target truth is drawn without a valid prior/estimate; run phase, planning epoch, plan version, both brain states, mission timeline, LLM thinking and Memory Steam match backend ledgers. Assert zero browser console errors, page errors and failed API/WebSocket requests.

- [ ] **Step 9: Execute the full `main.py` acceptance run.**

```bash
PYTHONPATH=src python scripts/monitor_main_battle.py \
  --main main.py \
  --scenario configs/scenario/uuv_only_single_target.yaml \
  --wall-timeout-s 1200 \
  --expected-duration-s 28800 \
  --require-real-provider \
  --output-report docs/verification/main-live-battle-acceptance.json
```

Expected: exit 0; phase COMPLETED; all required blue stages present; at least one valid counter-tracking evidence chain; 17 entity audits with zero violations; zero browser/request errors; output below 250 MiB; shutdown below 10 seconds.

- [ ] **Step 10: Write the human-readable acceptance record.**

In `docs/verification/main-live-battle-acceptance.md`, record:

- Git commit and config SHA-256 digests;
- wall-clock start/end and first-plan latency;
- every required stage timestamp and plan version;
- the redacted source-ID chain for tracking and counter-tracking;
- per-entity observed maxima versus configured limits;
- API p95, browser errors, output bytes and shutdown duration;
- links to retained desktop/mobile stage screenshots;
- final PASS/FAIL and every violation.

Do not include API keys, raw prompts/completions, private target coordinates or private truth depth.

- [ ] **Step 11: Verify cleanup and commit only passing evidence.**

```bash
pgrep -af 'python main.py|vite' || true
git status --short
git add src/underwater_tracking/verification/physics_invariants.py \
  scripts/monitor_main_battle.py \
  tests/verification/test_physics_invariants.py \
  src/underwater_tracking/simulation/engine.py \
  src/underwater_tracking/runtime/run_controller.py \
  src/underwater_tracking/api/app.py \
  src/underwater_tracking/ui/playwright.live.config.ts \
  docs/verification/main-live-battle-acceptance.json \
  docs/verification/main-live-battle-acceptance.md
git commit -m "test: verify full live adversarial battle"
```

If the real provider is unavailable, the task remains blocked and no PASS evidence commit is allowed. If any physical or behavioral violation occurs, retain the redacted failure report outside the release commit, fix the owning task, rerun Task 11 from the beginning, and accept only a complete passing run.
