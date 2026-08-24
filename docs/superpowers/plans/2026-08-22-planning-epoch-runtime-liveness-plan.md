# Planning Epoch and Runtime Liveness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace revision-equality planning with persistent planning epochs and semantic revalidation so slow real LLM cycles can commit without blocking physics, API traffic, or shutdown.

**Architecture:** Add an immutable `PlanningEpoch` and durable repository, route UUV-only central commits through a semantic revalidation port, and let `_AgentLoop` apply completed results regardless of harmless physics revision drift. Keep a latest-value situation mailbox plus a deduplicated critical-event mailbox. Add cooperative physics yielding and one bounded, idempotent shutdown owner so API and Vite stay responsive while providers are slow.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, SQLite WAL, LangGraph, threading, FastAPI/Uvicorn, pytest, Ruff, mypy.

## Global Constraints

- `MissionController` remains the only owner of committed UUV mission state.
- A physics revision change alone never invalidates a planning result.
- One scenario runs at most one central planning epoch at a time.
- Every committed UUV-only plan references a persisted semantic revalidation report.
- LLM work never runs on the physics worker or FastAPI event-loop thread.
- Ordinary observations update the latest situation but do not create one planning epoch per frame.
- Failed first planning leaves all UUVs onboard; failed replanning retains only a still-valid active plan.
- Shutdown is idempotent and bounded; it closes the Vite process group, provider clients, workers, publishers, and repositories exactly once.
- No task in this plan changes target mission logic, initialization truth, memory semantics, or UI presentation beyond planning health fields.

---

### Task 1: Define durable planning epoch and commit-result contracts

**Files:**
- Create: `src/underwater_tracking/domain/planning_epoch_models.py`
- Create: `src/underwater_tracking/persistence/planning_epochs.py`
- Modify: `src/underwater_tracking/persistence/sqlite.py`
- Create: `tests/domain/test_planning_epoch_models.py`
- Create: `tests/agent/test_planning_epoch_repository.py`

**Interfaces:**

```python
class PlanningEpochStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMMITTED = "committed"
    INVALIDATED = "invalidated"
    REJECTED = "rejected"
    FAILED = "failed"

class PlanningEpoch(StrictModel):
    epoch_id: str
    scenario_id: str
    base_physics_revision: int
    base_sim_time_s: int
    observation_batch_id: str
    critical_event_ids: tuple[str, ...]
    public_target_prior_ids: tuple[str, ...]
    public_target_estimate_ids: tuple[str, ...]
    resource_manifest_hash: str
    active_plan_version: int
    expert_request_version: int | None = None
    status: PlanningEpochStatus = PlanningEpochStatus.QUEUED

class EpochCommitResult(StrictModel):
    epoch_id: str
    status: Literal["committed", "invalidated", "rejected", "failed"]
    plan_id: str | None = None
    plan_version: int | None = None
    validation_report_id: str | None = None
    executable_plan: ExecutableMissionPlan | None = None
    invalidated_reason: str | None = None
    failure_category: Literal["timeout", "provider", "schema", "configuration", "internal"] | None = None
    failure_message: str | None = None
    consumed_event_ids: tuple[str, ...] = ()

class PlanningEpochCapture(StrictModel):
    epoch: PlanningEpoch
    situation: SituationSnapshot
    mission: MissionSnapshot

class PlanningEpochRepository:
    def create(self, capture: PlanningEpochCapture) -> None: ...
    def get(self, epoch_id: str) -> PlanningEpoch: ...
    def get_capture(self, epoch_id: str) -> PlanningEpochCapture: ...
    def mark_running(self, epoch_id: str) -> None: ...
    def finish(self, result: EpochCommitResult) -> None: ...
    def latest(self, scenario_id: str) -> tuple[PlanningEpoch, EpochCommitResult | None] | None: ...
    def close(self) -> None: ...
```

- [ ] **Step 1: Write model tests for frozen fields, unique event IDs, terminal status consistency, and invalidated reasons.**

```python
def test_invalidated_epoch_requires_reason() -> None:
    with pytest.raises(ValueError, match="invalidated_reason"):
        EpochCommitResult(
            epoch_id="epoch:S1:1",
            status="invalidated",
            validation_report_id="validation:S1:1",
        )

def test_committed_epoch_requires_plan_identity() -> None:
    with pytest.raises(ValueError, match="plan_id"):
        EpochCommitResult(
            epoch_id="epoch:S1:1",
            status="committed",
            validation_report_id="validation:S1:1",
        )
```

- [ ] **Step 2: Write repository round-trip and idempotency tests.**

```python
def test_epoch_terminal_result_is_idempotent_but_not_replaceable(tmp_path: Path) -> None:
    repo = PlanningEpochRepository(tmp_path / "agent.db")
    epoch = make_epoch("epoch:S1:1")
    capture = make_capture(epoch)
    result = make_failed_result(epoch.epoch_id, error="provider timeout")
    repo.create(capture)
    repo.mark_running(epoch.epoch_id)
    repo.finish(result)
    repo.finish(result)
    with pytest.raises(ValueError, match="already finished"):
        repo.finish(make_failed_result(epoch.epoch_id, error="different failure"))
```

- [ ] **Step 3: Run the new tests and verify imports fail.**

```bash
PYTHONPATH=src python -m pytest tests/domain/test_planning_epoch_models.py tests/agent/test_planning_epoch_repository.py -q
```

Expected: collection fails because `planning_epoch_models` and `planning_epochs` do not exist.

- [ ] **Step 4: Add strict Pydantic models and validators.**

Use `ConfigDict(extra="forbid", frozen=True)`. Reject duplicate critical event, prior, and estimate IDs. Require `plan_id`, positive `plan_version`, `validation_report_id`, and `executable_plan` only for `committed`; require `invalidated_reason` and a report for `invalidated`; require a report for post-candidate `rejected`. A pre-candidate/provider `failed` result may omit the report but requires a bounded `failure_category` and sanitized `failure_message`. Reject a committed result when `executable_plan.revision != plan_version`.

- [ ] **Step 5: Add schema version 11 and durable tables.**

```sql
CREATE TABLE IF NOT EXISTS planning_epochs (
    epoch_id TEXT PRIMARY KEY,
    scenario_id TEXT NOT NULL,
    base_physics_revision INTEGER NOT NULL,
    base_sim_time_s INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS planning_epoch_inputs (
    epoch_id TEXT PRIMARY KEY REFERENCES planning_epochs(epoch_id),
    observation_batch_id TEXT NOT NULL,
    situation_payload TEXT NOT NULL,
    mission_payload TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS planning_event_retries (
    scenario_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    retry_not_before_utc_ms INTEGER,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (scenario_id, event_id)
);

CREATE TABLE IF NOT EXISTS planning_revalidation_reports (
    report_id TEXT PRIMARY KEY,
    epoch_id TEXT NOT NULL REFERENCES planning_epochs(epoch_id),
    valid INTEGER NOT NULL,
    current_physics_revision INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS planning_epoch_results (
    epoch_id TEXT PRIMARY KEY REFERENCES planning_epochs(epoch_id),
    status TEXT NOT NULL,
    plan_id TEXT,
    plan_version INTEGER,
    validation_report_id TEXT REFERENCES planning_revalidation_reports(report_id),
    payload TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
```

Use canonical `json_dumps`, `json.loads`, `BEGIN IMMEDIATE`, and exact-payload idempotency following `PlanRepository` conventions. `create()` stores epoch metadata and the complete immutable capture in one transaction. The result payload durably contains the committed executable plan so startup recovery can restore `MissionController` after an interruption.

- [ ] **Step 6: Run repository tests and static checks.**

```bash
PYTHONPATH=src python -m pytest tests/domain/test_planning_epoch_models.py tests/agent/test_planning_epoch_repository.py tests/agent/test_repositories.py -q
ruff check src/underwater_tracking/domain/planning_epoch_models.py src/underwater_tracking/persistence/planning_epochs.py tests/domain/test_planning_epoch_models.py tests/agent/test_planning_epoch_repository.py
mypy src/underwater_tracking/domain/planning_epoch_models.py src/underwater_tracking/persistence/planning_epochs.py
```

Expected: all commands pass.

- [ ] **Step 7: Commit the contract and repository.**

```bash
git add src/underwater_tracking/domain/planning_epoch_models.py src/underwater_tracking/persistence/planning_epochs.py src/underwater_tracking/persistence/sqlite.py tests/domain/test_planning_epoch_models.py tests/agent/test_planning_epoch_repository.py
git commit -m "feat: persist planning epoch lifecycle"
```

---

### Task 2: Add a latest-situation and critical-event epoch coordinator

**Files:**
- Create: `src/underwater_tracking/runtime/planning_epoch.py`
- Create: `tests/runtime/test_planning_epoch.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class EpochTrigger:
    event_id: str
    event_type: str
    sim_time_s: int
    priority: int

class PlanningEpochCoordinator:
    def observe(self, situation: SituationSnapshot) -> None: ...
    def request(self, triggers: tuple[EpochTrigger, ...]) -> None: ...
    def next_epoch(self, mission: MissionSnapshot) -> PlanningEpochCapture | None: ...
    def mark_running(self, epoch_id: str) -> None: ...
    def finish(self, result: EpochCommitResult) -> None: ...
    def latest_situation(self) -> SituationSnapshot | None: ...
    def health(self) -> PlanningEpochHealth: ...
    def close(self) -> None: ...
```

`PlanningEpochHealth` exposes `status`, `epoch_id`, `queued_event_count`, `started_at_ms`, `last_result_status`, `last_error`, `retry_attempt`, `retry_not_before_utc_ms`, and `dead_letter_event_ids`.

- [ ] **Step 1: Add failing tests proving ordinary revisions do not queue epochs and events are deduplicated.**

```python
def test_observe_only_replaces_latest_situation() -> None:
    coordinator = make_coordinator()
    coordinator.observe(situation(revision=1))
    coordinator.observe(situation(revision=200))
    assert coordinator.latest_situation().snapshot_revision == 200
    assert coordinator.next_epoch(empty_mission()) is None

def test_request_deduplicates_event_ids() -> None:
    coordinator = make_coordinator()
    trigger = EpochTrigger("event-1", "initialization", 30, 100)
    coordinator.request((trigger, trigger))
    capture = coordinator.next_epoch(empty_mission())
    assert capture is not None
    assert capture.epoch.critical_event_ids == ("event-1",)
```

- [ ] **Step 2: Add concurrency tests with one running epoch and a newer mailbox event.**

Assert `next_epoch()` returns `None` while one epoch runs; after `finish()`, the next epoch contains the mailbox event but not events consumed by the completed result.

Add retry tests: transient failures for the same trigger are eligible after 5, 15, and 45 seconds; after the third failed retry the trigger moves to dead-letter and no epoch is queued until a new event or explicit expert retry arrives. Schema/config errors skip retries and dead-letter immediately.

- [ ] **Step 3: Run the tests and confirm failure.**

```bash
PYTHONPATH=src python -m pytest tests/runtime/test_planning_epoch.py -q
```

- [ ] **Step 4: Implement the coordinator with one `RLock`, immutable snapshots, sorted event IDs, and no worker thread.**

The coordinator owns scheduling state only. `_AgentLoop` remains the owner of the worker. Hash the resource manifest from sorted `MissionSnapshot.uuv_resources` JSON; do not include volatile wall-clock fields. Capture active `target_search_priors` and sensor estimate IDs separately, and persist the complete capture before returning it. Persist retry eligibility as UTC epoch milliseconds from an injected UTC clock; after restart compare it with the new process's UTC clock. A monotonic clock may drive only the current process's sleep duration, computed as `max(0, retry_utc - now_utc)`, and is never persisted. Retry counters and dead-letter IDs are persisted so restart cannot reset the ceiling.

- [ ] **Step 5: Verify deterministic epoch IDs and mailbox behavior.**

```bash
PYTHONPATH=src python -m pytest tests/runtime/test_planning_epoch.py -q
ruff check src/underwater_tracking/runtime/planning_epoch.py tests/runtime/test_planning_epoch.py
mypy src/underwater_tracking/runtime/planning_epoch.py
```

- [ ] **Step 6: Commit.**

```bash
git add src/underwater_tracking/runtime/planning_epoch.py tests/runtime/test_planning_epoch.py
git commit -m "feat: coordinate planning epochs and event mailbox"
```

---

### Task 3: Implement semantic revalidation for UUV executable plans

**Files:**
- Create: `src/underwater_tracking/planning/mission_revalidation.py`
- Modify: `src/underwater_tracking/planning/mission_validation.py`
- Modify: `src/underwater_tracking/planning/carrier_tasks.py`
- Modify: `src/underwater_tracking/persistence/planning_epochs.py`
- Create: `tests/planning/test_mission_revalidation.py`
- Modify: `tests/planning/test_carrier_tasks.py`
- Modify: `tests/agent/test_planning_epoch_repository.py`

**Interfaces:**

```python
class RevalidationIssue(StrictModel):
    code: Literal[
        "target_missing", "region_missing", "carrier_missing", "uuv_missing", "owner_changed",
        "deployment_changed", "resource_unavailable", "estimate_outside_envelope",
        "prior_expired", "prior_changed", "active_plan_advanced",
        "expert_version_advanced", "trigger_recovered",
    ]
    entity_id: str
    message: str

class MissionRevalidationReport(StrictModel):
    report_id: str
    epoch_id: str
    current_physics_revision: int
    current_plan_version: int
    valid: bool
    issues: tuple[RevalidationIssue, ...]
    rebased_plan: ExecutableMissionPlan | None

def revalidate_executable_mission_plan(
    *,
    epoch: PlanningEpoch,
    candidate: ExecutableMissionPlan,
    current_situation: SituationSnapshot,
    current_mission: MissionSnapshot,
    current_expert_request_version: int | None,
    recovered_event_ids: frozenset[str],
) -> MissionRevalidationReport: ...

class PlanningEpochRepository:
    def finish_with_revalidation(
        self, report: MissionRevalidationReport, result: EpochCommitResult
    ) -> None: ...
    def get_revalidation(self, report_id: str) -> MissionRevalidationReport: ...
```

- [ ] **Step 1: Add failing tests for harmless revision drift.**

```python
def test_revision_drift_rebases_eta_without_changing_strategy() -> None:
    planned = candidate(entry_s=300)
    report = revalidate_executable_mission_plan(
        epoch=epoch(base_revision=1, active_plan_version=0),
        candidate=planned,
        current_situation=situation(revision=20, sim_time_s=100),
        current_mission=mission(plan_revision=0),
        current_expert_request_version=None,
        recovered_event_ids=frozenset(),
    )
    assert report.valid is True
    assert report.rebased_plan is not None
    assert report.rebased_plan.region_assignments[0].region_id == planned.region_assignments[0].region_id
```

- [ ] **Step 2: Add invalidation tests for changed owner, unhealthy UUV, advanced plan/expert version, missing region, exhausted mileage, and recovered trigger.**

Each test asserts one stable issue code and `rebased_plan is None`. A target-estimate movement inside the candidate region envelope remains valid; movement outside invalidates.

Define the envelope as each selected region polygon buffered outward by 500 m. A captured prior is unchanged only when its ID, validity interval, covariance and canonical content hash match; it is expired when `current_situation.sim_time_s >= valid_until_s`. A recovery cancels a trigger only when event type, entity ID and resource episode match. An event is “covered” only when the candidate carries its ID in `trigger_event_ids` and its affected entity appears in the executable assignments. Resource feasibility uses configured route distance plus region path plus return distance, maximum mileage, current energy and reserve thresholds.

- [ ] **Step 3: Run the tests and verify failure.**

```bash
PYTHONPATH=src python -m pytest tests/planning/test_mission_revalidation.py tests/planning/test_carrier_tasks.py -q
```

- [ ] **Step 4: Implement validation as pure functions.**

Reuse `validate_executable_mission_plan()` for structural constraints. Recompute only carrier route origin, stop ETA, moving fleet rendezvous endpoint, and resource projections. Preserve candidate target IDs, region IDs, priorities, UUV owner IDs, task modes, and plan concept byte-for-byte.

Add report serialization and exact round-trip/idempotency tests. For invalidated/rejected outcomes, `PlanningEpochRepository.finish_with_revalidation()` writes the report and terminal result in one transaction. A valid report is returned to the commit port and is not terminally persisted until Task 4 atomically commits it with both plan representations. A different payload for the same `report_id` must fail.

- [ ] **Step 5: Run focused tests and static checks.**

```bash
PYTHONPATH=src python -m pytest tests/planning/test_mission_revalidation.py tests/planning/test_mission_optimizer.py tests/planning/test_carrier_tasks.py -q
ruff check src/underwater_tracking/planning/mission_revalidation.py tests/planning/test_mission_revalidation.py
mypy src/underwater_tracking/planning/mission_revalidation.py
```

- [ ] **Step 6: Commit.**

```bash
git add src/underwater_tracking/planning/mission_revalidation.py src/underwater_tracking/planning/mission_validation.py src/underwater_tracking/planning/carrier_tasks.py src/underwater_tracking/persistence/planning_epochs.py tests/planning/test_mission_revalidation.py tests/planning/test_carrier_tasks.py tests/agent/test_planning_epoch_repository.py
git commit -m "feat: revalidate slow mission plans against live state"
```

---

### Task 4: Route UUV-only graph commits through the epoch commit port

**Files:**
- Modify: `src/underwater_tracking/agent/state.py`
- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Modify: `src/underwater_tracking/agent/runtime.py`
- Modify: `src/underwater_tracking/agent/nodes/commit.py`
- Modify: `src/underwater_tracking/persistence/plans.py`
- Create: `src/underwater_tracking/persistence/uuv_plan_commits.py`
- Create: `src/underwater_tracking/runtime/scenario_transition.py`
- Modify: `src/underwater_tracking/runtime/mission_controller.py`
- Modify: `tests/agent/test_commit.py`
- Modify: `tests/agent/test_central_graph.py`
- Modify: `tests/agent/test_regional_plan_pipeline.py`
- Modify: `tests/runtime/test_mission_controller.py`
- Create: `tests/agent/test_uuv_plan_commit_repository.py`
- Create: `tests/runtime/test_scenario_transition.py`

**Interfaces:**

```python
class EpochCommitPort(Protocol):
    def commit(
        self,
        *,
        epoch: PlanningEpoch,
        audit_projection: TrackingPlan,
        executable_plan: ExecutableMissionPlan,
    ) -> EpochCommitResult: ...

class ScenarioTransitionCoordinator:
    @contextmanager
    def transition(self, kind: Literal["plan", "observation"]) -> Iterator[None]: ...

class CarrierDependencies:
    planning_epoch_provider: Callable[[], PlanningEpoch | None] | None = None
    epoch_commit_port: EpochCommitPort | None = None

def CarrierRuntime.tick(self, *, epoch: PlanningEpoch | None = None) -> dict[str, Any]: ...
```

- [ ] **Step 1: Replace the old stale-result expectation with semantic commit tests.**

For legacy `TrackingPlan`, preserve exact snapshot equality. For UUV-only `ExecutableMissionPlan`, assert revision drift reaches `EpochCommitPort`; a returned `invalidated` result becomes `commit_status="invalidated"` and does not expose a selected plan.

- [ ] **Step 2: Run the graph tests and verify failure.**

```bash
PYTHONPATH=src python -m pytest tests/agent/test_commit.py tests/agent/test_central_graph.py tests/agent/test_regional_plan_pipeline.py -q
```

- [ ] **Step 3: Extend graph state and commit status.**

Use:

```python
commit_status: Literal[
    "committed", "hold_current", "stale", "invalidated", "rejected", "failed"
] | None
planning_epoch: PlanningEpoch | None
epoch_commit_result: EpochCommitResult | None
```

The UUV-only `CommitPlanNode` must call `epoch_commit_port.commit()`. It must not call `snapshot_is_current()` and must not write a plan before semantic revalidation. The legacy branch keeps the existing equality guard.

- [ ] **Step 4: Commit durable and in-memory state under one rollback boundary.**

Add a UUV-specific commit repository on one SQLite connection plus controller checkpointing:

```python
class UUVPlanCommitRepository:
    def prepare(
        self,
        *,
        epoch: PlanningEpoch,
        report: MissionRevalidationReport,
        audit_projection: TrackingPlan,
        executable_plan: ExecutableMissionPlan,
        expected_active_plan_revision: int,
    ) -> PreparedUUVCommit: ...

class PreparedUUVCommit:
    def finish(self, result: EpochCommitResult) -> None: ...
    def rollback(self) -> None: ...

def MissionController.checkpoint(self) -> MissionControllerCheckpoint: ...
def MissionController.restore(self, checkpoint: MissionControllerCheckpoint) -> None: ...
def MissionController.apply_revalidated_plan(
    self,
    plan: ExecutableMissionPlan,
    *,
    expected_current_revision: int,
) -> bool: ...
```

`EpochCommitPort` first acquires the scenario's shared `ScenarioTransitionCoordinator`, then `prepare()` opens `BEGIN IMMEDIATE`, verifies both current plan revisions, and stages the revalidation report, executable payload and inactive audit projection without committing. The port checkpoints `MissionController`, calls `apply_revalidated_plan()`, then `finish()` marks the audit projection active, records the terminal result, and commits. Any SQL/apply exception calls `restore()` and rolls back. A process crash rolls back SQLite and loses the uncommitted in-memory state together. UUV-only execution creates no legacy `PlanCommand` rows. Keep existing `PlanRepository.commit()` semantics for legacy tests. Add fault-injection tests before apply, after apply, and during SQL commit, plus a two-thread plan/observation interleaving test proving one transition finishes before the other starts and rollback never restores across a committed transition. The lock order is transition coordinator before SQLite; no LLM call occurs while held.

- [ ] **Step 5: Verify graph and repository compatibility.**

```bash
PYTHONPATH=src python -m pytest tests/agent/test_commit.py tests/agent/test_central_graph.py tests/agent/test_regional_plan_pipeline.py tests/agent/test_repositories.py tests/agent/test_uuv_plan_commit_repository.py tests/runtime/test_mission_controller.py tests/runtime/test_scenario_transition.py -q
ruff check src/underwater_tracking/agent src/underwater_tracking/persistence/plans.py src/underwater_tracking/persistence/uuv_plan_commits.py src/underwater_tracking/runtime/scenario_transition.py tests/agent/test_commit.py
```

- [ ] **Step 6: Commit.**

```bash
git add src/underwater_tracking/agent/state.py src/underwater_tracking/agent/graphs/central.py src/underwater_tracking/agent/runtime.py src/underwater_tracking/agent/nodes/commit.py src/underwater_tracking/persistence/plans.py src/underwater_tracking/persistence/uuv_plan_commits.py src/underwater_tracking/runtime/scenario_transition.py src/underwater_tracking/runtime/mission_controller.py tests/agent/test_commit.py tests/agent/test_central_graph.py tests/agent/test_regional_plan_pipeline.py tests/runtime/test_mission_controller.py tests/agent/test_uuv_plan_commit_repository.py tests/runtime/test_scenario_transition.py
git commit -m "fix: commit uuv plans through semantic epoch validation"
```

---

### Task 5: Integrate epochs into `_AgentLoop` and remove revision-discard livelock

**Files:**
- Modify: `src/underwater_tracking/cli.py`
- Modify: `tests/agent/test_background_cycle.py`
- Modify: `tests/agent/test_runtime_master_slave_adversary.py`
- Modify: `tests/integration/test_agent_loop.py`

**Interfaces:**

Replace `_BackgroundCarrierCycle.situation` identity with:

```python
@dataclass(slots=True)
class _BackgroundCarrierCycle:
    epoch: PlanningEpoch
    situation: SituationSnapshot
    adversary_contexts: tuple[AdversaryEscapeInput, ...]
    slave_contexts: tuple[SlaveSonarContext, ...]
    result: dict[str, Any] | None = None
    error: BaseException | None = None
    done: bool = False
```

- [ ] **Step 1: Replace `test_stale_background_result_is_discarded_and_latest_cycle_started`.**

```python
def test_completed_epoch_is_applied_after_harmless_revision_drift() -> None:
    loop = loop_with_completed_epoch(base_revision=1, live_revision=200)
    loop.apply_background_cycle()
    assert loop.mission_controller.snapshot().plan_revision == 1
    assert loop._epoch_coordinator.health().last_result_status == "committed"
```

Add a second test where the commit port returns `invalidated`; assert no mission plan is applied and exactly one mailbox epoch starts with the still-unconsumed event.

- [ ] **Step 2: Add a delayed-LLM integration test.**

Use a blocking fake master LLM, advance 20 observation revisions while it waits, release it, and assert `plan_version == 1`, one epoch result is committed, and no second initialization epoch is created.

Add a restart test with a committed epoch result but a freshly constructed `MissionController`; startup must restore the exact persisted `executable_plan` before accepting another epoch.

- [ ] **Step 3: Run the focused tests and confirm the old discard behavior fails.**

```bash
PYTHONPATH=src python -m pytest tests/agent/test_background_cycle.py tests/integration/test_agent_loop.py -q
```

- [ ] **Step 4: Construct `PlanningEpochRepository` and `PlanningEpochCoordinator` in `_AgentLoop`.**

`on_situation()` always calls `coordinator.observe(situation)`. Initialization, key runtime events, and expert requests call `coordinator.request()`. `_start_background_cycle()` starts only when `next_epoch(mission_snapshot)` returns an epoch.

Before queuing initialization, reconcile the latest committed epoch result into an empty/older `MissionController`. Reject a hash/revision mismatch as startup corruption; do not silently synthesize a replacement plan.

- [ ] **Step 5: Remove the revision comparison at `cli.py:1387`.**

`apply_background_cycle()` consumes `EpochCommitResult`. Apply only a newly committed mission plan. For invalidated/rejected/failed results, preserve the current executable plan and return unconsumed critical events to the coordinator. The coordinator applies its retry delay/ceiling or dead-letter rule before another epoch can start; `_AgentLoop` must not immediately resubmit the same event. Do not apply local decisions whose evidence contract fails; leave their stage-three validator as the only future extension point.

- [ ] **Step 6: Keep synchronous finite runs on the same epoch path.**

`_run_synchronous_carrier_cycle()` creates and completes a `PlanningEpoch` inline rather than bypassing epoch persistence. This preserves deterministic unit tests while ensuring finite `agent-run` and live `serve` share commit semantics.

- [ ] **Step 7: Run the carrier/runtime integration suite.**

```bash
PYTHONPATH=src python -m pytest tests/agent/test_background_cycle.py tests/agent/test_runtime_master_slave_adversary.py tests/integration/test_agent_loop.py tests/integration/test_uuv_only_replan_loop.py -q
ruff check src/underwater_tracking/cli.py tests/agent/test_background_cycle.py tests/integration/test_agent_loop.py
```

- [ ] **Step 8: Commit.**

```bash
git add src/underwater_tracking/cli.py tests/agent/test_background_cycle.py tests/agent/test_runtime_master_slave_adversary.py tests/integration/test_agent_loop.py
git commit -m "fix: apply completed planning epochs after physics advances"
```

---

### Task 6: Preserve API responsiveness during unpaced physics and slow providers

**Files:**
- Modify: `src/underwater_tracking/runtime/run_controller.py`
- Modify: `src/underwater_tracking/api/app.py`
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `tests/runtime/test_run_controller.py`
- Modify: `tests/api/test_app_lifespan.py`
- Create: `tests/integration/test_slow_llm_api_responsiveness.py`

**Interfaces:**

```python
class PlanningHealthView(StrictModel):
    status: Literal["idle", "queued", "running", "committed", "invalidated", "degraded"]
    epoch_id: str | None = None
    base_physics_revision: int | None = None
    current_physics_revision: int | None = None
    queued_event_count: int = 0
    last_result_status: str | None = None
    last_error: str | None = None
```

- [ ] **Step 1: Add a worker-yield test for `speed=0`.**

Patch `bundle.stop.wait` and assert the unpaced loop calls `bundle.stop.wait(0.001)` after every physics step. This releases the GIL and lets API and shutdown threads run without imposing simulation-time pacing.

- [ ] **Step 2: Add a live responsiveness test.**

Start `RunController` with a fake LLM blocked on an event. While blocked, perform 20 `/api/health` requests and assert each returns within 500 ms under the test environment, `planning.status == "running"`, and `plan_version == 0`.

- [ ] **Step 3: Run tests and verify the responsiveness contract fails.**

```bash
PYTHONPATH=src python -m pytest tests/runtime/test_run_controller.py tests/api/test_app_lifespan.py tests/integration/test_slow_llm_api_responsiveness.py -q
```

- [ ] **Step 4: Yield cooperatively in the unpaced worker and expose epoch health.**

Do not sleep when positive `effective_speed` already waits against a wall deadline. `/api/health` reads immutable coordinator health and must not acquire the long-lived carrier or engine mutation lock.

- [ ] **Step 5: Verify.**

```bash
PYTHONPATH=src python -m pytest tests/runtime/test_run_controller.py tests/api/test_app_lifespan.py tests/integration/test_slow_llm_api_responsiveness.py -q
ruff check src/underwater_tracking/runtime/run_controller.py src/underwater_tracking/api/app.py tests/integration/test_slow_llm_api_responsiveness.py
```

- [ ] **Step 6: Commit.**

```bash
git add src/underwater_tracking/runtime/run_controller.py src/underwater_tracking/api/app.py src/underwater_tracking/domain/ui_models.py tests/runtime/test_run_controller.py tests/api/test_app_lifespan.py tests/integration/test_slow_llm_api_responsiveness.py
git commit -m "fix: keep health api responsive during live planning"
```

---

### Task 7: Make provider cancellation and process shutdown bounded

**Files:**
- Modify: `src/underwater_tracking/agent/llm.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `src/underwater_tracking/runtime/run_controller.py`
- Modify: `main.py`
- Modify: `tests/agent/test_background_cycle.py`
- Modify: `tests/runtime/test_run_controller.py`
- Modify: `tests/main/test_main.py`
- Create: `tests/integration/test_main_shutdown.py`

**Interfaces:**

```python
class CancelledLLMError(LLMError): ...

class HTTPStructuredLLM:
    def cancel(self) -> None: ...
    def close(self) -> None: ...

class RunController:
    def close(self, *, timeout_s: float = 10.0) -> bool: ...
    def abort(self) -> None: ...
```

- [ ] **Step 1: Add idempotent cancellation tests.**

Assert `cancel()` closes the active HTTP client once, future invocations raise `CancelledLLMError`, and repeated `close()`/`cancel()` calls do not fail.

- [ ] **Step 2: Add subprocess shutdown acceptance.**

Start `main.py` with a local blocking fake-provider fixture and temporary ports. Wait for `/api/health`, send one `SIGINT`, and assert within 10 seconds:

```python
assert process.wait(timeout=10) == 130
assert not port_is_open(api_port)
assert not port_is_open(ui_port)
```

- [ ] **Step 3: Run tests and confirm current shutdown fails the one-signal contract.**

```bash
PYTHONPATH=src python -m pytest tests/main/test_main.py tests/integration/test_main_shutdown.py tests/runtime/test_run_controller.py -q
```

- [ ] **Step 4: Implement one shutdown owner.**

On the first signal, set the run stop event, stop accepting epochs, call provider `cancel()`, stop MemoryWorker and publisher, join bounded workers, close repositories, then stop Vite in `main.py`'s `finally`. Remove the `os._exit()` branch after the integration test proves all owned non-daemon resources close normally.

- [ ] **Step 5: Ensure close failures are explicit.**

`RunController.close()` returns `False` on timeout. `_serve()` returns a nonzero status and logs the names of remaining resources. It must not silently report normal completion while a provider or worker remains owned.

- [ ] **Step 6: Run shutdown and API suites.**

```bash
PYTHONPATH=src python -m pytest tests/agent/test_background_cycle.py tests/runtime/test_run_controller.py tests/api/test_app_lifespan.py tests/main/test_main.py tests/integration/test_main_shutdown.py -q
ruff check main.py src/underwater_tracking/agent/llm.py src/underwater_tracking/cli.py src/underwater_tracking/runtime/run_controller.py tests/integration/test_main_shutdown.py
```

- [ ] **Step 7: Commit.**

```bash
git add main.py src/underwater_tracking/agent/llm.py src/underwater_tracking/cli.py src/underwater_tracking/runtime/run_controller.py tests/agent/test_background_cycle.py tests/runtime/test_run_controller.py tests/main/test_main.py tests/integration/test_main_shutdown.py
git commit -m "fix: bound provider cancellation and live shutdown"
```

---

### Task 8: Run the phase-one gate and record evidence

**Files:**
- Create: `docs/superpowers/reports/2026-08-22-planning-epoch-runtime-liveness-acceptance.md`

**Interfaces:**
- Consumes: planning epoch repository, coordinator, semantic commit port, health view, and bounded shutdown from Tasks 1-7.
- Produces: a phase-one acceptance report required before phase two begins.

- [ ] **Step 1: Run focused tests.**

```bash
PYTHONPATH=src python -m pytest tests/domain/test_planning_epoch_models.py tests/agent/test_planning_epoch_repository.py tests/runtime/test_planning_epoch.py tests/runtime/test_scenario_transition.py tests/planning/test_mission_revalidation.py tests/agent/test_uuv_plan_commit_repository.py tests/agent/test_background_cycle.py tests/agent/test_commit.py tests/integration/test_slow_llm_api_responsiveness.py tests/integration/test_main_shutdown.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run backend quality gates.**

```bash
PYTHONPATH=src python -m pytest -m "not real_llm and not long_running" -q
ruff check main.py src tests
mypy src/underwater_tracking
```

Expected: all commands pass with zero failures and zero mypy errors.

- [ ] **Step 3: Run a controlled slow-provider live smoke.**

Run the local delayed provider for longer than 20 observation revisions. Record health response latency, epoch base/current revisions, committed `plan_version`, and shutdown duration. Acceptance requires `plan_version >= 1`, p95 health latency below 500 ms in the test environment, and one-signal shutdown below 10 seconds.

- [ ] **Step 4: Write the acceptance report with exact commands, exit codes, timings, epoch IDs, plan versions, and any environment-specific limits.**

- [ ] **Step 5: Commit the report.**

```bash
git add docs/superpowers/reports/2026-08-22-planning-epoch-runtime-liveness-acceptance.md
git commit -m "test: document planning epoch runtime acceptance"
```
