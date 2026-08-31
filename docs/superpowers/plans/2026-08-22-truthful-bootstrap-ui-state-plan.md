# Truthful Bootstrap and UI State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make initialization, UUV inventory, target priors, execution groups, brain status, map framing, and sidebar content reflect one authoritative runtime state from the first frame through physical deployment.

**Architecture:** Introduce a typed public target-search prior separate from hidden target truth and sensor estimates. Initialize all UUV resources in `MissionController`, keep onboard plan assignments separate from waterborne execution groups, and derive operation-frame brain status from actual epoch/LLM ledger records. Update the React map and sidebar to render those explicit contracts without fallback fabrication.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, deterministic simulation, SQLite, FastAPI operational frames, React 18, TypeScript, Vitest, Playwright, pytest, Ruff, mypy.

## Global Constraints

- Live initialization contains exactly one carrier, three mother ships, twelve onboard UUVs, one submarine target, and zero USVs.
- UUV ownership is immutable and available in the first `MissionSnapshot` and first `OperationalFrame`.
- Hidden target truth is used only by physics and sensor gating.
- A public target prior is source-attributed; a sensor estimate always has real observation IDs.
- No `(0, 0)` or map-center estimate is synthesized when no prior or observation exists.
- An onboard UUV may have a planned assignment but never belongs to an execution group.
- Deployment event, physical exposure, execution-group creation, and frame publication occur in one observation boundary.
- UI components do not synthesize missing brain records, target records, UUV owners, or online states.
- Phase one planning-epoch acceptance must pass before this plan's live acceptance begins.

---

### Task 1: Add a source-attributed public target-search prior

**Files:**
- Modify: `src/underwater_tracking/config/models.py`
- Modify: `src/underwater_tracking/domain/models.py`
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `configs/scenario/uuv_only_single_target.yaml`
- Modify: `tests/config/test_uuv_only_config.py`
- Create: `tests/domain/test_target_search_prior.py`

**Interfaces:**

```python
class TargetSearchPriorConfig(StrictModel):
    prior_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    source: IntelligenceSource
    issued_at_s: int = Field(ge=0)
    valid_until_s: int = Field(gt=0)
    center_xy: tuple[float, float]
    covariance_xy: tuple[tuple[float, float], tuple[float, float]]
    confidence: float = Field(ge=0, le=1)

class ScenarioConfig(StrictModel):
    target_search_priors: tuple[TargetSearchPriorConfig, ...] = ()

class TargetSearchPrior(StrictModel):
    prior_id: str
    target_id: str
    source: IntelligenceSource
    issued_at_s: int
    valid_until_s: int
    center_xy: tuple[float, float]
    covariance_xy: tuple[tuple[float, float], tuple[float, float]]
    confidence: float

class TargetPriorView(StrictModel):
    prior_id: str
    target_id: str
    source: IntelligenceSource
    issued_at_s: int
    valid_until_s: int
    center: Point2D
    covariance_ellipse: CovarianceEllipse
    confidence: float
```

- [ ] **Step 1: Add failing validation tests.**

```python
def test_default_target_prior_is_public_and_not_truth_equal() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    prior = config.scenario.target_search_priors[0]
    truth = config.environment.submarines[0].position_xy
    assert prior.target_id == "target_00"
    assert prior.source is IntelligenceSource.TECHNICAL_RECONNAISSANCE
    assert prior.center_xy != truth
    assert prior.valid_until_s > prior.issued_at_s
```

Add rejection tests for duplicate prior IDs, unknown target IDs, non-positive-definite covariance, non-finite values, reversed validity, and a center outside map bounds.

Add an expiry test: at `sim_time_s == valid_until_s`, the prior disappears from the active publication set and exactly one `target_prior_expired` critical event is emitted. Without a replacement prior or sensor estimate, a new plan cannot continue treating the expired region as evidence.

- [ ] **Step 2: Run tests and verify the config rejects the new field.**

```bash
PYTHONPATH=src python -m pytest tests/config/test_uuv_only_config.py tests/domain/test_target_search_prior.py -q
```

- [ ] **Step 3: Implement strict prior models and cross-validation in `AppConfig`.**

Validate target IDs after platform-core files are loaded. Require covariance symmetry and positive eigenvalues. Do not compare a prior with hidden truth in production validation; the non-equality assertion is an acceptance-fixture guard only.

- [ ] **Step 4: Add a realistic default prior near the task corridor.**

```yaml
target_search_priors:
  - prior_id: intel-target-00-initial
    target_id: target_00
    source: technical_reconnaissance
    issued_at_s: 0
    valid_until_s: 1800
    center_xy: [-4200.0, -6200.0]
    covariance_xy: [[360000.0, 0.0], [0.0, 360000.0]]
    confidence: 0.45
```

This is public mission intelligence, not a sensor estimate and not exact target truth.

- [ ] **Step 5: Verify.**

```bash
PYTHONPATH=src python -m pytest tests/config/test_uuv_only_config.py tests/domain/test_target_search_prior.py -q
ruff check src/underwater_tracking/config/models.py src/underwater_tracking/domain/models.py src/underwater_tracking/domain/ui_models.py tests/domain/test_target_search_prior.py
mypy src/underwater_tracking/config/models.py src/underwater_tracking/domain/models.py src/underwater_tracking/domain/ui_models.py
```

- [ ] **Step 6: Commit.**

```bash
git add configs/scenario/uuv_only_single_target.yaml src/underwater_tracking/config/models.py src/underwater_tracking/domain/models.py src/underwater_tracking/domain/ui_models.py tests/config/test_uuv_only_config.py tests/domain/test_target_search_prior.py
git commit -m "feat: model source attributed target search priors"
```

---

### Task 2: Remove bootstrap execution groups and fake target estimates

**Files:**
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/groups/manager.py`
- Modify: `src/underwater_tracking/domain/models.py`
- Create: `src/underwater_tracking/runtime/observation_boundary.py`
- Modify: `src/underwater_tracking/runtime/scenario_transition.py`
- Modify: `src/underwater_tracking/runtime/mission_controller.py`
- Modify: `src/underwater_tracking/api/live.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `tests/integration/test_platform_core_scenario.py`
- Modify: `tests/simulation/test_uuv_only_carrier_group.py`
- Create: `tests/simulation/test_execution_group_activation.py`
- Create: `tests/runtime/test_observation_boundary.py`
- Modify: `tests/runtime/test_scenario_transition.py`

**Interfaces:**

```python
class SituationSnapshot(StrictModel):
    target_search_priors: tuple[TargetSearchPrior, ...] = ()

class ExecutionGroupState(StrictModel):
    group_id: str
    target_id: str
    region_id: str
    member_ids: tuple[str, ...]
    mode: Literal["active_scan", "passive_track", "returning"]

class SimulationEngine:
    def activate_execution_group(
        self,
        *,
        target_id: str,
        region_id: str,
        member_ids: tuple[str, ...],
    ) -> ExecutionGroupState: ...

    def deactivate_execution_group(self, group_id: str) -> None: ...

class ObservationBoundaryCommitter:
    def __init__(self, transitions: ScenarioTransitionCoordinator, ...) -> None: ...
    def commit(self, delta: PhysicalObservationBatch) -> CommittedStateBundle: ...
```

- [ ] **Step 1: Add an initial-snapshot regression test.**

```python
def test_explicit_uuv_only_initialization_has_no_execution_group_or_estimate() -> None:
    engine = build_default_engine()
    snapshot = engine.publication_situation()
    assert snapshot.group_reports == ()
    assert snapshot.execution_groups == ()
    assert tuple(prior.prior_id for prior in snapshot.target_search_priors) == (
        "intel-target-00-initial",
    )
    assert engine._assignments == {}
    assert engine._latest_reports == {}
```

- [ ] **Step 2: Add activation tests.**

Reject activation if any member is onboard, failed, owned by another planned carrier mission, duplicated, or not exposed by the same committed deployment boundary. Accept only waterborne members and verify `ExecutionGroupState.member_ids` changes atomically without creating a target belief or tracking `GroupReport`. Add a second test showing that a later real fused belief with non-empty `source_observation_ids` creates the first tracking report and permits `PASSIVE_TRACK`.

Inject failures after controller apply and after execution-group reconciliation. Assert engine/controller checkpoints, revisions and latest published frame remain unchanged. Race an observation commit against a phase-one plan commit using barriers; both must use the exact same coordinator instance, serialize completely, and preserve both successful revisions regardless of acquisition order.

- [ ] **Step 3: Run tests and confirm `_initialize_explicit_groups()` causes failure.**

```bash
PYTHONPATH=src python -m pytest tests/integration/test_platform_core_scenario.py tests/simulation/test_uuv_only_carrier_group.py tests/simulation/test_execution_group_activation.py tests/runtime/test_observation_boundary.py tests/runtime/test_scenario_transition.py -q
```

- [ ] **Step 4: Delete `_COARSE_PRIOR` use and `_initialize_explicit_groups()` from the explicit UUV-only path.**

Do not change legacy synthetic scenarios. Publish configured `TargetSearchPrior` records separately. `GroupManager.create()` may keep its legacy coarse-prior argument for legacy tests, but the new `activate_execution_group()` never takes or creates a `TargetBelief`. Only the later observation-fusion path may call the legacy tracking-report constructor, and only with real `source_observation_ids`.

- [ ] **Step 5: Wire physical deployment to group activation.**

At the observation boundary that consumes `uuv_deployed`, collect newly exposed UUV IDs by active regional mission. Create the execution group only after all required members are exposed. If that condition is not met before the batch deployment deadline, mark already exposed members `RETURN_REQUIRED` and let their original mother ship recover them; do not create a partial group. Recovery removes members; an empty group is deactivated. Planned assignments remain only in `MissionSnapshot`/regional plan views, not `SituationSnapshot` or `GroupReport`.

`ObservationBoundaryCommitter` receives the same scenario-scoped `ScenarioTransitionCoordinator` instance as `EpochCommitPort`, holds it for the full transition, checkpoints engine/controller state, applies the physical batch to the controller, reconciles execution groups, and returns one frozen bundle carrying the same `physics_revision` and `mission_revision` to the publisher. On injected failure it restores both checkpoints and publishes no frame. `_AgentLoop` is the sole dependency-composition site, so creating separate locks is impossible in live mode.

- [ ] **Step 6: Verify no slave/adversary context exists at initialization.**

```python
snapshot = engine.publication_situation()
assert snapshot.execution_groups == ()
assert engine.build_slave_contexts(snapshot) == ()
assert all(not item.detected_platforms for item in engine.build_adversary_inputs(snapshot))
```

Phase three may add one target-owned `target_mission_initialized` input at time zero. This phase-two assertion forbids blue-platform evidence before deployment; it does not require the adversary input tuple itself to remain empty.

- [ ] **Step 7: Run focused tests and commit.**

```bash
PYTHONPATH=src python -m pytest tests/integration/test_platform_core_scenario.py tests/simulation/test_uuv_only_carrier_group.py tests/simulation/test_execution_group_activation.py tests/runtime/test_observation_boundary.py tests/runtime/test_scenario_transition.py tests/groups -q
ruff check src/underwater_tracking/simulation/engine.py tests/simulation/test_execution_group_activation.py
git add src/underwater_tracking/simulation/engine.py src/underwater_tracking/groups/manager.py src/underwater_tracking/domain/models.py src/underwater_tracking/runtime/observation_boundary.py src/underwater_tracking/runtime/scenario_transition.py src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/api/live.py src/underwater_tracking/cli.py tests/integration/test_platform_core_scenario.py tests/simulation/test_uuv_only_carrier_group.py tests/simulation/test_execution_group_activation.py tests/runtime/test_observation_boundary.py tests/runtime/test_scenario_transition.py
git commit -m "fix: create execution groups only after physical deployment"
```

---

### Task 3: Initialize `MissionController` with authoritative UUV inventory

**Files:**
- Modify: `src/underwater_tracking/runtime/mission_controller.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `tests/runtime/test_mission_controller.py`
- Modify: `tests/api/test_uuv_only_frame_contract.py`

**Interfaces:**

```python
class MissionController:
    def __init__(
        self,
        *,
        scenario_id: str,
        initial_uuv_resources: Mapping[str, UUVResourceState],
        uuv_owner_by_id: Mapping[str, str],
        ...,
    ) -> None: ...
```

`initial_uuv_resources` is required in UUV-only live mode and contains all 12 UUV IDs with `mileage_m=0`, configured energy/health, `deployment_state="onboard"`, and permanent `carrier_id`.

- [ ] **Step 1: Add a first-snapshot inventory test.**

```python
def test_controller_registers_all_onboard_resources_before_any_plan() -> None:
    controller = controller_from_default_config()
    snapshot = controller.snapshot()
    assert len(snapshot.uuv_resources) == 12
    assert set(snapshot.uuv_modes.values()) == {UUVMissionMode.ONBOARD}
    assert {resource.carrier_id for resource in snapshot.uuv_resources.values()} == {
        "carrier_02", "carrier_03", "carrier_04"
    }
```

- [ ] **Step 2: Add atomic owner and deployment tests.**

Reject any observation or plan that changes `carrier_id`; preserve resource entries across apply/replan/recovery; ensure mileage is monotonic and onboard resource observations still update health and energy.

- [ ] **Step 3: Run tests and verify current empty resource maps fail.**

```bash
PYTHONPATH=src python -m pytest tests/runtime/test_mission_controller.py tests/api/test_uuv_only_frame_contract.py -q
```

- [ ] **Step 4: Build initial resources in `_mission_controller_for()`.**

Use only config fields and permanent ownership. `SimulationEngine` submits physical observations for all UUVs, including onboard inventory, so `_record_resource_observations()` must iterate the configured resource map rather than only existing modes created by a plan.

- [ ] **Step 5: Verify and commit.**

```bash
PYTHONPATH=src python -m pytest tests/runtime/test_mission_controller.py tests/api/test_uuv_only_frame_contract.py tests/integration/test_uuv_only_mission_acceptance.py -q
ruff check src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/cli.py tests/runtime/test_mission_controller.py
git add src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/cli.py src/underwater_tracking/simulation/engine.py tests/runtime/test_mission_controller.py tests/api/test_uuv_only_frame_contract.py
git commit -m "fix: initialize authoritative mother ship uuv inventory"
```

---

### Task 4: Publish priors, planned assignments, and ledger-derived brain activity

**Files:**
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/persistence/ledger.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/api/live.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `tests/agent/test_repositories.py`
- Modify: `tests/api/test_frame_pipeline.py`
- Modify: `tests/api/test_uuv_only_frame_contract.py`

**Interfaces:**

```python
class PlannedAssignmentView(StrictModel):
    target_id: str
    region_id: str
    uuv_ids: tuple[str, ...]
    carrier_id: str
    plan_version: int
    status: Literal["planned", "transporting", "ready_to_deploy"]

class ExecutionGroupView(StrictModel):
    group_id: str
    target_id: str
    region_id: str
    member_ids: tuple[str, ...]
    mode: Literal["active_scan", "passive_track", "returning"]

class BrainActivityRecord(StrictModel):
    brain_id: str
    role: Literal["master", "slave", "adversary"]
    status: Literal["unconfigured", "ready", "running", "succeeded", "degraded", "failed"]
    operation: str | None
    sim_time_s: int | None
    evidence_platform_ids: tuple[str, ...]
    message: str

class OperationalFrame(StrictModel):
    target_priors: tuple[TargetPriorView, ...] = ()
    planned_assignments: tuple[PlannedAssignmentView, ...] = ()
    execution_groups: tuple[ExecutionGroupView, ...] = ()
```

```python
def DecisionLedger.latest_role_activity(
    self, scenario_id: str
) -> Mapping[Literal["master", "slave", "adversary"], BrainActivityRecord]: ...
```

- [ ] **Step 1: Add ledger role-activity tests.**

Map master operations (`intent`, `regional_strategy`, `commit`) to master, `slave_sonar_decision` to slave, and `adversary_escape`/the stage-three adversary operation to adversary. A call with an error category produces `failed` or `degraded`; absence of a call produces `ready` only if that role is configured.

- [ ] **Step 2: Add frame truth tests.**

Initial frame assertions:

```python
assert frame.target_estimates == ()
assert len(frame.target_priors) == 1
assert frame.groups == ()
assert frame.execution_groups == ()
assert frame.planned_assignments == ()
assert len(frame.uuv_resources) == 12
brains = {brain.role: brain for brain in frame.brains}
assert brains["slave"].status == "ready"
assert brains["slave"].connected_platform_ids == ()
assert brains["adversary"].status == "ready"
assert brains["adversary"].connected_platform_ids == ()
```

After an adversary ledger decision whose evidence contains `uuv_04`, assert only `("uuv_04",)` appears in the adversary brain connection list. Fix wire ordering to master/slave/adversary, but all tests select by role. `ready` means configured and never invoked; `running` means an active call; a terminal status remains the latest role activity, with its timestamp, until the next call replaces it.

- [ ] **Step 3: Run tests and verify the existing `_build_brain_views()` fails them.**

```bash
PYTHONPATH=src python -m pytest tests/agent/test_repositories.py tests/api/test_frame_pipeline.py tests/api/test_uuv_only_frame_contract.py -q
```

- [ ] **Step 4: Replace heuristic `_build_brain_views()`.**

Pass role activity from `OperationalFramePublisher` into `build_operational_frame()`. Never infer activity from `snapshot.uuvs` or `snapshot.group_reports`. Use active planning epoch health for `master=running`; use ledger evidence for completed status. Derive planned assignments from `MissionSnapshot.regions` and carrier missions, never from `GroupReport`.

Publish one deterministic bootstrap frame before opening the initialization event barrier or starting master/adversary workers. That frame is the only frame required to show every configured, never-invoked role as `ready`; immediately after publication, enqueue time-zero initialization events and allow later frames to show `running` or a terminal status. Add a race test with a zero-latency fake adversary proving it cannot update the bootstrap frame.

- [ ] **Step 5: Preserve compatibility defaults.**

Legacy replay may omit `target_priors`, `planned_assignments`, and new brain statuses; `legacy_frame_adapter` maps old statuses to `degraded` or `ready` without inventing evidence IDs.

- [ ] **Step 6: Verify and commit.**

```bash
PYTHONPATH=src python -m pytest tests/agent/test_repositories.py tests/api/test_frame_pipeline.py tests/api/test_uuv_only_frame_contract.py tests/api/test_live_publisher.py -q
ruff check src/underwater_tracking/domain/ui_models.py src/underwater_tracking/persistence/ledger.py src/underwater_tracking/api
mypy src/underwater_tracking/domain/ui_models.py src/underwater_tracking/persistence/ledger.py src/underwater_tracking/api/frame_builder.py
git add src/underwater_tracking/domain/ui_models.py src/underwater_tracking/persistence/ledger.py src/underwater_tracking/api/frame_builder.py src/underwater_tracking/api/live.py src/underwater_tracking/cli.py tests/agent/test_repositories.py tests/api/test_frame_pipeline.py tests/api/test_uuv_only_frame_contract.py
git commit -m "fix: publish ledger derived operational truth"
```

---

### Task 5: Render authoritative initial, inventory, and brain states in React

**Files:**
- Modify: `src/underwater_tracking/ui/src/types/frames.ts`
- Modify: `src/underwater_tracking/ui/src/types/frames.contract.ts`
- Modify: `src/underwater_tracking/ui/src/components/CanvasMap.tsx`
- Modify: `src/underwater_tracking/ui/src/components/CanvasMap.test.ts`
- Modify: `src/underwater_tracking/ui/src/components/RightSidebar.tsx`
- Modify: `src/underwater_tracking/ui/src/components/RightSidebar.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/CarrierStatusPanel.tsx`
- Modify: `src/underwater_tracking/ui/src/components/CarrierStatusPanel.test.tsx`

**Interfaces:**

```typescript
export type BrainStatus =
  | "unconfigured" | "ready" | "running"
  | "succeeded" | "degraded" | "failed";

export interface TargetPriorView {
  prior_id: string;
  target_id: string;
  source: string;
  issued_at_s: number;
  valid_until_s: number;
  center: Point2D;
  covariance_ellipse: CovarianceEllipse;
  confidence: number;
}
```

- [ ] **Step 1: Add camera tests with no target estimate.**

```typescript
it("frames the carrier group and public search prior before detection", () => {
  const bounds = cameraBoundsForFrame(initialFrame, DEFAULT_VIEW_CONFIG, false);
  expect(pointInside(bounds, initialFrame.carriers[0].position)).toBe(true);
  expect(pointInside(bounds, initialFrame.target_priors[0].center)).toBe(true);
});
```

Assert an empty prior/estimate frame still includes all carrier positions and uses a bounded mission-area fallback.

- [ ] **Step 2: Add sidebar tests for all 12 owners and truthful brain states.**

Remove the adversary fallback object at `RightSidebar.tsx:89`. Assert owner labels use `uuv_resources`, planned assignments say “计划分配” rather than “已编组”, and ready brains do not display online/executing copy.

- [ ] **Step 3: Run Vitest and confirm failures.**

```bash
npm --prefix src/underwater_tracking/ui test -- --run src/components/CanvasMap.test.ts src/components/RightSidebar.test.tsx src/components/CarrierStatusPanel.test.tsx
```

- [ ] **Step 4: Update camera input ordering.**

In local focus mode, include carrier positions, waterborne UUV positions, active regional geometry, target estimates, and target priors. Detection circles remain tied to actual adversary/estimate state and are not drawn around a prior. Keep minimum-span expansion to avoid over-zoom.

- [ ] **Step 5: Update sidebar and inventory presentation.**

Use new brain status labels. Do not synthesize adversary cards. Keep onboard UUV rows selectable in inventory, but map selection must not create a spatial hit target for them. Show permanent mother ID even before any plan.

- [ ] **Step 6: Verify frontend tests and build.**

```bash
npm --prefix src/underwater_tracking/ui test -- --run
npm --prefix src/underwater_tracking/ui run build
```

- [ ] **Step 7: Commit.**

```bash
git add src/underwater_tracking/ui/src/types/frames.ts src/underwater_tracking/ui/src/types/frames.contract.ts src/underwater_tracking/ui/src/components/CanvasMap.tsx src/underwater_tracking/ui/src/components/CanvasMap.test.ts src/underwater_tracking/ui/src/components/RightSidebar.tsx src/underwater_tracking/ui/src/components/RightSidebar.test.tsx src/underwater_tracking/ui/src/components/CarrierStatusPanel.tsx src/underwater_tracking/ui/src/components/CarrierStatusPanel.test.tsx
git commit -m "fix: render truthful bootstrap and inventory state"
```

---

### Task 6: Verify four atomic mission frames from the default entry path

**Files:**
- Create: `tests/integration/test_truthful_bootstrap_deployment_frames.py`
- Modify: `tests/integration/test_uuv_initialization_local_perception.py`

**Interfaces:**
- Consumes: default config, phase-one controlled LLM, `OperationalFramePublisher`, and physical deployment events.
- Produces: four persisted frames named by semantic checkpoint, not hard-coded tick number.

- [ ] **Step 1: Write a semantic frame collector.**

```python
def collect_transition_frames(frames: Iterable[OperationalFrame]) -> dict[str, OperationalFrame]:
    return {
        "initial": first(frame.plan_version == 0),
        "pre_deploy": first(frame.planned_assignments and not any_exposed(frame)),
        "deploy": first(has_event(frame, "uuv_deployed")),
        "post_deploy": first(frame.execution_groups and any_exposed(frame)),
    }
```

- [ ] **Step 2: Assert exact cross-frame invariants.**

Initial has 12 owned onboard resources, one prior, no estimate/tracking/execution group. Pre-deploy has planned assignments but no exposed assigned UUV. Deploy has event, exposure, and `ExecutionGroupView` in the same frame while `groups` remains empty until real fused target evidence exists. Post-deploy has active-scan slave contexts only for execution-group members. No frame contains USV fields or onboard map exposure.

Also assert the configured target is 2.5-4.0 km from the nearest mother ship and outside 1200 m local detection at time zero; two engines built from the same config/seed have byte-equal initial authoritative snapshots. Advance the undispatched formation along one route segment and assert `carrier_01` follows its loop while each standby mother ship preserves its configured slot offset within tolerance.

- [ ] **Step 3: Remove `_co_locate_test_carriers()` from the acceptance path.**

The integration test must use the unmodified default config and controlled phase-one LLM output. Test helpers may accelerate entity speed or service duration only through a dedicated acceptance config override that preserves initial positions and ownership.

- [ ] **Step 4: Run integration tests.**

```bash
PYTHONPATH=src python -m pytest tests/integration/test_truthful_bootstrap_deployment_frames.py tests/integration/test_uuv_initialization_local_perception.py -q
```

- [ ] **Step 5: Commit.**

```bash
git add tests/integration/test_truthful_bootstrap_deployment_frames.py tests/integration/test_uuv_initialization_local_perception.py
git commit -m "test: verify truthful bootstrap through physical deployment"
```

---

### Task 7: Run the phase-two gate and record evidence

**Files:**
- Create: `docs/superpowers/reports/2026-08-22-truthful-bootstrap-ui-state-acceptance.md`

**Interfaces:**
- Consumes: all phase-two contracts and the phase-one acceptance report.
- Produces: a phase-two gate required before adversary/kinematics work is accepted.

- [ ] **Step 1: Run backend gates.**

```bash
PYTHONPATH=src python -m pytest tests/config/test_uuv_only_config.py tests/domain/test_target_search_prior.py tests/integration/test_platform_core_scenario.py tests/simulation/test_execution_group_activation.py tests/runtime/test_observation_boundary.py tests/runtime/test_mission_controller.py tests/api/test_uuv_only_frame_contract.py tests/integration/test_truthful_bootstrap_deployment_frames.py -q
ruff check main.py src tests
mypy src/underwater_tracking
```

- [ ] **Step 2: Run frontend gates.**

```bash
npm --prefix src/underwater_tracking/ui test -- --run
npm --prefix src/underwater_tracking/ui run build
```

- [ ] **Step 3: Capture initial and deployed API/frame evidence.**

Record carrier positions, target prior, target estimate count, each UUV owner/deployment/exposure/group, brain statuses, event IDs, and `plan_version`. Do not use or modify the user's reference screenshots.

- [ ] **Step 4: Write the report with exact commands and serialized invariant tables.**

- [ ] **Step 5: Commit the report.**

```bash
git add docs/superpowers/reports/2026-08-22-truthful-bootstrap-ui-state-acceptance.md
git commit -m "test: document truthful bootstrap ui acceptance"
```
