# UUV-Only Production Closed Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incomplete UUV-only execution path with a deterministic production loop in which verified LLM plans drive physical multi-carrier deployment, UUV sensing, handoff, resource rotation, recovery, and event-triggered replanning.

**Architecture:** `ExecutableMissionPlan` is the only plan that can control a UUV-only run. `MissionController` owns immutable mission state, lifecycle transitions, resource episodes, and mission events. `SimulationEngine` owns kinematics and observations, while a narrow execution adapter reconciles controller commands with the carrier fleet and UUV entities. `CarrierRuntime` and every CLI entry point apply the latest verified executable plan; the legacy `TrackingPlan` path remains available only for non-UUV scenarios and old replay reads.

**Tech Stack:** Python 3.11/3.12-compatible Pydantic models, NumPy/SciPy, LangGraph, deterministic carrier/UUV simulation, pytest, Ruff, JSONL replay, and existing React/API frame contracts.

## Global Constraints

- UUVs are the only sensing and tracking platforms; carriers provide deployment, recovery, route, and inventory logistics only.
- A UUV-only run must create and expose every configured carrier, and every carrier mission must start and finish at its home battle group.
- `MissionController` is the sole source of region lifecycle, UUV mission mode, resource rotation, handoff, recovery, event, and executable-plan revision state.
- `SimulationEngine` never invokes an LLM and never uses legacy `TrackingPlan` to control a UUV-only run.
- LLM output is a candidate policy; deterministic validation, live resource checks, capacity checks, and complete A* route validation precede plan application.
- Entry confirmation remains configurable with defaults of probability `0.70` and two consecutive observation cycles.
- Mileage includes UUV transit, scan/track motion, and travel to recovery; recovery resets the sortie counters only after a completed carrier recovery and health check.
- Strategic mission events are classified by `EventMonitor`, enter the LangGraph replan path, and carry plan revision/resource-episode context.
- An invalid or unavailable LLM preserves the last verified executable plan and records `llm_degraded`; it never fabricates a successful new plan.
- New operational frames and new JSONL records omit USV fields; legacy readers accept and discard old USV fields.
- Fixed-seed integration tests must prove deterministic plan, route, event, and frame hashes without exposing target truth.

---

### Task 1: Define live mission resources and complete event contracts

**Files:**
- Modify: `src/underwater_tracking/domain/mission_models.py`
- Modify: `src/underwater_tracking/runtime/mission_controller.py`
- Modify: `src/underwater_tracking/agent/nodes/event_monitor.py`
- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Test: `tests/runtime/test_mission_controller.py`
- Test: `tests/agent/test_event_monitor.py`

**Interfaces:**
- Add immutable `UUVResourceState(uuv_id, carrier_id, mileage_m, energy_fraction, healthy, capability_active, deployment_state)`.
- Extend `MissionSnapshot` with `uuv_resources` and `resource_episode_by_uuv` mappings while preserving existing fields.
- `MissionController.advance()` accepts `recovered_uuv_ids`, `carrier_dispatch_completed`, `carrier_recovery_completed`, `entry_probability`, `handoff_ready`, `successor_passive_ready`, and estimated intent/confidence observations.
- `MissionController.acknowledge_recovery(uuv_id, sim_time_s)` moves a recovered UUV to `ONBOARD`, clears its recovery episode, restores the carrier ready inventory, and emits `carrier_recovery_completed` once per episode.
- `EventMonitor.classify()` recognizes every design event plus `llm_degraded`; unknown events remain errors.

- [x] **Step 1: Write failing controller tests for real resource episodes.**

```python
def test_exhausted_uuv_emits_recovery_and_reenters_ready_pool_after_ack() -> None:
    controller = controller_with_plan(single_uuv_plan("U01", carrier_id="carrier_01"))

    first = controller.advance(
        10,
        {
            "deployed_uuv_ids": {"region-1": ("U01",)},
            "mileage_m": {"U01": 50_000.0},
            "energy_fraction": {"U01": 0.8},
        },
    )
    assert first.uuv_modes["U01"] is UUVMissionMode.RETURN_REQUIRED
    assert {event.event_type for event in first.events} >= {
        "uuv_range_exhausted",
        "region_coverage_degraded",
    }

    recovered = controller.advance(20, {"recovered_uuv_ids": ("U01",)})
    assert recovered.uuv_modes["U01"] is UUVMissionMode.ONBOARD
    assert recovered.carrier_missions["carrier_01"].ready_uuv_ids == ("U01",)
    assert any(event.event_type == "carrier_recovery_completed" for event in recovered.events)

def test_handoff_creates_recovery_for_predecessor_uuvs() -> None:
    controller = controller_with_handoff_plan("U01", "U02")
    snapshot = controller.advance(
        30,
        {
            "deployed_uuv_ids": {"region-a": ("U01",), "region-b": ("U02",)},
            "entry_probability": {"region-a": 0.9, "region-b": 0.9},
            "handoff_ready": {"region-a": "region-b"},
            "successor_passive_ready": {"region-b": True},
        },
    )
    assert snapshot.uuv_modes["U01"] is UUVMissionMode.RETURN_REQUIRED
    assert snapshot.regions[0].lifecycle in {
        RegionLifecycle.TRACKING_COMPLETED,
        RegionLifecycle.CARRIER_RECOVERY,
    }
```

- [x] **Step 2: Run the focused tests and confirm the missing transitions.**

```bash
PYTHONPATH=src pytest -q tests/runtime/test_mission_controller.py::test_exhausted_uuv_emits_recovery_and_reenters_ready_pool_after_ack tests/runtime/test_mission_controller.py::test_handoff_creates_recovery_for_predecessor_uuvs
```

Expected: failure because the controller does not accept recovery acknowledgements and handoff does not change predecessor UUV modes.

- [x] **Step 3: Implement resource episodes, recovery acknowledgements, and event classification.**

Track a per-UUV episode integer. Include it in event IDs as `scenario:plan_revision:event_type:entity:episode:sim_time`, clear the episode only after recovery health check, and use the `(event_type, entity_id, episode)` tuple for idempotence. On return, always locate the UUV through the plan batch/carrier mapping; do not require it to still be in an inventory group. On handoff, transition the predecessor region through `HANDOFF_PENDING` to `TRACKING_COMPLETED`, set its assigned UUVs to `RETURN_REQUIRED`, add them to the carrier recoverable inventory, and leave the successor in `PASSIVE_TRACK` only when its readiness observation is true.

Add the eleven design events and `llm_degraded` to the strategic event set in `event_monitor.py`. In `central.py`, route these events to the regional strategy/replan branch instead of the legacy regional feedback branch. Preserve existing old event classifications for legacy scenarios.

- [x] **Step 4: Run controller, event, and regression tests.**

```bash
PYTHONPATH=src pytest -q tests/runtime/test_mission_controller.py tests/agent/test_event_monitor.py tests/agent/test_central_graph.py
ruff check src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/agent/nodes/event_monitor.py src/underwater_tracking/agent/graphs/central.py tests/runtime/test_mission_controller.py tests/agent/test_event_monitor.py
```

- [x] **Step 5: Commit.**

```bash
git add src/underwater_tracking/domain/mission_models.py src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/agent/nodes/event_monitor.py src/underwater_tracking/agent/graphs/central.py tests/runtime/test_mission_controller.py tests/agent/test_event_monitor.py
git commit -m "fix: close uuv mission resource and event state"
```

### Task 2: Make planning consume live UUV and carrier capability

**Files:**
- Modify: `src/underwater_tracking/planning/mission_optimizer.py`
- Modify: `src/underwater_tracking/planning/carrier_tasks.py`
- Modify: `src/underwater_tracking/planning/hungarian.py`
- Modify: `src/underwater_tracking/planning/astar.py`
- Modify: `src/underwater_tracking/domain/mission_models.py`
- Test: `tests/planning/test_mission_optimizer.py`
- Test: `tests/planning/test_carrier_tasks.py`
- Test: `tests/planning/test_hungarian.py`

**Interfaces:**
- `_platform_pool(snapshot, home_battle_group_id)` returns every healthy carrier and every UUV whose deployment, health, capability, energy, and range make it eligible.
- `MissionOptimizer.optimize()` produces batches keyed by actual carrier ID and never selects failed, returning, recovering, or below-threshold UUVs.
- `CarrierTaskPlanner.build_tasks()` emits deterministic deploy/recover tasks for every batch and validates carrier membership and inventory.
- `CarrierTaskPlanner.build_routes()` returns complete home-to-stops-to-home `CarrierMissionModel` routes for all carriers with assigned tasks.
- `HungarianMatcher` rejects a slot when the full route with the new stop cannot return home or when post-assignment ready inventory violates reserve requirements.

- [x] **Step 1: Write failing live-resource and multi-carrier tests.**

```python
def test_optimizer_excludes_low_energy_returning_and_failed_uuvs() -> None:
    snapshot = snapshot_with_resources(
        uuvs={
            "U01": resource(energy_fraction=0.9, deployment_state="onboard"),
            "U02": resource(energy_fraction=0.05, deployment_state="onboard"),
            "U03": resource(energy_fraction=0.9, deployment_state="returning"),
            "U04": resource(energy_fraction=0.9, healthy=False, deployment_state="onboard"),
        }
    )
    plan = MissionOptimizer().optimize(snapshot, (candidate("region-1", minimum=1),))
    assert plan.all_uuv_ids == ("U01",)

def test_routes_are_generated_for_two_carriers_and_return_home() -> None:
    plan = plan_with_batches_on_two_carriers()
    missions = CarrierTaskPlanner().build_routes(plan, carrier_states(), map_bounds=(-20, 20, -20, 20))
    assert set(missions) == {"carrier_01", "carrier_02"}
    assert all(route.route_xy[0] == route.route_xy[-1] for route in missions.values())
```

- [x] **Step 2: Run the focused tests and observe current failures.**

```bash
PYTHONPATH=src pytest -q tests/planning/test_mission_optimizer.py tests/planning/test_carrier_tasks.py tests/planning/test_hungarian.py
```

Expected: low-energy UUVs are currently selected and the optimizer creates only one carrier mission with no route.

- [x] **Step 3: Implement live resource filtering and route materialization.**

Read `platform_snapshot.roster.uuvs` and the new `uuv_resources` mapping when present. Exclude `failed`, `returning`, `recovering`, unhealthy, inactive-capability, and `energy_fraction <= min_energy_fraction` UUVs. Read `platform_snapshot.carriers` when available, falling back to the legacy singular carrier for old snapshots. Assign batches to deterministic carrier slots by ready count, route distance, and carrier ID. Use `CarrierTaskPlanner` plus `AStarRoutePlanner` to create route stops and recoveries; write route status and inventory counts back into each `CarrierMissionModel` before returning the executable plan.

Update the A* route validator to use current carrier position as the start, every committed service stop in order, and the home point as a mandatory final node. Update Hungarian costs to include incremental distance, ETA slack, required UUV count, remaining ready inventory, and future reserve loss; reject infeasible assignments instead of returning a straight-line fallback.

- [x] **Step 4: Verify planning behavior.**

```bash
PYTHONPATH=src pytest -q tests/planning/test_mission_optimizer.py tests/planning/test_carrier_tasks.py tests/planning/test_hungarian.py tests/planning/test_astar.py
ruff check src/underwater_tracking/planning src/underwater_tracking/domain/mission_models.py tests/planning
```

- [x] **Step 5: Commit.**

```bash
git add src/underwater_tracking/planning/mission_optimizer.py src/underwater_tracking/planning/carrier_tasks.py src/underwater_tracking/planning/hungarian.py src/underwater_tracking/planning/astar.py src/underwater_tracking/domain/mission_models.py tests/planning/test_mission_optimizer.py tests/planning/test_carrier_tasks.py tests/planning/test_hungarian.py
git commit -m "fix: make uuv planning resource and carrier aware"
```

### Task 3: Implement a physical carrier fleet and mission reconciliation

**Files:**
- Modify: `src/underwater_tracking/simulation/carrier.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/simulation/uuv.py`
- Modify: `src/underwater_tracking/domain/models.py`
- Modify: `src/underwater_tracking/domain/platforms.py`
- Test: `tests/simulation/test_carrier.py`
- Test: `tests/simulation/test_deployment_lifecycle.py`
- Create: `tests/integration/test_uuv_only_physical_execution.py`

**Interfaces:**
- `SimulationEngine._carrier_entities` stores every configured carrier; `_carrier_entity` remains an alias to the primary carrier only for legacy code paths.
- `SimulationEngine.apply_verified_mission_plan(plan)` validates live UUV/carrier IDs, installs complete routes, atomically applies the controller plan, and calls `_reconcile_uuv_mission_state()`.
- `_reconcile_uuv_mission_state()` maps controller modes to `request_uuv_deployment`, `set_sensor_mode`, and `request_uuv_recovery`; it never calls legacy `apply_tracking_plan` for UUV-only runs.
- `CarrierEntity` exposes deterministic mission-stop arrival detection and a finite route that ends at the configured home point.
- `SituationSnapshot` and `PlatformSnapshot` expose a plural carrier view while retaining the primary singular carrier for backward-compatible readers.

- [x] **Step 1: Write failing physical-loop tests.**

```python
def test_verified_plan_moves_two_carriers_and_executes_three_stops() -> None:
    engine = small_uuv_only_engine_with_two_carriers()
    plan = three_stop_plan_for_two_carriers()
    assert engine.apply_verified_mission_plan(plan) is True

    frames = advance_until(engine, lambda frame: engine.mission_snapshot().plan_revision == plan.revision and all_carriers_home(engine))
    assert {carrier.carrier_id for carrier in engine.carrier_states()} == {"carrier_01", "carrier_02"}
    assert deployment_events(frames).count("uuv_deployed") >= 2
    assert recovery_events(frames).count("uuv_recovered") >= 1
    assert every_route_returns_home(engine)

def test_resource_exhaustion_requests_recovery_and_resets_sortie_counters() -> None:
    engine = small_uuv_only_engine_with_two_carriers(max_uuv_mileage_m=10.0)
    engine.apply_verified_mission_plan(single_uuv_plan("U01"))
    engine.force_mission_distance_for_test("U01", 10.0)
    frames = advance_until(engine, lambda: engine.mission_snapshot().uuv_modes["U01"] is UUVMissionMode.ONBOARD)
    assert contains_event(frames, "uuv_range_exhausted")
    assert contains_event(frames, "carrier_recovery_completed")
    assert engine.mission_distance("U01") == 0.0
```

- [x] **Step 2: Run the new tests and confirm physical execution is absent.**

```bash
PYTHONPATH=src pytest -q tests/simulation/test_carrier.py tests/simulation/test_deployment_lifecycle.py tests/integration/test_uuv_only_physical_execution.py
```

Expected: `apply_verified_mission_plan` may update the controller but no configured second carrier moves and no controller mode is reconciled to a physical deployment/recovery.

- [x] **Step 3: Add carrier-fleet state and the reconciliation adapter.**

Construct one `CarrierEntity` per configured carrier, maintain `_uuv_carrier_ids`, and make `_advance_world()` step every carrier. An onboard or returning UUV follows its assigned carrier, not the primary carrier alias. At each route stop, execute the matching deployment/recovery task exactly once; deployment changes the engine state and sensor mode, while recovery places the UUV onboard, resets speed/waypoints/mileage, restores maintenance energy, and sends `recovered_uuv_ids` to the controller at the next observation boundary.

`apply_verified_mission_plan` must reject unknown IDs, duplicate carrier ownership, a route whose first point is not the carrier's current position, or a route whose last point is not home. It must apply the controller plan only after all validation succeeds. The reconciler sets active-scan UUVs to active sonar, passive-track UUVs to passive mode, and returns UUVs to the assigned carrier recovery task.

Add plural carrier states to snapshots and use the primary carrier only when constructing legacy fields. The new UUV-only frame builder will consume the plural field.

- [x] **Step 4: Verify physical lifecycle and legacy scenarios.**

```bash
PYTHONPATH=src pytest -q tests/simulation/test_carrier.py tests/simulation/test_deployment_lifecycle.py tests/integration/test_uuv_only_physical_execution.py tests/integration/test_platform_core_scenario.py
ruff check src/underwater_tracking/simulation/carrier.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/domain/models.py src/underwater_tracking/domain/platforms.py tests/simulation tests/integration/test_uuv_only_physical_execution.py
```

- [x] **Step 5: Commit.**

```bash
git add src/underwater_tracking/simulation/carrier.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/simulation/uuv.py src/underwater_tracking/domain/models.py src/underwater_tracking/domain/platforms.py tests/simulation/test_carrier.py tests/simulation/test_deployment_lifecycle.py tests/integration/test_uuv_only_physical_execution.py
git commit -m "feat: execute uuv plans through the carrier fleet"
```

### Task 4: Connect verified executable plans to every production entry point

**Files:**
- Modify: `src/underwater_tracking/runtime/run_controller.py`
- Modify: `src/underwater_tracking/agent/runtime.py`
- Modify: `src/underwater_tracking/agent/nodes/optimize.py`
- Modify: `src/underwater_tracking/agent/nodes/commit.py`
- Modify: `src/underwater_tracking/cli.py`
- Test: `tests/agent/test_agent_loop.py`
- Modify: `tests/agent/test_regional_plan_pipeline.py`
- Create: `tests/integration/test_uuv_only_runtime_entrypoints.py`

**Interfaces:**
- `CarrierRuntime.active_mission_plan()` returns the latest verified `ExecutableMissionPlan` from the graph checkpoint.
- `_AgentLoop._apply_new_commands()` applies `active_mission_plan()` through `engine.apply_verified_mission_plan()` when `config.scenario.uuv_only` is true; it does not call `apply_tracking_plan` or legacy commands on that path.
- `_simulate`, `_agent_run`, and `_serve` all construct the same mission controller/execution bundle for UUV-only scenarios.
- `CommitNode` records an executable-plan revision and commit result, while legacy plan persistence remains only for non-UUV compatibility.

- [x] **Step 1: Write failing production-entry tests.**

```python
def test_agent_run_applies_executable_plan_instead_of_legacy_tracking_plan(monkeypatch) -> None:
    engine = RecordingEngine(uuv_only=True)
    loop = loop_with_verified_executable_plan(engine)
    loop._apply_new_commands()
    assert engine.applied_executable_revisions == [1]
    assert engine.applied_tracking_plan_ids == []

def test_all_uuv_only_cli_entrypoints_attach_a_mission_controller(monkeypatch) -> None:
    for entrypoint in (_simulate, _agent_run, _serve):
        engine = capture_engine_created_by(entrypoint, uuv_only_config())
        assert engine.mission_snapshot() is not None
```

- [x] **Step 2: Run the focused tests and confirm the old path is still selected.**

```bash
PYTHONPATH=src pytest -q tests/agent/test_agent_loop.py tests/agent/test_regional_plan_pipeline.py tests/integration/test_uuv_only_runtime_entrypoints.py
```

- [x] **Step 3: Wire executable plans and preserve last verified plan on LLM failure.**

Return `executable_mission_plan` from the final graph state and expose it through `CarrierRuntime.active_mission_plan()`. At the safe physics boundary, apply it once per revision. Use a single `_build_mission_runtime_bundle()` helper for `simulate`, `agent-run`, and `serve`, passing the controller into the engine and runtime callback in UUV-only mode. Keep existing `TrackingPlan` application unchanged for legacy scenarios.

When graph execution raises `LLMError` or candidate validation fails, retain the previous executable plan revision, append an `llm_degraded` event with the failed revision and reason, and return a non-committed cycle result. Do not clear the controller's active plan.

- [x] **Step 4: Verify runtime wiring and fallback.**

```bash
PYTHONPATH=src pytest -q tests/agent/test_agent_loop.py tests/agent/test_regional_plan_pipeline.py tests/integration/test_uuv_only_runtime_entrypoints.py tests/agent/test_llm_outage.py
ruff check src/underwater_tracking/runtime src/underwater_tracking/agent/runtime.py src/underwater_tracking/agent/nodes/optimize.py src/underwater_tracking/agent/nodes/commit.py src/underwater_tracking/cli.py tests/agent tests/integration/test_uuv_only_runtime_entrypoints.py
```

- [x] **Step 5: Commit.**

```bash
git add src/underwater_tracking/runtime/run_controller.py src/underwater_tracking/agent/runtime.py src/underwater_tracking/agent/nodes/optimize.py src/underwater_tracking/agent/nodes/commit.py src/underwater_tracking/cli.py tests/agent/test_agent_loop.py tests/agent/test_regional_plan_pipeline.py tests/integration/test_uuv_only_runtime_entrypoints.py
git commit -m "fix: apply verified executable plans in production loops"
```

### Task 5: Replace old UUV-only frame output and add event-driven replan integration

**Files:**
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/api/replay.py`
- Modify: `src/underwater_tracking/api/frame_logger.py`
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Test: `tests/api/test_uuv_only_replay_acceptance.py`
- Test: `tests/api/test_frame_pipeline.py`
- Create: `tests/integration/test_uuv_only_replan_loop.py`

**Interfaces:**
- `build_uuv_only_frame()` projects plural carrier missions, region lifecycle, UUV modes/resources, grid evidence, and event ledger without `usvs` keys.
- `read_legacy_frame()` accepts old `usvs` fields and discards them before validation.
- `SimulationEngine` emits a new mission event only once per resource episode and passes it to the runtime callback in the same observation boundary.
- `test_uuv_only_replan_loop` uses a deterministic LLM provider and asserts event → graph → higher executable revision → physical reconciliation.

- [x] **Step 1: Write failing output and replan tests.**

```python
def test_new_uuv_only_jsonl_omits_usv_fields() -> None:
    frame = run_one_uuv_only_step()
    payload = json.loads(frame_json(frame))
    assert "usvs" not in payload
    assert "USV" not in json.dumps(payload)

def test_range_event_produces_a_new_verified_executable_revision() -> None:
    trace = run_deterministic_replan_trace(event="uuv_range_exhausted")
    assert trace.event_types.index("uuv_range_exhausted") < trace.plan_revisions[-1]
    assert trace.plan_revisions == sorted(set(trace.plan_revisions))
    assert trace.engine_applied_revisions[-1] > trace.engine_applied_revisions[0]
```

- [x] **Step 2: Run the tests and confirm old serialization/replan behavior.**

```bash
PYTHONPATH=src pytest -q tests/api/test_uuv_only_replay_acceptance.py tests/api/test_frame_pipeline.py tests/integration/test_uuv_only_replan_loop.py
```

- [x] **Step 3: Implement the serialization and event-to-plan loop.**

Build the UUV-only frame from `MissionSnapshot` and live situation state. Remove `usvs` before new-frame serialization and before `FrameLogger` writes JSONL. Add a compatibility reader that strips `usvs`/legacy USV IDs before constructing current models. Feed controller events to `CarrierRuntime.submit_events()` at the next graph cycle; the graph must preserve the active executable plan until a strictly newer validated plan is committed.

- [x] **Step 4: Verify API, replay, and event replan behavior.**

```bash
PYTHONPATH=src pytest -q tests/api/test_uuv_only_replay_acceptance.py tests/api/test_frame_pipeline.py tests/integration/test_uuv_only_replan_loop.py tests/integration/test_uuv_only_physical_execution.py
ruff check src/underwater_tracking/api src/underwater_tracking/domain/ui_models.py src/underwater_tracking/simulation/engine.py tests/api tests/integration/test_uuv_only_replan_loop.py
```

- [x] **Step 5: Commit.**

```bash
git add src/underwater_tracking/simulation/engine.py src/underwater_tracking/api/frame_builder.py src/underwater_tracking/api/replay.py src/underwater_tracking/api/frame_logger.py src/underwater_tracking/domain/ui_models.py tests/api/test_uuv_only_replay_acceptance.py tests/api/test_frame_pipeline.py tests/integration/test_uuv_only_replan_loop.py
git commit -m "feat: close uuv mission event and replay loop"
```

### Task 6: Add fixed-seed production acceptance and remove obsolete UUV-only control paths

**Files:**
- Create: `tests/integration/test_uuv_only_production_closed_loop.py`
- Modify: `tests/integration/test_uuv_only_mission.py`
- Modify: `tests/integration/test_uuv_only_mission_acceptance.py`
- Modify: `src/underwater_tracking/agent/nodes/optimize.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `docs/superpowers/audits/2026-08-19-uuv-only-carrier-region-mission-verification.md`

**Interfaces:**
- The production acceptance test drives real `SimulationEngine`, real `MissionController`, real carrier entities, and a deterministic LLM adapter; it does not manually call `MissionController.advance()` to simulate physical events.
- The acceptance trace includes two carriers, one carrier with three service stops, complete home return, active scan, passive track, handoff, predecessor recovery, mileage-triggered rotation, intent/confidence replan, degraded resource handling, and no USV output.
- UUV-only optimizer nodes no longer materialize or commit a legacy `TrackingPlan` as the execution result; the legacy object is created only in non-UUV graph branches.

- [x] **Step 1: Add the fixed-seed acceptance test and make it fail against the current path.**

```python
def test_fixed_seed_production_trace_is_closed_and_deterministic() -> None:
    first = run_production_trace(seed=20260820, provider="deterministic-test-provider-v1")
    second = run_production_trace(seed=20260820, provider="deterministic-test-provider-v1")
    assert first.plan_hash == second.plan_hash
    assert first.route_hash == second.route_hash
    assert first.frame_hash == second.frame_hash
    assert first.carrier_count >= 2
    assert first.max_single_carrier_stops >= 3
    assert first.all_carriers_returned_home
    assert first.lifecycle_sequence_contains("ACTIVE_SCAN", "PASSIVE_TRACK", "HANDOFF_PENDING")
    assert first.event_types.count("uuv_range_exhausted") >= 1
    assert first.replanned_after("uuv_range_exhausted")
    assert first.replanned_after("target_intent_changed")
    assert first.usv_field_count == 0
```

- [x] **Step 2: Run the acceptance test before replacing the old path.**

```bash
PYTHONPATH=src pytest -q tests/integration/test_uuv_only_production_closed_loop.py
```

Expected: failure because CLI/runtime still controls the simulation with legacy plans and the engine still has only the primary carrier in the physical path.

- [x] **Step 3: Remove obsolete UUV-only control branches.**

Delete the UUV-only calls to `apply_tracking_plan`, legacy return-list handling, and legacy sensor command application from `_AgentLoop`; retain them only under `if not config.scenario.uuv_only`. In the optimizer graph, pass the executable plan through commit state and never overwrite it with a legacy plan. Update the previous acceptance tests to assert the production adapter rather than directly invoking controller transitions; keep direct controller tests for transition-unit coverage.

- [x] **Step 4: Run full backend acceptance and update the audit with evidence.**

```bash
PYTHONPATH=src pytest -q tests/integration/test_uuv_only_production_closed_loop.py tests/integration/test_uuv_only_mission.py tests/integration/test_uuv_only_mission_acceptance.py
PYTHONPATH=src pytest -q
```

Replace the old audit's unsupported “通过” claims with a requirement-to-evidence table that cites the new physical acceptance trace and lists any unrelated existing visual baseline difference separately.

- [x] **Step 5: Commit.**

```bash
git add tests/integration/test_uuv_only_production_closed_loop.py tests/integration/test_uuv_only_mission.py tests/integration/test_uuv_only_mission_acceptance.py src/underwater_tracking/agent/nodes/optimize.py src/underwater_tracking/cli.py docs/superpowers/audits/2026-08-19-uuv-only-carrier-region-mission-verification.md
git commit -m "test: prove the uuv-only production closed loop"
```

### Task 7: Final verification and branch handoff

**Files:**
- No production files unless verification exposes a regression.
- Review: all commits on `fix/uuv-only-production-loop` against `master`.

- [x] **Step 1: Run backend, frontend, and contract verification.**

```bash
PYTHONPATH=src pytest -q
npm test -- --run
npm run build
git diff --check master...HEAD
```

- [x] **Step 2: Run the targeted UI/replay acceptance command.**

```bash
npm run test:e2e -- --grep "uuv-only|mission|replay"
```

Record any pre-existing screenshot baseline difference separately from functional failures.

- [x] **Step 3: Request code review using the final branch diff.**

Provide the reviewer the design document, this plan, `master` as the base, and the final branch SHA. Fix every Critical or Important finding, then rerun the affected verification command.

- [x] **Step 4: Prepare integration.**

Use `finishing-a-development-branch` after all tests and review are complete. Present the user with the verified branch summary and merge options; do not claim completion until fresh commands show the stated result.
