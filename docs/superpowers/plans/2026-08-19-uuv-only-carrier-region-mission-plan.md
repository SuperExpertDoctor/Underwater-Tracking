# UUV-Only Carrier Region Mission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement the confirmed UUV-only regional mission loop: deterministic prediction-grid regions, LLM-selected candidate policies, rolling UUV batch optimization, multi-stop carrier deployment/recovery, strict region handoff, versioned replanning, and operator/replay views without publishing USV data in new runs.

**Architecture:** Keep IMM/UIF, prediction, LangGraph, FastAPI/WebSocket, React, and the deterministic simulation engine as reusable services. Add a new mission-domain boundary containing MissionController, MissionOptimizer, CarrierTaskPlanner, HungarianMatcher, and AStarRoutePlanner; the controller owns lifecycle and plan versions while the engine owns only physics and observations. Legacy regional/USV structures are adapted at the read boundary for old replay files, but the new run path produces only UUV mission state and carrier logistics state.

**Tech Stack:** Python 3.11/3.12-compatible Pydantic models, NumPy/SciPy, LangGraph structured LLM ports, FastAPI/WebSocket, JSONL replay, React 18 + TypeScript + Vite, Vitest, Playwright, pytest, Ruff, and Hypothesis where useful.

## Global Constraints

- UUV is the only platform for active sonar scanning, passive bearing, and cooperative tracking; the carrier is logistics-only.
- USV is not created, observed, grouped, allocated, drawn, or written to a new operational frame; old replay fields may be read and ignored.
- Every carrier mission starts and ends at the same home battle group; task regions and other carriers are never final dwell points.
- MissionController is the sole source of region lifecycle, sensor mode, handoff, recovery, event, and plan-execution state.
- SimulationEngine advances physics and generates observations; it does not own mission lifecycle transitions or LLM state mutation.
- LLM output is a candidate policy only; deterministic validation, optimization, capacity checks, and route checks must succeed before atomic plan application.
- Region entry uses configurable region_entry_probability_threshold default 0.70 and region_transition_confirm_cycles default 2.
- Probability controls grid evidence intensity and ranking; task-region rendering remains a yellow fill with opacity 0.66.
- New plans use immutable snapshots and monotonically increasing revisions; an invalid or unavailable LLM keeps the last verified plan and records degradation.
- Operational frames never contain target truth; same seed plus the same deterministic LLM provider must produce the same plan, route, and replay bytes.
- UUV mileage includes transit to the region, internal scan/track motion, and travel to the recovery point; range/energy thresholds trigger return and replanning.
- A* treats task-region interiors as forbidden, keeps safety boundaries high-cost/forbidden, and validates the complete route back to the home battle group after every inserted stop.
- New work begins only after this plan is committed; implementation occurs on a new branch created from the merged master.

---

### Task 1: Establish the UUV-only configuration and legacy boundary

**Files:**
- Modify: src/underwater_tracking/config/models.py
- Modify: src/underwater_tracking/config/platform_core.py
- Modify: src/underwater_tracking/config/loader.py
- Modify: configs/scenario/default.yaml
- Create: configs/scenario/uuv_only_single_target.yaml
- Modify: src/underwater_tracking/domain/models.py
- Modify: src/underwater_tracking/domain/platforms.py
- Test: tests/agent/test_agent_loader.py
- Test: tests/config/test_loader.py
- Create: tests/config/test_uuv_only_config.py

**Interfaces:**
- UUVOnlyEnvironmentConfig exposes home_battle_group_id, carriers, uuvs, submarines, and map bounds; it has no required USV collection.
- AppConfig.uuv_only is True for the new scenario and false for legacy scenarios.
- load_app_config maps legacy singular carrier/usvs fields only for legacy mode and never injects them into a UUV-only runtime.
- PlatformRoster.uuvs remains available to tracking code; new UUV-only snapshots expose an empty USV collection at the compatibility boundary.

- [ ] Step 1: Add failing configuration tests and correct the stale merged assertion.

~~~python
def test_default_target_count_matches_single_target_design() -> None:
    config = load_app_config("configs/scenario/default.yaml")
    assert config.scenario.initial_target_count == 1
    assert config.scenario.max_target_count == 4

def test_uuv_only_config_rejects_usv_runtime_entries() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    assert config.scenario.uuv_only is True
    assert config.environment is not None
    assert config.environment.usvs == ()

def test_uuv_only_requires_one_home_battle_group_and_target() -> None:
    with pytest.raises(ValidationError, match="home battle group"):
        load_app_config(_write_uuv_only_config(home_battle_group_id=""))
~~~

Change the stale assertion in tests/agent/test_agent_loader.py from 2 to 1; do not change the intended single-target default back to 2.

- [ ] Step 2: Run the focused tests and record the pre-implementation failures.

~~~bash
PYTHONPATH=src python -m pytest tests/agent/test_agent_loader.py tests/config/test_loader.py tests/config/test_uuv_only_config.py -q
~~~

Expected before implementation: the new UUV-only model/loader symbols are missing; after only the stale assertion edit, existing loader tests pass and the new tests fail at the missing UUV-only contract.

- [ ] Step 3: Implement the explicit UUV-only schema and compatibility loader.

Use immutable tuples and strict validation. The effective invariant is:

~~~python
if config.scenario.uuv_only:
    if not config.environment or config.environment.usvs:
        raise ValueError("uuv-only scenario must not load USV entries")
    if len(config.environment.carriers) < 1:
        raise ValueError("uuv-only scenario requires at least one carrier")
    if len(config.environment.submarines) != 1:
        raise ValueError("phase one requires exactly one submarine")
~~~

Keep legacy carrier and usvs fields readable for old configs, but normalize them into an empty new-runtime USV tuple when uuv_only is true. Add the default scenario file with one target, at least two carriers, twelve UUVs, one home battle group, and deterministic map bounds.

- [ ] Step 4: Verify configuration behavior and style.

~~~bash
PYTHONPATH=src python -m pytest tests/agent/test_agent_loader.py tests/config/test_loader.py tests/config/test_uuv_only_config.py -q
ruff check src/underwater_tracking/config src/underwater_tracking/domain tests/config tests/agent/test_agent_loader.py
~~~

Expected: focused tests pass and Ruff reports no errors.

- [ ] Step 5: Commit.

~~~bash
git add src/underwater_tracking/config/models.py src/underwater_tracking/config/platform_core.py src/underwater_tracking/config/loader.py configs/scenario/default.yaml configs/scenario/uuv_only_single_target.yaml src/underwater_tracking/domain/models.py src/underwater_tracking/domain/platforms.py tests/agent/test_agent_loader.py tests/config/test_loader.py tests/config/test_uuv_only_config.py
git commit -m "feat: define uuv-only mission configuration boundary"
~~~

### Task 2: Add mission-domain state and compatibility views

**Files:**
- Create: src/underwater_tracking/domain/mission_models.py
- Modify: src/underwater_tracking/domain/regional_models.py
- Modify: src/underwater_tracking/domain/models.py
- Create: src/underwater_tracking/domain/mission_adapters.py
- Test: tests/domain/test_mission_models.py
- Test: tests/domain/test_regional_models.py

**Interfaces:**
- UUVMissionMode is one of ONBOARD, TRANSIT_TO_REGION, ACTIVE_SCAN, PASSIVE_TRACK, RETURN_REQUIRED, RECOVERING, FAILED.
- RegionLifecycle is one of PLANNED, CARRIER_DEPLOYING, ACTIVE_SCAN, PASSIVE_TRACK, HANDOFF_PENDING, TRACKING_COMPLETED, CARRIER_RECOVERY, RECOVERED, DEGRADED, UNCOVERED.
- PredictionGridCell carries target, revision, integer grid coordinates, square bounds, probability, entry/exit time, IMM model probabilities, covariance summary, intent evidence, and region ID.
- PredictionGrid carries target, revision, origin, cell size, sorted cells, and centerline region IDs.
- RegionalMissionViewModel and CarrierMissionModel are immutable executable-state models; LLM policy models remain candidate-only.
- legacy_frame_to_uuv_view(payload) accepts old usvs fields and returns a payload with those fields ignored.

- [ ] Step 1: Write model round-trip and invariant tests.

~~~python
def test_uuv_mode_and_region_lifecycle_are_closed_sets() -> None:
    assert UUVMissionMode("ACTIVE_SCAN") == "ACTIVE_SCAN"
    with pytest.raises(ValueError):
        UUVMissionMode("RELAY")

def test_prediction_grid_ids_are_stable_within_revision() -> None:
    grid = PredictionGrid(
        target_id="T1",
        revision=4,
        origin=(0.0, 0.0),
        cell_size_m=500.0,
        cells=(PredictionGridCell(
            target_id="T1",
            revision=4,
            grid_x=2,
            grid_y=-1,
            bounds=(-100.0, 900.0, -600.0, 400.0),
            probability=0.8,
            first_entry_s=30,
            last_exit_s=90,
            imm_model_probabilities={"CV": 0.6, "CT": 0.4},
            covariance_summary=(100.0, 100.0, 0.0),
            intent_label="transit",
            intent_confidence=0.8,
            region_id="T1:r4:cell:2:-1",
        ),),
        centerline_region_ids=("T1:r4:cell:2:-1",),
    )
    assert grid.cell("T1", 4, 2, -1).region_id == "T1:r4:cell:2:-1"

def test_legacy_usv_fields_are_ignored_but_new_view_has_none() -> None:
    view = legacy_frame_to_uuv_view({"uuvs": [], "usvs": [{"usv_id": "USV1"}]})
    assert view["uuvs"] == []
    assert "usvs" not in view
~~~

- [ ] Step 2: Run the tests to confirm missing contracts.

~~~bash
PYTHONPATH=src python -m pytest tests/domain/test_mission_models.py tests/domain/test_regional_models.py -q
~~~

- [ ] Step 3: Implement immutable mission models and adapter functions.

Use ConfigDict(extra="forbid", frozen=True), finite numeric fields, sorted tuple normalization, and explicit state-transition validation. Keep legacy RegionTask readable, but define new mission models without assigned USV IDs, USV roles, or relay requirements.

- [ ] Step 4: Run focused domain verification.

~~~bash
PYTHONPATH=src python -m pytest tests/domain/test_mission_models.py tests/domain/test_regional_models.py tests/domain/test_models.py -q
ruff check src/underwater_tracking/domain tests/domain
~~~

- [ ] Step 5: Commit.

~~~bash
git add src/underwater_tracking/domain/mission_models.py src/underwater_tracking/domain/regional_models.py src/underwater_tracking/domain/models.py src/underwater_tracking/domain/mission_adapters.py tests/domain/test_mission_models.py tests/domain/test_regional_models.py
git commit -m "feat: add immutable uuv mission domain state"
~~~

### Task 3: Build deterministic prediction grids and square candidate regions

**Files:**
- Create: src/underwater_tracking/planning/prediction_grid.py
- Modify: src/underwater_tracking/planning/regions.py
- Create: src/underwater_tracking/planning/candidate_regions.py
- Modify: src/underwater_tracking/config/models.py
- Test: tests/planning/test_prediction_grid.py
- Test: tests/planning/test_candidate_regions.py

**Interfaces:**
- build_prediction_grid(belief, prediction, intent, revision, config) returns PredictionGrid.
- generate_candidate_regions(grid, map_bounds_xy) returns a deterministic tuple of CandidateRegion.
- candidate_region_to_task(region, plan_revision) returns a validated regional mission candidate.
- cell size is clamp-and-round(sqrt(predicted_envelope_area / target_grid_cell_count), min_cell_size_m, max_cell_size_m).

- [ ] Step 1: Add failing grid and geometry tests.

Test that identical belief/prediction/revision inputs produce identical cell IDs, that high covariance or low confidence does not create a smaller cell size than low covariance/high confidence, that cells stay inside bounds, and that a candidate is a contiguous axis-aligned square whose side is an integer multiple of the grid cell size.

- [ ] Step 2: Run focused tests before implementation.

~~~bash
PYTHONPATH=src python -m pytest tests/planning/test_prediction_grid.py tests/planning/test_candidate_regions.py -q
~~~

- [ ] Step 3: Implement grid evidence and candidate generation.

Use only TargetBelief, prediction points/corridor, IMM model probabilities, and intent evidence. Store probability, entry/exit windows, covariance summary, and region linkage on each cell. Enumerate square windows in deterministic (side, start_grid_x, start_grid_y, first_entry_s) order; reject candidates outside the map, disconnected cells, missing time windows, or missing perimeter deployment/recovery points.

The generator must never accept an arbitrary LLM coordinate. The only valid region reference is a deterministic candidate ID present in the generated tuple.

- [ ] Step 4: Verify geometry and property invariants.

~~~bash
PYTHONPATH=src python -m pytest tests/planning/test_prediction_grid.py tests/planning/test_candidate_regions.py tests/planning/test_regional_standoff.py -q
ruff check src/underwater_tracking/planning src/underwater_tracking/config tests/planning
~~~

- [ ] Step 5: Commit.

~~~bash
git add src/underwater_tracking/planning/prediction_grid.py src/underwater_tracking/planning/regions.py src/underwater_tracking/planning/candidate_regions.py src/underwater_tracking/config/models.py tests/planning/test_prediction_grid.py tests/planning/test_candidate_regions.py
git commit -m "feat: build deterministic prediction grids and candidate regions"
~~~

### Task 4: Restrict regional LLM strategy to validated UUV candidates

**Files:**
- Modify: src/underwater_tracking/domain/regional_models.py
- Create: src/underwater_tracking/planning/regional_plan_validator.py
- Modify: src/underwater_tracking/agent/prompts.py
- Modify: src/underwater_tracking/agent/nodes/regions.py
- Modify: src/underwater_tracking/agent/nodes/regional_strategy.py
- Modify: src/underwater_tracking/agent/graphs/central.py
- Test: tests/planning/test_regional_plan_validator.py
- Test: tests/agent/test_regional_strategy.py
- Modify: tests/agent/test_regional_plan_pipeline.py

**Interfaces:**
- RegionalMissionCandidate contains only a candidate ID, grid cells, time window, and deterministic perimeter points.
- validate_uuv_strategy(candidate_set, strategy, available_uuv_ids) returns ValidatedRegionalStrategy and rejects unknown regions, unknown UUV IDs, USV fields, coordinates outside candidates, invalid counts, bad handoff references, and missing evidence.
- RegionalStrategyGenerationNode receives candidate regions and emits a candidate policy; it never mutates SituationSnapshot or platform state.

- [ ] Step 1: Add failing validator and prompt-contract tests.

~~~python
def test_strategy_rejects_unknown_uuv_and_region() -> None:
    with pytest.raises(RegionalPlanError, match="unknown region"):
        validate_uuv_strategy(candidates, strategy_with_region("T1:r9"), {"U1"})
    with pytest.raises(RegionalPlanError, match="unknown UUV"):
        validate_uuv_strategy(candidates, strategy_with_uuv("U99"), {"U1"})

def test_strategy_payload_contains_no_usv_candidates() -> None:
    payload = node.build_payload(snapshot, plan, intents)
    assert all(item["kind"] == "uuv" for item in payload["platform_candidates"])
    assert "assigned_usv_ids" not in json.dumps(payload)
~~~

- [ ] Step 2: Run the focused tests to confirm the old mixed-domain contract fails.

~~~bash
PYTHONPATH=src python -m pytest tests/planning/test_regional_plan_validator.py tests/agent/test_regional_strategy.py tests/agent/test_regional_plan_pipeline.py -q
~~~

- [ ] Step 3: Implement the UUV-only policy schema and deterministic validator.

Replace the regional prompt's allowed modes with active_scan, passive_track, and handoff_reserve; expose candidate IDs and UUV capability/resource fields only. Keep the old mixed-domain parser solely behind the legacy replay/compatibility path. Validation must check assigned_uuv_ids is a subset of available_uuv_ids, active_scan capability, non-overlapping windows, predecessor/successor candidate membership, and evidence IDs.

- [ ] Step 4: Verify LLM, fallback, and atomic-candidate behavior.

~~~bash
PYTHONPATH=src python -m pytest tests/planning/test_regional_plan_validator.py tests/agent/test_regional_strategy.py tests/agent/test_regional_plan_pipeline.py tests/agent/test_central_graph.py -q
ruff check src/underwater_tracking/agent src/underwater_tracking/planning src/underwater_tracking/domain tests/agent tests/planning
~~~

Expected: an unavailable or malformed LLM produces a recorded degradation and does not mutate the currently active plan.

- [ ] Step 5: Commit.

~~~bash
git add src/underwater_tracking/domain/regional_models.py src/underwater_tracking/planning/regional_plan_validator.py src/underwater_tracking/agent/prompts.py src/underwater_tracking/agent/nodes/regions.py src/underwater_tracking/agent/nodes/regional_strategy.py src/underwater_tracking/agent/graphs/central.py tests/planning/test_regional_plan_validator.py tests/agent/test_regional_strategy.py tests/agent/test_regional_plan_pipeline.py
git commit -m "feat: validate uuv-only regional strategy candidates"
~~~

### Task 5: Implement rolling UUV batch optimization and reserve protection

**Files:**
- Create: src/underwater_tracking/planning/mission_optimizer.py
- Modify: src/underwater_tracking/planning/allocation.py
- Modify: src/underwater_tracking/domain/mission_models.py
- Modify: src/underwater_tracking/agent/nodes/optimize.py
- Test: tests/planning/test_mission_optimizer.py
- Modify: tests/planning/test_regional_allocation.py
- Modify: tests/agent/test_central_graph.py

**Interfaces:**
- MissionOptimizer.optimize(snapshot, candidates) returns ExecutableMissionPlan.
- required_active_uuvs(region, snapshot) and required_passive_uuvs(region, snapshot) are deterministic and do not read truth.
- evaluate_batch(plan, batch, future_requirements) returns coverage, FIM/tracking benefit, energy, delay, route, churn, and reserve feasibility.
- ExecutableMissionPlan includes uuv_batches_by_carrier, reserved_uuv_ids, region_assignments, carrier_missions, and revision.

- [ ] Step 1: Write failing optimizer tests.

Cover current region minimum passive scale, future high-probability region reserve, rejecting a larger batch that makes the next region infeasible, preferring a higher total marginal benefit over immediate coverage, and emitting DEGRADED/UNCOVERED without fabricating UUV IDs.

~~~python
result = optimizer.optimize(snapshot, candidates)
assert result.reserved_uuv_ids == ("U05", "U06")
assert result.batches[0].uuv_ids == ("U01", "U02")
assert "U99" not in result.all_uuv_ids
~~~

- [ ] Step 2: Run the focused tests to confirm no rolling optimizer exists.

~~~bash
PYTHONPATH=src python -m pytest tests/planning/test_mission_optimizer.py tests/planning/test_regional_allocation.py -q
~~~

- [ ] Step 3: Implement deterministic marginal-gain selection.

At each revision, calculate current/future minimum demand, enumerate sorted feasible batch prefixes, reject any prefix that violates future reserve, and maximize:

~~~text
coverage_probability + active_scan_coverage + passive_fim_quality
+ handoff_continuity + tracking_available_time
- carrier_distance - uuv_energy - deploy_delay - churn - reserve_consumption
~~~

Tie-break by hard violations, lower cost, earlier window, batch size, then sorted UUV IDs. Preserve manually locked UUV-target assignments as hard constraints.

- [ ] Step 4: Verify optimizer behavior and graph integration.

~~~bash
PYTHONPATH=src python -m pytest tests/planning/test_mission_optimizer.py tests/planning/test_regional_allocation.py tests/agent/test_central_graph.py -q
ruff check src/underwater_tracking/planning src/underwater_tracking/agent/nodes/optimize.py tests/planning
~~~

- [ ] Step 5: Commit.

~~~bash
git add src/underwater_tracking/planning/mission_optimizer.py src/underwater_tracking/planning/allocation.py src/underwater_tracking/domain/mission_models.py src/underwater_tracking/agent/nodes/optimize.py tests/planning/test_mission_optimizer.py tests/planning/test_regional_allocation.py tests/agent/test_central_graph.py
git commit -m "feat: optimize rolling uuv batches with future reserves"
~~~

### Task 6: Add multi-stop carrier tasks, Hungarian slots, and complete A* return validation

**Files:**
- Create: src/underwater_tracking/planning/astar.py
- Create: src/underwater_tracking/planning/hungarian.py
- Create: src/underwater_tracking/planning/carrier_tasks.py
- Modify: src/underwater_tracking/simulation/carrier.py
- Modify: src/underwater_tracking/domain/mission_models.py
- Test: tests/planning/test_astar.py
- Test: tests/planning/test_hungarian.py
- Test: tests/planning/test_carrier_tasks.py
- Modify: tests/simulation/test_carrier.py

**Interfaces:**
- AStarRoutePlanner.plan(start, stops, home, forbidden_regions, map_bounds) returns RoutePlan or None and always includes home as the final point.
- CarrierTaskPlanner.build_tasks(plan, carriers) creates deployment/recovery stops on region perimeters.
- HungarianMatcher.match(tasks, virtual_service_slots) returns deterministic CarrierSlotAssignment values, with slots such as carrier_01.slot_1.
- A carrier remains matchable while TO_DEPLOY, DEPLOYING, EN_ROUTE_NEXT_DEPLOY, or RETURNING_TO_FLEET only when it has ready UUVs, is healthy, can insert the task, and still returns home.

- [ ] Step 1: Write route/matching failure tests.

Assert that a route through a region interior is rejected, an inserted third stop is revalidated against the full route to home, an impossible slot falls back to another carrier or yields no assignment, capacities reduce ready_uuv_count, and every route begins/ends at the same home point.

- [ ] Step 2: Run focused tests before implementation.

~~~bash
PYTHONPATH=src python -m pytest tests/planning/test_astar.py tests/planning/test_hungarian.py tests/planning/test_carrier_tasks.py tests/simulation/test_carrier.py -q
~~~

- [ ] Step 3: Implement grid routing and stable matching.

Use integer map cells and deterministic neighbor order [(0, -1), (-1, 0), (1, 0), (0, 1)]. Mark region interiors forbidden, safety rings as forbidden/high cost, and map boundaries/dynamic obstacles as forbidden. Recompute start -> all committed stops -> home on insertion. Build the Hungarian cost matrix from incremental A* distance, ETA/time-window slack, required UUV count, ready inventory, return distance, and future reserve loss; reject any assignment whose route validator fails after matching.

- [ ] Step 4: Verify multi-stop and capacity behavior.

~~~bash
PYTHONPATH=src python -m pytest tests/planning/test_astar.py tests/planning/test_hungarian.py tests/planning/test_carrier_tasks.py tests/simulation/test_carrier.py -q
ruff check src/underwater_tracking/planning src/underwater_tracking/simulation/carrier.py tests/planning tests/simulation/test_carrier.py
~~~

- [ ] Step 5: Commit.

~~~bash
git add src/underwater_tracking/planning/astar.py src/underwater_tracking/planning/hungarian.py src/underwater_tracking/planning/carrier_tasks.py src/underwater_tracking/simulation/carrier.py src/underwater_tracking/domain/mission_models.py tests/planning/test_astar.py tests/planning/test_hungarian.py tests/planning/test_carrier_tasks.py tests/simulation/test_carrier.py
git commit -m "feat: plan multi-stop carrier deployment and recovery routes"
~~~

### Task 7: Make MissionController the lifecycle source and integrate UUV-only physics

**Files:**
- Create: src/underwater_tracking/runtime/mission_controller.py
- Modify: src/underwater_tracking/runtime/run_controller.py
- Modify: src/underwater_tracking/simulation/engine.py
- Modify: src/underwater_tracking/simulation/uuv.py
- Modify: src/underwater_tracking/simulation/carrier.py
- Modify: src/underwater_tracking/cli.py
- Modify: src/underwater_tracking/api/live.py
- Test: tests/runtime/test_mission_controller.py
- Create: tests/integration/test_uuv_only_mission.py
- Modify: tests/simulation/test_active_sonar.py
- Modify: tests/simulation/test_carrier.py

**Interfaces:**
- MissionController.snapshot() returns an immutable MissionSnapshot.
- MissionController.apply_verified_plan(plan) atomically replaces the executable plan only when plan.revision is greater than the current revision.
- MissionController.advance(sim_time_s, observations) drives transitions and emits RuntimeEvent values.
- MissionController.events covers target_intent_changed, imm_confidence_shifted, target_entered_region, target_exit_predicted, handoff_completed, uuv_range_exhausted, uuv_energy_depleted, uuv_failed, region_coverage_degraded, carrier_dispatch_completed, and carrier_recovery_completed.

- [ ] Step 1: Add failing lifecycle and integration tests.

Use a fixed seed and deterministic LLM provider. Assert: carrier starts at home, deploys to at least three regions, UUVs enter ACTIVE_SCAN, target-entry confirmation switches the region to PASSIVE_TRACK, handoff prepares the successor before the current region closes, exhausted mileage creates RETURN_REQUIRED and a recovery task, and no operational snapshot/log contains usv keys or USV IDs.

- [ ] Step 2: Run the focused tests before the controller exists.

~~~bash
PYTHONPATH=src python -m pytest tests/runtime/test_mission_controller.py tests/integration/test_uuv_only_mission.py -q
~~~

- [ ] Step 3: Implement controller transitions and engine boundary.

Move region state, UUV mode, inventory, carrier stops, event ledger, and plan application into MissionController. Keep SimulationEngine.step() responsible for moving UUV/carriers and producing observations only; call controller advancement at the observation boundary. Do not call LLMs or mutate controller state from the physics loop. For old legacy engine scenarios, preserve the adapter path and route only the uuv_only scenario through the mission controller.

The transition guard is explicit:

~~~python
if entry_probability >= config.region_entry_probability_threshold:
    region.entry_confirmations += 1
if region.entry_confirmations >= config.region_transition_confirm_cycles:
    region.lifecycle = "PASSIVE_TRACK"
    region.sensor_mode = "passive"
~~~

At mileage/energy exhaustion, emit one idempotent event, change mode to RETURN_REQUIRED, enqueue recovery, and request a new immutable planning snapshot.

- [ ] Step 4: Verify deterministic mission integration.

~~~bash
PYTHONPATH=src python -m pytest tests/runtime/test_mission_controller.py tests/integration/test_uuv_only_mission.py tests/simulation/test_active_sonar.py tests/simulation/test_carrier.py -q
PYTHONPATH=src ruff check src/underwater_tracking/runtime src/underwater_tracking/simulation src/underwater_tracking/api/live.py tests/runtime tests/integration
~~~

- [ ] Step 5: Commit.

~~~bash
git add src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/runtime/run_controller.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/simulation/uuv.py src/underwater_tracking/simulation/carrier.py src/underwater_tracking/cli.py src/underwater_tracking/api/live.py tests/runtime/test_mission_controller.py tests/integration/test_uuv_only_mission.py tests/simulation/test_active_sonar.py tests/simulation/test_carrier.py
git commit -m "feat: orchestrate uuv-only mission lifecycle"
~~~

### Task 8: Publish prediction, mission, carrier, event, and legacy-compatible frame views

**Files:**
- Modify: src/underwater_tracking/domain/ui_models.py
- Modify: src/underwater_tracking/api/frame_builder.py
- Create: src/underwater_tracking/api/legacy_frame_adapter.py
- Modify: src/underwater_tracking/api/frame_logger.py
- Modify: src/underwater_tracking/api/replay.py
- Modify: src/underwater_tracking/api/app.py
- Modify: src/underwater_tracking/runtime/run_catalog.py
- Test: tests/api/test_uuv_only_frame_contract.py
- Modify: tests/api/test_frame_pipeline.py
- Modify: tests/api/test_app.py
- Create: tests/api/test_replay_compatibility.py

**Interfaces:**
- PredictionGridView exposes target, revision, origin, cell size, and sorted cell evidence.
- RegionalMissionView exposes square geometry, lifecycle, time window, active/passive/reserve UUV IDs, coverage, tracking quality, handoff, and carrier task.
- CarrierMissionView exposes home ID, mission type, route status, stops, route, and onboard/ready/reserved/recoverable counts.
- OperationalFrame accepts old usvs during validation but the new frame serializer omits empty legacy USV fields.
- read_legacy_frame(payload) returns an OperationalFrame while ignoring old USV fields.

- [ ] Step 1: Add failing frame and replay contract tests.

~~~python
def test_new_uuv_only_frame_has_no_usv_payload() -> None:
    payload = build_uuv_only_frame(
        snapshot=uuv_only_snapshot_fixture(),
        mission=mission_fixture(),
        events=(),
    ).model_dump(mode="json")
    assert "usvs" not in payload
    assert "USV" not in json.dumps(payload)

def test_legacy_frame_reads_and_ignores_usv_fields() -> None:
    frame = read_legacy_frame({**legacy_payload, "usvs": [{"usv_id": "USV1"}]})
    assert frame.uuvs
    assert "usvs" not in frame.model_dump(mode="json")
~~~

- [ ] Step 2: Run focused tests before frame changes.

~~~bash
PYTHONPATH=src python -m pytest tests/api/test_uuv_only_frame_contract.py tests/api/test_frame_pipeline.py tests/api/test_replay_compatibility.py tests/api/test_app.py -q
~~~

- [ ] Step 3: Implement the frame projection and serialization boundary.

Build views exclusively from MissionSnapshot, SituationSnapshot, belief/prediction, and event ledger. Use fixed map layer order: probability grid, trajectory centerline/direction, yellow rgba(245, 194, 64, 0.66) region squares, UUV scan/passive geometry, carrier A* route, handoff/recovery edges, and event markers. Serialize new frames with an operational_frame_payload() helper that removes legacy USV fields when their collection is empty; replay validation still accepts old JSONL fields and discards them.

- [ ] Step 4: Verify API, replay, and frame contracts.

~~~bash
PYTHONPATH=src python -m pytest tests/api/test_uuv_only_frame_contract.py tests/api/test_frame_pipeline.py tests/api/test_replay_compatibility.py tests/api/test_app.py -q
ruff check src/underwater_tracking/api src/underwater_tracking/domain/ui_models.py tests/api
~~~

- [ ] Step 5: Commit.

~~~bash
git add src/underwater_tracking/domain/ui_models.py src/underwater_tracking/api/frame_builder.py src/underwater_tracking/api/legacy_frame_adapter.py src/underwater_tracking/api/frame_logger.py src/underwater_tracking/api/replay.py src/underwater_tracking/api/app.py src/underwater_tracking/runtime/run_catalog.py tests/api/test_uuv_only_frame_contract.py tests/api/test_frame_pipeline.py tests/api/test_replay_compatibility.py tests/api/test_app.py
git commit -m "feat: publish uuv mission and carrier operational views"
~~~

### Task 9: Render probability grids, UUV modes, carrier routes, and mission handoffs

**Files:**
- Modify: src/underwater_tracking/ui/src/types/frames.ts
- Modify: src/underwater_tracking/ui/src/frameTypes.ts
- Modify: src/underwater_tracking/ui/src/components/CanvasMap.tsx
- Modify: src/underwater_tracking/ui/src/components/map/RegionOverlay.tsx
- Modify: src/underwater_tracking/ui/src/components/RegionTimelinePanel.tsx
- Modify: src/underwater_tracking/ui/src/components/CarrierStatusPanel.tsx
- Modify: src/underwater_tracking/ui/src/components/BottomDrawer.tsx
- Modify: src/underwater_tracking/ui/src/App.css
- Test: src/underwater_tracking/ui/src/components/CanvasMap.test.ts
- Test: src/underwater_tracking/ui/src/components/map/RegionOverlay.test.tsx
- Test: src/underwater_tracking/ui/src/components/CarrierStatusPanel.test.tsx
- Test: src/underwater_tracking/ui/src/components/RegionTimelinePanel.test.tsx
- Test: src/underwater_tracking/ui/src/e2e/visualCommandCenterFlow.test.ts

**Interfaces:**
- PredictionGridView, RegionalMissionView, CarrierMissionView, and MissionEventView TypeScript types mirror backend JSON exactly.
- drawPredictionGrid() maps probability to evidence color without changing the yellow region semantic.
- drawCarrierRoute() draws each committed stop and the final home return; drawUuvMissionMode() distinguishes active scan, passive track, return, recovering, and failed.

- [ ] Step 1: Add failing TypeScript contract and rendering tests.

Assert that a high-probability cell is visually stronger than a low-probability cell, every region remains yellow at opacity 0.66, active scan and passive track use distinct legends, the carrier route ends at home, reserve UUVs appear in the inventory panel, and legacy frames with USV fields do not render USV markers.

- [ ] Step 2: Run the focused frontend tests before implementation.

~~~bash
npm --prefix src/underwater_tracking/ui test -- --run src/components/CanvasMap.test.ts src/components/map/RegionOverlay.test.tsx src/components/CarrierStatusPanel.test.tsx src/components/RegionTimelinePanel.test.tsx
~~~

- [ ] Step 3: Implement map and panel projections.

Use frame data only; do not infer platform state from DOM or target truth. Keep minimum pointer hit areas unchanged while reducing visual marker size where needed. Show route stops, carrier counts, region lifecycle, handoff predecessor/successor, current plan revision, event reasons, and “沿用上一版计划” when LLM degradation is active.

- [ ] Step 4: Run all frontend tests, build, and browser acceptance.

~~~bash
npm --prefix src/underwater_tracking/ui test -- --run
npm --prefix src/underwater_tracking/ui run build
npm --prefix src/underwater_tracking/ui run test:e2e -- --grep "uuv-only|mission|replay"
~~~

- [ ] Step 5: Commit.

~~~bash
git add src/underwater_tracking/ui/src/types/frames.ts src/underwater_tracking/ui/src/frameTypes.ts src/underwater_tracking/ui/src/components/CanvasMap.tsx src/underwater_tracking/ui/src/components/map/RegionOverlay.tsx src/underwater_tracking/ui/src/components/RegionTimelinePanel.tsx src/underwater_tracking/ui/src/components/CarrierStatusPanel.tsx src/underwater_tracking/ui/src/components/BottomDrawer.tsx src/underwater_tracking/ui/src/App.css src/underwater_tracking/ui/src/components/CanvasMap.test.ts src/underwater_tracking/ui/src/components/map/RegionOverlay.test.tsx src/underwater_tracking/ui/src/components/CarrierStatusPanel.test.tsx src/underwater_tracking/ui/src/components/RegionTimelinePanel.test.tsx src/underwater_tracking/ui/src/e2e/visualCommandCenterFlow.test.ts
git commit -m "feat: visualize uuv mission regions and carrier routes"
~~~

### Task 10: Complete deterministic integration, replay, and failure-path acceptance

**Files:**
- Create: tests/integration/test_uuv_only_mission_acceptance.py
- Create: tests/api/test_uuv_only_replay_acceptance.py
- Modify: tests/integration/test_platform_core_scenario.py
- Modify: tests/simulation/test_false_alarm.py
- Modify: tests/agent/test_replan_latch.py
- Modify: tests/api/test_frame_pipeline.py
- Modify: docs/audit-hyperparameters.md
- Create: docs/superpowers/audits/2026-08-19-uuv-only-carrier-region-mission-verification.md

**Interfaces:**
- run_uuv_only_acceptance(seed) returns an AcceptanceTrace containing plan revisions, region states, UUV modes, carrier stops, route validity, events, and frame payload hashes.
- assert_uuv_only_acceptance(trace) is the single end-to-end completion assertion used by pytest and the audit record.

- [ ] Step 1: Add the fixed-seed acceptance test and failure-path cases.

The trace must prove all of the following in one deterministic scenario: one target; at least two carriers; one carrier visits at least three deployment points and returns home; no route crosses a region; active scan precedes passive track; entry threshold requires two consecutive confirmations; handoff activates the successor before closing the predecessor; mileage triggers recovery/replanning; intent/confidence changes create a new grid revision; insufficient resources leave a region degraded/uncovered; malformed/failed LLM output retains the previous plan; old replay with USV fields loads; new frame and JSONL payloads contain no USV data; same seed/provider produces identical normalized trace and route.

- [ ] Step 2: Run the targeted acceptance test before the final full suite.

~~~bash
PYTHONPATH=src python -m pytest tests/integration/test_uuv_only_mission_acceptance.py tests/api/test_uuv_only_replay_acceptance.py -q
~~~

- [ ] Step 3: Add deterministic route/frame hash and event-order assertions.

Normalize only stable JSON keys and compare:

~~~python
assert trace_a.plan_hashes == trace_b.plan_hashes
assert trace_a.route_hashes == trace_b.route_hashes
assert trace_a.frame_hashes == trace_b.frame_hashes
assert trace_a.event_types == tuple(sorted(trace_a.event_types, key=trace_a.event_order.__getitem__))
~~~

Ensure event IDs include plan revision and simulation time, and repeated exhaustion/replan events are latched idempotently.

- [ ] Step 4: Run the complete verification matrix.

~~~bash
PYTHONPATH=src python -m pytest -q
ruff check src tests
python -m compileall -q src tests
npm --prefix src/underwater_tracking/ui test -- --run
npm --prefix src/underwater_tracking/ui run build
npm --prefix src/underwater_tracking/ui run test:e2e
git diff --check
git status --short --branch
~~~

Expected: the only known baseline correction is the stale default-target assertion fixed in Task 1; all backend/frontend tests, build, e2e checks, and diff checks pass.

- [ ] Step 5: Record verification evidence and commit the audit.

~~~bash
git add tests/integration/test_uuv_only_mission_acceptance.py tests/api/test_uuv_only_replay_acceptance.py tests/integration/test_platform_core_scenario.py tests/simulation/test_false_alarm.py tests/agent/test_replan_latch.py tests/api/test_frame_pipeline.py docs/audit-hyperparameters.md docs/superpowers/audits/2026-08-19-uuv-only-carrier-region-mission-verification.md
git commit -m "test: verify uuv-only carrier region mission end to end"
~~~

## Plan Self-Review Checklist

- The design platform boundary is covered by Tasks 1, 7, 8, and 9.
- Prediction-grid geometry, stable IDs, probability evidence, and square candidate constraints are covered by Tasks 2 and 3.
- LLM candidate-only behavior and deterministic validation are covered by Task 4.
- Batch marginal optimization and future reserve constraints are covered by Task 5.
- Multi-stop carrier routes, capacity, Hungarian virtual slots, A* forbidden regions, and mandatory home return are covered by Task 6.
- Lifecycle, sensor switching, handoff, mileage, energy, failure, and event-driven replanning are covered by Task 7 and Task 10.
- Operational frames, old replay compatibility, new-frame USV omission, API contracts, and map layers are covered by Tasks 8 and 9.
- Deterministic integration, backend/API/frontend/e2e verification, and audit evidence are covered by Task 10.
- There are no placeholder implementation steps; every task names exact files, interfaces, tests, commands, and commit boundaries.
