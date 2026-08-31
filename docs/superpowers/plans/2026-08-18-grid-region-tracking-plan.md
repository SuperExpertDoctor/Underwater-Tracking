# Grid-Region Tracking Implementation Plan

> Progress: Tasks 1-4 implementation slices and the Task 5 compatibility derivation are committed on `feature/grid-region-tracking`. Runtime pytest execution is pending because the remote `.venv` lacks `pytest`, `pydantic`, and the compatible x86_64 dependency wheels; static compilation and diff checks pass.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make spatial square regions the primary tracking-plan contract while preserving target/segment compatibility views and real LangGraph/LLM ownership.

**Architecture:** Add strict regional Pydantic contracts beside the existing target contracts. Deterministic prediction-side code computes adaptive axis-aligned cells and temporal visits; the real regional strategy LLM chooses bounded policies for generated IDs; deterministic validation/allocation derives concrete UUV/USV roles, links, sonar modes, handoffs, and legacy views. Graph, persistence, replay, and UI migrate through explicit regional fields.

**Tech Stack:** Python 3.11, Pydantic v2, LangGraph, NumPy/SciPy, SQLite JSON payloads, React/TypeScript/Vite, Vitest, Playwright.

## Constraints

- Work on `feature/grid-region-tracking` in `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/.worktrees/grid-region-tracking`.
- Keep `Segment`, `SegmentPlan`, `member_ids_by_target`, and `waypoints_by_member` as derived compatibility views.
- Every production behavior change follows failing test, minimal implementation, focused green test, then commit.
- Regional LLM input may include generated IDs, geometry, intent, prediction, and operational evidence, but never simulator truth, platform IDs, arbitrary coordinates, or infeasible modes.
- A second structured-content failure pauses through the existing error path; it must not be replaced by a deterministic strategy.

## Task 1: Domain contracts and configuration

**Files:** Create `src/underwater_tracking/domain/regional_models.py`, `tests/domain/test_regional_models.py`, and `tests/config/test_regional_config.py`. Modify `src/underwater_tracking/domain/agent_models.py`, `src/underwater_tracking/domain/__init__.py`, `src/underwater_tracking/config/models.py`, and `configs/tracking.yaml`.

- [x] Write failing tests for `GridSpec`, `TimeWindow`, `RegionCell`, `SonarPolicy`, `CommunicationRequirement`, `RegionTask`, `TargetRegionPlan`, and config round trips.

```python
def test_region_id_is_derived_from_integer_grid_coordinates() -> None:
    cell = RegionCell(
        region_id="T1:cell:2:-1", target_id="T1", grid_x=2, grid_y=-1,
        min_x=200.0, max_x=300.0, min_y=-100.0, max_y=0.0,
        center_xy=(250.0, -50.0), cell_size_m=100.0,
        first_entry_s=120, last_exit_s=180,
    )
    assert cell.region_id == "T1:cell:2:-1"

def test_region_task_requires_passive_sonar() -> None:
    with pytest.raises(ValidationError, match="passive"):
        RegionTask(**task_payload(sonar_policy={"passive_required": False}))

def test_regional_plan_round_trip_preserves_evidence() -> None:
    plan = make_single_region_plan()
    assert TargetRegionPlan.model_validate_json(plan.model_dump_json()) == plan
```

  Run: `.venv/bin/python -m pytest tests/domain/test_regional_models.py tests/config/test_regional_config.py -q`.
  Expected: FAIL because the regional contracts do not exist.
- [x] Implement strict `GridSpec` with origin, coordinate convention, target cell count, min/max/rounding size, lateral width, uncertainty margin, UUV/USV requirements, and relay overlap policy. Implement square `RegionCell`, ordered `TimeWindow`, role/sonar/communication contracts, `RegionTask`, `RegionalPolicy`, `RegionalStrategySet`, and `TargetRegionPlan`. Enforce deterministic IDs, positive bounds, disjoint visit windows, and mandatory passive sonar.
- [x] Add `regional_plans` and `region_tasks` to `TrackingPlan`, plus a helper deriving legacy member/role views. Add `TrackingConfig.grid` and YAML defaults. Export public contracts.
- [ ] Run the focused tests and `ruff check` on changed Python files. Expected: PASS.
- [x] Commit: `git commit -m "feat: add regional tracking contracts"`.

## Task 2: Deterministic adaptive rasterization

**Files:** Create `src/underwater_tracking/planning/regions.py` and `tests/planning/test_regions.py`.

- [x] Write failing tests for adaptive size calculation, clamping/rounding, stable cells, mandatory -2..+2 normal offsets, corridor widening, clipping, deduplication, multiple visit windows, temporal adjacency, and fallback evidence.

```python
def test_cell_size_uses_area_then_clamps_and_rounds() -> None:
    spec = GridSpec(target_grid_cells=16, min_cell_size_m=100.0,
                    max_cell_size_m=400.0, cell_size_rounding_m=50.0)
    assert compute_cell_size(10_000.0, spec) == 100.0
    assert compute_cell_size(10_000_000.0, spec) == 400.0
    assert compute_cell_size(90_000.0, spec) == 250.0

def test_straight_prediction_contains_the_mandatory_lateral_band() -> None:
    plan = generate_target_region_plan(straight_prediction(), intent(), bounds(), fixed_spec())
    keys = {(cell.grid_x, cell.grid_y) for cell in plan.cells}
    assert {(1, -2), (1, -1), (1, 0), (1, 1), (1, 2)} <= keys
```

  Run: `.venv/bin/python -m pytest tests/planning/test_regions.py -q`.
  Expected: FAIL because `planning.regions` does not exist.
- [x] Implement `compute_cell_size(envelope_area_m2, grid_spec)`, `generate_target_region_plan(prediction, intent, map_bounds_xy, grid_spec)`, and `rectangles_overlap(left, right)`. Estimate the clipped envelope from centerline bounds plus corridor radius, use `sqrt(area / target_grid_cells)`, clamp, and round.
- [x] Rasterize with `floor((coordinate - origin) / cell_size)`. Estimate tangents from adjacent samples, add normal offsets for the configured lateral width, add corridor-intersected cells within the uncertainty margin, retain only complete squares inside map bounds, deduplicate coordinates, and calculate visit windows from sample times. Sort by `(first_entry_s, grid_x, grid_y)` and link only temporal neighbors. Build one default passive `RegionTask` per cell and propagate prediction/intent evidence and fallback.
- [x] Run static compilation and `git diff --check`; runtime focused tests remain blocked by the remote dependency environment. Commit `feat: generate deterministic regional grid plans`.

## Task 3: Regional strategy LLM node

**Files:** Create `src/underwater_tracking/agent/nodes/regional_strategy.py`, `src/underwater_tracking/agent/nodes/regions.py`, and `tests/agent/test_regional_strategy.py`. Modify `src/underwater_tracking/agent/prompts.py` and `src/underwater_tracking/agent/state.py`.

- [x] Write failing tests that payloads contain every region and evidence, contain no platform IDs/arbitrary coordinates, reject unknown/duplicate/missing policies, and retry `LLMContentError` exactly once.

```python
def test_payload_is_region_scoped_and_has_no_platform_ids() -> None:
    payload = node.build_payload(snapshot, target_region_plan, {"T1": hypothesis})
    assert [item["region_id"] for item in payload["regions"]] == ["T1:cell:0:0"]
    assert "uuv_ids" not in json.dumps(payload)

def test_policy_validation_requires_exactly_one_policy_per_region() -> None:
    with pytest.raises(ValueError, match="missing regional policy"):
        validate_regional_strategy(target_region_plan, RegionalStrategySet(policies=()))
```

- [x] Implement `RegionalStrategyGenerationNode.build_payload` and `__call__`, exact-one-policy validation, regional prompt/version, request/response hashes, and the existing bounded correction behavior. The policy schema may set priority, quality, coverage mode, role counts, sonar policy, communication requirements, handoff IDs, rationale, and evidence only.
- [x] Implement the graph adapter that reads predictions/intents, invokes deterministic region generation, stores `TargetRegionPlan` values, then invokes the real regional strategy node. Run static compilation; runtime focused tests remain blocked. Commit `feat: add regional strategy LLM contract`.

## Task 4: Regional validation and allocation

**Files:** Create `src/underwater_tracking/planning/regional_validation.py`, `src/underwater_tracking/planning/regional_allocation.py`, `tests/planning/test_regional_validation.py`, and `tests/planning/test_regional_allocation.py`.

- [x] Write failing tests for overlap/bounds/time ordering, passive sonar, active policy/capability, carrier support radius, communication paths, double booking, allowed relay overlap, and explicit degradation.

```python
def test_far_usv_is_rejected() -> None:
    assert "usv_outside_carrier_radius" in validate_regional_plan(plan_with_far_usv(), roster())

def test_missing_platform_degrades_region_instead_of_dropping_it() -> None:
    result = allocate_regional_tasks(tasks_requiring_two_uuvs(), roster_with_one_uuv())
    assert result.tasks["T1:cell:0:0"].assignment_status == "degraded"
    assert result.tasks["T1:cell:0:0"].degraded_reasons
```

- [x] Implement candidate filtering by kind/capability, time-to-region, kinematics, range/endurance, deployment, carrier radius, links, sonar cooldown, and reservations. Assign in order: current passive coverage, next-region handoff reserve, communication path, authorized active verification, then rotations. Use deterministic score `(-coverage, travel, energy, churn, relay_instability, platform_id)`.
- [x] Return regional tasks, role assignments, sonar/link decisions, handoff metadata, derived waypoints, and machine-readable degraded/uncovered reasons. Run static compilation; runtime focused tests remain blocked. Commit `feat: allocate regional roles and validate links`.

## Task 5: Regional plan pipeline and compatibility views

**Files:** Modify `src/underwater_tracking/agent/nodes/optimize.py`, `verify.py`, `commit.py`, and `src/underwater_tracking/domain/agent_models.py`. Create `tests/agent/test_regional_plan_pipeline.py`.

- [x] Write failing tests proving regional tasks derive `member_ids_by_target`, `roles_by_member`, and `waypoints_by_member`, while degraded tasks remain present and do not become a target-only plan.
- [ ] Make `OptimizeNode` consume regional plans/policies and call `allocate_regional_tasks`. Construct `TrackingPlan.region_tasks` first, then derive legacy fields. Add regional degradation and relay metrics without removing existing metrics.
- [ ] Make plan verification invoke regional validation before old allocation/waypoint checks. Add optional `PlanCommand.region_id`; preserve `group_id` and all old execution fields. Commit `feat: derive legacy plans from regional tasks` after focused regional and existing plan-pipeline tests pass.

## Task 6: Strategic/tactical LangGraph routing

**Files:** Modify `src/underwater_tracking/agent/graphs/central.py` and `src/underwater_tracking/agent/state.py`. Create `tests/agent/test_regional_graph.py`.

- [ ] Write failing tests for strategic prediction -> region generation -> regional strategy -> verification -> allocation; tactical reuse without strategy LLM calls; and strategic escalation on relay-radius loss, intent change, lost/reacquired target, hard covariance, endurance conflict, intelligence/scheme change, or regional human directive.
- [ ] Add `region_generation` and `regional_strategy_generation` to the strategic route. Tactical cycles reuse a valid active regional policy and optimize movement/sonar/link continuity only. Add regional state channels and populate the old strategy set with a compatibility proposal.
- [ ] Run regional graph, central graph, and agent-loop tests. Commit `feat: route carrier graph through regional strategy`.

## Task 7: Persistence, directives, evidence, and replay

**Files:** Modify `src/underwater_tracking/persistence/plans.py`, `ledger.py`, `sqlite.py`, `src/underwater_tracking/api/frame_builder.py`, `live.py`, `replay.py`, and domain decision/frame models. Create `tests/persistence/test_regional_replay.py`.

- [ ] Write failing round-trip tests for GridSpec, cell geometry, tasks, roles, links, sonar decisions, revision, degraded reasons, LLM hashes, trigger events, and exact region-scoped human feedback.
- [ ] Store regional models in existing JSON payload columns, add explicit regional decision evidence, add region lookup where needed, and expose cells/tasks/roles/links/sonar/handoff/causal factor in live and replay frames. Keep old frame fields optional and deserializable.
- [ ] Run persistence, frame-contract, hub, and replay tests. Commit `feat: persist regional revisions and replay data`.

## Task 8: Regional operational UI

**Files:** Create `src/underwater_tracking/domain/regional_ui_models.py`, `src/underwater_tracking/ui/src/components/map/RegionOverlay.tsx`, its test, `RegionalOperationsPanel.tsx`, and its test. Modify `domain/ui_models.py`, `ui/src/frameTypes.ts`, `CanvasMap.tsx`, `RightSidebar.tsx`, and `DirectiveComposer.tsx`.

- [ ] Write failing Vitest tests for cell labels/status colors, current/next handoff, roles/links/sonar display, responsive layout, and exact clicked-region directive scope.
- [ ] Render a clipped square-grid overlay with probability/priority/coverage/handoff/reserve/degraded/uncovered states, labels, centerline/corridor, carrier/detection ranges, relay links, and role markers. Add one compact RegionTask row with time, geometry, IDs/roles, sonar, relay chain, health, rationale, and evidence. Preserve target/group UI as compatibility information.
- [ ] Run `npm test -- --run` and `npm run build`; commit `feat: render regional operations views`.

## Task 9: End-to-end acceptance and verification

**Files:** Create `tests/integration/test_regional_acceptance.py`. Modify existing agent-loop/headless-loop/command-center tests and `README.md`.

- [ ] Add a single-target scenario proving estimated centerline, ordered non-overlapping cells, one regional policy per cell, UUV/USV roles, passive/active decisions, relay handoff, plan revision, and replay reconstruction. Assert all LLM payloads are operational and truth-free.
- [ ] Assert `region_tasks` remain the source of truth, legacy views equal deterministic derivations, IDs are stable, and no cells overlap.
- [ ] On a supported Python 3.11 x86_64 host run `python -m pytest -q`, UI unit tests, `npm run build`, and Playwright desktop/mobile E2E. The current remote host has a cpython-311-i386-linux-gnu interpreter on x86_64, so NumPy/ormsgpack/sqlite-vec wheels cannot install; record that as an environment blocker rather than weakening project bounds or committing generated environment files.
- [ ] Commit `test: verify regional tracking acceptance flow`, then run `git status --short --branch` and `git log --oneline --decorate -12`; expect a clean feature/grid-region-tracking branch.

## Self-review

Tasks 1-2 cover GridSpec, RegionCell, adaptive rasterization, non-overlap, visits, evidence, and fallback. Tasks 3 and 6 cover regional LLM ownership and strategic/tactical routing. Tasks 4-5 cover roles, sonar, relays, communication, allocation, waypoints, degradation, and compatibility. Task 7 covers persistence, directives, revisions, evidence, and replay. Task 8 covers the map/panel/feedback UI. Task 9 covers acceptance and migration invariants. All referenced public types are introduced before use, all steps name files and commands, and no unrelated refactor is included.
