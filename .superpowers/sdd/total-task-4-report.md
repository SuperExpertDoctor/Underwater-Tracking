# Task 4 Report: Persist Regional Revisions and Replay Data

## Delivered

- Added `PlanRepository.list_regional_revisions`, which projects historical
  target regional plans from the existing canonical `plans.payload` JSON.
  It includes the parent plan revision, trigger events, and evidence without
  creating a second regional serialization format.
- Stored target-scoped regional LLM request/response hash pairs in the
  canonical plan payload. Replay and regional revision projections expose
  only those hashes, never raw prompts or response bodies.
- Regional candidates preserve the authoritative `StrategySet.trigger_event_ids`,
  so committed regional revisions and replay frames retain their causal links.
- SQLite persistence remains at schema v3. Its idempotent
  `idx_llm_calls_scenario_operation` index supports scenario/operation LLM
  metadata lookup without changing the canonical plan payload format.
- Extended regional frame views with grid, visit window, roles, sonar and
  communication details, evidence, degradation reasons, revision, current and
  next handoff, and causal trigger event IDs.
- Kept all newly added regional frame fields optional or defaulted so prior
  JSONL frames remain valid replay inputs. The live publisher already routes
  through `build_operational_frame`, and `ReplayService` already validates the
  same `OperationalFrame` contract.

## Test Coverage

- Complete regional persistence round-trip: GridSpec, cells, visit windows,
  roles, communication links, sonar policy, handoff, degradation, evidence,
  trigger events, and revision.
- Target-scoped LLM request/response hash round-trip through candidate,
  committed plan, regional revision, frame, and JSONL replay.
- Ordered regional frame tasks, group-quality proxy, effects, current/next
  handoff, target-isolated causal events (including exclusion of unscoped
  events), and bounded effect ratios.
- OptimizeNode regional-plan pipeline preserves strategy trigger events through
  the committed-plan projection.
- Current regional JSONL plus legacy optional fields and `handed_off` task
  statuses replay successfully; target-scoped LLM hashes and causal event IDs
  remain optional replay-compatible fields.

## Commands Run

```bash
PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest tests/api/test_frame_builder_regional_views.py tests/persistence/test_regional_replay.py tests/agent/test_regional_plan_pipeline.py::test_optimize_node_uses_authoritative_single_uuv_relay_policy -q --timeout=20
# 11 passed in 1.00s

PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest tests/api/test_frame_builder_regional_views.py tests/api/test_frame_contracts.py tests/persistence/test_regional_replay.py tests/agent/test_regional_plan_pipeline.py tests/agent/test_repositories.py -q --timeout=20
# 51 passed in 3.66s

PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m compileall -q src/underwater_tracking/persistence src/underwater_tracking/api src/underwater_tracking/domain
# exit 0

git diff --check
# exit 0

PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest -q --timeout=20
# blocked during collection: `hypothesis` is not installed in lang_py310
# (tests/planning/test_waypoints.py and tests/property/test_foundation_invariants.py)
```

## Remaining Risk

- The full Python suite cannot currently collect in `lang_py310` until its
  missing `hypothesis` test dependency is installed; focused Task 4 tests and
  the related repository regression tests passed.
