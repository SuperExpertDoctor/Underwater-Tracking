# Task 4 Report: Persist Regional Revisions and Replay Data

## Delivered

- Added `PlanRepository.list_regional_revisions`, which projects historical
  target regional plans from the existing canonical `plans.payload` JSON.
  It includes the parent plan revision, trigger events, and evidence without
  creating a second regional serialization format.
- Added scenario/operation filtering for persisted LLM metadata hashes and an
  SQLite index for that lookup. Schema version is now 3.
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
- LLM request/response hash lookup round-trip.
- Ordered regional frame tasks, group-quality proxy, effects, current/next
  handoff, causal events, and bounded effect status.
- Current regional JSONL and legacy regional JSONL shapes replay successfully.

## Commands Run

```bash
PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest tests/api/test_frame_builder_regional_views.py tests/persistence/test_regional_replay.py -q --timeout=20
# 5 passed in 1.07s

PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest tests/api/test_frame_builder_regional_views.py tests/api/test_frame_contracts.py tests/persistence/test_regional_replay.py tests/agent/test_repositories.py -q --timeout=20
# 36 passed in 3.14s

PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m compileall -q src/underwater_tracking/persistence src/underwater_tracking/api src/underwater_tracking/domain
# exit 0

git diff --check
# exit 0

PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest -q --timeout=20
# blocked during collection: `hypothesis` is not installed in lang_py310
# (tests/planning/test_waypoints.py and tests/property/test_foundation_invariants.py)
```

## Remaining Risk

- Regional LLM hashes are durable in the existing `llm_calls` metadata table
  and can now be filtered by scenario/operation, but the table has no target
  identifier. A single batched regional strategy call therefore cannot yet be
  attributed to an individual target in the frame without adding target scope
  to that pre-existing call contract.
- The full Python suite cannot currently collect in `lang_py310` until its
  missing `hypothesis` test dependency is installed; focused Task 4 tests and
  the related repository regression tests passed.
