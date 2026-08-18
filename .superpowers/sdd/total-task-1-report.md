# Task 1 Implementation Report

## Outcome

Regional tasks are now the authoritative planning record in the carrier plan
pipeline. Optimize materializes regional policies against the live roster,
derives target-level compatibility fields from those tasks, carries regional
metrics in state and plan records, validates regional constraints before legacy
checks, and preserves an optional regional address on execution commands.

## Files Changed

- `src/underwater_tracking/agent/nodes/optimize.py`
  - Materializes regional policies and allocations, then derives legacy member,
    role, waypoint, and active-UUV projections from the resulting region tasks.
  - Publishes authoritative `region_tasks` and `regional_metrics` in state.
- `src/underwater_tracking/agent/nodes/verify.py`
  - Adds regional validation for known cells, role assignment, handoff links,
    sonar, communication, relay range, and overlapping occupancy through the
    regional validator.
- `src/underwater_tracking/agent/nodes/commit.py`
  - Executes regional validation before legacy plan checks.
  - Associates a command with `region_id` only when the target-level group maps
    unambiguously to one regional task.
- `src/underwater_tracking/domain/agent_models.py`
  - Adds `RegionalPlanMetrics`, plan-level regional metrics, optional command
    `region_id`, and authoritative regional-task legacy projection.
- `src/underwater_tracking/agent/state.py`
  - Adds `regional_metrics` to carrier state.
- `tests/agent/test_regional_plan_pipeline.py`
  - Covers regional authority over legacy compatibility fields, deterministic
    member roles and waypoints, uncovered/degraded preservation, planning proxy
    metrics, command compatibility, and regional validation surfacing.

`tests/agent/test_plan_pipeline.py` was not modified; it was executed as the
required compatibility suite.

## Design Decisions

- `region_tasks` override the task copies embedded in `regional_plans` whenever
  present. The plan remains the source of cells and geometry; the task map is
  the live allocation source.
- Degraded and uncovered tasks are retained in both `region_tasks` and
  `RegionalPlanMetrics`. Coverage rate counts active and degraded regions as
  having some assigned coverage; uncovered regions reduce it.
- Quality, coverage, and relay fields are explicitly named planning proxies via
  `metrics_are_planning_proxies`; they are not sensor truth.
- Target-level fields remain compatibility projections. A command gains a
  region address only if its member set has exactly one matching region task,
  avoiding false regional attribution for an aggregate target command.

## TDD Evidence

1. Added the regional-authority behavior test before implementation.
2. Red run:
   `PYTHONPATH=src conda run -n lang_py310 python -m pytest tests/agent/test_regional_plan_pipeline.py -q`
   Output: `1 failed`; the candidate retained `('LEGACY',)` instead of the
   region-derived `('U1', 'U2')`.
3. Implemented the minimal projection and metrics path.
4. Green run: `1 passed in 0.25s`.
5. Added command and regional-validation behaviors before their implementation.
6. Red run: `2 failed`; `PlanCommand` had no `region_id` field and
   `validate_plan` returned no regional issue.
7. Implemented the command field and regional validation integration.
8. Green run: `3 passed in 0.30s`.

## Verification

Executed from the repository root:

```bash
PYTHONPATH=src conda run -n lang_py310 python -m pytest tests/agent/test_regional_plan_pipeline.py tests/agent/test_plan_pipeline.py -q
```

Output: `21 passed in 3.89s`.

```bash
PYTHONPATH=src conda run -n lang_py310 python -m pytest tests/planning/test_regional_allocation.py tests/agent/test_regional_strategy.py tests/agent/test_commit.py -q
```

Output: `16 passed in 0.31s`.

```bash
PYTHONPATH=src conda run -n lang_py310 python -m ruff check src/underwater_tracking/agent/nodes/optimize.py src/underwater_tracking/agent/nodes/verify.py src/underwater_tracking/agent/nodes/commit.py src/underwater_tracking/agent/state.py src/underwater_tracking/domain/agent_models.py tests/agent/test_regional_plan_pipeline.py
```

Output: `All checks passed!`.

```bash
PYTHONPATH=src conda run -n lang_py310 python -m mypy src/underwater_tracking/agent/nodes/optimize.py src/underwater_tracking/agent/nodes/verify.py src/underwater_tracking/agent/nodes/commit.py src/underwater_tracking/agent/state.py src/underwater_tracking/domain/agent_models.py
```

Output: `Success: no issues found in 5 source files`.

```bash
git diff --check
```

Output: no whitespace errors.

## Self-Review

- The regional plan cells remain immutable geometry inputs while the keyed task
  map remains authoritative for allocations and degraded status.
- Regional validation runs first in `validate_plan`; its issues are preserved
  alongside legacy validation issues rather than replacing them.
- The command change is backward compatible: `region_id` defaults to `None`
  and existing target/group execution fields are unchanged.
- Only Task 1 source, test, and report files are staged for the commit.

## Concerns

- A target-level `PlanCommand` can represent several regional tasks. In that
  case `region_id` intentionally remains `None`; consumers should use the
  committed plan's authoritative `region_tasks` for the full regional schedule.
- Coverage rate is an operational planning proxy and treats degraded regions as
  partially covered. It must not be interpreted as measured sensor coverage.
