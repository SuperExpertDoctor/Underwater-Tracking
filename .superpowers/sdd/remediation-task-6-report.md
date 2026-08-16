# Remediation Task 6 report

Base: `6cedec5`

## Root causes

- `SimulationEngine.apply_plan_command` projected a complete strategic roster
  into replacement pairs and only considered non-deployed outgoing members.
  Pure growth, pure shrink, and healthy same-size replacement therefore never
  reached `GroupManager`.
- Group commands could only express replacements, so the group graph could
  not atomically replace the roster or update positions.
- UUV validation enforced only status-to-deployment contradictions; carrier
  validation enforced list disjointness but not status/list/speed agreement.
- `AssignmentPanel` retained local selected IDs after a later frame made an
  ID non-deployable.

## Changes

- Added optional authoritative roster and position fields to internal group
  commands. The group node atomically applies authoritative additions,
  removals, and replacements, refreshes positions/revision/report, and emits
  deterministic membership events. Replacement-only commands retain their
  legacy behavior and `member_failed` events.
- Engine commands now pass the full committed roster plus positions to the
  group manager. After the group graph applies it, `_uuv_groups` and frame
  assignments are synchronized to the same roster.
- Added bidirectional UUV status/deployment checks, canonical carrier
  status derivation/validation, and missing-field-only normalization for
  mapping and typed legacy carriers.
- The assignment panel now derives submit eligibility from the latest frame
  and clears stale selections after availability changes.

## Verification

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests/groups/test_group_graph.py tests/simulation/test_deployment_lifecycle.py tests/simulation/test_carrier.py tests/domain/test_models.py tests/api/test_frame_contracts.py tests/agent/test_assignment_directives.py tests/agent/test_active_verification.py` — 76 passed.
- `npm test -- --run src/components/assistant/AssignmentPanel.test.tsx` — 5 passed.
- `git diff --check`, `.venv/bin/ruff check src tests`, `.venv/bin/mypy src`,
  `npx tsc --noEmit`, and `npm run build` — passed.

## Full-suite note

The unscoped Python suite was stopped at the requested focused-test boundary.
Before stopping, its first failure without credentials was
`tests/agent/test_llm_port.py::test_llm_config_points_at_longcat_provider`
because `UNDERWATER_TRACKING_API_KEY` was unset. Supplying a placeholder key
allowed that config assertion but caused `@pytest.mark.real_llm` tests to make
real provider calls and fail with HTTP 401. No product code was changed for
either environment-only failure.
