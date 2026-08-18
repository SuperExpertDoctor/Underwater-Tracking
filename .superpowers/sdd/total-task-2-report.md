# Task 2 Report: LLM-Authoritative Regional Teams

## Scope

Task 2 makes regional policy membership LLM-authoritative. Explicit UUV and
USV IDs, including empty selections, are carried into regional tasks without
count-driven fallback selection. Availability, duplicate-ID, tracking-domain,
relay, capability, link, motion, and safety validation remain deterministic.
Feedback directives now remain context-only and cannot carry assignments.

## Files Changed

- `src/underwater_tracking/domain/regional_models.py`
- `src/underwater_tracking/planning/regional_allocation.py`
- `src/underwater_tracking/planning/regional_validation.py`
- `src/underwater_tracking/domain/agent_models.py`
- `src/underwater_tracking/agent/nodes/directives.py`
- `src/underwater_tracking/agent/runtime.py`
- `src/underwater_tracking/agent/nodes/strategy.py`
- `src/underwater_tracking/agent/nodes/regional_strategy.py`
- `src/underwater_tracking/agent/prompts.py`
- `tests/planning/test_regional_allocation.py`
- `tests/planning/test_regional_validation.py`
- `tests/agent/test_assignment_directives.py`

## TDD Evidence

### Red

Added allocation tests before changing production code, then ran:

```bash
PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest \
  tests/planning/test_regional_allocation.py::test_materializer_preserves_explicit_llm_members_and_empty_membership \
  tests/planning/test_regional_allocation.py::test_materializer_degrades_unknown_duplicate_and_unavailable_llm_members \
  tests/planning/test_regional_validation.py::test_heuristic_usv_rejects_uuv_primary_tracking \
  tests/agent/test_assignment_directives.py::test_applied_feedback_reaches_strategy_context_without_reserving_uuvs -q
```

Output: `2 failed, 2 passed in 0.75s`.

- The first failure showed `required_uuv_count=4` being overwritten to `1`.
- The second showed duplicate LLM IDs being silently de-duplicated instead of
  producing a degradation reason.

Added the feedback-assignment contract test before its model validation and
ran:

```bash
PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest \
  tests/agent/test_assignment_directives.py::test_feedback_directive_cannot_carry_an_assignment -q
```

Output: `1 failed in 0.54s`; no `ValueError` was raised.

### Green

After the minimal implementation, ran the new behavior tests:

```bash
PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest \
  tests/planning/test_regional_allocation.py::test_materializer_preserves_explicit_llm_members_and_empty_membership \
  tests/planning/test_regional_allocation.py::test_materializer_degrades_unknown_duplicate_and_unavailable_llm_members \
  tests/planning/test_regional_validation.py::test_heuristic_usv_rejects_uuv_primary_tracking \
  tests/agent/test_assignment_directives.py::test_applied_feedback_reaches_strategy_context_without_reserving_uuvs \
  tests/agent/test_assignment_directives.py::test_feedback_directive_cannot_carry_an_assignment -q
```

Output: `5 passed in 0.69s`.

## Verification

Required Task 2 suites:

```bash
PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest \
  tests/planning/test_regional_allocation.py \
  tests/planning/test_regional_validation.py \
  tests/agent/test_directives.py \
  tests/agent/test_assignment_directives.py -q
```

Output: exit code `0`; the subsequent `-ra` verification collected `32` tests
across the four required modules and also exited `0`.

Scoped static checks:

```bash
PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m ruff check \
  src/underwater_tracking/domain/regional_models.py \
  src/underwater_tracking/planning/regional_allocation.py \
  src/underwater_tracking/planning/regional_validation.py \
  src/underwater_tracking/domain/agent_models.py \
  src/underwater_tracking/agent/nodes/directives.py \
  src/underwater_tracking/agent/runtime.py \
  src/underwater_tracking/agent/nodes/strategy.py \
  src/underwater_tracking/agent/nodes/regional_strategy.py \
  src/underwater_tracking/agent/prompts.py \
  tests/planning/test_regional_allocation.py \
  tests/planning/test_regional_validation.py \
  tests/agent/test_directives.py \
  tests/agent/test_assignment_directives.py
```

Output: `All checks passed!`.

```bash
git diff --check
```

Output: no whitespace errors.

## Self-Review

- `materialize_regional_plan` now copies LLM member tuples directly and retains
  `required_uuv_count` and `required_usv_count` as explanatory metadata.
- Empty LLM membership remains empty and is surfaced as `uncovered`; no
  advisory count causes automatic member selection.
- Unknown, duplicate, reserved, and unavailable explicit members become
  deterministic degradation reasons; no substitute platforms are selected.
- `uuv_primary_usv_relay`, `heuristic_uuv`, and `heuristic_usv` remain separate
  active tracking domains. Relay mode permits multiple UUVs plus a relay-only
  USV; heuristic modes reject mixed domains.
- Feedback directives retain preview/apply and event/version behavior, enter
  generic and regional strategy context with target and region scope, and are
  blocked from carrying assignment fields or reserving UUVs.
- Regional and generic strategy payloads include prior regional effects,
  degradation reasons, snapshot/plan/prediction identifiers, and feedback.

## Concerns

- The required `tests/agent/test_directives.py` module remains configured to
  use the repository's live LLM fixture. It participated in the required
  successful suite run; deterministic feedback coverage was intentionally
  added to `test_assignment_directives.py` so it does not depend on a model
  response.

## Follow-Up: Review Fixes

### Findings Addressed

- `uuv_primary_usv_relay` now treats relay communication as a mode-level hard
  constraint. Disabling `usv_relay_required` reports
  `relay_communication_required:<region_id>`, and selected relay members have
  their acoustic paths checked regardless of that flag.
- Feedback directives now reject every planning-mutation field:
  `locked_members`, priorities, minimum quality, disabled/return UUVs, and
  assignment fields. Scope, feedback text/regions, confidence, and lifecycle
  metadata remain valid.
- `RegionalPolicy.assigned_uuv_ids` and `assigned_usv_ids` are now required
  structured fields. `()` remains a valid intentional empty selection, while
  omission is a schema error.

### TDD Evidence

Before the follow-up implementation, ran:

```bash
PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest \
  tests/planning/test_regional_validation.py::test_uuv_primary_mode_requires_relay_communication_and_path \
  tests/planning/test_regional_allocation.py::test_regional_policy_requires_explicit_member_lists \
  tests/agent/test_assignment_directives.py::test_feedback_directive_cannot_carry_planning_constraints -q
```

Output: `7 failed in 0.60s`.

- Relay mode produced no issue when relay communication was disabled and the
  UUV-to-USV path exceeded the acoustic range.
- A policy without member-list fields validated successfully.
- All five feedback constraint payloads validated successfully.

After the implementation, ran the covering regressions:

```bash
PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest \
  tests/planning/test_regional_validation.py::test_uuv_primary_mode_requires_relay_communication_and_path \
  tests/planning/test_regional_allocation.py::test_regional_policy_requires_explicit_member_lists \
  tests/agent/test_assignment_directives.py::test_feedback_directive_cannot_carry_an_assignment \
  tests/agent/test_assignment_directives.py::test_feedback_directive_cannot_carry_planning_constraints -q
```

Output: `8 passed in 0.50s`.

The related regional-policy fixture suite was also run after making its
intentional empty member lists explicit:

```bash
PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest \
  tests/agent/test_regional_strategy.py -q
```

Output: `5 passed in 0.04s`.

### Final Verification

```bash
PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest \
  tests/planning/test_regional_allocation.py \
  tests/planning/test_regional_validation.py \
  tests/agent/test_directives.py \
  tests/agent/test_assignment_directives.py -q
```

Output: exit code `0`.

```bash
PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m ruff check \
  src/underwater_tracking/domain/regional_models.py \
  src/underwater_tracking/planning/regional_allocation.py \
  src/underwater_tracking/planning/regional_validation.py \
  src/underwater_tracking/domain/agent_models.py \
  src/underwater_tracking/agent/nodes/directives.py \
  src/underwater_tracking/agent/runtime.py \
  src/underwater_tracking/agent/nodes/strategy.py \
  src/underwater_tracking/agent/nodes/regional_strategy.py \
  src/underwater_tracking/agent/prompts.py \
  tests/planning/test_regional_allocation.py \
  tests/planning/test_regional_validation.py \
  tests/agent/test_directives.py \
  tests/agent/test_assignment_directives.py \
  tests/agent/test_regional_strategy.py
```

Output: `All checks passed!`.

```bash
git diff --check
```

Output: no whitespace errors.
