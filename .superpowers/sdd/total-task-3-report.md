# Task 3 Report: Regional Strategy Routing

## Review Fixes

- The strategic carrier path is now `trajectory_prediction ->
  regional_generation -> regional_strategy -> regional_strategy_adapter ->
  verify_strategy -> resource_optimizer -> verify_plan -> commit_plan`.
  `strategy_generation` is no longer a carrier graph node, so the legacy
  strategy LLM cannot replace a regional decision.
- `RegionalStrategyToStrategySetNode` supplies only the target-level fields
  required by the existing verifier.  The original `regional_policies` pass
  through unchanged and `OptimizeNode` materializes them into region tasks;
  assigned UUV/USV members, tracking mode, and relay requirements remain
  regional-policy authority.
- Tactical cycles retain the checkpointed regional policies and do not invoke
  the regional strategy LLM.  The restart regression test compares the first
  strategic policies and resulting region tasks with the tactical cycle.
- `assess_regional_replan_events` is a reusable deterministic boundary
  interface invoked by `CarrierRuntime` before each graph cycle.  It emits
  evidence-backed target loss/reacquisition, covariance, endurance, link, and
  intent-change events.  Loss state is checkpointed so a later observation
  automatically emits `target_reacquired`; intent changes in group reports
  automatically escalate without a manually submitted confirmed event.
- Deterministic regional errors still defer through `handle_error`.
  `LLMError` still escapes the wiring node and preserves the runtime's
  retry/pause transaction behavior.

## Tests

- RED: `PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m
  pytest tests/agent/test_regional_graph.py::test_state_assessment_reacquires_a_previously_lost_target
  --timeout=20 -q` failed with the expected missing `lost_target_ids`
  interface error.
- GREEN: `PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m
  pytest tests/agent/test_regional_graph.py --timeout=20 -q` passed with
  `19 passed in 0.70s`.
- Scoped controlled suite: `PYTHONPATH=src conda run --no-capture-output -n
  lang_py310 python -m pytest tests/agent/test_central_graph.py
  tests/agent/test_regional_graph.py tests/integration/test_agent_loop.py -m
  'not real_llm' --timeout=20 -q -rA` produced `27 passed, 7 deselected, 1
  failed` in `6.05s`.  All central/regional graph assertions passed.

## Remaining Risk

- `tests/integration/test_agent_loop.py::test_checkpoint_failure_stops_commits_but_not_group_updates`
  still fails its 30-second report-cadence assertion.  This is the same
  unrelated simulation/checkpoint behavior reported before this review fix;
  it is outside Task 3's permitted files for engine/CLI wiring.
- The runtime interface now consumes snapshot threshold signals, but the
  simulation engine and CLI's lower-level production of relay-radius,
  endurance, link, covariance, reacquisition, and feedback signals remains
  Task 5/11 cross-task wiring risk.  They can use
  `CarrierRuntime.submit_regional_replan` or the reusable assessment interface.
- The live-provider strategic/tactical regression is marked `real_llm` and was
  deselected from the bounded controlled suite; it requires a healthy configured
  provider endpoint.
