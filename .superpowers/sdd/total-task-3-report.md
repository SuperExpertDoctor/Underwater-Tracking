# Task 3 Report: Regional Strategy Routing

## Review Fixes

- The strategic carrier path is now `trajectory_prediction ->
  regional_generation -> regional_strategy -> regional_strategy_adapter ->
  verify_strategy -> resource_optimizer -> verify_plan -> commit_plan`.
  `strategy_generation` is no longer a carrier graph node, so the legacy
  strategy LLM cannot replace a regional decision.
- `RegionalStrategyToStrategySetNode` supplies only the target-level fields
  required by the existing verifier.  The original `regional_policies` pass
  through unchanged.  When regional plans are present, `OptimizeNode` now
  materializes their explicit assignments and builds the `TrackingPlan`
  directly from those tasks, without calling the generic 2--4 UUV allocator.
  Assigned UUV/USV members, tracking mode, relay topology, waypoints, and
  deterministic degraded/uncovered status therefore remain authoritative from
  verification through optimization and commit.
- Tactical cycles retain the checkpointed regional policies and do not invoke
  the regional strategy LLM.  The restart regression test compares the first
  strategic policies and resulting region tasks with the tactical cycle.
- `assess_regional_replan_events` is a reusable deterministic boundary
  interface invoked by `CarrierRuntime` before each graph cycle.  It does not
  infer `target_lost` from a missing `GroupReport`, and it does not promote a
  raw `intent_change_detected` marker to `intent_change_confirmed`; those need
  EventMonitor/Task 5 evidence and confidence gates.  It still emits a
  checkpoint-backed `target_reacquired`, covariance, communication-link, and
  active-plan endurance signals.  With a complete platform snapshot, it now
  detects an assigned USV outside the carrier support radius from their actual
  positions and emits `relay_radius_exceeded` with platform evidence.
- Deterministic regional errors still defer through `handle_error`.
  `LLMError` still escapes the wiring node and preserves the runtime's
  retry/pause transaction behavior.

## Tests

### Final Review Fix

- RED: the focused review tests failed as expected on the prior implementation:
  missing reports emitted `target_lost`, raw intent markers emitted
  `intent_change_confirmed`, unassigned UUVs emitted endurance events, relay
  radius had no automatic event, and a valid one-UUV relay policy produced a
  generic `degraded` plan.
- GREEN: `PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m
  pytest tests/agent/test_regional_graph.py::test_state_assessment_emits_evidence_backed_replan_events
  tests/agent/test_regional_graph.py::test_state_assessment_does_not_treat_a_missing_report_as_target_loss
  tests/agent/test_regional_graph.py::test_state_assessment_limits_endurance_and_relay_checks_to_active_assignments
  tests/agent/test_regional_plan_pipeline.py::test_optimize_node_uses_authoritative_single_uuv_relay_policy
  --timeout=20 -q` passed: `4 passed in 0.56s`.
- Controlled graph/regional suite: `PYTHONPATH=src conda run
  --no-capture-output -n lang_py310 python -m pytest
  tests/agent/test_regional_graph.py tests/agent/test_regional_plan_pipeline.py
  tests/agent/test_central_graph.py -m 'not real_llm' --timeout=20 -q` passed:
  `39 passed, 4 deselected in 2.15s`.
- The live restart regression now includes an explicit `PlatformSnapshot` and
  compares regional policy UUV/USV IDs, tracking mode, relay role, and the
  regional-strategy LLM call count across strategic then tactical restart.
  Its direct run timed out at the configured 20-second limit while waiting for
  the external provider during intent analysis; it did not reach Task 3 logic.

- RED: `PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m
  pytest tests/agent/test_regional_graph.py::test_state_assessment_reacquires_a_previously_lost_target
  --timeout=20 -q` failed with the expected missing `lost_target_ids`
  interface error.
- GREEN: `PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m
  pytest tests/agent/test_regional_graph.py --timeout=20 -q` passed with
  `19 passed in 0.70s`.
- Scoped controlled suite: `PYTHONPATH=src conda run --no-capture-output -n
  lang_py310 python -m pytest tests/agent/test_central_graph.py
  tests/agent/test_regional_graph.py tests/agent/test_regional_plan_pipeline.py
  tests/integration/test_agent_loop.py -m 'not real_llm' --timeout=20 -q -rA`
  produced `39 passed, 7 deselected, 1 failed` in `6.41s`.  All
  central/regional graph assertions passed; the one integration failure is the
  pre-existing checkpoint/report-cadence issue below.

## Remaining Risk

- `tests/integration/test_agent_loop.py::test_checkpoint_failure_stops_commits_but_not_group_updates`
  still fails its 30-second report-cadence assertion.  This is the same
  unrelated simulation/checkpoint behavior reported before this review fix;
  it is outside Task 3's permitted files for engine/CLI wiring.
- The runtime now computes relay radius from an assigned USV and the carrier
  when both positions are present.  Upstream production of EventMonitor's
  gated loss/intent evidence, and simulation/CLI wiring for feedback, remains
  Task 5/11 follow-up; raw group-report flags are intentionally not treated as
  confirmed strategic events.
- The live-provider strategic/tactical regression is marked `real_llm` and was
  excluded from the controlled suite.  A direct bounded run timed out in the
  provider HTTPS read, so it requires a healthy configured provider endpoint.
