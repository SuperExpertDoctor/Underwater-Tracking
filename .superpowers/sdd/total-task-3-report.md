# Task 3 Report: Regional Strategy Routing

## Changes

- Routed strategic carrier cycles through deterministic region generation and the
  regional strategy LLM before the legacy semantic verification and optimizer
  chain. Tactical cycles retain the existing direct optimizer continuation.
- Added carrier-owned regional replan signals for feedback, relay radius,
  endurance, communication link, covariance, and target reacquisition. These
  signals are classified as strategic without weakening the generic event
  monitor's unknown-event validation.
- Deferred deterministic regional and event-classification failures to
  `handle_error`; `LLMError` still propagates to `CarrierRuntime`, which keeps
  its existing retry/pause transaction semantics.
- Added a typed runtime entry point for queuing a regional replan and persisted
  the current regional replan reasons in carrier state.
- Added graph tests for strategic trigger routing, deterministic error routing,
  LLM pause behavior, graph edge order, tactical no-LLM continuation, and the
  live agent-loop assertion for a regional strategy call.

## Tests

- RED: `PYTHONPATH=src conda run -n lang_py310 python -m pytest tests/agent/test_regional_graph.py -q`
  initially failed for all six new regional replan signals because the strict
  generic event monitor rejected their event types.
- GREEN: the same command passed with `16 passed in 0.76s` after carrier-level
  classification and error routing were added.
- Tactical continuation: `PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest tests/agent/test_central_graph.py::test_tactical_route_never_calls_llm --timeout=20 -q`
  passed with `1 passed in 0.92s`.
- Controlled graph suite: `PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest tests/agent/test_central_graph.py tests/agent/test_regional_graph.py -m 'not real_llm' --timeout=20 -q -rA`
  passed with `24 passed, 4 deselected in 1.84s`.
- Compile check: `PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m compileall -q src`
  exited successfully.
- Controlled agent-loop run: `PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest tests/agent/test_central_graph.py tests/agent/test_regional_graph.py tests/integration/test_agent_loop.py -m 'not real_llm' --timeout=20 -q -rA`
  completed in `6.14s` with `24 passed, 7 deselected, 1 failed`.

## Unresolved Risk

- `tests/integration/test_agent_loop.py::test_checkpoint_failure_stops_commits_but_not_group_updates`
  fails its 30-second report-cadence assertion. The failure is outside Task 3
  routing: this test has no Task 3 integration diff, while the shared worktree
  contains intentional uncommitted changes in `simulation/target.py` and
  `tests/simulation/test_engine.py`. It was not modified here to preserve task
  scope.
- Real-provider tests were not left running without a bound after earlier
  no-output processes were terminated. The live E2E assertion for
  `regional_strategy` is present but remains to be run in an environment with a
  bounded, healthy LLM endpoint.
