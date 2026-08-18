# Task 5 Report: Smooth Adversarial Maneuvers and Rapid Blue Response

## Changes

- Expiring an adversary hold now clears its waypoint before the next physics command. The target still uses the shared bounded-motion integrator, so position, acceleration, and turn rate remain bounded without steering toward an expired waypoint.
- The deterministic adversary gate permits a first request, cooldown expiry, a strategic non-ping trigger, or a twice-confirmed material belief revision. Different `active_ping` IDs and informational blue-side feedback cannot bypass the cooldown.
- `SimulationEngine.apply_plan_command` rejects non-increasing `(scenario_id, target_id)` revisions before deployment, waypoint, execution-record, or audit writes. USV hold records a persistent zero-speed position hold; track, relay, and return clear that hold state.
- Fast regional feedback and relay-link-loss events require two consecutive observation updates, are latched until recovery, are included in explicit-world rollback state, and are delivered in the same carrier snapshot. They use the existing central mappings for `regional_feedback_received`, `communication_link_lost`, and `intent_change_confirmed`; `central.py` required no change.
- Maneuver chains close only for a newer regional or relay command. Audit rows carry `chain_id`, `decision_id`, `prediction_revision`, `plan_revision`, and simulation-time latency; stale or unrelated same-target commands leave the chain intact.

## Tests

Executed with `PYTHONPATH=src conda run --no-capture-output -n lang_py310 python -m pytest ... -q --timeout=20`:

- `tests/simulation/test_target.py tests/simulation/test_engine.py tests/agent/test_adversary_graph.py tests/agent/test_runtime_master_slave_adversary.py`: 27 passed.
- The red phase was observed for stale target waypoint steering, active-ping cooldown bypass, stale plan-command execution, non-stationary USV hold, missing fast-replan hysteresis, and an unrelated command closing a maneuver chain. Carrier delivery initially exposed an adversary feedback-loop regression; informational regional feedback is now gated out of immediate adversary requests.

## Known Risks

- The 60-second adversary cooldown and two-observation thresholds are fixed deterministic defaults. They should become scenario configuration if operational tuning needs to vary by target class or relay doctrine.
- A held USV intentionally remains stationary. Long holds while the carrier departs can eventually meet the existing carrier-support boundary; command planners remain responsible for issuing a relay, track, or return action before that operational limit is reached.
- The causal chain records engine application of a qualifying command, not proof of physical tracking effectiveness. Regional quality remains an estimator-derived feedback signal rather than a direct sensor measurement.
