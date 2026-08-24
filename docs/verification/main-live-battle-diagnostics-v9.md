# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `dad5829030dddafbdc7f7451c235e6e20cc97d52`
- Config SHA-256: `6fee101518f0d238ffc31dab93deae557ad1ca34f7833680b07f7b11a785428a`
- Wall-clock start (UTC): `2026-08-23T13:38:18.887310+00:00`
- Wall-clock end (UTC): `2026-08-23T13:42:30.602861+00:00`
- First plan latency: `unavailable` s
- Final run phase: `awaiting_retry`
- Final simulation time: `0` s
- Final plan version: `0`
- Motion audits: `17`
- Physics frames observed/expected: `1/181`
- Browser errors: `0`
- Failed requests: `0`
- Memory events: `0`
- API p95: `9.181` ms
- Output bytes: `2579766`
- Shutdown: `1.166` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `0` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_02` | `0` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_03` | `0` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_04` | `0` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `target_00` | `0` | `8.0/14.0` | `0.0/0.08` | `0.0/0.1` | `0.0/0.010471975511965976` | `n/a` | `0` |
| `uuv_00` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_01` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_02` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_03` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_04` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_05` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_06` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_07` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_08` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_09` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_10` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_11` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |

## Evidence Chains


## Screenshots

- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile.png](screenshots/mobile.png)
- [screenshots/desktop-latest.png](screenshots/desktop-latest.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)

## Violations

- planning_awaiting_retry:CancelledLLMError: LLM call cancelled
- initial_plan_not_committed
- missing_stages:active_scan,carrier_dispatch,carrier_returned,handoff,passive_track,recovery,resource_threshold,uuv_deployed,uuv_recovered
- memory_stream_empty
- real_provider_unavailable
- adversary_llm_decision_not_observed
- no_observed_steps:carrier_01
- no_observed_steps:carrier_02
- no_observed_steps:carrier_03
- no_observed_steps:carrier_04
- no_observed_steps:target_00
- no_observed_steps:uuv_00
- no_observed_steps:uuv_01
- no_observed_steps:uuv_02
- no_observed_steps:uuv_03
- no_observed_steps:uuv_04
- no_observed_steps:uuv_05
- no_observed_steps:uuv_06
- no_observed_steps:uuv_07
- no_observed_steps:uuv_08
- no_observed_steps:uuv_09
- no_observed_steps:uuv_10
- no_observed_steps:uuv_11
- missing_counter_tracking_evidence_chain
- missing_adversary_llm_decision
- battle_not_completed
