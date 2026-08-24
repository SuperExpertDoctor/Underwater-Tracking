# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `83f0662988654d7703b174db2cea081a8ef54d5e`
- Config SHA-256: `69eb837c7e6eeb9a572d4730c5260a24f79ec2a4a8939868a502e854d1b9929b`
- Wall-clock start (UTC): `2026-08-24T00:42:32.843582+00:00`
- Wall-clock end (UTC): `2026-08-24T00:45:35.342148+00:00`
- First plan latency: `unavailable` s
- Final run phase: `awaiting_retry`
- Final simulation time: `0` s
- Final plan version: `0`
- Motion audits: `17`
- Physics frames observed/expected: `1/361`
- Browser errors: `1`
- Failed requests: `0`
- Memory events: `13`
- API p95: `9.719` ms
- Output bytes: `3267960`
- Shutdown: `1.705` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Total / teleport / boundary / owner / route / formation / resource |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `0` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_02` | `0` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_03` | `0` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_04` | `0` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `target_00` | `0` | `8.0/14.0` | `0.0/0.08` | `0.0/0.1` | `0.0/0.010471975511965976` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_00` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_01` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_02` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_03` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_04` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_05` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_06` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_07` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_08` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_09` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_10` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_11` | `0` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |

## Evidence Chains


## Blue Tracking Chains

| Target | Carrier / candidate | UUVs | Dispatch | Deploy | Active ping | Estimates | Handoff | Resource | Recovery | Recovered | Carrier return | Plan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |

## Prediction Intent Chains

| Target | Diff / thresholds | Window (s) | Suspicion | Intent provider / calls | Confirmation | Plan | Response latency | Blue response |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |

## Screenshots

- [screenshots/desktop-latest.png](screenshots/desktop-latest.png)
- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)
- [screenshots/mobile.png](screenshots/mobile.png)

## Browser Diagnostics

- `mobile:console:Failed to load resource: the server responded with a status of 500 (Internal Server Error)`

## Violations

- planning_awaiting_retry:CancelledLLMError: LLM call cancelled
- memory_source_missing_from_operational_views
- initial_plan_not_committed
- persisted_replay_terminal_mismatch
- missing_stages:active_scan,carrier_dispatch,carrier_returned,handoff,passive_track,recovery,resource_threshold,uuv_deployed,uuv_recovered
- real_provider_unavailable
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
- missing_blue_tracking_evidence_chain
- missing_prediction_intent_evidence_chain
- missing_counter_tracking_evidence_chain
- browser_errors:1
- battle_not_completed
