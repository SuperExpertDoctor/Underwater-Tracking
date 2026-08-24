# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `83f0662988654d7703b174db2cea081a8ef54d5e`
- Config SHA-256: `69eb837c7e6eeb9a572d4730c5260a24f79ec2a4a8939868a502e854d1b9929b`
- Wall-clock start (UTC): `2026-08-23T22:11:09.662375+00:00`
- Wall-clock end (UTC): `2026-08-23T22:12:38.431166+00:00`
- First plan latency: `85.63178688497283` s
- Final run phase: `failed`
- Final simulation time: `0` s
- Final plan version: `1`
- Motion audits: `17`
- Physics frames observed/expected: `1/721`
- Browser errors: `0`
- Failed requests: `0`
- Memory events: `6`
- API p95: `26.797` ms
- Output bytes: `7164594`
- Shutdown: `2.268` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `initial_plan_committed` | `0` | `1` |

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
- [screenshots/desktop-stage-initial_plan_committed.png](screenshots/desktop-stage-initial_plan_committed.png)
- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)
- [screenshots/mobile-stage-initial_plan_committed.png](screenshots/mobile-stage-initial_plan_committed.png)
- [screenshots/mobile.png](screenshots/mobile.png)

## Violations

- planning_health_frame_mismatch
- memory_source_missing_from_operational_views
- run_phase:failed
- persisted_replay_terminal_mismatch
- missing_stages:active_scan,carrier_dispatch,carrier_returned,handoff,passive_track,recovery,resource_threshold,uuv_deployed,uuv_recovered
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
- battle_not_completed
