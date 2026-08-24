# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `83f0662988654d7703b174db2cea081a8ef54d5e`
- Config SHA-256: `69eb837c7e6eeb9a572d4730c5260a24f79ec2a4a8939868a502e854d1b9929b`
- Wall-clock start (UTC): `2026-08-24T01:17:22.277199+00:00`
- Wall-clock end (UTC): `2026-08-24T01:19:25.607543+00:00`
- First plan latency: `66.60202425601892` s
- Final run phase: `running`
- Final simulation time: `985` s
- Final plan version: `3`
- Motion audits: `17`
- Physics frames observed/expected: `228/361`
- Browser errors: `2`
- Failed requests: `0`
- Memory events: `16`
- API p95: `86.833` ms
- Output bytes: `15404688`
- Shutdown: `1.614` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `130` | `1` |
| `carrier_dispatch` | `90` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `passive_track` | `150` | `1` |
| `recovery` | `920` | `3` |
| `resource_threshold` | `630` | `2` |
| `uuv_deployed` | `90` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Total / teleport / boundary / owner / route / formation / resource |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `227` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_02` | `227` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_03` | `227` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_04` | `227` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `target_00` | `227` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `7.105427357601002e-16/0.1` | `0.010471975511966214/0.010471975511965976` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_00` | `227` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_01` | `227` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_02` | `227` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_03` | `227` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_04` | `227` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_05` | `227` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_06` | `227` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_07` | `227` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_08` | `227` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_09` | `227` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_10` | `227` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_11` | `227` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |

## Evidence Chains


## Blue Tracking Chains

| Target | Carrier / candidate | UUVs | Dispatch | Deploy | Active ping | Estimates | Handoff | Resource | Recovery | Recovered | Carrier return | Plan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |

## Prediction Intent Chains

| Target | Diff / thresholds | Window (s) | Suspicion | Intent provider / calls | Confirmation | Plan | Response latency | Blue response |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |

## Screenshots

- [screenshots/desktop-latest.png](screenshots/desktop-latest.png)
- [screenshots/desktop-stage-active_scan.png](screenshots/desktop-stage-active_scan.png)
- [screenshots/desktop-stage-carrier_dispatch.png](screenshots/desktop-stage-carrier_dispatch.png)
- [screenshots/desktop-stage-initial_plan_committed.png](screenshots/desktop-stage-initial_plan_committed.png)
- [screenshots/desktop-stage-passive_track.png](screenshots/desktop-stage-passive_track.png)
- [screenshots/desktop-stage-recovery.png](screenshots/desktop-stage-recovery.png)
- [screenshots/desktop-stage-resource_threshold.png](screenshots/desktop-stage-resource_threshold.png)
- [screenshots/desktop-stage-uuv_deployed.png](screenshots/desktop-stage-uuv_deployed.png)
- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)
- [screenshots/mobile-stage-active_scan.png](screenshots/mobile-stage-active_scan.png)
- [screenshots/mobile-stage-carrier_dispatch.png](screenshots/mobile-stage-carrier_dispatch.png)
- [screenshots/mobile-stage-initial_plan_committed.png](screenshots/mobile-stage-initial_plan_committed.png)
- [screenshots/mobile-stage-passive_track.png](screenshots/mobile-stage-passive_track.png)
- [screenshots/mobile-stage-recovery.png](screenshots/mobile-stage-recovery.png)
- [screenshots/mobile-stage-resource_threshold.png](screenshots/mobile-stage-resource_threshold.png)
- [screenshots/mobile-stage-uuv_deployed.png](screenshots/mobile-stage-uuv_deployed.png)
- [screenshots/mobile.png](screenshots/mobile.png)

## Browser Diagnostics

- `desktop:pageerror:missing_ui_surface:sim_time`
- `mobile:pageerror:missing_ui_surface:sim_time`

## Violations

- planning_degraded:carrier graph completed without a terminal result for the active epoch
- persisted_replay_terminal_mismatch
- missing_stages:carrier_returned,handoff,uuv_recovered
- real_provider_unavailable
- missing_blue_tracking_evidence_chain
- incomplete_blue_tracking_candidates:target_00:cell:-3:-4
- missing_prediction_diff
- missing_counter_tracking_evidence_chain
- browser_errors:2
- battle_not_completed
