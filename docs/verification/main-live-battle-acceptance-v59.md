# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `41d2c3a788192d706f5e87e4f0c4223870bf2d5f`
- Config SHA-256: `0ea7495cff65ff4f1ef44fb9ae8110d467a5027badb6ea932418637bba95cf31`
- Wall-clock start (UTC): `2026-08-24T16:44:55.738186+00:00`
- Wall-clock end (UTC): `2026-08-24T17:05:42.305729+00:00`
- First plan latency: `57.62121961195953` s
- Final run phase: `running`
- Final simulation time: `26825` s
- Final plan version: `28`
- Motion audits: `17`
- Physics frames observed/expected: `5503/5761`
- Browser errors: `9`
- Failed requests: `0`
- Memory events: `100`
- API p95: `160.224` ms
- Output bytes: `197460041`
- Shutdown: `2.915` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `90` | `1` |
| `carrier_dispatch` | `90` | `1` |
| `carrier_returned` | `7320` | `2` |
| `initial_plan_committed` | `0` | `1` |
| `passive_track` | `90` | `1` |
| `recovery` | `2515` | `2` |
| `resource_threshold` | `2310` | `1` |
| `uuv_deployed` | `90` | `1` |
| `uuv_recovered` | `3160` | `2` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Total / teleport / boundary / owner / route / formation / resource |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `5502` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_02` | `5502` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25000000000000006/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_03` | `5502` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_04` | `5502` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `target_00` | `5502` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `7.105427357601002e-16/0.1` | `0.010471975511966214/0.010471975511965976` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_00` | `5502` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_01` | `5502` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_02` | `5502` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_03` | `5502` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982987/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_04` | `5502` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_05` | `5502` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_06` | `5502` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_07` | `5502` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_08` | `5502` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_09` | `5502` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_10` | `5502` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_11` | `5502` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |

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
- [screenshots/desktop-stage-carrier_returned.png](screenshots/desktop-stage-carrier_returned.png)
- [screenshots/desktop-stage-initial_plan_committed.png](screenshots/desktop-stage-initial_plan_committed.png)
- [screenshots/desktop-stage-passive_track.png](screenshots/desktop-stage-passive_track.png)
- [screenshots/desktop-stage-recovery.png](screenshots/desktop-stage-recovery.png)
- [screenshots/desktop-stage-resource_threshold.png](screenshots/desktop-stage-resource_threshold.png)
- [screenshots/desktop-stage-uuv_deployed.png](screenshots/desktop-stage-uuv_deployed.png)
- [screenshots/desktop-stage-uuv_recovered.png](screenshots/desktop-stage-uuv_recovered.png)
- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)
- [screenshots/mobile-stage-active_scan.png](screenshots/mobile-stage-active_scan.png)
- [screenshots/mobile-stage-carrier_dispatch.png](screenshots/mobile-stage-carrier_dispatch.png)
- [screenshots/mobile-stage-carrier_returned.png](screenshots/mobile-stage-carrier_returned.png)
- [screenshots/mobile-stage-initial_plan_committed.png](screenshots/mobile-stage-initial_plan_committed.png)
- [screenshots/mobile-stage-passive_track.png](screenshots/mobile-stage-passive_track.png)
- [screenshots/mobile-stage-recovery.png](screenshots/mobile-stage-recovery.png)
- [screenshots/mobile-stage-resource_threshold.png](screenshots/mobile-stage-resource_threshold.png)
- [screenshots/mobile-stage-uuv_deployed.png](screenshots/mobile-stage-uuv_deployed.png)
- [screenshots/mobile-stage-uuv_recovered.png](screenshots/mobile-stage-uuv_recovered.png)
- [screenshots/mobile.png](screenshots/mobile.png)

## Browser Diagnostics

- `desktop:pageerror:ui_consistency_probe:TimeoutError`
- `desktop:pageerror:ui_consistency_probe:TimeoutError`
- `desktop:pageerror:ui_sim_time_stale`
- `desktop:pageerror:ui_sim_time_stale`
- `desktop:pageerror:ui_sim_time_stale`
- `desktop:pageerror:ui_sim_time_stale`
- `desktop:pageerror:ui_sim_time_stale`
- `mobile:pageerror:ui_consistency_probe:TimeoutError`
- `mobile:pageerror:ui_sim_time_stale`

## Violations

- ledger_trigger_missing_from_events
- plan_timeline_event_missing_from_events
- llm_thinking_source_missing_from_events
- memory_source_missing_from_operational_views
- wall_timeout
- persisted_replay_terminal_mismatch
- missing_stages:handoff
- missing_blue_tracking_evidence_chain
- incomplete_blue_tracking_candidates:target_00:cell:1:-5
- missing_prediction_diff
- missing_counter_tracking_evidence_chain
- browser_errors:9
- battle_not_completed
