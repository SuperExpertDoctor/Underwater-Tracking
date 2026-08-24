# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `83f0662988654d7703b174db2cea081a8ef54d5e`
- Config SHA-256: `69eb837c7e6eeb9a572d4730c5260a24f79ec2a4a8939868a502e854d1b9929b`
- Wall-clock start (UTC): `2026-08-24T01:05:27.874374+00:00`
- Wall-clock end (UTC): `2026-08-24T01:09:34.868779+00:00`
- First plan latency: `163.3704796070233` s
- Final run phase: `running`
- Final simulation time: `1800` s
- Final plan version: `3`
- Motion audits: `17`
- Physics frames observed/expected: `364/361`
- Browser errors: `5`
- Failed requests: `1`
- Memory events: `25`
- API p95: `37.072` ms
- Output bytes: `20429767`
- Shutdown: `2.462` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `90` | `1` |
| `carrier_dispatch` | `90` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `passive_track` | `90` | `1` |
| `recovery` | `930` | `3` |
| `resource_threshold` | `480` | `2` |
| `uuv_deployed` | `90` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Total / teleport / boundary / owner / route / formation / resource |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `363` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_02` | `363` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_03` | `363` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_04` | `363` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `target_00` | `363` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `3.552713678800501e-16/0.1` | `0.010471975511966214/0.010471975511965976` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_00` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_01` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982988/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_02` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_03` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982988/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_04` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982988/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_05` | `363` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_06` | `363` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_07` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_08` | `363` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_09` | `363` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_10` | `363` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_11` | `363` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |

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
- [screenshots/mobile-stage-resource_threshold.png](screenshots/mobile-stage-resource_threshold.png)
- [screenshots/mobile-stage-uuv_deployed.png](screenshots/mobile-stage-uuv_deployed.png)
- [screenshots/mobile.png](screenshots/mobile.png)

## Browser Diagnostics

- `desktop:console:Failed to load resource: the server responded with a status of 422 (Unprocessable Content) (url=http://127.0.0.1:58909/api/assistant/memory/stream?user_id=operator&conversation_id=conversation-ebb0dd17-fcca-4a3e-9c2a-ffe911bb733a&after_cursor=11&limit=100&scenario_id=uuv-only-single-target)`
- `desktop:console:Failed to load resource: the server responded with a status of 500 (Internal Server Error) (url=http://127.0.0.1:58909/api/assistant/memory/stream?user_id=operator&conversation_id=conversation-ebb0dd17-fcca-4a3e-9c2a-ffe911bb733a&after_cursor=16&limit=100&scenario_id=uuv-only-single-target)`
- `desktop:pageerror:missing_ui_surface:sim_time`
- `mobile:console:Failed to load resource: the server responded with a status of 500 (Internal Server Error) (url=http://127.0.0.1:58909/api/assistant/memory/stream?user_id=operator&conversation_id=conversation-4de47555-d50a-4975-8fd9-1c3e0a337492&after_cursor=2&limit=100&scenario_id=uuv-only-single-target)`
- `mobile:pageerror:missing_ui_surface:sim_time`
- `mobile:requestfailed:memory_ui_http_422`

## Violations

- missing_stages:carrier_returned,handoff,uuv_recovered
- physics_frame_count_mismatch:364!=361
- physics_step_count_mismatch:carrier_01:363!=360
- physics_step_count_mismatch:carrier_02:363!=360
- physics_step_count_mismatch:carrier_03:363!=360
- physics_step_count_mismatch:carrier_04:363!=360
- physics_step_count_mismatch:target_00:363!=360
- physics_step_count_mismatch:uuv_00:363!=360
- physics_step_count_mismatch:uuv_01:363!=360
- physics_step_count_mismatch:uuv_02:363!=360
- physics_step_count_mismatch:uuv_03:363!=360
- physics_step_count_mismatch:uuv_04:363!=360
- physics_step_count_mismatch:uuv_05:363!=360
- physics_step_count_mismatch:uuv_06:363!=360
- physics_step_count_mismatch:uuv_07:363!=360
- physics_step_count_mismatch:uuv_08:363!=360
- physics_step_count_mismatch:uuv_09:363!=360
- physics_step_count_mismatch:uuv_10:363!=360
- physics_step_count_mismatch:uuv_11:363!=360
- missing_blue_tracking_evidence_chain
- incomplete_blue_tracking_candidates:target_00:cell:-1:-3,target_00:cell:-3:2
- missing_prediction_diff
- missing_counter_tracking_evidence_chain
- browser_errors:5
- failed_requests:1
- battle_phase_not_completed:running
