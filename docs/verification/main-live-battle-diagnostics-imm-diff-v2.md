# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `83f0662988654d7703b174db2cea081a8ef54d5e`
- Config SHA-256: `6fee101518f0d238ffc31dab93deae557ad1ca34f7833680b07f7b11a785428a`
- Wall-clock start (UTC): `2026-08-23T17:29:08.111260+00:00`
- Wall-clock end (UTC): `2026-08-23T17:33:26.620560+00:00`
- First plan latency: `118.2276508109644` s
- Final run phase: `running`
- Final simulation time: `1800` s
- Final plan version: `4`
- Motion audits: `17`
- Physics frames observed/expected: `364/361`
- Browser errors: `0`
- Failed requests: `1`
- Memory events: `13`
- API p95: `91.494` ms
- Output bytes: `50402978`
- Shutdown: `7.88` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `95` | `1` |
| `carrier_dispatch` | `90` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `passive_track` | `90` | `1` |
| `recovery` | `590` | `2` |
| `uuv_deployed` | `90` | `1` |
| `uuv_recovered` | `985` | `3` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `363` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_02` | `363` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_03` | `363` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_04` | `363` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `target_00` | `363` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `3.552713678800501e-16/0.1` | `0.010471975511966214/0.010471975511965976` | `n/a` | `0` |
| `uuv_00` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_01` | `363` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_02` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_03` | `363` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_04` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982988/0.05235987755982988` | `n/a` | `0` |
| `uuv_05` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_06` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_07` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_08` | `363` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_09` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_10` | `363` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_11` | `363` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |

## Evidence Chains


## Prediction Intent Chains

| Target | Diff / thresholds | Window (s) | Suspicion | Intent provider / calls | Confirmation | Plan | Response latency | Blue response |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |

## Screenshots

- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile.png](screenshots/mobile.png)
- [screenshots/desktop-latest.png](screenshots/desktop-latest.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)

## Violations

- missing_stages:carrier_returned,handoff,resource_threshold
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
- battle_evidence_unavailable
- verification_requests_failed:1
- missing_counter_tracking_evidence_chain
- battle_phase_not_completed:running
