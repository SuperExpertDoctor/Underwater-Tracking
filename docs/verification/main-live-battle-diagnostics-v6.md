# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `dad5829030dddafbdc7f7451c235e6e20cc97d52`
- Config SHA-256: `6fee101518f0d238ffc31dab93deae557ad1ca34f7833680b07f7b11a785428a`
- Wall-clock start (UTC): `2026-08-23T12:28:54.358847+00:00`
- Wall-clock end (UTC): `2026-08-23T12:34:22.253853+00:00`
- First plan latency: `119.34333126398269` s
- Final run phase: `running`
- Final simulation time: `1810` s
- Final plan version: `1`
- Motion audits: `17`
- Physics frames observed/expected: `381/361`
- Browser errors: `0`
- Failed requests: `1`
- Memory events: `20`
- API p95: `130.037` ms
- Output bytes: `33052826`
- Shutdown: `6.628` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `95` | `1` |
| `carrier_dispatch` | `90` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `passive_track` | `90` | `1` |
| `recovery` | `625` | `1` |
| `uuv_deployed` | `90` | `1` |
| `uuv_recovered` | `1000` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `380` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_02` | `380` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_03` | `380` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_04` | `380` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `target_00` | `380` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `3.552713678800501e-16/0.1` | `0.010471975511966214/0.010471975511965976` | `n/a` | `0` |
| `uuv_00` | `380` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_01` | `380` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_02` | `380` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_03` | `380` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_04` | `380` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_05` | `380` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_06` | `380` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_07` | `380` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_08` | `380` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_09` | `380` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_10` | `380` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_11` | `380` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |

## Evidence Chains


## Screenshots

- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile.png](screenshots/mobile.png)
- [screenshots/desktop-latest.png](screenshots/desktop-latest.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)

## Violations

- memory_request_failed:HTTPError
- simulation_exceeded_duration
- missing_stages:carrier_returned,handoff,resource_threshold
- physics_frame_count_mismatch:381!=361
- physics_step_count_mismatch:carrier_01:380!=360
- physics_step_count_mismatch:carrier_02:380!=360
- physics_step_count_mismatch:carrier_03:380!=360
- physics_step_count_mismatch:carrier_04:380!=360
- physics_step_count_mismatch:target_00:380!=360
- physics_step_count_mismatch:uuv_00:380!=360
- physics_step_count_mismatch:uuv_01:380!=360
- physics_step_count_mismatch:uuv_02:380!=360
- physics_step_count_mismatch:uuv_03:380!=360
- physics_step_count_mismatch:uuv_04:380!=360
- physics_step_count_mismatch:uuv_05:380!=360
- physics_step_count_mismatch:uuv_06:380!=360
- physics_step_count_mismatch:uuv_07:380!=360
- physics_step_count_mismatch:uuv_08:380!=360
- physics_step_count_mismatch:uuv_09:380!=360
- physics_step_count_mismatch:uuv_10:380!=360
- physics_step_count_mismatch:uuv_11:380!=360
- missing_counter_tracking_evidence_chain
- battle_phase_not_completed:running
