# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `dad5829030dddafbdc7f7451c235e6e20cc97d52`
- Config SHA-256: `6fee101518f0d238ffc31dab93deae557ad1ca34f7833680b07f7b11a785428a`
- Wall-clock start (UTC): `2026-08-23T03:09:38.998562+00:00`
- Wall-clock end (UTC): `2026-08-23T03:12:47.621791+00:00`
- First plan latency: `119.32634920597775` s
- Final run phase: `running`
- Final simulation time: `3050` s
- Final plan version: `1`
- Motion audits: `17`
- Physics frames observed/expected: `615/601`
- Browser errors: `1`
- Failed requests: `1`
- Memory events: `12`
- API p95: `118.794` ms
- Output bytes: `21989855`
- Shutdown: `5.523` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `655` | `1` |
| `carrier_dispatch` | `655` | `1` |
| `initial_plan_committed` | `65` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `614` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_02` | `614` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_03` | `614` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_04` | `614` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `target_00` | `614` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `3.552713678800501e-16/0.1` | `0.010471975511966214/0.010471975511965976` | `n/a` | `0` |
| `uuv_00` | `614` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_01` | `614` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_02` | `614` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_03` | `614` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_04` | `614` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_05` | `614` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_06` | `614` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_07` | `614` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_08` | `614` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_09` | `614` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_10` | `614` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_11` | `614` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |

## Evidence Chains


## Screenshots

- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile.png](screenshots/mobile.png)
- [screenshots/desktop-latest.png](screenshots/desktop-latest.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)

## Violations

- planning_health_frame_mismatch
- memory_request_failed:HTTPError
- simulation_exceeded_duration
- missing_stages:carrier_returned,handoff,passive_track,recovery,resource_threshold,uuv_deployed,uuv_recovered
- physics_frame_count_mismatch:615!=601
- physics_step_count_mismatch:carrier_01:614!=600
- physics_step_count_mismatch:carrier_02:614!=600
- physics_step_count_mismatch:carrier_03:614!=600
- physics_step_count_mismatch:carrier_04:614!=600
- physics_step_count_mismatch:target_00:614!=600
- physics_step_count_mismatch:uuv_00:614!=600
- physics_step_count_mismatch:uuv_01:614!=600
- physics_step_count_mismatch:uuv_02:614!=600
- physics_step_count_mismatch:uuv_03:614!=600
- physics_step_count_mismatch:uuv_04:614!=600
- physics_step_count_mismatch:uuv_05:614!=600
- physics_step_count_mismatch:uuv_06:614!=600
- physics_step_count_mismatch:uuv_07:614!=600
- physics_step_count_mismatch:uuv_08:614!=600
- physics_step_count_mismatch:uuv_09:614!=600
- physics_step_count_mismatch:uuv_10:614!=600
- physics_step_count_mismatch:uuv_11:614!=600
- missing_counter_tracking_evidence_chain
- browser_errors:1
- battle_phase_not_completed:running
