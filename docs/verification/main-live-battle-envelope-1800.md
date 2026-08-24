# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `dad5829030dddafbdc7f7451c235e6e20cc97d52`
- Config SHA-256: `6fee101518f0d238ffc31dab93deae557ad1ca34f7833680b07f7b11a785428a`
- Wall-clock start (UTC): `2026-08-23T07:26:23.918572+00:00`
- Wall-clock end (UTC): `2026-08-23T07:28:50.373669+00:00`
- First plan latency: `95.82944923097966` s
- Final run phase: `running`
- Final simulation time: `1810` s
- Final plan version: `1`
- Motion audits: `17`
- Physics frames observed/expected: `375/361`
- Browser errors: `2`
- Failed requests: `0`
- Memory events: `8`
- API p95: `133.539` ms
- Output bytes: `20822897`
- Shutdown: `5.526` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `carrier_dispatch` | `590` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `uuv_deployed` | `590` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `374` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_02` | `374` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_03` | `374` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_04` | `374` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `target_00` | `374` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `3.552713678800501e-16/0.1` | `0.010471975511966014/0.010471975511965976` | `n/a` | `0` |
| `uuv_00` | `374` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_01` | `374` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_02` | `374` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_03` | `374` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_04` | `374` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_05` | `374` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_06` | `374` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_07` | `374` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_08` | `374` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_09` | `374` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_10` | `374` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_11` | `374` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |

## Evidence Chains


## Screenshots

- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile.png](screenshots/mobile.png)
- [screenshots/desktop-latest.png](screenshots/desktop-latest.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)

## Violations

- planning_health_frame_mismatch
- simulation_exceeded_duration
- missing_stages:active_scan,carrier_returned,handoff,passive_track,recovery,resource_threshold,uuv_recovered
- physics_frame_count_mismatch:375!=361
- physics_step_count_mismatch:carrier_01:374!=360
- physics_step_count_mismatch:carrier_02:374!=360
- physics_step_count_mismatch:carrier_03:374!=360
- physics_step_count_mismatch:carrier_04:374!=360
- physics_step_count_mismatch:target_00:374!=360
- physics_step_count_mismatch:uuv_00:374!=360
- physics_step_count_mismatch:uuv_01:374!=360
- physics_step_count_mismatch:uuv_02:374!=360
- physics_step_count_mismatch:uuv_03:374!=360
- physics_step_count_mismatch:uuv_04:374!=360
- physics_step_count_mismatch:uuv_05:374!=360
- physics_step_count_mismatch:uuv_06:374!=360
- physics_step_count_mismatch:uuv_07:374!=360
- physics_step_count_mismatch:uuv_08:374!=360
- physics_step_count_mismatch:uuv_09:374!=360
- physics_step_count_mismatch:uuv_10:374!=360
- physics_step_count_mismatch:uuv_11:374!=360
- missing_counter_tracking_evidence_chain
- browser_errors:2
- battle_phase_not_completed:running
