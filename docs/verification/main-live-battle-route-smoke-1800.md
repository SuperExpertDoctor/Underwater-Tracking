# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `dad5829030dddafbdc7f7451c235e6e20cc97d52`
- Config SHA-256: `6fee101518f0d238ffc31dab93deae557ad1ca34f7833680b07f7b11a785428a`
- Wall-clock start (UTC): `2026-08-23T08:41:07.245195+00:00`
- Wall-clock end (UTC): `2026-08-23T08:44:19.203511+00:00`
- First plan latency: `135.35861737799132` s
- Final run phase: `running`
- Final simulation time: `1820` s
- Final plan version: `1`
- Motion audits: `17`
- Physics frames observed/expected: `376/361`
- Browser errors: `2`
- Failed requests: `1`
- Memory events: `8`
- API p95: `139.826` ms
- Output bytes: `19668634`
- Shutdown: `5.776` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `845` | `1` |
| `carrier_dispatch` | `840` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `uuv_deployed` | `840` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `375` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_02` | `375` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_03` | `375` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_04` | `375` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `target_00` | `375` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `3.552713678800501e-16/0.1` | `0.010471975511966014/0.010471975511965976` | `n/a` | `0` |
| `uuv_00` | `375` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_01` | `375` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_02` | `375` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_03` | `375` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_04` | `375` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_05` | `375` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_06` | `375` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_07` | `375` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_08` | `375` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_09` | `375` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_10` | `375` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_11` | `375` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |

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
- missing_stages:carrier_returned,handoff,passive_track,recovery,resource_threshold,uuv_recovered
- physics_frame_count_mismatch:376!=361
- physics_step_count_mismatch:carrier_01:375!=360
- physics_step_count_mismatch:carrier_02:375!=360
- physics_step_count_mismatch:carrier_03:375!=360
- physics_step_count_mismatch:carrier_04:375!=360
- physics_step_count_mismatch:target_00:375!=360
- physics_step_count_mismatch:uuv_00:375!=360
- physics_step_count_mismatch:uuv_01:375!=360
- physics_step_count_mismatch:uuv_02:375!=360
- physics_step_count_mismatch:uuv_03:375!=360
- physics_step_count_mismatch:uuv_04:375!=360
- physics_step_count_mismatch:uuv_05:375!=360
- physics_step_count_mismatch:uuv_06:375!=360
- physics_step_count_mismatch:uuv_07:375!=360
- physics_step_count_mismatch:uuv_08:375!=360
- physics_step_count_mismatch:uuv_09:375!=360
- physics_step_count_mismatch:uuv_10:375!=360
- physics_step_count_mismatch:uuv_11:375!=360
- missing_counter_tracking_evidence_chain
- browser_errors:2
- battle_phase_not_completed:running
