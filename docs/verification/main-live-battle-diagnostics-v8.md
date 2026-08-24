# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `dad5829030dddafbdc7f7451c235e6e20cc97d52`
- Config SHA-256: `6fee101518f0d238ffc31dab93deae557ad1ca34f7833680b07f7b11a785428a`
- Wall-clock start (UTC): `2026-08-23T13:09:09.688266+00:00`
- Wall-clock end (UTC): `2026-08-23T13:11:58.875736+00:00`
- First plan latency: `129.208982859971` s
- Final run phase: `running`
- Final simulation time: `915` s
- Final plan version: `1`
- Motion audits: `17`
- Physics frames observed/expected: `192/181`
- Browser errors: `1`
- Failed requests: `0`
- Memory events: `8`
- API p95: `90.741` ms
- Output bytes: `18778351`
- Shutdown: `5.676` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `95` | `1` |
| `carrier_dispatch` | `90` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `passive_track` | `90` | `1` |
| `recovery` | `625` | `1` |
| `uuv_deployed` | `90` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `191` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_02` | `191` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_03` | `191` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_04` | `191` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `target_00` | `191` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `3.552713678800501e-16/0.1` | `0.010471975511966214/0.010471975511965976` | `n/a` | `0` |
| `uuv_00` | `191` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_01` | `191` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_02` | `191` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_03` | `191` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_04` | `191` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_05` | `191` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_06` | `191` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_07` | `191` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_08` | `191` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_09` | `191` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_10` | `191` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_11` | `191` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |

## Evidence Chains


## Screenshots

- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile.png](screenshots/mobile.png)
- [screenshots/desktop-latest.png](screenshots/desktop-latest.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)

## Violations

- simulation_exceeded_duration
- missing_stages:carrier_returned,handoff,resource_threshold,uuv_recovered
- physics_frame_count_mismatch:192!=181
- physics_step_count_mismatch:carrier_01:191!=180
- physics_step_count_mismatch:carrier_02:191!=180
- physics_step_count_mismatch:carrier_03:191!=180
- physics_step_count_mismatch:carrier_04:191!=180
- physics_step_count_mismatch:target_00:191!=180
- physics_step_count_mismatch:uuv_00:191!=180
- physics_step_count_mismatch:uuv_01:191!=180
- physics_step_count_mismatch:uuv_02:191!=180
- physics_step_count_mismatch:uuv_03:191!=180
- physics_step_count_mismatch:uuv_04:191!=180
- physics_step_count_mismatch:uuv_05:191!=180
- physics_step_count_mismatch:uuv_06:191!=180
- physics_step_count_mismatch:uuv_07:191!=180
- physics_step_count_mismatch:uuv_08:191!=180
- physics_step_count_mismatch:uuv_09:191!=180
- physics_step_count_mismatch:uuv_10:191!=180
- physics_step_count_mismatch:uuv_11:191!=180
- missing_counter_tracking_evidence_chain
- browser_errors:1
- battle_phase_not_completed:running
