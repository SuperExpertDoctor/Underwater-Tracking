# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `dad5829030dddafbdc7f7451c235e6e20cc97d52`
- Config SHA-256: `6fee101518f0d238ffc31dab93deae557ad1ca34f7833680b07f7b11a785428a`
- Wall-clock start (UTC): `2026-08-23T08:17:41.393336+00:00`
- Wall-clock end (UTC): `2026-08-23T08:19:27.705030+00:00`
- First plan latency: `85.25729561096523` s
- Final run phase: `running`
- Final simulation time: `620` s
- Final plan version: `1`
- Motion audits: `17`
- Physics frames observed/expected: `135/121`
- Browser errors: `0`
- Failed requests: `0`
- Memory events: `7`
- API p95: `125.287` ms
- Output bytes: `12079240`
- Shutdown: `5.476` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `initial_plan_committed` | `0` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `134` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_02` | `134` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_03` | `134` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_04` | `134` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `target_00` | `134` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `3.552713678800501e-16/0.1` | `0.010471975511966014/0.010471975511965976` | `n/a` | `0` |
| `uuv_00` | `134` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_01` | `134` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_02` | `134` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_03` | `134` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_04` | `134` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_05` | `134` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_06` | `134` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_07` | `134` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_08` | `134` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_09` | `134` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_10` | `134` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_11` | `134` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |

## Evidence Chains


## Screenshots

- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile.png](screenshots/mobile.png)
- [screenshots/desktop-latest.png](screenshots/desktop-latest.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)

## Violations

- planning_health_frame_mismatch
- simulation_exceeded_duration
- missing_stages:active_scan,carrier_dispatch,carrier_returned,handoff,passive_track,recovery,resource_threshold,uuv_deployed,uuv_recovered
- physics_frame_count_mismatch:135!=121
- physics_step_count_mismatch:carrier_01:134!=120
- physics_step_count_mismatch:carrier_02:134!=120
- physics_step_count_mismatch:carrier_03:134!=120
- physics_step_count_mismatch:carrier_04:134!=120
- physics_step_count_mismatch:target_00:134!=120
- physics_step_count_mismatch:uuv_00:134!=120
- physics_step_count_mismatch:uuv_01:134!=120
- physics_step_count_mismatch:uuv_02:134!=120
- physics_step_count_mismatch:uuv_03:134!=120
- physics_step_count_mismatch:uuv_04:134!=120
- physics_step_count_mismatch:uuv_05:134!=120
- physics_step_count_mismatch:uuv_06:134!=120
- physics_step_count_mismatch:uuv_07:134!=120
- physics_step_count_mismatch:uuv_08:134!=120
- physics_step_count_mismatch:uuv_09:134!=120
- physics_step_count_mismatch:uuv_10:134!=120
- physics_step_count_mismatch:uuv_11:134!=120
- missing_counter_tracking_evidence_chain
- battle_phase_not_completed:running
