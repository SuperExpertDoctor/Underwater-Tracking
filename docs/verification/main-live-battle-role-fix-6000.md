# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `dad5829030dddafbdc7f7451c235e6e20cc97d52`
- Config SHA-256: `6fee101518f0d238ffc31dab93deae557ad1ca34f7833680b07f7b11a785428a`
- Wall-clock start (UTC): `2026-08-23T03:39:56.146158+00:00`
- Wall-clock end (UTC): `2026-08-23T03:44:15.632175+00:00`
- First plan latency: `105.4425582770491` s
- Final run phase: `running`
- Final simulation time: `6020` s
- Final plan version: `1`
- Motion audits: `17`
- Physics frames observed/expected: `1240/1201`
- Browser errors: `0`
- Failed requests: `0`
- Memory events: `10`
- API p95: `225.43` ms
- Output bytes: `41476807`
- Shutdown: `10.114` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `625` | `1` |
| `carrier_dispatch` | `620` | `1` |
| `carrier_returned` | `4820` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `passive_track` | `630` | `1` |
| `recovery` | `2100` | `1` |
| `uuv_deployed` | `620` | `1` |
| `uuv_recovered` | `2130` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `1239` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_02` | `1239` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_03` | `1239` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_04` | `1239` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `target_00` | `1239` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `7.105427357601002e-16/0.1` | `0.010471975511966214/0.010471975511965976` | `n/a` | `0` |
| `uuv_00` | `1239` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_01` | `1239` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_02` | `1239` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_03` | `1239` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_04` | `1239` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_05` | `1239` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_06` | `1239` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_07` | `1239` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_08` | `1239` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_09` | `1239` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_10` | `1239` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_11` | `1239` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |

## Evidence Chains


## Screenshots

- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile.png](screenshots/mobile.png)
- [screenshots/desktop-latest.png](screenshots/desktop-latest.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)

## Violations

- planning_health_frame_mismatch
- simulation_exceeded_duration
- missing_stages:handoff,resource_threshold
- api_p95_exceeded_200ms
- physics_frame_count_mismatch:1240!=1201
- physics_step_count_mismatch:carrier_01:1239!=1200
- physics_step_count_mismatch:carrier_02:1239!=1200
- physics_step_count_mismatch:carrier_03:1239!=1200
- physics_step_count_mismatch:carrier_04:1239!=1200
- physics_step_count_mismatch:target_00:1239!=1200
- physics_step_count_mismatch:uuv_00:1239!=1200
- physics_step_count_mismatch:uuv_01:1239!=1200
- physics_step_count_mismatch:uuv_02:1239!=1200
- physics_step_count_mismatch:uuv_03:1239!=1200
- physics_step_count_mismatch:uuv_04:1239!=1200
- physics_step_count_mismatch:uuv_05:1239!=1200
- physics_step_count_mismatch:uuv_06:1239!=1200
- physics_step_count_mismatch:uuv_07:1239!=1200
- physics_step_count_mismatch:uuv_08:1239!=1200
- physics_step_count_mismatch:uuv_09:1239!=1200
- physics_step_count_mismatch:uuv_10:1239!=1200
- physics_step_count_mismatch:uuv_11:1239!=1200
- missing_counter_tracking_evidence_chain
- battle_phase_not_completed:running
- shutdown_exceeded_10s
