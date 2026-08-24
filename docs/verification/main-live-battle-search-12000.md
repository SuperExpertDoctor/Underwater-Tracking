# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `dad5829030dddafbdc7f7451c235e6e20cc97d52`
- Config SHA-256: `6fee101518f0d238ffc31dab93deae557ad1ca34f7833680b07f7b11a785428a`
- Wall-clock start (UTC): `2026-08-23T06:06:08.096919+00:00`
- Wall-clock end (UTC): `2026-08-23T06:11:22.002848+00:00`
- First plan latency: `81.03728071996011` s
- Final run phase: `running`
- Final simulation time: `8650` s
- Final plan version: `4`
- Motion audits: `17`
- Physics frames observed/expected: `1795/2401`
- Browser errors: `1`
- Failed requests: `1`
- Memory events: `14`
- API p95: `230.826` ms
- Output bytes: `83678184`
- Shutdown: `6.426` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `845` | `1` |
| `carrier_dispatch` | `840` | `1` |
| `carrier_returned` | `3510` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `recovery` | `1830` | `1` |
| `uuv_deployed` | `840` | `1` |
| `uuv_recovered` | `2345` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `1794` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_02` | `1794` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_03` | `1794` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_04` | `1794` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `target_00` | `1794` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `7.105427357601002e-16/0.1` | `0.010471975511966214/0.010471975511965976` | `n/a` | `0` |
| `uuv_00` | `1794` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_01` | `1794` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_02` | `1794` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982988/0.05235987755982988` | `n/a` | `0` |
| `uuv_03` | `1794` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982988/0.05235987755982988` | `n/a` | `0` |
| `uuv_04` | `1794` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_05` | `1794` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_06` | `1794` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_07` | `1794` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_08` | `1794` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_09` | `1794` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_10` | `1794` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_11` | `1794` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |

## Evidence Chains


## Screenshots

- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile.png](screenshots/mobile.png)
- [screenshots/desktop-latest.png](screenshots/desktop-latest.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)

## Violations

- planning_health_frame_mismatch
- memory_request_failed:HTTPError
- wall_timeout
- missing_stages:handoff,passive_track,resource_threshold
- api_p95_exceeded_200ms
- missing_counter_tracking_evidence_chain
- browser_errors:1
- battle_not_completed
