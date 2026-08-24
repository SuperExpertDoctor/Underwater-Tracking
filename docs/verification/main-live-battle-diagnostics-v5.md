# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `dad5829030dddafbdc7f7451c235e6e20cc97d52`
- Config SHA-256: `6fee101518f0d238ffc31dab93deae557ad1ca34f7833680b07f7b11a785428a`
- Wall-clock start (UTC): `2026-08-23T12:17:17.375747+00:00`
- Wall-clock end (UTC): `2026-08-23T12:28:07.796174+00:00`
- First plan latency: `119.47548106196336` s
- Final run phase: `running`
- Final simulation time: `1745` s
- Final plan version: `1`
- Motion audits: `17`
- Physics frames observed/expected: `351/361`
- Browser errors: `1`
- Failed requests: `0`
- Memory events: `47`
- API p95: `66.854` ms
- Output bytes: `29950124`
- Shutdown: `10.114` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `95` | `1` |
| `carrier_dispatch` | `90` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `passive_track` | `90` | `1` |
| `recovery` | `590` | `1` |
| `uuv_deployed` | `90` | `1` |
| `uuv_recovered` | `1000` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `350` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `carrier_02` | `350` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_03` | `350` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0` |
| `carrier_04` | `350` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0` |
| `target_00` | `350` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `3.552713678800501e-16/0.1` | `0.010471975511966214/0.010471975511965976` | `n/a` | `0` |
| `uuv_00` | `350` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_01` | `350` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_02` | `350` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_03` | `350` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_04` | `350` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_05` | `350` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_06` | `350` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0` |
| `uuv_07` | `350` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0` |
| `uuv_08` | `350` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_09` | `350` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_10` | `350` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |
| `uuv_11` | `350` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0` |

## Evidence Chains


## Screenshots

- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile.png](screenshots/mobile.png)
- [screenshots/desktop-latest.png](screenshots/desktop-latest.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)

## Violations

- planning_degraded:deployment_changed; deployment_changed; deployment_changed; deployment_changed; resource_unavailable; resource_unavailable
- missing_stages:carrier_returned,handoff,resource_threshold
- real_provider_unavailable
- missing_counter_tracking_evidence_chain
- browser_errors:1
- battle_not_completed
- shutdown_exceeded_10s
