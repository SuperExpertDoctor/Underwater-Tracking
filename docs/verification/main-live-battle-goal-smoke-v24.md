# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `83f0662988654d7703b174db2cea081a8ef54d5e`
- Config SHA-256: `69eb837c7e6eeb9a572d4730c5260a24f79ec2a4a8939868a502e854d1b9929b`
- Wall-clock start (UTC): `2026-08-24T02:08:49.560926+00:00`
- Wall-clock end (UTC): `2026-08-24T02:16:09.813889+00:00`
- First plan latency: `74.60717493796255` s
- Final run phase: `running`
- Final simulation time: `1325` s
- Final plan version: `9`
- Motion audits: `17`
- Physics frames observed/expected: `269/361`
- Browser errors: `0`
- Failed requests: `41`
- Memory events: `57`
- API p95: `621.443` ms
- Output bytes: `24162660`
- Shutdown: `3.528` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `130` | `1` |
| `carrier_dispatch` | `90` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `passive_track` | `150` | `1` |
| `recovery` | `920` | `3` |
| `resource_threshold` | `630` | `2` |
| `uuv_deployed` | `90` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Total / teleport / boundary / owner / route / formation / resource |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `268` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_02` | `268` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_03` | `268` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_04` | `268` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `target_00` | `268` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `3.552713678800501e-16/0.1` | `0.010471975511966014/0.010471975511965976` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_00` | `268` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_01` | `268` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_02` | `268` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_03` | `268` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_04` | `268` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_05` | `268` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_06` | `268` | `4.0/4.0` | `0.1/0.1` | `0.0/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_07` | `268` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_08` | `268` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_09` | `268` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_10` | `268` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_11` | `268` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |

## Evidence Chains


## Blue Tracking Chains

| Target | Carrier / candidate | UUVs | Dispatch | Deploy | Active ping | Estimates | Handoff | Resource | Recovery | Recovered | Carrier return | Plan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |

## Prediction Intent Chains

| Target | Diff / thresholds | Window (s) | Suspicion | Intent provider / calls | Confirmation | Plan | Response latency | Blue response |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |

## Screenshots

- [screenshots/desktop-latest.png](screenshots/desktop-latest.png)
- [screenshots/desktop-stage-active_scan.png](screenshots/desktop-stage-active_scan.png)
- [screenshots/desktop-stage-carrier_dispatch.png](screenshots/desktop-stage-carrier_dispatch.png)
- [screenshots/desktop-stage-initial_plan_committed.png](screenshots/desktop-stage-initial_plan_committed.png)
- [screenshots/desktop-stage-passive_track.png](screenshots/desktop-stage-passive_track.png)
- [screenshots/desktop-stage-recovery.png](screenshots/desktop-stage-recovery.png)
- [screenshots/desktop-stage-resource_threshold.png](screenshots/desktop-stage-resource_threshold.png)
- [screenshots/desktop-stage-uuv_deployed.png](screenshots/desktop-stage-uuv_deployed.png)
- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)
- [screenshots/mobile-stage-active_scan.png](screenshots/mobile-stage-active_scan.png)
- [screenshots/mobile-stage-carrier_dispatch.png](screenshots/mobile-stage-carrier_dispatch.png)
- [screenshots/mobile-stage-initial_plan_committed.png](screenshots/mobile-stage-initial_plan_committed.png)
- [screenshots/mobile-stage-passive_track.png](screenshots/mobile-stage-passive_track.png)
- [screenshots/mobile-stage-recovery.png](screenshots/mobile-stage-recovery.png)
- [screenshots/mobile-stage-resource_threshold.png](screenshots/mobile-stage-resource_threshold.png)
- [screenshots/mobile-stage-uuv_deployed.png](screenshots/mobile-stage-uuv_deployed.png)
- [screenshots/mobile.png](screenshots/mobile.png)

## Browser Diagnostics

- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `desktop:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`
- `mobile:requestfailed:memory_ui_probe:TimeoutError`

## Violations

- wall_timeout
- persisted_replay_terminal_mismatch
- missing_stages:carrier_returned,handoff,uuv_recovered
- verification_evidence_request_failed:TimeoutError
- verification_evidence_unavailable
- adversary_llm_decision_not_observed
- api_p95_exceeded_200ms
- missing_blue_tracking_evidence_chain
- real_provider_attestation_unavailable
- battle_evidence_unavailable
- verification_requests_failed:1
- missing_counter_tracking_evidence_chain
- missing_adversary_llm_decision
- failed_requests:39
- battle_not_completed
