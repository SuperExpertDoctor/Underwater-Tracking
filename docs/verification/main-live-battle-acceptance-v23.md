# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `83f0662988654d7703b174db2cea081a8ef54d5e`
- Config SHA-256: `69eb837c7e6eeb9a572d4730c5260a24f79ec2a4a8939868a502e854d1b9929b`
- Wall-clock start (UTC): `2026-08-24T01:57:05.154102+00:00`
- Wall-clock end (UTC): `2026-08-24T02:00:19.916892+00:00`
- First plan latency: `142.9897340129828` s
- Final run phase: `failed`
- Final simulation time: `930` s
- Final plan version: `3`
- Motion audits: `17`
- Physics frames observed/expected: `187/5761`
- Browser errors: `59`
- Failed requests: `0`
- Memory events: `12`
- API p95: `38.467` ms
- Output bytes: `15336431`
- Shutdown: `1.571` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `125` | `1` |
| `carrier_dispatch` | `125` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `passive_track` | `150` | `1` |
| `recovery` | `915` | `3` |
| `resource_threshold` | `630` | `2` |
| `uuv_deployed` | `125` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Total / teleport / boundary / owner / route / formation / resource |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `186` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_02` | `186` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_03` | `186` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_04` | `186` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.0/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `target_00` | `186` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `7.105427357601002e-16/0.1` | `0.010471975511966214/0.010471975511965976` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_00` | `186` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_01` | `186` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_02` | `186` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_03` | `186` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_04` | `186` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_05` | `186` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_06` | `186` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982989/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_07` | `186` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_08` | `186` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_09` | `186` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_10` | `186` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_11` | `186` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |

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

- `desktop:pageerror:ui_surface_probe:TimeoutError`
- `desktop:pageerror:missing_ui_surface:mission_panel`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`
- `desktop:pageerror:missing_ui_tab:时间线`
- `desktop:pageerror:missing_ui_tab:LLM 思考过程`
- `desktop:pageerror:missing_ui_tab:Memory Steam`

## Violations

- run_phase:failed
- persisted_replay_terminal_mismatch
- missing_stages:carrier_returned,handoff,uuv_recovered
- missing_blue_tracking_evidence_chain
- missing_prediction_diff
- missing_counter_tracking_evidence_chain
- browser_errors:59
- battle_not_completed
