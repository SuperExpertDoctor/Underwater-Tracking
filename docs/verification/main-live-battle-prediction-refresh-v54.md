# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `6d3373b7cd7fc4ebd318a5122c0112cf130bf4b8`
- Config SHA-256: `989d9241d6247162b335d7ee6658133165d00056b6a4282d45d02cfa8bd42951`
- Wall-clock start (UTC): `2026-08-24T14:44:51.772929+00:00`
- Wall-clock end (UTC): `2026-08-24T14:48:46.562062+00:00`
- First plan latency: `67.04011577903293` s
- Final run phase: `completed`
- Final simulation time: `4500` s
- Final plan version: `1`
- Motion audits: `17`
- Physics frames observed/expected: `901/901`
- Browser errors: `0`
- Failed requests: `0`
- Memory events: `20`
- API p95: `88.253` ms
- Output bytes: `96758823`
- Shutdown: `10.235` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `90` | `1` |
| `carrier_dispatch` | `90` | `1` |
| `carrier_returned` | `4050` | `1` |
| `handoff` | `1260` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `passive_track` | `90` | `1` |
| `recovery` | `1260` | `1` |
| `resource_threshold` | `375` | `1` |
| `uuv_deployed` | `90` | `1` |
| `uuv_recovered` | `1520` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Total / teleport / boundary / owner / route / formation / resource |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `900` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_02` | `900` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_03` | `900` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_04` | `900` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `target_00` | `900` | `8.400000000000002/14.0` | `0.0799999999999983/0.08` | `7.105427357601002e-16/0.1` | `0.010471975511966214/0.010471975511965976` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_00` | `900` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_01` | `900` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_02` | `900` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_03` | `900` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_04` | `900` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_05` | `900` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_06` | `900` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_07` | `900` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_08` | `900` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_09` | `900` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_10` | `900` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_11` | `900` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |

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
- [screenshots/desktop-stage-carrier_returned.png](screenshots/desktop-stage-carrier_returned.png)
- [screenshots/desktop-stage-handoff.png](screenshots/desktop-stage-handoff.png)
- [screenshots/desktop-stage-initial_plan_committed.png](screenshots/desktop-stage-initial_plan_committed.png)
- [screenshots/desktop-stage-passive_track.png](screenshots/desktop-stage-passive_track.png)
- [screenshots/desktop-stage-resource_threshold.png](screenshots/desktop-stage-resource_threshold.png)
- [screenshots/desktop-stage-uuv_deployed.png](screenshots/desktop-stage-uuv_deployed.png)
- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)
- [screenshots/mobile-stage-active_scan.png](screenshots/mobile-stage-active_scan.png)
- [screenshots/mobile-stage-carrier_dispatch.png](screenshots/mobile-stage-carrier_dispatch.png)
- [screenshots/mobile-stage-carrier_returned.png](screenshots/mobile-stage-carrier_returned.png)
- [screenshots/mobile-stage-handoff.png](screenshots/mobile-stage-handoff.png)
- [screenshots/mobile-stage-initial_plan_committed.png](screenshots/mobile-stage-initial_plan_committed.png)
- [screenshots/mobile-stage-passive_track.png](screenshots/mobile-stage-passive_track.png)
- [screenshots/mobile-stage-resource_threshold.png](screenshots/mobile-stage-resource_threshold.png)
- [screenshots/mobile-stage-uuv_deployed.png](screenshots/mobile-stage-uuv_deployed.png)
- [screenshots/mobile.png](screenshots/mobile.png)

## Browser Diagnostics


## Violations

- ledger_plan_version_ahead_of_frame
- plan_timeline_version_ahead_of_frame
- stage_order:handoff_after_resource_threshold
- missing_blue_tracking_evidence_chain
- incomplete_blue_tracking_candidates:target_00:cell:-2:-4
- missing_intent_confirmation
- background_carrier_drain_failed
- missing_counter_tracking_evidence_chain
- shutdown_exceeded_10s
- main_process_exit:-9
