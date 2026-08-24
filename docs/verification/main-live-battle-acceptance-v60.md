# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `41d2c3a788192d706f5e87e4f0c4223870bf2d5f`
- Config SHA-256: `0ea7495cff65ff4f1ef44fb9ae8110d467a5027badb6ea932418637bba95cf31`
- Wall-clock start (UTC): `2026-08-24T17:40:08.162671+00:00`
- Wall-clock end (UTC): `2026-08-24T18:10:49.400781+00:00`
- First plan latency: `105.21807617903687` s
- Final run phase: `running`
- Final simulation time: `24140` s
- Final plan version: `30`
- Motion audits: `17`
- Physics frames observed/expected: `4900/5761`
- Browser errors: `10`
- Failed requests: `0`
- Memory events: `100`
- API p95: `195.073` ms
- Output bytes: `223745139`
- Shutdown: `9.286` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |
| `active_scan` | `90` | `1` |
| `carrier_dispatch` | `90` | `1` |
| `carrier_returned` | `10980` | `10` |
| `handoff` | `1260` | `1` |
| `initial_plan_committed` | `0` | `1` |
| `passive_track` | `90` | `1` |
| `recovery` | `1980` | `1` |
| `resource_threshold` | `1980` | `1` |
| `uuv_deployed` | `90` | `1` |
| `uuv_recovered` | `3060` | `1` |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Total / teleport / boundary / owner / route / formation / resource |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `carrier_01` | `4899` | `4.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_02` | `4899` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_03` | `4899` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `carrier_04` | `4899` | `8.0/8.0` | `0.0/0.25` | `0.0/0.25` | `0.25/0.25` | `n/a` | `0/0/0/0/0/0/0` |
| `target_00` | `4899` | `8.400000000000002/14.0` | `0.08000000000000007/0.08` | `0.10000000000000106/0.1` | `0.010471975511966214/0.010471975511965976` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_00` | `4899` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_01` | `4899` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_02` | `4899` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_03` | `4899` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_04` | `4899` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_05` | `4899` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_06` | `4899` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_07` | `4899` | `4.0/4.0` | `0.1/0.1` | `0.1/0.1` | `0.05235987755982992/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_08` | `4899` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_09` | `4899` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_10` | `4899` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |
| `uuv_11` | `4899` | `0.0/4.0` | `0.0/0.1` | `0.0/0.1` | `0.0/0.05235987755982988` | `n/a` | `0/0/0/0/0/0/0` |

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
- [screenshots/desktop-stage-recovery.png](screenshots/desktop-stage-recovery.png)
- [screenshots/desktop-stage-resource_threshold.png](screenshots/desktop-stage-resource_threshold.png)
- [screenshots/desktop-stage-uuv_deployed.png](screenshots/desktop-stage-uuv_deployed.png)
- [screenshots/desktop-stage-uuv_recovered.png](screenshots/desktop-stage-uuv_recovered.png)
- [screenshots/desktop.png](screenshots/desktop.png)
- [screenshots/mobile-latest.png](screenshots/mobile-latest.png)
- [screenshots/mobile-stage-active_scan.png](screenshots/mobile-stage-active_scan.png)
- [screenshots/mobile-stage-carrier_dispatch.png](screenshots/mobile-stage-carrier_dispatch.png)
- [screenshots/mobile-stage-carrier_returned.png](screenshots/mobile-stage-carrier_returned.png)
- [screenshots/mobile-stage-handoff.png](screenshots/mobile-stage-handoff.png)
- [screenshots/mobile-stage-initial_plan_committed.png](screenshots/mobile-stage-initial_plan_committed.png)
- [screenshots/mobile-stage-passive_track.png](screenshots/mobile-stage-passive_track.png)
- [screenshots/mobile-stage-recovery.png](screenshots/mobile-stage-recovery.png)
- [screenshots/mobile-stage-resource_threshold.png](screenshots/mobile-stage-resource_threshold.png)
- [screenshots/mobile-stage-uuv_deployed.png](screenshots/mobile-stage-uuv_deployed.png)
- [screenshots/mobile-stage-uuv_recovered.png](screenshots/mobile-stage-uuv_recovered.png)
- [screenshots/mobile.png](screenshots/mobile.png)

## Browser Diagnostics

- `desktop:pageerror:missing_ui_content:memory_event_content:TimeoutError`
- `desktop:pageerror:missing_ui_content:memory_event_content:TimeoutError`
- `desktop:pageerror:missing_ui_content:llm_thinking_content:TimeoutError`
- `desktop:pageerror:missing_ui_content:llm_thinking_content:TimeoutError`
- `desktop:pageerror:missing_ui_content:llm_thinking_content:TimeoutError`
- `desktop:pageerror:missing_ui_content:llm_thinking_content:TimeoutError`
- `mobile:pageerror:missing_ui_content:llm_thinking_content:TimeoutError`
- `mobile:pageerror:missing_ui_content:llm_thinking_content:TimeoutError`
- `mobile:pageerror:missing_ui_content:memory_event_content:TimeoutError`
- `mobile:pageerror:missing_ui_content:task_timeline_content:TimeoutError`

## Violations

- ledger_trigger_missing_from_events
- plan_timeline_event_missing_from_events
- llm_thinking_source_missing_from_events
- memory_source_missing_from_operational_views
- wall_timeout
- persisted_replay_terminal_mismatch
- missing_blue_tracking_evidence_chain
- incomplete_blue_tracking_candidates:target_00:cell:-1:-10,target_00:cell:-1:-9
- missing_intent_confirmation
- missing_counter_tracking_evidence_chain
- browser_errors:10
- battle_not_completed
