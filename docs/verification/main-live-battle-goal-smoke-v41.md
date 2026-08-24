# Main Live Battle Acceptance

- Status: **BLOCKED/FAIL**
- Git commit: `83f0662988654d7703b174db2cea081a8ef54d5e`
- Config SHA-256: `989d9241d6247162b335d7ee6658133165d00056b6a4282d45d02cfa8bd42951`
- Wall-clock start (UTC): `unavailable`
- Wall-clock end (UTC): `2026-08-24T09:06:31.082997+00:00`
- First plan latency: `unavailable` s
- Final run phase: `unknown`
- Final simulation time: `0` s
- Final plan version: `0`
- Motion audits: `0`
- Physics frames observed/expected: `0/unavailable`
- Browser errors: `1`
- Failed requests: `3`
- Memory events: `0`
- API p95: `0.0` ms
- Output bytes: `0`
- Shutdown: `10.726` s

## Stage Evidence

| Stage | Simulation time (s) | Plan version |
| --- | ---: | ---: |

## Entity Motion Audits

| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Total / teleport / boundary / owner / route / formation / resource |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |

## Evidence Chains


## Blue Tracking Chains

| Target | Carrier / candidate | UUVs | Dispatch | Deploy | Active ping | Estimates | Handoff | Resource | Recovery | Recovered | Carrier return | Plan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |

## Prediction Intent Chains

| Target | Diff / thresholds | Window (s) | Suspicion | Intent provider / calls | Confirmation | Plan | Response latency | Blue response |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |

## Screenshots


## Browser Diagnostics

- `browser_audit_unavailable`

## Violations

- api_boot_timeout
- physics_audit_unavailable
- physics_entity_count_mismatch
- missing_motion_entities:carrier_01,carrier_02,carrier_03,carrier_04,target_00,uuv_00,uuv_01,uuv_02,uuv_03,uuv_04,uuv_05,uuv_06,uuv_07,uuv_08,uuv_09,uuv_10,uuv_11
- motion_limits_entity_set_mismatch
- physics_frame_coverage_unavailable
- missing_blue_tracking_evidence_chain
- real_provider_attestation_unavailable
- battle_evidence_unavailable
- verification_requests_failed:2
- missing_counter_tracking_evidence_chain
- missing_adversary_llm_decision
- browser_errors:1
- failed_requests:1
- battle_not_completed
- shutdown_exceeded_10s
- main_process_exit:-9
