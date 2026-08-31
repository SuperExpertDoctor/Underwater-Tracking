# Truthful Bootstrap and UI State Acceptance

Date: 2026-08-22
Branch: `feature/end-to-end-adversarial-runtime-remediation`
Seed: `20260820` for the default-entry semantic timeline

## Backend gate

Command:

```text
PYTHONPATH=src python -m pytest tests/config/test_uuv_only_config.py tests/domain/test_target_search_prior.py tests/integration/test_platform_core_scenario.py tests/simulation/test_execution_group_activation.py tests/runtime/test_observation_boundary.py tests/runtime/test_mission_controller.py tests/api/test_uuv_only_frame_contract.py tests/integration/test_truthful_bootstrap_deployment_frames.py tests/integration/test_uuv_initialization_local_perception.py -q
```

Result: `113 passed in 36.03s`.

The default entry path now records semantic checkpoints rather than relying on
fixed ticks: the initial frame contains 12 onboard UUV resources and one
source-attributed prior; the pre-deployment frame contains a planned mother-ship
assignment with no exposed UUV; the deployment frame contains the deployment
event, exposed assigned UUVs, and an execution group while `groups` and target
estimates remain empty; later frames retain the physical group without inventing
a belief. The test also verifies no USV payload, deterministic initial snapshots,
the configured target distance/detection boundary, carrier route motion, and
mother-ship slot offsets.

The existing local-perception integration test now uses the unmodified default
configuration and `_mission_controller_for`; it no longer calls the old
co-location helper.

## Frontend gate

```text
npm --prefix src/underwater_tracking/ui test -- --run
```

Result: `22 files, 106 tests passed`.

```text
npm --prefix src/underwater_tracking/ui run build
```

Result: TypeScript checking and Vite production build passed.

The added UI checks cover camera framing from carrier positions and a public
search prior without an estimate, all 12 permanent UUV owner labels, planned
assignment wording, and modern ready-brain rendering without legacy adversary
card synthesis.

## Static checks

```text
ruff check main.py src tests
```

Result: passed.

```text
mypy src/underwater_tracking
```

Result: non-zero with `211 errors in 29 files`. The remaining errors are the
repository's broad existing NumPy, planning, legacy adversary, and runtime
typing debt. The new `ledger.py` and `observation_boundary.py` implementations
are clean under targeted mypy; the phase-one shutdown timeout annotation was
also corrected. No credentials or real-provider tests were used for this gate.

## Evidence

The semantic integration test is
`tests/integration/test_truthful_bootstrap_deployment_frames.py`. The physical
boundary regression is covered by
`tests/runtime/test_observation_boundary.py` and
`tests/simulation/test_execution_group_activation.py`. User-provided reference
screenshots and untracked design documents were not modified or staged.
