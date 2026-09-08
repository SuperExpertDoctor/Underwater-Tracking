# UUV Tracking Control C-01-C-06 Verification

Status: **VERIFIED FOR BRANCH PUBLICATION**. The UUV tracking-control C-01-C-06
implementation passes the focused matrix, the complete repository pytest suite, the
long-duration integration checks, and a deterministic two-run production audit. This
status applies only to the UUV tracking-control responsibility; it does not claim that the
separate four-person joint acceptance is complete.

## Source State

- Worktree: `D:\Air\反Q\Underwater-Tracking-uuv-tracking-control-c01-c06-20260906`
- Branch: `fix/uuv-tracking-control-c01-c06-20260906`
- Base: `origin/master` at `2c444bb347c17fa25aa7ac136c70d3a234bf5fc6`
- Design commit: `745477f`
- Plan commit: `f2d47d4`
- Runtime: CPython 3.12.12 in `.python/` and `.venv/`
- Dependencies: Pydantic 2.13.5, NumPy 2.5.2, SciPy 1.18.1, Shapely 2.1.2,
  pytest 8.4.2, Hypothesis 6.167.1, Ruff 0.16.6, Mypy 1.20.2
- Scenario: `configs/scenario/uuv_only_single_target.yaml`
- Audit seed and duration: seed 42, 120 steps, 5 s physics step, repeat 2

## Baseline Evidence

Before production-code changes, the focused suite passed 210 tests. The original seed-42
audit returned a superficial PASS and equal digest
`3f943557289c189af7f60347c4ee308072352a500f50f14e3cf73c8b1f96a7d2`, but its
600 s state remained at execution revision 1 with four active groups, no passive group,
no owner, one public observation, and one active ping. That result did not prove C-01-C-06.

## Implemented Contracts

- C-01: recoverable forward snapshot admission, atomic rollback, progress preservation,
  deployment-generation-safe resource allocation, idempotent semantic projection, and
  four simultaneous source-linked replacement pairs with 24 unique live members.
- C-02: active echo to public group fusion to finite region probability to two-cycle
  confirmation, with structured resets and one atomic passive owner.
- C-03: scheduled physical transmissions separated from echoes, source-backed ping
  evidence, clipped union footprint coverage, route progress, scan rounds, thresholded
  completion, a 256-item public evidence window, and a fixed-memory whole-round coverage
  raster.
- C-04: per-member successor readiness, adjacent-only handoff, old-owner continuity
  through 0/3-2/3, and atomic 3/3 ownership transfer before old-group exit.
- C-05: one incoming/outgoing pair per region, latest-pending revision handling, bounded
  live resources, exit-before-release, and dedicated-to-regional restoration.
- C-06: typed region, group, tracking-control, readiness, replacement, reason, and evidence
  fields projected through the canonical `OperationalFrame` payload.

The final review pass also closed six cross-cutting defects: resource episodes are
validated for every runtime member; engine, controller, entity materialization, and
verification-monitor registration roll back atomically; replacement completion preserves
the latest pending geometry/generation; runtime handoff cannot fall back to legacy ID
sets; scan counters reset only on a newer round while evidence IDs remain bounded; and
the direct frame-builder path projects regional missions from the same mission snapshot.
An incoming replacement with zero physical pings now preserves the outgoing generation's
public scan state until evidence-backed ownership of that region changes. A final C-03
review also moved round rollover to the route-reload instant so the next round's first
pings survive, retained footprints evicted from the public evidence window in a bounded
50 m raster mask, and reclaimed all scan state for retired deployment generations.

## Focused Verification

All commands set command-local `PYTHONPATH=src`, disable the pytest cache provider, and
use a new `.test-artifacts/` basetemp.

The final focused matrix covered the agent runtime, domain models, coordinator, mission
controller, active sonar, group activation, execution-frame contract, live pipeline,
initialization, live validation, and coverage audit modules. The already-separated
eight-hour case was excluded from this matrix because the complete repository run below
executes it:

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp=.test-artifacts/review-focused-matrix-04 `
  tests/agent/test_runtime.py tests/api/test_execution_frame_contract.py `
  tests/domain/test_execution_models.py tests/domain/test_mission_models.py `
  tests/integration/test_live_tracking_health_pipeline.py `
  tests/integration/test_uuv_initialization_local_perception.py `
  tests/planning/test_coverage_paths.py `
  tests/runtime/test_execution_coordinator.py `
  tests/runtime/test_execution_snapshot_factory.py `
  tests/runtime/test_mission_controller.py `
  tests/runtime/test_task_group_instances.py `
  tests/simulation/test_active_sonar.py `
  tests/simulation/test_execution_group_activation.py `
  tests/verification/test_live_demo_monitor.py `
  tests/verification/test_uuv_tracking_coverage_runner.py `
  -k "not tracking_pipeline_remains_bounded_and_executable_for_eight_hours"
```

Result: exit 0, **406 passed, 1 deselected** in 344.89 s.

Additional current evidence:

- Reviewer regression bundle for all six initial findings: 8 passed.
- Final C-03 route-reload, bounded-evidence, cumulative-coverage, and generation-cleanup
  regressions: 4 passed.
- C-03 engine/controller/audit matrix: 174 passed.
- Region replacement, scan-round audit, and 120-step authority checks: 5 passed.
- Post-style controller and authoritative 120-step scenario: 2 passed in 22.40 s.
- Eight-hour production pipeline, before the final complete run: 1 passed in 1719.65 s.
- Configured local embedding readiness: 25 passed, 1 environment-gated test skipped.
- Restored default live-acceptance driver: 16 passed, 1 skipped.
- `git diff --check`: exit 0.

The complete repository command was:

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp=.test-artifacts/full-suite-final-05
```

Result: exit 0, **2302 passed, 72 skipped, 5 warnings** in 1359.00 s. The
warnings are third-party Starlette and NumPy deprecations; there were no test failures or
collection errors. The run includes the 28,800 s checkpoint pipeline and 680-step
resource-generation integrations.

## Production Audit

Command:

```powershell
.venv\Scripts\python.exe scripts/run_uuv_tracking_coverage_audit.py `
  --config configs/scenario/uuv_only_single_target.yaml `
  --seed 42 --steps 120 --repeat 2 `
  --work-dir outputs/codex-uuv-tracking-control-20260906/final-work-attempt-09 `
  --evidence-dir outputs/codex-uuv-tracking-control-20260906/final-evidence-attempt-09
```

Result: exit 0, **PASS**. Both traces have digest
`85216f108d3fcc2656a9ace1e6d26ae87c7e225496107efbe5e90131426db60f`.

Key metrics from `final-evidence-attempt-09/metrics.json`:

| Metric | Result |
| --- | ---: |
| Active pings | 222 |
| Source-linked active echoes | 8 |
| Active regions with physical pings | 4 |
| Maximum runtime scan coverage | 0.9858454236 |
| Maximum coverage gap area | 0.0 m2 |
| Entry confirmations | 2/2 |
| Tracking owner observed | true |
| Maximum owners in one frame | 1 |
| Owner-gap frames | 0 |
| Maximum runtime groups | 8 |
| Maximum visible UUVs | 24 |
| Replacement pairs | 4 |
| Minimum fleet separation | 443.0814 m |
| Physics violations | 0 |

The named hard checks
`c01_revision_replacement_contract`, `c02_public_fusion_tracking_owner`,
`c03_physical_scan_coverage`, `c04_atomic_handoff_continuity`,
`c05_resource_bounded_replacement`, and `c06_canonical_operational_frame` are all true.

## Static Verification

- Ruff 0.16.6 with `--no-cache` passes for every changed file except
  `cli.py`, `runtime/execution_coordinator.py`, `simulation/engine.py`, and the exactly
  restored historical acceptance driver.
- `cli.py`, `runtime/execution_coordinator.py`, and `simulation/engine.py` report 6, 1,
  and 13 findings respectively; Ruff against their `origin/master` contents reports the
  same rule sets and counts. The restored driver's SHA-1 matches `2c444bb^`, so its eight
  findings are also inherited. The branch adds no Ruff finding.
- Repository-wide Mypy 1.20.2 with Python 3.12 and `--no-incremental` exits 1 with
  **319 errors in 59 files**. The repository baseline begins with absent SciPy stubs and
  includes broad pre-existing typing failures across planning, persistence, runtime,
  simulation, API, and CLI modules. No dependency was installed and this branch does not
  claim repository-wide Mypy conformance.

## Repository-Wide Recovery

Commit `2c444bb` deleted `tools/run_default_live_acceptance.py` while retaining both
importing tests. With explicit authorization, the file was restored byte-for-byte from
`2c444bb^`; both Git blob hashes are
`391b32f1622616652c40e8a9f927a14b3a9bf89f`.

The configured sentence-transformer snapshot was downloaded only below the isolated
worktree's ignored `.cache/sentence-transformers/` path. On Windows the Hugging Face
symlink fallback materialized duplicate framework weights, so the local cache occupies
approximately 4.62 GB rather than only the unique model payload. It is not committed.

The diagnostic failures and collection errors recorded earlier are all resolved or
updated to the intended C-01-C-06 runtime contract. The authoritative final evidence is
the complete green suite and attempt-09 production audit above. Generated `.test-artifacts/`,
`build/`, `outputs/`, and `.cache/` paths remain untracked and were neither deleted nor
committed.
