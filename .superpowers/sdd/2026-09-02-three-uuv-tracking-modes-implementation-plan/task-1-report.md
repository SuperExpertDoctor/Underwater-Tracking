# Task 1 Report: Single Tracking Policy Configuration

Status: DONE_WITH_CONCERNS

## RED evidence

- After adding the brief's policy tests, the focused command failed during collection because `TrackingPolicyConfig` did not yet exist (`ImportError`).
- After adding cross-configuration mismatch tests with the validator absent, the focused suite reported `4 failed, 19 passed`; all failures were expected `DID NOT RAISE` failures for mismatched target and sonar ranges.
- The supplied `Literal[4]` and `Literal[3]` fields initially produced Pydantic's generic literal messages, so a pre-validator was added to preserve the brief's required error messages.

## GREEN evidence

- `conda run --no-capture-output -n underwater-tracking python -m pytest tests/config/test_uuv_only_config.py -q`: `23 passed`.
- `conda run --no-capture-output -n underwater-tracking python -m pytest tests/config -q`: `87 passed`.
- `conda run --no-capture-output -n underwater-tracking python -m ruff check src/underwater_tracking/config tests/config/test_uuv_only_config.py`: `All checks passed!`.
- `git diff --check`: no whitespace errors.

## Changes

- Added frozen, strict `TrackingPolicyConfig` with the exact policy values and geometry/range/mileage invariants.
- Added `ScenarioConfig.tracking_policy` with `default_factory`; retained flat legacy fields.
- Added UUV-only AppConfig checks tying target detection and `uuv_dual_sonar` active/passive ranges to the policy. Non-UUV-only paths remain unbound to these checks.
- Updated the UUV-only scenario, environment, and sonar YAML values; set the legacy entry buffer to `0.0`.
- Added policy, invariant, and cross-configuration tests in `tests/config/test_uuv_only_config.py`.

## Self-review and concerns

- Current diff contains only the five requested task files.
- The requested commit was not created because the user interrupted the workflow before staging/commit.
- A standalone compatibility probe failed with `AttributeError` because `conda python -c` imported the installed package instead of the worktree; the configured pytest runs imported the worktree and passed all 87 config tests.
- Full runtime/integration verification was not run after the interruption.

## Fix Round 4: UUV-only Runtime Policy Wiring

Status: FIXED

### Root cause

- `_mission_controller_for` passed the flat region threshold, confirmation-cycle, and mileage-fraction fields into UUV-only runtime.
- `SimulationEngine._mission_entry_polygon` read the flat entry buffer for UUV-only execution.
- Dedicated release reused the legacy warning-fraction-derived threshold.

### Changes

- UUV-only controller construction now takes thresholds, mileage, and reserve values from `ScenarioConfig.tracking_policy`.
- `MissionController` accepts an explicit dedicated remaining-mileage reserve while retaining the legacy fraction-derived fallback for existing callers.
- UUV-only entry probability uses the executable polygon directly; legacy engine paths retain `region_entry_buffer_m` expansion.
- Added regression coverage proving mutated flat fields cannot change UUV-only behavior and that legacy geometry remains compatible.

### Verification

- RED command: `conda run --no-capture-output -n underwater-tracking python -m pytest tests/runtime/test_mission_controller.py::test_uuv_only_controller_uses_tracking_policy_when_legacy_fields_change tests/simulation/test_uuv_only_carrier_group.py::test_uuv_only_entry_polygon_ignores_legacy_entry_buffer tests/simulation/test_uuv_only_carrier_group.py::test_legacy_entry_polygon_still_uses_legacy_entry_buffer -q`
  Output: `2 failed, 1 passed in 2.62s` (the expected stale controller and entry-buffer reads).
- GREEN command: `conda run --no-capture-output -n underwater-tracking python -m pytest tests/runtime/test_mission_controller.py::test_uuv_only_controller_uses_tracking_policy_when_legacy_fields_change tests/simulation/test_uuv_only_carrier_group.py::test_uuv_only_entry_polygon_ignores_legacy_entry_buffer tests/simulation/test_uuv_only_carrier_group.py::test_legacy_entry_polygon_still_uses_legacy_entry_buffer -q`
  Output: `3 passed in 2.06s`.
- Ruff command: `conda run --no-capture-output -n underwater-tracking python -m ruff check src/underwater_tracking/cli.py src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/simulation/engine.py tests/runtime/test_mission_controller.py tests/simulation/test_uuv_only_carrier_group.py`
  Output: `Found 21 errors.` All reported violations are pre-existing in `cli.py`, `engine.py`, or the existing simulation test module; none are in the Round 4 additions.
