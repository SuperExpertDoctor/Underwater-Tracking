# USV Ray Frame Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent explicit platform-core frame publication from failing on USV
bearing rays while preserving USV bearings in internal group fusion.

**Architecture:** Keep `SituationSnapshot` and `domain/models.py` unchanged.
Use the existing public UUV state index in `frame_builder` as the only safe ray
origin source, filtering unsupported observers before constructing
`BearingRayView`. Exercise the full path through direct adapter tests,
publisher/JSONL tests, and explicit engine integration tests.

**Tech Stack:** Python 3.11, Pydantic models, pytest, Ruff, mypy.

## Global Constraints

- Do not modify `src/underwater_tracking/domain/models.py`.
- Do not modify agent/planning, runtime/cli, or tracking files.
- Do not remove USV bearings from internal GroupManager fusion or belief input.
- Do not introduce target truth or guessed observer positions into public frames.
- Do not silently treat an unknown observer as a UUV.

### Task 1: Add failing frame-contract regression tests

**Files:**
- Modify: `tests/api/test_frame_pipeline.py`
- Modify: `tests/api/test_live_publisher.py`
- Modify: `tests/integration/test_platform_core_scenario.py`

- [ ] **Step 1: Write tests for filtering and end-to-end publication.**
  Construct a contact with one public UUV ray and one `usv_00` ray, assert the
  direct builder emits only the UUV ray, and assert the publisher can log and
  publish an explicit snapshot without raising. Add an explicit engine carrier
  test asserting USV observation IDs remain in the GroupManager belief source
  IDs.

- [ ] **Step 2: Run the focused tests and verify the expected failure.**

  Run:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/api/test_frame_pipeline.py tests/api/test_live_publisher.py tests/integration/test_platform_core_scenario.py -q`

  Expected: the new builder/publisher assertion fails with the current
  unknown-UUV `ValueError`, while the existing engine fusion assertion remains
  green.

### Task 2: Implement the minimal public projection fix

**Files:**
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/simulation/engine.py`

- [ ] **Step 1: Filter rays by an explicit public origin.**
  Build the existing UUV index, keep only observations whose `uuv_id` is in
  that index, and pass those observations to `_build_ray`. Keep `_build_ray`
  strict for direct callers so an invalid observer cannot be silently renamed
  or assigned a fallback origin.

- [ ] **Step 2: Document and test the internal/public split.**
  Add a concise engine comment at the contact/ray handoff stating that all
  converted bearings, including USVs, remain internal while the public builder
  may project only UUV-origin rays.

- [ ] **Step 3: Run focused tests and verify they pass.**

  Run the focused pytest command from Task 1 and confirm zero failures.

### Task 3: Verify and commit

**Files:**
- Verify only the files above and the approved documentation.

- [ ] **Step 1: Run the requested explicit integration and API contract tests.**

  Run:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_platform_core_scenario.py tests/api/test_frame_contracts.py tests/api/test_frame_pipeline.py tests/api/test_live_publisher.py -q`

- [ ] **Step 2: Run static checks under the project Python 3.11 environment.**

  Run:
  `PYTHONPATH=src .venv/bin/python -m ruff check src tests`

  Run:
  `PYTHONPATH=src .venv/bin/python -m mypy src/underwater_tracking`

- [ ] **Step 3: Inspect scope and commit.**
  Run `git diff --check`, inspect `git diff` and `git status`, then commit only
  the implementation/tests with a message such as
  `fix: keep USV bearings out of unsafe public rays`.
