# UUV Control and Full Tracking Closure Integration Design

**Date:** 2026-09-07  
**Status:** High-level approach approved; ready for implementation review  
**Base:** `branch1`  
**Integrated branch:** `origin/fix/uuv-tracking-control-c01-c06-20260906`

## 1. Goal and scope

Merge the UUV control remediation into `branch1`, reconcile the overlapping
frontend/backend contracts, and close every issue I-01 through I-06 in
`docs/three-uuv-tracking-team-remediation-and-acceptance.md`.

The result must be a real end-to-end loop:

```text
physical simulation
  -> public observations and fusion
  -> region probability and confirmation
  -> deterministic UUV state transition
  -> authoritative OperationalFrame
  -> HTTP/WebSocket/JSONL/Replay/UI
```

Production acceptance must use the actual simulation, controller, frame builder,
and UI contracts. Mock data, synthetic state-machine fixtures, evaluation truth,
or browser-side state inference cannot be used as production evidence.

The final long run will represent at least eight hours of simulation time while
finishing within approximately eight minutes of wall-clock time. The exact
acceleration/step command and measured wall time will be recorded in the
verification report.

## 2. Integration strategy

Work first in an isolated branch based on the current `branch1`. Merge the full
UUV remediation branch there so its history and all C-01 through C-06 changes
remain available. Resolve conflicts by preserving the following ownership rules:

1. `MissionController`, the execution coordinator, and the simulation engine own
   deployment, sensor mode, confirmation, owner, handoff, replacement, release,
   and safety decisions.
2. The public fused estimate and world-model outputs contain provenance,
   revision, age, validity, and no target truth.
3. `OperationalFrame` is the single cross-module projection. Its values are
   copied from authoritative runtime state, not reconstructed by the frontend.
4. HTTP, WebSocket, JSONL, Replay, and the UI serialize the same frame contract.
5. LLM and assistant outputs remain high-level, evidence-referenced suggestions;
   they cannot assign physical coordinates, UUV resources, owners, or state
   transitions outside deterministic validation.

Existing uncommitted files in the user's checkout are left untouched. The final
integration back into `branch1` is performed only after the isolated branch has
passed its verification gates.

## 3. Closure work by documented issue

### I-01: Fresh execution and estimate recovery

- Admit higher execution revisions atomically before expiry when valid.
- Recover after a rejected or expired candidate without poisoning later valid
  candidates or resetting live group progress.
- Keep public estimate, prediction, and world-model revisions aligned.
- Make current, stale/degraded, and expired/unavailable states explicit in every
  consumer and verify the LLM timeout path does not stop physics.

### I-02: Public-fusion passive acquisition

- Use current-cycle public fused observations to calculate finite region
  probabilities.
- Require two consecutive confirmations at or above `0.70`.
- Atomically switch all three members to `PASSIVE_TRACK` and install one owner.
- Reset counters with structured reasons for missing, invalid, low, or exited
  evidence. Truth and evaluation frames never enter this decision.

### I-03: Physical active coverage

- Emit source-backed active transmissions on the configured schedule even when no
  target echo exists.
- Track route progress, scan round, physical ping count, and deduplicated ping
  footprints separately.
- Set scan completion only from runtime coverage reaching the configured
  threshold; route completion alone cannot fabricate coverage.
- Keep the 300 m minimum separation contract and deterministic repeated runs.

### I-04: Authoritative status explanation

- Publish region probability, confirmation progress, group lifecycle, sensor mode,
  owner, successor readiness, revisions, and structured blocking reasons in one
  bounded frame.
- Make the assistant explain those fields by frame ID and execution revision.
- Make the frontend display backend values directly, including unavailable data,
  without guessing state from geometry or history.

### I-05: Handoff, replacement, and dedicated restoration

- Require adjacent, deployed, healthy, passive successor members with valid
  observations from the same cycle.
- Expose `0/3` through `3/3` readiness and keep the old owner until takeover.
- In the successful frame, publish one new owner and the old group as `EXITING`;
  release only after boundary evidence and `DISAPPEARED`.
- Preserve incoming/outgoing replacement pairs, use latest-revision-wins
  semantics, and restore regional mode after a valid three-member takeover of a
  dedicated owner approaching the 7000 m threshold.

### I-06: Truth-safe formal and evaluation views

- Keep `OperationalFrame` and `EvaluationFrame` separate.
- Permit truth and error overlays only in explicitly labelled evaluation mode.
- Ensure formal HTTP, WebSocket, JSONL, Replay, world-model, LLM, and UI paths
  contain no target truth fields.
- Switching display mode changes presentation only; it cannot change control
  inputs or state transitions.

## 4. Verification design

Implementation follows red-green-refactor for each missing behavior. Verification
is layered:

1. Run focused C-01 through C-06 tests with fresh pytest basetemps.
2. Run the complete pytest suite with `conda run --no-capture-output -n
   underwater-tracking` and UTF-8 output settings.
3. Run Ruff and Mypy with caches disabled.
4. Run two identical real seed-42 production traces. Each trace must reach at
   least `28800` simulation seconds within about `480` wall-clock seconds and
   must contain source-backed active pings, passive acquisition, owner handoff,
   zero owner-gap frames, boundary disappearance, and the required frame
   transport equality.
5. Run the accelerated long-run failure/recovery checks for expired snapshots,
   LLM timeout/unavailability, interrupted confirmation, missing successor
   observation, expired prediction, replacement, and dedicated restoration.
6. Record commands, versions, exit codes, deterministic digests, output paths,
   and any unverified criteria in a committed verification report.

Unit and fixture tests remain useful for individual state transitions, but they
cannot substitute for the real production trace in the final acceptance.

## 5. Non-goals and safety constraints

- Do not merge into `master` or push remote branches unless separately requested.
- Do not delete or overwrite existing evidence directories.
- Do not change the legacy ROS workspaces or add new physics models outside the
  documented simplified sonar contract.
- Do not introduce mock values into production code, UI hooks, API defaults, or
  acceptance evidence.
- Do not use target physical truth to advance formal tracking state.

