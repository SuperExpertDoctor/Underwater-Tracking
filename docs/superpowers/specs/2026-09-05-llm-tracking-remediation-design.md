# LLM Tracking Remediation Design

## Status

Approved for implementation by the user's explicit request to implement every task in `docs/llm-tracking-remediation-plan.md` and merge the result into `branch1`. That plan is the authoritative product specification.

## Goal

Keep deterministic tracking, prediction, mission control, and physical execution authoritative and live even when the LLM, assistant, or memory subsystem is slow, unavailable, stale, or malformed. Add an observable closure from public source data through execution refresh, diagnosis, memory, replay, and the seed-42 audit.

## Design decisions

1. The deterministic baseline is committed first. Semantic LLM output is optional and can only attach bounded advice to an already committed baseline.
2. Execution refresh is decided from the snapshot deadline plus public source and prediction revisions. A source gap cannot renew an expired snapshot.
3. MissionController remains the owner of lifecycle, ownership, handoff, exit, and mileage/physical constraints. The LLM cannot write coordinates, waypoints, UUV IDs, sensor settings, lifecycle, ownership, or confirmations.
4. Four operational questions are answered deterministically from the current frame, execution snapshot, mission state, and public evidence. Other assistant requests remain explicitly degraded when the LLM is unavailable.
5. Memory stores only public, source-backed information. Episode identity includes the opening execution revision, so expiry/rejection and recovery cannot silently merge unrelated execution state.
6. Live, JSONL, and replay use the same execution refresh projection. Missing fields in older replay records become `idle`/`unknown`, never an invented committed state.
7. Audit evidence is deterministic, local, network-blocked, seed-42 reproducible, and records both safety metrics and LLM/assistant/memory closure metrics.

## Data flow

`public observations -> deterministic prediction -> refresh decision -> MissionController/ExecutionCoordinator CAS commit -> OperationalFrame`

The assistant reads the same frame and execution evidence. Memory receives bounded public event payloads asynchronously. LLM semantic advice is revision-bound and discarded if its base execution or source evidence is stale.

## Verification

Each task adds focused regression tests before production code. The final verification runs the focused suite, the existing suite excluding the two pre-existing `tools`-import collection failures, and the seed-42 audit twice. The full suite's collection limitation is retained as an explicit residual risk and is not hidden by changing unrelated acceptance tests.
