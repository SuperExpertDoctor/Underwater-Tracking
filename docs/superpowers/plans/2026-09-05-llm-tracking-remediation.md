# LLM Tracking Remediation Implementation Plan

> Execute this plan task by task in the isolated `feature/llm-tracking-remediation` worktree. Every Python test and audit command uses `conda run --no-capture-output -n underwater-tracking`.

## Task 1: Deadline-aware execution refresh

- Add failing tests in `tests/agent/test_execution_refresh.py` for missing, expired, margin, source-revision, prediction-revision, not-due, and simulation-time rollback decisions.
- Add `ExecutionRefreshConfig` and validation in `src/underwater_tracking/config/models.py`; expose it from `configs/agent.yaml`.
- Add `src/underwater_tracking/agent/execution_refresh.py` with the specified frozen decision dataclass and pure decision function.
- Parameterize `build_execution_snapshot` in `src/underwater_tracking/runtime/execution_snapshot_factory.py` with configured validity while retaining the compatibility default.
- Run the focused tests and commit the task.

## Task 2: Non-blocking cycle ordering

- Add regression coverage for deterministic prediction/mission progress when an LLM provider fails or sleeps, and for 30-second retry behavior.
- Make `AgentLoop.on_situation` refresh deterministic state before optional LLM work and convert LLM failures into degraded optional state rather than a physics-path exception.
- Apply the configured refresh policy to the background mailbox and synchronous path.
- Run agent/background and live pipeline tests and commit the task.

## Task 3: Expiry and recovery protocol

- Add tests for expired snapshots with no newer public source, deterministic recovery after a newer source, and the one-round recovery-to-next-round LLM ordering.
- Register bounded execution refresh events and publish due/attempted/committed/rejected/waiting/recovered transitions.
- Preserve expired snapshots during source gaps without moving their validity interval; use CAS for recovery with `current + 1` execution revision.
- Run execution factory, event, and integration tests and commit the task.

## Task 4: Bounded semantic optimization

- Add tests that accept only semantic advice bound to current situation/execution/prediction/evidence revisions and reject stale or physical proposals.
- Add a typed semantic-advice payload and whitelist validation in the coordinator, central graph, strategy node, and prompts.
- Extend the physical fingerprint to cover every physical execution field while leaving only bounded semantic advice mutable.
- Run strategy/prompt/coordinator/integration tests and commit the task.

## Task 5: Deterministic operational diagnosis

- Add tests for the four required operational questions with missing LLM and missing authority evidence.
- Add strict diagnosis models and a public-evidence diagnosis builder. Bind answers to frame, execution revision, and evidence IDs; compute ages from simulation time.
- Route the four question classes around LLM classification and retain a degraded response for other questions when the provider is unavailable.
- Run conversation and API/agent tests and commit the task.

## Task 6: Revision-aware memory closure

- Add summary/source/service tests for execution health, region entry, coverage, ownership, handoff, source gaps, expiry/rejection, and recovery episode identity.
- Extend public summary models and the memory source whitelist with bounded execution and diagnosis fields.
- Add deterministic episode ingestion keyed by scenario, episode type, entity, and opening execution revision; keep source IDs and make service ingestion asynchronous/failure-tolerant.
- Run the memory suite and commit the task.

## Task 7: Live/replay frame contract

- Add frame contract and replay tests for refresh status, due/attempt/result/reason/source-revision fields, event cardinality, and legacy defaults.
- Extend domain/UI models and `frame_builder.py` so live, JSONL, and replay serialize the same bounded projection.
- Ensure reason codes are registry-backed and never contain stack traces or prompts.
- Run API/replay/UI contract tests and commit the task.

## Task 8: Truth-boundary enforcement

- Add recursive prompt, summary, and conversation tests for all forbidden truth keys, including nested payloads and legacy aliases.
- Apply one bounded sanitizer/validator at planning, assistant, memory, and replay boundaries; reject or drop forbidden fields without exposing them in diagnostics.
- Run the truthfulness and memory/conversation suites and commit the task.

## Task 9: Seed-42 acceptance audit

- Add audit tests for the four input cases, no-network provider blocking, two identical 3600-second runs, deterministic event sequences, assistant cases, and memory source IDs.
- Extend the audit runner and script to write `outputs/llm-tracking-remediation-seed42/` and all eleven remediation metrics.
- Run the audit twice plus the full relevant test suite, inspect JSON evidence, and commit the task.

## Final integration

- Run fresh verification immediately before integration.
- Merge `feature/llm-tracking-remediation` into `branch1` without overwriting unrelated user changes.
- Verify `branch1` contains the feature commits and rerun the focused smoke checks from the merged checkout.
