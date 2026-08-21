# Memory Review Fixes V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the five Important review findings while preserving real LLM/embedding boundaries, user/scenario isolation, read-only preview/apply behavior, and worker lifecycle semantics.

**Architecture:** SQLite migration first inspects the live schema and repairs memory tables inside one transaction, rebuilding tables when constraints or required columns are incomplete. Persistence APIs normalize legacy NULL scenarios to `__legacy__`, while conversation reads and writes enforce exact scenario provenance. Evidence candidates are formed as a base snapshot/evidence set plus verified memory sources, and `_AgentLoop.close` serializes shutdown attempts with retryable timeout behavior.

**Tech Stack:** Python 3.11+, stdlib `sqlite3`, Pydantic, pytest, ruff, mypy.

## Global Constraints

- Real LLM and embedding providers remain the semantic boundaries; no static memory or embedding fallback is added.
- All persistence writes remain transactional and user/scenario scoped.
- Existing preview/apply and read-only evidence behavior remains unchanged.
- A failed migration leaves no partial `*_legacy` table and can be retried from the original schema.
- A worker stop timeout leaves resources alive so a later close can complete.

---

### Task 1: Regression tests for resilient migration and scenario-aware work dedupe

**Files:**
- Modify: `tests/persistence/test_memory_repository.py`
- Modify: `tests/memory/test_service.py`

**Interfaces:**
- Tests exercise `open_database`, `LongTermMemoryRepository.enqueue_work`, `get_work_by_source_key`, and `ShortTermContextRepository` public APIs.

- [x] Add a v8 partial-table fixture missing columns from the current schema and assert migration completes, preserves rows, creates no `*_legacy` tables, and is idempotent.
- [x] Add a v9/v10 fixture with an old `memory_work_items` unique constraint, nullable scenarios, and a missing stream column; assert rollback on injected migration failure and successful retry.
- [x] Add a work-item test proving the same `(user_id, source_key)` can queue independently in two scenarios, while duplicate enqueue in one scenario is idempotent; cover NULL/legacy lookup.
- [x] Run the focused tests and record the expected failures before production changes.

### Task 2: Regression tests for strict short-term scenario provenance

**Files:**
- Modify: `tests/persistence/test_memory_repository.py`
- Modify: `tests/memory/test_service.py`
- Modify: `tests/memory/test_source_reader.py`

**Interfaces:**
- Tests exercise `append_messages`, `append_messages_and_enqueue_work`, `get_short_term`, `get_messages`, `MemoryService.prepare_context`, and `MemorySourceReader.read_conversation/load_work_sources`.

- [x] Assert append rejects a message whose scenario differs from the target, including the atomic append/enqueue path, without persisting either side of the transaction.
- [x] Seed a legacy malformed JSON context and assert get/prepare/read/load filter or reject wrong-scenario messages rather than exposing them.
- [x] Run the focused tests and record the expected failures before production changes.

### Task 3: Regression tests for evidence candidate union

**Files:**
- Modify: `tests/agent/test_questions.py`
- Modify: `tests/agent/test_conversation.py`

**Interfaces:**
- Tests exercise `build_question_payload`, `answer_question`, and the conversation evidence branch through existing deterministic LLM harnesses.

- [x] Assert `allowed_evidence_ids` cannot remove valid current snapshot/QuestionEvidence IDs.
- [x] Add a memory-hit case with one independently valid current evidence ID and one verified memory source; assert both remain citable and memory-only unverified sources are absent.
- [x] Run the focused tests and record the expected failures before production changes.

### Task 4: Regression tests for concurrent and retryable close

**Files:**
- Modify: `tests/agent/test_runtime.py`
- Modify: `tests/agent/test_background_cycle.py`

**Interfaces:**
- Tests exercise `_AgentLoop.close` with controllable worker/resource doubles and concurrent threads.

- [x] Assert concurrent close calls do not duplicate any resource close and only one caller performs shutdown.
- [x] Assert a worker stop timeout returns false while resources remain open, then a later close completes all resources exactly once.
- [x] Run the focused tests and record the expected failures before production changes.

### Task 5: Implement migration, persistence, evidence, and close fixes

**Files:**
- Modify: `src/underwater_tracking/persistence/sqlite.py`
- Modify: `src/underwater_tracking/persistence/memory.py`
- Modify: `src/underwater_tracking/memory/service.py`
- Modify: `src/underwater_tracking/memory/source_reader.py`
- Modify: `src/underwater_tracking/agent/nodes/questions.py`
- Modify: `src/underwater_tracking/agent/nodes/conversation.py`
- Modify: `src/underwater_tracking/cli.py`

**Interfaces:**
- Migration preserves `open_database` and `SCHEMA_VERSION`.
- Work-item lookup accepts an optional explicit `scenario_id` and maps NULL to `__legacy__`.
- Conversation persistence methods reject mismatched message scenarios.
- Evidence functions return the union of base evidence and verified memory source IDs.
- `_AgentLoop.close() -> bool` remains retryable and idempotent.

- [x] Implement live-schema inspection, atomic repair/rebuild, rollback, and index recreation.
- [x] Implement scenario-aware work dedupe and strict message/read filtering.
- [x] Implement evidence candidate union and preserve memory-source verification.
- [x] Implement condition-serialized close with identity-based once-only resource closure.
- [x] Run focused tests, then the requested migration/persistence/memory/conversation/runtime/CLI suites.
- [x] Run `ruff`, scoped `mypy`, and diff-check; inspect the final diff for unrelated changes.
- [x] Commit with a message beginning `fix:` and verify `git status --short` is empty.
