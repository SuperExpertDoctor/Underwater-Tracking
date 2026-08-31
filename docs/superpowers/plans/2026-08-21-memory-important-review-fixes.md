# Memory Important Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make memory source discovery fair and bounded, preserve strict conversation provenance, persist bounded observation inputs, and cover the contracts with regression tests.

**Architecture:** Persist source-scope last-seen timestamps and per-repository discovery continuations in `memory_source_cursors`. Treat conversation cursors as absolute message offsets and derive the effective loaded provenance before invoking the reasoner. Store a bounded sanitized observation projection in the durable work payload while retaining authoritative source IDs for worker reloading.

**Tech Stack:** Python 3.11, SQLite, Pydantic, pytest, ruff, mypy.

## Global Constraints

- Source discovery remains bounded per call and must eventually visit scenarios beyond the first 32.
- All source reads and writes remain user-scoped and transactional.
- Real `MemoryReasoner` calls remain the semantic processing boundary; no keyword or static memory extraction rules may be added.
- Observation payloads must be sanitized and byte-bounded; complete authoritative records remain in their source repositories.
- Missing conversation messages must never be passed as provenance to filtering, extraction, or persisted memory versions.

### Task 1: Fair source discovery and persistence cursor

**Files:**
- Modify: `src/underwater_tracking/persistence/sqlite.py`
- Modify: `src/underwater_tracking/persistence/memory.py`
- Modify: `src/underwater_tracking/persistence/events.py`
- Modify: `src/underwater_tracking/persistence/ledger.py`
- Modify: `src/underwater_tracking/persistence/plans.py`
- Modify: `src/underwater_tracking/memory/source_reader.py`
- Test: `tests/memory/test_source_reader.py`
- Test: `tests/persistence/test_memory_repository.py`

- [x] Write a regression test with more than 32 scenarios and assert successive bounded discovery calls return later scenarios.
- [x] Run the focused test and verify it fails because discovery always selects the first page.
- [x] Add bounded repository pagination, durable discovery cursors, a last-seen scope update, and an index supporting scope recency ordering.
- [x] Run the focused tests and verify cold-start scope recovery and cursor persistence.

### Task 2: Absolute conversation cursor and effective provenance

**Files:**
- Modify: `src/underwater_tracking/memory/source_reader.py`
- Modify: `src/underwater_tracking/memory/worker.py`
- Test: `tests/memory/test_source_reader.py`
- Test: `tests/memory/test_worker.py`

- [x] Add a rolling-window test that proves absolute offsets do not replay retained messages.
- [x] Add a worker test where one queued message has been evicted and assert the reasoner and persisted memory receive only loaded IDs, with an auditable degraded event for the missing ID.
- [x] Run both tests RED.
- [x] Implement absolute-offset slicing and effective per-type source ID projection from loaded sources.
- [x] Run the focused tests GREEN and confirm retry, isolation, and transaction behavior remain unchanged.

### Task 3: Bounded sanitized observation work payload

**Files:**
- Modify: `src/underwater_tracking/domain/memory_models.py`
- Modify: `src/underwater_tracking/memory/service.py`
- Modify: `src/underwater_tracking/memory/source_reader.py`
- Test: `tests/memory/test_service.py`
- Test: `tests/memory/test_worker.py`

- [x] Add a test asserting observation work durably contains bounded sanitized source text/projection and authoritative source IDs.
- [x] Add a worker test asserting the persisted observation projection is actually consumed by the worker/reasoner when the authoritative repository is unavailable for reload.
- [x] Run both tests RED.
- [x] Add a strict bounded work payload field and sanitize/truncate observation input before enqueueing; preserve source repository reload semantics.
- [x] Run focused memory and persistence tests GREEN.

### Task 4: Full verification and commit

**Files:**
- Modify: only files covered by Tasks 1-3.

- [x] Run focused memory/persistence tests.
- [x] Run related full tests.
- [x] Run ruff and scoped mypy.
- [x] Review the diff and confirm no unrelated files changed.
- [x] Commit with a message beginning `fix:`.
