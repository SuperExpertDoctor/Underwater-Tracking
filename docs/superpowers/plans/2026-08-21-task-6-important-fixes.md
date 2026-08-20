# Task 6 Important Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair all Task 6 Important review findings and prevent cross-user memory-version existence disclosure while preserving real provider success paths and legacy conversation compatibility.

**Architecture:** Make bundle shutdown transactional at the ownership boundary: a bundle remains installed until its worker and loop have actually closed, and failed shutdown can be retried. Construct worker-side repositories, audit ledger, embedding provider, and memory LLM as independent resources. Make memory deletion resolve the stored family's source conversation before choosing an audit stream, and make API scope failures indistinguishable from missing resources.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, SQLite, pytest, ruff, mypy.

## Global Constraints

- `_AgentLoop.close()` returning `False` means worker-owned resources remain live and close remains retryable.
- Missing memory/chat credentials degrade status and preserve raw-message persistence; they must not raise `SystemExit` during app construction.
- Real embedding and LLM providers remain the semantic boundary; no fake local provider or fake success result is allowed.
- `/api/conversation/messages` keeps its existing non-empty-string compatibility; strict opaque validation is limited to memory APIs.
- Versions and deletion must not disclose another user's family existence.

### Task 1: Add failing lifecycle, isolation, provenance, compatibility, and privacy tests

**Files:**
- Modify: `tests/runtime/test_run_controller.py`
- Modify: `tests/api/test_app_lifespan.py`
- Modify: `tests/agent/test_background_cycle.py`
- Modify: `tests/agent/test_runtime_master_slave_adversary.py`
- Modify: `tests/api/test_memory_routes.py`
- Modify: `tests/api/test_conversation.py`
- Modify: `tests/integration/test_memory_api_real_sqlite.py`

**Interfaces:**
- Exercise `RunController._close_bundle`, `RunController.close`, FastAPI lifespan, `_AgentLoop.close`, `MemoryService.delete`, and the public HTTP routes.

- [x] Add a worker-stop-timeout test proving the installed bundle and all resources remain available for a retry.
- [x] Add a lifespan test whose queue close raises and assert controller close still runs.
- [x] Add worker construction tests proving worker repositories, ledger, embedding provider, and memory LLM are distinct objects from foreground resources.
- [x] Add deletion tests for wrong conversation and cross-user scope; assert 4xx and no event in the wrong stream.
- [x] Add missing-chat-credential app construction/status tests and Unicode/space legacy conversation-id tests.
- [x] Add a real SQLite versions test asserting a foreign user receives exactly 404.
- [x] Run the focused tests and confirm each new regression test fails for the current implementation.

### Task 2: Implement resilient shutdown and lifespan cleanup

**Files:**
- Modify: `src/underwater_tracking/runtime/run_controller.py`
- Modify: `src/underwater_tracking/api/app.py`

**Interfaces:**
- Preserve `RunController.close() -> None` and `_AgentLoop.close() -> bool`; failed bundle shutdown remains retryable.

- [x] Keep a bundle installed until `_close_bundle` reports complete; do not close bundle resources after a false worker stop.
- [x] Make `_close_bundle` stop/join the worker, honor `loop.close()` false, and only write the manifest and release ownership after successful close.
- [x] Let `start_run` retain the previous bundle if replacement cleanup is incomplete.
- [x] Put queue close and controller close in independent `try/finally` cleanup paths so queue exceptions cannot skip controller cleanup.
- [x] Run lifecycle tests green.

### Task 3: Isolate worker resources and degraded credentials

**Files:**
- Modify: `src/underwater_tracking/cli.py`
- Modify: `src/underwater_tracking/memory/worker.py`
- Modify: `src/underwater_tracking/agent/llm.py` or the existing shared HTTP helper

**Interfaces:**
- Worker receives its own repository/ledger/provider/LLM instances and closes only resources it owns.
- Missing memory credentials produce a degraded `MemoryService`; missing chat credentials do not abort app creation.

- [x] Build worker embedding and audit LLM clients from the same real configuration with independent ledger/provider instances.
- [x] Keep raw message acceptance and queue persistence available when credentials are absent, while exposing degraded planning/chat and memory status.
- [x] Preserve configured real provider calls and audit metadata when credentials are present.
- [x] Run isolation and credential tests green.

### Task 4: Enforce family source provenance and opaque version privacy

**Files:**
- Modify: `src/underwater_tracking/memory/service.py`
- Modify: `src/underwater_tracking/persistence/memory.py`
- Modify: `src/underwater_tracking/api/app.py`
- Modify: `src/underwater_tracking/api/dependencies.py`

**Interfaces:**
- `MemoryService.delete(..., conversation_id=None) -> bool` resolves the family source conversation and rejects an explicit mismatch.
- `versions` maps both absent and foreign-user families to an empty result/404 at the HTTP boundary without revealing existence.

- [x] Resolve the family's real `source_message_ids`/conversation scope before emitting deletion audit events.
- [x] Reject wrong conversation or foreign-user deletion before mutating memory or writing a stream event.
- [x] Keep explicit no-conversation families on an explicit unscoped audit stream rather than accepting arbitrary conversation input.
- [x] Return uniform 404 behavior for foreign versions and retain 4xx validation for malformed memory identifiers.
- [x] Run API, persistence, and integration tests green.

### Task 5: Preserve legacy conversation validation and complete verification

**Files:**
- Modify: `src/underwater_tracking/api/app.py`
- Modify: tests touched by Tasks 1-4 as needed

- [x] Separate legacy conversation field validation from memory opaque identifier validation while retaining non-empty strings and length limits.
- [x] Run all requested API/lifespan/agent-loop/conversation/memory/persistence/runtime tests.
- [x] Run `ruff`, scoped `mypy`, and the repository diff-check command.
- [x] Inspect the final diff, commit with a message beginning `fix:`, and verify the worktree is clean.
