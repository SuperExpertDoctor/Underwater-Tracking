# Task 4 Report: MemoryService and Persistent MemoryWorker

## Scope

Implemented only the requested memory modules and their focused tests:

- `src/underwater_tracking/memory/service.py`
- `src/underwater_tracking/memory/source_reader.py`
- `src/underwater_tracking/memory/worker.py`
- `tests/memory/test_service.py`
- `tests/memory/test_source_reader.py`
- `tests/memory/test_worker.py`

No conversation runtime, API, UI, or simulation code was changed.

## RED / GREEN

RED: the first focused test run failed during collection because the requested
`memory.service`, `memory.source_reader`, and `memory.worker` modules did not
exist. This established the tests were exercising new behavior rather than
passing against pre-existing code.

GREEN: implemented the service request-path persistence, cursor reader, and
durable worker. The first green run exposed a `NameError` for a missing
`MemoryWorkType` import in the worker dispatch path. Root cause analysis
confirmed the dispatch branch referenced the enum without importing it; the
single import correction made the original focused suite pass.

Additional RED tests were then added for real-filter rejection, transient retry
and bounded degradation, and short-term conversation source cursors. A fixture
method-placement error was corrected before those tests exercised production
behavior. The expanded focused suite then passed.

Final RED: the reviewer pass found that a newly extracted memory had no vector,
so it could never be retrieved. A test was changed to inject a recording real
embedding provider and assert its vector/version were persisted; it failed
because the worker did not accept that dependency. GREEN: the worker now
requires an injected real embedding provider before creating a version and
persists that provider's vector/version. It has no static, keyword, or mock
embedding fallback.

## Behavior Covered

- `prepare_context` reads short-term context separately and delegates only to
  the retriever for long-term material.
- `accept_turn` persists bounded raw messages, queues a durable work item, and
  emits a bounded pending memory event without calling a reasoner.
- Observation queueing is source-key deduplicated.
- Source reads project safe event data, retain stable source IDs, and advance
  durable cursors. Conversation cursors retain message IDs.
- Worker leasing runs filter, extract, version creation, and threshold-gated
  compression on its own named thread, and persists a real embedding before
  committing an extracted long-term version.
- Filter rejection is controlled solely by the reasoner result; no greeting or
  keyword rule exists in service or worker code.
- Transient LLM failure is retried with exponential backoff and becomes
  degraded at the configured attempt bound.
- `stop()` wakes the event-wait loop and joins within its bounded timeout.
- Stream records use only `MemoryStreamEvent` status/type contracts and do not
  carry LLM thinking data.

## Verification

- `PYTHONPATH=src pytest -q tests/memory/test_service.py tests/memory/test_source_reader.py tests/memory/test_worker.py`
  - `9 passed`
- `PYTHONPATH=src pytest -q tests/memory tests/persistence/test_memory_repository.py`
  - `47 passed, 1 skipped`
- `PYTHONPATH=src ruff check src/underwater_tracking/memory/service.py src/underwater_tracking/memory/source_reader.py src/underwater_tracking/memory/worker.py tests/memory/test_service.py tests/memory/test_source_reader.py tests/memory/test_worker.py`
  - passed
- `PYTHONPATH=src mypy src/underwater_tracking/memory/service.py src/underwater_tracking/memory/source_reader.py src/underwater_tracking/memory/worker.py`
  - passed
- `git diff --check`
  - passed

## Self-Review

Reviewed the new modules against the task brief. The worker owns a dedicated
`underwater-memory-worker` daemon thread, uses `Event.wait` rather than sleep,
and does not import or invoke `CarrierRuntime`, simulation locks, or runtime
locks. Queue leasing and work-state transitions stay in repository transactions;
the reasoner is called outside those repository operations. Content decisions
are delegated to `MemoryReasoner`; deterministic compression thresholds and
cursor advancement are kept separate.

## Concerns

- Task scope explicitly excludes runtime/API wiring. A later task must construct
  the worker with production source repositories and start/stop it with the
  application lifecycle.
- The current persistence contract has no public bulk maintenance API for
  decay/archive updates. The worker therefore keeps maintenance as a bounded,
  deterministic no-content hook; a subsequent persistence-contract task should
  expose those atomic repository operations before enabling archival mutation.
- A later runtime task must inject the production `HTTPEmbeddingProvider` when
  constructing the worker. The worker deliberately degrades work rather than
  manufacture an embedding when that dependency is absent.
