# Ontology Knowledge Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every active ontology-query path so no ontology client, evidence, or memory provenance can influence an LLM decision, while preserving ordinary strategy, rule-world-model, and non-ontology memory behavior.

**Architecture:** Remove the optional ontology service at configuration and composition boundaries first, then remove its query/evidence contracts from strategy state, ledger, question payloads, and operational timeline views. Remove the ontology-only memory provenance field while retaining message, event, decision, and plan provenance. Old SQLite tables and columns are tolerated but are no longer created, queried, propagated, or rendered.

**Tech Stack:** Python 3.11, Pydantic 2, SQLite, pytest, Ruff, mypy, React 18, TypeScript, Vitest.

## Global Constraints

- The active runtime must not create, inject, call, persist, serialize, or render ontology queries.
- Preserve ordinary strategy LLM calls and `plan_adjustment_suggestions`; they must not depend on an external knowledge provider.
- Preserve message, event, decision, plan, and non-ontology memory provenance.
- Do not add a replacement external knowledge service or a test-only branch.
- Existing SQLite databases may contain `knowledge_queries` tables and `source_knowledge_ids` columns; do not destructively drop them, but active code must not query or expose them.
- Historical design, plan, audit documents, and Git history are retained; active code/config/tests/tools are the deletion surface.
- Follow TDD: write and run the focused failing test before each production behavior change, then run focused green tests and the listed regression set.

---

### Task 1: Remove Ontology Configuration, Client, and Runtime Composition

**Files:**
- Delete: `src/underwater_tracking/knowledge/__init__.py`
- Delete: `src/underwater_tracking/knowledge/client.py`
- Delete: `configs/knowledge.yaml`
- Modify: `src/underwater_tracking/config/loader.py`
- Modify: `src/underwater_tracking/config/models.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Modify: `src/underwater_tracking/agent/nodes/strategy.py`
- Modify: `tests/cli/test_cli.py`
- Modify: `tests/agent/test_strategy.py`
- Modify: `tests/agent/test_background_cycle.py`
- Delete: `tests/knowledge/test_client.py`

**Interfaces:**
- `AppConfig` no longer has a `knowledge` field, and the loader no longer reads `knowledge.yaml`.
- `CarrierDependencies` no longer has `knowledge_client`.
- `StrategyGenerationNode` accepts no knowledge provider. It still builds ordinary strategy payloads and keeps `plan_adjustment_suggestions` independent of ontology.

- [ ] **Step 1: Write failing composition tests**

Add assertions to the existing CLI and strategy tests before removing production code:

```python
def test_agent_dependencies_have_no_ontology_client(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = make_dependency_test_loop(monkeypatch)
    dependencies = loop._deps()
    assert not hasattr(dependencies, "knowledge_client")
    assert not hasattr(loop, "_knowledge_client")


def test_strategy_generation_has_no_external_knowledge_field() -> None:
    node = StrategyGenerationNode(fake_llm())
    payload = node.build_payload({}, "balanced")
    assert "external_knowledge" not in payload
```

Remove tests whose only purpose is to call the ontology client, but retain ordinary strategy and suggestion tests with no provider argument.

- [ ] **Step 2: Run the focused tests and verify the expected red state**

Run:

```powershell
python -m pytest tests/cli/test_cli.py tests/agent/test_strategy.py tests/agent/test_background_cycle.py -q
```

Expected: FAIL during collection or assertions because the old config, dependency, and strategy surfaces still expose ontology fields.

- [ ] **Step 3: Remove configuration and composition wiring**

Delete the `KnowledgeConfig` model and `AppConfig.knowledge`; remove `("knowledge", "knowledge.yaml")` from `_OPTIONAL_SECTIONS`. Delete `_AgentLoop._build_knowledge_client`, its constructor assignment, dependency argument, and shutdown call. Remove the `KnowledgeProvider` import and field from `CarrierDependencies`.

- [ ] **Step 4: Remove ontology behavior from the legacy strategy node**

Delete the `KnowledgeProvider`/`KnowledgeQueryResult` imports, constructor parameter, `_query_knowledge`, `external_knowledge` payload parameter, query-id output, and ontology prompt text. Keep strategy proposal generation and make suggestion generation depend only on an explicit normal strategy-suggestion path, never on a knowledge provider.

- [ ] **Step 5: Remove the client and client-only tests/config**

Delete the two `src/underwater_tracking/knowledge` files, `configs/knowledge.yaml`, and `tests/knowledge/test_client.py`. Update remaining fixtures to construct the revised interfaces without broad casts.

- [ ] **Step 6: Run focused green checks and commit**

Run:

```powershell
python -m pytest tests/cli/test_cli.py tests/agent/test_strategy.py tests/agent/test_background_cycle.py -q
python -m ruff check src/underwater_tracking/config/loader.py src/underwater_tracking/config/models.py src/underwater_tracking/cli.py src/underwater_tracking/agent/graphs/central.py src/underwater_tracking/agent/nodes/strategy.py tests/cli/test_cli.py tests/agent/test_strategy.py tests/agent/test_background_cycle.py
```

Then commit only this task's files:

```powershell
git add src/underwater_tracking/knowledge configs/knowledge.yaml src/underwater_tracking/config/loader.py src/underwater_tracking/config/models.py src/underwater_tracking/cli.py src/underwater_tracking/agent/graphs/central.py src/underwater_tracking/agent/nodes/strategy.py tests/knowledge tests/cli/test_cli.py tests/agent/test_strategy.py tests/agent/test_background_cycle.py
git commit -m "refactor: remove ontology runtime wiring"
```

---

### Task 2: Remove Ontology Query Evidence from State, Ledger, Questions, and Frames

**Files:**
- Modify: `src/underwater_tracking/agent/state.py`
- Modify: `src/underwater_tracking/domain/agent_models.py`
- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Modify: `src/underwater_tracking/persistence/ledger.py`
- Modify: `src/underwater_tracking/persistence/sqlite.py`
- Modify: `src/underwater_tracking/agent/nodes/questions.py`
- Modify: `src/underwater_tracking/agent/nodes/conversation.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/ui/src/types/frames.ts`
- Modify: `src/underwater_tracking/ui/src/App.css`
- Modify: `tests/agent/test_conversation.py`
- Modify: `tests/agent/test_questions.py`
- Modify: `tests/agent/test_central_graph.py`
- Modify: `tests/api/test_frame_pipeline.py`
- Modify: `tests/api/test_frame_contracts.py`
- Modify: `tests/persistence/test_ledger.py`

**Interfaces:**
- `DecisionRecord`, `CentralState`, and question evidence contain no ontology query IDs or query records.
- `DecisionLedger` no longer exposes ontology query read/write methods; existing database tables remain inert.
- Timeline factor kinds contain event/evidence/directive only; the frontend has no ontology factor style.

- [ ] **Step 1: Write failing evidence-contract tests**

Add tests that exercise real question/frame builders without ontology records:

```python
def test_question_payload_has_no_ontology_queries() -> None:
    evidence = retrieve_question_evidence(snapshot, ledger, events)
    payload = build_question_payload("why", entities, snapshot, evidence, None)
    assert "knowledge_queries" not in payload
    assert all("knowledge" not in str(value).lower() for value in payload.values())


def test_plan_timeline_has_no_ontology_factor() -> None:
    frame = build_live_frame_without_ontology_records()
    assert all(factor.kind != "knowledge" for row in frame.plan_timeline for factor in row.factors)
```

- [ ] **Step 2: Run the focused tests and verify the expected red state**

Run:

```powershell
python -m pytest tests/agent/test_questions.py tests/agent/test_conversation.py tests/api/test_frame_pipeline.py tests/api/test_frame_contracts.py -q
```

Expected: FAIL because the current builders still import/read/query ontology evidence and emit ontology fields.

- [ ] **Step 3: Remove state and decision fields**

Delete `knowledge_query_ids` from `CarrierState` and `DecisionRecord`, remove central graph checkpoint serialization of that field, and retain all ordinary evidence IDs and plan-adjustment suggestions.

- [ ] **Step 4: Remove ledger query APIs without destructive migration**

Delete `KnowledgeQueryRun`, `DecisionLedger.save_knowledge_query`, and `DecisionLedger.list_knowledge_queries`. Remove only the `CREATE TABLE` and index statements for new databases; do not add `DROP TABLE` or rewrite existing database files.

- [ ] **Step 5: Simplify question, conversation, and frame projections**

Remove query imports, query lookup, query rendering, knowledge-only citation text, and ontology source verification. Preserve event/decision/plan/message evidence and existing bounded question behavior. Remove the `knowledge` timeline factor kind and its CSS/type declarations.

- [ ] **Step 6: Update focused tests and run green checks**

Delete ontology-only test cases and update shared fixtures to omit removed fields. Run:

```powershell
python -m pytest tests/agent/test_questions.py tests/agent/test_conversation.py tests/agent/test_central_graph.py tests/api/test_frame_pipeline.py tests/api/test_frame_contracts.py tests/persistence/test_ledger.py -q
python -m ruff check src/underwater_tracking/agent/state.py src/underwater_tracking/domain/agent_models.py src/underwater_tracking/agent/graphs/central.py src/underwater_tracking/persistence/ledger.py src/underwater_tracking/persistence/sqlite.py src/underwater_tracking/agent/nodes/questions.py src/underwater_tracking/agent/nodes/conversation.py src/underwater_tracking/api/frame_builder.py tests/agent tests/api tests/persistence/test_ledger.py
```

- [ ] **Step 7: Commit this task**

```powershell
git add src/underwater_tracking/agent/state.py src/underwater_tracking/domain/agent_models.py src/underwater_tracking/agent/graphs/central.py src/underwater_tracking/persistence/ledger.py src/underwater_tracking/persistence/sqlite.py src/underwater_tracking/agent/nodes/questions.py src/underwater_tracking/agent/nodes/conversation.py src/underwater_tracking/api/frame_builder.py src/underwater_tracking/ui/src/types/frames.ts src/underwater_tracking/ui/src/App.css tests/agent tests/api tests/persistence/test_ledger.py
git commit -m "refactor: remove ontology evidence surfaces"
```

---

### Task 3: Remove Ontology-Only Memory Provenance and UI Rendering

**Files:**
- Modify: `src/underwater_tracking/domain/memory_models.py`
- Modify: `src/underwater_tracking/memory/source_reader.py`
- Modify: `src/underwater_tracking/memory/reasoner.py`
- Modify: `src/underwater_tracking/memory/service.py`
- Modify: `src/underwater_tracking/memory/worker.py`
- Modify: `src/underwater_tracking/persistence/memory.py`
- Modify: `src/underwater_tracking/persistence/sqlite.py`
- Modify: `src/underwater_tracking/ui/src/services/memoryApi.ts`
- Modify: `src/underwater_tracking/ui/src/components/MemorySteam.tsx`
- Modify: `src/underwater_tracking/ui/src/components/MemoryWindow.tsx`
- Modify: `src/underwater_tracking/ui/src/components/assistant/SmartAssistantPanel.tsx`
- Modify: `tests/memory/test_models.py`
- Modify: `tests/memory/test_service.py`
- Modify: `tests/memory/test_worker.py`
- Modify: `tests/persistence/test_memory_repository.py`
- Modify: `src/underwater_tracking/ui/src/components/MemorySteam.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/assistant/MemoryWindow.test.tsx`

**Interfaces:**
- Memory source groups contain message, event, decision, and plan IDs only.
- Existing legacy `source_knowledge_ids` database columns/JSON keys are ignored during reads and never included in new writes or API responses.
- `source_reader` maps plan references to `source_plan_ids`, not the removed ontology group.

- [ ] **Step 1: Write failing memory contract tests**

Add a round-trip test that proves ordinary provenance survives without an ontology field:

```python
def test_memory_round_trip_keeps_non_ontology_sources() -> None:
    memory = make_memory(source_event_ids=("event-1",), source_plan_ids=("plan-1",))
    save_memory(memory)
    loaded = load_memory(memory.memory_id)
    assert loaded.source_event_ids == ("event-1",)
    assert loaded.source_plan_ids == ("plan-1",)
    assert not hasattr(loaded, "source_knowledge_ids")
```

Update UI fixtures to assert event/plan labels remain visible without a knowledge label.

- [ ] **Step 2: Run memory tests and verify the expected red state**

Run:

```powershell
python -m pytest tests/memory/test_models.py tests/memory/test_service.py tests/memory/test_worker.py tests/persistence/test_memory_repository.py -q
```

Expected: FAIL because memory models, worker payloads, persistence SQL, and UI projections still require `source_knowledge_ids`.

- [ ] **Step 3: Remove the ontology source group from Python contracts and flow**

Delete the field from memory models and source extraction structures. Remove it from reasoner payloads, service merge/deduplication logic, worker extraction and provenance, and persistence inserts/selects. Keep the other four source groups and map source-reader plan IDs into `source_plan_ids`.

- [ ] **Step 4: Preserve legacy storage without exposing it**

Leave old SQLite columns in repair/migration definitions where required to load existing databases, but do not select them into active models, write them in new inserts, or copy them into API payloads. Do not delete or rewrite existing database files.

- [ ] **Step 5: Remove frontend types and labels**

Delete `source_knowledge_ids` from memory API types and components. Remove the knowledge source category and keep event/decision/plan/message source labels and empty-state behavior intact.

- [ ] **Step 6: Run Python and frontend focused checks**

Run:

```powershell
python -m pytest tests/memory/test_models.py tests/memory/test_service.py tests/memory/test_worker.py tests/persistence/test_memory_repository.py -q
Set-Location src/underwater_tracking/ui
npx vitest run src/components/MemorySteam.test.tsx src/components/assistant/MemoryWindow.test.tsx
npm run build
```

- [ ] **Step 7: Commit this task**

```powershell
Set-Location ../../..
git add src/underwater_tracking/domain/memory_models.py src/underwater_tracking/memory src/underwater_tracking/persistence/memory.py src/underwater_tracking/persistence/sqlite.py src/underwater_tracking/ui/src/services/memoryApi.ts src/underwater_tracking/ui/src/components/MemorySteam.tsx src/underwater_tracking/ui/src/components/MemoryWindow.tsx src/underwater_tracking/ui/src/components/assistant/SmartAssistantPanel.tsx tests/memory tests/persistence/test_memory_repository.py src/underwater_tracking/ui/src/components/MemorySteam.test.tsx src/underwater_tracking/ui/src/components/assistant/MemoryWindow.test.tsx
git commit -m "refactor: remove ontology memory provenance"
```

---

### Task 4: Verify Active-Source Removal and Full Regression Boundaries

**Files:**
- Modify only files already listed in Tasks 1-3 if verification exposes a defect.
- Do not modify historical documents under `docs/superpowers/specs`, `docs/superpowers/plans`, or `docs/superpowers/audits`.

- [ ] **Step 1: Add an active-source regression test**

Create `tests/verification/test_ontology_removal.py` that scans only active paths and fails if any active source contains these symbols: `OntologyKnowledgeClient`, `KnowledgeProvider`, `KnowledgeQueryResult`, `KnowledgeQueryRun`, `knowledge_client`, `KnowledgeConfig`, `knowledge_queries`, or `source_knowledge_ids`. The scan must exclude the new test itself and historical docs.

- [ ] **Step 2: Run the active-source test and inspect remaining references**

Run:

```powershell
python -m pytest tests/verification/test_ontology_removal.py -q
rg -n -i "ontology|knowledge_client|KnowledgeConfig|KnowledgeProvider|KnowledgeQueryResult|KnowledgeQueryRun|knowledge_queries|source_knowledge_ids" src tests tools configs
```

Expected: the test passes and `rg` returns no active-code/config/test/tool matches. Historical docs may still contain the terms by design.

- [ ] **Step 3: Run the complete non-external regression boundary**

Run:

```powershell
python -m pytest -q -m "not real_llm and not live_acceptance and not long_running"
python -m ruff check src tests tools
Set-Location src/underwater_tracking/ui
npm test
npm run build
```

Treat failures as defects only when they are caused by Tasks 1-3; preserve and report unrelated baseline/environment failures without weakening tests.

- [ ] **Step 4: Inspect the diff and commit only necessary verification fixes**

From the repository root run:

```powershell
git status --short
git diff --check
git diff --name-only branch1...HEAD
```

Confirm no pre-existing user changes from the parent checkout are present, no historical docs were altered, and no ontology source remains in active paths.

- [ ] **Step 5: Commit any cohesive verification fix**

If a verification defect required a code/test fix, commit it separately after rerunning its failing gate:

```powershell
git commit -m "test: verify ontology removal"
```
