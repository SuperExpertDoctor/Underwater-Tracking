# Ontology Knowledge Removal Design

Date: 2026-08-28

Status: approved scope from the operator; implementation follows the existing
live-visualization branch.

## Goal

Remove the active ontology knowledge-query feature so it cannot influence LLM
planning or create an unreachable evidence path. Keep ordinary strategy LLM
decisions, the rule world model, and the remaining memory pipeline working.

## Scope

The active runtime must no longer create, inject, call, persist, serialize, or
render ontology queries.

Remove:

- `src/underwater_tracking/knowledge/` and its tests;
- `KnowledgeConfig`, `configs/knowledge.yaml`, and the loader entry;
- `_AgentLoop` knowledge-client construction, dependency injection, and close;
- `CarrierDependencies.knowledge_client`;
- ontology provider/query support from `StrategyGenerationNode`, including
  ontology-derived payload fields and query-id output;
- ontology query state, decision, question-evidence, and frame-timeline fields;
- `KnowledgeQueryRun` and the DecisionLedger query read/write API;
- creation of the new `knowledge_queries` table and index. Existing tables in
  old run databases are left untouched and ignored, avoiding destructive data
  loss during migration;
- `source_knowledge_ids` from the active memory domain, worker, service,
  persistence projection, API types, and UI. Old database columns and old JSON
  keys are tolerated as legacy input and are not exposed or propagated.

Preserve:

- non-ontology strategy payload factors and ordinary LLM calls;
- `plan_adjustment_suggestions` as a normal strategy feature, independent of
  any external knowledge provider;
- event, decision, plan, message, and memory provenance that is not an
  ontology-query ID;
- historical design/plan/audit documents and existing Git history;
- legacy database readability where it does not require querying the removed
  ontology table.

## Runtime Flow

The strategy path is:

```text
current state -> bounded strategy payload -> ordinary StrategyProposal LLM
             -> local verification and planning
```

There is no ontology side request before the proposal call. Missing ontology
configuration is no longer an accepted reason for a degraded strategy path.

## Compatibility and Failure Handling

Existing SQLite databases may contain the removed table/columns. Schema setup
does not create a new ontology table, and active code never queries it. Legacy
memory rows load their supported provenance fields; the removed ontology field
is discarded. New payloads and frames contain no ontology query IDs.

## Verification

Focused tests must prove that:

1. scenario configuration loads without `knowledge.yaml`;
2. the CLI dependency object has no ontology client;
3. strategy generation performs no ontology query and emits no ontology field;
4. question and frame evidence remain valid without ontology records;
5. memory provenance keeps event/decision/plan/message sources while omitting
   ontology sources;
6. an active-source search finds no ontology client/provider/query symbols in
   `src/underwater_tracking`, `tests`, `tools`, or `configs` (historical docs
   are excluded intentionally).
