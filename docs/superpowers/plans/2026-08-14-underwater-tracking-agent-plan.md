# Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan:** LangGraph Underwater Tracking Assistant

**Goal:** Add a persistent carrier-level LangGraph that performs event-routed intent analysis, strategy generation, verified plan commit, memory compression, expert directives, and evidence-backed questions without blocking group tracking.

**Architecture:** The carrier graph consumes immutable `SituationSnapshot` objects and separates strategic, tactical, informational, directive, and question branches. LLM calls are behind a provider-neutral port; every semantic output passes a dedicated Verify subgraph, while plan optimization and all commits remain deterministic and version-checked.

**Tech Stack:** LangGraph 1.x, Pydantic 2, SQLite checkpointer/store, provider-neutral chat HTTP client, Pytest, FastAPI TestClient for later API seams.

---

**Prerequisite:** Complete `2026-08-14-underwater-tracking-foundation-plan.md` with a green suite.

## File map

- `configs/agent.yaml`, `configs/llm.yaml`: event, retry, history, model, and prompt settings.
- `src/underwater_tracking/agent/state.py`: carrier graph state.
- `src/underwater_tracking/agent/llm.py`: provider port, HTTP adapter, Mock LLM.
- `src/underwater_tracking/agent/prompts.py`: versioned intent, strategy, directive, and explanation prompts.
- `src/underwater_tracking/agent/nodes/*.py`: one focused carrier responsibility per file.
- `src/underwater_tracking/agent/graphs/{verify,central}.py`: graph assembly only.
- `src/underwater_tracking/persistence/{events,plans,ledger,checkpoints}.py`: durable repositories.
- `src/underwater_tracking/domain/agent_models.py`: intent, strategy, directive, plan, and decision contracts.

### Task 1: Add agent contracts and configuration

**Files:**
- Create: `configs/agent.yaml`
- Create: `configs/llm.yaml`
- Create: `src/underwater_tracking/domain/agent_models.py`
- Modify: `src/underwater_tracking/config/models.py`
- Modify: `src/underwater_tracking/config/loader.py`
- Test: `tests/agent/test_agent_models.py`

- [ ] **Step 1: Write the failing strict-model test**

```python
import pytest
from pydantic import ValidationError
from underwater_tracking.domain.agent_models import IntentHypothesis, StrategyProposal


def test_intent_requires_evidence_and_strategy_cannot_assign_uuvs():
    with pytest.raises(ValidationError):
        IntentHypothesis(label="evade", confidence=0.8, evidence_ids=[])
    with pytest.raises(ValidationError):
        StrategyProposal(
            concept="balanced", target_priorities={"T1": 1.0},
            required_quality={"T1": 0.7}, evidence_ids=["B:T1:900"],
            member_ids_by_target={"T1": ["U1", "U2"]},
        )
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/agent/test_agent_models.py -v`

Expected: FAIL importing `agent_models`.

- [ ] **Step 3: Implement strict agent models**

```python
from typing import Literal
from pydantic import Field, model_validator
from underwater_tracking.domain.models import StrictModel


IntentLabel = Literal["transit", "patrol", "loiter", "evade", "approach", "withdraw", "unknown"]
Concept = Literal["quality_first", "balanced", "resource_saving", "hold_current"]


class IntentHypothesis(StrictModel):
    label: IntentLabel
    confidence: float = Field(ge=0, le=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    alternatives: dict[IntentLabel, float] = Field(default_factory=dict)
    planning_effects: tuple[str, ...] = ()
    model_id: str
    prompt_version: str


class StrategyProposal(StrictModel):
    concept: Concept
    target_priorities: dict[str, float]
    required_quality: dict[str, float]
    reinforcement_policy: dict[str, str]
    releasable_soft_constraints: tuple[str, ...]
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    rationale: str


class ExpertDirective(StrictModel):
    directive_id: str
    raw_text: str
    target_scope: tuple[str, ...]
    locked_members: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    target_priorities: dict[str, float] = {}
    minimum_quality: dict[str, float] = {}
    disabled_uuv_ids: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)
    conflicts: tuple[str, ...] = ()
    status: Literal["preview", "applied", "rejected", "needs_clarification"] = "preview"

    @model_validator(mode="after")
    def ambiguity_requires_clarification(self):
        if self.confidence < 0.70 and self.status == "applied":
            raise ValueError("low-confidence directives cannot be applied")
        return self
```

Also define immutable `PredictedTrackRef`, `StrategySet`, `TrackingPlan`, `PlanCommand`, `PlanDiff`, `ValidationIssue`, `ValidationReport`, and `DecisionRecord` with the exact fields from the approved specification. Keep all final member IDs and waypoints in `TrackingPlan`, never `StrategyProposal`.

- [ ] **Step 4: Add validated config defaults**

`configs/agent.yaml` must set transport retries `3`, semantic repairs `2`, warning/critical persistence, event cooldown, history token threshold, and intent-change confirmation `confidence=0.70`, `margin=0.15`, `consecutive=2`. `configs/llm.yaml` must contain model/base URL/temperature/timeouts but reference the API key by environment variable name only.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/agent/test_agent_models.py tests/config -v`

Expected: all tests PASS.

```powershell
git add configs/agent.yaml configs/llm.yaml src/underwater_tracking/domain/agent_models.py src/underwater_tracking/config tests/agent/test_agent_models.py
git commit -m "feat: define strict assistant contracts"
```

### Task 2: Implement durable event, plan, and decision repositories

**Files:**
- Create: `src/underwater_tracking/persistence/sqlite.py`
- Create: `src/underwater_tracking/persistence/events.py`
- Create: `src/underwater_tracking/persistence/plans.py`
- Create: `src/underwater_tracking/persistence/ledger.py`
- Create: `src/underwater_tracking/persistence/checkpoints.py`
- Test: `tests/agent/test_repositories.py`

- [ ] **Step 1: Write a failing atomic-plan test**

```python
from underwater_tracking.persistence.plans import PlanRepository, StaleSnapshotError


def test_plan_commit_rejects_stale_snapshot(tmp_path, valid_plan):
    repo = PlanRepository(tmp_path / "run.db")
    repo.set_snapshot_revision("S1", 5)
    valid_plan.base_snapshot_revision = 4
    with pytest.raises(StaleSnapshotError):
        repo.commit(valid_plan)
    assert repo.get_active("S1") is None
```

- [ ] **Step 2: Implement SQLite migrations and repositories**

Create tables `runtime_events`, `snapshots`, `plans`, `plan_commands`, `decision_records`, `llm_calls`, `expert_directives`, and `question_runs`. Use WAL mode, foreign keys, explicit transactions, canonical JSON serialization with sorted keys, and integer simulation timestamps. `PlanRepository.commit()` must begin an immediate transaction, compare the stored scenario revision, insert the validated plan, supersede the previous active plan, and commit only if all writes succeed.

- [ ] **Step 3: Implement graph persistence factory**

Expose `create_checkpointer(database_path)` returning the LangGraph SQLite saver and `create_store(database_path)` for long-term memory namespaces. Tests use an in-memory saver; runtime uses SQLite.

- [ ] **Step 4: Test rollback and restart recovery**

Inject a failure between plan insert and supersede update, assert rollback leaves the old plan active, reopen the repository, and assert the active revision is unchanged.

Run: `python -m pytest tests/agent/test_repositories.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit persistence**

```powershell
git add src/underwater_tracking/persistence tests/agent/test_repositories.py
git commit -m "feat: add durable event plan and decision stores"
```

### Task 3: Add a provider-neutral LLM port and deterministic mocks

**Files:**
- Create: `src/underwater_tracking/agent/llm.py`
- Create: `tests/fixtures/llm_responses.py`
- Test: `tests/agent/test_llm_port.py`

- [ ] **Step 1: Write failing transport and structured-output tests**

```python
from underwater_tracking.agent.llm import MockStructuredLLM, TransientLLMError
from underwater_tracking.domain.agent_models import IntentHypothesis


def test_mock_llm_returns_validated_model():
    llm = MockStructuredLLM({"intent": {"label": "transit", "confidence": 0.8,
        "evidence_ids": ["B:T1:900"], "model_id": "mock", "prompt_version": "intent-v1"}})
    result = llm.invoke_structured("intent", {}, IntentHypothesis)
    assert result.label == "transit"


def test_transient_failures_retry_exactly_three_times(counting_transport):
    counting_transport.fail_with(TransientLLMError, times=3)
    with pytest.raises(TransientLLMError):
        counting_transport.client.invoke_structured("intent", {}, IntentHypothesis)
    assert counting_transport.calls == 3
```

- [ ] **Step 2: Implement the port**

Define a generic `StructuredLLM` protocol whose `invoke_structured(operation: str, payload: dict[str, object], response_model: type[T]) -> T` method is the only call surface. Concrete clients must return `response_model.model_validate(response_json)`. Implement `HTTPStructuredLLM` using configured base URL, bearer token read at call time from the configured environment variable, timeout, exponential backoff with deterministic jitter source injection, and request metadata hooks. Retry only timeout, connection, 429, and 5xx failures. Implement `MockStructuredLLM` as a queue keyed by operation, supporting valid objects, invalid dictionaries, and injected exceptions.

- [ ] **Step 3: Redact secrets and persist call metadata**

Record operation, model, prompt version, request hash, response hash, latency, token metadata when available, and error category. Never persist authorization headers, API keys, or the full environment.

- [ ] **Step 4: Run and commit**

Run: `python -m pytest tests/agent/test_llm_port.py -v`

Expected: all tests PASS.

```powershell
git add src/underwater_tracking/agent/llm.py tests/fixtures/llm_responses.py tests/agent/test_llm_port.py
git commit -m "feat: add provider neutral structured llm port"
```

### Task 4: Classify and coalesce strategic, tactical, and informational events

**Files:**
- Create: `src/underwater_tracking/agent/nodes/event_monitor.py`
- Create: `src/underwater_tracking/agent/state.py`
- Test: `tests/agent/test_event_monitor.py`

- [ ] **Step 1: Write a failing hysteresis test**

```python
from underwater_tracking.agent.nodes.event_monitor import EventMonitor


def test_quality_warning_requires_two_minutes_and_deduplicates():
    monitor = EventMonitor(warning_threshold=0.65, warning_hold_s=120, cooldown_s=300)
    assert monitor.observe_quality("G-T1", 30, 0.60) == ()
    assert monitor.observe_quality("G-T1", 120, 0.60) == ()
    events = monitor.observe_quality("G-T1", 150, 0.60)
    assert [event.event_type for event in events] == ["group_quality_warning"]
    assert monitor.observe_quality("G-T1", 180, 0.59) == ()
```

- [ ] **Step 2: Implement explicit state and route decisions**

`CarrierState` contains scenario ID, snapshot revision/reference, pending/coalesced events, route, intent hypotheses, predictions, strategy set, validation attempts, candidate plans, selected plan, latest directive/question, history summaries, errors, and output messages. It stores references rather than raw observation histories.

Map initialization, target add/remove/lost, confirmed intent change, major failure, infeasible repair, and applied directives to strategic. Map quality warning, geometry degradation, battery rotation, and replaceable single failure to tactical. Map progress timers and questions to informational/question branches. Coalesce entity/type duplicates in the cooldown window and retain the latest payload.

- [ ] **Step 3: Add escalation tests**

Test immediate critical quality, two consecutive intent analyses with confidence/margin gates, tactical repair infeasibility escalating to strategic, and a failed member causing strategic routing when the group drops below two.

Run: `python -m pytest tests/agent/test_event_monitor.py -v`

Expected: all tests PASS.

- [ ] **Step 4: Commit event routing**

```powershell
git add src/underwater_tracking/agent/state.py src/underwater_tracking/agent/nodes/event_monitor.py tests/agent/test_event_monitor.py
git commit -m "feat: add tiered event monitor"
```

### Task 5: Implement versioned prompts, intent analysis, and strategy generation

**Files:**
- Create: `src/underwater_tracking/agent/prompts.py`
- Create: `src/underwater_tracking/agent/nodes/intent.py`
- Create: `src/underwater_tracking/agent/nodes/strategy.py`
- Test: `tests/agent/test_semantic_nodes.py`

- [ ] **Step 1: Write failing prompt-boundary tests**

```python
def test_intent_payload_uses_history_features_not_truth(intent_node, snapshot):
    payload = intent_node.build_payload(snapshot, target_id="T1")
    assert "truth" not in repr(payload).lower()
    assert payload["trajectory_features"]
    assert payload["sampled_belief_history"]


def test_major_event_requests_three_concepts(strategy_node, strategic_state):
    result = strategy_node(strategic_state)
    assert {item.concept for item in result["strategy_set"]} == {
        "quality_first", "balanced", "resource_saving"
    }
```

- [ ] **Step 2: Create immutable prompt templates**

Define `INTENT_PROMPT_VERSION = "intent-v1"`, `STRATEGY_PROMPT_VERSION = "strategy-v1"`, `DIRECTIVE_PROMPT_VERSION = "directive-v1"`, and `EXPLANATION_PROMPT_VERSION = "explanation-v1"`. Each system prompt states the allowed evidence, output schema purpose, no-truth rule, and prohibition on final member/waypoint output. Payload builders sort IDs and serialize only required fields.

- [ ] **Step 3: Implement semantic nodes**

`intent_analysis` loops over targets needing analysis and invokes `IntentHypothesis`. `strategy_generation` requests exactly three proposals for strategic events and one `hold_current` or modified proposal for periodic review. Both nodes attach model/prompt versions and raw response hashes to state.

- [ ] **Step 4: Run and commit**

Run: `python -m pytest tests/agent/test_semantic_nodes.py -v`

Expected: all tests PASS with Mock LLM.

```powershell
git add src/underwater_tracking/agent/prompts.py src/underwater_tracking/agent/nodes/intent.py src/underwater_tracking/agent/nodes/strategy.py tests/agent/test_semantic_nodes.py
git commit -m "feat: add intent and strategy semantic nodes"
```

### Task 6: Build the semantic Verify subgraph with two repairs

**Files:**
- Create: `src/underwater_tracking/agent/nodes/verify.py`
- Create: `src/underwater_tracking/agent/graphs/verify.py`
- Test: `tests/agent/test_verify_graph.py`

- [ ] **Step 1: Write a failing repair-limit test**

```python
def test_verify_repairs_twice_then_degrades(verify_graph, invalid_strategy_queue):
    result = verify_graph.invoke({
        "candidate": invalid_strategy_queue.first,
        "attempt": 0,
        "max_repairs": 2,
        "last_valid_strategy": invalid_strategy_queue.last_valid,
    })
    assert result["repair_attempts"] == 2
    assert result["verified_strategy"] == invalid_strategy_queue.last_valid
    assert result["degraded"] is True
```

- [ ] **Step 2: Implement validators**

Check strict Pydantic validity, known target IDs, finite priorities/quality, complete target coverage, evidence ID existence, allowed soft constraints, no member IDs/waypoints, and consistency with expert hard constraints. Return a sorted tuple of `ValidationIssue(code, field, message, observed, expected)`.

- [ ] **Step 3: Assemble the repair graph**

Graph nodes: `validate → route_validity`; valid ends, invalid with attempts remaining goes to `repair`, then back to `validate`; exhausted goes to `fallback`. `repair` sends the original candidate plus machine-readable issues and the unchanged schema. `fallback` selects the last valid strategy or a deterministic emergency strategy that prioritizes already tracked high-priority targets.

- [ ] **Step 4: Test valid-first-pass, repaired, and exhausted cases**

Assert `0`, `1`, and `2` repair counts respectively; assert no invalid object appears under `verified_strategy`; assert transport retries do not increment semantic attempts.

Run: `python -m pytest tests/agent/test_verify_graph.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit Verify**

```powershell
git add src/underwater_tracking/agent/nodes/verify.py src/underwater_tracking/agent/graphs/verify.py tests/agent/test_verify_graph.py
git commit -m "feat: add bounded semantic verify graph"
```

### Task 7: Adapt deterministic optimization into versioned plan commit

**Files:**
- Create: `src/underwater_tracking/agent/nodes/snapshot.py`
- Create: `src/underwater_tracking/agent/nodes/optimize.py`
- Create: `src/underwater_tracking/agent/nodes/commit.py`
- Test: `tests/agent/test_plan_pipeline.py`

- [ ] **Step 1: Write a failing stale-plan test**

```python
def test_stale_plan_is_rejected_before_broadcast(plan_pipeline, repositories):
    state = plan_pipeline.make_state(snapshot_revision=4)
    candidate = plan_pipeline.optimize(state)
    repositories.snapshots.set_revision("S1", 5)
    result = plan_pipeline.commit(state, candidate)
    assert result["commit_status"] == "stale"
    assert repositories.commands.list_for_scenario("S1") == []
```

- [ ] **Step 2: Implement snapshot and optimizer adapters**

Build an immutable snapshot from current group reports, resources, events, active plan, and applied directives. For every verified strategy, call Foundation allocation and waypoint services, calculate expected quality/resources/energy/churn, and construct a candidate `TrackingPlan`. Sort candidates lexicographically by hard violations, active count, economic cost, then stable concept order.

- [ ] **Step 3: Implement independent plan validation and atomic commit**

Re-check target coverage, 2–4 members, uniqueness, resource health, return reserve, waypoint bounds/separation/kinematics, evidence references, and base snapshot revision. Commit the selected plan in one repository transaction, create one `PlanCommand` per group, then publish commands only after transaction success.

- [ ] **Step 4: Test no-change and emergency plans**

Assert a periodic review with no material objective improvement returns `hold_current` without a new revision. Assert an infeasible previous plan after a failure yields a `DEGRADED` emergency plan that retains highest-priority feasible targets.

Run: `python -m pytest tests/agent/test_plan_pipeline.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit plan pipeline**

```powershell
git add src/underwater_tracking/agent/nodes/snapshot.py src/underwater_tracking/agent/nodes/optimize.py src/underwater_tracking/agent/nodes/commit.py tests/agent/test_plan_pipeline.py
git commit -m "feat: add versioned deterministic plan pipeline"
```

### Task 8: Assemble the persistent carrier graph

**Files:**
- Create: `src/underwater_tracking/agent/graphs/central.py`
- Create: `src/underwater_tracking/agent/runtime.py`
- Test: `tests/agent/test_central_graph.py`

- [ ] **Step 1: Write failing route integration tests**

```python
def test_tactical_route_never_calls_llm(carrier, quality_warning_state, spy_llm):
    result = carrier.invoke(quality_warning_state)
    assert result["route"] == "tactical"
    assert spy_llm.calls == []
    assert result["selected_plan"] is not None


def test_strategic_route_runs_full_chain(carrier, target_added_state, spy_llm):
    result = carrier.invoke(target_added_state)
    assert [call.operation for call in spy_llm.calls] == ["intent", "strategy"]
    assert result["commit_status"] == "committed"
```

- [ ] **Step 2: Assemble the graph exactly once**

Add `ingest`, `event_monitor`, `build_snapshot`, `intent_analysis`, `trajectory_prediction`, `strategy_generation`, Verify subgraph, `resource_optimizer`, `verify_plan`, `commit_plan`, `record_decision`, `progress_report`, and `handle_error`. Conditional edges route strategic through the full chain, tactical through prediction/optimization, and informational through record/report. Questions and directives enter dedicated branches added in later tasks.

- [ ] **Step 3: Compile with injected dependencies**

`build_carrier_graph(dependencies, checkpointer, store)` accepts repositories, LLM, predictor, optimizer, and clock as ports. `CarrierRuntime` owns a scenario thread ID and offers `submit_event`, `tick`, `get_state`, and `resume` without exposing graph internals.

- [ ] **Step 4: Test checkpoint restart**

Run a strategic commit, close runtime, reopen with the same SQLite database and thread ID, submit a tactical event, and assert the plan revision continues monotonically from the prior checkpoint.

Run: `python -m pytest tests/agent/test_central_graph.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit carrier graph**

```powershell
git add src/underwater_tracking/agent/graphs/central.py src/underwater_tracking/agent/runtime.py tests/agent/test_central_graph.py
git commit -m "feat: assemble persistent carrier langgraph"
```

### Task 9: Add evidence-preserving History compression

**Files:**
- Create: `src/underwater_tracking/agent/nodes/history.py`
- Create: `src/underwater_tracking/agent/graphs/history.py`
- Test: `tests/agent/test_history.py`

- [ ] **Step 1: Write a failing evidence-retention test**

```python
def test_history_summary_keeps_retrievable_evidence(history_graph, event_store):
    result = history_graph.invoke({"scenario_id": "S1", "window_end_s": 3600})
    assert result["operational_summary"].evidence_ids
    for evidence_id in result["operational_summary"].evidence_ids:
        assert event_store.get(evidence_id) is not None
```

- [ ] **Step 2: Implement three summary models and trigger policy**

Create `OperationalSummary`, `DecisionSummary`, and `ConversationSummary`; each contains covered interval, facts, unresolved risks, and evidence IDs. Trigger compression when time window, event count, message count, or token estimate crosses configuration. Summaries are appended by namespace and never delete source records.

- [ ] **Step 3: Implement context assembly**

`build_planning_context()` returns current snapshot, active plan, applied directives, last critical events, and only summaries matching target/event evidence. Enforce a deterministic character/token budget by stable priority and truncation at record boundaries.

- [ ] **Step 4: Run and commit**

Run: `python -m pytest tests/agent/test_history.py -v`

Expected: all tests PASS.

```powershell
git add src/underwater_tracking/agent/nodes/history.py src/underwater_tracking/agent/graphs/history.py tests/agent/test_history.py
git commit -m "feat: add evidence preserving history compression"
```

### Task 10: Implement non-blocking expert directives

**Files:**
- Create: `src/underwater_tracking/agent/nodes/directives.py`
- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Test: `tests/agent/test_directives.py`

- [ ] **Step 1: Write a failing non-blocking ambiguity test**

```python
def test_ambiguous_directive_requests_clarification_without_stopping_plan(runtime):
    before = runtime.active_plan()
    preview = runtime.preview_directive("多派一些艇过去")
    assert preview.status == "needs_clarification"
    assert runtime.active_plan() == before
    runtime.tick()
    assert runtime.group_updates_advanced()
```

- [ ] **Step 2: Implement preview, conflict check, and apply**

`preview_directive(text)` invokes the directive schema, validates IDs/resources/quality and conflicts, persists the preview, and never changes graph state. `apply_directive(id)` rejects low-confidence/conflicting previews, marks a clean preview applied, emits `expert_directive_applied`, and returns immediately while the existing plan remains active until commit.

- [ ] **Step 3: Add structured shortcuts**

Implement typed helpers `lock_group_members`, `set_target_priority`, `set_minimum_quality`, and `disable_uuv`; they create the same `ExpertDirective` model and pass the same validator.

- [ ] **Step 4: Run and commit**

Run: `python -m pytest tests/agent/test_directives.py -v`

Expected: all tests PASS.

```powershell
git add src/underwater_tracking/agent/nodes/directives.py src/underwater_tracking/agent/graphs/central.py tests/agent/test_directives.py
git commit -m "feat: add non blocking expert directives"
```

### Task 11: Implement evidence-backed questions and isolated counterfactuals

**Files:**
- Create: `src/underwater_tracking/agent/nodes/questions.py`
- Create: `src/underwater_tracking/agent/counterfactual.py`
- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Test: `tests/agent/test_questions.py`

- [ ] **Step 1: Write a failing read-only question test**

```python
def test_question_and_counterfactual_do_not_change_online_plan(runtime):
    before = runtime.active_plan()
    answer = runtime.ask("为什么没有给 T2 增派 UUV？", counterfactual={"T2.min_quality": 0.85})
    assert answer.evidence_ids
    assert answer.counterfactual_plan_id.startswith("dry-run:")
    assert runtime.active_plan() == before
```

- [ ] **Step 2: Implement evidence retrieval**

Parse target/UUV/plan IDs using deterministic entity matching, query DecisionLedger, plan diffs, validation issues, and observations by evidence ID, then build a bounded payload. Reject answers whose cited evidence IDs are absent.

- [ ] **Step 3: Implement dry-run isolation**

Clone the historical snapshot into an in-memory repository with `run_id="dry-run:<uuid>"`, apply allowed counterfactual overrides, run the deterministic optimizer, and return plan diff/objective changes. The dry-run dependency container must not receive the online `PlanRepository.commit` method or event publisher.

- [ ] **Step 4: Run and commit**

Run: `python -m pytest tests/agent/test_questions.py -v`

Expected: all tests PASS.

```powershell
git add src/underwater_tracking/agent/nodes/questions.py src/underwater_tracking/agent/counterfactual.py src/underwater_tracking/agent/graphs/central.py tests/agent/test_questions.py
git commit -m "feat: add evidence backed expert questions"
```

### Task 12: Integrate the Agent CLI and verify degradation

**Files:**
- Modify: `src/underwater_tracking/cli.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Create: `tests/integration/test_agent_loop.py`
- Modify: `README.md`

- [ ] **Step 1: Write a failing end-to-end Mock LLM test**

Run a 90-minute simulated scenario with initialization, a target addition, one malformed strategy repaired on retry, a UUV failure, a directive, and a question. Assert monotonic plan revisions, no invalid commit, group updates every 30 seconds, evidence IDs in the answer, and at least one tactical route with zero LLM calls.

- [ ] **Step 2: Add `agent-run` CLI command**

The command loads config, creates SQLite repositories and checkpointer, selects `--llm mock` or configured provider, wires engine group reports into `CarrierRuntime`, applies emitted plan commands back to group manager, and writes run manifest plus JSONL.

```powershell
python -m underwater_tracking.cli agent-run --config configs/scenario/default.yaml --steps 540 --seed 42 --llm mock
```

- [ ] **Step 3: Inject storage and provider failures**

Test provider exhaustion returns the last valid strategy/emergency plan, checkpoint failure prevents new central commits while group local updates advance, and restart restores the last committed revision.

- [ ] **Step 4: Run the complete Agent verification**

```powershell
python -m pytest tests/agent tests/integration/test_agent_loop.py -q
python -m ruff check src/underwater_tracking/agent src/underwater_tracking/persistence tests/agent
python -m mypy src/underwater_tracking/agent src/underwater_tracking/persistence
python -m underwater_tracking.cli agent-run --config configs/scenario/default.yaml --steps 540 --seed 42 --llm mock
```

Expected: all commands exit `0`; no invalid plan is committed; the CLI writes plan and decision records.

- [ ] **Step 5: Document and commit the Agent runtime**

```powershell
git add src/underwater_tracking/cli.py src/underwater_tracking/simulation/engine.py tests/integration/test_agent_loop.py README.md
git commit -m "feat: integrate resilient langgraph tracking assistant"
```
