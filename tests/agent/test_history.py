# tests/agent/test_history.py
"""Evidence-preserving History compression tests (spec 9, plan Task 9).

Covers the brief's verbatim evidence-retention test (every summary evidence
id must resolve via ``event_store.get``), the three summary namespaces
appended to the long-term memory store, the deterministic trigger policy
(window/event/message/token thresholds, reading the agent configuration),
the append-only guarantee (compression never deletes source records), and
``build_planning_context``: only summaries matching the current target/event
evidence are loaded, and the deterministic budget truncates at whole-record
boundaries. All behaviour is deterministic: no randomness anywhere.
"""

import pytest

from underwater_tracking.agent.graphs.history import build_history_graph, build_planning_context
from underwater_tracking.agent.nodes.history import (
    ConversationSummary,
    HistoryTriggerPolicy,
    OperationalSummary,
)
from underwater_tracking.config.models import AgentConfig
from underwater_tracking.domain.agent_models import (
    DecisionRecord,
    ExpertDirective,
    TrackingPlan,
    Waypoint,
)
from underwater_tracking.persistence.checkpoints import create_store
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository

SCENARIO_ID = "S1"
WINDOW_END_S = 3600


def _seed_events(store: EventRepository) -> None:
    """Deterministic scenario history: bearings, a quality warning, a
    resource-critical event, and an applied-directive event — every event id
    below resolves through ``EventRepository.get``."""
    store.append(
        event_id="B:T1:300",
        event_type="bearing",
        scenario_id=SCENARIO_ID,
        sim_time_s=300,
        payload={"azimuth_rad": 1.0, "quality": 0.82},
        target_id="T1",
    )
    store.append(
        event_id="E:directive:600",
        event_type="directive_applied",
        scenario_id=SCENARIO_ID,
        sim_time_s=600,
        payload={"directive_id": "X1"},
    )
    store.append(
        event_id="E:quality:900",
        event_type="quality_warning",
        scenario_id=SCENARIO_ID,
        sim_time_s=900,
        payload={"quality": 0.55},
        target_id="T1",
        severity="warning",
    )
    store.append(
        event_id="E:resource:1500",
        event_type="resource_low",
        scenario_id=SCENARIO_ID,
        sim_time_s=1500,
        payload={"energy": 0.25},
        severity="critical",
    )


def _seed_ledger(ledger: DecisionLedger) -> None:
    """One traceable decision, one applied directive, one question run."""
    ledger.record(
        DecisionRecord(
            decision_id="D:900",
            scenario_id=SCENARIO_ID,
            sim_time_s=900,
            trigger_event_ids=("E:quality:900",),
            snapshot_revision=2,
            input_evidence_ids=("B:T1:300", "E:quality:900"),
            rejected_candidates={"P2": "fails minimum quality"},
            final_plan_id="S1:plan:1",
        )
    )
    ledger.save_directive(
        ExpertDirective(
            directive_id="X1",
            raw_text="keep two boats on T1",
            target_scope=("T1",),
            confidence=0.9,
            status="applied",
        ),
        scenario_id=SCENARIO_ID,
    )
    ledger.save_directive(
        ExpertDirective(
            directive_id="X2",
            raw_text="move all boats to T2",
            target_scope=("T2",),
            confidence=0.4,
            status="needs_clarification",
        ),
        scenario_id=SCENARIO_ID,
    )
    ledger.save_question(
        run_id="Q1",
        scenario_id=SCENARIO_ID,
        question_text="why is T1 quality warning?",
        payload={"answer": "geometry degrades", "evidence_ids": ["E:quality:900"]},
    )


def _commit_plan(plans: PlanRepository) -> None:
    """Commit one minimal valid plan so the context can load it as active."""
    plans.set_snapshot_revision(SCENARIO_ID, 2)
    plans.commit(
        TrackingPlan(
            plan_id="S1:plan:1",
            scenario_id=SCENARIO_ID,
            revision=1,
            base_snapshot_revision=2,
            status="validating",
            target_priorities={"T1": 1.0},
            member_ids_by_target={"T1": ("U1", "U2")},
            roles_by_member={"U1": "lead", "U2": "wing"},
            waypoints_by_member={
                "U1": (Waypoint(x=0.0, y=0.0, arrive_at_s=30),),
                "U2": (Waypoint(x=300.0, y=0.0, arrive_at_s=30),),
            },
            predicted_active_count=2,
        )
    )


@pytest.fixture
def event_store(tmp_path) -> EventRepository:
    store = EventRepository(tmp_path / "run.db")
    _seed_events(store)
    yield store
    store.close()


@pytest.fixture
def ledger(tmp_path) -> DecisionLedger:
    repo = DecisionLedger(tmp_path / "run.db")
    _seed_ledger(repo)
    yield repo
    repo.close()


@pytest.fixture
def memory_store(tmp_path):
    return create_store(tmp_path / "mem.db")


@pytest.fixture
def history_graph(event_store, ledger, memory_store):
    return build_history_graph(event_store, ledger, memory_store)


def test_history_summary_keeps_retrievable_evidence(history_graph, event_store):
    result = history_graph.invoke({"scenario_id": "S1", "window_end_s": 3600})
    assert result["operational_summary"].evidence_ids
    for evidence_id in result["operational_summary"].evidence_ids:
        assert event_store.get(evidence_id) is not None


def test_all_summary_kinds_keep_resolvable_evidence(history_graph, event_store):
    result = history_graph.invoke({"scenario_id": "S1", "window_end_s": WINDOW_END_S})
    for summary in (
        result["operational_summary"],
        result["decision_summary"],
        result["conversation_summary"],
    ):
        assert summary.evidence_ids
        for evidence_id in summary.evidence_ids:
            assert event_store.get(evidence_id) is not None
    conversation = result["conversation_summary"]
    assert "keep two boats on T1" in conversation.expert_annotations
    assert "move all boats to T2" in conversation.clarifications
    assert "clarification pending: X2" in conversation.unresolved_risks
    assert "why is T1 quality warning?" in conversation.question_topics


def test_below_window_does_not_compress(history_graph):
    result = history_graph.invoke({"scenario_id": "S1", "window_end_s": 60})
    assert result["compressed"] is False
    assert result["operational_summary"] is None
    assert result["decision_summary"] is None
    assert result["conversation_summary"] is None


def test_summaries_appended_by_namespace(history_graph, memory_store):
    result = history_graph.invoke({"scenario_id": "S1", "window_end_s": WINDOW_END_S})
    operational = result["operational_summary"]
    decision = result["decision_summary"]
    conversation = result["conversation_summary"]
    item = memory_store.get(
        ("scenario", "S1", "history", "operational"), operational.summary_id
    )
    assert item is not None
    assert OperationalSummary.model_validate(item.value) == operational
    assert (
        memory_store.get(
            ("scenario", "S1", "history", "decision"), decision.summary_id
        )
        is not None
    )
    assert (
        memory_store.get(
            ("scenario", "S1", "history", "conversation"), conversation.summary_id
        )
        is not None
    )
    # The returned and stored summaries cover the compressed window: from
    # the earliest covered event up to the window end.
    assert (operational.start_time_s, operational.end_time_s) == (300, WINDOW_END_S)
    assert operational.scenario_id == SCENARIO_ID


def test_compression_never_deletes_source_records(history_graph, event_store, ledger):
    events_before = [e.event_id for e in event_store.list_events(scenario_id="S1")]
    history_graph.invoke({"scenario_id": "S1", "window_end_s": WINDOW_END_S})
    events_after = [e.event_id for e in event_store.list_events(scenario_id="S1")]
    assert events_after == events_before
    assert ledger.get("D:900") is not None


def test_trigger_policy_crosses_thresholds():
    policy = HistoryTriggerPolicy(token_threshold=6000)
    assert policy.should_compress(
        covered_window_s=0, covered_event_count=0, covered_message_count=0, estimated_tokens=6000
    )
    assert not policy.should_compress(
        covered_window_s=0, covered_event_count=0, covered_message_count=0, estimated_tokens=5999
    )
    assert policy.should_compress(
        covered_window_s=900, covered_event_count=1, covered_message_count=0, estimated_tokens=10
    )
    assert policy.should_compress(
        covered_window_s=0, covered_event_count=50, covered_message_count=0, estimated_tokens=0
    )
    assert policy.should_compress(
        covered_window_s=0, covered_event_count=0, covered_message_count=20, estimated_tokens=0
    )


def test_trigger_policy_reads_agent_config_threshold():
    config = AgentConfig()
    assert HistoryTriggerPolicy.from_agent_config(config).token_threshold == (
        config.history_token_threshold
    )


def test_planning_context_loads_only_matching_summaries(
    history_graph, event_store, ledger, memory_store, tmp_path
):
    history_graph.invoke({"scenario_id": "S1", "window_end_s": WINDOW_END_S})
    plans = PlanRepository(tmp_path / "run.db")
    _commit_plan(plans)
    try:
        context = build_planning_context(
            scenario_id=SCENARIO_ID,
            window_end_s=WINDOW_END_S,
            events=event_store,
            ledger=ledger,
            plans=plans,
            store=memory_store,
            relevant_evidence_ids=("B:T1:300",),
        )
    finally:
        plans.close()
    assert context.summaries
    # Only summaries whose evidence matches the current target/event evidence.
    assert all(frozenset(summary.evidence_ids) & {"B:T1:300"} for summary in context.summaries)
    assert not any(isinstance(s, ConversationSummary) for s in context.summaries)
    # Snapshot, active plan, directives and last critical events are loaded.
    assert context.active_plan is not None
    assert context.active_plan.plan_id == "S1:plan:1"
    assert [d.directive_id for d in context.applied_directives] == ["X1"]
    assert [e.event_id for e in context.critical_events] == [
        "E:quality:900",
        "E:resource:1500",
    ]
    assert context.text == "\n".join(context.records)


def test_planning_context_budget_truncates_at_record_boundaries(
    history_graph, event_store, ledger, memory_store, tmp_path
):
    history_graph.invoke({"scenario_id": "S1", "window_end_s": WINDOW_END_S})
    plans = PlanRepository(tmp_path / "run.db")
    try:
        tight = build_planning_context(
            scenario_id=SCENARIO_ID,
            window_end_s=WINDOW_END_S,
            events=event_store,
            ledger=ledger,
            plans=plans,
            store=memory_store,
            budget_chars=150,
        )
        wide = build_planning_context(
            scenario_id=SCENARIO_ID,
            window_end_s=WINDOW_END_S,
            events=event_store,
            ledger=ledger,
            plans=plans,
            store=memory_store,
            budget_chars=10_000,
        )
        empty = build_planning_context(
            scenario_id=SCENARIO_ID,
            window_end_s=WINDOW_END_S,
            events=event_store,
            ledger=ledger,
            plans=plans,
            store=memory_store,
            budget_chars=5,
        )
    finally:
        plans.close()
    assert tight.text_chars <= 150
    assert tight.text == "\n".join(tight.records)
    assert len(tight.text) < len(wide.text)
    assert empty.text == ""
    assert empty.records == ()


def test_planning_context_is_deterministic(
    history_graph, event_store, ledger, memory_store, tmp_path
):
    history_graph.invoke({"scenario_id": "S1", "window_end_s": WINDOW_END_S})
    plans = PlanRepository(tmp_path / "run.db")
    try:
        first = build_planning_context(
            scenario_id=SCENARIO_ID,
            window_end_s=WINDOW_END_S,
            events=event_store,
            ledger=ledger,
            plans=plans,
            store=memory_store,
            budget_chars=500,
        )
        second = build_planning_context(
            scenario_id=SCENARIO_ID,
            window_end_s=WINDOW_END_S,
            events=event_store,
            ledger=ledger,
            plans=plans,
            store=memory_store,
            budget_chars=500,
        )
    finally:
        plans.close()
    assert first.text == second.text
    assert first.records == second.records
