"""Durable repository tests: events, plans, ledger, and graph persistence.

Covers the brief's atomic-plan stale-commit test verbatim, the commit
transactionality/rollback requirement, restart recovery, canonical JSON,
WAL/foreign-key database setup, and the LangGraph SQLite checkpointer/store
factories. ``valid_plan`` is a function-scoped fixture so every test gets an
independent mutable plan.
"""

import pytest

from underwater_tracking.domain.agent_models import (
    DecisionRecord,
    ExpertDirective,
    PlanCommand,
    TrackingPlan,
    Waypoint,
)
from underwater_tracking.persistence.checkpoints import create_checkpointer, create_store
from underwater_tracking.persistence.events import EventRepository, StoredEvent
from underwater_tracking.persistence.ledger import DecisionLedger, LlmCallRecord, QuestionRun
from underwater_tracking.persistence.plans import PlanRepository, StaleSnapshotError
from underwater_tracking.persistence.sqlite import SCHEMA_VERSION, json_dumps, open_database

_ALL_TABLES = {
    "runtime_events",
    "snapshots",
    "plans",
    "plan_commands",
    "decision_records",
    "llm_calls",
    "expert_directives",
    "question_runs",
}


def build_plan(*, plan_id: str, revision: int, base_revision: int = 5) -> TrackingPlan:
    """A valid, strict ``TrackingPlan`` for scenario ``S1``."""
    return TrackingPlan(
        plan_id=plan_id,
        scenario_id="S1",
        revision=revision,
        base_snapshot_revision=base_revision,
        status="validating",
        valid_until_s=3600,
        concept="balanced",
        target_priorities={"T1": 1.0},
        member_ids_by_target={"T1": ("U1", "U2")},
        roles_by_member={"U1": "lead", "U2": "wing"},
        waypoints_by_member={"U1": (Waypoint(x=100.0, y=200.0, arrive_at_s=30),)},
        predicted_active_count=2,
    )


@pytest.fixture
def valid_plan() -> TrackingPlan:
    return build_plan(plan_id="P1", revision=1)


def test_plan_commit_rejects_stale_snapshot(tmp_path, valid_plan):
    repo = PlanRepository(tmp_path / "run.db")
    repo.set_snapshot_revision("S1", 5)
    valid_plan.base_snapshot_revision = 4
    with pytest.raises(StaleSnapshotError):
        repo.commit(valid_plan)
    assert repo.get_active("S1") is None


def test_stale_commit_keeps_existing_active_plan(tmp_path, valid_plan):
    repo = PlanRepository(tmp_path / "run.db")
    repo.set_snapshot_revision("S1", 5)
    repo.commit(valid_plan)
    stale = build_plan(plan_id="P2", revision=2, base_revision=4)
    with pytest.raises(StaleSnapshotError):
        repo.commit(stale)
    active = repo.get_active("S1")
    assert active is not None
    assert active.plan_id == "P1"
    assert active.revision == 1


def test_plan_commit_makes_active_and_supersedes_previous(tmp_path, valid_plan):
    repo = PlanRepository(tmp_path / "run.db")
    repo.set_snapshot_revision("S1", 5)
    repo.commit(valid_plan)
    first = repo.get_active("S1")
    assert first is not None
    assert first.plan_id == "P1"
    assert first.status == "active"
    repo.commit(build_plan(plan_id="P2", revision=2))
    second = repo.get_active("S1")
    assert second is not None
    assert second.plan_id == "P2"
    assert second.revision == 2
    assert second.status == "active"
    superseded = repo.get_plan("P1")
    assert superseded is not None
    assert superseded.status == "superseded"


def test_commit_rolls_back_when_supersede_fails(tmp_path, valid_plan, monkeypatch):
    """Inject a failure between plan insert and supersede: the whole commit
    must roll back, the old plan stays active, and reopening the database
    shows the same active revision."""
    repo = PlanRepository(tmp_path / "run.db")
    repo.set_snapshot_revision("S1", 5)
    repo.commit(valid_plan)
    successor = build_plan(plan_id="P2", revision=2)

    def fail_after_insert() -> None:
        raise RuntimeError("injected failure between insert and supersede")

    monkeypatch.setattr(repo, "_after_plan_insert", fail_after_insert)
    with pytest.raises(RuntimeError, match="injected failure"):
        repo.commit(successor)

    active = repo.get_active("S1")
    assert active is not None
    assert active.plan_id == "P1"
    assert active.revision == 1
    assert repo.get_plan("P2") is None

    reopened = PlanRepository(tmp_path / "run.db")
    recovered = reopened.get_active("S1")
    assert recovered is not None
    assert recovered.plan_id == "P1"
    assert recovered.revision == 1


def test_snapshot_revision_upserts(tmp_path):
    repo = PlanRepository(tmp_path / "run.db")
    assert repo.get_snapshot_revision("S1") == 0
    repo.set_snapshot_revision("S1", 5)
    assert repo.get_snapshot_revision("S1") == 5
    repo.set_snapshot_revision("S1", 6, snapshot_hash="h6")
    assert repo.get_snapshot_revision("S1") == 6


def test_plan_commands_roundtrip(tmp_path, valid_plan):
    repo = PlanRepository(tmp_path / "run.db")
    repo.set_snapshot_revision("S1", 5)
    repo.commit(valid_plan)
    command = PlanCommand(
        command_id="C1",
        plan_id="P1",
        plan_revision=1,
        scenario_id="S1",
        group_id="G1",
        target_id="T1",
        sim_time_s=600,
        member_ids=("U1", "U2"),
        waypoints_by_member={"U1": (Waypoint(x=100.0, y=200.0, arrive_at_s=30),)},
        actions={"U1": "transit"},
    )
    repo.save_command(command)
    commands = repo.list_commands("P1")
    assert len(commands) == 1
    assert commands[0].command_id == "C1"
    assert commands[0].member_ids == ("U1", "U2")
    assert commands[0].actions == {"U1": "transit"}
    assert repo.list_commands("P2") == []


def test_event_repository_appends_and_replays(tmp_path):
    repo = EventRepository(tmp_path / "run.db")
    first = repo.append(
        event_id="E1",
        event_type="bearing",
        scenario_id="S1",
        sim_time_s=300,
        payload={"azimuth_rad": 1.0},
    )
    second = repo.append(
        event_id="E2",
        event_type="bearing",
        scenario_id="S1",
        sim_time_s=310,
        target_id="T1",
        severity="warning",
        payload={"azimuth_rad": 1.1},
    )
    events = repo.list_events(scenario_id="S1")
    assert [event.event_id for event in events] == ["E1", "E2"]
    assert isinstance(events[0], StoredEvent)
    assert events[0].id < events[1].id
    assert events[1].target_id == "T1"
    assert events[1].severity == "warning"
    assert events[1].payload == {"azimuth_rad": 1.1}

    since = repo.list_events(scenario_id="S1", since_id=first)
    assert [event.event_id for event in since] == ["E2"]

    by_type = repo.list_events(scenario_id="S1", event_type="bearing")
    assert len(by_type) == 2
    assert repo.list_events(scenario_id="S1", event_type="contact") == []
    assert repo.list_events(scenario_id="S1", since_id=second) == []


def test_decision_ledger_roundtrip(tmp_path):
    ledger = DecisionLedger(tmp_path / "run.db")
    record = DecisionRecord(
        decision_id="D1",
        scenario_id="S1",
        sim_time_s=600,
        trigger_event_ids=("E1",),
        snapshot_revision=5,
        snapshot_hash="abc",
        model_version="m1",
        prompt_version="p1",
        schema_version="s1",
        rejected_candidates={"P2": "fails minimum quality"},
    )
    ledger.record(record)
    got = ledger.get("D1")
    assert got is not None
    assert got.decision_id == "D1"
    assert got.snapshot_revision == 5
    assert got.rejected_candidates == {"P2": "fails minimum quality"}
    assert [item.decision_id for item in ledger.list_decisions(scenario_id="S1")] == ["D1"]
    assert ledger.get("missing") is None
    assert ledger.list_decisions(scenario_id="S2") == []


def test_ledger_records_llm_call_metadata(tmp_path):
    ledger = DecisionLedger(tmp_path / "run.db")
    call_id = ledger.record_llm_call(
        operation="intent",
        model="mock",
        prompt_version="intent-v1",
        request_hash="abc",
        response_hash="def",
        latency_ms=120,
        token_count=512,
        sim_time_s=600,
        scenario_id="S1",
    )
    rows = ledger.list_llm_calls()
    assert len(rows) == 1
    call = rows[0]
    assert isinstance(call, LlmCallRecord)
    assert call.id == call_id
    assert call.operation == "intent"
    assert call.token_count == 512
    assert call.latency_ms == 120
    assert call.error_category == ""


def test_ledger_persists_expert_directives(tmp_path):
    ledger = DecisionLedger(tmp_path / "run.db")
    directive = ExpertDirective(
        directive_id="X1",
        raw_text="keep two boats on T1",
        target_scope=("T1",),
        confidence=0.9,
        status="applied",
    )
    ledger.save_directive(directive, scenario_id="S1")
    listed = ledger.list_directives(scenario_id="S1")
    assert len(listed) == 1
    assert listed[0].directive_id == "X1"
    assert listed[0].raw_text == "keep two boats on T1"
    assert listed[0].status == "applied"
    assert ledger.list_directives(scenario_id="S2") == []


def test_ledger_records_question_runs(tmp_path):
    ledger = DecisionLedger(tmp_path / "run.db")
    ledger.save_question(
        run_id="Q1",
        scenario_id="S1",
        question_text="why U2 on T1?",
        payload={"answer": "geometry", "evidence_ids": ["D1"]},
    )
    rows = ledger.list_questions(scenario_id="S1")
    assert len(rows) == 1
    question = rows[0]
    assert isinstance(question, QuestionRun)
    assert question.run_id == "Q1"
    assert question.payload["answer"] == "geometry"
    assert question.payload["evidence_ids"] == ["D1"]


def test_open_database_uses_wal_foreign_keys_and_migrations(tmp_path):
    conn = open_database(tmp_path / "run.db")
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert _ALL_TABLES <= tables
    finally:
        conn.close()


def test_canonical_json_sorts_keys():
    assert json_dumps({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_create_checkpointer_persists_across_reopen(tmp_path):
    path = tmp_path / "graph.db"
    saver = create_checkpointer(path)
    checkpoint = {
        "v": 1,
        "id": "c1",
        "ts": "2026-08-14T00:00:00Z",
        "channel_values": {"events": []},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    thread = {"configurable": {"thread_id": "S1", "checkpoint_ns": ""}}
    saver.put(thread, checkpoint, {}, {})
    reopened = create_checkpointer(path)
    got = reopened.get_tuple(thread)
    assert got is not None
    assert got.checkpoint["channel_values"] == {"events": []}


def test_create_store_roundtrips(tmp_path):
    store = create_store(tmp_path / "graph.db")
    store.put(("scenario", "S1"), "summary", {"events": [1, 2]})
    item = store.get(("scenario", "S1"), "summary")
    assert item is not None
    assert item.value == {"events": [1, 2]}


def test_factory_connections_set_busy_timeout_and_wal(tmp_path):
    """The checkpointer and store share the WAL write lock with the agent
    repositories, so their connections must carry the same busy timeout and
    WAL mode (SqliteSaver has no retry; a lock collision must wait, not
    crash the graph step)."""
    saver = create_checkpointer(tmp_path / "graph.db")
    assert saver.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert saver.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    store = create_store(tmp_path / "graph.db")
    assert store.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
