"""Durable repository tests: events, plans, ledger, and graph persistence.

Covers the brief's atomic-plan stale-commit test verbatim, the commit
transactionality/rollback requirement, restart recovery, canonical JSON,
WAL/foreign-key database setup, and the LangGraph SQLite checkpointer/store
factories. ``valid_plan`` is a function-scoped fixture so every test gets an
independent mutable plan.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

from underwater_tracking.domain.agent_models import (
    DecisionRecord,
    ExpertDirective,
    IntentVerificationCallRef,
    PlanCommand,
    TrackingPlan,
    TrajectoryDiffGateState,
    TrajectoryDiffResult,
    Waypoint,
)
from underwater_tracking.persistence.checkpoints import create_checkpointer, create_store
from underwater_tracking.persistence.events import EventRepository, StoredEvent
from underwater_tracking.persistence.ledger import DecisionLedger, LlmCallRecord, QuestionRun
from underwater_tracking.persistence.plans import PlanRepository, StaleSnapshotError
from underwater_tracking.persistence.sqlite import (
    SCHEMA_VERSION,
    database_write_lock,
    json_dumps,
    open_database,
    transaction,
)
from underwater_tracking.world_model.models import (
    DataStatus,
    HorizonCoverage,
    HorizonName,
    WorldModelForecast,
)

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


def test_ledger_serializes_concurrent_llm_metadata_writes(tmp_path):
    ledger = DecisionLedger(tmp_path / "run.db")
    barrier = Barrier(3)

    def write_call(index: int) -> int:
        barrier.wait()
        return ledger.record_llm_call(
            operation=f"regional_strategy:{index}",
            model="master-model",
            prompt_version="regional-v1",
            request_hash=str(index),
            response_hash=str(index),
            scenario_id="S1",
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(write_call, index) for index in range(3)]
        call_ids = [future.result() for future in futures]

    assert len(set(call_ids)) == 3
    assert len(ledger.list_llm_calls(scenario_id="S1")) == 3


def test_shared_plan_repository_serializes_read_and_write_calls(tmp_path):
    """The graph and publisher may use one PlanRepository concurrently."""
    repository = PlanRepository(tmp_path / "run.db")
    errors: list[BaseException] = []
    barrier = Barrier(3)

    def writer() -> None:
        barrier.wait()
        try:
            for revision in range(50):
                repository.set_snapshot_revision("S1", revision)
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)

    def reader() -> None:
        barrier.wait()
        try:
            for _ in range(50):
                repository.get_snapshot_revision("S1")
                repository.get_active("S1")
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(writer), executor.submit(reader)]
        barrier.wait()
        for future in futures:
            future.result()

    assert errors == []


def test_ledger_projects_latest_activity_by_brain_role(tmp_path):
    ledger = DecisionLedger(tmp_path / "run.db")
    ledger.record_llm_call(
        operation="intent",
        model="master-model",
        prompt_version="intent-v1",
        sim_time_s=10,
        scenario_id="S1",
    )
    ledger.record_llm_call(
        operation="commit",
        model="master-model",
        prompt_version="commit-v1",
        sim_time_s=20,
        scenario_id="S1",
    )
    ledger.record_llm_call(
        operation="slave_sonar_decision",
        model="slave-model",
        prompt_version="slave-v1",
        sim_time_s=30,
        scenario_id="S1",
    )
    ledger.record_llm_call(
        operation="adversary_escape",
        model="adversary-model",
        prompt_version="adversary-v1",
        error_category="timeout",
        sim_time_s=40,
        scenario_id="S1",
    )

    activity = ledger.latest_role_activity("S1")
    assert activity["master"].operation == "commit"
    assert activity["master"].status == "succeeded"
    assert activity["slave"].operation == "slave_sonar_decision"
    assert activity["slave"].status == "succeeded"
    assert activity["adversary"].operation == "adversary_escape"
    assert activity["adversary"].status == "degraded"

    ledger.record(
        DecisionRecord(
            decision_id="D-adversary",
            scenario_id="S1",
            sim_time_s=50,
            snapshot_revision=1,
            model_version="adversary-v2",
            input_evidence_ids=("uuv_04", "carrier_01", "target_00"),
            final_plan_id="P-adversary",
        )
    )
    updated = ledger.latest_role_activity("S1")
    assert updated["adversary"].status == "succeeded"
    assert updated["adversary"].evidence_platform_ids == (
        "carrier_01",
        "uuv_04",
    )


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
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 60000
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


def test_create_checkpointer_prunes_old_checkpoints_and_writes(tmp_path):
    path = tmp_path / "bounded-graph.db"
    saver = create_checkpointer(path, max_checkpoints=2)
    thread = {"configurable": {"thread_id": "S1", "checkpoint_ns": ""}}

    for index in range(5):
        checkpoint = {
            "v": 1,
            "id": f"c{index}",
            "ts": f"2026-08-14T00:00:0{index}Z",
            "channel_values": {"step": index},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        saved = saver.put(thread, checkpoint, {}, {})
        saver.put_writes(saved, [("events", index)], f"task-{index}")

    retained = saver.conn.execute(
        "SELECT checkpoint_id FROM checkpoints ORDER BY checkpoint_id"
    ).fetchall()
    writes = saver.conn.execute("SELECT DISTINCT checkpoint_id FROM writes").fetchall()
    assert [row[0] for row in retained] == ["c3", "c4"]
    assert [row[0] for row in writes] == ["c3", "c4"]


def test_prediction_diff_gate_types_survive_sqlite_checkpoint_reopen(tmp_path):
    path = tmp_path / "prediction-diff.db"
    call = IntentVerificationCallRef(
        model="LongCat-Flash-Chat",
        prompt_version="intent-v2",
        request_hash="request-1",
        response_hash="response-1",
        sim_time_s=60,
        scenario_id="S1",
    )
    gate = TrajectoryDiffGateState(
        target_id="T1",
        consecutive_count=2,
        latched=True,
        verification_pending=True,
        suspicion_diff_id="D1",
        latest_diff_id="D2",
        intent_verification_calls=(call,),
    )
    diff = TrajectoryDiffResult(
        diff_id="D2",
        target_id="T1",
        current_prediction_id="P2",
        current_sim_time_s=60,
        status="comparable",
        normalized_threshold=2.45,
        absolute_floor_m=250,
        reset_normalized_threshold=1.75,
        reset_absolute_floor_m=150,
        threshold_schema_version="trajectory-diff-v1",
        confirmation_cycles=2,
    )
    checkpoint = {
        "v": 1,
        "id": "prediction-diff-c1",
        "ts": "2026-08-24T00:00:00Z",
        "channel_values": {"gate": gate, "diff": diff},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    thread = {"configurable": {"thread_id": "S1", "checkpoint_ns": ""}}
    create_checkpointer(path).put(thread, checkpoint, {}, {})

    restored = create_checkpointer(path).get_tuple(thread)

    assert restored is not None
    assert restored.checkpoint["channel_values"]["gate"] == gate
    assert isinstance(
        restored.checkpoint["channel_values"]["gate"].intent_verification_calls[0],
        IntentVerificationCallRef,
    )
    assert restored.checkpoint["channel_values"]["diff"] == diff


def test_world_model_forecast_survives_sqlite_checkpoint_reopen(tmp_path):
    path = tmp_path / "world-model.db"
    forecast = WorldModelForecast(
        scenario_id="S1",
        target_id="T1",
        as_of_s=60,
        source_prediction_id="prediction-1",
        data_status=DataStatus.READY,
        trajectory_fallback_used=False,
        imm_model_probabilities={"cv": 1.0},
        horizons=(
            HorizonCoverage(name=HorizonName.H1, start_offset_s=0, end_offset_s=120, sample_count=2, covered=True),
            HorizonCoverage(name=HorizonName.H2, start_offset_s=120, end_offset_s=300, sample_count=2, covered=True),
            HorizonCoverage(name=HorizonName.H3, start_offset_s=300, end_offset_s=900, sample_count=2, covered=True),
            HorizonCoverage(name=HorizonName.H4, start_offset_s=900, end_offset_s=1800, sample_count=2, covered=True),
        ),
        events=(),
    )
    checkpoint = {
        "v": 1,
        "id": "world-model-c1",
        "ts": "2026-08-26T00:00:00Z",
        "channel_values": {"world_model_forecasts": {"T1": forecast}},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    thread = {"configurable": {"thread_id": "S1", "checkpoint_ns": ""}}
    create_checkpointer(path).put(thread, checkpoint, {}, {})

    restored = create_checkpointer(path).get_tuple(thread)

    assert restored is not None
    restored_forecast = restored.checkpoint["channel_values"]["world_model_forecasts"]["T1"]
    assert restored_forecast == forecast
    assert isinstance(restored_forecast, WorldModelForecast)
    assert restored_forecast.data_status is DataStatus.READY


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
    assert saver.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 60000
    assert saver.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    store = create_store(tmp_path / "graph.db")
    assert store.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 60000
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_factory_connections_share_the_repository_write_lock(tmp_path):
    """Concurrent bootstrap writers must serialize before touching SQLite."""
    path = tmp_path / "shared.db"
    repository_conn = open_database(path)
    saver = create_checkpointer(path)
    lock = database_write_lock(repository_conn)
    assert lock is database_write_lock(saver.conn)

    entered = Event()
    release = Event()

    def hold_repository_transaction() -> None:
        with transaction(repository_conn):
            entered.set()
            assert release.wait(timeout=2.0)

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
    with ThreadPoolExecutor(max_workers=2) as executor:
        holder = executor.submit(hold_repository_transaction)
        assert entered.wait(timeout=2.0)
        writer = executor.submit(saver.put, thread, checkpoint, {}, {})
        assert not writer.done()
        release.set()
        holder.result(timeout=2.0)
        writer.result(timeout=2.0)

    repository_conn.close()
    saver.conn.close()
