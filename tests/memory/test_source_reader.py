from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import pytest

from underwater_tracking.domain.agent_models import DecisionRecord, StrategyProposal
from underwater_tracking.domain.memory_models import MemoryWorkPayload
from underwater_tracking.memory.source_reader import MemorySourceReader
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.memory import LongTermMemoryRepository, ShortTermContextRepository
from underwater_tracking.domain.memory_models import ShortTermMessage


class _ScenarioRepository:
    def __init__(self, prefix: str, count: int) -> None:
        self._scenario_ids = tuple(f"{prefix}-{index:02d}" for index in range(count))

    def list_scenario_ids(self, limit: int = 100, *, offset: int = 0) -> tuple[str, ...]:
        return self._scenario_ids[offset : offset + limit]


def test_source_reader_does_not_advance_cursor_before_work_is_enqueued(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    events = EventRepository(database)
    memory = LongTermMemoryRepository(database)
    events.append(
        event_id="event-1",
        event_type="bearing",
        scenario_id="scenario-1",
        sim_time_s=10,
        payload={"summary": "contact bearing changed", "target_id": "target-1", "ignored": "x"},
    )
    reader = MemorySourceReader(memory, event_repository=events)

    sources = reader.read_new("operator", "scenario-1")

    assert len(sources) == 1
    source = sources[0]
    assert source.source_key == "runtime_event:scenario-1:event-1"
    assert source.source_event_ids == ("event-1",)
    assert source.payload["event_type"] == "bearing"
    assert "ignored" not in source.payload
    assert source.cursor == 1
    assert memory.get_source_cursor("operator", "scenario-1", "runtime_event") == 0
    assert reader.read_new("operator", "scenario-1") == sources


def test_source_reader_projects_periodic_summary_text_and_event_provenance(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    events = EventRepository(database)
    memory = LongTermMemoryRepository(database)
    events.append(
        event_id="periodic_situation_summary:scenario-1:600",
        event_type="periodic_situation_summary",
        scenario_id="scenario-1",
        sim_time_s=600,
        payload={
            "summary": "time=600; plan=4; regions=R1:ACTIVE_SCAN:0.80",
            "source_event_ids": ["bearing-1", "bearing-2"],
            "uuv_counts": {"total": 2},
        },
    )
    reader = MemorySourceReader(memory, event_repository=events)

    source = reader.read_new("operator", "scenario-1")[0]

    assert source.text == "time=600; plan=4; regions=R1:ACTIVE_SCAN:0.80"
    assert source.payload["summary"] == source.text
    assert source.source_event_ids == ("periodic_situation_summary:scenario-1:600",)
    assert "uuv_counts" not in source.payload


def test_source_reader_retains_prediction_diff_audit_ids_once(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    events = EventRepository(database)
    memory = LongTermMemoryRepository(database)
    events.append(
        event_id="suspected-1",
        event_type="target_intent_change_suspected",
        scenario_id="scenario-1",
        sim_time_s=60,
        target_id="T1",
        severity="tactical",
        payload={
            "diff_id": "D1",
            "previous_prediction_id": "P1",
            "current_prediction_id": "P2",
            "observation_ids": ("O1", "O2"),
            "absolute_rms_m": 300.0,
            "normalized_rms": 3.0,
            "absolute_floor_m": 250.0,
            "normalized_threshold": 2.45,
            "consecutive_count": 2,
            "source": "trajectory_diff",
            "raw_prompt": "must never enter memory",
        },
    )
    events.append(
        event_id="confirmed-1",
        event_type="target_intent_changed",
        scenario_id="scenario-1",
        sim_time_s=90,
        target_id="T1",
        severity="strategic",
        payload={
            "diff_id": "D1",
            "suspicion_event_id": "suspected-1",
            "observation_ids": ("O2",),
            "evidence_ids": ("D1", "O2", "suspected-1"),
            "previous_label": "transit",
            "label": "evade",
            "confidence": 0.85,
            "llm_operation": "intent",
            "llm_model": "real-intent-model",
            "llm_prompt_version": "intent-v2",
            "llm_request_hash": "request-hash",
            "llm_response_hash": "response-hash",
            "source": "real_intent_llm",
            "raw_prompt": "must never enter memory",
        },
    )
    reader = MemorySourceReader(memory, event_repository=events)

    sources = reader.read_new("operator", "scenario-1")

    assert [source.source_event_ids for source in sources] == [
        ("suspected-1",),
        ("confirmed-1",),
    ]
    assert sources[0].payload["diff_id"] == "D1"
    assert sources[0].payload["previous_prediction_id"] == "P1"
    assert sources[0].payload["current_prediction_id"] == "P2"
    assert sources[0].payload["observation_ids"] == ("O1", "O2")
    assert "raw_prompt" not in sources[0].payload
    assert sources[1].payload["suspicion_event_id"] == "suspected-1"
    assert sources[1].payload["llm_request_hash"] == "request-hash"
    assert sources[1].payload["llm_response_hash"] == "response-hash"
    assert "raw_prompt" not in sources[1].payload
    assert reader.read_new("operator", "scenario-1") == sources


def test_source_reader_uses_conversation_cursor_and_preserves_message_ids(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    memory = LongTermMemoryRepository(database)
    short_term = ShortTermContextRepository(database)
    short_term.append_messages(
        "operator", "conversation-1",
        (
            ShortTermMessage(message_id="message-1", scenario_id="scenario-1", role="user", text="first source"),
            ShortTermMessage(message_id="message-2", scenario_id="scenario-1", role="assistant", text="second source"),
        ),
        scenario_id="scenario-1",
    )
    reader = MemorySourceReader(memory, short_term_repository=short_term)

    sources = reader.read_conversation("operator", "scenario-1", "conversation-1")

    assert sources[0].source_message_ids == ("message-1", "message-2")
    assert sources[0].cursor == 2
    assert memory.get_source_cursor("operator", "scenario-1", "conversation:conversation-1") == 0
    assert reader.read_conversation("operator", "scenario-1", "conversation-1") == sources


def test_source_reader_does_not_share_conversation_messages_or_cursors_between_scenarios(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    memory = LongTermMemoryRepository(database)
    short_term = ShortTermContextRepository(database)
    for scenario_id, message_id in (("scenario-a", "message-a"), ("scenario-b", "message-b")):
        short_term.append_messages(
            "operator",
            "conversation-1",
            (ShortTermMessage(message_id=message_id, scenario_id=scenario_id, role="user", text=scenario_id),),
            scenario_id=scenario_id,
        )
    reader = MemorySourceReader(memory, short_term_repository=short_term)

    first = reader.read_conversation("operator", "scenario-a", "conversation-1")[0]
    second = reader.read_conversation("operator", "scenario-b", "conversation-1")[0]

    assert first.source_message_ids == ("message-a",)
    assert second.source_message_ids == ("message-b",)
    assert first.source_key != second.source_key
    assert first.source_cursor_type != second.source_cursor_type


def test_load_work_sources_reads_only_the_named_messages_after_new_messages_arrive(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    memory = LongTermMemoryRepository(database)
    short_term = ShortTermContextRepository(database)
    short_term.append_messages(
        "operator",
        "conversation-1",
        (
            ShortTermMessage(message_id="message-1", scenario_id="scenario-1", role="user", text="first source"),
            ShortTermMessage(message_id="message-2", scenario_id="scenario-1", role="assistant", text="second source"),
        ),
        scenario_id="scenario-1",
    )
    reader = MemorySourceReader(memory, short_term_repository=short_term)

    payload = MemoryWorkPayload(source_message_ids=("message-1",))
    short_term.append_messages(
        "operator",
        "conversation-1",
        (ShortTermMessage(message_id="message-3", scenario_id="scenario-1", role="user", text="new unrelated source"),),
        scenario_id="scenario-1",
    )

    sources = reader.load_work_sources(
        "operator", "scenario-1", payload, conversation_id="conversation-1"
    )

    assert len(sources) == 1
    assert sources[0].source_message_ids == ("message-1",)
    assert sources[0].text == "first source"
    assert "new unrelated source" not in sources[0].text


def test_source_reader_never_loads_a_wrong_scenario_message_from_corrupt_context(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    memory = LongTermMemoryRepository(database)
    short_term = ShortTermContextRepository(database)
    short_term.append_messages(
        "operator",
        "conversation-1",
        (ShortTermMessage(message_id="valid", scenario_id="scenario-a", role="user", text="valid"),),
        scenario_id="scenario-a",
    )
    short_term._conn.execute(
        "UPDATE short_term_contexts SET recent_messages = ?"
        " WHERE user_id = ? AND scenario_id = ? AND conversation_id = ?",
        (
            json.dumps(
                [
                    {"message_id": "valid", "scenario_id": "scenario-a", "role": "user", "text": "valid"},
                    {"message_id": "wrong", "scenario_id": "scenario-b", "role": "user", "text": "wrong"},
                ]
            ),
            "operator",
            "scenario-a",
            "conversation-1",
        ),
    )
    reader = MemorySourceReader(memory, short_term_repository=short_term)

    assert [message.message_id for message in short_term.get_messages(
        "operator", "conversation-1", ("valid", "wrong"), scenario_id="scenario-a"
    )] == ["valid"]
    assert reader.read_conversation("operator", "scenario-a", "conversation-1")[0].source_message_ids == ("valid",)
    with pytest.raises(ValueError, match="source_message_ids"):
        reader.load_work_sources(
            "operator",
            "scenario-a",
            MemoryWorkPayload(source_message_ids=("valid", "wrong")),
            conversation_id="conversation-1",
        )


def test_load_work_sources_rejects_missing_scenario_scope(tmp_path: Path) -> None:
    memory = LongTermMemoryRepository(tmp_path / "memory.db")
    reader = MemorySourceReader(memory)

    with pytest.raises(ValueError, match="scenario_id"):
        reader.load_work_sources("operator", None, MemoryWorkPayload())


def test_decision_source_text_keeps_traceable_fields_and_bounds_large_payload(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    memory = LongTermMemoryRepository(database)
    decisions = DecisionLedger(database)
    decisions.record(
        DecisionRecord(
            decision_id="decision-1",
            scenario_id="scenario-1",
            sim_time_s=12,
            candidates=(
                StrategyProposal(
                    concept="hold_current",
                    target_priorities={"target-1": 0.9},
                    required_quality={"target-1": 0.7},
                    reinforcement_policy={"max_additional_groups": "1"},
                    releasable_soft_constraints=("constraint-1",),
                    evidence_ids=("event-1",),
                    rationale="candidate rationale",
                ),
            ),
            candidate_plan_ids=("plan-1",),
            rejected_candidates={"candidate-2": "unsafe"},
            final_plan_id="plan-1",
        )
    )
    reader = MemorySourceReader(memory, decision_ledger=decisions, batch_limit=4)

    source = reader.load_work_sources(
        "operator",
        "scenario-1",
        MemoryWorkPayload(source_decision_ids=("decision-1",)),
    )[0]

    assert len(source.text.encode("utf-8")) <= 4000
    assert '"candidates"' in source.text
    assert "hold_current" in source.text
    assert '"candidate_plan_ids"' in source.text
    assert '"rejected_candidates"' in source.text
    assert '"final_plan_id"' in source.text
    assert source.text != "{}"


def test_source_reader_discovers_existing_scenarios_without_work_and_stays_bounded(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "memory.db"
    memory = LongTermMemoryRepository(database)
    events = EventRepository(database)
    events.append(
        event_id="event-1",
        event_type="bearing",
        scenario_id="scenario-1",
        sim_time_s=10,
        payload={"summary": "existing event"},
    )
    reader = MemorySourceReader(memory, event_repository=events, batch_limit=2)

    assert reader.discover_scopes("operator") == (("operator", "scenario-1"),)
    assert memory.list_source_scopes(limit=2) == (("operator", "scenario-1"),)

    def fail_unbounded(*args, **kwargs):
        del args, kwargs
        raise sqlite3.OperationalError("database is busy")

    monkeypatch.setattr(events, "list_scenario_ids", fail_unbounded)
    with pytest.raises(sqlite3.OperationalError, match="database is busy"):
        reader.discover_scopes("operator")


def test_source_reader_discovers_beyond_first_page_with_persistent_continuation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    memory = LongTermMemoryRepository(database)
    events = EventRepository(database)
    for index in range(33):
        events.append(
            event_id=f"event-{index}",
            event_type="bearing",
            scenario_id=f"scenario-{index:02d}",
            sim_time_s=index,
            payload={"summary": f"scenario {index}"},
        )
    reader = MemorySourceReader(memory, event_repository=events, batch_limit=32)

    first = reader.discover_scopes("operator")
    second = reader.discover_scopes("operator")

    assert len(first) == 32
    assert second == (("operator", "scenario-32"),)
    assert memory.get_source_cursor(
        "operator", "__memory_scope_discovery__", "__scope_discovery__:runtime_event"
    ) == 33


def test_source_reader_round_robins_all_repositories_with_one_bounded_continuation(
    tmp_path: Path,
) -> None:
    memory = LongTermMemoryRepository(tmp_path / "memory.db")
    reader = MemorySourceReader(
        memory,
        event_repository=_ScenarioRepository("event", 4),
        decision_ledger=_ScenarioRepository("decision", 4),
        plan_repository=_ScenarioRepository("plan", 4),
        batch_limit=3,
    )

    pages = [reader.discover_scopes("operator") for _ in range(4)]

    assert pages == [
        (("operator", "event-00"), ("operator", "decision-00"), ("operator", "plan-00")),
        (("operator", "event-01"), ("operator", "decision-01"), ("operator", "plan-01")),
        (("operator", "event-02"), ("operator", "decision-02"), ("operator", "plan-02")),
        (("operator", "event-03"), ("operator", "decision-03"), ("operator", "plan-03")),
    ]
    assert all(len(page) <= 3 for page in pages)


def test_source_reader_does_not_advance_discovery_when_scope_registration_fails(
    tmp_path: Path, monkeypatch
) -> None:
    memory = LongTermMemoryRepository(tmp_path / "memory.db")
    reader = MemorySourceReader(
        memory,
        event_repository=_ScenarioRepository("event", 1),
        batch_limit=1,
    )
    initial = memory.get_source_discovery_state("operator", 1)
    monkeypatch.setattr(
        memory,
        "_register_source_scope",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("registration failed")),
    )

    with pytest.raises(RuntimeError, match="registration failed"):
        reader.discover_scopes("operator")

    assert memory.get_source_discovery_state("operator", 1) == initial
    assert memory.list_source_scopes() == ()


def test_read_conversation_uses_absolute_cursor_after_rolling_window_eviction(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    memory = LongTermMemoryRepository(database)
    short_term = ShortTermContextRepository(database)
    short_term.append_messages(
        "operator",
        "conversation-1",
        tuple(
            ShortTermMessage(message_id=f"message-{index}", scenario_id="scenario-1", role="user", text=f"source {index}")
            for index in range(130)
        ),
        scenario_id="scenario-1",
    )
    reader = MemorySourceReader(memory, short_term_repository=short_term, batch_limit=32)

    memory.set_source_cursor(
        "operator", "scenario-1", "conversation:scenario-1:conversation-1", 2
    )
    first = reader.read_conversation("operator", "scenario-1", "conversation-1")[0]
    memory.advance_source_cursor(
        "operator", "scenario-1", "conversation:scenario-1:conversation-1", first.cursor
    )
    second = reader.read_conversation("operator", "scenario-1", "conversation-1")[0]

    assert first.source_message_ids == tuple(f"message-{index}" for index in range(2, 34))
    assert first.cursor == 34
    assert second.source_message_ids == tuple(f"message-{index}" for index in range(34, 66))
    assert second.cursor == 66


def test_read_conversation_reads_immutable_messages_after_window_compression(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    memory = LongTermMemoryRepository(database)
    short_term = ShortTermContextRepository(database)
    short_term.append_messages(
        "operator",
        "conversation-1",
        tuple(
            ShortTermMessage(
                message_id=f"message-{index}",
                scenario_id="scenario-1",
                role="user",
                text=f"source {index}",
            )
            for index in range(130)
        ),
        scenario_id="scenario-1",
    )
    memory.set_source_cursor(
        "operator", "scenario-1", "conversation:scenario-1:conversation-1", 0
    )
    reader = MemorySourceReader(memory, short_term_repository=short_term, batch_limit=32)

    source = reader.read_conversation("operator", "scenario-1", "conversation-1")[0]
    assert source.source_message_ids == tuple(f"message-{index}" for index in range(0, 32))
