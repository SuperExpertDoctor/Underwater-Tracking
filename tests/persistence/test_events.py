from __future__ import annotations

import sqlite3

from underwater_tracking.domain.models import EventAudience
from underwater_tracking.persistence.events import EventRepository


def test_append_if_absent_is_idempotent_per_event_id(tmp_path) -> None:
    repository = EventRepository(tmp_path / "events.db")

    first = repository.append_if_absent(
        event_id="periodic_situation_summary:S1:600",
        event_type="periodic_situation_summary",
        scenario_id="S1",
        sim_time_s=600,
        payload={"summary": "first"},
    )
    duplicate = repository.append_if_absent(
        event_id="periodic_situation_summary:S1:600",
        event_type="periodic_situation_summary",
        scenario_id="S1",
        sim_time_s=600,
        payload={"summary": "duplicate"},
    )
    later = repository.append_if_absent(
        event_id="periodic_situation_summary:S1:1200",
        event_type="periodic_situation_summary",
        scenario_id="S1",
        sim_time_s=1200,
        payload={"summary": "later"},
    )

    assert first is not None
    assert duplicate is None
    assert later is not None
    assert [event.event_id for event in repository.list_events(scenario_id="S1")] == [
        "periodic_situation_summary:S1:600",
        "periodic_situation_summary:S1:1200",
    ]

    repository.close()


def test_event_audiences_are_persisted_and_private_decisions_are_not_blue_visible(
    tmp_path,
) -> None:
    repository = EventRepository(tmp_path / "events.db")
    event_id = "target_mission_decision:T1:D1"
    repository.append(
        event_id=event_id,
        event_type="target_mission_decision",
        scenario_id="S1",
        sim_time_s=30,
        payload={"decision_id": "D1", "guidance_waypoint_xy": (1.0, 2.0)},
        target_id="T1",
    )

    stored = repository.get(event_id)
    assert stored is not None
    assert EventAudience.OPERATOR_AUDIT in stored.audiences
    assert EventAudience.MEMORY_SOURCE in stored.audiences
    assert EventAudience.BLUE_PLANNING not in stored.audiences
    repository.close()


def test_legacy_event_table_migrates_audiences_and_private_history(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE runtime_events ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, "
        "event_type TEXT NOT NULL, scenario_id TEXT NOT NULL, target_id TEXT, "
        "sim_time_s INTEGER NOT NULL, severity TEXT NOT NULL DEFAULT 'info', "
        "payload TEXT NOT NULL, created_at INTEGER NOT NULL)"
    )
    connection.execute(
        "INSERT INTO runtime_events VALUES "
        "(1, 'old-public', 'state_changed', 'S1', NULL, 0, 'info', '{}', 0), "
        "(2, 'old-private', 'target_mission_decision', 'S1', 'T1', 0, 'info', '{}', 0)"
    )
    connection.commit()
    connection.close()

    repository = EventRepository(path)
    public = repository.get("old-public")
    private = repository.get("old-private")
    assert public is not None and EventAudience.BLUE_PLANNING in public.audiences
    assert private is not None and EventAudience.BLUE_PLANNING not in private.audiences
    assert EventAudience.OPERATOR_AUDIT in private.audiences
    repository.close()
