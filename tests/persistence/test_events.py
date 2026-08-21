from __future__ import annotations

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
