from __future__ import annotations

from pathlib import Path

from underwater_tracking.memory.source_reader import MemorySourceReader
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.memory import LongTermMemoryRepository, ShortTermContextRepository
from underwater_tracking.domain.memory_models import ShortTermMessage


def test_source_reader_reads_new_runtime_events_once_and_advances_cursor(tmp_path: Path) -> None:
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
    assert source.source_key == "runtime_event:event-1"
    assert source.source_event_ids == ("event-1",)
    assert source.payload["event_type"] == "bearing"
    assert "ignored" not in source.payload
    assert memory.get_source_cursor("operator", "scenario-1", "runtime_event") == 1
    assert reader.read_new("operator", "scenario-1") == ()


def test_source_reader_uses_conversation_cursor_and_preserves_message_ids(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    memory = LongTermMemoryRepository(database)
    short_term = ShortTermContextRepository(database)
    short_term.append_messages(
        "operator", "conversation-1",
        (
            ShortTermMessage(message_id="message-1", role="user", text="first source"),
            ShortTermMessage(message_id="message-2", role="assistant", text="second source"),
        ),
    )
    reader = MemorySourceReader(memory, short_term_repository=short_term)

    sources = reader.read_conversation("operator", "scenario-1", "conversation-1")

    assert sources[0].source_message_ids == ("message-1", "message-2")
    assert memory.get_source_cursor("operator", "scenario-1", "conversation:conversation-1") == 2
    assert reader.read_conversation("operator", "scenario-1", "conversation-1") == ()
