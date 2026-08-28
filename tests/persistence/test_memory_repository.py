"""SQLite persistence contracts for the asynchronous memory pipeline."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from threading import Event, Thread

import pytest

import underwater_tracking.persistence.sqlite as sqlite_module

from underwater_tracking.domain.memory_models import (
    MemoryStreamEvent,
    MemoryStreamEventType,
    MemoryStreamStatus,
    MemoryType,
    MemoryVersion,
    MemoryWorkItem,
    MemoryWorkStatus,
    MemoryWorkType,
    ShortTermMessage,
)
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.memory import (
    LongTermMemoryRepository,
    ShortTermContextRepository,
    VersionConflictError,
)
from underwater_tracking.persistence.sqlite import SCHEMA_VERSION, open_database


def _memory(
    memory_id: str,
    *,
    family_id: str = "family-1",
    version: int = 1,
    user_id: str = "operator",
    supersedes_memory_id: str | None = None,
    importance: float = 0.7,
) -> MemoryVersion:
    return MemoryVersion(
        memory_id=memory_id,
        memory_family_id=family_id,
        version=version,
        user_id=user_id,
        memory_type=MemoryType.SEMANTIC,
        summary=f"summary {memory_id}",
        importance_score=importance,
        embedding=(0.1, 0.2),
        supersedes_memory_id=supersedes_memory_id,
        source_event_ids=("event-1",),
        change_reason="created" if version == 1 else "updated",
    )


def _work(work_id: str, *, user_id: str = "operator") -> MemoryWorkItem:
    return MemoryWorkItem(
        work_id=work_id,
        user_id=user_id,
        conversation_id="conversation-1",
        scenario_id="scenario-1",
        work_type=MemoryWorkType.CONVERSATION_TURN,
    )


def _stream(event_id: str, *, user_id: str = "operator", conversation_id: str = "conversation-1") -> MemoryStreamEvent:
    return MemoryStreamEvent(
        cursor=0,
        event_id=event_id,
        user_id=user_id,
        conversation_id=conversation_id,
        status=MemoryStreamStatus.COMPLETED,
        type=MemoryStreamEventType.MEMORY_EXTRACTED,
    )


def test_new_database_creates_memory_tables(tmp_path):
    conn = open_database(tmp_path / "memory.db")
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {
            "short_term_contexts",
            "long_term_memories",
            "memory_work_items",
                "memory_stream_events",
                "memory_source_cursors",
                "memory_source_discovery",
                "short_term_messages",
            } <= tables
        memory_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(long_term_memories)")
        }
        assert "source_knowledge_ids" not in memory_columns
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        conn.close()


def test_v3_database_migrates_without_losing_existing_rows(tmp_path):
    path = tmp_path / "v3.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE runtime_events (
            id INTEGER PRIMARY KEY, event_id TEXT UNIQUE, event_type TEXT NOT NULL,
            scenario_id TEXT NOT NULL, target_id TEXT, sim_time_s INTEGER NOT NULL,
            severity TEXT NOT NULL, payload TEXT NOT NULL, created_at INTEGER NOT NULL
        );
        CREATE TABLE plans (
            plan_id TEXT PRIMARY KEY, scenario_id TEXT NOT NULL, revision INTEGER NOT NULL,
            base_snapshot_revision INTEGER NOT NULL, status TEXT NOT NULL,
            valid_from_s INTEGER NOT NULL, valid_until_s INTEGER NOT NULL,
            payload TEXT NOT NULL, created_at INTEGER NOT NULL
        );
        CREATE TABLE llm_calls (
            id INTEGER PRIMARY KEY, operation TEXT NOT NULL, model TEXT NOT NULL,
            prompt_version TEXT NOT NULL, request_hash TEXT NOT NULL, response_hash TEXT NOT NULL,
            latency_ms INTEGER NOT NULL, token_count INTEGER NOT NULL, error_category TEXT NOT NULL,
            sim_time_s INTEGER NOT NULL, scenario_id TEXT NOT NULL, created_at INTEGER NOT NULL
        );
        INSERT INTO runtime_events VALUES (1, 'event-1', 'bearing', 'scenario-1', NULL, 1, 'info', '{}', 1);
        INSERT INTO plans VALUES ('plan-1', 'scenario-1', 1, 0, 'active', 0, 1, '{}', 1);
        INSERT INTO llm_calls VALUES (1, 'memory_filter', 'model', 'v1', '', '', 0, 0, '', 0, 'scenario-1', 1);
        PRAGMA user_version = 3;
        """
    )
    conn.close()

    migrated = open_database(path)
    try:
        assert migrated.execute("SELECT event_id FROM runtime_events").fetchone()[0] == "event-1"
        assert migrated.execute("SELECT plan_id FROM plans").fetchone()[0] == "plan-1"
        assert migrated.execute("SELECT operation FROM llm_calls").fetchone()[0] == "memory_filter"
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        migrated.close()
    reopened = open_database(path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        reopened.close()


def test_v4_database_migrates_v5_columns_without_losing_memory_and_context_rows(tmp_path):
    path = tmp_path / "v4.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE short_term_contexts (
            user_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
            summary_text TEXT NOT NULL DEFAULT '', summary_version INTEGER NOT NULL DEFAULT 0,
            recent_messages TEXT NOT NULL DEFAULT '[]', message_count INTEGER NOT NULL DEFAULT 0,
            estimated_tokens INTEGER NOT NULL DEFAULT 0, compression_count INTEGER NOT NULL DEFAULT 0,
            last_compressed_at INTEGER, compression_status TEXT NOT NULL DEFAULT 'pending',
            updated_at INTEGER NOT NULL, PRIMARY KEY (user_id, conversation_id)
        );
        CREATE TABLE long_term_memories (
            memory_id TEXT PRIMARY KEY, memory_family_id TEXT NOT NULL, version INTEGER NOT NULL,
            user_id TEXT NOT NULL, memory_type TEXT NOT NULL, summary TEXT NOT NULL,
            importance_score REAL NOT NULL, embedding TEXT NOT NULL, embedding_version TEXT NOT NULL,
            status TEXT NOT NULL, supersedes_memory_id TEXT, source_message_ids TEXT NOT NULL DEFAULT '[]',
            source_event_ids TEXT NOT NULL DEFAULT '[]', source_decision_ids TEXT NOT NULL DEFAULT '[]',
            source_knowledge_ids TEXT NOT NULL DEFAULT '[]', change_reason TEXT NOT NULL,
            created_at INTEGER NOT NULL, last_accessed_at INTEGER, access_count INTEGER NOT NULL DEFAULT 0,
            sim_time_s REAL, UNIQUE (user_id, memory_family_id, version)
        );
        INSERT INTO short_term_contexts(user_id, conversation_id, recent_messages, updated_at)
        VALUES ('operator', 'conversation-1', '[]', 1);
        INSERT INTO long_term_memories(
            memory_id, memory_family_id, version, user_id, memory_type, summary,
            importance_score, embedding, embedding_version, status, change_reason, created_at
        ) VALUES ('memory-v4', 'family-v4', 1, 'operator', 'semantic', 'kept',
                  0.7, '[0.1]', 'v1', 'active', 'created', 1);
        PRAGMA user_version = 4;
        """
    )
    conn.close()

    migrated = open_database(path)
    try:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert migrated.execute(
            "SELECT last_compression_work_id FROM short_term_contexts"
        ).fetchone()[0] is None
        assert migrated.execute(
            "SELECT importance_baseline, memory_work_id FROM long_term_memories"
        ).fetchone()[0] == 0.7
        assert migrated.execute("SELECT summary FROM long_term_memories").fetchone()[0] == "kept"
    finally:
        migrated.close()


def test_v5_database_rebuilds_scenario_scoped_memory_constraints(tmp_path):
    path = tmp_path / "v5.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE short_term_contexts (
            user_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
            summary_text TEXT NOT NULL DEFAULT '', summary_version INTEGER NOT NULL DEFAULT 0,
            recent_messages TEXT NOT NULL DEFAULT '[]', message_count INTEGER NOT NULL DEFAULT 0,
            estimated_tokens INTEGER NOT NULL DEFAULT 0, compression_count INTEGER NOT NULL DEFAULT 0,
            last_compressed_at INTEGER, compression_status TEXT NOT NULL DEFAULT 'pending',
            last_compression_work_id TEXT, updated_at INTEGER NOT NULL,
            PRIMARY KEY (user_id, conversation_id)
        );
        CREATE TABLE long_term_memories (
            memory_id TEXT PRIMARY KEY, memory_work_id TEXT, memory_family_id TEXT NOT NULL,
            version INTEGER NOT NULL, user_id TEXT NOT NULL, memory_type TEXT NOT NULL,
            summary TEXT NOT NULL, importance_score REAL NOT NULL, importance_baseline REAL NOT NULL,
            embedding TEXT NOT NULL, embedding_version TEXT NOT NULL, status TEXT NOT NULL,
            supersedes_memory_id TEXT, source_message_ids TEXT NOT NULL DEFAULT '[]',
            source_event_ids TEXT NOT NULL DEFAULT '[]', source_decision_ids TEXT NOT NULL DEFAULT '[]',
            source_knowledge_ids TEXT NOT NULL DEFAULT '[]', source_plan_ids TEXT NOT NULL DEFAULT '[]',
            change_reason TEXT NOT NULL, created_at INTEGER NOT NULL, last_accessed_at INTEGER,
            access_count INTEGER NOT NULL DEFAULT 0, sim_time_s REAL,
            UNIQUE (user_id, memory_family_id, version)
        );
        INSERT INTO long_term_memories(
            memory_id, memory_family_id, version, user_id, memory_type, summary,
            importance_score, importance_baseline, embedding, embedding_version, status,
            source_knowledge_ids, change_reason, created_at
        ) VALUES ('legacy-memory', 'family-legacy', 1, 'operator', 'semantic', 'legacy',
                  0.7, 0.7, '[0.1]', 'v1', 'active', '["query-legacy"]', 'created', 1);
        PRAGMA user_version = 5;
        """
    )
    conn.close()

    migrated = open_database(path)
    try:
        columns = {row[1] for row in migrated.execute("PRAGMA table_info(long_term_memories)")}
        assert "scenario_id" in columns
        assert "source_knowledge_ids" in columns
        assert migrated.execute(
            "SELECT source_knowledge_ids FROM long_term_memories"
            " WHERE memory_id = 'legacy-memory'"
        ).fetchone()[0] == '["query-legacy"]'
        assert migrated.execute(
            "SELECT scenario_id FROM long_term_memories WHERE memory_id = 'legacy-memory'"
        ).fetchone()[0] == "__legacy__"
        indexes = {
            row[1]
            for row in migrated.execute("PRAGMA index_list(long_term_memories)")
        }
        assert "idx_long_term_memories_one_active_family" in indexes
        assert tuple(
            row[2]
            for row in migrated.execute(
                "PRAGMA index_info(idx_long_term_memories_one_active_family)"
            )
        ) == ("user_id", "memory_family_id", "scenario_id")
        assert migrated.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = 'idx_long_term_memories_one_active_family'"
        ).fetchone()[0] == 1
        assert migrated.execute(
            "SELECT COUNT(*) FROM long_term_memories WHERE scenario_id = 'scenario-a'"
        ).fetchone()[0] == 0
    finally:
        migrated.close()


def test_partial_v8_database_migrates_each_existing_memory_table(tmp_path):
    path = tmp_path / "partial-v8.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE short_term_contexts (
            user_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
            summary_text TEXT NOT NULL DEFAULT '', summary_version INTEGER NOT NULL DEFAULT 0,
            recent_messages TEXT NOT NULL DEFAULT '[]', message_count INTEGER NOT NULL DEFAULT 0,
            estimated_tokens INTEGER NOT NULL DEFAULT 0, compression_count INTEGER NOT NULL DEFAULT 0,
            last_compressed_at INTEGER, compression_status TEXT NOT NULL DEFAULT 'pending',
            last_compression_work_id TEXT, updated_at INTEGER NOT NULL,
            PRIMARY KEY (user_id, conversation_id)
        );
        INSERT INTO short_term_contexts(user_id, conversation_id, updated_at)
        VALUES ('operator', 'conversation-1', 1);
        PRAGMA user_version = 8;
        """
    )
    conn.close()

    migrated = open_database(path)
    try:
        assert migrated.execute(
            "SELECT scenario_id FROM short_term_contexts"
        ).fetchone()[0] == "__legacy__"
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        migrated.close()


def test_partial_v8_database_repairs_missing_columns_without_legacy_tables(tmp_path):
    path = tmp_path / "partial-v8-missing-columns.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE short_term_contexts (
            user_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
            summary_text TEXT NOT NULL DEFAULT '', summary_version INTEGER NOT NULL DEFAULT 0,
            recent_messages TEXT NOT NULL DEFAULT '[]', message_count INTEGER NOT NULL DEFAULT 0,
            estimated_tokens INTEGER NOT NULL DEFAULT 0, compression_count INTEGER NOT NULL DEFAULT 0,
            last_compressed_at INTEGER, updated_at INTEGER NOT NULL,
            PRIMARY KEY (user_id, conversation_id)
        );
        INSERT INTO short_term_contexts(user_id, conversation_id, recent_messages, updated_at)
        VALUES ('operator', 'conversation-1', '[]', 1);
        PRAGMA user_version = 8;
        """
    )
    conn.close()

    migrated = open_database(path)
    try:
        columns = {row[1] for row in migrated.execute("PRAGMA table_info(short_term_contexts)")}
        assert {
            "scenario_id",
            "compression_status",
            "last_compression_work_id",
            "compressed_message_count",
        } <= columns
        assert migrated.execute("SELECT COUNT(*) FROM short_term_contexts").fetchone()[0] == 1
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert migrated.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE '%_legacy'"
        ).fetchone()[0] == 0
    finally:
        migrated.close()

    reopened = open_database(path)
    reopened.close()


def test_v9_work_schema_rebuilds_dedupe_and_rolls_back_failed_migration(tmp_path, monkeypatch):
    path = tmp_path / "v9-work-items.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE memory_work_items (
            work_id TEXT PRIMARY KEY, source_key TEXT NOT NULL, user_id TEXT NOT NULL,
            conversation_id TEXT, scenario_id TEXT, work_type TEXT NOT NULL,
            payload TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            available_at INTEGER NOT NULL, created_at INTEGER NOT NULL, completed_at INTEGER,
            last_error TEXT, claimed_by TEXT, lease_expires_at INTEGER,
            UNIQUE (user_id, source_key)
        );
        INSERT INTO memory_work_items(
            work_id, source_key, user_id, scenario_id, work_type, payload, status,
            available_at, created_at
        ) VALUES ('legacy-work', 'same-source', 'operator', NULL, 'observation', '{}', 'pending', 1, 1);
        CREATE TABLE memory_stream_events (
            cursor INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
            user_id TEXT NOT NULL, conversation_id TEXT, status TEXT NOT NULL, type TEXT NOT NULL,
            payload TEXT NOT NULL, memory_id TEXT, memory_family_id TEXT, version INTEGER,
            created_at INTEGER NOT NULL, sim_time_s REAL
        );
        PRAGMA user_version = 9;
        """
    )
    conn.close()

    original_indexes = sqlite_module._CREATE_INDEXES
    monkeypatch.setattr(
        sqlite_module,
        "_CREATE_INDEXES",
        original_indexes + ("CREATE INDEX broken_migration_index ON missing_table(value)",),
    )
    with pytest.raises(sqlite3.OperationalError, match="missing_table"):
        open_database(path)

    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 9
        assert check.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE '%_legacy'"
        ).fetchone()[0] == 0
        assert check.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memory_work_items'"
        ).fetchone()[0].count("UNIQUE (user_id, source_key)") == 1
    finally:
        check.close()

    monkeypatch.setattr(sqlite_module, "_CREATE_INDEXES", original_indexes)
    migrated = open_database(path)
    try:
        unique_indexes = [
            row[1]
            for row in migrated.execute("PRAGMA index_list(memory_work_items)")
            if row[2]
        ]
        assert any(
            tuple(row[2] for row in migrated.execute(f"PRAGMA index_info({index})"))
            == ("user_id", "scenario_id", "source_key")
            for index in unique_indexes
        )
        assert migrated.execute(
            "SELECT scenario_id FROM memory_work_items WHERE work_id = 'legacy-work'"
        ).fetchone()[0] == "__legacy__"
    finally:
        migrated.close()


def test_source_cursor_duplicate_rows_abort_migration_and_retry_after_repair(
    tmp_path,
) -> None:
    path = tmp_path / "duplicate-source-cursor.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE memory_source_cursors (
            user_id TEXT, scenario_id TEXT, source_type TEXT, source_cursor INTEGER
        );
        INSERT INTO memory_source_cursors VALUES
            ('operator', 'scenario-1', 'runtime_event', 1),
            ('operator', 'scenario-1', 'runtime_event', 2);
        PRAGMA user_version = 10;
        """
    )
    conn.close()

    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        open_database(path)

    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 10
        assert check.execute("SELECT COUNT(*) FROM memory_source_cursors").fetchone()[0] == 2
        assert "updated_at" not in {
            row[1] for row in check.execute("PRAGMA table_info(memory_source_cursors)")
        }
    finally:
        check.close()

    repair = sqlite3.connect(path)
    repair.execute(
        "DELETE FROM memory_source_cursors WHERE user_id = ? AND scenario_id = ?"
        " AND source_type = ? AND source_cursor = ?",
        ("operator", "scenario-1", "runtime_event", 2),
    )
    repair.commit()
    repair.close()

    migrated = open_database(path)
    try:
        assert migrated.execute("SELECT COUNT(*) FROM memory_source_cursors").fetchone()[0] == 1
        assert {
            "user_id",
            "scenario_id",
            "source_type",
            "source_cursor",
            "updated_at",
        } <= {row[1] for row in migrated.execute("PRAGMA table_info(memory_source_cursors)")}
    finally:
        migrated.close()
    open_database(path).close()


def test_source_discovery_null_user_aborts_migration_and_retry_repairs_table(tmp_path) -> None:
    path = tmp_path / "null-source-discovery.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE memory_source_discovery (
            user_id TEXT, repository_index INTEGER, offsets TEXT
        );
        INSERT INTO memory_source_discovery VALUES (NULL, 0, '[]');
        PRAGMA user_version = 10;
        """
    )
    conn.close()

    with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
        open_database(path)

    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 10
        assert check.execute("SELECT user_id FROM memory_source_discovery").fetchone()[0] is None
        assert "updated_at" not in {
            row[1] for row in check.execute("PRAGMA table_info(memory_source_discovery)")
        }
    finally:
        check.close()

    repair = sqlite3.connect(path)
    repair.execute("DELETE FROM memory_source_discovery")
    repair.execute(
        "INSERT INTO memory_source_discovery VALUES (?, ?, ?)",
        ("operator", 0, "[]"),
    )
    repair.commit()
    repair.close()

    migrated = open_database(path)
    try:
        assert tuple(
            migrated.execute(
                "SELECT user_id, repository_index, offsets FROM memory_source_discovery"
            ).fetchone()
        ) == ("operator", 0, "[]")
        assert tuple(
            row[1] for row in migrated.execute("PRAGMA index_list(memory_source_discovery)")
        )
    finally:
        migrated.close()
    open_database(path).close()


def test_short_term_context_updates_within_user_and_isolates_other_users(tmp_path):
    repo = ShortTermContextRepository(tmp_path / "memory.db")
    first = repo.append_messages(
        "operator",
        "conversation-1",
        (ShortTermMessage(message_id="message-1", role="user", text="one"),),
    )
    second = repo.append_messages(
        "operator",
        "conversation-1",
        (ShortTermMessage(message_id="message-2", role="assistant", text="two"),),
    )
    repo.append_messages(
        "other-user",
        "conversation-1",
        (ShortTermMessage(message_id="message-3", role="user", text="private"),),
    )

    assert first.message_count == 1
    assert second.message_count == 2
    assert [message.message_id for message in second.recent_messages] == ["message-1", "message-2"]
    assert repo.get_short_term("other-user", "conversation-1").recent_messages[0].text == "private"
    assert repo.get_short_term("operator", "missing") is None

    context_before_compression = repo.get_short_term("operator", "conversation-1")
    assert context_before_compression is not None
    repo.append_messages(
        "operator",
        "conversation-1",
        (ShortTermMessage(message_id="message-3", role="user", text="new"),),
    )
    compressed = repo.save_compressed_context(
        "operator",
        "conversation-1",
        expected_summary_version=0,
        summary="compressed",
        retained_messages=(ShortTermMessage(message_id="message-2", role="assistant", text="two"),),
        expected_message_count=context_before_compression.message_count,
    )
    assert compressed.summary_version == 1
    assert compressed.message_count == 3
    assert compressed.compressed_message_count == 2
    with pytest.raises(VersionConflictError):
        repo.save_compressed_context(
            "operator", "conversation-1", 0, "stale", ()
        )


def test_short_term_context_isolated_by_scenario_and_cursor(tmp_path):
    repo = ShortTermContextRepository(tmp_path / "memory.db")
    repo.append_messages(
        "operator",
        "conversation-1",
        (ShortTermMessage(message_id="message-a", scenario_id="scenario-a", role="user", text="a"),),
        scenario_id="scenario-a",
    )
    repo.append_messages(
        "operator",
        "conversation-1",
        (ShortTermMessage(message_id="message-b", scenario_id="scenario-b", role="user", text="b"),),
        scenario_id="scenario-b",
    )

    assert [item.message_id for item in repo.get_short_term(
        "operator", "conversation-1", "scenario-a"
    ).recent_messages] == ["message-a"]
    assert [item.message_id for item in repo.get_short_term(
        "operator", "conversation-1", "scenario-b"
    ).recent_messages] == ["message-b"]


def test_memory_records_keep_execution_context_across_reopen(tmp_path):
    path = tmp_path / "memory-context.db"
    short_term = ShortTermContextRepository(path)
    message = ShortTermMessage(
        message_id="message-context",
        scenario_id="scenario-1",
        role="user",
        text="context",
        execution_revision=7,
        frame_id=42,
    )
    context = short_term.append_messages(
        "operator", "conversation-1", (message,), scenario_id="scenario-1"
    )
    assert context.recent_messages[0].execution_revision == 7
    short_term.close()

    long_term = LongTermMemoryRepository(path)
    memory = _memory("memory-context").model_copy(
        update={"scenario_id": "scenario-1", "execution_revision": 7, "frame_id": 42}
    )
    long_term.create_memory_version(memory, expected_previous_version=0)
    event = _stream("stream-context").model_copy(
        update={
            "scenario_id": "scenario-1",
            "execution_revision": 7,
            "frame_id": 42,
        }
    )
    stored_event = long_term.append_stream_event(event)

    reopened_short_term = ShortTermContextRepository(path)
    reopened_long_term = LongTermMemoryRepository(path)
    assert reopened_short_term.get_short_term(
        "operator", "conversation-1", "scenario-1"
    ).execution_revision == 7
    assert reopened_short_term.list_messages(
        "operator", "conversation-1", scenario_id="scenario-1"
    )[0].frame_id == 42
    assert reopened_long_term.get_memory(
        "operator", "memory-context", "scenario-1"
    ).execution_revision == 7
    assert reopened_long_term.list_stream_events(
        "operator", "conversation-1", scenario_id="scenario-1"
    )[0].frame_id == 42
    assert stored_event.execution_revision == 7


def test_memory_round_trip_keeps_non_ontology_sources(tmp_path) -> None:
    repo = LongTermMemoryRepository(tmp_path / "memory.db")
    memory = _memory("memory-provenance").model_copy(
        update={
            "source_message_ids": ("message-1",),
            "source_event_ids": ("event-1",),
            "source_decision_ids": ("decision-1",),
            "source_plan_ids": ("plan-1",),
        }
    )

    repo.create_memory_version(memory, expected_previous_version=0)
    loaded = repo.get_memory("operator", memory.memory_id)

    assert loaded is not None
    assert loaded.source_message_ids == ("message-1",)
    assert loaded.source_event_ids == ("event-1",)
    assert loaded.source_decision_ids == ("decision-1",)
    assert loaded.source_plan_ids == ("plan-1",)
    assert not hasattr(loaded, "source_knowledge_ids")


def test_short_term_messages_without_matching_scenario_are_rejected(tmp_path):
    repo = ShortTermContextRepository(tmp_path / "memory.db")
    with pytest.raises(ValueError, match="scenario"):
        repo.append_messages(
            "operator",
            "conversation-1",
            (ShortTermMessage(message_id="message-a", role="user", text="a"),),
            scenario_id="scenario-a",
        )


def test_short_term_append_and_enqueue_rejects_mismatched_message_atomically(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "memory.db")
    item = _work("work-mismatch")
    with pytest.raises(ValueError, match="scenario"):
        repo.append_messages_and_enqueue_work(
            "operator",
            "conversation-1",
            (ShortTermMessage(message_id="wrong", scenario_id="scenario-2", role="user", text="wrong"),),
            item,
            "conversation:scenario-1:conversation-1:wrong",
            scenario_id="scenario-1",
            source_type="conversation:scenario-1:conversation-1",
        )
    assert repo.get_work(item.work_id) is None
    assert repo._conn.execute("SELECT COUNT(*) FROM short_term_contexts").fetchone()[0] == 0


def test_memory_stream_events_are_isolated_by_scenario(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "memory.db")
    for scenario_id, event_id in (("scenario-a", "stream-a"), ("scenario-b", "stream-b")):
        repo.append_stream_event(
            MemoryStreamEvent(
                cursor=0,
                event_id=event_id,
                user_id="operator",
                conversation_id="conversation-1",
                scenario_id=scenario_id,
                status=MemoryStreamStatus.COMPLETED,
                type=MemoryStreamEventType.MEMORY_ACCESSED,
            )
        )

    assert [event.event_id for event in repo.list_stream_events(
        "operator", "conversation-1", scenario_id="scenario-a"
    )] == ["stream-a"]
    assert [event.event_id for event in repo.list_stream_events(
        "operator", "conversation-1", scenario_id="scenario-b"
    )] == ["stream-b"]


def test_memory_stream_combines_scenario_and_conversation_events_on_one_cursor(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "memory.db")
    scenario_event = repo.append_stream_event(
        MemoryStreamEvent(
            cursor=0,
            event_id="stream-scenario",
            user_id="operator",
            scenario_id="scenario-a",
            conversation_id=None,
            status=MemoryStreamStatus.COMPLETED,
            type=MemoryStreamEventType.CONTEXT_LOADED,
        )
    )
    conversation_event = repo.append_stream_event(
        MemoryStreamEvent(
            cursor=0,
            event_id="stream-conversation",
            user_id="operator",
            scenario_id="scenario-a",
            conversation_id="conversation-a",
            status=MemoryStreamStatus.COMPLETED,
            type=MemoryStreamEventType.MEMORY_EXTRACTED,
        )
    )
    repo.append_stream_event(
        MemoryStreamEvent(
            cursor=0,
            event_id="stream-other-conversation",
            user_id="operator",
            scenario_id="scenario-a",
            conversation_id="conversation-b",
            status=MemoryStreamStatus.COMPLETED,
            type=MemoryStreamEventType.MEMORY_ACCESSED,
        )
    )

    combined = repo.list_stream_events(
        "operator", "conversation-a", scenario_id="scenario-a", limit=10
    )
    assert [event.event_id for event in combined] == [
        scenario_event.event_id,
        conversation_event.event_id,
    ]
    assert [event.event_id for event in repo.list_stream_events(
        "operator",
        "conversation-a",
        scenario_id="scenario-a",
        after_cursor=scenario_event.cursor,
        limit=10,
    )] == [conversation_event.event_id]
    assert [event.event_id for event in repo.list_stream_events(
        "operator",
        "conversation-a",
        scenario_id="scenario-a",
        include_scenario_events=False,
        limit=10,
    )] == [conversation_event.event_id]


def test_memory_stream_read_uses_a_wal_read_connection(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "memory.db")
    repo.append_stream_event(
        MemoryStreamEvent(
            cursor=0,
            event_id="stream-locked-read",
            user_id="operator",
            scenario_id="scenario-a",
            conversation_id=None,
            status=MemoryStreamStatus.COMPLETED,
            type=MemoryStreamEventType.CONTEXT_LOADED,
        )
    )
    lock = sqlite_module.database_write_lock(repo._conn)
    read_finished = Event()
    read_result: list[list[MemoryStreamEvent]] = []

    def read_stream() -> None:
        read_result.append(
            repo.list_stream_events(
                "operator",
                "conversation-a",
                scenario_id="scenario-a",
            )
        )
        read_finished.set()

    lock.acquire()
    thread = Thread(target=read_stream)
    thread.start()
    try:
        assert read_finished.wait(0.5)
    finally:
        lock.release()
    thread.join(timeout=1.0)

    assert read_finished.is_set()
    assert [event.event_id for event in read_result[0]] == ["stream-locked-read"]


def test_same_family_can_start_independent_versions_in_each_scenario(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "memory.db")
    repo.create_memory_version(
        _memory("memory-a", family_id="shared-family").model_copy(update={"scenario_id": "scenario-a"}),
        expected_previous_version=0,
    )
    repo.create_memory_version(
        _memory("memory-b", family_id="shared-family").model_copy(update={"scenario_id": "scenario-b"}),
        expected_previous_version=0,
    )

    assert [item.version for item in repo.list_versions("operator", "shared-family", "scenario-a")] == [1]
    assert [item.version for item in repo.list_versions("operator", "shared-family", "scenario-b")] == [1]
    assert {item.memory_id for item in repo.list_active("operator", {"scenario_id": "scenario-a"})} == {"memory-a"}
    assert {item.memory_id for item in repo.list_active("operator", {"scenario_id": "scenario-b"})} == {"memory-b"}


def test_long_term_memory_versions_are_atomic_stable_and_user_scoped(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "memory.db")
    first = _memory("memory-1")
    repo.create_memory_version(first, expected_previous_version=0)
    second = _memory("memory-2", version=2, supersedes_memory_id="memory-1")
    repo.create_memory_version(second, expected_previous_version=1)
    third = _memory("memory-3", version=3, supersedes_memory_id="memory-2")
    repo.create_memory_version(third, expected_previous_version=2)

    assert [item.version for item in repo.list_versions("operator", "family-1")] == [1, 2, 3]
    assert [item.memory_id for item in repo.list_active("operator", limit=10)] == ["memory-3"]
    with pytest.raises(VersionConflictError):
        repo.create_memory_version(
        _memory("memory-4", version=3, supersedes_memory_id="memory-2"),
            expected_previous_version=2,
        )
    assert [item.memory_id for item in repo.list_active("other-user", limit=10)] == []


def test_memory_read_queries_never_project_legacy_ontology_column(tmp_path) -> None:
    repo = LongTermMemoryRepository(tmp_path / "memory.db")
    repo._conn.execute(
        "ALTER TABLE long_term_memories ADD COLUMN source_knowledge_ids"
        " TEXT NOT NULL DEFAULT '[]'"
    )
    work = _work("work-projection")
    assert repo.enqueue_work(work, "projection-source")
    memory = _memory("memory-projection")
    repo.create_memory_version(memory, expected_previous_version=0, work_id=work.work_id)

    requested_columns: list[str] = []

    def deny_legacy_column(
        action: int,
        table_name: str | None,
        column_name: str | None,
        database_name: str | None,
        trigger_name: str | None,
    ) -> int:
        del database_name, trigger_name
        if action == sqlite3.SQLITE_READ and table_name == "long_term_memories":
            requested_columns.append(column_name or "")
            if column_name == "source_knowledge_ids":
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    repo._conn.set_authorizer(deny_legacy_column)
    repo._read_conn.set_authorizer(deny_legacy_column)

    idempotent = repo.create_memory_version(
        memory, expected_previous_version=0, work_id=work.work_id
    )
    for loaded in (
        idempotent,
        repo.get_memory_for_work("operator", work.work_id),
        repo.list_active("operator")[0],
        repo.list_versions("operator", memory.memory_family_id)[0],
        repo.get_memory("operator", memory.memory_id),
    ):
        assert loaded is not None
        assert loaded.memory_id == memory.memory_id
        assert loaded.source_event_ids == ("event-1",)
    assert "source_knowledge_ids" not in requested_columns


def test_long_term_repository_accepts_the_contract_maximum_embedding_size(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "memory.db")
    memory = _memory("memory-maximum-embedding").model_copy(
        update={
            "embedding": tuple(
                (index + 0.12345678901234567) / 16_384 for index in range(16_384)
            )
        }
    )

    repo.create_memory_version(memory, expected_previous_version=0)
    assert len(repo.list_active("operator", limit=1)[0].embedding) == 16_384


def test_delete_memory_family_leaves_original_event_and_decision_audit_rows(tmp_path):
    path = tmp_path / "memory.db"
    events = EventRepository(path)
    events.append(
        event_id="event-1", event_type="bearing", scenario_id="scenario-1", sim_time_s=1, payload={}
    )
    ledger = DecisionLedger(path)
    ledger.save_question(
        run_id="question-1", scenario_id="scenario-1", question_text="why", payload={}
    )
    repo = LongTermMemoryRepository(path)
    repo.create_memory_version(_memory("memory-1"), expected_previous_version=0)

    assert repo.mark_deleted("operator", "memory-1") is True
    assert repo.list_active("operator", limit=10) == []
    assert events.get("event-1") is not None
    assert ledger.list_questions(scenario_id="scenario-1")[0].run_id == "question-1"


def test_work_queue_claims_retries_and_deduplicates_source_keys(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "memory.db")
    now = datetime.now(UTC) + timedelta(seconds=1)
    assert repo.enqueue_work(_work("work-1"), "event:event-1") is True
    assert repo.enqueue_work(_work("work-2"), "event:event-1") is False
    assert repo.enqueue_work(
        _work("work-other-user", user_id="other-user").model_copy(
            update={"available_at": now + timedelta(seconds=1)}
        ),
        "event:event-1",
    ) is True
    claimed = repo.claim_work("worker-a", now, lease_timeout_s=30)
    assert claimed is not None and claimed.work_id == "work-1" and claimed.attempts == 1
    assert repo.claim_work("worker-b", now, lease_timeout_s=30) is None

    reclaimed = repo.claim_work("worker-b", now + timedelta(seconds=31), lease_timeout_s=30)
    assert reclaimed is not None and reclaimed.work_id == "work-1" and reclaimed.attempts == 2
    assert repo.fail_work(
        "work-1", "worker-b", MemoryWorkStatus.PENDING, "temporary failure", now
    ) is True
    retried = repo.claim_work("worker-c", now, lease_timeout_s=30)
    assert retried is not None and retried.attempts == 3 and retried.last_error == "temporary failure"
    assert repo.complete_work("work-1", "worker-c") is True
    other_user_item = repo.claim_work("worker-c", now + timedelta(seconds=1), lease_timeout_s=30)
    assert other_user_item is not None and other_user_item.work_id == "work-other-user"
    assert repo.complete_work("work-other-user", "worker-c") is True
    assert repo.claim_work("worker-c", now + timedelta(seconds=1), lease_timeout_s=30) is None


def test_work_queue_deduplication_includes_scenario_and_preserves_legacy_null(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "memory.db")
    scenario_a = _work("work-a").model_copy(update={"scenario_id": "scenario-a"})
    scenario_b = _work("work-b").model_copy(update={"scenario_id": "scenario-b"})
    legacy = _work("work-legacy").model_copy(update={"scenario_id": None})

    assert repo.enqueue_work(scenario_a, "same-source") is True
    assert repo.enqueue_work(scenario_b, "same-source") is True
    assert repo.enqueue_work(scenario_a.model_copy(update={"work_id": "duplicate-a"}), "same-source") is False
    assert repo.enqueue_work(legacy, "same-source") is True

    assert repo.get_work_by_source_key("operator", "same-source", scenario_id="scenario-a").work_id == "work-a"
    assert repo.get_work_by_source_key("operator", "same-source", scenario_id="scenario-b").work_id == "work-b"
    assert repo.get_work_by_source_key("operator", "same-source", scenario_id=None).work_id == "work-legacy"


def test_work_queue_degrades_items_after_max_attempts(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "memory.db")
    now = datetime.now(UTC) + timedelta(seconds=1)

    assert repo.enqueue_work(_work("work-failed"), "event:event-failed") is True
    first = repo.claim_work("worker-a", now, lease_timeout_s=30, max_attempts=2)
    assert first is not None and first.attempts == 1
    assert repo.fail_work(
        "work-failed",
        "worker-a",
        MemoryWorkStatus.PENDING,
        "temporary failure",
        now,
        max_attempts=2,
    )
    second = repo.claim_work("worker-b", now, lease_timeout_s=30, max_attempts=2)
    assert second is not None and second.attempts == 2
    assert repo.fail_work(
        "work-failed",
        "worker-b",
        MemoryWorkStatus.PENDING,
        "final failure",
        now,
        max_attempts=2,
    )
    failed = repo._conn.execute(
        "SELECT status, completed_at, last_error FROM memory_work_items WHERE work_id = ?",
        ("work-failed",),
    ).fetchone()
    assert failed["status"] == MemoryWorkStatus.DEGRADED.value
    assert failed["completed_at"] is not None
    assert failed["last_error"] == "final failure"
    assert repo.claim_work("worker-c", now, lease_timeout_s=30, max_attempts=2) is None

    assert repo.enqueue_work(_work("work-expired"), "event:event-expired") is True
    claimed = repo.claim_work("worker-a", now, lease_timeout_s=1, max_attempts=1)
    assert claimed is not None and claimed.work_id == "work-expired" and claimed.attempts == 1
    assert repo.claim_work(
        "worker-b", now + timedelta(seconds=2), lease_timeout_s=1, max_attempts=1
    ) is None
    expired = repo._conn.execute(
        "SELECT status, completed_at FROM memory_work_items WHERE work_id = ?",
        ("work-expired",),
    ).fetchone()
    assert expired["status"] == MemoryWorkStatus.DEGRADED.value
    assert expired["completed_at"] is not None


def test_source_cursor_rolls_back_when_atomic_enqueue_fails(tmp_path, monkeypatch):
    repo = LongTermMemoryRepository(tmp_path / "memory.db")

    def fail_cursor(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("cursor write failed")

    monkeypatch.setattr(repo, "_upsert_source_cursor", fail_cursor)
    with pytest.raises(RuntimeError, match="cursor write failed"):
        repo.enqueue_work_and_advance_cursor(
            _work("work-atomic"),
            "runtime_event:event-atomic",
            "scenario-1",
            "runtime_event",
            1,
        )

    assert repo.get_source_cursor("operator", "scenario-1", "runtime_event") == 0
    assert repo._conn.execute(
        "SELECT COUNT(*) FROM memory_work_items WHERE work_id = ?", ("work-atomic",)
    ).fetchone()[0] == 0


def test_source_cursors_stream_cursors_and_access_metrics_are_scoped_and_bounded(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "memory.db")
    repo.create_memory_version(_memory("memory-1"), expected_previous_version=0)
    repo.advance_source_cursor("operator", "scenario-1", "runtime_event", 3)
    repo.advance_source_cursor("operator", "scenario-1", "runtime_event", 8)
    assert repo.get_source_cursor("operator", "scenario-1", "runtime_event") == 8
    assert repo.get_source_cursor("other-user", "scenario-1", "runtime_event") == 0

    first = repo.append_stream_event(_stream("stream-1"))
    second = repo.append_stream_event(_stream("stream-2"))
    repo.append_stream_event(_stream("stream-3", user_id="other-user"))
    repo.append_stream_event(_stream("stream-4", conversation_id="other-conversation"))
    listed = repo.list_stream_events("operator", "conversation-1", after_cursor=first.cursor, limit=999)
    assert [event.event_id for event in listed] == ["stream-2"]
    assert second.cursor > first.cursor
    assert repo.list_stream_events("operator", "conversation-1", after_cursor=second.cursor, limit=1) == []

    repo.record_access("operator", ("memory-1",))
    active = repo.list_active("operator", filters={"min_importance_score": 0.7}, limit=1)[0]
    assert active.access_count == 1
    assert active.last_accessed_at is not None
    assert active.importance_score == 0.7
