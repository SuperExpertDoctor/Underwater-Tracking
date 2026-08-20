"""SQLite persistence contracts for the asynchronous memory pipeline."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

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
            } <= tables
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
            change_reason, created_at
        ) VALUES ('legacy-memory', 'family-legacy', 1, 'operator', 'semantic', 'legacy',
                  0.7, 0.7, '[0.1]', 'v1', 'active', 'created', 1);
        PRAGMA user_version = 5;
        """
    )
    conn.close()

    migrated = open_database(path)
    try:
        columns = {row[1] for row in migrated.execute("PRAGMA table_info(long_term_memories)")}
        assert "scenario_id" in columns
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
        assert {"scenario_id", "compression_status", "last_compression_work_id"} <= columns
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

    compressed = repo.save_compressed_context(
        "operator",
        "conversation-1",
        expected_summary_version=0,
        summary="compressed",
        retained_messages=(ShortTermMessage(message_id="message-2", role="assistant", text="two"),),
    )
    assert compressed.summary_version == 1
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
