# src/underwater_tracking/persistence/sqlite.py
"""SQLite connection factory, schema migrations, and canonical JSON.

All agent persistence shares one SQLite database file per run (spec 5.4):
LangGraph checkpoints, runtime events, plan versions, the DecisionLedger,
expert directives, and LLM call metadata. The database is opened in WAL mode
with foreign keys enabled and autocommit (``isolation_level=None``) so every
multi-statement write manages its own explicit ``BEGIN IMMEDIATE`` /
``COMMIT`` transaction (spec 8.4; pre-flight ruling: stdlib sqlite3, WAL, no
extra dependencies). JSON payloads are serialized canonically with sorted
keys so byte-equal round trips are stable for replay and diffing; simulation
timestamps are integers (``sim_time_s``) and wall-clock rows carry integer
millisecond ``created_at`` values.
"""

from __future__ import annotations

import json
import sqlite3
from functools import wraps
from threading import Lock, RLock
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 17
LEGACY_SCENARIO_ID = "__legacy__"
_BUSY_TIMEOUT_MS = 60_000

_DATABASE_LOCKS: dict[str, RLock] = {}
_DATABASE_LOCKS_GUARD = Lock()
_FALLBACK_DATABASE_LOCK = RLock()


def _database_lock_key(database_path: str | Path) -> str:
    raw_path = os.fspath(database_path)
    if raw_path == ":memory:" or raw_path.startswith("file:"):
        return raw_path
    return str(Path(raw_path).expanduser().resolve())


def _database_write_lock(database_path: str | Path) -> RLock:
    key = _database_lock_key(database_path)
    with _DATABASE_LOCKS_GUARD:
        lock = _DATABASE_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _DATABASE_LOCKS[key] = lock
        return lock


class SQLiteConnection(sqlite3.Connection):
    """Connection carrying the process-wide write lock for its database."""

    def __init__(self, database_path: str | bytes | os.PathLike[str], *args, **kwargs):
        super().__init__(database_path, *args, **kwargs)
        self.database_write_lock = _database_write_lock(database_path)


def connect_database(
    database_path: str | Path, *, row_factory: bool = False
) -> sqlite3.Connection:
    """Open a configured connection without running the application schema migration."""
    conn = sqlite3.connect(
        os.fspath(database_path),
        check_same_thread=False,
        isolation_level=None,
        timeout=_BUSY_TIMEOUT_MS / 1000,
        factory=SQLiteConnection,
    )
    if row_factory:
        conn.row_factory = sqlite3.Row
    with database_write_lock(conn):
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA journal_mode=WAL")
    return conn


def database_write_lock(conn: sqlite3.Connection) -> RLock:
    """Return the process-wide write lock associated with a SQLite connection."""
    return getattr(conn, "database_write_lock", _FALLBACK_DATABASE_LOCK)


def synchronized_database_method(method: Callable[..., Any]) -> Callable[..., Any]:
    """Serialize one repository call, including its cursor fetches."""
    @wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        with database_write_lock(self._conn):
            return method(self, *args, **kwargs)

    return wrapped

_CREATE_TABLES = (
    """
    CREATE TABLE IF NOT EXISTS runtime_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        event_type TEXT NOT NULL,
        scenario_id TEXT NOT NULL,
        target_id TEXT,
        sim_time_s INTEGER NOT NULL,
        severity TEXT NOT NULL DEFAULT 'info',
        audiences_json TEXT NOT NULL DEFAULT '["blue_planning","memory_source","operator_audit"]',
        payload TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS snapshots (
        scenario_id TEXT PRIMARY KEY,
        revision INTEGER NOT NULL,
        snapshot_hash TEXT NOT NULL DEFAULT '',
        updated_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS plans (
        plan_id TEXT PRIMARY KEY,
        scenario_id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        base_snapshot_revision INTEGER NOT NULL,
        status TEXT NOT NULL,
        valid_from_s INTEGER NOT NULL,
        valid_until_s INTEGER NOT NULL,
        payload TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        UNIQUE (scenario_id, revision)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS plan_commands (
        command_id TEXT PRIMARY KEY,
        plan_id TEXT NOT NULL REFERENCES plans(plan_id),
        plan_revision INTEGER NOT NULL,
        scenario_id TEXT NOT NULL,
        group_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        sim_time_s INTEGER NOT NULL,
        payload TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS planning_epochs (
        epoch_id TEXT PRIMARY KEY,
        scenario_id TEXT NOT NULL,
        base_physics_revision INTEGER NOT NULL,
        base_sim_time_s INTEGER NOT NULL,
        status TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS planning_epoch_inputs (
        epoch_id TEXT PRIMARY KEY REFERENCES planning_epochs(epoch_id),
        observation_batch_id TEXT NOT NULL,
        situation_payload TEXT NOT NULL,
        mission_payload TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS planning_event_retries (
        scenario_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        retry_not_before_utc_ms INTEGER,
        status TEXT NOT NULL,
        payload TEXT NOT NULL,
        PRIMARY KEY (scenario_id, event_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS planning_revalidation_reports (
        report_id TEXT PRIMARY KEY,
        epoch_id TEXT NOT NULL REFERENCES planning_epochs(epoch_id),
        valid INTEGER NOT NULL,
        current_physics_revision INTEGER NOT NULL,
        payload TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS planning_epoch_results (
        epoch_id TEXT PRIMARY KEY REFERENCES planning_epochs(epoch_id),
        status TEXT NOT NULL,
        plan_id TEXT,
        plan_version INTEGER,
        validation_report_id TEXT REFERENCES planning_revalidation_reports(report_id),
        payload TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS execution_revisions (
        commit_id TEXT PRIMARY KEY,
        scenario_id TEXT NOT NULL,
        execution_revision INTEGER NOT NULL,
        candidate_execution_revision INTEGER,
        base_execution_revision INTEGER,
        status TEXT NOT NULL,
        source_snapshot_revision INTEGER,
        active_plan_preserved INTEGER NOT NULL DEFAULT 0,
        reason TEXT NOT NULL DEFAULT '',
        snapshot_payload TEXT,
        result_payload TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS decision_records (
        decision_id TEXT PRIMARY KEY,
        scenario_id TEXT NOT NULL,
        sim_time_s INTEGER NOT NULL,
        snapshot_revision INTEGER NOT NULL DEFAULT 0,
        payload TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llm_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        operation TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        request_hash TEXT NOT NULL DEFAULT '',
        response_hash TEXT NOT NULL DEFAULT '',
        latency_ms INTEGER NOT NULL DEFAULT 0,
        token_count INTEGER NOT NULL DEFAULT 0,
        error_category TEXT NOT NULL DEFAULT '',
        sim_time_s INTEGER NOT NULL DEFAULT 0,
        scenario_id TEXT NOT NULL DEFAULT '',
        base_execution_revision INTEGER,
        failed_fields_json TEXT NOT NULL DEFAULT '[]',
        active_plan_preserved INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS expert_directives (
        directive_id TEXT PRIMARY KEY,
        scenario_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'preview',
        confidence REAL NOT NULL,
        payload TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS question_runs (
        run_id TEXT PRIMARY KEY,
        scenario_id TEXT NOT NULL,
        question_text TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'completed',
        payload TEXT NOT NULL,
        execution_revision INTEGER,
        frame_id INTEGER,
        created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_queries (
        query_id TEXT PRIMARY KEY,
        scenario_id TEXT NOT NULL,
        sim_time_s INTEGER NOT NULL,
        query_text TEXT NOT NULL,
        mode TEXT NOT NULL,
        status TEXT NOT NULL,
        response_hash TEXT NOT NULL DEFAULT '',
        payload TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS short_term_contexts (
        user_id TEXT NOT NULL,
        scenario_id TEXT NOT NULL DEFAULT '__legacy__',
        conversation_id TEXT NOT NULL,
        summary_text TEXT NOT NULL DEFAULT '',
        summary_version INTEGER NOT NULL DEFAULT 0,
        recent_messages TEXT NOT NULL DEFAULT '[]',
        message_count INTEGER NOT NULL DEFAULT 0,
        compressed_message_count INTEGER NOT NULL DEFAULT 0,
        estimated_tokens INTEGER NOT NULL DEFAULT 0,
        compression_count INTEGER NOT NULL DEFAULT 0,
        last_compressed_at INTEGER,
        compression_status TEXT NOT NULL DEFAULT 'pending',
        last_compression_work_id TEXT,
        execution_revision INTEGER,
        frame_id INTEGER,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY (user_id, scenario_id, conversation_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS short_term_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        scenario_id TEXT NOT NULL DEFAULT '__legacy__',
        conversation_id TEXT NOT NULL,
        message_id TEXT NOT NULL,
        turn_id TEXT,
        role TEXT NOT NULL,
        text TEXT NOT NULL,
        source_evidence_ids TEXT NOT NULL DEFAULT '[]',
        execution_revision INTEGER,
        frame_id INTEGER,
        created_at INTEGER NOT NULL,
        UNIQUE (user_id, scenario_id, conversation_id, message_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS long_term_memories (
        memory_id TEXT PRIMARY KEY,
        memory_work_id TEXT,
        memory_family_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        user_id TEXT NOT NULL,
        scenario_id TEXT NOT NULL DEFAULT '__legacy__',
        memory_type TEXT NOT NULL,
        summary TEXT NOT NULL,
        importance_score REAL NOT NULL,
        importance_baseline REAL NOT NULL DEFAULT 0.0,
        embedding TEXT NOT NULL,
        embedding_version TEXT NOT NULL,
        status TEXT NOT NULL,
        supersedes_memory_id TEXT,
        source_message_ids TEXT NOT NULL DEFAULT '[]',
        source_event_ids TEXT NOT NULL DEFAULT '[]',
        source_decision_ids TEXT NOT NULL DEFAULT '[]',
        source_knowledge_ids TEXT NOT NULL DEFAULT '[]',
        source_plan_ids TEXT NOT NULL DEFAULT '[]',
        change_reason TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        last_accessed_at INTEGER,
        access_count INTEGER NOT NULL DEFAULT 0,
        sim_time_s REAL,
        execution_revision INTEGER,
        frame_id INTEGER,
        UNIQUE (user_id, memory_family_id, scenario_id, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_work_items (
        work_id TEXT PRIMARY KEY,
        source_key TEXT NOT NULL,
        user_id TEXT NOT NULL,
        conversation_id TEXT,
        scenario_id TEXT NOT NULL DEFAULT '__legacy__',
        work_type TEXT NOT NULL,
        payload TEXT NOT NULL,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        available_at INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        completed_at INTEGER,
        last_error TEXT,
        claimed_by TEXT,
        lease_expires_at INTEGER,
        UNIQUE (user_id, scenario_id, source_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_stream_events (
        cursor INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        user_id TEXT NOT NULL,
        scenario_id TEXT NOT NULL DEFAULT '__legacy__',
        conversation_id TEXT,
        status TEXT NOT NULL,
        type TEXT NOT NULL,
        payload TEXT NOT NULL,
        memory_id TEXT,
        memory_family_id TEXT,
        version INTEGER,
        created_at INTEGER NOT NULL,
        sim_time_s REAL,
        execution_revision INTEGER,
        frame_id INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_source_cursors (
        user_id TEXT NOT NULL,
        scenario_id TEXT NOT NULL,
        source_type TEXT NOT NULL,
        source_cursor INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY (user_id, scenario_id, source_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_source_discovery (
        user_id TEXT PRIMARY KEY,
        repository_index INTEGER NOT NULL,
        offsets TEXT NOT NULL,
        updated_at INTEGER NOT NULL
    )
    """,
)

_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_runtime_events_scenario ON runtime_events(scenario_id, sim_time_s)",
    "CREATE INDEX IF NOT EXISTS idx_runtime_events_type ON runtime_events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_plans_scenario_status ON plans(scenario_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_plan_commands_plan ON plan_commands(plan_id)",
    "CREATE INDEX IF NOT EXISTS idx_planning_epochs_scenario ON planning_epochs(scenario_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_execution_revisions_scenario ON execution_revisions(scenario_id, status, execution_revision)",
    "CREATE INDEX IF NOT EXISTS idx_planning_event_retries_status ON planning_event_retries(scenario_id, status, retry_not_before_utc_ms)",
    "CREATE INDEX IF NOT EXISTS idx_decision_records_scenario ON decision_records(scenario_id, sim_time_s)",
    "CREATE INDEX IF NOT EXISTS idx_llm_calls_scenario_operation ON llm_calls(scenario_id, operation, id)",
    "CREATE INDEX IF NOT EXISTS idx_expert_directives_scenario ON expert_directives(scenario_id)",
    "CREATE INDEX IF NOT EXISTS idx_question_runs_scenario ON question_runs(scenario_id)",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_queries_scenario ON knowledge_queries(scenario_id, sim_time_s)",
    "CREATE INDEX IF NOT EXISTS idx_short_term_contexts_updated ON short_term_contexts(user_id, scenario_id, updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_short_term_messages_scope ON short_term_messages(user_id, scenario_id, conversation_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_long_term_memories_lookup ON long_term_memories(user_id, status, memory_type, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_long_term_memories_scenario ON long_term_memories(user_id, scenario_id, status, created_at)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_long_term_memories_one_active_family"
    " ON long_term_memories(user_id, memory_family_id, scenario_id) WHERE status = 'active'",
    "CREATE INDEX IF NOT EXISTS idx_memory_work_items_available"
    " ON memory_work_items(status, available_at)",
    "CREATE INDEX IF NOT EXISTS idx_memory_work_items_lease"
    " ON memory_work_items(status, lease_expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_memory_stream_events_cursor"
    " ON memory_stream_events(user_id, scenario_id, conversation_id, cursor)",
    "CREATE INDEX IF NOT EXISTS idx_memory_source_cursors_scope_last_seen"
    " ON memory_source_cursors(source_type, updated_at, user_id, scenario_id)",
)


def open_database(database_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) a migrated SQLite database in WAL mode.

    The connection runs in autocommit mode (``isolation_level=None``) so
    callers control transactions explicitly via :func:`transaction`.
    ``check_same_thread=False`` lets the runtime share one repository from
    the engine loop and the LangGraph thread pool; concurrent writers are
    serialized by WAL plus a busy timeout. ``busy_timeout`` is set before
    ``journal_mode=WAL`` so a connection that loses the first-open WAL
    conversion race waits instead of failing at the default zero timeout.
    """
    conn = connect_database(database_path, row_factory=True)
    try:
        with database_write_lock(conn):
            conn.execute("PRAGMA foreign_keys=ON")
            _migrate(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Repair the live schema atomically and stamp the supported version."""
    stored = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if stored > SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema version {stored} is newer than supported"
            f" {SCHEMA_VERSION}; upgrade the code before opening this database"
        )
    with transaction(conn):
        for statement in _CREATE_TABLES:
            conn.execute(statement)
        _recover_abandoned_repairs(conn)
        _repair_execution_context_columns(conn)
        _repair_runtime_events(conn)
        _repair_short_term_contexts(conn)
        _repair_short_term_messages(conn)
        _repair_long_term_memories(conn)
        _repair_memory_work_items(conn)
        _repair_memory_stream_events(conn)
        _repair_memory_source_cursors(conn)
        _repair_memory_source_discovery(conn)
        _repair_llm_calls(conn)
        for statement in _CREATE_INDEXES:
            conn.execute(statement)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})")}


def _repair_runtime_events(conn: sqlite3.Connection) -> None:
    """Add audience metadata and quarantine historical private decisions."""
    columns = _table_columns(conn, "runtime_events")
    if "audiences_json" not in columns:
        conn.execute(
            "ALTER TABLE runtime_events ADD COLUMN audiences_json TEXT NOT NULL "
            "DEFAULT '[\"blue_planning\",\"memory_source\",\"operator_audit\"]'"
        )
    conn.execute(
        "UPDATE runtime_events SET audiences_json = "
        "'[\"adversary_private\",\"memory_source\",\"operator_audit\"]' "
        "WHERE event_type = 'target_mission_decision'"
    )


def _primary_key_columns(conn: sqlite3.Connection, table_name: str) -> tuple[str, ...]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return tuple(row[1] for row in sorted(rows, key=lambda row: row[5]) if row[5])


def _unique_index_columns(conn: sqlite3.Connection, table_name: str) -> set[tuple[str, ...]]:
    unique_columns: set[tuple[str, ...]] = set()
    for row in conn.execute(f"PRAGMA index_list({table_name})"):
        if not row[2]:
            continue
        unique_columns.add(tuple(item[2] for item in conn.execute(f"PRAGMA index_info({row[1]})")))
    return unique_columns


def _recover_abandoned_repairs(conn: sqlite3.Connection) -> None:
    """Recover databases left by the pre-transaction migration implementation."""
    for table_name in (
        "short_term_contexts",
        "short_term_messages",
        "long_term_memories",
        "memory_work_items",
        "memory_stream_events",
        "memory_source_cursors",
        "memory_source_discovery",
    ):
        legacy_name = f"{table_name}_legacy"
        if not _table_exists(conn, legacy_name):
            continue
        current_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        legacy_count = conn.execute(f"SELECT COUNT(*) FROM {legacy_name}").fetchone()[0]
        if current_count == 0 and legacy_count > 0:
            conn.execute(f"DROP TABLE {table_name}")
            conn.execute(f"ALTER TABLE {legacy_name} RENAME TO {table_name}")
        elif legacy_count == 0:
            conn.execute(f"DROP TABLE {legacy_name}")
        else:
            raise sqlite3.IntegrityError(
                f"abandoned migration tables {table_name} and {legacy_name} both contain rows"
            )


def _repair_short_term_contexts(conn: sqlite3.Connection) -> None:
    columns = (
        ("user_id", "'operator'"),
        ("scenario_id", f"'{LEGACY_SCENARIO_ID}'"),
        ("conversation_id", "''"),
        ("summary_text", "''"),
        ("summary_version", "0"),
        ("recent_messages", "'[]'"),
        ("message_count", "0"),
        ("compressed_message_count", "0"),
        ("estimated_tokens", "0"),
        ("compression_count", "0"),
        ("last_compressed_at", "NULL"),
        ("compression_status", "'pending'"),
        ("last_compression_work_id", "NULL"),
        ("execution_revision", "NULL"),
        ("frame_id", "NULL"),
        ("updated_at", "0"),
    )
    _repair_table(
        conn,
        "short_term_contexts",
        columns,
        """
        CREATE TABLE {table} (
            user_id TEXT NOT NULL,
            scenario_id TEXT NOT NULL DEFAULT '__legacy__',
            conversation_id TEXT NOT NULL,
            summary_text TEXT NOT NULL DEFAULT '',
            summary_version INTEGER NOT NULL DEFAULT 0,
            recent_messages TEXT NOT NULL DEFAULT '[]',
            message_count INTEGER NOT NULL DEFAULT 0,
            compressed_message_count INTEGER NOT NULL DEFAULT 0,
            estimated_tokens INTEGER NOT NULL DEFAULT 0,
            compression_count INTEGER NOT NULL DEFAULT 0,
            last_compressed_at INTEGER,
            compression_status TEXT NOT NULL DEFAULT 'pending',
            last_compression_work_id TEXT,
            execution_revision INTEGER,
            frame_id INTEGER,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (user_id, scenario_id, conversation_id)
        )
        """,
        primary_key=("user_id", "scenario_id", "conversation_id"),
        required_not_null=("scenario_id",),
    )


def _repair_short_term_messages(conn: sqlite3.Connection) -> None:
    """Backfill immutable message rows from pre-v12 rolling contexts."""
    if not _table_exists(conn, "short_term_contexts"):
        return
    rows = conn.execute(
        "SELECT user_id, scenario_id, conversation_id, recent_messages, updated_at "
        "FROM short_term_contexts"
    ).fetchall()
    for row in rows:
        try:
            messages = json.loads(row["recent_messages"])
        except (TypeError, ValueError):
            continue
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            message_id = message.get("message_id")
            role = message.get("role")
            text = message.get("text")
            if not all(isinstance(value, str) and value for value in (message_id, role, text)):
                continue
            created_at = message.get("created_at")
            created_ms = row["updated_at"]
            if isinstance(created_at, str):
                try:
                    created_ms = int(datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp() * 1000)
                except ValueError:
                    pass
            conn.execute(
                "INSERT OR IGNORE INTO short_term_messages "
                "(user_id, scenario_id, conversation_id, message_id, turn_id, role, text, "
                "source_evidence_ids, execution_revision, frame_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["user_id"],
                    row["scenario_id"],
                    row["conversation_id"],
                    message_id,
                    message.get("turn_id"),
                    role,
                    text,
                    json_dumps(message.get("source_evidence_ids", ())),
                    message.get("execution_revision"),
                    message.get("frame_id"),
                    created_ms,
                ),
            )


def _repair_long_term_memories(conn: sqlite3.Connection) -> None:
    columns = (
        ("memory_id", "''"),
        ("memory_work_id", "NULL"),
        ("memory_family_id", "''"),
        ("version", "1"),
        ("user_id", "'operator'"),
        ("scenario_id", f"'{LEGACY_SCENARIO_ID}'"),
        ("memory_type", "'semantic'"),
        ("summary", "''"),
        ("importance_score", "0.0"),
        ("importance_baseline", "importance_score"),
        ("embedding", "'[]'"),
        ("embedding_version", "'v1'"),
        ("status", "'active'"),
        ("supersedes_memory_id", "NULL"),
        ("source_message_ids", "'[]'"),
        ("source_event_ids", "'[]'"),
        ("source_decision_ids", "'[]'"),
        ("source_knowledge_ids", "'[]'"),
        ("source_plan_ids", "'[]'"),
        ("change_reason", "'created'"),
        ("created_at", "0"),
        ("last_accessed_at", "NULL"),
        ("access_count", "0"),
        ("sim_time_s", "NULL"),
        ("execution_revision", "NULL"),
        ("frame_id", "NULL"),
    )
    _repair_table(
        conn,
        "long_term_memories",
        columns,
        """
        CREATE TABLE {table} (
            memory_id TEXT PRIMARY KEY,
            memory_work_id TEXT,
            memory_family_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            user_id TEXT NOT NULL,
            scenario_id TEXT NOT NULL DEFAULT '__legacy__',
            memory_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            importance_score REAL NOT NULL,
            importance_baseline REAL NOT NULL DEFAULT 0.0,
            embedding TEXT NOT NULL,
            embedding_version TEXT NOT NULL,
            status TEXT NOT NULL,
            supersedes_memory_id TEXT,
            source_message_ids TEXT NOT NULL DEFAULT '[]',
            source_event_ids TEXT NOT NULL DEFAULT '[]',
            source_decision_ids TEXT NOT NULL DEFAULT '[]',
            source_knowledge_ids TEXT NOT NULL DEFAULT '[]',
            source_plan_ids TEXT NOT NULL DEFAULT '[]',
            change_reason TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            last_accessed_at INTEGER,
            access_count INTEGER NOT NULL DEFAULT 0,
            sim_time_s REAL,
            execution_revision INTEGER,
            frame_id INTEGER,
            UNIQUE (user_id, memory_family_id, scenario_id, version)
        )
        """,
        unique_key=("user_id", "memory_family_id", "scenario_id", "version"),
        required_not_null=("scenario_id",),
    )


def _repair_memory_work_items(conn: sqlite3.Connection) -> None:
    columns = (
        ("work_id", "''"),
        ("source_key", "''"),
        ("user_id", "'operator'"),
        ("conversation_id", "NULL"),
        ("scenario_id", f"'{LEGACY_SCENARIO_ID}'"),
        ("work_type", "'maintenance'"),
        ("payload", "'{}'"),
        ("status", "'pending'"),
        ("attempts", "0"),
        ("available_at", "0"),
        ("created_at", "0"),
        ("completed_at", "NULL"),
        ("last_error", "NULL"),
        ("claimed_by", "NULL"),
        ("lease_expires_at", "NULL"),
    )
    _repair_table(
        conn,
        "memory_work_items",
        columns,
        """
        CREATE TABLE {table} (
            work_id TEXT PRIMARY KEY,
            source_key TEXT NOT NULL,
            user_id TEXT NOT NULL,
            conversation_id TEXT,
            scenario_id TEXT NOT NULL DEFAULT '__legacy__',
            work_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            available_at INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            completed_at INTEGER,
            last_error TEXT,
            claimed_by TEXT,
            lease_expires_at INTEGER,
            UNIQUE (user_id, scenario_id, source_key)
        )
        """,
        unique_key=("user_id", "scenario_id", "source_key"),
        required_not_null=("scenario_id",),
    )


def _repair_memory_stream_events(conn: sqlite3.Connection) -> None:
    columns = (
        ("cursor", "0"),
        ("event_id", "''"),
        ("user_id", "'operator'"),
        ("scenario_id", f"'{LEGACY_SCENARIO_ID}'"),
        ("conversation_id", "NULL"),
        ("status", "'degraded'"),
        ("type", "'work_degraded'"),
        ("payload", "'{}'"),
        ("memory_id", "NULL"),
        ("memory_family_id", "NULL"),
        ("version", "NULL"),
        ("created_at", "0"),
        ("sim_time_s", "NULL"),
        ("execution_revision", "NULL"),
        ("frame_id", "NULL"),
    )
    _repair_table(
        conn,
        "memory_stream_events",
        columns,
        """
        CREATE TABLE {table} (
            cursor INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            user_id TEXT NOT NULL,
            scenario_id TEXT NOT NULL DEFAULT '__legacy__',
            conversation_id TEXT,
            status TEXT NOT NULL,
            type TEXT NOT NULL,
            payload TEXT NOT NULL,
            memory_id TEXT,
            memory_family_id TEXT,
            version INTEGER,
            created_at INTEGER NOT NULL,
            sim_time_s REAL,
            execution_revision INTEGER,
            frame_id INTEGER
        )
        """,
        primary_key=("cursor",),
        unique_key=("event_id",),
        required_not_null=("scenario_id",),
    )


def _repair_memory_source_cursors(conn: sqlite3.Connection) -> None:
    columns = (
        ("user_id", "'operator'"),
        ("scenario_id", f"'{LEGACY_SCENARIO_ID}'"),
        ("source_type", "''"),
        ("source_cursor", "0"),
        ("updated_at", "0"),
    )
    _repair_table(
        conn,
        "memory_source_cursors",
        columns,
        """
        CREATE TABLE {table} (
            user_id TEXT NOT NULL,
            scenario_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_cursor INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (user_id, scenario_id, source_type)
        )
        """,
        primary_key=("user_id", "scenario_id", "source_type"),
        required_not_null=("user_id", "scenario_id", "source_type", "source_cursor", "updated_at"),
    )


def _repair_memory_source_discovery(conn: sqlite3.Connection) -> None:
    columns = (
        ("user_id", "'operator'"),
        ("repository_index", "0"),
        ("offsets", "'[]'"),
        ("updated_at", "0"),
    )
    _repair_table(
        conn,
        "memory_source_discovery",
        columns,
        """
        CREATE TABLE {table} (
            user_id TEXT NOT NULL PRIMARY KEY,
            repository_index INTEGER NOT NULL,
            offsets TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """,
        primary_key=("user_id",),
        required_not_null=("user_id", "repository_index", "offsets", "updated_at"),
    )


def _repair_llm_calls(conn: sqlite3.Connection) -> None:
    """Add execution-strategy audit metadata without rewriting call history."""
    columns = _table_columns(conn, "llm_calls")
    if "base_execution_revision" not in columns:
        conn.execute(
            "ALTER TABLE llm_calls ADD COLUMN base_execution_revision INTEGER"
        )
    if "failed_fields_json" not in columns:
        conn.execute(
            "ALTER TABLE llm_calls ADD COLUMN failed_fields_json TEXT NOT NULL DEFAULT '[]'"
        )
    if "active_plan_preserved" not in columns:
        conn.execute(
            "ALTER TABLE llm_calls ADD COLUMN active_plan_preserved INTEGER NOT NULL DEFAULT 0"
        )


def _repair_execution_context_columns(conn: sqlite3.Connection) -> None:
    """Add nullable execution coordinates while preserving older run databases."""
    for table_name in (
        "question_runs",
        "short_term_contexts",
        "short_term_messages",
        "long_term_memories",
        "memory_stream_events",
    ):
        columns = _table_columns(conn, table_name)
        for column_name in ("execution_revision", "frame_id"):
            if column_name not in columns:
                conn.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} INTEGER"
                )


def _repair_table(
    conn: sqlite3.Connection,
    table_name: str,
    columns: tuple[tuple[str, str], ...],
    create_sql: str,
    *,
    primary_key: tuple[str, ...] | None = None,
    unique_key: tuple[str, ...] | None = None,
    required_not_null: tuple[str, ...] = (),
) -> None:
    if not _table_exists(conn, table_name):
        return
    existing_columns = _table_columns(conn, table_name)
    needs_rebuild = not {name for name, _ in columns} <= existing_columns
    if primary_key is not None and _primary_key_columns(conn, table_name) != primary_key:
        needs_rebuild = True
    if unique_key is not None and unique_key not in _unique_index_columns(conn, table_name):
        needs_rebuild = True
    actual_not_null = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table_name})")
        if row[3]
    }
    if not set(required_not_null) <= actual_not_null:
        needs_rebuild = True
    if not needs_rebuild:
        return

    for index_name in (
        "idx_short_term_contexts_updated",
        "idx_long_term_memories_lookup",
        "idx_long_term_memories_scenario",
        "idx_long_term_memories_one_active_family",
        "idx_memory_work_items_available",
        "idx_memory_work_items_lease",
        "idx_memory_stream_events_cursor",
        "idx_memory_source_cursors_scope_last_seen",
    ):
        conn.execute(f"DROP INDEX IF EXISTS {index_name}")
    repair_name = f"{table_name}__repair"
    conn.execute(f"DROP TABLE IF EXISTS {repair_name}")
    conn.execute(create_sql.format(table=repair_name))
    source_columns = existing_columns
    expressions: list[str] = []
    for name, default in columns:
        if name not in source_columns:
            if name == "importance_baseline" and "importance_score" not in source_columns:
                expressions.append("0.0")
            elif name == "cursor":
                expressions.append("rowid")
            else:
                expressions.append(default)
        elif name == "scenario_id":
            expressions.append(f"COALESCE({name}, '{LEGACY_SCENARIO_ID}')")
        else:
            expressions.append(name)
    names = ", ".join(name for name, _ in columns)
    conn.execute(
        f"INSERT INTO {repair_name} ({names}) "
        f"SELECT {', '.join(expressions)} FROM {table_name}"
    )
    conn.execute(f"DROP TABLE {table_name}")
    conn.execute(f"ALTER TABLE {repair_name} RENAME TO {table_name}")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Run the wrapped statements inside one IMMEDIATE write transaction.

    ``BEGIN IMMEDIATE`` takes the write lock up front (no deadlock between a
    later read in the transaction and a competing writer) and rolls back on
    any exception, committing only when every statement succeeded.
    """
    with database_write_lock(conn):
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")


def json_dumps(value: object) -> str:
    """Canonical JSON: sorted keys, compact separators, UTF-8 not escaped."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def now_ms() -> int:
    """Wall-clock time in integer milliseconds (for ``created_at`` columns)."""
    return time.time_ns() // 1_000_000
