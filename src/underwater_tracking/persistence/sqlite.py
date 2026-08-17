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
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 2
_BUSY_TIMEOUT_MS = 60_000

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
)

_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_runtime_events_scenario ON runtime_events(scenario_id, sim_time_s)",
    "CREATE INDEX IF NOT EXISTS idx_runtime_events_type ON runtime_events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_plans_scenario_status ON plans(scenario_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_plan_commands_plan ON plan_commands(plan_id)",
    "CREATE INDEX IF NOT EXISTS idx_decision_records_scenario ON decision_records(scenario_id, sim_time_s)",
    "CREATE INDEX IF NOT EXISTS idx_expert_directives_scenario ON expert_directives(scenario_id)",
    "CREATE INDEX IF NOT EXISTS idx_question_runs_scenario ON question_runs(scenario_id)",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_queries_scenario ON knowledge_queries(scenario_id, sim_time_s)",
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
    conn = sqlite3.connect(
        str(database_path),
        check_same_thread=False,
        isolation_level=None,
        timeout=_BUSY_TIMEOUT_MS / 1000,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Create the agent schema and stamp ``user_version`` (future hook).

    A database stamped newer than :data:`SCHEMA_VERSION` is rejected (the
    code that wrote it is older than the database); an unstamped or older
    database is migrated with idempotent ``CREATE IF NOT EXISTS`` statements
    and re-stamped.
    """
    stored = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if stored > SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema version {stored} is newer than supported"
            f" {SCHEMA_VERSION}; upgrade the code before opening this database"
        )
    if stored < SCHEMA_VERSION:
        for statement in (*_CREATE_TABLES, *_CREATE_INDEXES):
            conn.execute(statement)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Run the wrapped statements inside one IMMEDIATE write transaction.

    ``BEGIN IMMEDIATE`` takes the write lock up front (no deadlock between a
    later read in the transaction and a competing writer) and rolls back on
    any exception, committing only when every statement succeeded.
    """
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
