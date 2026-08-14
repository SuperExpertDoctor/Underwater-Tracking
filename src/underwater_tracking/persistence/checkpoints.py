# src/underwater_tracking/persistence/checkpoints.py
"""LangGraph checkpoint and long-term-memory factories (spec 8.4).

The carrier graph persists checkpoints to SQLite so runs survive restart and
support recovery, state inspection, and time-travel debugging; the long-term
memory store backs scenario-level summary namespaces and expert requests
sharing one store per scenario. Both factories open their own connection to
the given database file with the same WAL + busy-timeout settings as the
agent repositories, because the checkpointer is the highest-frequency writer
in the runtime and shares the single WAL write lock with EventStore appends
(SqliteSaver has no retry, so a contention failure would crash a graph
step). Graph tests use the in-memory savers; the runtime uses these
SQLite-backed factories.

Import note: ``SqliteSaver`` lives in the ``langgraph-checkpoint-sqlite``
companion package (installed with langgraph 1.2.x); the import path is
``langgraph.checkpoint.sqlite``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

_BUSY_TIMEOUT_MS = 5000


def _open_factory_conn(database_path: str | Path) -> sqlite3.Connection:
    """Open a factory connection in autocommit with WAL and a busy timeout.

    ``busy_timeout`` must be set before ``journal_mode=WAL``: when two
    connections first open the same fresh file, the loser of the WAL-mode
    conversion must wait on the write lock instead of failing instantly at
    the default timeout of zero.
    """
    conn = sqlite3.connect(
        str(database_path), check_same_thread=False, isolation_level=None
    )
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def create_checkpointer(database_path: str | Path) -> SqliteSaver:
    """Open a LangGraph SQLite checkpointer on the given database file."""
    conn = _open_factory_conn(database_path)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def create_store(database_path: str | Path) -> SqliteStore:
    """Open a long-term memory store on the given database file."""
    return SqliteStore(_open_factory_conn(database_path))
