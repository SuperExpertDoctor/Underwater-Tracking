# src/underwater_tracking/persistence/checkpoints.py
"""LangGraph checkpoint and long-term-memory factories (spec 8.4).

The carrier graph persists checkpoints to SQLite so runs survive restart and
support recovery, state inspection, and time-travel debugging; the long-term
memory store backs scenario-level summary namespaces and expert requests
sharing one store per scenario. Both factories open their own connection to
the given database file (WAL mode allows it to coexist with the repository
connections). Graph tests use the in-memory savers; the runtime uses these
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


def create_checkpointer(database_path: str | Path) -> SqliteSaver:
    """Open a LangGraph SQLite checkpointer on the given database file."""
    conn = sqlite3.connect(str(database_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def create_store(database_path: str | Path) -> SqliteStore:
    """Open a long-term memory store on the given database file."""
    conn = sqlite3.connect(
        str(database_path), check_same_thread=False, isolation_level=None
    )
    return SqliteStore(conn)
