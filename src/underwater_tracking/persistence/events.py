# src/underwater_tracking/persistence/events.py
"""Append-only runtime event store over SQLite.

Raw observations and carrier events are written to ``runtime_events``
(spec 8.4: high-frequency raw observations live in the EventStore, not the
graph checkpoint) and replayed in insertion order via ``list_events``.
Payloads are canonical JSON with integer simulation timestamps.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from underwater_tracking.persistence.sqlite import json_dumps, now_ms, open_database, transaction

_DEFAULT_LIMIT = 1000


@dataclass(frozen=True)
class StoredEvent:
    """One persisted runtime event row, read back in replay order."""

    id: int
    event_id: str
    event_type: str
    scenario_id: str
    target_id: str | None
    sim_time_s: int
    severity: str
    payload: dict[str, Any]
    created_at: int


class EventRepository:
    """Append-only repository for raw observations and carrier events."""

    def __init__(self, database_path: str | Path) -> None:
        self._conn = open_database(database_path)

    def close(self) -> None:
        self._conn.close()

    def append(
        self,
        *,
        event_id: str,
        event_type: str,
        scenario_id: str,
        sim_time_s: int,
        payload: dict[str, Any],
        target_id: str | None = None,
        severity: str = "info",
    ) -> int:
        """Append one event and return its monotonically increasing row id.

        Each append is its own immediate transaction: the insert is
        atomic and durable before this method returns.
        """
        with transaction(self._conn):
            cursor = self._conn.execute(
                "INSERT INTO runtime_events"
                " (event_id, event_type, scenario_id, target_id, sim_time_s,"
                "  severity, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    event_type,
                    scenario_id,
                    target_id,
                    sim_time_s,
                    severity,
                    json_dumps(payload),
                    now_ms(),
                ),
            )
        return int(cursor.lastrowid or 0)

    def get(self, event_id: str) -> StoredEvent | None:
        """Return the stored event with this unique ``event_id`` (or None).

        ``runtime_events.event_id`` is unique, so the lookup is
        unambiguous; this is the evidence-id retrieval path for summary
        lookups and expert questions (spec 9, 10.2).
        """
        row = self._conn.execute(
            "SELECT id, event_id, event_type, scenario_id, target_id, sim_time_s,"
            " severity, payload, created_at FROM runtime_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return self._decode(row) if row is not None else None

    def list_events(
        self,
        *,
        scenario_id: str | None = None,
        event_type: str | None = None,
        target_id: str | None = None,
        since_id: int = 0,
        limit: int = _DEFAULT_LIMIT,
    ) -> list[StoredEvent]:
        """Return matching events in insertion order, optionally after ``since_id``."""
        clauses = ["id > ?"]
        params: list[object] = [since_id]
        if scenario_id is not None:
            clauses.append("scenario_id = ?")
            params.append(scenario_id)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        if target_id is not None:
            clauses.append("target_id = ?")
            params.append(target_id)
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT id, event_id, event_type, scenario_id, target_id, sim_time_s,"
            f" severity, payload, created_at FROM runtime_events"
            f" WHERE {' AND '.join(clauses)} ORDER BY id LIMIT ?",
            params,
        ).fetchall()
        return [self._decode(row) for row in rows]

    def _decode(self, row: sqlite3.Row) -> StoredEvent:
        return StoredEvent(
            id=int(row["id"]),
            event_id=row["event_id"],
            event_type=row["event_type"],
            scenario_id=row["scenario_id"],
            target_id=row["target_id"],
            sim_time_s=int(row["sim_time_s"]),
            severity=row["severity"],
            payload=json.loads(row["payload"]),
            created_at=int(row["created_at"]),
        )
