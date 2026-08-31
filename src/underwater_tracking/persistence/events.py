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

from underwater_tracking.domain.event_registry import event_audiences
from underwater_tracking.domain.models import DEFAULT_EVENT_AUDIENCES, EventAudience
from underwater_tracking.persistence.sqlite import (
    json_dumps,
    now_ms,
    open_database,
    synchronized_database_method,
    transaction,
)

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
    audiences: frozenset[EventAudience]
    payload: dict[str, Any]
    created_at: int


class EventRepository:
    """Append-only repository for raw observations and carrier events."""

    def __init__(self, database_path: str | Path) -> None:
        self._conn = open_database(database_path)

    def close(self) -> None:
        self._conn.close()

    @synchronized_database_method
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
        audiences: frozenset[EventAudience] | None = None,
    ) -> int:
        """Append one event and return its monotonically increasing row id.

        Each append is its own immediate transaction: the insert is
        atomic and durable before this method returns.
        """
        stored_audiences = _audiences_for(event_type, audiences)
        with transaction(self._conn):
            cursor = self._conn.execute(
                "INSERT INTO runtime_events"
                " (event_id, event_type, scenario_id, target_id, sim_time_s,"
                "  severity, audiences_json, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    event_type,
                    scenario_id,
                    target_id,
                    sim_time_s,
                    severity,
                    json_dumps(sorted(audience.value for audience in stored_audiences)),
                    json_dumps(payload),
                    now_ms(),
                ),
            )
        return int(cursor.lastrowid or 0)

    @synchronized_database_method
    def append_if_absent(
        self,
        *,
        event_id: str,
        event_type: str,
        scenario_id: str,
        sim_time_s: int,
        payload: dict[str, Any],
        target_id: str | None = None,
        severity: str = "info",
        audiences: frozenset[EventAudience] | None = None,
    ) -> int | None:
        """Append one event unless its stable ID already exists."""
        stored_audiences = _audiences_for(event_type, audiences)
        with transaction(self._conn):
            cursor = self._conn.execute(
                "INSERT INTO runtime_events"
                " (event_id, event_type, scenario_id, target_id, sim_time_s,"
                "  severity, audiences_json, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(event_id) DO NOTHING",
                (
                    event_id,
                    event_type,
                    scenario_id,
                    target_id,
                    sim_time_s,
                    severity,
                    json_dumps(sorted(audience.value for audience in stored_audiences)),
                    json_dumps(payload),
                    now_ms(),
                ),
            )
        return int(cursor.lastrowid or 0) if cursor.rowcount == 1 else None

    @synchronized_database_method
    def get(self, event_id: str) -> StoredEvent | None:
        """Return the stored event with this unique ``event_id`` (or None).

        ``runtime_events.event_id`` is unique, so the lookup is
        unambiguous; this is the evidence-id retrieval path for summary
        lookups and expert questions (spec 9, 10.2).
        """
        row = self._conn.execute(
            "SELECT id, event_id, event_type, scenario_id, target_id, sim_time_s,"
            " severity, audiences_json, payload, created_at FROM runtime_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return self._decode(row) if row is not None else None

    @synchronized_database_method
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
            f" severity, audiences_json, payload, created_at FROM runtime_events"
            f" WHERE {' AND '.join(clauses)} ORDER BY id LIMIT ?",
            params,
        ).fetchall()
        return [self._decode(row) for row in rows]

    @synchronized_database_method
    def list_scenario_ids(self, limit: int = 100, *, offset: int = 0) -> tuple[str, ...]:
        """Return a bounded set of scenarios that have persisted events."""
        bounded_limit = max(0, min(limit, 100))
        bounded_offset = max(0, offset)
        if bounded_limit == 0:
            return ()
        rows = self._conn.execute(
            "SELECT DISTINCT scenario_id FROM runtime_events ORDER BY scenario_id LIMIT ? OFFSET ?",
            (bounded_limit, bounded_offset),
        ).fetchall()
        return tuple(row["scenario_id"] for row in rows)

    def _decode(self, row: sqlite3.Row) -> StoredEvent:
        return StoredEvent(
            id=int(row["id"]),
            event_id=row["event_id"],
            event_type=row["event_type"],
            scenario_id=row["scenario_id"],
            target_id=row["target_id"],
            sim_time_s=int(row["sim_time_s"]),
            severity=row["severity"],
            audiences=_decode_audiences(row["audiences_json"], row["event_type"]),
            payload=json.loads(row["payload"]),
            created_at=int(row["created_at"]),
        )


def _audiences_for(
    event_type: str, audiences: frozenset[EventAudience] | None
) -> frozenset[EventAudience]:
    try:
        registered = event_audiences(event_type)
    except ValueError:
        registered = None
    if audiences is not None:
        requested = frozenset(audiences)
        if not requested:
            raise ValueError("event audiences must not be empty")
        if registered is not None and requested != registered:
            raise ValueError(
                f"event audiences do not match the registry for {event_type!r}"
            )
        return requested
    return registered if registered is not None else DEFAULT_EVENT_AUDIENCES


def _decode_audiences(value: object, event_type: str) -> frozenset[EventAudience]:
    try:
        values = json.loads(str(value))
        audiences = frozenset(EventAudience(item) for item in values)
        if audiences:
            return audiences
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return _audiences_for(event_type, None)
