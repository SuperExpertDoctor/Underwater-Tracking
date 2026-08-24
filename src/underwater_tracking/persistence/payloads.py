"""Durable, bounded mapping for LangGraph payload references."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
import sqlite3
from threading import RLock
import time
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from underwater_tracking.persistence.checkpoints import (
    _ALLOWED_MSGPACK_MODULES,
    _open_factory_conn,
)
from underwater_tracking.persistence.sqlite import database_write_lock


class RuntimePayloadStore(MutableMapping[str, Any]):
    """Persist graph payloads while keeping hot references in a small cache."""

    def __init__(
        self,
        database_path: str,
        *,
        owner: str,
        cache_limit: int = 64,
        database_limit: int = 256,
    ) -> None:
        if cache_limit < 1:
            raise ValueError("cache_limit must be positive")
        if database_limit < 1:
            raise ValueError("database_limit must be positive")
        self._owner = owner
        self._cache_limit = cache_limit
        self._database_limit = database_limit
        self._conn = _open_factory_conn(database_path)
        self._lock = RLock()
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._serde = JsonPlusSerializer(
            allowed_msgpack_modules=(*_ALLOWED_MSGPACK_MODULES,)
        )
        with database_write_lock(self._conn):
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_payloads (
                    owner TEXT NOT NULL,
                    ref TEXT NOT NULL,
                    payload_type TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    updated_at_ns INTEGER NOT NULL,
                    PRIMARY KEY (owner, ref)
                )
                """
            )

    def __getitem__(self, ref: str) -> Any:
        with self._lock:
            cached = self._cache.get(ref)
            if cached is not None:
                self._cache.move_to_end(ref)
                return cached
            row = self._conn.execute(
                """
                SELECT payload_type, payload
                FROM runtime_payloads
                WHERE owner = ? AND ref = ?
                """,
                (self._owner, ref),
            ).fetchone()
            if row is None:
                raise KeyError(ref)
            value = self._serde.loads_typed((row[0], bytes(row[1])))
            self._cache[ref] = value
            self._cache.move_to_end(ref)
            self._trim_cache()
            return value

    def __setitem__(self, ref: str, value: Any) -> None:
        payload_type, payload = self._serde.dumps_typed(value)
        with self._lock:
            with database_write_lock(self._conn):
                self._conn.execute(
                    """
                    INSERT INTO runtime_payloads
                        (owner, ref, payload_type, payload, updated_at_ns)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(owner, ref) DO UPDATE SET
                        payload_type = excluded.payload_type,
                        payload = excluded.payload,
                        updated_at_ns = excluded.updated_at_ns
                    """,
                    (
                        self._owner,
                        ref,
                        payload_type,
                        sqlite3.Binary(payload),
                        time.time_ns(),
                    ),
                )
                self._cache[ref] = value
                self._cache.move_to_end(ref)
                self._trim_cache()
                self._prune_database()

    def __delitem__(self, ref: str) -> None:
        with self._lock:
            with database_write_lock(self._conn):
                cursor = self._conn.execute(
                    "DELETE FROM runtime_payloads WHERE owner = ? AND ref = ?",
                    (self._owner, ref),
                )
                if cursor.rowcount == 0:
                    raise KeyError(ref)
                self._cache.pop(ref, None)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ref FROM runtime_payloads WHERE owner = ? ORDER BY updated_at_ns",
                (self._owner,),
            ).fetchall()
        return iter(tuple(row[0] for row in rows))

    def __len__(self) -> int:
        with self._lock:
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM runtime_payloads WHERE owner = ?",
                    (self._owner,),
                ).fetchone()[0]
            )

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None  # type: ignore[assignment]

    def _trim_cache(self) -> None:
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)

    def _prune_database(self) -> None:
        retained = tuple(
            row[0]
            for row in self._conn.execute(
                """
                SELECT ref
                FROM runtime_payloads
                WHERE owner = ?
                ORDER BY updated_at_ns DESC, ref DESC
                LIMIT ?
                """,
                (self._owner, self._database_limit),
            ).fetchall()
        )
        if not retained:
            return
        placeholders = ",".join("?" for _ in retained)
        self._conn.execute(
            f"""
            DELETE FROM runtime_payloads
            WHERE owner = ? AND ref NOT IN ({placeholders})
            """,
            (self._owner, *retained),
        )
        retained_set = set(retained)
        for ref in tuple(self._cache):
            if ref not in retained_set:
                del self._cache[ref]
