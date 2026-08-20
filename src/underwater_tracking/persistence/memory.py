"""SQLite repositories for bounded short-term and versioned long-term memory.

The repositories intentionally own their SQLite connections.  A memory worker
can therefore open and close its own repository without sharing a LangGraph
checkpointer connection with the simulation thread.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from underwater_tracking.domain.memory_models import (
    MemoryStatus,
    MemoryStreamStatus,
    MemoryStreamEvent,
    MemoryVersion,
    MemoryWorkItem,
    MemoryWorkStatus,
    ShortTermContext,
    ShortTermMessage,
)
from underwater_tracking.persistence.sqlite import json_dumps, now_ms, open_database, transaction

_MAX_JSON_BYTES = 256 * 1024
_MAX_LIST_LIMIT = 100
_MAX_STREAM_LIMIT = 100


class VersionConflictError(RuntimeError):
    """Raised when an optimistic short-term or long-term version is stale."""


def _validate_user_id(user_id: str) -> str:
    if not isinstance(user_id, str) or not user_id.strip() or len(user_id) > 120:
        raise ValueError("user_id must be a non-blank string no longer than 120 characters")
    return user_id


def _datetime_to_ms(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1000)


def _datetime_from_ms(value: int | None) -> datetime | None:
    return datetime.fromtimestamp(value / 1000, UTC) if value is not None else None


def _bounded_json(value: object, *, label: str) -> str:
    encoded = json_dumps(value)
    if len(encoded.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds {_MAX_JSON_BYTES} bytes")
    return encoded


def _bounded_limit(limit: int, maximum: int = _MAX_LIST_LIMIT) -> int:
    if not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    return max(0, min(limit, maximum))


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _estimate_tokens(messages: Sequence[ShortTermMessage]) -> int:
    """Keep a deterministic, bounded estimate alongside retained messages."""
    return sum((len(message.text) + 3) // 4 for message in messages)


class ShortTermContextRepository:
    """Rolling short-term context scoped by ``(user_id, conversation_id)``."""

    def __init__(self, database_path: str | Path) -> None:
        self._conn = open_database(database_path)

    def close(self) -> None:
        self._conn.close()

    def get_short_term(self, user_id: str, conversation_id: str) -> ShortTermContext | None:
        _validate_user_id(user_id)
        row = self._conn.execute(
            "SELECT * FROM short_term_contexts WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        ).fetchone()
        return self._decode(row) if row is not None else None

    def append_messages(
        self,
        user_id: str,
        conversation_id: str,
        messages: Sequence[ShortTermMessage],
    ) -> ShortTermContext:
        """Append messages atomically and retain only the bounded recent window."""
        _validate_user_id(user_id)
        incoming = tuple(messages)
        if not all(isinstance(message, ShortTermMessage) for message in incoming):
            raise TypeError("messages must contain ShortTermMessage instances")
        with transaction(self._conn):
            existing = self.get_short_term(user_id, conversation_id)
            retained = (existing.recent_messages if existing is not None else ()) + incoming
            retained = retained[-128:]
            updated_at = now_ms()
            if existing is None:
                context = ShortTermContext(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    recent_messages=retained,
                    message_count=len(incoming),
                    estimated_tokens=_estimate_tokens(retained),
                    updated_at=datetime.fromtimestamp(updated_at / 1000, UTC),
                )
                self._insert(context, updated_at)
            else:
                context = existing.model_copy(
                    update={
                        "recent_messages": retained,
                        "message_count": existing.message_count + len(incoming),
                        "estimated_tokens": _estimate_tokens(retained),
                        "updated_at": _datetime_from_ms(updated_at),
                    }
                )
                self._update(context, updated_at)
        return context

    def save_compressed_context(
        self,
        user_id: str,
        conversation_id: str,
        expected_summary_version: int,
        summary: str,
        retained_messages: Sequence[ShortTermMessage],
    ) -> ShortTermContext:
        """Replace a summary only when its caller read the expected version."""
        _validate_user_id(user_id)
        if expected_summary_version < 0:
            raise ValueError("expected_summary_version must be non-negative")
        retained = tuple(retained_messages)[-128:]
        if not all(isinstance(message, ShortTermMessage) for message in retained):
            raise TypeError("retained_messages must contain ShortTermMessage instances")
        updated_at = now_ms()
        with transaction(self._conn):
            existing = self.get_short_term(user_id, conversation_id)
            if existing is None:
                if expected_summary_version != 0:
                    raise VersionConflictError("short-term context does not exist at requested version")
                context = ShortTermContext(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    summary_text=summary,
                    summary_version=1,
                    recent_messages=retained,
                    message_count=len(retained),
                    estimated_tokens=_estimate_tokens(retained),
                    compression_count=1,
                    last_compressed_at=datetime.fromtimestamp(updated_at / 1000, UTC),
                    compression_status=MemoryStreamStatus.COMPLETED,
                    updated_at=datetime.fromtimestamp(updated_at / 1000, UTC),
                )
                self._insert(context, updated_at)
            else:
                if existing.summary_version != expected_summary_version:
                    raise VersionConflictError(
                        f"expected short-term summary version {expected_summary_version}, "
                        f"found {existing.summary_version}"
                    )
                context = existing.model_copy(
                    update={
                        "summary_text": summary,
                        "summary_version": existing.summary_version + 1,
                        "recent_messages": retained,
                        "estimated_tokens": _estimate_tokens(retained),
                        "compression_count": existing.compression_count + 1,
                        "last_compressed_at": _datetime_from_ms(updated_at),
                        "compression_status": MemoryStreamStatus.COMPLETED,
                        "updated_at": datetime.fromtimestamp(updated_at / 1000, UTC),
                    }
                )
                self._update(context, updated_at)
        return context

    def _insert(self, context: ShortTermContext, updated_at: int) -> None:
        self._conn.execute(
            "INSERT INTO short_term_contexts"
            " (user_id, conversation_id, summary_text, summary_version, recent_messages,"
            "  message_count, estimated_tokens, compression_count, last_compressed_at,"
            "  compression_status, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self._context_values(context, updated_at),
        )

    def _update(self, context: ShortTermContext, updated_at: int) -> None:
        cursor = self._conn.execute(
            "UPDATE short_term_contexts SET summary_text = ?, summary_version = ?,"
            " recent_messages = ?, message_count = ?, estimated_tokens = ?,"
            " compression_count = ?, last_compressed_at = ?, compression_status = ?,"
            " updated_at = ? WHERE user_id = ? AND conversation_id = ?",
            (
                context.summary_text,
                context.summary_version,
                _bounded_json(
                    [message.model_dump(mode="json") for message in context.recent_messages],
                    label="recent_messages",
                ),
                context.message_count,
                context.estimated_tokens,
                context.compression_count,
                _datetime_to_ms(context.last_compressed_at)
                if context.last_compressed_at is not None
                else None,
                _enum_value(context.compression_status),
                updated_at,
                context.user_id,
                context.conversation_id,
            ),
        )
        if cursor.rowcount != 1:
            raise VersionConflictError("short-term context disappeared during update")

    @staticmethod
    def _context_values(context: ShortTermContext, updated_at: int) -> tuple[object, ...]:
        return (
            context.user_id,
            context.conversation_id,
            context.summary_text,
            context.summary_version,
            _bounded_json(
                [message.model_dump(mode="json") for message in context.recent_messages],
                label="recent_messages",
            ),
            context.message_count,
            context.estimated_tokens,
            context.compression_count,
            _datetime_to_ms(context.last_compressed_at)
            if context.last_compressed_at is not None
            else None,
            _enum_value(context.compression_status),
            updated_at,
        )

    @staticmethod
    def _decode(row: sqlite3.Row) -> ShortTermContext:
        return ShortTermContext.model_validate(
            {
                "user_id": row["user_id"],
                "conversation_id": row["conversation_id"],
                "summary_text": row["summary_text"],
                "summary_version": row["summary_version"],
                "recent_messages": json.loads(row["recent_messages"]),
                "message_count": row["message_count"],
                "estimated_tokens": row["estimated_tokens"],
                "compression_count": row["compression_count"],
                "last_compressed_at": _datetime_from_ms(row["last_compressed_at"]),
                "compression_status": row["compression_status"],
                "updated_at": _datetime_from_ms(row["updated_at"]),
            }
        )


class LongTermMemoryRepository:
    """Versioned long-term memory plus worker queue, cursors, and stream audit."""

    def __init__(self, database_path: str | Path) -> None:
        self._conn = open_database(database_path)

    def close(self) -> None:
        self._conn.close()

    def create_memory_version(
        self, memory: MemoryVersion, expected_previous_version: int
    ) -> MemoryVersion:
        """Atomically supersede one active version and insert its successor."""
        user_id = _validate_user_id(memory.user_id)
        if expected_previous_version < 0 or memory.version != expected_previous_version + 1:
            raise ValueError("memory.version must be expected_previous_version + 1")
        if memory.status is not MemoryStatus.ACTIVE:
            raise ValueError("new memory versions must start active")
        with transaction(self._conn):
            latest = self._conn.execute(
                "SELECT memory_id, version FROM long_term_memories"
                " WHERE user_id = ? AND memory_family_id = ?"
                " ORDER BY version DESC LIMIT 1",
                (user_id, memory.memory_family_id),
            ).fetchone()
            actual_version = int(latest["version"]) if latest is not None else 0
            if actual_version != expected_previous_version:
                raise VersionConflictError(
                    f"expected previous memory version {expected_previous_version}, "
                    f"found {actual_version}"
                )
            if expected_previous_version == 0:
                if memory.supersedes_memory_id is not None:
                    raise ValueError("initial memory version cannot supersede another version")
            else:
                assert latest is not None
                if memory.supersedes_memory_id != latest["memory_id"]:
                    raise ValueError("new memory version must supersede the latest memory id")
                superseded = self._conn.execute(
                    "UPDATE long_term_memories SET status = 'superseded'"
                    " WHERE memory_id = ? AND user_id = ? AND status = 'active'",
                    (latest["memory_id"], user_id),
                )
                if superseded.rowcount != 1:
                    raise VersionConflictError("latest memory version is no longer active")
            self._insert_memory(memory)
        return memory

    def list_active(
        self,
        user_id: str,
        filters: Mapping[str, object] | None = None,
        limit: int = _MAX_LIST_LIMIT,
    ) -> list[MemoryVersion]:
        _validate_user_id(user_id)
        bounded_limit = _bounded_limit(limit)
        if bounded_limit == 0:
            return []
        clauses = ["user_id = ?", "status = 'active'"]
        params: list[object] = [user_id]
        selected = filters or {}
        memory_type = selected.get("memory_type")
        if memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(getattr(memory_type, "value", memory_type))
        family_id = selected.get("memory_family_id")
        if family_id is not None:
            clauses.append("memory_family_id = ?")
            params.append(family_id)
        min_importance = selected.get("min_importance_score")
        if min_importance is not None:
            clauses.append("importance_score >= ?")
            params.append(min_importance)
        created_after = selected.get("created_after")
        if created_after is not None:
            clauses.append("created_at >= ?")
            params.append(_datetime_to_ms(created_after) if isinstance(created_after, datetime) else created_after)
        created_before = selected.get("created_before")
        if created_before is not None:
            clauses.append("created_at <= ?")
            params.append(_datetime_to_ms(created_before) if isinstance(created_before, datetime) else created_before)
        rows = self._conn.execute(
            "SELECT * FROM long_term_memories WHERE "
            + " AND ".join(clauses)
            + " ORDER BY importance_score DESC, created_at DESC, memory_id DESC LIMIT ?",
            (*params, bounded_limit),
        ).fetchall()
        return [self._decode_memory(row) for row in rows]

    def list_versions(self, user_id: str, memory_family_id: str) -> list[MemoryVersion]:
        _validate_user_id(user_id)
        rows = self._conn.execute(
            "SELECT * FROM long_term_memories WHERE user_id = ? AND memory_family_id = ?"
            " ORDER BY version ASC, memory_id ASC",
            (user_id, memory_family_id),
        ).fetchall()
        return [self._decode_memory(row) for row in rows]

    def mark_deleted(self, user_id: str, memory_id: str) -> bool:
        """Delete a whole logical family without deleting its source audit rows."""
        _validate_user_id(user_id)
        with transaction(self._conn):
            row = self._conn.execute(
                "SELECT memory_family_id FROM long_term_memories WHERE user_id = ? AND memory_id = ?",
                (user_id, memory_id),
            ).fetchone()
            if row is None:
                return False
            self._conn.execute(
                "UPDATE long_term_memories SET status = 'deleted'"
                " WHERE user_id = ? AND memory_family_id = ?",
                (user_id, row["memory_family_id"]),
            )
        return True

    def record_access(self, user_id: str, memory_ids: Sequence[str]) -> int:
        """Increment access metadata only for current active memory versions."""
        _validate_user_id(user_id)
        unique_ids = tuple(dict.fromkeys(memory_ids))
        if not unique_ids:
            return 0
        placeholders = ", ".join("?" for _ in unique_ids)
        with transaction(self._conn):
            cursor = self._conn.execute(
                "UPDATE long_term_memories SET access_count = access_count + 1,"
                " last_accessed_at = ? WHERE user_id = ? AND status = 'active'"
                f" AND memory_id IN ({placeholders})",
                (now_ms(), user_id, *unique_ids),
            )
        return cursor.rowcount

    def enqueue_work(self, item: MemoryWorkItem, source_key: str) -> bool:
        _validate_user_id(item.user_id)
        if item.status is not MemoryWorkStatus.PENDING:
            raise ValueError("enqueued work must start pending")
        if not isinstance(source_key, str) or not source_key or len(source_key) > 500:
            raise ValueError("source_key must be a non-empty string no longer than 500 characters")
        with transaction(self._conn):
            cursor = self._conn.execute(
                "INSERT INTO memory_work_items"
                " (work_id, source_key, user_id, conversation_id, scenario_id, work_type, payload,"
                "  status, attempts, available_at, created_at, completed_at, last_error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT DO NOTHING",
                (
                    item.work_id,
                    source_key,
                    item.user_id,
                    item.conversation_id,
                    item.scenario_id,
                    item.work_type.value,
                    _bounded_json(item.payload.model_dump(mode="json"), label="work payload"),
                    item.status.value,
                    item.attempts,
                    _datetime_to_ms(item.available_at),
                    _datetime_to_ms(item.created_at),
                    _datetime_to_ms(item.completed_at) if item.completed_at is not None else None,
                    item.last_error,
                ),
            )
        return cursor.rowcount == 1

    def claim_work(
        self, worker_id: str, now: datetime, lease_timeout_s: float
    ) -> MemoryWorkItem | None:
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id must be a non-empty string")
        if lease_timeout_s <= 0:
            raise ValueError("lease_timeout_s must be positive")
        now_at = _datetime_to_ms(now)
        lease_expires_at = _datetime_to_ms(now + timedelta(seconds=lease_timeout_s))
        with transaction(self._conn):
            row = self._conn.execute(
                "SELECT work_id FROM memory_work_items WHERE"
                " (status = 'pending' AND available_at <= ?)"
                " OR (status = 'processing' AND lease_expires_at <= ?)"
                " ORDER BY available_at ASC, created_at ASC, work_id ASC LIMIT 1",
                (now_at, now_at),
            ).fetchone()
            if row is None:
                return None
            cursor = self._conn.execute(
                "UPDATE memory_work_items SET status = 'processing', attempts = attempts + 1,"
                " claimed_by = ?, lease_expires_at = ?, completed_at = NULL"
                " WHERE work_id = ? AND ((status = 'pending' AND available_at <= ?)"
                " OR (status = 'processing' AND lease_expires_at <= ?))",
                (worker_id, lease_expires_at, row["work_id"], now_at, now_at),
            )
            if cursor.rowcount != 1:
                return None
            claimed = self._conn.execute(
                "SELECT * FROM memory_work_items WHERE work_id = ?", (row["work_id"],)
            ).fetchone()
        return self._decode_work(claimed) if claimed is not None else None

    def complete_work(self, work_id: str, worker_id: str) -> bool:
        with transaction(self._conn):
            cursor = self._conn.execute(
                "UPDATE memory_work_items SET status = 'completed', completed_at = ?,"
                " claimed_by = NULL, lease_expires_at = NULL"
                " WHERE work_id = ? AND status = 'processing' AND claimed_by = ?",
                (now_ms(), work_id, worker_id),
            )
        return cursor.rowcount == 1

    def fail_work(
        self,
        work_id: str,
        worker_id: str,
        status: MemoryWorkStatus,
        error: str,
        retry_at: datetime | None,
    ) -> bool:
        if status not in {
            MemoryWorkStatus.PENDING,
            MemoryWorkStatus.DEGRADED,
            MemoryWorkStatus.FAILED,
        }:
            raise ValueError("failed work status must be pending, degraded, or failed")
        if not error or len(error) > 1000:
            raise ValueError("error must be between 1 and 1000 characters")
        completed_at = now_ms() if status in {MemoryWorkStatus.DEGRADED, MemoryWorkStatus.FAILED} else None
        with transaction(self._conn):
            cursor = self._conn.execute(
                "UPDATE memory_work_items SET status = ?, last_error = ?, available_at = ?,"
                " completed_at = ?, claimed_by = NULL, lease_expires_at = NULL"
                " WHERE work_id = ? AND status = 'processing' AND claimed_by = ?",
                (
                    status.value,
                    error,
                    _datetime_to_ms(retry_at) if retry_at is not None else now_ms(),
                    completed_at,
                    work_id,
                    worker_id,
                ),
            )
        return cursor.rowcount == 1

    def get_source_cursor(self, user_id: str, scenario_id: str, source_type: str) -> int:
        _validate_user_id(user_id)
        row = self._conn.execute(
            "SELECT source_cursor FROM memory_source_cursors"
            " WHERE user_id = ? AND scenario_id = ? AND source_type = ?",
            (user_id, scenario_id, source_type),
        ).fetchone()
        return int(row["source_cursor"]) if row is not None else 0

    def advance_source_cursor(
        self, user_id: str, scenario_id: str, source_type: str, source_cursor: int
    ) -> int:
        _validate_user_id(user_id)
        if not isinstance(source_cursor, int) or source_cursor < 0:
            raise ValueError("source_cursor must be a non-negative integer")
        with transaction(self._conn):
            self._conn.execute(
                "INSERT INTO memory_source_cursors"
                " (user_id, scenario_id, source_type, source_cursor, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(user_id, scenario_id, source_type) DO UPDATE SET"
                " source_cursor = MAX(memory_source_cursors.source_cursor, excluded.source_cursor),"
                " updated_at = excluded.updated_at",
                (user_id, scenario_id, source_type, source_cursor, now_ms()),
            )
            advanced = self.get_source_cursor(user_id, scenario_id, source_type)
        return advanced

    def append_stream_event(self, event: MemoryStreamEvent) -> MemoryStreamEvent:
        _validate_user_id(event.user_id)
        with transaction(self._conn):
            cursor = self._conn.execute(
                "INSERT INTO memory_stream_events"
                " (event_id, user_id, conversation_id, status, type, payload, memory_id,"
                "  memory_family_id, version, created_at, sim_time_s)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.user_id,
                    event.conversation_id,
                    event.status.value,
                    event.type.value,
                    _bounded_json(event.payload.model_dump(mode="json"), label="stream payload"),
                    event.memory_id,
                    event.memory_family_id,
                    event.version,
                    _datetime_to_ms(event.created_at),
                    event.sim_time_s,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM memory_stream_events WHERE cursor = ?", (cursor.lastrowid,)
            ).fetchone()
        assert row is not None
        return self._decode_stream(row)

    def list_stream_events(
        self, user_id: str, conversation_id: str, *, after_cursor: int = 0, limit: int = _MAX_STREAM_LIMIT
    ) -> list[MemoryStreamEvent]:
        _validate_user_id(user_id)
        if not isinstance(after_cursor, int) or after_cursor < 0:
            raise ValueError("after_cursor must be a non-negative integer")
        bounded_limit = _bounded_limit(limit, _MAX_STREAM_LIMIT)
        if bounded_limit == 0:
            return []
        rows = self._conn.execute(
            "SELECT * FROM memory_stream_events WHERE user_id = ? AND conversation_id = ?"
            " AND cursor > ? ORDER BY cursor ASC LIMIT ?",
            (user_id, conversation_id, after_cursor, bounded_limit),
        ).fetchall()
        return [self._decode_stream(row) for row in rows]

    def _insert_memory(self, memory: MemoryVersion) -> None:
        self._conn.execute(
            "INSERT INTO long_term_memories"
            " (memory_id, memory_family_id, version, user_id, memory_type, summary, importance_score,"
            "  embedding, embedding_version, status, supersedes_memory_id, source_message_ids,"
            "  source_event_ids, source_decision_ids, source_knowledge_ids, change_reason, created_at,"
            "  last_accessed_at, access_count, sim_time_s)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                memory.memory_id,
                memory.memory_family_id,
                memory.version,
                memory.user_id,
                memory.memory_type.value,
                memory.summary,
                memory.importance_score,
                _bounded_json(list(memory.embedding), label="embedding"),
                memory.embedding_version,
                memory.status.value,
                memory.supersedes_memory_id,
                _bounded_json(list(memory.source_message_ids), label="source_message_ids"),
                _bounded_json(list(memory.source_event_ids), label="source_event_ids"),
                _bounded_json(list(memory.source_decision_ids), label="source_decision_ids"),
                _bounded_json(list(memory.source_knowledge_ids), label="source_knowledge_ids"),
                memory.change_reason,
                _datetime_to_ms(memory.created_at),
                _datetime_to_ms(memory.last_accessed_at)
                if memory.last_accessed_at is not None
                else None,
                memory.access_count,
                memory.sim_time_s,
            ),
        )

    @staticmethod
    def _decode_memory(row: sqlite3.Row) -> MemoryVersion:
        return MemoryVersion.model_validate(
            {
                "memory_id": row["memory_id"],
                "memory_family_id": row["memory_family_id"],
                "version": row["version"],
                "user_id": row["user_id"],
                "memory_type": row["memory_type"],
                "summary": row["summary"],
                "importance_score": row["importance_score"],
                "embedding": json.loads(row["embedding"]),
                "embedding_version": row["embedding_version"],
                "status": row["status"],
                "supersedes_memory_id": row["supersedes_memory_id"],
                "source_message_ids": json.loads(row["source_message_ids"]),
                "source_event_ids": json.loads(row["source_event_ids"]),
                "source_decision_ids": json.loads(row["source_decision_ids"]),
                "source_knowledge_ids": json.loads(row["source_knowledge_ids"]),
                "change_reason": row["change_reason"],
                "created_at": _datetime_from_ms(row["created_at"]),
                "last_accessed_at": _datetime_from_ms(row["last_accessed_at"]),
                "access_count": row["access_count"],
                "sim_time_s": row["sim_time_s"],
            }
        )

    @staticmethod
    def _decode_work(row: sqlite3.Row) -> MemoryWorkItem:
        return MemoryWorkItem.model_validate(
            {
                "work_id": row["work_id"],
                "user_id": row["user_id"],
                "conversation_id": row["conversation_id"],
                "scenario_id": row["scenario_id"],
                "work_type": row["work_type"],
                "payload": json.loads(row["payload"]),
                "status": row["status"],
                "attempts": row["attempts"],
                "available_at": _datetime_from_ms(row["available_at"]),
                "created_at": _datetime_from_ms(row["created_at"]),
                "completed_at": _datetime_from_ms(row["completed_at"]),
                "last_error": row["last_error"],
            }
        )

    @staticmethod
    def _decode_stream(row: sqlite3.Row) -> MemoryStreamEvent:
        return MemoryStreamEvent.model_validate(
            {
                "cursor": row["cursor"],
                "event_id": row["event_id"],
                "user_id": row["user_id"],
                "conversation_id": row["conversation_id"],
                "status": row["status"],
                "type": row["type"],
                "payload": json.loads(row["payload"]),
                "memory_id": row["memory_id"],
                "memory_family_id": row["memory_family_id"],
                "version": row["version"],
                "created_at": _datetime_from_ms(row["created_at"]),
                "sim_time_s": row["sim_time_s"],
            }
        )
