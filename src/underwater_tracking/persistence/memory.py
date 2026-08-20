"""SQLite repositories for bounded short-term and versioned long-term memory.

The repositories intentionally own their SQLite connections.  A memory worker
can therefore open and close its own repository without sharing a LangGraph
checkpointer connection with the simulation thread.
"""

from __future__ import annotations

import json
import math
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
from underwater_tracking.persistence.sqlite import (
    LEGACY_SCENARIO_ID,
    json_dumps,
    now_ms,
    open_database,
    transaction,
)

_MAX_JSON_BYTES = 256 * 1024
_MAX_EMBEDDING_JSON_BYTES = 512 * 1024
_MAX_LIST_LIMIT = 100
_MAX_STREAM_LIMIT = 100
_DEFAULT_MAX_ATTEMPTS = 3
_SOURCE_SCOPE_TYPE = "__scenario_scope__"


class VersionConflictError(RuntimeError):
    """Raised when an optimistic short-term or long-term version is stale."""


def _validate_user_id(user_id: str) -> str:
    if not isinstance(user_id, str) or not user_id.strip() or len(user_id) > 120:
        raise ValueError("user_id must be a non-blank string no longer than 120 characters")
    return user_id


def _scenario_key(scenario_id: str | None) -> str:
    if scenario_id is None:
        return LEGACY_SCENARIO_ID
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError("scenario_id must be a non-blank string")
    return scenario_id


def _datetime_to_ms(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1000)


def _datetime_from_ms(value: int | None) -> datetime | None:
    return datetime.fromtimestamp(value / 1000, UTC) if value is not None else None


def _bounded_json(value: object, *, label: str, maximum_bytes: int = _MAX_JSON_BYTES) -> str:
    encoded = json_dumps(value)
    if len(encoded.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{label} exceeds {maximum_bytes} bytes")
    return encoded


def _bounded_limit(limit: int, maximum: int = _MAX_LIST_LIMIT) -> int:
    if not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    return max(0, min(limit, maximum))


def _validate_max_attempts(max_attempts: int) -> int:
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    return max_attempts


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _estimate_tokens(messages: Sequence[ShortTermMessage]) -> int:
    """Keep a deterministic, bounded estimate alongside retained messages."""
    return sum((len(message.text) + 3) // 4 for message in messages)


def _normalize_messages_for_scope(
    messages: Sequence[ShortTermMessage], scenario_id: str | None
) -> tuple[ShortTermMessage, ...]:
    normalized: list[ShortTermMessage] = []
    for message in messages:
        normalized.append(
            message
            if message.scenario_id is not None or scenario_id is None
            else message.model_copy(update={"scenario_id": scenario_id})
        )
    return tuple(normalized)


def _append_messages_in_transaction(
    conn: sqlite3.Connection,
    user_id: str,
    conversation_id: str,
    messages: Sequence[ShortTermMessage],
    scenario_id: str | None = None,
) -> ShortTermContext:
    """Append messages using the caller's already-open transaction."""
    messages = _normalize_messages_for_scope(messages, scenario_id)
    row = conn.execute(
        "SELECT * FROM short_term_contexts"
        " WHERE user_id = ? AND scenario_id = ? AND conversation_id = ?",
        (user_id, _scenario_key(scenario_id), conversation_id),
    ).fetchone()
    existing = ShortTermContextRepository._decode(row) if row is not None else None
    existing_messages = existing.recent_messages if existing is not None else ()
    seen_ids = {message.message_id for message in existing_messages}
    unique_incoming: list[ShortTermMessage] = []
    for message in messages:
        if message.message_id not in seen_ids:
            seen_ids.add(message.message_id)
            unique_incoming.append(message)
    if existing is not None and not unique_incoming:
        return existing

    retained = (existing_messages + tuple(unique_incoming))[-128:]
    updated_at = now_ms()
    if existing is None:
        context = ShortTermContext(
            user_id=user_id,
            scenario_id=scenario_id,
            conversation_id=conversation_id,
            recent_messages=retained,
            message_count=len(unique_incoming),
            estimated_tokens=_estimate_tokens(retained),
            updated_at=datetime.fromtimestamp(updated_at / 1000, UTC),
        )
        _insert_short_term_context(conn, context, updated_at)
        return context

    context = existing.model_copy(
        update={
            "recent_messages": retained,
            "message_count": existing.message_count + len(unique_incoming),
            "estimated_tokens": _estimate_tokens(retained),
            "updated_at": _datetime_from_ms(updated_at),
        }
    )
    _update_short_term_context(conn, context, updated_at)
    return context


def _short_term_values(
    context: ShortTermContext, updated_at: int, operation_id: str | None = None
) -> tuple[object, ...]:
    return (
        context.user_id,
        _scenario_key(context.scenario_id),
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
        operation_id,
        updated_at,
    )


def _insert_short_term_context(
    conn: sqlite3.Connection,
    context: ShortTermContext,
    updated_at: int,
    operation_id: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO short_term_contexts"
        " (user_id, scenario_id, conversation_id, summary_text, summary_version, recent_messages,"
        "  message_count, estimated_tokens, compression_count, last_compressed_at,"
        "  compression_status, last_compression_work_id, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        _short_term_values(context, updated_at, operation_id),
    )


def _update_short_term_context(
    conn: sqlite3.Connection,
    context: ShortTermContext,
    updated_at: int,
    operation_id: str | None = None,
) -> None:
    cursor = conn.execute(
        "UPDATE short_term_contexts SET summary_text = ?, summary_version = ?,"
        " recent_messages = ?, message_count = ?, estimated_tokens = ?,"
        " compression_count = ?, last_compressed_at = ?, compression_status = ?,"
        " last_compression_work_id = COALESCE(?, last_compression_work_id), updated_at = ?"
        " WHERE user_id = ? AND scenario_id = ? AND conversation_id = ?",
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
            operation_id,
            updated_at,
            context.user_id,
            _scenario_key(context.scenario_id),
            context.conversation_id,
        ),
    )
    if cursor.rowcount != 1:
        raise VersionConflictError("short-term context disappeared during update")


class ShortTermContextRepository:
    """Rolling short-term context scoped by ``(user_id, scenario_id, conversation_id)``."""

    def __init__(self, database_path: str | Path) -> None:
        self._conn = open_database(database_path)

    def close(self) -> None:
        self._conn.close()

    def get_short_term(
        self, user_id: str, conversation_id: str, scenario_id: str | None = None
    ) -> ShortTermContext | None:
        _validate_user_id(user_id)
        row = self._conn.execute(
            "SELECT * FROM short_term_contexts"
            " WHERE user_id = ? AND scenario_id = ? AND conversation_id = ?",
            (user_id, _scenario_key(scenario_id), conversation_id),
        ).fetchone()
        return self._decode(row) if row is not None else None

    def append_messages(
        self,
        user_id: str,
        conversation_id: str,
        messages: Sequence[ShortTermMessage],
        *,
        scenario_id: str | None = None,
    ) -> ShortTermContext:
        """Append messages atomically and retain only the bounded recent window."""
        _validate_user_id(user_id)
        incoming = tuple(messages)
        if not all(isinstance(message, ShortTermMessage) for message in incoming):
            raise TypeError("messages must contain ShortTermMessage instances")
        with transaction(self._conn):
            context = _append_messages_in_transaction(
                self._conn, user_id, conversation_id, incoming, scenario_id
            )
        return context

    def save_compressed_context(
        self,
        user_id: str,
        conversation_id: str,
        expected_summary_version: int,
        summary: str,
        retained_messages: Sequence[ShortTermMessage],
        operation_id: str | None = None,
        *,
        scenario_id: str | None = None,
    ) -> ShortTermContext:
        """Replace a summary only when its caller read the expected version."""
        _validate_user_id(user_id)
        if expected_summary_version < 0:
            raise ValueError("expected_summary_version must be non-negative")
        retained = tuple(retained_messages)[-128:]
        if not all(isinstance(message, ShortTermMessage) for message in retained):
            raise TypeError("retained_messages must contain ShortTermMessage instances")
        retained = _normalize_messages_for_scope(retained, scenario_id)
        updated_at = now_ms()
        with transaction(self._conn):
            existing = self.get_short_term(user_id, conversation_id, scenario_id)
            if (
                existing is not None
                and operation_id is not None
                and self._last_compression_work_id(user_id, conversation_id, scenario_id) == operation_id
            ):
                return existing
            if existing is None:
                if expected_summary_version != 0:
                    raise VersionConflictError("short-term context does not exist at requested version")
                context = ShortTermContext(
                    user_id=user_id,
                    scenario_id=scenario_id,
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
                self._insert(context, updated_at, operation_id)
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
                self._update(context, updated_at, operation_id)
        return context

    def get_messages(
        self,
        user_id: str,
        conversation_id: str,
        message_ids: Sequence[str],
        *,
        scenario_id: str | None = None,
    ) -> tuple[ShortTermMessage, ...]:
        """Return exactly the retained messages named by a work item."""
        context = self.get_short_term(user_id, conversation_id, scenario_id)
        if context is None:
            return ()
        by_id = {message.message_id: message for message in context.recent_messages}
        return tuple(
            by_id[message_id]
            for message_id in dict.fromkeys(message_ids)
            if message_id in by_id
            and (scenario_id is None or by_id[message_id].scenario_id == scenario_id)
        )

    def _last_compression_work_id(
        self, user_id: str, conversation_id: str, scenario_id: str | None = None
    ) -> str | None:
        row = self._conn.execute(
            "SELECT last_compression_work_id FROM short_term_contexts"
            " WHERE user_id = ? AND scenario_id = ? AND conversation_id = ?",
            (user_id, _scenario_key(scenario_id), conversation_id),
        ).fetchone()
        return row["last_compression_work_id"] if row is not None else None

    def _insert(
        self, context: ShortTermContext, updated_at: int, operation_id: str | None = None
    ) -> None:
        _insert_short_term_context(self._conn, context, updated_at, operation_id)

    def _update(
        self, context: ShortTermContext, updated_at: int, operation_id: str | None = None
    ) -> None:
        _update_short_term_context(self._conn, context, updated_at, operation_id)

    @staticmethod
    def _context_values(
        context: ShortTermContext, updated_at: int, operation_id: str | None = None
    ) -> tuple[object, ...]:
        return _short_term_values(context, updated_at, operation_id)

    @staticmethod
    def _decode(row: sqlite3.Row) -> ShortTermContext:
        scenario_id = None if row["scenario_id"] == LEGACY_SCENARIO_ID else row["scenario_id"]
        messages = _normalize_messages_for_scope(
            tuple(ShortTermMessage.model_validate(item) for item in json.loads(row["recent_messages"])),
            scenario_id,
        )
        return ShortTermContext.model_validate(
            {
                "user_id": row["user_id"],
                "scenario_id": scenario_id,
                "conversation_id": row["conversation_id"],
                "summary_text": row["summary_text"],
                "summary_version": row["summary_version"],
                "recent_messages": messages,
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
        self,
        memory: MemoryVersion,
        expected_previous_version: int,
        *,
        work_id: str | None = None,
    ) -> MemoryVersion:
        """Atomically supersede one active version and insert its successor."""
        user_id = _validate_user_id(memory.user_id)
        scenario_key = _scenario_key(memory.scenario_id)
        if expected_previous_version < 0 or memory.version != expected_previous_version + 1:
            raise ValueError("memory.version must be expected_previous_version + 1")
        if memory.status is not MemoryStatus.ACTIVE:
            raise ValueError("new memory versions must start active")
        with transaction(self._conn):
            if work_id is not None:
                existing_work = self._conn.execute(
                    "SELECT * FROM long_term_memories"
                    " WHERE user_id = ? AND scenario_id = ? AND memory_work_id = ?",
                    (user_id, scenario_key, work_id),
                ).fetchone()
                if existing_work is not None:
                    return self._decode_memory(existing_work)
            latest = self._conn.execute(
                "SELECT memory_id, version FROM long_term_memories"
                " WHERE user_id = ? AND memory_family_id = ?"
                " AND scenario_id = ?"
                " ORDER BY version DESC LIMIT 1",
                (user_id, memory.memory_family_id, scenario_key),
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
            self._insert_memory(memory, work_id)
        return memory

    def get_memory_for_work(
        self, user_id: str, work_id: str, scenario_id: str | None = None
    ) -> MemoryVersion | None:
        _validate_user_id(user_id)
        scenario_clause = ""
        params: tuple[object, ...] = (user_id, work_id)
        if scenario_id is not None:
            scenario_clause = " AND scenario_id = ?"
            params += (_scenario_key(scenario_id),)
        row = self._conn.execute(
            "SELECT * FROM long_term_memories WHERE user_id = ? AND memory_work_id = ?"
            + scenario_clause,
            params,
        ).fetchone()
        return self._decode_memory(row) if row is not None else None

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
        scenario_id = selected.get("scenario_id")
        if scenario_id is not None:
            if not isinstance(scenario_id, str):
                raise ValueError("scenario_id must be a string")
            clauses.append("scenario_id = ?")
            params.append(_scenario_key(scenario_id))
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

    def list_versions(
        self, user_id: str, memory_family_id: str, scenario_id: str | None = None
    ) -> list[MemoryVersion]:
        _validate_user_id(user_id)
        scenario_clause = ""
        params: tuple[object, ...] = (user_id, memory_family_id)
        if scenario_id is not None:
            scenario_clause = " AND scenario_id = ?"
            params += (_scenario_key(scenario_id),)
        rows = self._conn.execute(
            "SELECT * FROM long_term_memories WHERE user_id = ? AND memory_family_id = ?"
            + scenario_clause
            + " ORDER BY version ASC, memory_id ASC",
            params,
        ).fetchall()
        return [self._decode_memory(row) for row in rows]

    def mark_deleted(
        self, user_id: str, memory_id: str, scenario_id: str | None = None
    ) -> bool:
        """Delete a whole logical family without deleting its source audit rows."""
        _validate_user_id(user_id)
        with transaction(self._conn):
            clauses = ["user_id = ?", "memory_id = ?"]
            params: list[object] = [user_id, memory_id]
            if scenario_id is not None:
                clauses.append("scenario_id = ?")
                params.append(_scenario_key(scenario_id))
            row = self._conn.execute(
                "SELECT memory_family_id, scenario_id FROM long_term_memories WHERE "
                + " AND ".join(clauses),
                params,
            ).fetchone()
            if row is None:
                return False
            self._conn.execute(
                "UPDATE long_term_memories SET status = 'deleted'"
                " WHERE user_id = ? AND memory_family_id = ? AND scenario_id = ?",
                (user_id, row["memory_family_id"], row["scenario_id"]),
            )
        return True

    def record_access(
        self, user_id: str, memory_ids: Sequence[str], scenario_id: str | None = None
    ) -> int:
        """Increment access metadata only for current active memory versions."""
        _validate_user_id(user_id)
        unique_ids = tuple(dict.fromkeys(memory_ids))
        if not unique_ids:
            return 0
        placeholders = ", ".join("?" for _ in unique_ids)
        with transaction(self._conn):
            scenario_clause = ""
            extra_params: tuple[object, ...] = ()
            if scenario_id is not None:
                scenario_clause = " AND scenario_id = ?"
                extra_params = (_scenario_key(scenario_id),)
            cursor = self._conn.execute(
                "UPDATE long_term_memories SET access_count = access_count + 1,"
                " last_accessed_at = ? WHERE user_id = ? AND status = 'active'"
                f" AND memory_id IN ({placeholders}){scenario_clause}",
                (now_ms(), user_id, *unique_ids, *extra_params),
            )
        return cursor.rowcount

    def maintain_active(
        self,
        user_id: str,
        now: datetime,
        *,
        decay_half_life_s: float,
        archive_threshold: float,
        limit: int = 32,
        scenario_id: str | None = None,
    ) -> list[tuple[str, MemoryStatus, float]]:
        """Persist deterministic decay and archive decisions for one user."""
        _validate_user_id(user_id)
        if decay_half_life_s <= 0 or not 0 <= archive_threshold <= 1:
            raise ValueError("maintenance thresholds must be within valid ranges")
        bounded_limit = _bounded_limit(limit)
        if bounded_limit == 0:
            return []
        now_at = _datetime_to_ms(now)
        scenario_clause = ""
        params: list[object] = [user_id]
        if scenario_id is not None:
            scenario_clause = " AND scenario_id = ?"
            params.append(_scenario_key(scenario_id))
        rows = self._conn.execute(
            "SELECT memory_id, importance_baseline, created_at, last_accessed_at, access_count"
            " FROM long_term_memories WHERE user_id = ? AND status = 'active'"
            f"{scenario_clause} ORDER BY importance_score ASC, created_at ASC, memory_id ASC LIMIT ?",
            (*params, bounded_limit),
        ).fetchall()
        updates: list[tuple[str, MemoryStatus, float]] = []
        for row in rows:
            reference_at = row["last_accessed_at"] or row["created_at"]
            age_s = max(0.0, (now_at - int(reference_at)) / 1000.0)
            decay = 0.5 ** (age_s / decay_half_life_s)
            frequency = min(math.log1p(int(row["access_count"])) / math.log(11.0), 1.0)
            score = min(1.0, float(row["importance_baseline"]) * decay * (1.0 + 0.2 * frequency))
            status = MemoryStatus.ARCHIVED if score < archive_threshold else MemoryStatus.ACTIVE
            updates.append((row["memory_id"], status, score))
        if not updates:
            return []
        with transaction(self._conn):
            for memory_id, status, score in updates:
                self._conn.execute(
                    "UPDATE long_term_memories SET importance_score = ?, status = ?"
                    " WHERE user_id = ? AND memory_id = ? AND status = 'active'"
                    + (" AND scenario_id = ?" if scenario_id is not None else ""),
                    (score, status.value, user_id, memory_id)
                    + ((_scenario_key(scenario_id),) if scenario_id is not None else ()),
                )
        return updates

    def enqueue_work(self, item: MemoryWorkItem, source_key: str) -> bool:
        _validate_user_id(item.user_id)
        if item.status is not MemoryWorkStatus.PENDING:
            raise ValueError("enqueued work must start pending")
        if not isinstance(source_key, str) or not source_key or len(source_key) > 500:
            raise ValueError("source_key must be a non-empty string no longer than 500 characters")
        with transaction(self._conn):
            cursor = self._insert_work(item, source_key)
            if cursor.rowcount == 1 and item.scenario_id is not None:
                self._register_source_scope(item.user_id, item.scenario_id)
        return cursor.rowcount == 1

    def append_messages_and_enqueue_work(
        self,
        user_id: str,
        conversation_id: str,
        messages: Sequence[ShortTermMessage],
        item: MemoryWorkItem,
        source_key: str,
        *,
        scenario_id: str,
        source_type: str,
    ) -> bool:
        """Atomically persist a conversation window, work item, and cursor."""
        _validate_user_id(user_id)
        if item.user_id != user_id or item.conversation_id != conversation_id:
            raise ValueError("conversation work must match the supplied user and conversation")
        if item.scenario_id != scenario_id:
            raise ValueError("conversation work must match the supplied scenario")
        if item.status is not MemoryWorkStatus.PENDING:
            raise ValueError("enqueued work must start pending")
        if not source_key or len(source_key) > 500:
            raise ValueError("source_key must be a non-empty string no longer than 500 characters")
        incoming = tuple(messages)
        if not all(isinstance(message, ShortTermMessage) for message in incoming):
            raise TypeError("messages must contain ShortTermMessage instances")
        with transaction(self._conn):
            cursor = self._insert_work(item, source_key)
            if cursor.rowcount != 1:
                return False
            context = _append_messages_in_transaction(
                self._conn, user_id, conversation_id, incoming, scenario_id
            )
            if not scenario_id or not source_type:
                raise ValueError("scenario_id and source_type must be non-empty")
            self._register_source_scope(user_id, scenario_id)
            self._upsert_source_cursor(
                user_id, scenario_id, source_type, context.message_count
            )
        return True

    def append_messages_enqueue_work_and_stream_event(
        self,
        user_id: str,
        conversation_id: str,
        messages: Sequence[ShortTermMessage],
        item: MemoryWorkItem,
        source_key: str,
        *,
        scenario_id: str,
        source_type: str,
        event: MemoryStreamEvent,
    ) -> tuple[bool, MemoryStreamEvent | None]:
        """Atomically publish a conversation work item and its queued event."""
        _validate_user_id(user_id)
        if item.user_id != user_id or item.conversation_id != conversation_id:
            raise ValueError("conversation work must match the supplied user and conversation")
        if item.scenario_id != scenario_id:
            raise ValueError("conversation work must match the supplied scenario")
        if event.user_id != user_id or event.conversation_id != conversation_id:
            raise ValueError("queued event must match the supplied user and conversation")
        if event.scenario_id != scenario_id:
            raise ValueError("queued event must match the supplied scenario")
        if item.status is not MemoryWorkStatus.PENDING:
            raise ValueError("enqueued work must start pending")
        if not source_key or len(source_key) > 500:
            raise ValueError("source_key must be a non-empty string no longer than 500 characters")
        incoming = tuple(messages)
        if not all(isinstance(message, ShortTermMessage) for message in incoming):
            raise TypeError("messages must contain ShortTermMessage instances")
        with transaction(self._conn):
            cursor = self._insert_work(item, source_key)
            if cursor.rowcount != 1:
                existing = self.get_work_by_source_key(user_id, source_key)
                existing_event = (
                    self.get_stream_event_for_work(
                        user_id,
                        conversation_id,
                        existing.work_id,
                        existing.scenario_id,
                    )
                    if existing is not None
                    else None
                )
                return False, existing_event
            context = _append_messages_in_transaction(
                self._conn, user_id, conversation_id, incoming, scenario_id
            )
            if not scenario_id or not source_type:
                raise ValueError("scenario_id and source_type must be non-empty")
            self._register_source_scope(user_id, scenario_id)
            self._upsert_source_cursor(
                user_id, scenario_id, source_type, context.message_count
            )
            stream_cursor = self._insert_stream_event(event)
            row = self._conn.execute(
                "SELECT * FROM memory_stream_events WHERE cursor = ?",
                (stream_cursor.lastrowid,),
            ).fetchone()
        assert row is not None
        return True, self._decode_stream(row)

    def enqueue_work_and_advance_cursor(
        self,
        item: MemoryWorkItem,
        source_key: str,
        scenario_id: str,
        source_type: str,
        source_cursor: int,
    ) -> bool:
        """Atomically make a source work item durable and acknowledge its cursor."""
        _validate_user_id(item.user_id)
        if not scenario_id or not source_type:
            raise ValueError("scenario_id and source_type must be non-empty")
        if item.scenario_id != scenario_id:
            raise ValueError("work item must match the supplied scenario")
        if not isinstance(source_cursor, int) or source_cursor < 0:
            raise ValueError("source_cursor must be a non-negative integer")
        with transaction(self._conn):
            cursor = self._insert_work(item, source_key)
            if cursor.rowcount == 1:
                self._register_source_scope(item.user_id, scenario_id)
                self._upsert_source_cursor(item.user_id, scenario_id, source_type, source_cursor)
        return cursor.rowcount == 1

    def _insert_work(self, item: MemoryWorkItem, source_key: str) -> sqlite3.Cursor:
        return self._conn.execute(
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

    def claim_work(
        self,
        worker_id: str,
        now: datetime,
        lease_timeout_s: float,
        *,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    ) -> MemoryWorkItem | None:
        """Claim one retryable item, degrading work after its final allowed attempt."""
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id must be a non-empty string")
        if lease_timeout_s <= 0:
            raise ValueError("lease_timeout_s must be positive")
        max_attempts = _validate_max_attempts(max_attempts)
        now_at = _datetime_to_ms(now)
        lease_expires_at = _datetime_to_ms(now + timedelta(seconds=lease_timeout_s))
        with transaction(self._conn):
            self._conn.execute(
                "UPDATE memory_work_items SET status = 'degraded',"
                " last_error = COALESCE(last_error, 'maximum retry attempts exhausted'),"
                " completed_at = ?, claimed_by = NULL, lease_expires_at = NULL"
                " WHERE attempts >= ? AND (status = 'pending'"
                " OR (status = 'processing' AND lease_expires_at <= ?))",
                (now_at, max_attempts, now_at),
            )
            row = self._conn.execute(
                "SELECT work_id FROM memory_work_items WHERE attempts < ? AND ("
                " (status = 'pending' AND available_at <= ?)"
                " OR (status = 'processing' AND lease_expires_at <= ?))"
                " ORDER BY available_at ASC, created_at ASC, work_id ASC LIMIT 1",
                (max_attempts, now_at, now_at),
            ).fetchone()
            if row is None:
                return None
            cursor = self._conn.execute(
                "UPDATE memory_work_items SET status = 'processing', attempts = attempts + 1,"
                " claimed_by = ?, lease_expires_at = ?, completed_at = NULL"
                " WHERE work_id = ? AND attempts < ? AND ((status = 'pending' AND available_at <= ?)"
                " OR (status = 'processing' AND lease_expires_at <= ?))",
                (worker_id, lease_expires_at, row["work_id"], max_attempts, now_at, now_at),
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

    def get_work(self, work_id: str) -> MemoryWorkItem | None:
        row = self._conn.execute(
            "SELECT * FROM memory_work_items WHERE work_id = ?", (work_id,)
        ).fetchone()
        return self._decode_work(row) if row is not None else None

    def get_work_by_source_key(self, user_id: str, source_key: str) -> MemoryWorkItem | None:
        _validate_user_id(user_id)
        row = self._conn.execute(
            "SELECT * FROM memory_work_items WHERE user_id = ? AND source_key = ?",
            (user_id, source_key),
        ).fetchone()
        return self._decode_work(row) if row is not None else None

    def get_stream_event_for_work(
        self,
        user_id: str,
        conversation_id: str | None,
        work_id: str,
        scenario_id: str | None = None,
    ) -> MemoryStreamEvent | None:
        scenario_clause = ""
        params: tuple[object, ...] = (user_id, conversation_id, work_id)
        if scenario_id is not None:
            scenario_clause = " AND scenario_id = ?"
            params += (_scenario_key(scenario_id),)
        row = self._conn.execute(
            "SELECT * FROM memory_stream_events"
            " WHERE user_id = ? AND conversation_id IS ?"
            " AND json_extract(payload, '$.work_id') = ?"
            + scenario_clause
            + " ORDER BY cursor DESC LIMIT 1",
            params,
        ).fetchone()
        return self._decode_stream(row) if row is not None else None

    def fail_work(
        self,
        work_id: str,
        worker_id: str,
        status: MemoryWorkStatus,
        error: str,
        retry_at: datetime | None,
        *,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    ) -> bool:
        """Record a failure, degrading a requested retry once attempts are exhausted."""
        if status not in {
            MemoryWorkStatus.PENDING,
            MemoryWorkStatus.DEGRADED,
            MemoryWorkStatus.FAILED,
        }:
            raise ValueError("failed work status must be pending, degraded, or failed")
        if not error or len(error) > 1000:
            raise ValueError("error must be between 1 and 1000 characters")
        max_attempts = _validate_max_attempts(max_attempts)
        with transaction(self._conn):
            row = self._conn.execute(
                "SELECT attempts FROM memory_work_items"
                " WHERE work_id = ? AND status = 'processing' AND claimed_by = ?",
                (work_id, worker_id),
            ).fetchone()
            if row is None:
                return False
            final_status = (
                MemoryWorkStatus.DEGRADED
                if status is MemoryWorkStatus.PENDING and row["attempts"] >= max_attempts
                else status
            )
            completed_at = (
                now_ms()
                if final_status in {MemoryWorkStatus.DEGRADED, MemoryWorkStatus.FAILED}
                else None
            )
            cursor = self._conn.execute(
                "UPDATE memory_work_items SET status = ?, last_error = ?, available_at = ?,"
                " completed_at = ?, claimed_by = NULL, lease_expires_at = NULL"
                " WHERE work_id = ? AND status = 'processing' AND claimed_by = ?",
                (
                    final_status.value,
                    error,
                    _datetime_to_ms(retry_at)
                    if retry_at is not None and final_status is MemoryWorkStatus.PENDING
                    else now_ms(),
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

    def get_source_discovery_state(
        self, user_id: str, repository_count: int
    ) -> tuple[int, tuple[int, ...]]:
        """Return the durable round-robin repository index and per-repository offsets."""
        _validate_user_id(user_id)
        if not isinstance(repository_count, int) or repository_count < 0:
            raise ValueError("repository_count must be a non-negative integer")
        if repository_count == 0:
            return 0, ()
        row = self._conn.execute(
            "SELECT repository_index, offsets FROM memory_source_discovery WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return 0, (0,) * repository_count
        try:
            stored_offsets = json.loads(row["offsets"])
        except (TypeError, ValueError) as error:
            raise RuntimeError("memory source discovery state is corrupt") from error
        if not isinstance(stored_offsets, list) or not all(
            isinstance(offset, int) and offset >= 0 for offset in stored_offsets
        ):
            raise RuntimeError("memory source discovery offsets are corrupt")
        offsets = tuple(stored_offsets[:repository_count]) + (0,) * max(
            0, repository_count - len(stored_offsets)
        )
        return int(row["repository_index"]) % repository_count, offsets

    def register_source_scopes_and_advance_discovery(
        self,
        user_id: str,
        scopes: Sequence[tuple[str, str]],
        repository_index: int,
        offsets: Sequence[int],
        *,
        legacy_cursors: Mapping[str, int] | None = None,
    ) -> None:
        """Commit discovered scopes and their continuation as one SQLite transaction."""
        _validate_user_id(user_id)
        if not isinstance(repository_index, int) or repository_index < 0:
            raise ValueError("repository_index must be a non-negative integer")
        if not offsets or any(not isinstance(offset, int) or offset < 0 for offset in offsets):
            raise ValueError("offsets must contain non-negative integers")
        with transaction(self._conn):
            for discovered_user_id, scenario_id in scopes:
                if discovered_user_id != user_id:
                    raise ValueError("discovered scope user_id must match the discovery user")
                _validate_user_id(discovered_user_id)
                if not isinstance(scenario_id, str) or not scenario_id.strip() or len(scenario_id) > 240:
                    raise ValueError("scenario_id must be a non-blank string no longer than 240 characters")
                self._register_source_scope(discovered_user_id, scenario_id)
            self._set_source_discovery_state(user_id, repository_index, offsets)
            for source_type, source_cursor in (legacy_cursors or {}).items():
                self._set_source_cursor(
                    user_id, "__memory_scope_discovery__", source_type, source_cursor
                )

    def claim_source_scope_page(
        self, limit: int = _MAX_LIST_LIMIT
    ) -> tuple[tuple[str, str], ...]:
        """Return and rotate one durable bounded scope page for the next worker round."""
        bounded_limit = _bounded_limit(limit)
        if bounded_limit == 0:
            return ()
        with transaction(self._conn):
            rows = self._conn.execute(
                "SELECT user_id, scenario_id, updated_at FROM memory_source_cursors"
                " WHERE source_type = ? ORDER BY updated_at, user_id, scenario_id LIMIT ?",
                (_SOURCE_SCOPE_TYPE, bounded_limit),
            ).fetchall()
            if not rows:
                return ()
            latest = self._conn.execute(
                "SELECT COALESCE(MAX(updated_at), 0) FROM memory_source_cursors"
                " WHERE source_type = ?",
                (_SOURCE_SCOPE_TYPE,),
            ).fetchone()
            next_updated_at = max(now_ms(), int(latest[0]) + 1 if latest is not None else 0)
            for row in rows:
                self._conn.execute(
                    "UPDATE memory_source_cursors SET updated_at = ?"
                    " WHERE user_id = ? AND scenario_id = ? AND source_type = ?",
                    (next_updated_at, row["user_id"], row["scenario_id"], _SOURCE_SCOPE_TYPE),
                )
        return tuple((row["user_id"], row["scenario_id"]) for row in rows)

    def register_source_scope(self, user_id: str, scenario_id: str) -> None:
        """Persist a bounded source scope independently of queued work."""
        _validate_user_id(user_id)
        if not isinstance(scenario_id, str) or not scenario_id.strip() or len(scenario_id) > 240:
            raise ValueError("scenario_id must be a non-blank string no longer than 240 characters")
        with transaction(self._conn):
            self._register_source_scope(user_id, scenario_id)

    def list_source_scopes(self, limit: int = _MAX_LIST_LIMIT) -> tuple[tuple[str, str], ...]:
        """Return persisted scopes with a hard upper bound."""
        bounded_limit = _bounded_limit(limit)
        if bounded_limit == 0:
            return ()
        rows = self._conn.execute(
            "SELECT user_id, scenario_id FROM memory_source_cursors"
            " WHERE source_type = ? ORDER BY updated_at, user_id, scenario_id LIMIT ?",
            (_SOURCE_SCOPE_TYPE, bounded_limit),
        ).fetchall()
        return tuple((row["user_id"], row["scenario_id"]) for row in rows)

    def advance_source_cursor(
        self, user_id: str, scenario_id: str, source_type: str, source_cursor: int
    ) -> int:
        _validate_user_id(user_id)
        if not isinstance(source_cursor, int) or source_cursor < 0:
            raise ValueError("source_cursor must be a non-negative integer")
        with transaction(self._conn):
            self._upsert_source_cursor(user_id, scenario_id, source_type, source_cursor)
            advanced = self.get_source_cursor(user_id, scenario_id, source_type)
        return advanced

    def set_source_cursor(
        self, user_id: str, scenario_id: str, source_type: str, source_cursor: int
    ) -> int:
        """Persist an exact continuation, including a wrapped discovery offset."""
        _validate_user_id(user_id)
        if not isinstance(source_cursor, int) or source_cursor < 0:
            raise ValueError("source_cursor must be a non-negative integer")
        with transaction(self._conn):
            self._set_source_cursor(user_id, scenario_id, source_type, source_cursor)
        return source_cursor

    def _upsert_source_cursor(
        self, user_id: str, scenario_id: str, source_type: str, source_cursor: int
    ) -> None:
        self._conn.execute(
            "INSERT INTO memory_source_cursors"
            " (user_id, scenario_id, source_type, source_cursor, updated_at)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(user_id, scenario_id, source_type) DO UPDATE SET"
            " source_cursor = MAX(memory_source_cursors.source_cursor, excluded.source_cursor),"
            " updated_at = excluded.updated_at",
            (user_id, scenario_id, source_type, source_cursor, now_ms()),
        )

    def _set_source_cursor(
        self, user_id: str, scenario_id: str, source_type: str, source_cursor: int
    ) -> None:
        self._conn.execute(
            "INSERT INTO memory_source_cursors"
            " (user_id, scenario_id, source_type, source_cursor, updated_at)"
            " VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_id, scenario_id, source_type) DO UPDATE SET"
            " source_cursor = excluded.source_cursor, updated_at = excluded.updated_at",
            (user_id, scenario_id, source_type, source_cursor, now_ms()),
        )

    def _register_source_scope(self, user_id: str, scenario_id: str) -> None:
        latest = self._conn.execute(
            "SELECT COALESCE(MAX(updated_at), 0) FROM memory_source_cursors"
            " WHERE source_type = ?",
            (_SOURCE_SCOPE_TYPE,),
        ).fetchone()
        updated_at = max(now_ms(), int(latest[0]) + 1 if latest is not None else 0)
        self._conn.execute(
            "INSERT INTO memory_source_cursors"
            " (user_id, scenario_id, source_type, source_cursor, updated_at)"
            " VALUES (?, ?, ?, 0, ?) ON CONFLICT(user_id, scenario_id, source_type) DO UPDATE SET"
            " updated_at = excluded.updated_at",
            (user_id, scenario_id, _SOURCE_SCOPE_TYPE, updated_at),
        )

    def _set_source_discovery_state(
        self, user_id: str, repository_index: int, offsets: Sequence[int]
    ) -> None:
        self._conn.execute(
            "INSERT INTO memory_source_discovery"
            " (user_id, repository_index, offsets, updated_at) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(user_id) DO UPDATE SET repository_index = excluded.repository_index,"
            " offsets = excluded.offsets, updated_at = excluded.updated_at",
            (user_id, repository_index, json_dumps(tuple(offsets)), now_ms()),
        )

    def append_stream_event(self, event: MemoryStreamEvent) -> MemoryStreamEvent:
        _validate_user_id(event.user_id)
        with transaction(self._conn):
            cursor = self._insert_stream_event(event)
            row = self._conn.execute(
                "SELECT * FROM memory_stream_events WHERE cursor = ?", (cursor.lastrowid,)
            ).fetchone()
        assert row is not None
        return self._decode_stream(row)

    def _insert_stream_event(self, event: MemoryStreamEvent) -> sqlite3.Cursor:
        return self._conn.execute(
            "INSERT INTO memory_stream_events"
            " (event_id, user_id, scenario_id, conversation_id, status, type, payload, memory_id,"
            "  memory_family_id, version, created_at, sim_time_s)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.user_id,
                _scenario_key(event.scenario_id),
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

    def list_stream_events(
        self,
        user_id: str,
        conversation_id: str,
        *,
        scenario_id: str | None = None,
        after_cursor: int = 0,
        limit: int = _MAX_STREAM_LIMIT,
    ) -> list[MemoryStreamEvent]:
        _validate_user_id(user_id)
        if not isinstance(after_cursor, int) or after_cursor < 0:
            raise ValueError("after_cursor must be a non-negative integer")
        bounded_limit = _bounded_limit(limit, _MAX_STREAM_LIMIT)
        if bounded_limit == 0:
            return []
        rows = self._conn.execute(
            "SELECT * FROM memory_stream_events WHERE user_id = ? AND conversation_id = ?"
            " AND scenario_id = ? AND cursor > ? ORDER BY cursor ASC LIMIT ?",
            (user_id, conversation_id, _scenario_key(scenario_id), after_cursor, bounded_limit),
        ).fetchall()
        return [self._decode_stream(row) for row in rows]

    def _insert_memory(self, memory: MemoryVersion, work_id: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO long_term_memories"
            " (memory_id, memory_work_id, memory_family_id, version, user_id, memory_type, summary,"
            "  scenario_id,"
            "  importance_score, importance_baseline,"
            "  embedding, embedding_version, status, supersedes_memory_id, source_message_ids,"
            "  source_event_ids, source_decision_ids, source_knowledge_ids, source_plan_ids, change_reason, created_at,"
            "  last_accessed_at, access_count, sim_time_s)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                memory.memory_id,
                work_id,
                memory.memory_family_id,
                memory.version,
                memory.user_id,
                memory.memory_type.value,
                memory.summary,
                _scenario_key(memory.scenario_id),
                memory.importance_score,
                memory.importance_score,
                _bounded_json(
                    list(memory.embedding),
                    label="embedding",
                    maximum_bytes=_MAX_EMBEDDING_JSON_BYTES,
                ),
                memory.embedding_version,
                memory.status.value,
                memory.supersedes_memory_id,
                _bounded_json(list(memory.source_message_ids), label="source_message_ids"),
                _bounded_json(list(memory.source_event_ids), label="source_event_ids"),
                _bounded_json(list(memory.source_decision_ids), label="source_decision_ids"),
                _bounded_json(list(memory.source_knowledge_ids), label="source_knowledge_ids"),
                _bounded_json(list(memory.source_plan_ids), label="source_plan_ids"),
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
                "scenario_id": None if row["scenario_id"] == LEGACY_SCENARIO_ID else row["scenario_id"],
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
                "source_plan_ids": json.loads(row["source_plan_ids"]),
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
                "scenario_id": None if row["scenario_id"] == LEGACY_SCENARIO_ID else row["scenario_id"],
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
