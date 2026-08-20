"""Request-path memory facade: durable writes now, LLM work later."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from underwater_tracking.domain.memory_models import (
    MemoryContext,
    MemoryStreamEvent,
    MemoryStreamEventType,
    MemoryStreamPayload,
    MemoryStreamStatus,
    MemoryWorkItem,
    MemoryWorkPayload,
    MemoryWorkType,
    MemoryVersion,
    ShortTermContext,
    ShortTermMessage,
)
from underwater_tracking.persistence.memory import LongTermMemoryRepository, ShortTermContextRepository


class MemoryRetrieverPort(Protocol):
    def retrieve(
        self,
        *,
        user_id: str,
        query: str,
        filters: Mapping[str, object] | None = None,
        now: datetime | None = None,
    ) -> MemoryContext: ...


class MemoryService:
    """Coordinates bounded memory persistence without performing LLM work."""

    def __init__(
        self,
        short_term_repository: ShortTermContextRepository,
        long_term_repository: LongTermMemoryRepository,
        retriever: MemoryRetrieverPort,
    ) -> None:
        self._short_term = short_term_repository
        self._long_term = long_term_repository
        self._retriever = retriever

    def prepare_context(
        self,
        user_id: str,
        conversation_id: str,
        query: str,
        filters: Mapping[str, object] | None = None,
        scenario_id: str | None = None,
    ) -> MemoryContext:
        """Read short-term state and retrieve long-term material independently."""
        del scenario_id  # Scenario is a source boundary; retrieval is user-scoped.
        short_term = self._short_term.get_short_term(user_id, conversation_id)
        retrieved = self._retriever.retrieve(
            user_id=user_id, query=query, filters=filters, now=None
        )
        return MemoryContext(
            user_id=user_id,
            short_term_context=short_term,
            long_term_material=retrieved.long_term_material,
            retrieved_memory_ids=retrieved.retrieved_memory_ids,
            memory_status=retrieved.memory_status,
            evidence_trace=retrieved.evidence_trace,
        )

    def accept_turn(
        self,
        turn: Mapping[str, object] | object,
        result: Mapping[str, object] | object | None,
        source_refs: Sequence[str] = (),
    ) -> dict[str, str]:
        """Persist original messages and queue later semantic processing."""
        user_id = _required_value(turn, "user_id")
        conversation_id = _required_value(turn, "conversation_id")
        incoming = _as_message(turn, role="user")
        messages = [incoming]
        if result is not None:
            messages.append(_as_message(result, role="assistant", turn_id=incoming.turn_id))
        message_ids = tuple(message.message_id for message in messages)
        item = MemoryWorkItem(
            work_id=_new_id("memory-work"),
            user_id=user_id,
            conversation_id=conversation_id,
            scenario_id=_value(turn, "scenario_id") or None,
            work_type=MemoryWorkType.CONVERSATION_TURN,
            payload=MemoryWorkPayload(
                source_message_ids=message_ids,
                source_event_ids=tuple(source_refs),
            ),
        )
        scenario_id = item.scenario_id
        queued = self._long_term.append_messages_and_enqueue_work(
            user_id,
            conversation_id,
            tuple(messages),
            item,
            f"conversation:{conversation_id}:{incoming.message_id}",
            scenario_id=scenario_id,
            source_type=f"conversation:{conversation_id}" if scenario_id else None,
        )
        if queued:
            self._emit(
                user_id=user_id,
                conversation_id=conversation_id,
                status=MemoryStreamStatus.PENDING,
                event_type=MemoryStreamEventType.WORK_QUEUED,
                work_id=item.work_id,
                source_ids=message_ids + tuple(source_refs),
            )
        return {"status": "queued" if queued else "duplicate", "work_id": item.work_id}

    def enqueue_observation(
        self,
        source_ref: Mapping[str, object] | object,
        payload: Mapping[str, object],
    ) -> dict[str, str]:
        """Queue a bounded source reference; simulation callers never wait on an LLM."""
        source_id = _required_value(source_ref, "source_id")
        user_id = _value(source_ref, "user_id") or "operator"
        scenario_id = _required_value(source_ref, "scenario_id")
        source_type = _value(source_ref, "source_type") or "observation"
        cursor_type = _value(source_ref, "source_cursor_type") or source_type
        source_key = _value(source_ref, "source_key") or f"{source_type}:{source_id}"
        source_cursor_value = (
            source_ref.get("source_cursor")
            if isinstance(source_ref, Mapping)
            else getattr(source_ref, "source_cursor", None)
        )
        source_cursor = (
            source_cursor_value
            if isinstance(source_cursor_value, int) and not isinstance(source_cursor_value, bool)
            else None
        )
        conversation_id = _value(source_ref, "conversation_id")
        source_ids = _source_ids_for_type(source_type, source_id)
        item = MemoryWorkItem(
            work_id=_new_id("memory-work"),
            user_id=user_id,
            conversation_id=conversation_id or None,
            scenario_id=scenario_id,
            work_type=MemoryWorkType.OBSERVATION,
            payload=source_ids,
        )
        del payload  # Raw payload remains in its authoritative source repository.
        if source_cursor is not None:
            queued = self._long_term.enqueue_work_and_advance_cursor(
                item,
                source_key,
                scenario_id,
                cursor_type,
                source_cursor,
            )
        else:
            queued = self._long_term.enqueue_work(item, source_key)
        if queued:
            self._emit(
                user_id=user_id,
                conversation_id=conversation_id or None,
                status=MemoryStreamStatus.PENDING,
                event_type=MemoryStreamEventType.WORK_QUEUED,
                work_id=item.work_id,
                source_ids=(source_id,),
            )
        return {"status": "queued" if queued else "duplicate", "work_id": item.work_id}

    def snapshot(self, user_id: str, conversation_id: str) -> ShortTermContext | None:
        return self._short_term.get_short_term(user_id, conversation_id)

    def versions(self, user_id: str, memory_family_id: str) -> list[MemoryVersion]:
        return self._long_term.list_versions(user_id, memory_family_id)

    def delete(self, user_id: str, memory_id: str) -> bool:
        return self._long_term.mark_deleted(user_id, memory_id)

    def stream(
        self, user_id: str, conversation_id: str, *, after_cursor: int = 0, limit: int = 100
    ) -> list[MemoryStreamEvent]:
        return self._long_term.list_stream_events(
            user_id, conversation_id, after_cursor=after_cursor, limit=limit
        )

    def emit_worker_event(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        status: MemoryStreamStatus,
        event_type: MemoryStreamEventType,
        work_id: str,
        source_ids: Sequence[str] = (),
        memory_id: str | None = None,
        memory_family_id: str | None = None,
        version: int | None = None,
    ) -> MemoryStreamEvent:
        return self._emit(
            user_id=user_id,
            conversation_id=conversation_id,
            status=status,
            event_type=event_type,
            work_id=work_id,
            source_ids=source_ids,
            memory_id=memory_id,
            memory_family_id=memory_family_id,
            version=version,
        )

    def _emit(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        status: MemoryStreamStatus,
        event_type: MemoryStreamEventType,
        work_id: str,
        source_ids: Sequence[str],
        memory_id: str | None = None,
        memory_family_id: str | None = None,
        version: int | None = None,
    ) -> MemoryStreamEvent:
        return self._long_term.append_stream_event(
            MemoryStreamEvent(
                cursor=0,
                event_id=_new_id("memory-event"),
                user_id=user_id,
                conversation_id=conversation_id,
                status=status,
                type=event_type,
                payload=MemoryStreamPayload(
                    work_id=work_id,
                    source_ids=tuple(source_ids),
                    memory_ids=(memory_id,) if memory_id is not None else (),
                    memory_family_id=memory_family_id,
                    version=version,
                ),
                memory_id=memory_id,
                memory_family_id=memory_family_id,
                version=version,
            )
        )


def _value(value: Mapping[str, object] | object, name: str) -> str:
    candidate: Any
    if isinstance(value, Mapping):
        candidate = value.get(name, "")
    else:
        candidate = getattr(value, name, "")
    return candidate if isinstance(candidate, str) else ""


def _required_value(value: Mapping[str, object] | object, name: str) -> str:
    result = _value(value, name)
    if not result:
        raise ValueError(f"{name} must be a non-empty string")
    return result


def _as_message(
    value: Mapping[str, object] | object, *, role: str, turn_id: str | None = None
) -> ShortTermMessage:
    selected_role = _value(value, "role") or role
    if selected_role not in {"expert", "user", "assistant"}:
        selected_role = role
    return ShortTermMessage(
        message_id=_value(value, "message_id") or _new_id("message"),
        turn_id=_value(value, "turn_id") or turn_id,
        role=cast(Literal["expert", "user", "assistant"], selected_role),
        text=_required_value(value, "text"),
        source_evidence_ids=tuple(_value(value, "source_evidence_ids").split())
        if _value(value, "source_evidence_ids")
        else (),
    )


def _source_ids_for_type(source_type: str, source_id: str) -> MemoryWorkPayload:
    if source_type == "decision":
        return MemoryWorkPayload(source_decision_ids=(source_id,))
    if source_type == "conversation" or source_type.startswith("conversation:"):
        return MemoryWorkPayload(source_message_ids=(source_id,))
    if source_type in {"knowledge", "plan"}:
        return MemoryWorkPayload(source_knowledge_ids=(source_id,))
    return MemoryWorkPayload(source_event_ids=(source_id,))


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid4().hex}"
