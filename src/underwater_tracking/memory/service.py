"""Request-path memory facade: durable writes now, LLM work later."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from underwater_tracking.domain.memory_models import (
    MemoryContext,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryEvidenceTrace,
    MemoryRetrievalHit,
    MemoryStreamEvent,
    MemoryStreamEventType,
    MemoryStreamPayload,
    MemoryStreamStatus,
    MemoryStreamReasonCode,
    MemoryType,
    MemoryWorkItem,
    MemoryWorkPayload,
    MEMORY_WORK_PAYLOAD_MAX_JSON_BYTES,
    MemoryWorkType,
    MemoryVersion,
    ShortTermContext,
    ShortTermMessage,
)
from underwater_tracking.domain.models import RuntimeEvent
from underwater_tracking.domain.public_payload import sanitize_public_payload
from underwater_tracking.persistence.memory import (
    LongTermMemoryRepository,
    ShortTermContextRepository,
)
from underwater_tracking.memory.source_reader import tracking_episode_kind


_SAFE_OBSERVATION_FIELDS = frozenset(
    {
        "event_id",
        "event_type",
        "target_id",
        "sim_time_s",
        "severity",
        "summary",
        "decision_id",
        "scenario_id",
        "plan_id",
        "revision",
        "status",
        "conversation_id",
        "absolute_floor_m",
        "absolute_rms_m",
        "active_scan_uuv_ids",
        "assignment_uuv_ids",
        "assigned_uuv_ids",
        "candidate_id",
        "carrier_id",
        "changes_since_previous",
        "confidence",
        "confirmed",
        "consecutive_count",
        "current_label",
        "current_plan_version",
        "current_prediction_id",
        "current",
        "deployment_state",
        "diff_id",
        "evidence_ids",
        "heading_rad",
        "label",
        "llm_model",
        "llm_operation",
        "llm_prompt_version",
        "llm_request_hash",
        "llm_response_hash",
        "memory_eligible",
        "motion_model",
        "normalized_rms",
        "normalized_threshold",
        "observation_ids",
        "passive_track_uuv_ids",
        "plan_revision",
        "plan_impact",
        "plan_version",
        "previous_confidence",
        "previous_label",
        "previous_plan_version",
        "previous_prediction_id",
        "previous",
        "probabilities",
        "reason",
        "region_id",
        "region_assignments",
        "reserve_uuv_ids",
        "route_status",
        "sensor_mode",
        "source",
        "source_observation_ids",
        "speed_mps",
        "status",
        "suspicion_event_id",
        "successor_region_id",
        "successor_uuv_ids",
        "uuv_id",
        "uuv_ids",
        "predecessor_region_id",
        "predecessor_uuv_ids",
    }
)
_MAX_OBSERVATION_PAYLOAD_BYTES = 8192
_MAX_STREAM_SOURCE_IDS = 64


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
        *,
        degraded_reason: str | None = None,
    ) -> None:
        self._short_term = short_term_repository
        self._long_term = long_term_repository
        self._retriever = retriever
        self.degraded_reason = degraded_reason

    def prepare_context(
        self,
        user_id: str,
        conversation_id: str,
        query: str,
        filters: Mapping[str, object] | None = None,
        scenario_id: str | None = None,
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> MemoryContext:
        """Read short-term state and retrieve long-term material independently."""
        short_term = self._short_term.get_short_term(user_id, conversation_id, scenario_id)
        selected_filters = dict(filters or {})
        if scenario_id:
            selected_filters["scenario_id"] = scenario_id
        if execution_revision is not None:
            selected_filters["execution_revision"] = execution_revision
        if frame_id is not None:
            selected_filters["frame_id"] = frame_id
        try:
            retrieved = self._retriever.retrieve(
                user_id=user_id, query=query, filters=selected_filters, now=None
            )
        except Exception as exc:  # noqa: BLE001 - memory is non-blocking
            return MemoryContext(
                user_id=user_id,
                scenario_id=scenario_id,
                short_term_context=_stamp_short_term(
                    short_term, execution_revision=execution_revision, frame_id=frame_id
                ),
                memory_status=MemoryStreamStatus.DEGRADED,
                degraded_reason=(
                    self.degraded_reason
                    or f"memory retrieval unavailable: {type(exc).__name__}"
                ),
                execution_revision=execution_revision,
                frame_id=frame_id,
            )
        scoped_hits = tuple(
            hit
            for hit in retrieved.long_term_material
            if hit.memory.user_id == user_id
            and (scenario_id is None or hit.memory.scenario_id == scenario_id)
        )
        scoped_hits = _stamp_memory_hits(
            scoped_hits, execution_revision=execution_revision, frame_id=frame_id
        )
        return MemoryContext(
            user_id=user_id,
            scenario_id=scenario_id,
            short_term_context=_stamp_short_term(
                short_term, execution_revision=execution_revision, frame_id=frame_id
            ),
            long_term_material=scoped_hits,
            retrieved_memory_ids=tuple(hit.memory.memory_id for hit in scoped_hits),
            memory_status=retrieved.memory_status,
            degraded_reason=retrieved.degraded_reason or self.degraded_reason,
            evidence_trace=retrieved.evidence_trace,
            execution_revision=execution_revision,
            frame_id=frame_id,
        )

    def memory_snapshot(
        self,
        user_id: str,
        conversation_id: str,
        *,
        scenario_id: str | None = None,
        query: str = "",
        memory_type: MemoryType | None = None,
        min_importance_score: float | None = None,
        limit: int = 100,
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> dict[str, object]:
        """Build the bounded API view without exposing repository internals."""
        short_term = self._short_term.get_short_term(user_id, conversation_id, scenario_id)
        filters: dict[str, object] = {}
        if memory_type is not None:
            filters["memory_type"] = memory_type
        if min_importance_score is not None:
            filters["min_importance_score"] = min_importance_score
        if scenario_id is not None:
            filters["scenario_id"] = scenario_id
        active = self._long_term.list_active(user_id, filters=filters, limit=limit)
        retrieved = (
            self.prepare_context(
                user_id,
                conversation_id,
                query,
                filters=filters,
                scenario_id=scenario_id,
                execution_revision=execution_revision,
                frame_id=frame_id,
            )
            if query.strip()
            else MemoryContext(
                user_id=user_id,
                scenario_id=scenario_id,
                memory_status=(
                    MemoryStreamStatus.DEGRADED
                    if self.degraded_reason is not None
                    else MemoryStreamStatus.COMPLETED
                ),
                degraded_reason=self.degraded_reason,
                execution_revision=execution_revision,
                frame_id=frame_id,
            )
        )
        short_term = _stamp_short_term(
            short_term, execution_revision=execution_revision, frame_id=frame_id
        )
        if execution_revision is not None or frame_id is not None:
            active = tuple(
                memory.model_copy(
                    update={
                        "execution_revision": execution_revision,
                        "frame_id": frame_id,
                    }
                )
                for memory in active
            )
        by_type: dict[str, list[MemoryVersion]] = {
            memory_type_value.value: [] for memory_type_value in MemoryType
        }
        for memory in active:
            by_type[memory.memory_type.value].append(memory)
        episodes = self.tracking_episodes(user_id, scenario_id, limit=limit)
        return {
            "user_id": user_id,
            "scenario_id": scenario_id,
            "conversation_id": conversation_id,
            "short_term": short_term,
            "episodic": by_type[MemoryType.EPISODIC.value],
            "semantic": by_type[MemoryType.SEMANTIC.value],
            "procedural": by_type[MemoryType.PROCEDURAL.value],
            "retrieved_hits": retrieved.long_term_material,
            "versions": active,
            "episodes": episodes,
            "memory_status": retrieved.memory_status.value,
            "degraded_reason": retrieved.degraded_reason or self.degraded_reason,
            "execution_revision": execution_revision,
            "frame_id": frame_id,
        }

    def accept_turn(
        self,
        turn: Mapping[str, object] | object,
        result: Mapping[str, object] | Sequence[Mapping[str, object] | object] | object | None,
        source_refs: Sequence[str] = (),
        *,
        source_groups: MemoryWorkPayload | None = None,
        plan_version: int | None = None,
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> dict[str, object]:
        """Persist original messages and queue later semantic processing."""
        user_id = _required_value(turn, "user_id")
        conversation_id = _required_value(turn, "conversation_id")
        scenario_id = _required_value(turn, "scenario_id")
        execution_revision = execution_revision or _int_value(turn, "execution_revision")
        frame_id = frame_id if frame_id is not None else _int_value(turn, "frame_id")
        stable_scope = (user_id, conversation_id, scenario_id)
        incoming = _as_message(
            turn,
            role="user",
            stable_scope=stable_scope,
            scenario_id=scenario_id,
            execution_revision=execution_revision,
            frame_id=frame_id,
        )
        messages = [incoming]
        for rendered in _result_messages(result):
            if _value(rendered, "message_id") == incoming.message_id:
                continue
            messages.append(
                _as_message(
                    rendered,
                    role="assistant",
                    turn_id=incoming.turn_id or incoming.message_id,
                    stable_scope=stable_scope,
                    scenario_id=scenario_id,
                    execution_revision=execution_revision,
                    frame_id=frame_id,
                )
            )
        message_ids = tuple(message.message_id for message in messages)
        work_id = _stable_id(
            "memory-work",
            user_id,
            conversation_id,
            scenario_id,
            message_ids,
            tuple(source_refs),
        )
        item = MemoryWorkItem(
            work_id=work_id,
            user_id=user_id,
            conversation_id=conversation_id,
            scenario_id=scenario_id,
            work_type=MemoryWorkType.CONVERSATION_TURN,
            payload=_conversation_source_payload(
                message_ids,
                source_refs,
                source_groups=source_groups,
                plan_version=plan_version,
                execution_revision=execution_revision,
                frame_id=frame_id,
            ),
        )
        queued_source_groups = _bound_source_groups(_source_groups(item.payload))
        degraded = self.degraded_reason is not None
        queued_event = MemoryStreamEvent(
            cursor=0,
            event_id=_new_id("memory-event"),
            user_id=user_id,
            scenario_id=scenario_id,
            conversation_id=conversation_id,
            status=MemoryStreamStatus.DEGRADED if degraded else MemoryStreamStatus.PENDING,
            type=(
                MemoryStreamEventType.WORK_DEGRADED
                if degraded
                else MemoryStreamEventType.WORK_QUEUED
            ),
            payload=MemoryStreamPayload(
                work_id=item.work_id,
                source_ids=_flatten_source_groups(queued_source_groups),
                source_message_ids=queued_source_groups[0],
                source_event_ids=queued_source_groups[1],
                source_decision_ids=queued_source_groups[2],
                source_plan_ids=queued_source_groups[3],
                plan_version=plan_version,
                execution_revision=execution_revision,
                frame_id=frame_id,
            ),
            execution_revision=execution_revision,
            frame_id=frame_id,
        )
        queued, persisted_event = self._long_term.append_messages_enqueue_work_and_stream_event(
            user_id,
            conversation_id,
            tuple(messages),
            item,
            f"conversation:{scenario_id}:{conversation_id}:{incoming.message_id}",
            scenario_id=scenario_id,
            source_type=f"conversation:{scenario_id}:{conversation_id}",
            event=queued_event,
        )
        return {
            "status": "degraded" if degraded else ("queued" if queued else "duplicate"),
            "work_id": item.work_id,
            "stream_cursor": persisted_event.cursor if persisted_event is not None else None,
            "degraded_reason": self.degraded_reason,
            "execution_revision": execution_revision,
            "frame_id": frame_id,
        }

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
        source_key = _value(source_ref, "source_key") or f"{scenario_id}:{source_type}:{source_id}"
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
        source_ids = source_ids.model_copy(
            update={
                "execution_revision": _int_value(source_ref, "execution_revision"),
                "frame_id": _int_value(source_ref, "frame_id"),
            }
        )
        item = MemoryWorkItem(
            work_id=_new_id("memory-work"),
            user_id=user_id,
            conversation_id=conversation_id or None,
            scenario_id=scenario_id,
            work_type=MemoryWorkType.OBSERVATION,
            payload=source_ids,
        )
        safe_payload, source_text = _bounded_observation_projection(payload)
        bounded_payload = _fit_observation_payload(
            source_ids,
            source_type=source_type,
            source_text=source_text,
            source_payload=safe_payload,
        )
        item = item.model_copy(
            update={
                "payload": bounded_payload,
            }
        )
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
                scenario_id=scenario_id,
                status=MemoryStreamStatus.PENDING,
                event_type=MemoryStreamEventType.WORK_QUEUED,
                work_id=item.work_id,
                source_ids=(source_id,),
                source_message_ids=source_ids.source_message_ids,
                source_event_ids=source_ids.source_event_ids,
                source_decision_ids=source_ids.source_decision_ids,
                source_plan_ids=source_ids.source_plan_ids,
                execution_revision=source_ids.execution_revision,
                frame_id=source_ids.frame_id,
            )
        return {"status": "queued" if queued else "duplicate", "work_id": item.work_id}

    def snapshot(
        self,
        user_id: str,
        conversation_id: str,
        scenario_id: str | None = None,
        *,
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> ShortTermContext | None:
        return _stamp_short_term(
            self._short_term.get_short_term(user_id, conversation_id, scenario_id),
            execution_revision=execution_revision,
            frame_id=frame_id,
        )

    def messages(
        self,
        user_id: str,
        conversation_id: str,
        message_ids: Sequence[str],
        scenario_id: str | None = None,
    ) -> tuple[ShortTermMessage, ...]:
        return self._short_term.get_messages(
            user_id, conversation_id, message_ids, scenario_id=scenario_id
        )

    def versions(
        self, user_id: str, memory_family_id: str, scenario_id: str | None = None
    ) -> list[MemoryVersion]:
        versions = self._long_term.list_versions(user_id, memory_family_id, scenario_id)
        if versions:
            return versions
        if self._long_term.memory_family_exists(memory_family_id, scenario_id):
            raise PermissionError("memory family belongs to another user")
        raise LookupError("memory family was not found")

    def tracking_episodes(
        self,
        user_id: str,
        scenario_id: str | None = None,
        *,
        kind: str | None = None,
        status: MemoryEpisodeStatus | str | None = None,
        limit: int = 100,
    ) -> list[MemoryEpisode]:
        """Read deterministic tracking episodes from the durable memory index."""
        return self._long_term.list_memory_episodes(
            user_id,
            scenario_id,
            kind=kind,
            status=status,
            limit=limit,
        )

    def ingest_tracking_events(
        self,
        user_id: str,
        scenario_id: str,
        events: Sequence[RuntimeEvent | Mapping[str, object] | object],
    ) -> tuple[MemoryEpisode, ...]:
        """Merge source-backed tracking transitions into deterministic episodes."""
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string")
        ordered = tuple(
            sorted(
                (
                    event
                    for event in events
                    if _episode_descriptor(event, scenario_id) is not None
                    and _event_id(event).strip()
                ),
                key=lambda event: (
                    _event_sim_time(event),
                    _event_id(event),
                ),
            )
        )
        changed: dict[tuple[str, str, str, int], MemoryEpisode] = {}
        for event in ordered:
            descriptor = _episode_descriptor(event, scenario_id)
            if descriptor is None:
                continue
            kind, entity_id, opening_revision, closes = descriptor
            current = self._long_term.get_memory_episode_by_key(
                user_id,
                scenario_id,
                kind,
                entity_id,
                opening_revision,
            )
            if closes and current is None:
                candidates = self._long_term.list_memory_episodes(
                    user_id,
                    scenario_id,
                    kind=kind,
                    status=MemoryEpisodeStatus.OPEN,
                    limit=100,
                )
                current = next(
                    (
                        candidate
                        for candidate in reversed(candidates)
                        if candidate.entity_id == entity_id
                    ),
                    None,
                )
            source_event_id = _event_id(event)
            sim_time_s = _event_sim_time(event)
            execution_revision = _event_execution_revision(event)
            frame_id = _event_frame_id(event)
            if current is None:
                current = MemoryEpisode(
                    episode_id=_stable_id(
                        "episode",
                        scenario_id,
                        kind,
                        entity_id,
                        opening_revision,
                    ),
                    user_id=user_id,
                    scenario_id=scenario_id,
                    kind=kind,
                    entity_id=entity_id,
                    opening_execution_revision=opening_revision,
                    status=(
                        MemoryEpisodeStatus.CLOSED
                        if closes
                        else MemoryEpisodeStatus.OPEN
                    ),
                    summary=_episode_summary(event, kind),
                    source_event_ids=(source_event_id,),
                    started_sim_time_s=sim_time_s,
                    ended_sim_time_s=sim_time_s if closes else None,
                    closing_execution_revision=(
                        _closing_execution_revision(event, execution_revision)
                        if closes
                        else None
                    ),
                    frame_id=frame_id,
                )
            else:
                current = current.model_copy(
                    update={
                        "source_event_ids": _merge_ids(
                            current.source_event_ids,
                            (source_event_id,),
                        ),
                        "status": (
                            MemoryEpisodeStatus.CLOSED
                            if closes
                            else current.status
                        ),
                        "ended_sim_time_s": (
                            sim_time_s if closes else current.ended_sim_time_s
                        ),
                        "closing_execution_revision": (
                            _closing_execution_revision(event, execution_revision)
                            if closes
                            else current.closing_execution_revision
                        ),
                        "frame_id": frame_id if frame_id is not None else current.frame_id,
                        "updated_at": datetime.now(UTC),
                    }
                )
            key = current.episode_key
            assert key is not None
            changed[key] = self._long_term.upsert_memory_episode(current)
        return tuple(changed.values())

    def ingest_events(
        self,
        user_id: str,
        scenario_id: str,
        events: Sequence[RuntimeEvent | Mapping[str, object] | object],
    ) -> tuple[MemoryEpisode, ...]:
        """Compatibility alias for deterministic tracking-event ingestion."""
        return self.ingest_tracking_events(user_id, scenario_id, events)

    def delete(
        self,
        user_id: str,
        memory_id: str,
        scenario_id: str | None = None,
        conversation_id: str | None = None,
    ) -> bool:
        target = self._long_term.get_memory(user_id, memory_id, scenario_id)
        if target is None or target.status.value == "deleted":
            return False
        source_conversation = self._long_term.get_memory_source_conversation(
            user_id, memory_id, scenario_id
        )
        if source_conversation is None:
            source_conversation = self._short_term.find_conversation_for_messages(
                user_id,
                target.source_message_ids,
                scenario_id=scenario_id,
            )
        if target.source_message_ids and source_conversation is None:
            raise ValueError("memory source conversation cannot be resolved")
        if conversation_id is not None and conversation_id != source_conversation:
            raise PermissionError("memory conversation scope mismatch")
        deleted = self._long_term.mark_deleted(user_id, memory_id, scenario_id)
        if deleted:
            self._emit(
                user_id=user_id,
                conversation_id=source_conversation,
                scenario_id=scenario_id,
                status=MemoryStreamStatus.COMPLETED,
                event_type=MemoryStreamEventType.MEMORY_DELETED,
                work_id=f"memory-delete:{memory_id}",
                source_ids=(
                    *target.source_message_ids,
                    *target.source_event_ids,
                    *target.source_decision_ids,
                    *target.source_plan_ids,
                ),
                source_message_ids=target.source_message_ids,
                source_event_ids=target.source_event_ids,
                source_decision_ids=target.source_decision_ids,
                source_plan_ids=target.source_plan_ids,
                memory_id=target.memory_id,
                memory_family_id=target.memory_family_id,
                version=target.version,
                memory_type=target.memory_type,
            )
        return deleted

    def stream(
        self,
        user_id: str,
        conversation_id: str,
        *,
        scenario_id: str | None = None,
        after_cursor: int = 0,
        limit: int = 100,
        include_scenario_events: bool = True,
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> list[MemoryStreamEvent]:
        events = self._long_term.list_stream_events(
            user_id,
            conversation_id,
            scenario_id=scenario_id,
            after_cursor=after_cursor,
            limit=limit,
            include_scenario_events=include_scenario_events,
        )
        if execution_revision is None and frame_id is None:
            return events
        return [
            event.model_copy(
                update={
                    "execution_revision": (
                        event.execution_revision
                        if event.execution_revision is not None
                        else execution_revision
                    ),
                    "frame_id": event.frame_id if event.frame_id is not None else frame_id,
                    "payload": event.payload.model_copy(
                        update={
                            "execution_revision": (
                                event.payload.execution_revision
                                if event.payload.execution_revision is not None
                                else execution_revision
                            ),
                            "frame_id": (
                                event.payload.frame_id
                                if event.payload.frame_id is not None
                                else frame_id
                            ),
                        }
                    ),
                }
            )
            for event in events
        ]

    def emit_evidence_trace_events(
        self,
        *,
        user_id: str,
        conversation_id: str,
        scenario_id: str,
        trace: MemoryEvidenceTrace,
        plan_version: int | None = None,
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> tuple[MemoryStreamEvent, ...]:
        """Publish a bounded, idempotent causal trace for Memory Steam."""
        if trace.user_id != user_id:
            raise ValueError("evidence trace user_id must match the event user_id")
        source_groups = (
            trace.source_message_ids,
            trace.source_event_ids,
            trace.source_decision_ids,
            trace.source_plan_ids,
        )
        existing = self._long_term.get_stream_event_for_work(
            user_id,
            conversation_id,
            trace.trace_id,
            scenario_id,
        )
        pending: list[MemoryStreamEvent] = []
        if existing is None:
            pending.append(
                self._build_event(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    scenario_id=scenario_id,
                    status=MemoryStreamStatus.PROCESSING,
                    event_type=MemoryStreamEventType.EVIDENCE_TRACE_STARTED,
                    work_id=trace.trace_id,
                    source_ids=_flatten_source_groups(source_groups),
                    source_message_ids=source_groups[0],
                    source_event_ids=source_groups[1],
                    source_decision_ids=source_groups[2],
                    source_plan_ids=source_groups[3],
                    memory_ids=trace.memory_ids,
                    memory_id=trace.memory_ids[0] if trace.memory_ids else None,
                    plan_version=plan_version,
                    execution_revision=execution_revision,
                    frame_id=frame_id,
                    event_id=_stable_id(
                        "memory-evidence-event",
                        user_id,
                        conversation_id,
                        scenario_id,
                        trace.trace_id,
                        MemoryStreamEventType.EVIDENCE_TRACE_STARTED.value,
                    ),
                )
            )
        if existing is None or existing.type is not MemoryStreamEventType.EVIDENCE_TRACE_COMPLETED:
            pending.append(
                self._build_event(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    scenario_id=scenario_id,
                    status=trace.status,
                    event_type=MemoryStreamEventType.EVIDENCE_TRACE_COMPLETED,
                    work_id=trace.trace_id,
                    source_ids=_flatten_source_groups(source_groups),
                    source_message_ids=source_groups[0],
                    source_event_ids=source_groups[1],
                    source_decision_ids=source_groups[2],
                    source_plan_ids=source_groups[3],
                    memory_ids=trace.memory_ids,
                    memory_id=trace.memory_ids[0] if trace.memory_ids else None,
                    plan_version=plan_version,
                    execution_revision=execution_revision,
                    frame_id=frame_id,
                    event_id=_stable_id(
                        "memory-evidence-event",
                        user_id,
                        conversation_id,
                        scenario_id,
                        trace.trace_id,
                        MemoryStreamEventType.EVIDENCE_TRACE_COMPLETED.value,
                    ),
                )
            )
        return self._long_term.append_stream_events(tuple(pending))

    def emit_worker_event(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        status: MemoryStreamStatus,
        event_type: MemoryStreamEventType,
        work_id: str,
        scenario_id: str | None = None,
        source_ids: Sequence[str] = (),
        source_message_ids: Sequence[str] = (),
        source_event_ids: Sequence[str] = (),
        source_decision_ids: Sequence[str] = (),
        source_plan_ids: Sequence[str] = (),
        memory_id: str | None = None,
        memory_ids: Sequence[str] = (),
        memory_family_id: str | None = None,
        version: int | None = None,
        plan_version: int | None = None,
        operation: Literal["create", "update", "ignore"] | None = None,
        memory_type: MemoryType | None = None,
        reason_code: MemoryStreamReasonCode | None = None,
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> MemoryStreamEvent:
        work: MemoryWorkItem | None = None
        has_typed_sources = any(
            (
                source_message_ids,
                source_event_ids,
                source_decision_ids,
                source_plan_ids,
            )
        )
        if scenario_id is None or not has_typed_sources or plan_version is None:
            work = self._long_term.get_work(work_id)
            if scenario_id is None:
                scenario_id = work.scenario_id if work is not None else None
        if work is not None:
            work_groups = _source_groups(work.payload)
            if not has_typed_sources:
                (
                    source_message_ids,
                    source_event_ids,
                    source_decision_ids,
                    source_plan_ids,
                ) = work_groups
            if plan_version is None:
                plan_version = _work_plan_version(work)
            if execution_revision is None:
                execution_revision = work.payload.execution_revision
            if frame_id is None:
                frame_id = work.payload.frame_id
        return self._emit(
            user_id=user_id,
            conversation_id=conversation_id,
            scenario_id=scenario_id,
            status=status,
            event_type=event_type,
            work_id=work_id,
            source_ids=source_ids,
            source_message_ids=source_message_ids,
            source_event_ids=source_event_ids,
            source_decision_ids=source_decision_ids,
            source_plan_ids=source_plan_ids,
            memory_id=memory_id,
            memory_ids=memory_ids,
            memory_family_id=memory_family_id,
            version=version,
            plan_version=plan_version,
            operation=operation,
            memory_type=memory_type,
            reason_code=reason_code,
            execution_revision=execution_revision,
            frame_id=frame_id,
        )

    def _emit(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        scenario_id: str | None,
        status: MemoryStreamStatus,
        event_type: MemoryStreamEventType,
        work_id: str,
        source_ids: Sequence[str],
        source_message_ids: Sequence[str] = (),
        source_event_ids: Sequence[str] = (),
        source_decision_ids: Sequence[str] = (),
        source_plan_ids: Sequence[str] = (),
        memory_id: str | None = None,
        memory_ids: Sequence[str] = (),
        memory_family_id: str | None = None,
        version: int | None = None,
        plan_version: int | None = None,
        operation: Literal["create", "update", "ignore"] | None = None,
        memory_type: MemoryType | None = None,
        reason_code: MemoryStreamReasonCode | None = None,
        event_id: str | None = None,
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> MemoryStreamEvent:
        return self._long_term.append_stream_event(
            self._build_event(
                user_id=user_id,
                conversation_id=conversation_id,
                scenario_id=scenario_id,
                status=status,
                event_type=event_type,
                work_id=work_id,
                source_ids=source_ids,
                source_message_ids=source_message_ids,
                source_event_ids=source_event_ids,
                source_decision_ids=source_decision_ids,
                source_plan_ids=source_plan_ids,
                memory_id=memory_id,
                memory_ids=memory_ids,
                memory_family_id=memory_family_id,
                version=version,
                plan_version=plan_version,
                operation=operation,
                memory_type=memory_type,
                reason_code=reason_code,
                event_id=event_id,
                execution_revision=execution_revision,
                frame_id=frame_id,
            )
        )

    def _build_event(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        scenario_id: str | None,
        status: MemoryStreamStatus,
        event_type: MemoryStreamEventType,
        work_id: str,
        source_ids: Sequence[str],
        source_message_ids: Sequence[str] = (),
        source_event_ids: Sequence[str] = (),
        source_decision_ids: Sequence[str] = (),
        source_plan_ids: Sequence[str] = (),
        memory_id: str | None = None,
        memory_ids: Sequence[str] = (),
        memory_family_id: str | None = None,
        version: int | None = None,
        plan_version: int | None = None,
        operation: Literal["create", "update", "ignore"] | None = None,
        memory_type: MemoryType | None = None,
        reason_code: MemoryStreamReasonCode | None = None,
        event_id: str | None = None,
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> MemoryStreamEvent:
        source_groups = _bound_source_groups(
            (
                source_message_ids,
                source_event_ids,
                source_decision_ids,
                source_plan_ids,
            )
        )
        bounded_memory_ids = _unique_ids(
            memory_ids or ((memory_id,) if memory_id is not None else ())
        )
        return MemoryStreamEvent(
            cursor=0,
            event_id=event_id or _new_id("memory-event"),
            user_id=user_id,
            scenario_id=scenario_id,
            conversation_id=conversation_id,
            status=status,
            type=event_type,
            payload=MemoryStreamPayload(
                reason_code=reason_code,
                work_id=work_id,
                source_ids=_unique_ids(source_ids) or _flatten_source_groups(source_groups),
                memory_ids=bounded_memory_ids,
                memory_family_id=memory_family_id,
                version=version,
                source_message_ids=source_groups[0],
                source_event_ids=source_groups[1],
                source_decision_ids=source_groups[2],
                source_plan_ids=source_groups[3],
                plan_version=plan_version,
                execution_revision=execution_revision,
                frame_id=frame_id,
                operation=operation,
                memory_type=memory_type,
            ),
            memory_id=memory_id,
            memory_family_id=memory_family_id,
            version=version,
            execution_revision=execution_revision,
            frame_id=frame_id,
        )


def _value(value: Mapping[str, object] | object, name: str) -> str:
    candidate: Any
    if isinstance(value, Mapping):
        candidate = value.get(name, "")
    else:
        candidate = getattr(value, name, "")
    return candidate if isinstance(candidate, str) else ""


def _event_payload(event: Mapping[str, object] | object) -> Mapping[str, object]:
    if isinstance(event, Mapping):
        payload = event.get("payload")
    else:
        payload = getattr(event, "payload", None)
    return payload if isinstance(payload, Mapping) else {}


def _event_field(event: Mapping[str, object] | object, name: str) -> object:
    if isinstance(event, Mapping) and name in event:
        return event[name]
    value = getattr(event, name, None)
    if value is not None:
        return value
    return _event_payload(event).get(name)


def _event_id(event: Mapping[str, object] | object) -> str:
    value = _event_field(event, "event_id")
    return value.strip() if isinstance(value, str) else ""


def _event_sim_time(event: Mapping[str, object] | object) -> int:
    value = _event_field(event, "sim_time_s")
    if isinstance(value, bool):
        return 0
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, numeric)


def _event_int(event: Mapping[str, object] | object, name: str) -> int | None:
    value = _event_field(event, name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _event_execution_revision(event: Mapping[str, object] | object) -> int | None:
    return _event_int(event, "execution_revision")


def _event_frame_id(event: Mapping[str, object] | object) -> int | None:
    return _event_int(event, "frame_id")


def _event_scenario_id(event: Mapping[str, object] | object) -> str | None:
    value = _event_field(event, "scenario_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _episode_descriptor(
    event: Mapping[str, object] | object,
    scenario_id: str,
) -> tuple[str, str, int, bool] | None:
    event_scenario_id = _event_scenario_id(event)
    if event_scenario_id is not None and event_scenario_id != scenario_id:
        return None
    event_type = _event_field(event, "event_type")
    if not isinstance(event_type, str):
        return None
    kind = tracking_episode_kind(event_type)
    if kind is None:
        return None
    payload = _event_payload(event)
    entity_value: object
    if kind == "execution_recovery":
        entity_value = "execution"
    elif kind == "passive_tracking":
        entity_value = payload.get("group_instance_id") or _event_field(event, "entity_id")
    elif kind == "tracking_handoff":
        entity_value = (
            payload.get("tracking_owner_group_id")
            or payload.get("replacement_group_instance_id")
            or _event_field(event, "entity_id")
        )
    elif kind == "uuv_replacement":
        entity_value = (
            payload.get("replacement_uuv_id")
            or payload.get("replacement")
            or _event_field(event, "entity_id")
        )
    else:
        entity_value = (
            payload.get("tracking_owner_group_id")
            or payload.get("group_instance_id")
            or _event_field(event, "entity_id")
        )
    entity_id = str(entity_value).strip() if entity_value is not None else ""
    if not entity_id:
        entity_id = scenario_id if kind != "execution_recovery" else "execution"
    opening_revision = _event_execution_revision(event) or 0
    if kind == "execution_recovery" and event_type == "execution_snapshot_recovered":
        expired_revision = payload.get("expired_execution_revision")
        if isinstance(expired_revision, int) and not isinstance(expired_revision, bool):
            opening_revision = max(0, expired_revision)
    closes = event_type in {
        "execution_snapshot_recovered",
        "regional_mode_restored",
        "task_group_disappeared",
        "uuv_boundary_replacement",
        "member_replaced",
    }
    return kind, entity_id, opening_revision, closes


def _episode_summary(event: Mapping[str, object] | object, kind: str) -> str:
    event_type = _event_field(event, "event_type")
    payload = _event_payload(event)
    reason = payload.get("reason_code", payload.get("reason"))
    if isinstance(reason, str) and reason.strip():
        return f"{kind}:{event_type}:{reason.strip()[:160]}"
    return f"{kind}:{event_type}"


def _closing_execution_revision(
    event: Mapping[str, object] | object,
    fallback: int | None,
) -> int | None:
    recovered = _event_payload(event).get("recovered_execution_revision")
    if isinstance(recovered, int) and not isinstance(recovered, bool) and recovered >= 0:
        return recovered
    return fallback


def _merge_ids(existing: Sequence[str], incoming: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*existing, *incoming)))[:128]


def _int_value(value: Mapping[str, object] | object, name: str) -> int | None:
    candidate: object
    if isinstance(value, Mapping):
        candidate = value.get(name)
    else:
        candidate = getattr(value, name, None)
    return candidate if isinstance(candidate, int) and not isinstance(candidate, bool) else None


def _stamp_short_term(
    context: ShortTermContext | None,
    *,
    execution_revision: int | None,
    frame_id: int | None,
) -> ShortTermContext | None:
    if context is None or (execution_revision is None and frame_id is None):
        return context
    messages = tuple(
        message.model_copy(
            update={
                "execution_revision": (
                    message.execution_revision
                    if message.execution_revision is not None
                    else execution_revision
                ),
                "frame_id": message.frame_id if message.frame_id is not None else frame_id,
            }
        )
        for message in context.recent_messages
    )
    return context.model_copy(
        update={
            "recent_messages": messages,
            "execution_revision": execution_revision,
            "frame_id": frame_id,
        }
    )


def _stamp_memory_hits(
    hits: Sequence[MemoryRetrievalHit],
    *,
    execution_revision: int | None,
    frame_id: int | None,
) -> tuple[MemoryRetrievalHit, ...]:
    if execution_revision is None and frame_id is None:
        return tuple(hits)
    return tuple(
        hit.model_copy(
            update={
                "memory": hit.memory.model_copy(
                    update={
                        "execution_revision": execution_revision,
                        "frame_id": frame_id,
                    }
                )
            }
        )
        for hit in hits
    )


def _required_value(value: Mapping[str, object] | object, name: str) -> str:
    result = _value(value, name)
    if not result:
        raise ValueError(f"{name} must be a non-empty string")
    return result


def _as_message(
    value: Mapping[str, object] | object,
    *,
    role: str,
    turn_id: str | None = None,
    stable_scope: tuple[str, str, str] | None = None,
    scenario_id: str | None = None,
    execution_revision: int | None = None,
    frame_id: int | None = None,
) -> ShortTermMessage:
    selected_role = _value(value, "role") or role
    if selected_role not in {"expert", "user", "assistant"}:
        selected_role = role
    source_evidence_ids = _sequence_value(value, "source_evidence_ids")
    if not source_evidence_ids:
        source_evidence_ids = _sequence_value(value, "evidence_ids")
    return ShortTermMessage(
        message_id=_value(value, "message_id")
        or _stable_message_id(
            value, role=selected_role, turn_id=turn_id, stable_scope=stable_scope
        ),
        scenario_id=scenario_id or _value(value, "scenario_id") or None,
        turn_id=_value(value, "turn_id") or turn_id,
        role=cast(Literal["expert", "user", "assistant"], selected_role),
        text=_required_value(value, "text"),
        source_evidence_ids=source_evidence_ids,
        execution_revision=execution_revision,
        frame_id=frame_id,
    )


def _sequence_value(value: Mapping[str, object] | object, name: str) -> tuple[str, ...]:
    candidate: object
    if isinstance(value, Mapping):
        candidate = value.get(name, ())
    else:
        candidate = getattr(value, name, ())
    if isinstance(candidate, str):
        return tuple(item for item in candidate.split() if item)
    if isinstance(candidate, Sequence) and not isinstance(candidate, (bytes, bytearray)):
        return tuple(item for item in candidate if isinstance(item, str) and item)
    return ()


def _result_messages(
    result: Mapping[str, object] | Sequence[Mapping[str, object] | object] | object | None,
) -> tuple[Mapping[str, object] | object, ...]:
    if result is None:
        return ()
    if isinstance(result, Mapping):
        return (result,)
    messages = getattr(result, "messages", None)
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes, bytearray)):
        return tuple(messages)
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        return tuple(result)
    return (result,)


def _source_ids_for_type(source_type: str, source_id: str) -> MemoryWorkPayload:
    if source_type == "decision":
        return MemoryWorkPayload(source_decision_ids=(source_id,))
    if source_type == "conversation" or source_type.startswith("conversation:"):
        return MemoryWorkPayload(source_message_ids=(source_id,))
    if source_type == "plan":
        return MemoryWorkPayload(source_plan_ids=(source_id,))
    return MemoryWorkPayload(source_event_ids=(source_id,))


def _source_groups(
    payload: MemoryWorkPayload,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(payload.source_message_ids),
        tuple(payload.source_event_ids),
        tuple(payload.source_decision_ids),
        tuple(payload.source_plan_ids),
    )


def _flatten_source_groups(
    groups: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    return _unique_ids(source_id for group in groups for source_id in group)


def _bound_source_groups(
    groups: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    remaining = _MAX_STREAM_SOURCE_IDS
    bounded: list[tuple[str, ...]] = []
    for group in groups:
        selected = _unique_ids(group)[:remaining]
        bounded.append(selected)
        remaining -= len(selected)
    return tuple(bounded)


def _unique_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))[:_MAX_STREAM_SOURCE_IDS]


def _work_plan_version(work: MemoryWorkItem) -> int | None:
    value = work.payload.source_payload.get("revision")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _conversation_source_payload(
    message_ids: Sequence[str],
    source_refs: Sequence[str],
    *,
    source_groups: MemoryWorkPayload | None = None,
    plan_version: int | None = None,
    execution_revision: int | None = None,
    frame_id: int | None = None,
) -> MemoryWorkPayload:
    """Keep conversation provenance typed without guessing from ID strings.

    Opaque evidence IDs are conservatively treated as event sources. Callers
    that know a source is a decision or plan can provide the
    corresponding typed group explicitly.
    """
    typed = source_groups or MemoryWorkPayload()
    source_payload = dict(typed.source_payload)
    if plan_version is not None:
        source_payload["revision"] = plan_version
    return MemoryWorkPayload(
        source_payload=source_payload,
        source_message_ids=_unique_ids((*message_ids, *typed.source_message_ids)),
        source_event_ids=_unique_ids((*typed.source_event_ids, *source_refs)),
        source_decision_ids=_unique_ids(typed.source_decision_ids),
        source_plan_ids=_unique_ids(typed.source_plan_ids),
        execution_revision=(
            execution_revision
            if execution_revision is not None
            else typed.execution_revision
        ),
        frame_id=frame_id if frame_id is not None else typed.frame_id,
    )


def _bounded_observation_projection(
    payload: Mapping[str, object],
) -> tuple[dict[str, str | int | float | bool | None], str]:
    """Keep only a small, non-authoritative projection for deferred work."""
    sanitized = sanitize_public_payload(payload)
    payload = sanitized if isinstance(sanitized, Mapping) else {}
    projected: dict[str, str | int | float | bool | None] = {}
    for key in sorted(_SAFE_OBSERVATION_FIELDS):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            projected[key] = _bounded_utf8(value, 1000) if isinstance(value, str) else value
        elif isinstance(value, (Mapping, Sequence)) and not isinstance(
            value, (bytes, bytearray, str)
        ):
            projected[key] = _bounded_utf8(
                json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                1000,
            )
    while (
        projected
        and len(json.dumps(projected, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
        > _MAX_OBSERVATION_PAYLOAD_BYTES
    ):
        projected.pop(next(reversed(projected)))

    raw_text = payload.get("text") or payload.get("summary")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raw_text = json.dumps(projected, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return projected, _bounded_utf8(raw_text.strip(), 4000)


def _fit_observation_payload(
    base: MemoryWorkPayload,
    *,
    source_type: str,
    source_text: str,
    source_payload: Mapping[str, str | int | float | bool | None],
) -> MemoryWorkPayload:
    """Fit all observation fields under one serialized payload budget."""
    text = source_text
    projected = dict(source_payload)
    base_fields = base.model_dump(mode="python")
    while True:
        candidate = dict(base_fields)
        candidate.update(
            source_type=source_type,
            source_text=text or None,
            source_payload=projected,
        )
        try:
            return MemoryWorkPayload.model_validate(candidate)
        except ValueError:
            if text:
                text = _bounded_utf8(text, max(0, len(text.encode("utf-8")) // 2))
            elif projected:
                projected.pop(next(reversed(projected)))
            else:
                raise ValueError(
                    "observation work payload cannot fit within "
                    f"{MEMORY_WORK_PAYLOAD_MAX_JSON_BYTES} bytes"
                )


def _bounded_utf8(value: str, maximum_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value
    return encoded[:maximum_bytes].decode("utf-8", errors="ignore")


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid4().hex}"


def _stable_message_id(
    value: Mapping[str, object] | object,
    *,
    role: str,
    turn_id: str | None,
    stable_scope: tuple[str, str, str] | None,
) -> str:
    user_id, conversation_id, scenario_id = stable_scope or ("", "", "")
    text = _required_value(value, "text")
    payload = json.dumps(
        {
            "conversation_id": conversation_id,
            "scenario_id": scenario_id,
            "role": role,
            "text": text,
            "turn_id": turn_id or "",
            "user_id": user_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"message:{hashlib.sha256(payload).hexdigest()}"


def _stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
