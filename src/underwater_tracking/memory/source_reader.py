"""Read bounded source records using durable per-source cursors."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from underwater_tracking.domain.event_registry import is_memory_source_event
from underwater_tracking.domain.models import EventAudience
from underwater_tracking.domain.memory_models import ShortTermMessage
from underwater_tracking.persistence.events import EventRepository, StoredEvent
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.memory import (
    LongTermMemoryRepository,
    ShortTermContextRepository,
)
from underwater_tracking.persistence.plans import PlanRepository


_DISCOVERY_SCENARIO_ID = "__memory_scope_discovery__"
_DISCOVERY_SOURCE_PREFIX = "__scope_discovery__:"
_PUBLIC_EVENT_PAYLOAD_FIELDS = frozenset(
    {
        "absolute_floor_m",
        "absolute_rms_m",
        "active_scan_uuv_ids",
        "assignment_uuv_ids",
        "assigned_uuv_ids",
        "capability_active",
        "candidate_id",
        "carrier_id",
        "coverage",
        "confidence",
        "confirmed",
        "consecutive_count",
        "current_label",
        "current_plan_version",
        "current_prediction_id",
        "current",
        "diff_id",
        "execution_revision",
        "deployment_state",
        "energy_fraction",
        "evidence_ids",
        "gap_s",
        "gap_threshold_s",
        "group_id",
        "hard_guard_reasons",
        "healthy",
        "heading_rad",
        "label",
        "llm_model",
        "llm_operation",
        "llm_prompt_version",
        "llm_request_hash",
        "llm_response_hash",
        "mileage_m",
        "motion_model",
        "normalized_rms",
        "normalized_threshold",
        "observation_ids",
        "passive_track_uuv_ids",
        "plan_revision",
        "plan_id",
        "plan_impact",
        "plan_version",
        "position_covariance_trace",
        "previous_confidence",
        "previous_label",
        "previous_plan_version",
        "previous_prediction_id",
        "previous",
        "probabilities",
        "quality",
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
        "target_id",
        "threshold",
        "tracking_quality",
        "uuv_id",
        "uuv_ids",
        "predecessor_region_id",
        "predecessor_uuv_ids",
    }
)
_PUBLIC_SOURCE_FIELDS = frozenset(
    {
        "audiences",
        "candidate_plan_ids",
        "candidates",
        "change_type",
        "changes_since_previous",
        "concept",
        "decision_id",
        "event_id",
        "event_type",
        "final_plan_diff",
        "final_plan_id",
        "input_evidence_ids",
        "knowledge_query_ids",
        "plan_adjustment_suggestions",
        "plan_id",
        "rationale",
        "rejected_candidates",
        "revision",
        "scenario_id",
        "severity",
        "sim_time_s",
        "snapshot_revision",
        "status",
        "summary",
        "target_id",
        "trigger_event_ids",
        "member_ids_by_target",
        "roles_by_member",
        "intent_refs",
        "prediction_refs",
        "rotation_uuv_ids",
        "active_uuv_ids",
        "standby_uuv_ids",
        "returning_uuv_ids",
        "failed_uuv_ids",
        "regional_plans",
        "region_tasks",
        "regional_metrics",
        "diff",
        "execution_revision",
        "frame_id",
    }
    | _PUBLIC_EVENT_PAYLOAD_FIELDS
)
_SEQUENCE_EVENT_FIELDS = frozenset({"evidence_ids", "observation_ids"})


@dataclass(frozen=True)
class MemorySource:
    source_key: str
    source_type: str
    cursor: int
    payload: Mapping[str, object]
    text: str
    memory_eligible: bool = True
    source_message_ids: tuple[str, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    source_decision_ids: tuple[str, ...] = ()
    source_knowledge_ids: tuple[str, ...] = ()
    source_plan_ids: tuple[str, ...] = ()
    source_cursor_type: str | None = None
    execution_revision: int | None = None
    frame_id: int | None = None


class MemorySourceProvenanceError(ValueError):
    """A durable work item references a source outside its authoritative scope."""

    def __init__(self, message: str, source_ids: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.source_ids = tuple(dict.fromkeys(source_ids))


class MemorySourceReader:
    """Projects source repositories into safe summaries without fabricating data."""

    def __init__(
        self,
        memory_repository: LongTermMemoryRepository,
        *,
        event_repository: EventRepository | None = None,
        decision_ledger: DecisionLedger | None = None,
        plan_repository: PlanRepository | None = None,
        short_term_repository: ShortTermContextRepository | None = None,
        batch_limit: int = 32,
    ) -> None:
        self._memory = memory_repository
        self._events = event_repository
        self._decisions = decision_ledger
        self._plans = plan_repository
        self._short_term = short_term_repository
        self._batch_limit = max(1, min(batch_limit, 100))

    def read_new(self, user_id: str, scenario_id: str) -> tuple[MemorySource, ...]:
        """Read bounded source rows and mark non-durable events for cursor skipping."""
        sources: list[MemorySource] = []
        if self._events is not None:
            cursor = self._memory.get_source_cursor(user_id, scenario_id, "runtime_event")
            rows = self._events.list_events(
                scenario_id=scenario_id, since_id=cursor, limit=self._batch_limit
            )
            for event in rows:
                sources.append(_event_source(event))
        if self._decisions is not None:
            sources.extend(self._new_decisions(user_id, scenario_id))
        if self._plans is not None:
            sources.extend(self._active_plan(user_id, scenario_id))
        return tuple(sources)

    def discover_scopes(
        self, user_id: str, limit: int | None = None
    ) -> tuple[tuple[str, str], ...]:
        """Discover a bounded fair page using one durable round-robin continuation."""
        bounded_limit = (
            self._batch_limit if limit is None else max(1, min(limit, self._batch_limit))
        )
        repositories = (
            ("runtime_event", self._events),
            ("decision", self._decisions),
            ("plan", self._plans),
        )
        available = tuple(
            (source_type, repository)
            for source_type, repository in repositories
            if repository is not None
        )
        if not available:
            return ()
        repository_index, offsets = self._memory.get_source_discovery_state(user_id, len(available))
        next_offsets = list(offsets)
        discovered: list[tuple[str, str]] = []
        seen: set[str] = set()
        exhausted: set[int] = set()
        while len(discovered) < bounded_limit and len(exhausted) < len(available):
            if repository_index in exhausted:
                repository_index = (repository_index + 1) % len(available)
                continue
            _, repository = available[repository_index]
            assert repository is not None
            page = repository.list_scenario_ids(1, offset=next_offsets[repository_index])
            if not page:
                exhausted.add(repository_index)
                repository_index = (repository_index + 1) % len(available)
                continue
            next_offsets[repository_index] += len(page)
            scenario_id = page[0]
            if scenario_id not in seen:
                discovered.append((user_id, scenario_id))
                seen.add(scenario_id)
            repository_index = (repository_index + 1) % len(available)
        if not discovered and len(exhausted) == len(available):
            next_offsets = [0] * len(available)
        self._memory.register_source_scopes_and_advance_discovery(
            user_id,
            tuple(discovered),
            repository_index,
            tuple(next_offsets),
            legacy_cursors={
                f"{_DISCOVERY_SOURCE_PREFIX}{source_type}": next_offsets[index]
                for index, (source_type, _) in enumerate(available)
            },
        )
        return tuple(discovered)

    def read_conversation(
        self, user_id: str, scenario_id: str, conversation_id: str
    ) -> tuple[MemorySource, ...]:
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string")
        if self._short_term is None:
            return ()
        context = self._short_term.get_short_term(user_id, conversation_id, scenario_id)
        if context is None:
            return ()
        cursor_type = f"conversation:{scenario_id}:{conversation_id}"
        cursor = self._memory.get_source_cursor(user_id, scenario_id, cursor_type)
        if cursor > context.message_count:
            raise MemorySourceProvenanceError(
                "conversation source cursor is ahead of the retained message count"
            )
        messages = self._short_term.list_messages(
            user_id,
            conversation_id,
            scenario_id=scenario_id,
            offset=cursor,
            limit=self._batch_limit,
        )
        if not messages and cursor < context.message_count:
            raise MemorySourceProvenanceError(
                "conversation source messages are missing from immutable storage"
            )
        if not messages:
            return ()
        source = MemorySource(
            source_key=f"conversation:{scenario_id}:{conversation_id}:{messages[-1].message_id}",
            source_type="conversation",
            cursor=cursor + len(messages),
            payload={
                "conversation_id": conversation_id,
                "scenario_id": scenario_id,
                "message_count": len(messages),
            },
            text="\n".join(message.text for message in messages),
            source_message_ids=tuple(message.message_id for message in messages),
            source_cursor_type=cursor_type,
            execution_revision=context.execution_revision,
            frame_id=context.frame_id,
        )
        return (source,)

    def load_work_sources(
        self,
        user_id: str,
        scenario_id: str | None,
        payload: object,
        *,
        conversation_id: str | None = None,
    ) -> tuple[MemorySource, ...]:
        """Re-read authoritative sources after a durable work item is leased."""
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string")
        sources: list[MemorySource] = []
        event_ids = tuple(getattr(payload, "source_event_ids", ()))
        decision_ids = tuple(getattr(payload, "source_decision_ids", ()))
        message_ids = tuple(getattr(payload, "source_message_ids", ()))
        explicit_plan_ids = tuple(getattr(payload, "source_plan_ids", ()))
        conversation_messages: tuple[ShortTermMessage, ...] = ()
        if message_ids:
            if conversation_id is None or self._short_term is None:
                raise MemorySourceProvenanceError(
                    "source_message_ids require an authoritative conversation scope",
                    message_ids,
                )
            conversation_messages = self._short_term.get_messages(
                user_id, conversation_id, message_ids, scenario_id=scenario_id
            )
            loaded_message_ids = {message.message_id for message in conversation_messages}
            missing_message_ids = tuple(
                message_id
                for message_id in dict.fromkeys(message_ids)
                if message_id not in loaded_message_ids
            )
            if missing_message_ids:
                raise MemorySourceProvenanceError(
                    "source_message_ids are missing or outside the target user/conversation/scenario",
                    missing_message_ids,
                )
        if self._events is not None:
            for event_id in event_ids:
                event = self._events.get(event_id)
                if (
                    event is not None
                    and event.scenario_id == scenario_id
                    and EventAudience.MEMORY_SOURCE in event.audiences
                ):
                    sources.append(_event_source(event))
        if self._decisions is not None:
            for decision_id in decision_ids:
                decision = self._decisions.get(decision_id)
                if decision is not None and decision.scenario_id == scenario_id:
                    sources.append(
                        MemorySource(
                            source_key=f"decision:{scenario_id}:{decision.decision_id}",
                            source_type="decision",
                            cursor=0,
                            payload={
                                "decision_id": decision.decision_id,
                                "sim_time_s": decision.sim_time_s,
                            },
                            text=_bounded_text(decision.model_dump(mode="json")),
                            source_decision_ids=(decision.decision_id,),
                            execution_revision=getattr(decision, "execution_revision", None),
                            frame_id=getattr(decision, "frame_id", None),
                        )
                    )
        if self._plans is not None:
            legacy_plan_ids = (
                tuple(getattr(payload, "source_knowledge_ids", ())) if not explicit_plan_ids else ()
            )
            for plan_id in explicit_plan_ids + legacy_plan_ids:
                plan = self._plans.get_plan(plan_id)
                if (
                    plan is not None
                    and plan.scenario_id == scenario_id
                    and plan.status in {"active", "degraded"}
                ):
                    sources.append(
                        MemorySource(
                            source_key=f"plan:{scenario_id}:{plan.plan_id}:{plan.revision}",
                            source_type="plan",
                            cursor=plan.revision,
                            payload={
                                "plan_id": plan.plan_id,
                                "revision": plan.revision,
                                "status": plan.status,
                            },
                            text=_bounded_text(plan.model_dump(mode="json")),
                            source_plan_ids=(plan.plan_id,) if explicit_plan_ids else (),
                            source_knowledge_ids=(plan.plan_id,) if not explicit_plan_ids else (),
                            execution_revision=getattr(plan, "execution_revision", None),
                            frame_id=getattr(plan, "frame_id", None),
                        )
                    )
        if self._short_term is not None and message_ids and conversation_id is not None:
            if conversation_messages:
                sources.append(
                    MemorySource(
                        source_key=(
                            f"conversation:{scenario_id}:{conversation_id}:"
                            f"{conversation_messages[0].message_id}"
                        ),
                        source_type="conversation",
                        cursor=0,
                        payload={
                            "conversation_id": conversation_id,
                            "scenario_id": scenario_id,
                            "message_count": len(conversation_messages),
                        },
                        text="\n".join(message.text for message in conversation_messages),
                        source_message_ids=tuple(
                            message.message_id for message in conversation_messages
                        ),
                        source_cursor_type=f"conversation:{scenario_id}:{conversation_id}",
                        execution_revision=next(
                            (
                                message.execution_revision
                                for message in conversation_messages
                                if message.execution_revision is not None
                            ),
                            None,
                        ),
                        frame_id=next(
                            (
                                message.frame_id
                                for message in conversation_messages
                                if message.frame_id is not None
                            ),
                            None,
                        ),
                    )
                )
        return tuple(_stamp_source_context(source, payload) for source in sources)

    def _new_decisions(self, user_id: str, scenario_id: str) -> Sequence[MemorySource]:
        assert self._decisions is not None
        cursor = self._memory.get_source_cursor(user_id, scenario_id, "decision")
        rows = self._decisions._conn.execute(
            "SELECT rowid, decision_id FROM decision_records WHERE scenario_id = ? AND rowid > ?"
            " ORDER BY rowid LIMIT ?",
            (scenario_id, cursor, self._batch_limit),
        ).fetchall()
        if not rows:
            return ()
        sources: list[MemorySource] = []
        for row in rows:
            decision = self._decisions.get(row["decision_id"])
            if decision is None:
                continue
            sources.append(
                MemorySource(
                    source_key=f"decision:{scenario_id}:{decision.decision_id}",
                    source_type="decision",
                    cursor=int(row["rowid"]),
                    payload={
                        "decision_id": decision.decision_id,
                        "sim_time_s": decision.sim_time_s,
                        "execution_revision": getattr(decision, "execution_revision", None),
                        "frame_id": getattr(decision, "frame_id", None),
                    },
                    text=_bounded_text(decision.model_dump(mode="json")),
                    source_decision_ids=(decision.decision_id,),
                    execution_revision=getattr(decision, "execution_revision", None),
                    frame_id=getattr(decision, "frame_id", None),
                )
            )
        return tuple(sources)

    def _active_plan(self, user_id: str, scenario_id: str) -> Sequence[MemorySource]:
        assert self._plans is not None
        plan = self._plans.get_active(scenario_id)
        if plan is None:
            return ()
        cursor = self._memory.get_source_cursor(user_id, scenario_id, "plan")
        if plan.revision <= cursor:
            return ()
        source = MemorySource(
            source_key=f"plan:{scenario_id}:{plan.plan_id}:{plan.revision}",
            source_type="plan",
            cursor=plan.revision,
            payload={"plan_id": plan.plan_id, "revision": plan.revision, "status": plan.status},
            text=_bounded_text(plan.model_dump(mode="json")),
            source_plan_ids=(plan.plan_id,),
            execution_revision=getattr(plan, "execution_revision", None),
            frame_id=getattr(plan, "frame_id", None),
        )
        return (source,)


def _event_source(event: StoredEvent) -> MemorySource:
    execution_revision = _event_context_value(event, "execution_revision")
    frame_id = _event_context_value(event, "frame_id")
    payload = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "target_id": event.target_id,
        "sim_time_s": event.sim_time_s,
        "severity": event.severity,
        "audiences": tuple(sorted(audience.value for audience in event.audiences)),
    }
    if execution_revision is not None:
        payload["execution_revision"] = execution_revision
    if frame_id is not None:
        payload["frame_id"] = frame_id
    summary = _bounded_runtime_summary(event.payload.get("summary"))
    if summary is not None:
        payload["summary"] = summary
    for field_name in sorted(_PUBLIC_EVENT_PAYLOAD_FIELDS):
        if field_name in event.payload:
            payload[field_name] = _bounded_event_value(field_name, event.payload[field_name])
    memory_eligible = (
        EventAudience.MEMORY_SOURCE in event.audiences
        and is_memory_source_event(event.event_type, event.payload)
    )
    evidence_text = _bounded_text(payload)
    if summary is not None and memory_eligible and event.event_type != "periodic_situation_summary":
        source_text = f"{summary}; evidence={evidence_text}"
    else:
        source_text = summary or evidence_text or f"{event.event_type} at {event.sim_time_s}"
    return MemorySource(
        source_key=f"runtime_event:{event.scenario_id}:{event.event_id}",
        source_type="runtime_event",
        cursor=event.id,
        payload=payload,
        text=source_text,
        memory_eligible=memory_eligible,
        source_event_ids=(event.event_id,),
        execution_revision=execution_revision,
        frame_id=frame_id,
    )


def _stamp_source_context(source: MemorySource, payload: object) -> MemorySource:
    execution_revision = source.execution_revision
    if execution_revision is None:
        execution_revision = _payload_context_value(payload, "execution_revision")
    frame_id = source.frame_id
    if frame_id is None:
        frame_id = _payload_context_value(payload, "frame_id")
    if execution_revision is None and frame_id is None:
        return source
    projected_payload = dict(source.payload)
    if execution_revision is not None:
        projected_payload.setdefault("execution_revision", execution_revision)
    if frame_id is not None:
        projected_payload.setdefault("frame_id", frame_id)
    return replace(
        source,
        payload=projected_payload,
        execution_revision=execution_revision,
        frame_id=frame_id,
    )


def _payload_context_value(payload: object, name: str) -> int | None:
    candidate = getattr(payload, name, None)
    return candidate if isinstance(candidate, int) and not isinstance(candidate, bool) else None


def _event_context_value(event: StoredEvent, name: str) -> int | None:
    direct = getattr(event, name, None)
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
    event_payload = event.payload.get(name)
    return (
        event_payload
        if isinstance(event_payload, int) and not isinstance(event_payload, bool)
        else None
    )


def _bounded_event_value(field_name: str, value: object) -> object:
    if field_name in _SEQUENCE_EVENT_FIELDS and isinstance(value, (list, tuple, frozenset)):
        return tuple(str(item) for item in tuple(value)[:16])
    return _bounded_value(value)


def _bounded_runtime_summary(value: object) -> str | None:
    """Read the prebuilt public summary without projecting raw event payloads."""
    if not isinstance(value, str):
        return None
    summary = value[:1000]
    while len(summary.encode("utf-8")) > 4000:
        summary = summary[:-1]
    return summary


def _bounded_text(value: Mapping[str, Any]) -> str:
    selected = {key: value[key] for key in sorted(value) if key in _PUBLIC_SOURCE_FIELDS}
    bounded = _bounded_value(selected)
    text = json.dumps(bounded, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if len(text.encode("utf-8")) <= 4000:
        return text or "source record"
    bounded_mapping = bounded if isinstance(bounded, Mapping) else {}
    raw_candidates = bounded_mapping.get("candidates", ())
    candidates = raw_candidates[:4] if isinstance(raw_candidates, list) else []
    raw_rejected = bounded_mapping.get("rejected_candidates", {})
    rejected = raw_rejected if isinstance(raw_rejected, Mapping) else {}
    compact = {
        "candidates": candidates,
        "rejected_candidates": dict(list(rejected.items())[:8]),
        "final_plan_id": bounded_mapping.get("final_plan_id"),
    }
    fallback = json.dumps(compact, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if len(fallback.encode("utf-8")) <= 4000:
        return fallback
    return json.dumps(
        {"candidates": [], "rejected_candidates": {}, "final_plan_id": compact["final_plan_id"]},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _bounded_value(value: object, depth: int = 0) -> object:
    """Bound nested source evidence while retaining structured decision fields."""
    if depth >= 4:
        return str(value)[:128]
    if isinstance(value, Mapping):
        return (
            {
                str(key): _bounded_value(child, depth + 1)
                for key, child in sorted(value.items(), key=lambda item: str(item[0]))
                if len(str(key)) <= 120
            }
            if len(value) <= 24
            else {
                str(key): _bounded_value(child, depth + 1)
                for key, child in sorted(value.items(), key=lambda item: str(item[0]))[:24]
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_bounded_value(child, depth + 1) for child in value[:8]]
    if isinstance(value, str):
        return value[:256]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:256]
