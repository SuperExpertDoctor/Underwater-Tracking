"""Read bounded source records using durable per-source cursors."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from underwater_tracking.persistence.events import EventRepository, StoredEvent
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.memory import LongTermMemoryRepository, ShortTermContextRepository
from underwater_tracking.persistence.plans import PlanRepository


_DISCOVERY_SCENARIO_ID = "__memory_scope_discovery__"
_DISCOVERY_SOURCE_PREFIX = "__scope_discovery__:"


@dataclass(frozen=True)
class MemorySource:
    source_key: str
    source_type: str
    cursor: int
    payload: Mapping[str, object]
    text: str
    source_message_ids: tuple[str, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    source_decision_ids: tuple[str, ...] = ()
    source_knowledge_ids: tuple[str, ...] = ()
    source_cursor_type: str | None = None


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
        """Read and advance only successfully projected source rows."""
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

    def discover_scopes(self, user_id: str, limit: int | None = None) -> tuple[tuple[str, str], ...]:
        """Discover a bounded fair page using one durable round-robin continuation."""
        bounded_limit = self._batch_limit if limit is None else max(1, min(limit, self._batch_limit))
        repositories = (
            ("runtime_event", self._events),
            ("decision", self._decisions),
            ("plan", self._plans),
        )
        available = tuple((source_type, repository) for source_type, repository in repositories if repository is not None)
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
        context = self._short_term.get_short_term(user_id, conversation_id)
        if context is None:
            return ()
        cursor_type = f"conversation:{scenario_id}:{conversation_id}"
        cursor = self._memory.get_source_cursor(user_id, scenario_id, cursor_type)
        if cursor == 0:
            cursor = self._memory.get_source_cursor(
                user_id, scenario_id, f"conversation:{conversation_id}"
            )
        first_retained_index = max(0, context.message_count - len(context.recent_messages))
        absolute_start = max(cursor, first_retained_index)
        start = absolute_start - first_retained_index
        messages = context.recent_messages[start : start + self._batch_limit]
        if not messages:
            return ()
        source = MemorySource(
            source_key=f"conversation:{conversation_id}:{messages[-1].message_id}",
            source_type="conversation",
            cursor=absolute_start + len(messages),
            payload={"conversation_id": conversation_id, "message_count": len(messages)},
            text="\n".join(message.text for message in messages),
            source_message_ids=tuple(message.message_id for message in messages),
            source_cursor_type=cursor_type,
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
        if self._events is not None:
            for event_id in event_ids:
                event = self._events.get(event_id)
                if event is not None and event.scenario_id == scenario_id:
                    sources.append(_event_source(event))
        if self._decisions is not None:
            for decision_id in decision_ids:
                decision = self._decisions.get(decision_id)
                if decision is not None and decision.scenario_id == scenario_id:
                    sources.append(
                        MemorySource(
                            source_key=f"decision:{decision.decision_id}",
                            source_type="decision",
                            cursor=0,
                            payload={"decision_id": decision.decision_id, "sim_time_s": decision.sim_time_s},
                            text=_bounded_text(decision.model_dump(mode="json")),
                            source_decision_ids=(decision.decision_id,),
                        )
                    )
        if self._plans is not None:
            for plan_id in tuple(getattr(payload, "source_knowledge_ids", ())):
                plan = self._plans.get_plan(plan_id)
                if plan is not None and plan.scenario_id == scenario_id:
                    sources.append(
                        MemorySource(
                            source_key=f"plan:{plan.plan_id}:{plan.revision}",
                            source_type="plan",
                            cursor=plan.revision,
                            payload={
                                "plan_id": plan.plan_id,
                                "revision": plan.revision,
                                "status": plan.status,
                            },
                            text=_bounded_text(plan.model_dump(mode="json")),
                            source_knowledge_ids=(plan.plan_id,),
                        )
                    )
        if self._short_term is not None and message_ids and conversation_id is not None:
            messages = self._short_term.get_messages(user_id, conversation_id, message_ids)
            if messages:
                sources.append(
                    MemorySource(
                        source_key=f"conversation:{conversation_id}:{messages[0].message_id}",
                        source_type="conversation",
                        cursor=0,
                        payload={
                            "conversation_id": conversation_id,
                            "message_count": len(messages),
                        },
                        text="\n".join(message.text for message in messages),
                        source_message_ids=tuple(message.message_id for message in messages),
                        source_cursor_type=f"conversation:{scenario_id}:{conversation_id}",
                    )
                )
        return tuple(sources)

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
                    source_key=f"decision:{decision.decision_id}",
                    source_type="decision",
                    cursor=int(row["rowid"]),
                    payload={"decision_id": decision.decision_id, "sim_time_s": decision.sim_time_s},
                    text=_bounded_text(decision.model_dump(mode="json")),
                    source_decision_ids=(decision.decision_id,),
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
            source_key=f"plan:{plan.plan_id}:{plan.revision}",
            source_type="plan",
            cursor=plan.revision,
            payload={"plan_id": plan.plan_id, "revision": plan.revision, "status": plan.status},
            text=_bounded_text(plan.model_dump(mode="json")),
            source_knowledge_ids=(plan.plan_id,),
        )
        return (source,)


def _event_source(event: StoredEvent) -> MemorySource:
    payload = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "target_id": event.target_id,
        "sim_time_s": event.sim_time_s,
        "severity": event.severity,
    }
    safe_payload = event.payload.get("summary")
    if isinstance(safe_payload, str):
        payload["summary"] = safe_payload[:1000]
    return MemorySource(
        source_key=f"runtime_event:{event.event_id}",
        source_type="runtime_event",
        cursor=event.id,
        payload=payload,
        text=str(payload.get("summary", f"{event.event_type} at {event.sim_time_s}")),
        source_event_ids=(event.event_id,),
    )


def _bounded_text(value: Mapping[str, Any]) -> str:
    allowed = {
        "decision_id",
        "scenario_id",
        "sim_time_s",
        "trigger_event_ids",
        "snapshot_revision",
        "input_evidence_ids",
        "candidates",
        "candidate_plan_ids",
        "rejected_candidates",
        "verification_records",
        "final_plan_id",
        "final_plan_diff",
        "knowledge_query_ids",
        "plan_adjustment_suggestions",
        "concept",
        "plan_id",
        "revision",
        "status",
        "summary",
        "rationale",
    }
    selected = {key: value[key] for key in sorted(value) if key in allowed}
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
        return {
            str(key): _bounded_value(child, depth + 1)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if len(str(key)) <= 120
        } if len(value) <= 24 else {
            str(key): _bounded_value(child, depth + 1)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))[:24]
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_bounded_value(child, depth + 1) for child in value[:8]]
    if isinstance(value, str):
        return value[:256]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:256]
