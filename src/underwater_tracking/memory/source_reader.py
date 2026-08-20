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
            if rows:
                self._memory.advance_source_cursor(user_id, scenario_id, "runtime_event", rows[-1].id)
        if self._decisions is not None:
            sources.extend(self._new_decisions(user_id, scenario_id))
        if self._plans is not None:
            sources.extend(self._active_plan(user_id, scenario_id))
        return tuple(sources)

    def read_conversation(
        self, user_id: str, scenario_id: str, conversation_id: str
    ) -> tuple[MemorySource, ...]:
        if self._short_term is None:
            return ()
        context = self._short_term.get_short_term(user_id, conversation_id)
        if context is None:
            return ()
        cursor = self._memory.get_source_cursor(user_id, scenario_id, f"conversation:{conversation_id}")
        first_retained_index = max(0, context.message_count - len(context.recent_messages))
        start = max(0, cursor - first_retained_index)
        messages = context.recent_messages[start : start + self._batch_limit]
        if not messages:
            return ()
        source = MemorySource(
            source_key=f"conversation:{conversation_id}:{messages[-1].message_id}",
            source_type="conversation",
            cursor=cursor + len(messages),
            payload={"conversation_id": conversation_id, "message_count": len(messages)},
            text="\n".join(message.text for message in messages),
            source_message_ids=tuple(message.message_id for message in messages),
        )
        self._memory.advance_source_cursor(user_id, scenario_id, f"conversation:{conversation_id}", source.cursor)
        return (source,)

    def load_work_sources(self, user_id: str, scenario_id: str | None, payload: object) -> tuple[MemorySource, ...]:
        """Re-read authoritative sources after a durable work item is leased."""
        del user_id
        sources: list[MemorySource] = []
        event_ids = tuple(getattr(payload, "source_event_ids", ()))
        decision_ids = tuple(getattr(payload, "source_decision_ids", ()))
        message_ids = tuple(getattr(payload, "source_message_ids", ()))
        if self._events is not None:
            for event_id in event_ids:
                event = self._events.get(event_id)
                if event is not None and (scenario_id is None or event.scenario_id == scenario_id):
                    sources.append(_event_source(event))
        if self._decisions is not None:
            for decision_id in decision_ids:
                decision = self._decisions.get(decision_id)
                if decision is not None and (scenario_id is None or decision.scenario_id == scenario_id):
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
        if self._short_term is not None and message_ids:
            # Conversation work always supplies its conversation context separately in the worker.
            del message_ids
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
        if sources:
            self._memory.advance_source_cursor(user_id, scenario_id, "decision", sources[-1].cursor)
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
        self._memory.advance_source_cursor(user_id, scenario_id, "plan", plan.revision)
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
    text = json.dumps(
        {key: value[key] for key in sorted(value) if key in {"summary", "rationale", "status"}},
        ensure_ascii=True,
        sort_keys=True,
    )
    return text[:4000] or "source record"
