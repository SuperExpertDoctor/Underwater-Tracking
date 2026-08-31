# src/underwater_tracking/agent/nodes/history.py
"""Evidence-preserving History compression (spec 9, plan Task 9).

The compression node maintains three summary namespaces —
``operational``, ``decision``, ``conversation`` — and appends one entry per
namespace per compressed window to the long-term memory store. Every entry
keeps the ``evidence_ids`` of the source records it was built from, and
compression is strictly append-only: it never deletes or rewrites the
original observations, plans, or ledger records (spec 9).

The trigger is a deterministic threshold policy over the covered time
window, the covered event count, the conversation message count, and a
character-derived token estimate; summary content is derived deterministically
from the stored records, so identical stores produce identical summaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.store.base import BaseStore
from pydantic import Field

from underwater_tracking.config.models import AgentConfig
from underwater_tracking.domain.agent_models import (
    DecisionRecord,
    ExpertDirective,
)
from underwater_tracking.domain.models import StrictModel
from underwater_tracking.persistence.events import EventRepository, StoredEvent
from underwater_tracking.persistence.ledger import DecisionLedger, QuestionRun
from underwater_tracking.persistence.sqlite import json_dumps

#: Event types that carry expert-conversation records in the event stream
#: (directive applications, clarifications, question runs).
_CONVERSATION_EVENT_PREFIXES = ("directive", "question", "clarification")

#: Severities treated as critical for the operational summary and context.
_CRITICAL_SEVERITIES = ("warning", "critical")

EVENT_LIST_LIMIT = 10_000


class OperationalSummary(StrictModel):
    """Operational digest: targets, group quality, resources, key events.

    Covers the interval ``[start_time_s, end_time_s]``; ``evidence_ids``
    are the source event ids the digest was built from (spec 9: every
    summary entry keeps its evidence ids).
    """

    namespace: Literal["operational"] = "operational"
    summary_id: str
    scenario_id: str
    start_time_s: int = Field(ge=0)
    end_time_s: int = Field(ge=0)
    targets: tuple[str, ...] = ()
    group_quality: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    key_events: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    unresolved_risks: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


class DecisionSummary(StrictModel):
    """Decision digest: active strategies, plan changes, failures, risks."""

    namespace: Literal["decision"] = "decision"
    summary_id: str
    scenario_id: str
    start_time_s: int = Field(ge=0)
    end_time_s: int = Field(ge=0)
    active_strategies: tuple[str, ...] = ()
    plan_changes: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    unresolved_risks: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


class ConversationSummary(StrictModel):
    """Conversation digest: expert annotations, clarifications, question topics."""

    namespace: Literal["conversation"] = "conversation"
    summary_id: str
    scenario_id: str
    start_time_s: int = Field(ge=0)
    end_time_s: int = Field(ge=0)
    expert_annotations: tuple[str, ...] = ()
    clarifications: tuple[str, ...] = ()
    question_topics: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    unresolved_risks: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


HistorySummary = OperationalSummary | DecisionSummary | ConversationSummary


class HistoryState(TypedDict, total=False):
    """State of the history-compression subgraph (spec 9)."""

    scenario_id: str
    window_end_s: int
    operational_summary: OperationalSummary | None
    decision_summary: DecisionSummary | None
    conversation_summary: ConversationSummary | None
    compressed: bool


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate: one token per four characters (spec 9)."""
    return (len(text) + 3) // 4


def _summary_id(scenario_id: str, namespace: str, start: int, end: int) -> str:
    return f"{scenario_id}:{namespace}:{start}:{end}"


def render_event(event: StoredEvent) -> str:
    return f"[{event.sim_time_s}] {event.event_type} ({event.severity}) {event.event_id}"


def build_operational_summary(
    *,
    scenario_id: str,
    window_end_s: int,
    events: Sequence[StoredEvent],
) -> OperationalSummary:
    """Deterministic operational digest of the events up to ``window_end_s``."""
    in_window = [event for event in events if event.sim_time_s <= window_end_s]
    start = min((event.sim_time_s for event in in_window), default=0)
    facts = tuple(render_event(event) for event in in_window)
    critical = [event for event in in_window if event.severity in _CRITICAL_SEVERITIES]
    return OperationalSummary(
        summary_id=_summary_id(scenario_id, "operational", start, window_end_s),
        scenario_id=scenario_id,
        start_time_s=start,
        end_time_s=window_end_s,
        targets=tuple(sorted({event.target_id for event in in_window if event.target_id})),
        group_quality=_quality_lines(in_window),
        resources=_resource_lines(in_window),
        key_events=tuple(render_event(event) for event in critical) or facts[-5:],
        facts=facts,
        unresolved_risks=tuple(
            f"risk at t={event.sim_time_s}: {event.event_type}"
            for event in critical
        ),
        evidence_ids=tuple(sorted({event.event_id for event in in_window})),
    )


def _quality_lines(events: Sequence[StoredEvent]) -> tuple[str, ...]:
    lines: list[str] = []
    for event in events:
        quality = event.payload.get("quality")
        if isinstance(quality, (int, float)) and not isinstance(quality, bool):
            target = event.target_id or "n/a"
            lines.append(f"t={event.sim_time_s} {target} quality={quality:.3f}")
    return tuple(lines)


def _resource_lines(events: Sequence[StoredEvent]) -> tuple[str, ...]:
    lines: list[str] = []
    for event in events:
        energy = event.payload.get("energy")
        if isinstance(energy, (int, float)) and not isinstance(energy, bool):
            lines.append(f"t={event.sim_time_s} {event.event_type} energy={energy:.3f}")
    return tuple(lines)


def build_decision_summary(
    *,
    scenario_id: str,
    window_end_s: int,
    decisions: Sequence[DecisionRecord],
) -> DecisionSummary:
    """Deterministic decision digest of the ledger records up to the window."""
    in_window = [d for d in decisions if d.sim_time_s <= window_end_s]
    start = min((d.sim_time_s for d in in_window), default=0)
    failures = tuple(
        f"{decision_id} rejected: {reason}"
        for d in in_window
        for decision_id, reason in sorted(d.rejected_candidates.items())
    )
    return DecisionSummary(
        summary_id=_summary_id(scenario_id, "decision", start, window_end_s),
        scenario_id=scenario_id,
        start_time_s=start,
        end_time_s=window_end_s,
        active_strategies=tuple(
            sorted({proposal.concept for d in in_window for proposal in d.candidates})
        ),
        plan_changes=tuple(
            f"{d.final_plan_id} rev={d.snapshot_revision}"
            for d in in_window
            if d.final_plan_id is not None
        ),
        failures=failures,
        facts=tuple(
            f"{d.decision_id} triggers={','.join(sorted(d.trigger_event_ids)) or '-'}"
            for d in sorted(in_window, key=lambda d: (d.sim_time_s, d.decision_id))
        ),
        unresolved_risks=tuple(f"unresolved risk: {failure}" for failure in failures),
        evidence_ids=tuple(
            sorted(
                {
                    event_id
                    for d in in_window
                    for event_id in (*d.trigger_event_ids, *d.input_evidence_ids)
                }
            )
        ),
    )


def build_conversation_summary(
    *,
    scenario_id: str,
    window_end_s: int,
    questions: Sequence[QuestionRun],
    directives: Sequence[ExpertDirective],
    events: Sequence[StoredEvent],
) -> ConversationSummary:
    """Deterministic conversation digest from directives and question runs.

    Directives and question runs carry no simulation timestamp, so the
    digest covers the whole scenario conversation up to the window; its
    evidence ids point at the conversation-related events in the window.
    """
    in_window = [event for event in events if event.sim_time_s <= window_end_s]
    annotations = tuple(
        d.raw_text for d in directives if d.status == "applied"
    )
    clarifications = tuple(
        d.raw_text for d in directives if d.status == "needs_clarification"
    )
    conversation_events = [
        event
        for event in in_window
        if event.event_type.startswith(_CONVERSATION_EVENT_PREFIXES)
    ]
    return ConversationSummary(
        summary_id=_summary_id(scenario_id, "conversation", 0, window_end_s),
        scenario_id=scenario_id,
        start_time_s=0,
        end_time_s=window_end_s,
        expert_annotations=annotations,
        clarifications=clarifications,
        question_topics=tuple(q.question_text for q in questions),
        facts=tuple(
            f"directive {d.directive_id} status={d.status}" for d in directives
        )
        + tuple(f"question {q.run_id} status={q.status}" for q in questions),
        unresolved_risks=tuple(
            f"clarification pending: {d.directive_id}"
            for d in directives
            if d.status == "needs_clarification"
        ),
        evidence_ids=tuple(
            sorted({event.event_id for event in conversation_events})
        ),
    )


def decode_summary(value: dict[str, Any]) -> HistorySummary:
    """Decode a stored summary value by its ``namespace`` discriminator."""
    namespace = value.get("namespace")
    if namespace == "operational":
        return OperationalSummary.model_validate(value)
    if namespace == "decision":
        return DecisionSummary.model_validate(value)
    if namespace == "conversation":
        return ConversationSummary.model_validate(value)
    raise ValueError(f"unknown history summary namespace: {namespace!r}")


def list_summaries(store: BaseStore, scenario_id: str) -> tuple[HistorySummary, ...]:
    """All stored summaries of a scenario, stable interval/id order."""
    items = store.search(("scenario", scenario_id, "history"), limit=EVENT_LIST_LIMIT)
    return tuple(
        sorted(
            (decode_summary(item.value) for item in items),
            key=lambda s: (s.start_time_s, s.summary_id),
        )
    )


@dataclass(frozen=True)
class HistoryTriggerPolicy:
    """Deterministic compression trigger (spec 9).

    Compression fires when any threshold is met or crossed: the covered
    time window, the number of covered events, the number of conversation
    messages, or the estimated token count. ``token_threshold`` mirrors
    ``AgentConfig.history_token_threshold`` (6000 in the shipped config);
    ``from_agent_config`` adopts the validated configuration value.
    """

    window_s: int = 900
    event_count: int = 50
    message_count: int = 20
    token_threshold: int = 6000

    def should_compress(
        self,
        *,
        covered_window_s: int,
        covered_event_count: int,
        covered_message_count: int,
        estimated_tokens: int,
    ) -> bool:
        return (
            covered_window_s >= self.window_s
            or covered_event_count >= self.event_count
            or covered_message_count >= self.message_count
            or estimated_tokens >= self.token_threshold
        )

    @classmethod
    def from_agent_config(cls, config: AgentConfig) -> HistoryTriggerPolicy:
        """Adopt the token threshold from the validated agent configuration."""
        return cls(token_threshold=config.history_token_threshold)


class CompressHistoryNode:
    """Append-only per-namespace history compression (spec 9).

    Evaluates the deterministic trigger over the covered window; when it
    fires, builds the operational, decision, and conversation summaries and
    appends one entry per namespace to the long-term memory store. Source
    records — events, plans, ledger records — are never deleted or modified.
    """

    def __init__(
        self,
        events: EventRepository,
        ledger: DecisionLedger,
        store: BaseStore,
        trigger: HistoryTriggerPolicy | None = None,
    ) -> None:
        self._events = events
        self._ledger = ledger
        self._store = store
        self._trigger = trigger or HistoryTriggerPolicy()

    def __call__(self, state: HistoryState) -> HistoryState:
        scenario_id = state.get("scenario_id")
        if not scenario_id:
            raise ValueError("compress_history requires scenario_id in state")
        window_end_s = state.get("window_end_s", 0)
        events = [
            event
            for event in self._events.list_events(
                scenario_id=scenario_id, limit=EVENT_LIST_LIMIT
            )
            if event.sim_time_s <= window_end_s
        ]
        questions = self._ledger.list_questions(scenario_id=scenario_id)
        estimated_tokens = estimate_tokens(
            "\n".join(
                render_event(event) + " " + json_dumps(event.payload) for event in events
            )
        )
        if not self._trigger.should_compress(
            covered_window_s=window_end_s,
            covered_event_count=len(events),
            covered_message_count=len(questions),
            estimated_tokens=estimated_tokens,
        ):
            return {
                "compressed": False,
                "operational_summary": None,
                "decision_summary": None,
                "conversation_summary": None,
            }
        operational = build_operational_summary(
            scenario_id=scenario_id, window_end_s=window_end_s, events=events
        )
        decision = build_decision_summary(
            scenario_id=scenario_id,
            window_end_s=window_end_s,
            decisions=self._ledger.list_decisions(scenario_id=scenario_id),
        )
        conversation = build_conversation_summary(
            scenario_id=scenario_id,
            window_end_s=window_end_s,
            questions=questions,
            directives=self._ledger.list_directives(scenario_id=scenario_id),
            events=events,
        )
        for summary in (operational, decision, conversation):
            self._store.put(
                ("scenario", scenario_id, "history", summary.namespace),
                summary.summary_id,
                summary.model_dump(mode="json"),
            )
        return {
            "compressed": True,
            "operational_summary": operational,
            "decision_summary": decision,
            "conversation_summary": conversation,
        }
