# src/underwater_tracking/agent/graphs/history.py
"""History compression subgraph and planning-context assembly (spec 9).

The graph runs the single ``compress_history`` node: evaluate the
deterministic trigger, and when it fires append one summary per namespace
to the long-term memory store — never deleting source records.

``build_planning_context`` assembles the bounded planning input the planner
prompts are allowed to load (spec 9): the current snapshot, the last valid
plan, the applied expert directives, the last critical events, and only the
historical summaries whose evidence matches the target/event evidence of
the current cycle — rendered under a deterministic character/token budget
with truncation at whole-record boundaries.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from underwater_tracking.agent.nodes.history import (
    CompressHistoryNode,
    HistoryState,
    HistorySummary,
    HistoryTriggerPolicy,
    render_event,
    list_summaries,
)
from underwater_tracking.domain.agent_models import ExpertDirective, TrackingPlan
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.persistence.events import EventRepository, StoredEvent
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository

_DEFAULT_BUDGET_CHARS = 20_000
_DEFAULT_CRITICAL_LIMIT = 10
_CRITICAL_SEVERITIES = ("warning", "critical")
EVENT_LIST_LIMIT = 10_000


def build_history_graph(
    events: EventRepository,
    ledger: DecisionLedger,
    store: BaseStore,
    *,
    trigger: HistoryTriggerPolicy | None = None,
) -> Any:
    """Compile the history-compression subgraph (spec 9, plan Task 9).

    ``events``/``ledger`` are the source repositories the summaries are
    derived from, ``store`` is the long-term memory store the summaries are
    appended to by namespace, and ``trigger`` is the deterministic
    threshold policy (defaults to the agent-config defaults). The graph is
    stateless: each invoke is an independent compression window.
    """
    builder = StateGraph(HistoryState)
    builder.add_node(
        "compress_history",
        CompressHistoryNode(events, ledger, store, trigger=trigger),
    )
    builder.add_edge(START, "compress_history")
    builder.add_edge("compress_history", END)
    return builder.compile()


@dataclass(frozen=True)
class PlanningContext:
    """Bounded planning input assembled by :func:`build_planning_context`.

    ``records`` are the whole rendered records admitted within the budget
    (truncation happens only at record boundaries, in stable priority
    order); ``text`` is their newline-joined rendering. The structured
    fields mirror exactly what the planning prompt may load per spec 9.
    """

    snapshot: SituationSnapshot | None
    active_plan: TrackingPlan | None
    applied_directives: tuple[ExpertDirective, ...]
    critical_events: tuple[StoredEvent, ...]
    summaries: tuple[HistorySummary, ...]
    records: tuple[str, ...]
    text: str
    text_chars: int
    budget_chars: int


def build_planning_context(
    *,
    scenario_id: str,
    window_end_s: int,
    events: EventRepository,
    ledger: DecisionLedger,
    plans: PlanRepository,
    store: BaseStore,
    relevant_evidence_ids: Collection[str] = (),
    budget_chars: int = _DEFAULT_BUDGET_CHARS,
    critical_severities: Sequence[str] = _CRITICAL_SEVERITIES,
    critical_limit: int = _DEFAULT_CRITICAL_LIMIT,
    snapshot_provider: Callable[[str], SituationSnapshot] | None = None,
    snapshot_ref: str | None = None,
) -> PlanningContext:
    """Assemble the bounded planning context for one cycle (spec 9).

    Loads only: the current snapshot (when a provider and ref are given),
    the last broadcast plan, the applied directives, the last critical
    events up to ``window_end_s``, and the stored historical summaries
    whose evidence ids intersect ``relevant_evidence_ids`` (all summaries
    when the relevance filter is empty). The rendering is deterministic:
    fixed section priority, stable record ordering, and whole-record
    truncation once ``budget_chars`` is exhausted.
    """
    in_window = [
        event
        for event in events.list_events(scenario_id=scenario_id, limit=EVENT_LIST_LIMIT)
        if event.sim_time_s <= window_end_s
    ]
    critical = [
        event
        for event in in_window
        if event.severity in critical_severities
    ][-max(1, critical_limit):]
    snapshot = (
        snapshot_provider(snapshot_ref)
        if snapshot_provider is not None and snapshot_ref is not None
        else None
    )
    active_plan = plans.get_active(scenario_id)
    directives = tuple(ledger.list_directives(scenario_id, status="applied"))
    summaries = list_summaries(store, scenario_id)
    if relevant_evidence_ids:
        relevant = frozenset(relevant_evidence_ids)
        summaries = tuple(
            summary for summary in summaries if relevant.intersection(summary.evidence_ids)
        )

    records = _render_records(snapshot, active_plan, directives, critical, summaries)
    admitted, text = _fit_budget(records, budget_chars)
    return PlanningContext(
        snapshot=snapshot,
        active_plan=active_plan,
        applied_directives=directives,
        critical_events=tuple(critical),
        summaries=summaries,
        records=admitted,
        text=text,
        text_chars=len(text),
        budget_chars=budget_chars,
    )


def _render_records(
    snapshot: SituationSnapshot | None,
    active_plan: TrackingPlan | None,
    directives: tuple[ExpertDirective, ...],
    critical: Sequence[StoredEvent],
    summaries: Sequence[HistorySummary],
) -> list[str]:
    """Whole-record renderings in stable priority order (spec 9)."""
    records: list[str] = []
    if snapshot is not None:
        groups = ", ".join(
            f"{report.group_id}:{report.target_id}" for report in snapshot.group_reports
        )
        records.append(
            f"snapshot rev={snapshot.snapshot_revision} t={snapshot.sim_time_s}"
            f" groups=[{groups}]"
        )
    if active_plan is not None:
        members = ", ".join(
            f"{target}=[{','.join(group)}]"
            for target, group in sorted(active_plan.member_ids_by_target.items())
        )
        records.append(
            f"plan {active_plan.plan_id} rev={active_plan.revision}"
            f" status={active_plan.status} members=[{members}]"
        )
    records.extend(
        f"directive {directive.directive_id} [{directive.status}]:"
        f" {directive.raw_text}"
        for directive in directives
    )
    records.extend(render_event(event) for event in critical)
    records.extend(_render_summary(summary) for summary in summaries)
    return records


def _render_summary(summary: HistorySummary) -> str:
    facts = "; ".join(summary.facts)
    risks = "; ".join(summary.unresolved_risks) or "-"
    evidence = ",".join(summary.evidence_ids) or "-"
    return (
        f"summary {summary.summary_id} [{summary.start_time_s},{summary.end_time_s}]"
        f" facts=[{facts}] risks=[{risks}] evidence=[{evidence}]"
    )


def _fit_budget(
    records: Sequence[str], budget_chars: int
) -> tuple[tuple[str, ...], str]:
    """Deterministic budget fit: whole records only, stable priority order.

    Records are admitted in order while the newline-joined text fits
    ``budget_chars``; a record that would exceed the budget ends the fit
    (never a partial record).
    """
    admitted: list[str] = []
    used = 0
    for record in records:
        cost = len(record) + (1 if admitted else 0)
        if used + cost > budget_chars:
            break
        admitted.append(record)
        used += cost
    return tuple(admitted), "\n".join(admitted)
