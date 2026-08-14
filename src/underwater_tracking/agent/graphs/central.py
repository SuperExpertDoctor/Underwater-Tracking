# src/underwater_tracking/agent/graphs/central.py
"""Persistent carrier central graph (spec 8.1, 8.2, plan Task 8).

The carrier assembles the full scenario loop exactly once:

    ingest -> event_monitor -> build_snapshot -> directive_branch

and then routes on the classified event tier (spec 8.2): STRATEGIC events
run the full semantic chain (intent -> trajectory prediction -> strategy
generation -> Verify subgraph -> resource optimization -> plan verification
-> commit), TACTICAL events run prediction and optimization only (no LLM
calls), and INFORMATIONAL events record and report. The directive branch
(spec 10.1, plan Task 10) surfaces the latest applied expert directive onto
the checkpointed state between snapshot assembly and routing; applied
directives route STRATEGIC so the next cycle re-plans with them as hard
constraints. Deferred node errors (e.g. an intent analysis without enough
estimated history, or a Verify fallback with no verified strategy) route to
``handle_error`` so the cycle completes with a recorded error instead of
crashing the run.

References, never raw histories: the live situation resolves under the
deterministic ref ``{scenario_id}:live`` (``live_situation_ref``), the
immutable planning snapshot under ``{scenario_id}:snapshot:{revision}``
(``build_snapshot_ref``), and candidate plans under
``{scenario_id}:candidate:{index}:{revision}``. Checkpoints persist the
state channels; payloads (snapshots, candidates) live in the injected
``store`` (spec 8.4).
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from underwater_tracking.agent.graphs.verify import build_verify_graph
from underwater_tracking.agent.llm import StructuredLLM
from underwater_tracking.agent.nodes.commit import CommitNode, validate_plan
from underwater_tracking.agent.nodes.directives import DirectiveNode
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.intent import (
    BeliefHistoryProvider,
    IntentAnalysisNode,
)
from underwater_tracking.agent.nodes.optimize import OptimizeNode, PlanningConfig
from underwater_tracking.agent.nodes.snapshot import (
    PlanningSnapshot,
    SnapshotNode,
    snapshot_hash,
)
from underwater_tracking.agent.nodes.strategy import StrategyGenerationNode
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import (
    DecisionRecord,
    PredictedTrackRef,
    StrategyProposal,
    StrategySet,
    TrackingPlan,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent, SituationSnapshot
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.simulation.clock import SimulationClock

# Deterministic track predictor port (spec 6.6).
TrajectoryPredictor = Callable[[SituationSnapshot, str], PredictedTrackRef]

# Shared immutable default for node constructors (B008: no call in defaults).
_DEFAULT_PLANNING_CONFIG = PlanningConfig()

# Severity order for the three-tier routing decision (spec 8.2).
_LEVEL_SEVERITY: dict[EventLevel, int] = {
    EventLevel.INFORMATIONAL: 1,
    EventLevel.TACTICAL: 2,
    EventLevel.STRATEGIC: 3,
}


class CentralState(CarrierState, total=False):
    """Carrier cycle state: the persistent channels plus cycle outcomes.

    ``commit_status``/``selected_plan`` surface the commit outcome of the
    cycle, ``node_error`` is the single deferred-error marker routed to
    ``handle_error``, and ``confirmed_intent_labels`` tracks the last
    confirmed intent label per target so the wiring never re-confirms an
    unchanged label.
    """

    commit_status: Literal["committed", "hold_current", "stale", "rejected"] | None
    selected_plan: TrackingPlan | None
    node_error: str | None
    confirmed_intent_labels: dict[str, str]


@dataclass(frozen=True)
class CarrierDependencies:
    """Injected carrier ports (spec 8.1, Task 8 Step 3).

    ``plans``/``events``/``ledger`` are the repositories, ``llm`` the
    structured LLM port, ``predictor`` the deterministic track predictor,
    ``optimizer`` the deterministic planning configuration, and ``clock``
    the simulation time source. ``situation_provider`` resolves the live
    situation under the deterministic ref ``{scenario_id}:live``; the
    optional ``monitor``/``last_bearing_time`` override the spec 8.2
    defaults (the runtime wires one monitor per scenario).
    """

    plans: PlanRepository
    events: EventRepository
    ledger: DecisionLedger
    llm: StructuredLLM[Any]
    predictor: TrajectoryPredictor
    situation_provider: Callable[[str], SituationSnapshot]
    optimizer: PlanningConfig = _DEFAULT_PLANNING_CONFIG
    clock: SimulationClock = field(default_factory=SimulationClock)
    belief_history: BeliefHistoryProvider | None = None
    monitor: EventMonitor | None = None
    last_bearing_time: Callable[[str], int | None] | None = None
    allowed_soft_constraints: tuple[str, ...] = ("energy_reserve_0.1",)
    semantic_repairs: int = 2
    critical_hold_s: int = 30
    target_lost_gap_s: int = 300
    covariance_cap_m2: float = 50_000.0
    model_id: str = "underwater-assistant-model"


def live_situation_ref(scenario_id: str) -> str:
    """Deterministic storage reference of the live situation (spec 8.4)."""
    return f"{scenario_id}:live"


def _highest_level(events: Sequence[RuntimeEvent]) -> EventLevel:
    """The most severe tier among the coalesced events (spec 8.2).

    An empty cycle — no pending events and no monitor-triggered events,
    e.g. a ``CarrierRuntime.resume()`` continuation right after a reopen —
    is routed as informational instead of crashing the invocation.
    """
    if not events:
        return EventLevel.INFORMATIONAL
    return max(events, key=lambda event: _LEVEL_SEVERITY[event.level]).level


class IngestNode:
    """Acquire the live situation and record its revision (spec 8.4).

    Resolves the deterministic live reference ``{scenario_id}:live`` and
    mirrors the situation's snapshot revision into the plan repository so
    the commit-time staleness check passes; the immutable planning
    snapshot is assembled and stored by ``build_snapshot``.
    """

    def __init__(
        self,
        situation_provider: Callable[[str], SituationSnapshot],
        plans: PlanRepository,
    ) -> None:
        self._situation_provider = situation_provider
        self._plans = plans

    def __call__(self, state: CentralState) -> CentralState:
        scenario_id = state.get("scenario_id")
        if not scenario_id:
            raise ValueError("ingest requires scenario_id in state")
        ref = state.get("snapshot_ref")
        if not ref:
            raise ValueError("ingest requires snapshot_ref in state")
        situation = self._situation_provider(ref)
        self._plans.set_snapshot_revision(
            scenario_id, situation.snapshot_revision, snapshot_hash(situation)
        )
        return {}


class EventMonitorNode:
    """Observe the live situation and classify the pending events (spec 8.2).

    Feeds quality samples into the monitor's hysteresis/streak logic and
    the target-loss gate (5-minute ungated-bearing gap AND covariance
    above the scenario cap), classifies every pending event onto its
    three-tier level, and derives the cycle route from the most severe
    coalesced event.
    """

    def __init__(
        self,
        monitor: EventMonitor,
        situation_provider: Callable[[str], SituationSnapshot],
        last_bearing_time: Callable[[str], int | None] | None = None,
    ) -> None:
        self._monitor = monitor
        self._situation_provider = situation_provider
        self._last_bearing_time = last_bearing_time

    def __call__(self, state: CentralState) -> CentralState:
        ref = state.get("snapshot_ref")
        assert ref is not None, "event_monitor requires snapshot_ref in state"
        situation = self._situation_provider(ref)
        observed: list[RuntimeEvent] = []
        for report in situation.group_reports:
            observed.extend(
                self._monitor.observe_quality(
                    report.group_id,
                    situation.sim_time_s,
                    report.quality.ewma,
                    hard_guard_reasons=report.quality.hard_guard_reasons,
                )
            )
            if self._last_bearing_time is not None:
                last_gated = self._last_bearing_time(report.target_id)
                if last_gated is not None:
                    observed.extend(
                        self._monitor.observe_bearing_gap(
                            report.group_id,
                            situation.sim_time_s,
                            last_gated_bearing_s=last_gated,
                            position_covariance_trace=sum(
                                report.belief.covariance[i][i]
                                for i in range(min(len(report.belief.covariance), 2))
                            ),
                        )
                    )
        classified: list[RuntimeEvent] = []
        for event in state.get("pending_events") or ():
            level = self._monitor.classify(event.event_type, payload=event.payload)
            classified.append(
                RuntimeEvent(
                    event_id=event.event_id,
                    scenario_id=event.scenario_id,
                    sim_time_s=event.sim_time_s,
                    event_type=event.event_type,
                    entity_id=event.entity_id,
                    level=level,
                    payload=event.payload,
                )
            )
        coalesced = (*observed, *classified)
        return {"coalesced_events": coalesced, "route": _highest_level(coalesced)}


class IntentWiringNode:
    """Intent analysis plus confirmed-label change detection (spec 12.2).

    Runs the injected ``IntentAnalysisNode`` for every snapshot target,
    then feeds each hypothesis to the monitor's confirmation gate only
    when its label differs from the last confirmed label of the target
    (the monitor cannot detect newness, so the wiring tracks it).
    Confirmed events join ``coalesced_events`` so the route stays
    strategic. A target with fewer than three estimated fixes raises inside
    the inner node; the ValueError is deferred as a node error instead of
    crashing the cycle.
    """

    def __init__(
        self,
        inner: IntentAnalysisNode,
        monitor: EventMonitor,
        situation_provider: Callable[[str], SituationSnapshot],
    ) -> None:
        self._inner = inner
        self._monitor = monitor
        self._situation_provider = situation_provider

    def __call__(self, state: CentralState) -> CentralState:
        try:
            analyzed = self._inner(state)
        except ValueError as exc:
            return {"node_error": f"intent_analysis failed: {exc}"}
        ref = state.get("snapshot_ref")
        if ref is None:
            return {"node_error": "intent_analysis requires snapshot_ref in state"}
        situation = self._situation_provider(ref)
        confirmed = dict(state.get("confirmed_intent_labels") or {})
        emitted: list[RuntimeEvent] = []
        for target_id, hypothesis in analyzed["intent_hypotheses"].items():
            if confirmed.get(target_id) == hypothesis.label:
                continue
            events = self._monitor.observe_intent_analysis(
                target_id,
                situation.sim_time_s,
                leading_label=hypothesis.label,
                confidence=hypothesis.confidence,
                runner_up_confidence=max(hypothesis.alternatives.values(), default=0.0),
            )
            if events:
                confirmed[target_id] = hypothesis.label
                emitted.extend(events)
        return {
            "intent_hypotheses": analyzed["intent_hypotheses"],
            "llm_provenance": analyzed["llm_provenance"],
            "confirmed_intent_labels": confirmed,
            "coalesced_events": (*(state.get("coalesced_events") or ()), *emitted),
        }


class TrajectoryPredictionNode:
    """Deterministic per-target track prediction (spec 6.6).

    Each target's estimated track is predicted through the injected
    deterministic predictor; references are attached to the state's
    ``predictions`` channel.
    """

    def __init__(
        self,
        predictor: TrajectoryPredictor,
        situation_provider: Callable[[str], SituationSnapshot],
    ) -> None:
        self._predictor = predictor
        self._situation_provider = situation_provider

    def __call__(self, state: CentralState) -> CentralState:
        ref = state.get("snapshot_ref")
        assert ref is not None, "trajectory_prediction requires snapshot_ref in state"
        situation = self._situation_provider(ref)
        predictions: dict[str, PredictedTrackRef] = {}
        for target_id in sorted(
            {report.target_id for report in situation.group_reports}
        ):
            predictions[target_id] = self._predictor(situation, target_id)
        return {"predictions": predictions}


class VerifyStrategyNode:
    """Per-proposal semantic verification via the bounded Verify subgraph.

    Each proposal is validated against the planning snapshot's targets and
    evidence with bounded repairs (spec 8.3). A proposal whose repair
    budget is exhausted — including the emergency fallback with no
    evidence — resolves to None and is dropped; when no proposal survives,
    the cycle continues through ``handle_error`` with a recorded error.
    """

    def __init__(
        self,
        verify_graph: Any,
        snapshot_provider: Callable[[str], PlanningSnapshot],
        allowed_soft_constraints: tuple[str, ...] = (),
        max_repairs: int = 2,
    ) -> None:
        self._verify_graph = verify_graph
        self._snapshot_provider = snapshot_provider
        self._allowed_soft_constraints = allowed_soft_constraints
        self._max_repairs = max_repairs

    def __call__(self, state: CentralState) -> CentralState:
        ref = state.get("snapshot_ref")
        if ref is None:
            return {"node_error": "verify_strategy requires snapshot_ref in state"}
        snapshot = self._snapshot_provider(ref)
        strategy_set = state.get("strategy_set")
        if strategy_set is None or not strategy_set.proposals:
            return {
                "node_error": "verify_strategy requires a non-empty strategy_set"
            }
        target_ids = tuple(
            sorted({report.target_id for report in snapshot.situation.group_reports})
        )
        evidence_ids = tuple(
            sorted(
                {
                    observation_id
                    for report in snapshot.situation.group_reports
                    for observation_id in report.belief.source_observation_ids
                }
            )
        )
        verified: list[StrategyProposal] = []
        for proposal in strategy_set:
            outcome = self._verify_graph.invoke(
                {
                    "candidate": proposal,
                    "target_ids": target_ids,
                    "evidence_ids": evidence_ids,
                    "allowed_soft_constraints": self._allowed_soft_constraints,
                    "max_repairs": self._max_repairs,
                    "scenario_id": snapshot.scenario_id,
                    "sim_time_s": snapshot.sim_time_s,
                }
            )
            candidate = outcome.get("verified_strategy")
            if isinstance(candidate, StrategyProposal):
                verified.append(candidate)
        if not verified:
            return {
                "node_error": (
                    "verify_strategy: no verified strategy survived semantic verification"
                ),
                "strategy_set": strategy_set,
            }
        return {
            "strategy_set": StrategySet(
                trigger_event_ids=strategy_set.trigger_event_ids,
                proposals=tuple(verified),
            )
        }


def _continuation_strategy_set(
    state: CentralState, snapshot: PlanningSnapshot
) -> StrategySet:
    """Deterministic no-LLM strategy set for tactical cycles (spec 8.2).

    Reuses the checkpointed strategic strategy set when one exists;
    otherwise every target is continued at full priority with the default
    requirements, evidenced by the group reports' own observations.
    """
    prior = state.get("strategy_set")
    if prior is not None and prior.proposals:
        return prior
    targets = tuple(
        dict.fromkeys(report.target_id for report in snapshot.situation.group_reports)
    )
    return StrategySet(
        trigger_event_ids=(),
        proposals=(
            StrategyProposal(
                concept="hold_current",
                target_priorities={target: 1.0 for target in targets},
                required_quality={target: 0.7 for target in targets},
                reinforcement_policy={
                    target: "release_when_stable" for target in targets
                },
                releasable_soft_constraints=("energy_reserve_0.1",),
                evidence_ids=tuple(
                    sorted(
                        {
                            observation_id
                            for report in snapshot.situation.group_reports
                            for observation_id in report.belief.source_observation_ids
                        }
                    )
                ),
                rationale="tactical continuation without semantic replanning",
            ),
        ),
    )


class ResourceOptimizerNode:
    """Optimize the strategies into candidate plans (spec 15.1, 8.2).

    Tactical cycles reuse the checkpointed strategy set, or fall back to a
    deterministic hold-current continuation — no LLM calls on the tactical
    branch. The selected candidate is stored by the inner OptimizeNode;
    any infeasibility is deferred as a node error.
    """

    def __init__(
        self,
        inner: OptimizeNode,
        snapshot_provider: Callable[[str], PlanningSnapshot],
    ) -> None:
        self._inner = inner
        self._snapshot_provider = snapshot_provider

    def __call__(self, state: CentralState) -> CentralState:
        ref = state.get("snapshot_ref")
        if ref is None:
            return {"node_error": "resource_optimizer requires snapshot_ref in state"}
        snapshot = self._snapshot_provider(ref)
        strategy_set = state.get("strategy_set")
        if strategy_set is None or not strategy_set.proposals:
            strategy_set = _continuation_strategy_set(state, snapshot)
        merged = cast(CentralState, {**state, "strategy_set": strategy_set})
        try:
            return cast(CentralState, self._inner(merged))
        except ValueError as exc:
            return {"node_error": f"resource_optimizer failed: {exc}"}


class VerifyPlanNode:
    """Independent validation of the selected candidate plan (spec 15.3).

    The selected plan is re-checked against the immutable planning
    snapshot by the commit validator; issues defer to ``handle_error`` so
    the cycle completes with a recorded error instead of committing an
    invalid plan.
    """

    def __init__(
        self,
        snapshot_provider: Callable[[str], PlanningSnapshot],
        store: MutableMapping[str, Any],
        config: PlanningConfig = _DEFAULT_PLANNING_CONFIG,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        self._store = store
        self._config = config

    def __call__(self, state: CentralState) -> CentralState:
        ref = state.get("selected_plan_ref")
        if ref is None:
            return {"node_error": "verify_plan requires selected_plan_ref in state"}
        selected = self._store.get(ref)
        if selected is None:
            return {"node_error": f"verify_plan: no candidate stored under {ref!r}"}
        snapshot_ref = state.get("snapshot_ref")
        assert snapshot_ref is not None, "verify_plan requires snapshot_ref in state"
        snapshot = self._snapshot_provider(snapshot_ref)
        issues = validate_plan(snapshot, selected, self._config)
        if issues:
            return {
                "node_error": "verify_plan rejected the selected plan: "
                + "; ".join(
                    f"{issue.code} on {issue.field}" for issue in issues[:3]
                )
            }
        return {"selected_plan": selected, "selected_plan_ref": ref}


class CommitPlanNode:
    """Atomic versioned commit of the selected plan (spec 15.3).

    Rejects (with a recorded status) when no candidate is selected; a
    committed or held plan surfaces the final plan on the state.
    """

    def __init__(
        self,
        inner: CommitNode,
        store: MutableMapping[str, Any],
    ) -> None:
        self._inner = inner
        self._store = store

    def __call__(self, state: CentralState) -> CentralState:
        ref = state.get("selected_plan_ref")
        if ref is None:
            return {"commit_status": "rejected", "selected_plan": None}
        candidate = self._store.get(ref)
        if candidate is None:
            return {"commit_status": "rejected", "selected_plan": None}
        result = self._inner(state, candidate)
        if result["commit_status"] in ("committed", "hold_current"):
            return {
                "commit_status": result["commit_status"],
                "selected_plan": candidate,
            }
        return {"commit_status": result["commit_status"], "selected_plan": None}


class RecordDecisionNode:
    """Persist one decision record and the coalesced events (spec 16, 8.4).

    Strategic and tactical cycles that selected a plan record the full
    decision with its candidate refs and final plan; informational cycles
    only append their events. Events are appended with their classified
    severity so the store replays them in insertion order.
    """

    def __init__(
        self,
        events: EventRepository,
        ledger: DecisionLedger,
        snapshot_provider: Callable[[str], PlanningSnapshot],
        store: MutableMapping[str, Any],
    ) -> None:
        self._events = events
        self._ledger = ledger
        self._snapshot_provider = snapshot_provider
        self._store = store

    def __call__(self, state: CentralState) -> CentralState:
        for event in state.get("coalesced_events") or ():
            self._events.append(
                event_id=event.event_id,
                event_type=event.event_type,
                scenario_id=event.scenario_id,
                sim_time_s=event.sim_time_s,
                payload=event.payload,
                target_id=event.entity_id,
                severity=event.level.value,
            )
        route = state.get("route")
        ref = state.get("selected_plan_ref")
        if route in (EventLevel.STRATEGIC, EventLevel.TACTICAL) and ref is not None:
            selected = self._store.get(ref)
            if selected is not None:
                snapshot_ref = state.get("snapshot_ref")
                assert snapshot_ref is not None, "record_decision requires snapshot_ref"
                snapshot = self._snapshot_provider(snapshot_ref)
                self._record_decision(state, snapshot, selected)
        return {}

    def _record_decision(
        self,
        state: CentralState,
        snapshot: PlanningSnapshot,
        selected: TrackingPlan,
    ) -> None:
        strategy_set = state.get("strategy_set")
        provenance = next(iter(state.get("llm_provenance", {}).values()), None)
        self._ledger.record(
            DecisionRecord(
                decision_id=(
                    f"{selected.scenario_id}:decision:{snapshot.snapshot_revision}"
                ),
                scenario_id=selected.scenario_id,
                sim_time_s=snapshot.sim_time_s,
                trigger_event_ids=(
                    strategy_set.trigger_event_ids if strategy_set is not None else ()
                ),
                snapshot_revision=snapshot.snapshot_revision,
                snapshot_hash=snapshot.digest,
                input_evidence_ids=tuple(
                    sorted(
                        {
                            observation_id
                            for report in snapshot.situation.group_reports
                            for observation_id in report.belief.source_observation_ids
                        }
                    )
                ),
                model_version=provenance.model if provenance is not None else "",
                prompt_version=(
                    provenance.prompt_version if provenance is not None else ""
                ),
                candidates=(
                    strategy_set.proposals if strategy_set is not None else ()
                ),
                candidate_plan_ids=tuple(
                    candidate.plan_id
                    for candidate_ref in state.get("candidate_plan_refs") or ()
                    if (candidate := self._store.get(candidate_ref)) is not None
                ),
                final_plan_id=selected.plan_id,
            )
        )


class ProgressReportNode:
    """Publish a deterministic progress summary (spec 8.1)."""

    def __init__(
        self,
        snapshot_provider: Callable[[str], PlanningSnapshot],
    ) -> None:
        self._snapshot_provider = snapshot_provider

    def __call__(self, state: CentralState) -> CentralState:
        ref = state.get("snapshot_ref")
        assert ref is not None, "progress_report requires snapshot_ref in state"
        snapshot = self._snapshot_provider(ref)
        plan = state.get("selected_plan")
        commit = state.get("commit_status")
        summary = (
            f"{snapshot.scenario_id} cycle: {len(snapshot.situation.group_reports)}"
            f" group(s), snapshot revision {snapshot.snapshot_revision}"
            + (
                f", plan {plan.plan_id} {commit}"
                if plan is not None and commit is not None
                else ""
            )
        )
        return {"output_messages": (*(state.get("output_messages") or ()), summary)}


class HandleErrorNode:
    """Record one deferred node error and let the run continue (spec 18)."""

    def __call__(self, state: CentralState) -> CentralState:
        message = state.get("node_error")
        if message is None:
            return {"node_error": None}
        return {
            "node_error": None,
            "errors": (*(state.get("errors") or ()), message),
            "output_messages": (*(state.get("output_messages") or ()), f"error: {message}"),
        }


def _route_events(state: CentralState) -> Literal["strategic", "tactical", "informational"]:
    """Three-tier routing decision (spec 8.2)."""
    route = state.get("route") or EventLevel.INFORMATIONAL
    if route == EventLevel.STRATEGIC:
        return "strategic"
    if route == EventLevel.TACTICAL:
        return "tactical"
    return "informational"


def _route_directive_branch(
    state: CentralState,
) -> Literal["strategic", "tactical", "informational", "error"]:
    """After the directive branch: defer branch errors, then the tier route.

    The directive branch (spec 10.1) resolves applied-directive events onto
    the state; a branch error (e.g. an event referencing an unknown
    directive id) defers to ``handle_error`` so the cycle still completes,
    while clean cycles continue onto the regular three-tier routing.
    """
    if state.get("node_error") is not None:
        return "error"
    return _route_events(state)


def _route_after_prediction(state: CentralState) -> Literal["strategic", "tactical"]:
    """After prediction: strategic runs the full semantic chain, tactical
    continues with optimization only (spec 8.2)."""
    return "tactical" if state.get("route") == EventLevel.TACTICAL else "strategic"


def _route_error(state: CentralState) -> Literal["continue", "error"]:
    """Defer any recorded node error to ``handle_error``."""
    return "error" if state.get("node_error") is not None else "continue"


def build_carrier_graph(
    dependencies: CarrierDependencies,
    checkpointer: BaseCheckpointSaver[Any],
    store: MutableMapping[str, Any],
) -> Any:
    """Assemble the persistent carrier central graph exactly once (spec 8.1).

    ``dependencies`` are the injected ports (repositories, LLM, predictor,
    optimizer, clock, providers); ``checkpointer`` persists the state
    channels across cycles (spec 8.4); ``store`` holds the immutable
    payloads — planning snapshots and candidate plans — referenced by the
    checkpoint. The wiring is deterministic: no randomness anywhere.
    """
    monitor = dependencies.monitor or EventMonitor(
        critical_hold_s=dependencies.critical_hold_s,
        target_lost_gap_s=dependencies.target_lost_gap_s,
        covariance_cap_m2=dependencies.covariance_cap_m2,
    )
    planning_provider: Callable[[str], PlanningSnapshot] = lambda ref: store[ref]
    situation_provider = dependencies.situation_provider
    intent_situation_provider: Callable[[str], SituationSnapshot] = (
        lambda ref: store[ref].situation
    )

    builder = StateGraph(CentralState)
    builder.add_node("ingest", IngestNode(situation_provider, dependencies.plans))
    builder.add_node(
        "event_monitor",
        EventMonitorNode(
            monitor, situation_provider, dependencies.last_bearing_time
        ),
    )
    builder.add_node(
        "build_snapshot",
        SnapshotNode(
            snapshot_provider=situation_provider,
            active_plan_provider=dependencies.plans.get_active,
            directives_provider=lambda scenario_id: tuple(
                dependencies.ledger.list_directives(scenario_id, status="applied")
            ),
            store=store,
        ),
    )
    builder.add_node(
        "intent_analysis",
        IntentWiringNode(
            IntentAnalysisNode(
                dependencies.llm,
                model_id=dependencies.model_id,
                belief_history=dependencies.belief_history,
                snapshot_provider=intent_situation_provider,
            ),
            monitor,
            intent_situation_provider,
        ),
    )
    builder.add_node(
        "trajectory_prediction",
        TrajectoryPredictionNode(dependencies.predictor, intent_situation_provider),
    )
    builder.add_node(
        "strategy_generation",
        StrategyGenerationNode(dependencies.llm, model_id=dependencies.model_id),
    )
    builder.add_node(
        "verify_strategy",
        VerifyStrategyNode(
            build_verify_graph(
                dependencies.llm,
                model_id=dependencies.model_id,
                allowed_soft_constraints=dependencies.allowed_soft_constraints,
            ),
            planning_provider,
            dependencies.allowed_soft_constraints,
            dependencies.semantic_repairs,
        ),
    )
    builder.add_node(
        "resource_optimizer",
        ResourceOptimizerNode(
            OptimizeNode(
                snapshot_provider=planning_provider,
                store=store,
                config=dependencies.optimizer,
            ),
            planning_provider,
        ),
    )
    builder.add_node(
        "verify_plan",
        VerifyPlanNode(planning_provider, store, dependencies.optimizer),
    )
    builder.add_node(
        "commit_plan",
        CommitPlanNode(
            CommitNode(
                repository=dependencies.plans,
                snapshot_provider=planning_provider,
                config=dependencies.optimizer,
            ),
            store,
        ),
    )
    builder.add_node(
        "record_decision",
        RecordDecisionNode(
            dependencies.events, dependencies.ledger, planning_provider, store
        ),
    )
    builder.add_node("progress_report", ProgressReportNode(planning_provider))
    builder.add_node("handle_error", HandleErrorNode())
    builder.add_node("directive_branch", DirectiveNode(dependencies.ledger))

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "event_monitor")
    builder.add_edge("event_monitor", "build_snapshot")
    builder.add_edge("build_snapshot", "directive_branch")
    builder.add_conditional_edges(
        "directive_branch",
        _route_directive_branch,
        {
            "strategic": "intent_analysis",
            "tactical": "trajectory_prediction",
            "informational": "record_decision",
            "error": "handle_error",
        },
    )
    builder.add_conditional_edges(
        "intent_analysis",
        _route_error,
        {"continue": "trajectory_prediction", "error": "handle_error"},
    )
    builder.add_conditional_edges(
        "trajectory_prediction",
        _route_after_prediction,
        {"strategic": "strategy_generation", "tactical": "resource_optimizer"},
    )
    builder.add_edge("strategy_generation", "verify_strategy")
    builder.add_conditional_edges(
        "verify_strategy",
        _route_error,
        {"continue": "resource_optimizer", "error": "handle_error"},
    )
    builder.add_conditional_edges(
        "resource_optimizer",
        _route_error,
        {"continue": "verify_plan", "error": "handle_error"},
    )
    builder.add_conditional_edges(
        "verify_plan",
        _route_error,
        {"continue": "commit_plan", "error": "handle_error"},
    )
    builder.add_edge("commit_plan", "record_decision")
    builder.add_edge("record_decision", "progress_report")
    builder.add_edge("progress_report", END)
    builder.add_edge("handle_error", END)
    return builder.compile(checkpointer=checkpointer)
