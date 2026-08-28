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
constraints. The question branch (spec 10.2, plan Task 11) surfaces the
latest answered question run onto the ``latest_question`` channel; question
events classify INFORMATIONAL, so the branch never re-plans. Deferred node
errors (e.g. an intent analysis without enough estimated history, or a
Verify failure with no verified strategy) route to ``handle_error`` so the
cycle completes with a recorded error instead of crashing the run.

References, never raw histories: the live situation resolves under the
deterministic ref ``{scenario_id}:live`` (``live_situation_ref``), the
immutable planning snapshot under ``{scenario_id}:snapshot:{revision}``
(``build_snapshot_ref``), and candidate plans under
``{scenario_id}:candidate:{index}:{revision}``. Checkpoints persist the
state channels; payloads (snapshots, candidates) live in the injected
``store`` (spec 8.4).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from math import isfinite, sqrt
import os
from time import monotonic
from typing import Any, Literal, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from underwater_tracking.agent.graphs.verify import build_verify_graph
from underwater_tracking.agent.event_policy import (
    EventDisposition,
    PlanImpactAssessment,
    evaluate_plan_impact,
)
from underwater_tracking.agent.llm import LLMError, StructuredLLM
from underwater_tracking.agent.nodes.active_verification import ActiveVerificationNode
from underwater_tracking.agent.nodes.commit import (
    CommitNode,
    EpochCommitPort,
    validate_plan,
)
from underwater_tracking.agent.nodes.directives import DirectiveNode
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.intent import (
    BeliefHistoryProvider,
    IntentAnalysisNode,
    _intent_evidence_ids,
)
from underwater_tracking.agent.nodes.optimize import OptimizeNode, PlanningConfig
from underwater_tracking.agent.nodes.questions import QuestionBranchNode
from underwater_tracking.agent.nodes.regional_strategy import RegionalStrategyGenerationNode
from underwater_tracking.agent.nodes.regions import RegionGenerationNode
from underwater_tracking.planning.execution_strategy import ExecutionStrategyRevisionNode
from underwater_tracking.agent.nodes.snapshot import (
    PlanningSnapshot,
    SnapshotNode,
    snapshot_hash,
)
from underwater_tracking.agent.state import CarrierState, RegionalReplanReason
from underwater_tracking.config.models import (
    IntentChangeConfirmation,
    RuntimeRetentionConfig,
    TrajectoryDiffConfig,
)
from underwater_tracking.domain.agent_models import (
    DecisionRecord,
    IntentHypothesis,
    PredictedTrackRef,
    StrategyProposal,
    StrategySet,
    IntentVerificationCallRef,
    TrajectoryDiffGateState,
    TrajectoryDiffResult,
    TrackingPlan,
)
from underwater_tracking.domain.models import (
    Contact,
    ContactClassification,
    EventAudience,
    EventLevel,
    RuntimeEvent,
    SituationSnapshot,
)
from underwater_tracking.domain.event_registry import EVENT_REGISTRY, event_definition
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.domain.planning_epoch_models import (
    EpochCommitResult,
    EpochFailureCategory,
    PlanningEpoch,
)
from underwater_tracking.domain.prediction_models import AcceptedPrediction, PredictionHealth
from underwater_tracking.domain.regional_models import (
    GridSpec,
    RegionalStrategySet,
    UUVRegionalStrategySet,
)
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.memory import ShortTermContextRepository
from underwater_tracking.memory.service import MemoryService
from underwater_tracking.persistence.plans import PlanRepository, StaleSnapshotError
from underwater_tracking.planning.mission_validation import (
    validate_executable_mission_plan,
)
from underwater_tracking.planning.reservations import ReservationRegistry
from underwater_tracking.prediction.diff import compare_predicted_tracks
from underwater_tracking.prediction.diff_gate import advance_diff_gate
from underwater_tracking.simulation.clock import SimulationClock
from underwater_tracking.world_model.models import RuleWorldModelConfig

# Deterministic track predictor port (spec 6.6).
TrajectoryPredictor = Callable[[SituationSnapshot, str], AcceptedPrediction]

# Shared immutable default for node constructors (B008: no call in defaults).
_DEFAULT_PLANNING_CONFIG = PlanningConfig()

# These public engineering bounds form the initial search envelope from
# intelligence priors and stay separate from private simulator truth.
_PUBLIC_TARGET_SPEED_BOUND_MPS = 14.0
_PUBLIC_SEARCH_SWEEP_SPEED_MPS = 3.0
# A prior provides a center but no heading evidence. Its envelope grows at the
# configured physical maximum until deployed UUVs produce observations.
_PUBLIC_SEARCH_RADIUS_GROWTH_MPS = _PUBLIC_TARGET_SPEED_BOUND_MPS

# Severity order for the three-tier routing decision (spec 8.2).
_LEVEL_SEVERITY: dict[EventLevel, int] = {
    EventLevel.CRITICAL: 4,
    EventLevel.INFORMATIONAL: 1,
    EventLevel.TACTICAL: 2,
    EventLevel.STRATEGIC: 3,
}

# These operational changes invalidate a regional policy and therefore need a
# fresh LLM strategy.  They intentionally live at the carrier boundary: the
# generic EventMonitor remains strict about event types it owns.
REGIONAL_REPLAN_EVENT_TYPES: dict[RegionalReplanReason, str] = {
    "regional_feedback": "regional_feedback_received",
    "endurance": "endurance_threshold_crossed",
    "communication_link": "communication_link_lost",
    "covariance": "covariance_threshold_exceeded",
    "target_reacquired": "target_reacquired",
}
_REGIONAL_REPLAN_REASONS = {
    event_type: reason for reason, event_type in REGIONAL_REPLAN_EVENT_TYPES.items()
}
_REGIONAL_REPLAN_REASONS["group_quality_critical"] = "regional_feedback"


class CentralState(CarrierState, total=False):
    """Carrier cycle state: the persistent channels plus cycle outcomes.

    ``commit_status``/``selected_plan`` surface the commit outcome of the
    cycle, ``node_error`` is the single deferred-error marker routed to
    ``handle_error``, and ``confirmed_intent_labels`` tracks the last
    confirmed intent label per target so the wiring never re-confirms an
    unchanged label.
    """

    commit_status: Literal[
        "committed",
        "hold_current",
        "stale",
        "invalidated",
        "rejected",
        "failed",
    ] | None
    selected_plan: TrackingPlan | None
    node_error: str | None
    confirmed_intent_labels: dict[str, str]
    epoch_finalization_route: Literal["record", "end"] | None


class PlanningEpochInvariantError(RuntimeError):
    """Raised when an active epoch reaches a graph boundary without one result."""


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
    trajectory_diff_config: TrajectoryDiffConfig = field(
        default_factory=TrajectoryDiffConfig
    )
    intent_change_confirmation: IntentChangeConfirmation = field(
        default_factory=IntentChangeConfirmation
    )
    grid_spec: GridSpec = field(default_factory=GridSpec)
    clock: SimulationClock = field(default_factory=SimulationClock)
    belief_history: BeliefHistoryProvider | None = None
    monitor: EventMonitor | None = None
    prediction_intent_monitor: EventMonitor | None = None
    last_bearing_time: Callable[[str], int | None] | None = None
    allowed_soft_constraints: tuple[str, ...] = ("energy_reserve_0.1",)
    semantic_repairs: int = 2
    regional_batch_size: int = 4
    regional_max_concurrency: int = 3
    semantic_correction_attempts: int = 1
    critical_hold_s: int = 30
    target_lost_gap_s: int = 300
    covariance_cap_m2: float = 50_000.0
    model_id: str = "underwater-assistant-model"
    reservations: ReservationRegistry | None = None
    uuv_only: bool = False
    execution_hard_stale_s: float = 900.0
    retention: RuntimeRetentionConfig = field(default_factory=RuntimeRetentionConfig)
    current_snapshot_revision: Callable[[], int] | None = None
    memory_service: MemoryService | None = None
    short_term_repository: ShortTermContextRepository | None = None
    memory_port: object | None = None
    planning_epoch_provider: Callable[[], PlanningEpoch | None] | None = None
    epoch_commit_port: EpochCommitPort | None = None
    world_model_config: RuleWorldModelConfig | None = None
    execution_strategy_node: ExecutionStrategyRevisionNode | None = None


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
        active_plan_provider: Callable[[str], TrackingPlan | None] | None = None,
    ) -> None:
        self._monitor = monitor
        self._situation_provider = situation_provider
        self._last_bearing_time = last_bearing_time
        self._active_plan_provider = active_plan_provider

    def __call__(self, state: CentralState) -> CentralState:
        ref = state.get("snapshot_ref")
        assert ref is not None, "event_monitor requires snapshot_ref in state"
        situation = self._situation_provider(ref)
        plan_context = self._plan_context(situation)
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
        target_ids_by_group = {
            report.group_id: report.target_id for report in situation.group_reports
        }
        target_ids_by_region = cast(
            Mapping[str, str], plan_context["target_ids_by_region"]
        )
        classified: list[RuntimeEvent] = []
        replan_reasons: list[RegionalReplanReason] = []
        source_events = (
            *observed,
            *(
                event
                for event in (state.get("pending_events") or ())
                if EventAudience.BLUE_PLANNING in event.audiences
            ),
        )
        for event in source_events:
            reason = _REGIONAL_REPLAN_REASONS.get(event.event_type)
            payload = dict(event.payload)
            if (
                "target_id" not in payload
                and event.entity_id in target_ids_by_group
            ):
                payload["target_id"] = target_ids_by_group[event.entity_id]
            region_id = payload.get("region_id")
            if (
                "target_id" not in payload
                and isinstance(region_id, str)
                and region_id in target_ids_by_region
            ):
                payload["target_id"] = target_ids_by_region[region_id]
            normalized_event = event.model_copy(update={"payload": payload})
            assessment = self._assess(normalized_event, plan_context)
            try:
                level = _event_level(normalized_event, assessment, self._monitor)
            except (TypeError, ValueError) as exc:
                return {"node_error": f"event_monitor failed: {exc}"}
            if reason is not None and assessment.plan_impact:
                replan_reasons.append(reason)
            classified.append(
                RuntimeEvent(
                    event_id=normalized_event.event_id,
                    scenario_id=normalized_event.scenario_id,
                    sim_time_s=normalized_event.sim_time_s,
                    event_type=normalized_event.event_type,
                    entity_id=normalized_event.entity_id,
                    level=level,
                    audiences=normalized_event.audiences,
                    payload={
                        **payload,
                        "plan_impact": assessment.plan_impact,
                        "impact_reason": assessment.reason,
                    },
                )
            )
        coalesced = _coalesce_plan_events(classified)
        lost_target_ids = set(state.get("lost_target_ids") or ())

        def event_target_id(event: RuntimeEvent) -> str | None:
            target_id = event.payload.get("target_id")
            if isinstance(target_id, str) and target_id:
                return target_id
            if event.entity_id is None:
                return None
            return target_ids_by_group.get(event.entity_id, event.entity_id)

        for event in coalesced:
            target_id = event_target_id(event)
            if event.event_type == "target_lost" and target_id is not None:
                lost_target_ids.add(target_id)
            elif event.event_type == "target_reacquired" and target_id is not None:
                lost_target_ids.discard(target_id)
        return {
            "coalesced_events": coalesced,
            "route": _highest_level(coalesced),
            "strategic_replan_reasons": tuple(dict.fromkeys(replan_reasons)),
            "known_target_ids": tuple(
                sorted(
                    set(state.get("known_target_ids") or ())
                    | {report.target_id for report in situation.group_reports}
                    | {
                        contact.contact_id
                        for contact in _known_submarine_contacts(situation)
                    }
                )
            ),
            "lost_target_ids": tuple(sorted(lost_target_ids)),
        }

    def _plan_context(self, situation: SituationSnapshot) -> dict[str, object]:
        scenario_id = getattr(situation, "scenario_id", self._monitor._scenario_id)
        plan = (
            self._active_plan_provider(scenario_id)
            if self._active_plan_provider is not None
            else None
        )
        tasks = getattr(plan, "region_tasks", {}) if plan is not None else {}
        active_tasks = tuple(
            task
            for task in tasks.values()
            if getattr(task, "assignment_status", "planned") == "active"
        )
        active_region_ids = tuple(sorted(task.region_id for task in active_tasks))
        active_target_ids = tuple(
            sorted(
                set(getattr(plan, "member_ids_by_target", {}) or {})
                | {task.target_id for task in active_tasks}
            )
        )
        active_uuv_ids = tuple(
            sorted(
                set(getattr(plan, "active_uuv_ids", ()) or ())
                | {
                    uuv_id
                    for task in active_tasks
                    for uuv_id in getattr(task, "assigned_uuv_ids", ())
                }
            )
        )
        required_quality = dict(getattr(plan, "required_quality", {}) or {})
        for task in active_tasks:
            required_quality[task.target_id] = max(
                required_quality.get(task.target_id, 0.0),
                float(getattr(task, "required_quality", 0.0)),
            )
        return {
            "active_target_ids": active_target_ids,
            "active_region_ids": active_region_ids,
            "active_uuv_ids": active_uuv_ids,
            "quality_by_target": {
                report.target_id: report.quality.ewma
                for report in situation.group_reports
            },
            "required_quality_by_target": required_quality,
            "target_ids_by_region": {
                str(task.region_id): str(task.target_id) for task in active_tasks
            },
        }

    @staticmethod
    def _assess(
        event: RuntimeEvent, context: Mapping[str, object]
    ) -> PlanImpactAssessment:
        payload = event.payload
        return evaluate_plan_impact(
            event,
            active_target_ids=cast(Sequence[str], context["active_target_ids"]),
            active_region_ids=cast(Sequence[str], context["active_region_ids"]),
            active_uuv_ids=cast(Sequence[str], context["active_uuv_ids"]),
            quality_by_target=cast(Mapping[str, float], context["quality_by_target"]),
            required_quality_by_target=cast(
                Mapping[str, float], context["required_quality_by_target"]
            ),
            target_corridor_changed=payload.get("target_corridor_changed") is True,
            resource_feasible=payload.get("resource_feasible") is not False,
            communication_healthy=payload.get("communication_healthy") is not False,
            time_window_feasible=payload.get("time_window_feasible") is not False,
        )


def _event_level(
    event: RuntimeEvent,
    assessment: PlanImpactAssessment,
    monitor: EventMonitor,
) -> EventLevel:
    if event_definition(event.event_type).plan_impact_policy == "always":
        return EventLevel.STRATEGIC
    # A tactical observation may require deterministic replanning without
    # becoming a semantic-strategy event.  Preserve that registered boundary
    # even when the producer marks the current plan as affected.
    if assessment.disposition is EventDisposition.TACTICAL:
        return EventLevel.TACTICAL
    if assessment.plan_impact:
        return EventLevel.STRATEGIC
    if assessment.disposition in {
        EventDisposition.AUDIT_ONLY,
        EventDisposition.CANDIDATE,
    }:
        if assessment.reason.startswith("unknown event"):
            return monitor.classify(event.event_type, payload=event.payload)
        return EventLevel.INFORMATIONAL
    if assessment.disposition is EventDisposition.KEY:
        return EventLevel.TACTICAL
    return monitor.classify(event.event_type, payload=event.payload)


_EVENT_COALESCE_FAMILY_BY_TYPE = {
    event_type: definition.coalescing_family
    for event_type, definition in EVENT_REGISTRY.items()
    if definition.coalescing_family is not None
}


def _coalesce_plan_events(events: Sequence[RuntimeEvent]) -> tuple[RuntimeEvent, ...]:
    """Collapse duplicate source signals before the planning boundary.

    The raw source ids remain in the selected event payload for evidence
    tracing, while one physical abnormal episode produces one routing input.
    """
    buckets: dict[tuple[str, str], list[RuntimeEvent]] = {}
    ordered_keys: list[tuple[str, str]] = []
    for index, event in enumerate(events):
        family = _EVENT_COALESCE_FAMILY_BY_TYPE.get(event.event_type)
        if family is None:
            key = (f"event:{index}", event.event_id)
        else:
            subject = (
                event.payload.get("target_id")
                or event.payload.get("region_id")
                or event.entity_id
                or event.event_id
            )
            key = (family, str(subject))
        if key not in buckets:
            buckets[key] = []
            ordered_keys.append(key)
        buckets[key].append(event)

    result: list[RuntimeEvent] = []
    for key in ordered_keys:
        members = buckets[key]
        selected = max(
            members,
            key=lambda event: (
                event.payload.get("plan_impact") is True,
                _LEVEL_SEVERITY[event.level],
                event.sim_time_s,
                event.event_id,
            ),
        )
        if len(members) > 1:
            payload = {
                **selected.payload,
                "coalesced_event_ids": tuple(
                    sorted(event.event_id for event in members)
                ),
                "coalesced_event_types": tuple(
                    sorted({event.event_type for event in members})
                ),
            }
            selected = selected.model_copy(update={"payload": payload})
        result.append(selected)
    return tuple(result)


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
        emitted_events: list[RuntimeEvent] = []
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
                emitted_events.extend(events)
        existing_events = tuple(state.get("coalesced_events") or ())
        events_by_id = {event.event_id: event for event in existing_events}
        events_by_id.update({event.event_id: event for event in emitted_events})
        return {
            "intent_hypotheses": analyzed["intent_hypotheses"],
            "llm_provenance": analyzed["llm_provenance"],
            "deterministic_intents": analyzed.get("deterministic_intents", {}),
            "intent_latches": analyzed.get("intent_latches", {}),
            "confirmed_intent_labels": confirmed,
            "coalesced_events": tuple(events_by_id.values()),
        }


class PredictionIntentWiringNode:
    """Verify a latched forecast divergence through semantic Intent LLM calls."""

    def __init__(
        self,
        inner: IntentAnalysisNode,
        monitor: EventMonitor,
        situation_provider: Callable[[str], SituationSnapshot],
        confirmation: IntentChangeConfirmation | None = None,
    ) -> None:
        self._inner = inner
        self._monitor = monitor
        self._situation_provider = situation_provider
        self._confirmation = confirmation or IntentChangeConfirmation()

    def __call__(self, state: CentralState) -> CentralState:
        requested_target_ids = (
            state.get("prediction_intent_verification_target_ids") or ()
        )
        if not requested_target_ids:
            return {
                "prediction_intent_confirmed": False,
                "prediction_intent_verification_target_ids": (),
            }
        ref = state.get("snapshot_ref")
        if ref is None:
            return {
                "node_error": "prediction_intent_analysis requires snapshot_ref in state"
            }
        situation = self._situation_provider(ref)
        gates = dict(state.get("prediction_diff_gates") or {})
        diffs = dict(state.get("prediction_diffs") or {})
        target_ids = tuple(
            target_id
            for target_id in requested_target_ids
            if (
                (gate := gates.get(target_id)) is not None
                and (diff := diffs.get(target_id)) is not None
                and (
                    gate.last_intent_verification_sim_time_s is None
                    or situation.sim_time_s
                    > gate.last_intent_verification_sim_time_s
                )
                and diff.diff_id != gate.last_intent_verification_diff_id
            )
        )
        if not target_ids:
            return {
                "prediction_intent_confirmed": False,
                "prediction_intent_verification_target_ids": requested_target_ids,
                "prediction_diff_gates": gates,
                "prediction_diffs": diffs,
            }
        try:
            analyzed = self._inner({**state, "intent_target_ids": target_ids})
        except ValueError as exc:
            return {"node_error": f"prediction_intent_analysis failed: {exc}"}
        hypotheses = analyzed.get("intent_hypotheses") or {}
        provenance = analyzed.get("llm_provenance") or {}
        confirmed = dict(state.get("confirmed_intent_labels") or {})
        remaining = [
            target_id
            for target_id in requested_target_ids
            if target_id not in target_ids
        ]
        confirmed_events: list[RuntimeEvent] = []

        for target_id in target_ids:
            hypothesis = hypotheses.get(target_id)
            gate = gates.get(target_id)
            diff = diffs.get(target_id)
            metadata = provenance.get(f"intent:{target_id}")
            if hypothesis is None or gate is None or diff is None or metadata is None:
                return {
                    "node_error": (
                        f"prediction_intent_analysis lacks auditable inputs for {target_id}"
                    )
                }
            previous_hypothesis = (state.get("intent_hypotheses") or {}).get(target_id)
            previous_label = (
                confirmed.get(target_id)
                or gate.intent_baseline_label
                or (None if previous_hypothesis is None else previous_hypothesis.label)
            )
            runner_up = max(hypothesis.alternatives.values(), default=0.0)
            passed = (
                hypothesis.confidence >= self._confirmation.confidence
                and hypothesis.confidence - runner_up >= self._confirmation.margin
            )
            call_ref = IntentVerificationCallRef(
                operation="intent",
                model=metadata.model,
                prompt_version=metadata.prompt_version,
                request_hash=metadata.request_hash,
                response_hash=metadata.response_hash,
                sim_time_s=metadata.sim_time_s,
                scenario_id=metadata.scenario_id,
            )
            verification_label = gate.intent_verification_label
            if verification_label != hypothesis.label:
                verification_calls = (call_ref,)
            else:
                verification_calls = (*gate.intent_verification_calls, call_ref)
            verification_calls = verification_calls[-diff.confirmation_cycles :]
            verification_update = {
                "intent_verification_calls": verification_calls,
                "intent_verification_label": hypothesis.label,
                "last_intent_verification_sim_time_s": situation.sim_time_s,
                "last_intent_verification_diff_id": diff.diff_id,
            }
            if previous_label is None or previous_label == hypothesis.label or not passed:
                gates[target_id] = gate.model_copy(
                    update={
                        "verification_pending": False,
                        "intent_verification_calls": (),
                        "intent_verification_label": None,
                        "intent_baseline_label": (
                            hypothesis.label
                            if passed and (
                                previous_label is None
                                or previous_label == hypothesis.label
                            )
                            else gate.intent_baseline_label
                        ),
                        "last_intent_verification_sim_time_s": situation.sim_time_s,
                        "last_intent_verification_diff_id": diff.diff_id,
                    }
                )
                continue

            if len(verification_calls) < diff.confirmation_cycles:
                remaining.append(target_id)
                gates[target_id] = gate.model_copy(
                    update={
                        "verification_pending": True,
                        **verification_update,
                    }
                )
                diffs[target_id] = diff.model_copy(
                    update={"gate_transition": "verifying"}
                )
                continue

            events = self._monitor.emit_confirmed_intent_change(
                target_id,
                situation.sim_time_s,
                leading_label=hypothesis.label,
                confidence=hypothesis.confidence,
                runner_up_confidence=runner_up,
            )

            confirmed[target_id] = hypothesis.label
            gates[target_id] = gate.model_copy(
                update={
                    "verification_pending": False,
                    **verification_update,
                    "intent_baseline_label": hypothesis.label,
                }
            )
            diffs[target_id] = diff.model_copy(
                update={"gate_transition": "confirmed"}
            )
            for event in events:
                confirmed_events.append(
                    event.model_copy(
                        update={
                            "payload": {
                                **event.payload,
                                "previous_label": previous_label,
                                "diff_id": gate.suspicion_diff_id or diff.diff_id,
                                "verification_diff_id": diff.diff_id,
                                "suspicion_event_id": gate.suspicion_event_id,
                                "observation_ids": diff.current_evidence_ids,
                                "evidence_ids": tuple(sorted(hypothesis.evidence_ids)),
                                "llm_operation": metadata.operation,
                                "llm_model": metadata.model,
                                "llm_prompt_version": metadata.prompt_version,
                                "llm_request_hash": metadata.request_hash,
                                "llm_response_hash": metadata.response_hash,
                                "intent_llm_calls": tuple(
                                    call.model_dump(mode="json")
                                    for call in verification_calls
                                ),
                                "source": "real_intent_llm",
                            }
                        }
                    )
                )

        return {
            "intent_hypotheses": hypotheses,
            "llm_provenance": provenance,
            "deterministic_intents": analyzed.get("deterministic_intents", {}),
            "intent_latches": analyzed.get("intent_latches", {}),
            "confirmed_intent_labels": confirmed,
            "prediction_diffs": diffs,
            "prediction_diff_gates": gates,
            "prediction_intent_verification_target_ids": tuple(sorted(remaining)),
            "prediction_intent_confirmed": bool(confirmed_events),
            "coalesced_events": (
                *(state.get("coalesced_events") or ()),
                *confirmed_events,
            ),
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
        *,
        uuv_only: bool = False,
        diff_config: TrajectoryDiffConfig | None = None,
    ) -> None:
        self._predictor = predictor
        self._situation_provider = situation_provider
        self._uuv_only = uuv_only
        self._diff_config = diff_config or TrajectoryDiffConfig()

    def __call__(self, state: CentralState) -> CentralState:
        ref = state.get("snapshot_ref")
        assert ref is not None, "trajectory_prediction requires snapshot_ref in state"
        situation = self._situation_provider(ref)
        prediction_revision = int(
            getattr(situation, "snapshot_revision", situation.sim_time_s)
        )
        cached_predictions = state.get("predictions") or {}
        active_prior_ids_by_target: dict[str, set[str]] = {}
        for prior in situation.target_search_priors:
            active_prior_ids_by_target.setdefault(prior.target_id, set()).add(
                prior.prior_id
            )
        stale_public_targets = {
            target_id
            for target_id, prediction in cached_predictions.items()
            if prediction.prediction_regime == "public_prior"
            and not any(
                prediction.prediction_id.startswith(
                    f"prior-prediction:{prior_id}:"
                )
                for prior_id in active_prior_ids_by_target.get(target_id, ())
            )
        }
        same_prediction_revision = (
            state.get("prediction_snapshot_revision") == prediction_revision
        )
        if same_prediction_revision and cached_predictions and stale_public_targets:
            seeded = _prior_seeded_planning_inputs(situation)
            refreshed_predictions = dict(seeded["predictions"])
            refreshed_predictions.update(
                {
                    target_id: prediction
                    for target_id, prediction in cached_predictions.items()
                    if prediction.prediction_regime != "public_prior"
                }
            )
            refreshed_intents = dict(seeded["intent_hypotheses"])
            refreshed_intents.update(
                {
                    target_id: hypothesis
                    for target_id, hypothesis in (
                        state.get("intent_hypotheses") or {}
                    ).items()
                    if hypothesis.model_id != "public-target-search-prior"
                }
            )
            unchanged_target_ids = {
                target_id
                for target_id, prediction in refreshed_predictions.items()
                if cached_predictions.get(target_id) == prediction
            }
            return {
                "predictions": refreshed_predictions,
                "accepted_predictions": {
                    **dict(seeded.get("accepted_predictions") or {}),
                    **{
                        target_id: accepted
                        for target_id, accepted in (
                            state.get("accepted_predictions") or {}
                        ).items()
                        if target_id in unchanged_target_ids
                        and refreshed_predictions[target_id].prediction_regime
                        != "public_prior"
                    },
                },
                "intent_hypotheses": refreshed_intents,
                "prediction_diffs": {
                    target_id: diff
                    for target_id, diff in (state.get("prediction_diffs") or {}).items()
                    if target_id in unchanged_target_ids
                },
                "prediction_diff_gates": {
                    target_id: gate
                    for target_id, gate in (
                        state.get("prediction_diff_gates") or {}
                    ).items()
                    if target_id in unchanged_target_ids
                },
                "prediction_snapshot_revision": prediction_revision,
                "prediction_intent_verification_target_ids": tuple(
                    target_id
                    for target_id in (
                        state.get("prediction_intent_verification_target_ids") or ()
                    )
                    if target_id in unchanged_target_ids
                ),
                "prediction_intent_confirmed": bool(
                    state.get("prediction_intent_confirmed")
                ),
                "coalesced_events": tuple(state.get("coalesced_events") or ()),
            }
        if (
            same_prediction_revision
            and cached_predictions
        ):
            # CarrierRuntime may have produced this deterministic fragment at
            # the observation boundary while a provider cycle was in flight.
            # Keep the exact evidence/gate transition and let this graph cycle
            # consume it for intent verification or planning.
            return {
                "predictions": dict(state.get("predictions") or {}),
                "accepted_predictions": dict(
                    state.get("accepted_predictions") or {}
                ),
                "prediction_diffs": dict(state.get("prediction_diffs") or {}),
                "prediction_diff_gates": dict(
                    state.get("prediction_diff_gates") or {}
                ),
                "prediction_snapshot_revision": prediction_revision,
                "prediction_intent_verification_target_ids": tuple(
                    state.get("prediction_intent_verification_target_ids") or ()
                ),
                "prediction_intent_confirmed": bool(
                    state.get("prediction_intent_confirmed")
                ),
                "coalesced_events": tuple(state.get("coalesced_events") or ()),
            }
        target_ids = {report.target_id for report in situation.group_reports}
        additional: CentralState = {}
        if not target_ids and self._uuv_only and situation.target_search_priors:
            seeded = _prior_seeded_planning_inputs(situation)
            predictions = seeded["predictions"]
            additional = {
                "intent_hypotheses": seeded["intent_hypotheses"],
                "accepted_predictions": seeded["accepted_predictions"],
            }
        elif not target_ids:
            # A temporary loss of public contact is not a new prediction. Keep
            # the last observation-derived forecast and gate until a later
            # observation can produce a comparable update. Public-prior
            # envelopes expire with their source intelligence and cannot be
            # retained as executable planning inputs.
            retained_predictions = {
                target_id: prediction
                for target_id, prediction in cached_predictions.items()
                if prediction.prediction_regime != "public_prior"
            }
            retained_intents = {
                target_id: hypothesis
                for target_id, hypothesis in (
                    state.get("intent_hypotheses") or {}
                ).items()
                if hypothesis.model_id != "public-target-search-prior"
            }
            retained_target_ids = set(retained_predictions)
            return {
                "predictions": retained_predictions,
                "accepted_predictions": {
                    target_id: accepted
                    for target_id, accepted in (
                        state.get("accepted_predictions") or {}
                    ).items()
                    if target_id in retained_target_ids
                },
                "intent_hypotheses": retained_intents,
                "prediction_diffs": {
                    target_id: diff
                    for target_id, diff in (state.get("prediction_diffs") or {}).items()
                    if target_id in retained_target_ids
                },
                "prediction_diff_gates": {
                    target_id: gate
                    for target_id, gate in (
                        state.get("prediction_diff_gates") or {}
                    ).items()
                    if target_id in retained_target_ids
                },
                "prediction_snapshot_revision": prediction_revision,
                "prediction_intent_verification_target_ids": (),
                "prediction_intent_confirmed": False,
                "coalesced_events": tuple(state.get("coalesced_events") or ()),
            }
        else:
            accepted_predictions = {
                target_id: self._predictor(situation, target_id)
                for target_id in sorted(target_ids)
            }
            predictions = {
                target_id: accepted.prediction
                for target_id, accepted in accepted_predictions.items()
                if accepted.health.status != "unavailable"
                and accepted.prediction is not None
            }
            additional["accepted_predictions"] = accepted_predictions
        return {
            **additional,
            **self._diff_updates(state, situation, predictions),
            "prediction_snapshot_revision": prediction_revision,
        }

    def _diff_updates(
        self,
        state: CentralState,
        situation: SituationSnapshot,
        predictions: Mapping[str, PredictedTrackRef],
    ) -> CentralState:
        previous_predictions = state.get("predictions") or {}
        previous_gates = state.get("prediction_diff_gates") or {}
        diffs: dict[str, TrajectoryDiffResult] = {}
        gates: dict[str, TrajectoryDiffGateState] = {}
        pending: list[str] = []
        emitted: list[RuntimeEvent] = []
        existing_events = state.get("coalesced_events") or ()
        existing_event_ids = {event.event_id for event in existing_events}

        for target_id, prediction in sorted(predictions.items()):
            diff = compare_predicted_tracks(
                previous_predictions.get(target_id),
                prediction,
                self._diff_config,
            )
            decision = advance_diff_gate(
                previous_gates.get(target_id),
                diff,
                self._diff_config,
            )
            gate = decision.state
            if decision.emit_suspicion:
                event = self._suspicion_event(situation, diff, gate)
                gate = gate.model_copy(update={"suspicion_event_id": event.event_id})
                if event.event_id not in existing_event_ids:
                    emitted.append(event)
                    existing_event_ids.add(event.event_id)
            if decision.request_intent_verification:
                pending.append(target_id)
            transition = (
                "reset"
                if decision.reset
                else "suspected"
                if decision.emit_suspicion
                else "verifying"
                if decision.request_intent_verification
                else "accumulating"
                if gate.consecutive_count
                else "none"
            )
            gates[target_id] = gate
            diffs[target_id] = diff.model_copy(
                update={
                    "consecutive_count": gate.consecutive_count,
                    "latched": gate.latched,
                    "gate_transition": transition,
                }
            )

        return {
            "predictions": dict(predictions),
            "prediction_diffs": diffs,
            "prediction_diff_gates": gates,
            "prediction_intent_verification_target_ids": tuple(sorted(pending)),
            "prediction_intent_confirmed": False,
            "coalesced_events": (*existing_events, *emitted),
        }

    @staticmethod
    def _suspicion_event(
        situation: SituationSnapshot,
        diff: TrajectoryDiffResult,
        gate: TrajectoryDiffGateState,
    ) -> RuntimeEvent:
        definition = event_definition("target_intent_change_suspected")
        event_id = (
            f"{situation.scenario_id}:target_intent_change_suspected:"
            f"{diff.target_id}:{situation.sim_time_s}"
        )
        return RuntimeEvent(
            event_id=event_id,
            scenario_id=situation.scenario_id,
            sim_time_s=situation.sim_time_s,
            event_type=definition.event_type,
            entity_id=diff.target_id,
            level=definition.default_level,
            audiences=definition.audiences,
            payload={
                "diff_id": diff.diff_id,
                "previous_prediction_id": diff.previous_prediction_id,
                "current_prediction_id": diff.current_prediction_id,
                "observation_ids": diff.current_evidence_ids,
                "absolute_rms_m": diff.absolute_rms_m,
                "normalized_rms": diff.normalized_rms,
                "exceeded": diff.exceeded,
                "absolute_floor_m": diff.absolute_floor_m,
                "normalized_threshold": diff.normalized_threshold,
                "consecutive_count": gate.consecutive_count,
                "overlap_start_s": diff.overlap_start_s,
                "overlap_end_s": diff.overlap_end_s,
                "comparison_step_s": diff.comparison_step_s,
                "sample_count": diff.sample_count,
                "source": "trajectory_diff",
            },
        )


def _known_submarine_contacts(
    situation: SituationSnapshot,
) -> tuple[Contact, ...]:
    """Return identified submarine contacts with usable public geometry."""
    return tuple(
        sorted(
            (
                contact
                for contact in getattr(situation, "contacts", ())
                if contact.classification is ContactClassification.SUBMARINE
                and contact.estimated_position_xy is not None
            ),
            key=lambda contact: contact.contact_id,
        )
    )


def _prior_seeded_planning_inputs(
    situation: SituationSnapshot,
) -> CentralState:
    """Build candidate-only planning inputs from public search intelligence.

    This corridor is a planning artifact, not a target estimate: it has no
    sensor history, is never copied into ``SituationSnapshot.group_reports``,
    and is replaced by the first real fused belief after deployment.
    """
    hypotheses: dict[str, IntentHypothesis] = {}
    predictions: dict[str, PredictedTrackRef] = {}
    accepted_predictions: dict[str, AcceptedPrediction] = {}
    for prior in situation.target_search_priors:
        horizon_s = float(prior.valid_until_s - situation.sim_time_s)
        if horizon_s <= 0.0:
            continue
        sample_count = 7
        sample_step_s = horizon_s / (sample_count - 1)
        times = tuple(
            float(situation.sim_time_s + index * sample_step_s)
            for index in range(sample_count)
        )
        points = tuple(
            (
                prior.center_xy[0]
                + _PUBLIC_SEARCH_SWEEP_SPEED_MPS * (time_s - situation.sim_time_s),
                prior.center_xy[1],
            )
            for time_s in times
        )
        radius_m = sqrt(max(prior.covariance_xy[0][0], prior.covariance_xy[1][1]))
        corridor_radii = tuple(
            radius_m
            + _PUBLIC_SEARCH_RADIUS_GROWTH_MPS
            * max(0.0, time_s - situation.sim_time_s)
            for time_s in times
        )
        hypotheses[prior.target_id] = IntentHypothesis(
            label="unknown",
            confidence=prior.confidence,
            evidence_ids=(prior.prior_id,),
            model_id="public-target-search-prior",
            prompt_version="prior-seeded-v1",
        )
        prediction = PredictedTrackRef(
            prediction_id=f"prior-prediction:{prior.prior_id}:{situation.sim_time_s}",
            target_id=prior.target_id,
            sim_time_s=situation.sim_time_s,
            horizon_s=horizon_s,
            sample_step_s=sample_step_s,
            times_s=times,
            points_xy=points,
            corridor_radius_m=corridor_radii,
            source_belief_history_ids=(),
            fallback_used=True,
            fallback_reason="public_target_search_envelope",
            prediction_regime="public_prior",
            imm_model_probabilities={},
        )
        predictions[prior.target_id] = prediction
        accepted_predictions[prior.target_id] = _assess_public_prior_envelope(
            prediction,
            situation=situation,
        )
    return {
        "intent_hypotheses": hypotheses,
        "predictions": predictions,
        "accepted_predictions": accepted_predictions,
    }


def _assess_public_prior_envelope(
    prediction: PredictedTrackRef,
    *,
    situation: SituationSnapshot,
) -> AcceptedPrediction:
    reasons = ["public_target_search_envelope"]
    point_count = len(prediction.points_xy)
    structurally_valid = (
        prediction.prediction_regime == "public_prior"
        and prediction.sim_time_s == situation.sim_time_s
        and point_count > 1
        and len(prediction.times_s) == point_count
        and len(prediction.corridor_radius_m) == point_count
        and all(isfinite(value) for value in prediction.times_s)
        and all(
            isfinite(coordinate)
            for point in prediction.points_xy
            for coordinate in point
        )
        and all(
            isfinite(radius) and radius >= 0.0
            for radius in prediction.corridor_radius_m
        )
    )
    if not structurally_valid:
        reasons.append("public_prior_envelope_invalid")
    health = PredictionHealth(
        status="degraded" if structurally_valid else "unavailable",
        regime="short_history",
        reason_codes=tuple(reasons),
        source_track_age_s=0.0,
        clipped_point_fraction=0.0,
        maximum_radius_m=max(prediction.corridor_radius_m, default=0.0),
        raw_prediction_id=prediction.prediction_id,
    )
    return AcceptedPrediction(
        prediction=prediction if structurally_valid else None,
        health=health,
    )


class RegionalGenerationWiringNode:
    """Defer deterministic regionalization errors to the cycle error route."""

    def __init__(self, inner: RegionGenerationNode) -> None:
        self._inner = inner

    def __call__(self, state: CentralState) -> CentralState:
        started = monotonic()
        _trace_regional_node("regional_generation:start")
        try:
            result = self._inner(state)
            _trace_regional_node(
                f"regional_generation:done:{monotonic() - started:.3f}s"
            )
            return cast(CentralState, result)
        except (TypeError, ValueError) as exc:
            _trace_regional_node(
                f"regional_generation:error:{monotonic() - started:.3f}s:{exc}"
            )
            return {"node_error": f"regional_generation failed: {exc}"}


class RegionalStrategyWiringNode:
    """Keep provider failures retryable while deferring semantic failures."""

    def __init__(self, inner: RegionalStrategyGenerationNode) -> None:
        self._inner = inner

    def __call__(self, state: CentralState) -> CentralState:
        started = monotonic()
        _trace_regional_node("regional_strategy:start")
        try:
            result = self._inner(state)
            _trace_regional_node(
                f"regional_strategy:done:{monotonic() - started:.3f}s"
            )
            return cast(CentralState, result)
        except LLMError:
            _trace_regional_node(
                f"regional_strategy:llm-error:{monotonic() - started:.3f}s"
            )
            raise
        except (TypeError, ValueError) as exc:
            _trace_regional_node(
                f"regional_strategy:error:{monotonic() - started:.3f}s:{exc}"
            )
            return {"node_error": f"regional_strategy failed: {exc}"}


class RegionalStrategyToStrategySetNode:
    """Adapt regional policy goals to the existing semantic verifier.

    Regional policies remain authoritative for members, tracking mode, and
    relay topology.  This compatibility proposal only supplies target-level
    fields required by verification; the optimizer materializes the original
    policies into the resulting regional tasks.
    """

    def __call__(self, state: CentralState) -> CentralState:
        regional_plans = state.get("regional_plans") or {}
        policies = state.get("regional_policies") or {}
        if not regional_plans or not policies:
            return {
                "node_error": (
                    "regional_strategy_adapter requires regional plans and policies"
                )
            }

        target_priorities: dict[str, float] = {}
        required_quality: dict[str, float] = {}
        evidence_ids: set[str] = set()
        semantic_proposals = state.get("execution_strategy_proposals") or {}
        semantic_reports = state.get("strategy_validation_reports") or {}
        for target_id, plan in sorted(regional_plans.items()):
            policy_set = policies.get(target_id)
            if not isinstance(
                policy_set, (RegionalStrategySet, UUVRegionalStrategySet)
            ) or not policy_set.policies:
                return {
                    "node_error": (
                        f"regional_strategy_adapter requires policies for target {target_id!r}"
                    )
                }
            semantic = semantic_proposals.get(target_id)
            report = semantic_reports.get(target_id)
            if semantic is not None and report is not None and report.valid:
                target_priorities[target_id] = max(
                    slot.priority for slot in semantic.region_slots
                )
            else:
                target_priorities[target_id] = max(
                    policy.priority for policy in policy_set.policies
                )
            required_quality[target_id] = max(
                policy.required_quality for policy in policy_set.policies
            )
            evidence_ids.update(plan.evidence_ids)
            for policy in policy_set.policies:
                evidence_ids.update(policy.evidence_ids)
            if semantic is not None and report is not None and report.valid:
                evidence_ids.update(semantic.evidence_ids)
                for slot in semantic.region_slots:
                    evidence_ids.update(slot.evidence_ids)

        return {
            "regional_policies": policies,
            "strategy_set": StrategySet(
                trigger_event_ids=tuple(
                    event.event_id for event in state.get("coalesced_events") or ()
                ),
                proposals=(
                    StrategyProposal(
                        concept="balanced",
                        target_priorities=target_priorities,
                        required_quality=required_quality,
                        reinforcement_policy={
                            target_id: "release_when_stable"
                            for target_id in target_priorities
                        },
                        releasable_soft_constraints=("energy_reserve_0.1",),
                        evidence_ids=tuple(sorted(evidence_ids)),
                        rationale=(
                            "compatibility view of the approved regional policies"
                        ),
                    ),
                ),
            )
        }


def assess_regional_replan_events(
    situation: SituationSnapshot,
    *,
    active_plan: TrackingPlan | None,
    known_target_ids: Sequence[str],
    lost_target_ids: Sequence[str] = (),
    covariance_cap_m2: float,
    endurance_threshold: float = 0.2,
) -> tuple[RuntimeEvent, ...]:
    """Build deterministic, evidence-backed regional-policy invalidations."""
    observed = {report.target_id: report for report in situation.group_reports}
    events: list[RuntimeEvent] = []

    def emit(event_type: str, entity_id: str, evidence: Sequence[str]) -> None:
        evidence_ids = tuple(sorted(set(evidence))) or (
            f"state:{entity_id}:{situation.sim_time_s}",
        )
        events.append(
            RuntimeEvent(
                event_id=(
                    f"{situation.scenario_id}:{event_type}:{entity_id}:"
                    f"{situation.sim_time_s}"
                ),
                scenario_id=situation.scenario_id,
                sim_time_s=situation.sim_time_s,
                event_type=event_type,
                entity_id=entity_id,
                level=EventLevel.INFORMATIONAL,
                payload={"evidence": evidence_ids},
            )
        )

    for target_id, report in sorted(observed.items()):
        evidence = tuple(getattr(report.belief, "source_observation_ids", ()))
        if target_id in lost_target_ids:
            emit("target_reacquired", target_id, evidence)
        covariance = getattr(report.belief, "covariance", ())
        trace = sum(covariance[index][index] for index in range(min(2, len(covariance))))
        if trace > covariance_cap_m2:
            emit("covariance_threshold_exceeded", target_id, evidence)

    region_tasks = getattr(active_plan, "region_tasks", {}) if active_plan else {}
    active_region_tasks = tuple(
        task for task in region_tasks.values() if task.assignment_status == "active"
    )
    assigned_uuv_ids = {
        uuv_id
        for task in active_region_tasks
        for uuv_id in task.assigned_uuv_ids
    }
    if active_plan is not None and not region_tasks:
        assigned_uuv_ids.update(
            uuv_id
            for members in active_plan.member_ids_by_target.values()
            for uuv_id in members
        )
    # The legacy TrackingPlan projection keeps UUV-only region tasks in the
    # PLANNED state even while the physical mission controller has dispatched
    # them.  Use the live execution groups as the authoritative resource
    # assignment for endurance checks; this remains public runtime state.
    assigned_uuv_ids.update(
        member_id
        for group in getattr(situation, "execution_groups", ())
        if getattr(group, "mode", None) in {"active_scan", "passive_track"}
        for member_id in getattr(group, "member_ids", ())
    )
    for uuv in situation.uuvs:
        if uuv.uuv_id not in assigned_uuv_ids:
            continue
        if uuv.energy_fraction < endurance_threshold:
            emit("endurance_threshold_crossed", uuv.uuv_id, ())

    if region_tasks:
        links = {
            f"{link.source_id}->{link.target_id}"
            for link in (
                situation.platform_snapshot.communication_links
                if situation.platform_snapshot is not None
                else ()
            )
        }
        for task in region_tasks.values():
            missing = set(task.communication_links) - links
            if missing:
                emit("communication_link_lost", task.region_id, tuple(sorted(missing)))

    return tuple(events)


def _trace_regional_node(message: str) -> None:
    """Enable short-lived node timing diagnostics without changing normal logs."""
    if os.environ.get("UNDERWATER_TRACKING_DEBUG_GRAPH") == "1":
        print(f"[graph] {message}", flush=True)


class VerifyStrategyNode:
    """Per-proposal semantic verification via the bounded Verify subgraph.

    Each proposal is validated against the planning snapshot's targets and
    evidence with bounded repairs (spec 8.3). A proposal whose repair
    budget is exhausted — including a content failure with no
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
            sorted(
                {
                    *(report.target_id for report in snapshot.situation.group_reports),
                    *(state.get("regional_plans") or {}),
                    *(
                        contact.contact_id
                        for contact in _known_submarine_contacts(snapshot.situation)
                    ),
                }
            )
        )
        evidence_ids = {
            observation_id
            for report in snapshot.situation.group_reports
            for observation_id in report.belief.source_observation_ids
        }
        for report in snapshot.situation.group_reports:
            if not report.belief.source_observation_ids:
                evidence_ids.update(
                    _intent_evidence_ids(snapshot.situation, report.target_id)
                )
        for regional_plan in (state.get("regional_plans") or {}).values():
            evidence_ids.update(regional_plan.evidence_ids)
            evidence_ids.update(
                evidence_id
                for cell in regional_plan.cells
                for evidence_id in cell.evidence_ids
            )
        evidence_ids.update(
            evidence_id
            for hypothesis in (state.get("intent_hypotheses") or {}).values()
            for evidence_id in hypothesis.evidence_ids
        )
        verified_evidence_ids = tuple(sorted(evidence_ids))
        verified: list[StrategyProposal] = []
        for proposal in strategy_set:
            outcome = self._verify_graph.invoke(
                {
                    "candidate": proposal,
                    "target_ids": target_ids,
                    "evidence_ids": verified_evidence_ids,
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


def _continuation_strategy_set(snapshot: PlanningSnapshot) -> StrategySet:
    """Continue an already approved plan during a tactical cycle (spec 8.2).

    Every target is continued at full priority with the default
    requirements, evidenced by the group reports' own observations. This is
    only valid when an approved active plan already exists; it is not a
    replacement for a missing strategic LLM decision.
    """
    targets = tuple(
        dict.fromkeys(report.target_id for report in snapshot.situation.group_reports)
    )
    evidence_ids = {
        observation_id
        for report in snapshot.situation.group_reports
        for observation_id in report.belief.source_observation_ids
    }
    if not evidence_ids:
        for target_id in targets:
            evidence_ids.update(
                _intent_evidence_ids(snapshot.situation, target_id)
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
                evidence_ids=tuple(sorted(evidence_ids)),
                rationale="tactical continuation of the approved active plan",
            ),
        ),
    )


class ResourceOptimizerNode:
    """Optimize the strategies into candidate plans (spec 15.1, 8.2).

    Tactical cycles reuse the checkpointed strategy set, or derive a
    deterministic hold-current continuation from the approved active plan —
    no LLM calls on the tactical branch. A checkpointed proposal is only reused while every one of its
    evidence ids still resolves in the current snapshot (observation ids
    rotate with simulation time, so stale evidence would deterministically
    fail the commit validator's evidence check); a proposal with stale
    evidence is dropped while ALL of its targets still exist in the
    situation, and when none survive the continuation is used only if an
    active plan exists. A proposal covering a vanished target is kept unchanged:
    the optimizer's deterministic infeasibility ("no group report for
    target ...") is the designed signal for a disappearing target, not a
    silent re-plan over the survivors. The selected candidate is stored by
    the inner OptimizeNode; any infeasibility is deferred as a node error.
    """

    def __init__(
        self,
        inner: OptimizeNode,
        snapshot_provider: Callable[[str], PlanningSnapshot],
        *,
        uuv_only: bool = False,
    ) -> None:
        self._inner = inner
        self._snapshot_provider = snapshot_provider
        self._uuv_only = uuv_only

    def __call__(self, state: CentralState) -> CentralState:
        ref = state.get("snapshot_ref")
        if ref is None:
            return {"node_error": "resource_optimizer requires snapshot_ref in state"}
        snapshot = self._snapshot_provider(ref)
        has_executable_active_plan = (
            self._uuv_only
            and isinstance(
                state.get("executable_mission_plan"), ExecutableMissionPlan
            )
        )
        strategy_set = state.get("strategy_set")
        if strategy_set is None or not strategy_set.proposals:
            if snapshot.active_plan is None and not has_executable_active_plan:
                return {
                    "node_error": (
                        "resource_optimizer requires an approved strategy or active plan"
                    )
                }
            strategy_set = _continuation_strategy_set(snapshot)
        else:
            # A strategic UUV cycle has just regenerated the immutable
            # candidates, regional policies, and their evidence.  Those
            # evidence ids intentionally include prediction/estimate refs,
            # not only raw observation ids, so the stale-proposal filter below
            # must not discard the current semantic decision before the UUV
            # optimizer can materialize it.
            if (
                state.get("route") == EventLevel.STRATEGIC
                and state.get("regional_candidates")
                and state.get("regional_policies")
            ):
                # Regional generation has already produced a fresh policy
                # set for this snapshot. Its evidence includes prediction
                # and region references, so filtering it against only raw
                # group-report observation ids would discard a valid
                # strategic decision before optimization. This applies to
                # both the UUV-only and mixed-domain regional pipelines.
                usable = tuple(strategy_set.proposals)
            else:
                known_evidence = {
                    observation_id
                    for report in snapshot.situation.group_reports
                    for observation_id in report.belief.source_observation_ids
                }
                tracked_targets = {
                    report.target_id for report in snapshot.situation.group_reports
                }
                usable = tuple(
                    proposal
                    for proposal in strategy_set.proposals
                    if all(
                        observation_id in known_evidence
                        for observation_id in proposal.evidence_ids
                    )
                    # A proposal covering a target that vanished from the
                    # situation is kept despite stale evidence: dropping it
                    # would replace the deterministic optimizer-infeasibility
                    # path ("no group report for target ...") with a silent
                    # re-plan over the survivors.
                    or not set(proposal.target_priorities) <= tracked_targets
                )
            if not usable:
                if snapshot.active_plan is None and not has_executable_active_plan:
                    return {
                        "node_error": (
                            "resource_optimizer cannot continue without an active plan"
                        )
                    }
                strategy_set = _continuation_strategy_set(snapshot)
            elif len(usable) < len(strategy_set.proposals):
                strategy_set = StrategySet(
                    trigger_event_ids=strategy_set.trigger_event_ids,
                    proposals=usable,
                )
        merged = cast(CentralState, {**state, "strategy_set": strategy_set})
        started = monotonic()
        _trace_regional_node("resource_optimizer:start")
        try:
            result = cast(CentralState, self._inner(merged))
            _trace_regional_node(
                f"resource_optimizer:done:{monotonic() - started:.3f}s"
            )
            return result
        except ValueError as exc:
            _trace_regional_node(
                f"resource_optimizer:error:{monotonic() - started:.3f}s:{exc}"
            )
            return {"node_error": f"resource_optimizer failed: {exc}"}


class VerifyPlanNode:
    """Independent validation of the selected candidate plan (spec 15.3).

    The selected plan is re-checked against the immutable planning
    snapshot by the commit validator; issues defer to ``handle_error`` so
    the cycle completes with a recorded error instead of committing an
    invalid plan. A hold-current selection (spec 15.2) picks the active
    broadcast plan itself, which was already validated at its commit;
    re-checking it against the newer snapshot would deterministically
    fail on rotated observation evidence ids and the base-revision check,
    so the held plan passes through to ``commit_plan``'s own hold branch.
    """

    def __init__(
        self,
        snapshot_provider: Callable[[str], PlanningSnapshot],
        store: MutableMapping[str, Any],
        config: PlanningConfig = _DEFAULT_PLANNING_CONFIG,
        *,
        uuv_only: bool = False,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        self._store = store
        self._config = config
        self._uuv_only = uuv_only

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
        if self._uuv_only:
            executable = state.get("executable_mission_plan")
            if not isinstance(executable, ExecutableMissionPlan):
                return {
                    "node_error": (
                        "verify_plan requires an executable UUV mission plan"
                    )
                }
            candidate_ids = _state_mission_candidate_ids(state)
            executable_issues = validate_executable_mission_plan(
                snapshot,
                executable,
                candidate_ids=candidate_ids,
            )
            if executable_issues:
                return {
                    "node_error": "verify_plan rejected executable mission: "
                    + "; ".join(executable_issues[:3])
                }
            return {"selected_plan": selected, "selected_plan_ref": ref}
        active = snapshot.active_plan
        if active is not None and selected.plan_id == active.plan_id:
            # Hold-current selection: the broadcast plan was already
            # validated at its commit, and its evidence ids / base
            # revision are necessarily stale relative to this snapshot.
            return {"selected_plan": selected, "selected_plan_ref": ref}
        plan_issues = validate_plan(snapshot, selected, self._config)
        if plan_issues:
            return {
                "node_error": "verify_plan rejected the selected plan: "
                + "; ".join(
                    f"{issue.code} on {issue.field}" for issue in plan_issues[:3]
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
        *,
        repository: PlanRepository | None = None,
        snapshot_provider: Callable[[str], PlanningSnapshot] | None = None,
        uuv_only: bool = False,
        planning_epoch_provider: Callable[[], PlanningEpoch | None] | None = None,
        epoch_commit_port: EpochCommitPort | None = None,
    ) -> None:
        self._inner = inner
        self._store = store
        self._repository = repository
        self._snapshot_provider = snapshot_provider
        self._uuv_only = uuv_only
        self._planning_epoch_provider = planning_epoch_provider
        self._epoch_commit_port = epoch_commit_port

    def __call__(self, state: CentralState) -> CentralState:
        ref = state.get("selected_plan_ref")
        if ref is None:
            return {"commit_status": "rejected", "selected_plan": None}
        candidate = self._store.get(ref)
        if candidate is None:
            return {"commit_status": "rejected", "selected_plan": None}
        if self._uuv_only:
            return self._commit_uuv_only(state, candidate)
        result = self._inner(state, candidate)
        if result["commit_status"] in ("committed", "hold_current"):
            return {
                "commit_status": result["commit_status"],
                "selected_plan": candidate,
            }
        return {"commit_status": result["commit_status"], "selected_plan": None}

    def _commit_uuv_only(
        self, state: CentralState, candidate: object
    ) -> CentralState:
        """Commit the executable plan and persist only its audit projection.

        The typed ``ExecutableMissionPlan`` is the sole execution contract.
        The stored ``TrackingPlan`` is retained so existing audit/replay
        queries can identify the current cycle and so the next immutable
        snapshot can expose a compatibility baseline.  No legacy group
        commands are generated or published on this path.
        """
        executable = state.get("executable_mission_plan")
        snapshot_ref = state.get("snapshot_ref")
        if (
            not isinstance(executable, ExecutableMissionPlan)
            or (self._epoch_commit_port is None and snapshot_ref is None)
            or not isinstance(candidate, TrackingPlan)
        ):
            return {"commit_status": "rejected", "selected_plan": None}

        epoch = state.get("planning_epoch")
        if epoch is None and self._planning_epoch_provider is not None:
            epoch = self._planning_epoch_provider()
        if self._epoch_commit_port is not None:
            if not isinstance(epoch, PlanningEpoch):
                return {"commit_status": "rejected", "selected_plan": None}
            if self._snapshot_provider is not None and snapshot_ref is not None:
                structural_issues = validate_executable_mission_plan(
                    self._snapshot_provider(snapshot_ref),
                    executable,
                    candidate_ids=_state_mission_candidate_ids(state),
                )
                structural_issues = tuple(
                    issue
                    for issue in structural_issues
                    if issue != "mission_revision_mismatch"
                )
                if structural_issues:
                    return {"commit_status": "rejected", "selected_plan": None}
            try:
                result = self._epoch_commit_port.commit(
                    epoch=epoch,
                    audit_projection=candidate,
                    executable_plan=executable,
                )
            except Exception as exc:  # noqa: BLE001 - terminal epoch outcome
                result = EpochCommitResult(
                    epoch_id=epoch.epoch_id,
                    status="failed",
                    failure_category="internal",
                    failure_message=f"{type(exc).__name__}: {exc}"[:2000],
                )
            return {
                "commit_status": result.status,
                "selected_plan": candidate if result.status == "committed" else None,
                # The optimizer's candidate may use a newer local revision
                # than the epoch commit after semantic rebasing.  Only the
                # committed result is authoritative for the next cycle.
                "executable_mission_plan": (
                    result.executable_plan
                    if result.status == "committed"
                    else None
                ),
                "epoch_commit_result": result,
            }

        if self._repository is None or self._snapshot_provider is None:
            return {"commit_status": "rejected", "selected_plan": None}

        assert snapshot_ref is not None
        snapshot = self._snapshot_provider(snapshot_ref)
        issues = validate_executable_mission_plan(
            snapshot,
            executable,
            candidate_ids=_state_mission_candidate_ids(state),
        )
        if issues:
            return {"commit_status": "rejected", "selected_plan": None}
        active = snapshot.active_plan
        if active is not None and candidate.plan_id == active.plan_id:
            return {
                "commit_status": "hold_current",
                "selected_plan": candidate,
            }
        if not self._inner.snapshot_is_current(snapshot.snapshot_revision):
            return {"commit_status": "stale", "selected_plan": None}
        try:
            # This transaction stores the compatibility/audit projection only.
            # UUV execution consumes ``executable_mission_plan`` directly.
            self._repository.commit(candidate)
        except StaleSnapshotError:
            return {"commit_status": "stale", "selected_plan": None}
        return {"commit_status": "committed", "selected_plan": candidate}


def _state_mission_candidate_ids(state: CentralState) -> tuple[str, ...]:
    """Flatten planner-owned candidate ids available in the current cycle."""
    candidates = state.get("regional_candidates") or {}
    return tuple(
        sorted(
            {
                candidate.candidate_id
                for target_candidates in candidates.values()
                for candidate in target_candidates
            }
        )
    )


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
            if self._events.get(event.event_id) is None:
                self._events.append(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    scenario_id=event.scenario_id,
                    sim_time_s=event.sim_time_s,
                    payload=event.payload,
                    target_id=event.entity_id,
                    severity=event.level.value,
                    audiences=event.audiences,
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
        trigger_event_ids = tuple(
            dict.fromkeys(
                event.event_id for event in state.get("coalesced_events") or ()
            )
        )
        if not trigger_event_ids and strategy_set is not None:
            trigger_event_ids = strategy_set.trigger_event_ids
        self._ledger.record(
            DecisionRecord(
                decision_id=(
                    f"{selected.scenario_id}:decision:{snapshot.snapshot_revision}"
                ),
                scenario_id=selected.scenario_id,
                sim_time_s=snapshot.sim_time_s,
                trigger_event_ids=trigger_event_ids,
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
                final_plan_id=(
                    selected.plan_id
                    if state.get("commit_status") in {"committed", "hold_current"}
                    else None
                ),
                expert_inputs=snapshot.applied_directives,
                plan_adjustment_suggestions=tuple(
                    state.get("plan_adjustment_suggestions") or ()
                ),
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
            # FinalizeEpochNode consumes this marker after the error has been
            # appended to the operator-visible cycle history.
            "node_error": message,
            "errors": (*(state.get("errors") or ()), message),
            "output_messages": (*(state.get("output_messages") or ()), f"error: {message}"),
        }


class FinalizeEpochNode:
    """Produce exactly one authoritative terminal result for an active epoch."""

    def __call__(self, state: CentralState) -> CentralState:
        epoch = state.get("planning_epoch")
        if epoch is None:
            return {"node_error": None, "epoch_finalization_route": "record"}

        existing = state.get("epoch_commit_result")
        if existing is not None:
            if not isinstance(existing, EpochCommitResult) or existing.epoch_id != epoch.epoch_id:
                raise PlanningEpochInvariantError(
                    "epoch commit result does not match the active planning epoch"
                )
            if state.get("node_error") is not None:
                raise PlanningEpochInvariantError(
                    "active planning epoch produced a second terminal outcome"
                )
            return {
                "epoch_commit_result": existing,
                "commit_status": existing.status,
                "selected_plan": (
                    state.get("selected_plan") if existing.status == "committed" else None
                ),
                "node_error": None,
                "epoch_finalization_route": "record",
            }

        message = state.get("node_error")
        commit_status = state.get("commit_status")
        if message is None and commit_status == "stale":
            result = EpochCommitResult(
                epoch_id=epoch.epoch_id,
                status="invalidated",
                validation_report_id=f"validation:{epoch.epoch_id}:stale",
                invalidated_reason="stale planning snapshot",
                consumed_event_ids=_state_event_ids(state),
            )
            return {
                "epoch_commit_result": result,
                "commit_status": result.status,
                "selected_plan": None,
                "node_error": None,
                "epoch_finalization_route": "record",
            }
        if message is None and commit_status == "rejected":
            message = "plan rejected without a validation message"
        if message is None and commit_status == "invalidated":
            message = "planning epoch invalidated without a reason"
        if message is None and commit_status not in {"rejected", "invalidated"}:
            raise PlanningEpochInvariantError(
                "active planning epoch completed without a terminal result"
            )
        assert message is not None

        category = _epoch_failure_category(message)
        if commit_status == "invalidated":
            result = EpochCommitResult(
                epoch_id=epoch.epoch_id,
                status="invalidated",
                validation_report_id=f"validation:{epoch.epoch_id}:invalidated",
                invalidated_reason=message[:2000],
                consumed_event_ids=_state_event_ids(state),
            )
        elif category in {"schema", "content", "semantic"}:
            result = EpochCommitResult(
                epoch_id=epoch.epoch_id,
                status="rejected",
                validation_report_id=f"validation:{epoch.epoch_id}:rejected",
                failure_category=category,
                failure_message=message[:2000],
                consumed_event_ids=_state_event_ids(state),
            )
        else:
            result = EpochCommitResult(
                epoch_id=epoch.epoch_id,
                status="failed",
                failure_category=category,
                failure_message=message[:2000],
                consumed_event_ids=_state_event_ids(state),
            )
        return {
            "epoch_commit_result": result,
            "commit_status": result.status,
            "selected_plan": None,
            "node_error": None,
            "epoch_finalization_route": "end",
        }


def _state_event_ids(state: CentralState) -> tuple[str, ...]:
    events = state.get("coalesced_events") or state.get("pending_events") or ()
    return tuple(event.event_id for event in events)


def _epoch_failure_category(message: str) -> EpochFailureCategory:
    lowered = message.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "provider" in lowered or "transport" in lowered or "http" in lowered:
        return "provider"
    if "schema" in lowered:
        return "schema"
    if "content" in lowered:
        return "content"
    if "semantic" in lowered or "regional policy" in lowered:
        return "semantic"
    return "internal"


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
    while clean cycles continue onto the question branch, which runs before
    the regular three-tier routing.
    """
    if state.get("node_error") is not None:
        return "error"
    return _route_events(state)


def _route_question_branch(
    state: CentralState,
) -> Literal[
    "strategic",
    "strategic_prediction",
    "tactical",
    "informational",
    "error",
]:
    """After the question branch: defer branch errors, then the tier route.

    The question branch (spec 10.2) resolves question-run events onto the
    ``latest_question`` channel; a branch error (e.g. an event referencing
    an unknown run id) defers to ``handle_error``, while clean cycles
    continue onto the regular three-tier routing.
    """
    if state.get("node_error") is not None:
        return "error"
    route = _route_events(state)
    if route != "strategic":
        return route
    events = state.get("coalesced_events") or state.get("pending_events") or ()
    if any(event.event_type in _INTENT_ANALYSIS_TRIGGER_TYPES for event in events):
        return "strategic"
    return "strategic_prediction"


_INTENT_ANALYSIS_TRIGGER_TYPES = frozenset(
    {
        "target_added",
        "target_reacquired",
        "intent_change_confirmed",
        "target_intent_changed",
        "imm_confidence_shifted",
    }
)

_UUV_PUBLIC_REGION_REFRESH_EVENT_TYPES = frozenset(
    {
        "target_estimate_updated",
        "target_maneuver_observed",
        "target_speed_regime_changed",
        "target_depth_regime_changed",
        "target_exit_predicted",
        "target_reacquired",
        "covariance_threshold_exceeded",
    }
)


def _requires_uuv_public_region_refresh(state: CentralState) -> bool:
    """Return whether fresh public prediction geometry needs new policies."""
    if not state.get("uuv_only") or not state.get("predictions"):
        return False
    if not state.get("regional_plans"):
        return False
    events = state.get("coalesced_events") or state.get("pending_events") or ()
    return any(
        event.event_type in _UUV_PUBLIC_REGION_REFRESH_EVENT_TYPES
        for event in events
    )


def _route_after_verification(state: CentralState) -> Literal["informational", "error"]:
    """After active-sonar verification: defer branch errors, then record.

    The verification node (spec 17.3) is deterministic and never routes
    the LLM chain; a missing situation reference defers to
    ``handle_error`` while clean cycles continue to record and report.
    """
    if state.get("node_error") is not None:
        return "error"
    return "informational"


def _route_after_epoch_finalization(state: CentralState) -> Literal["record", "end"]:
    return "record" if state.get("epoch_finalization_route") == "record" else "end"


def _route_after_prediction(
    state: CentralState,
) -> Literal["intent_verification", "strategic", "tactical"]:
    """After prediction: strategic runs the full semantic chain, tactical
    continues with optimization only (spec 8.2).

    When a target is temporarily absent from the estimated situation and no
    search prior is available, the predictor returns no fresh evidence. An
    already committed regional plan can still be continued deterministically;
    sending an empty regional graph to the semantic adapter would manufacture
    a planning failure rather than represent the loss of contact honestly.
    """
    if state.get("prediction_intent_verification_target_ids"):
        return "intent_verification"
    unavailable_targets = {
        target_id
        for target_id, accepted in (
            state.get("accepted_predictions") or {}
        ).items()
        if accepted.health.status == "unavailable"
    }
    if unavailable_targets & set(state.get("dynamic_region_chains") or {}):
        # The deterministic regional baseline owns the final fallback. Keep
        # unavailable health intact and roll the prior geometry forward before
        # the semantic-only policy stage runs.
        return "strategic"
    if _requires_uuv_public_region_refresh(state):
        # New public prediction geometry creates new region IDs. Re-enter the
        # regional provider so its authoritative policy remains tied to those
        # IDs; the engine separately preserves physical in-flight batches.
        return "strategic"
    if (
        state.get("regional_plans")
        and state.get("executable_mission_plan") is not None
    ):
        # Goal-mode UUV continuation is an auditable deterministic controller:
        # public prediction refresh -> temporal auction -> physical execution.
        # The real provider is still mandatory for bootstrap planning.
        return "tactical"
    if not state.get("predictions") and state.get("regional_plans"):
        return "tactical"
    return "tactical" if state.get("route") == EventLevel.TACTICAL else "strategic"


def _route_after_prediction_intent(
    state: CentralState,
) -> Literal["strategic", "tactical", "error"]:
    if state.get("node_error") is not None:
        return "error"
    return "strategic" if state.get("prediction_intent_confirmed") else "tactical"


def _route_error(state: CentralState) -> Literal["continue", "error"]:
    """Defer any recorded node error to ``handle_error``."""
    return "error" if state.get("node_error") is not None else "continue"


def _build_live_regional_generation(
    dependencies: CarrierDependencies,
    planning_provider: Callable[[str], PlanningSnapshot],
) -> RegionalGenerationWiringNode:
    """Construct the production region node with deterministic geometry ownership."""
    return RegionalGenerationWiringNode(
        RegionGenerationNode(
            snapshot_provider=planning_provider,
            map_bounds_provider=lambda snapshot: dependencies.optimizer.bounds,
            grid_spec=dependencies.grid_spec,
            llm=dependencies.llm,
            model_id=dependencies.model_id,
            required_quality=dependencies.optimizer.quality_warning,
            execution_strategy_node=dependencies.execution_strategy_node,
            semantic_only=True,
        )
    )


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
        intent_confirmation=dependencies.intent_change_confirmation,
        critical_hold_s=dependencies.critical_hold_s,
        target_lost_gap_s=dependencies.target_lost_gap_s,
        covariance_cap_m2=dependencies.covariance_cap_m2,
    )
    prediction_intent_monitor = dependencies.prediction_intent_monitor or EventMonitor(
        intent_confirmation=dependencies.intent_change_confirmation,
    )

    def planning_provider(ref: str) -> PlanningSnapshot:
        return cast(PlanningSnapshot, store[ref])

    situation_provider = dependencies.situation_provider

    def intent_situation_provider(ref: str) -> SituationSnapshot:
        return cast(SituationSnapshot, store[ref].situation)

    builder = StateGraph(CentralState)
    builder.add_node("ingest", IngestNode(situation_provider, dependencies.plans))
    builder.add_node(
        "event_monitor",
        EventMonitorNode(
            monitor,
            situation_provider,
            dependencies.last_bearing_time,
            dependencies.plans.get_active,
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
        TrajectoryPredictionNode(
            dependencies.predictor,
            intent_situation_provider,
            uuv_only=dependencies.uuv_only,
            diff_config=dependencies.trajectory_diff_config,
        ),
    )
    builder.add_node(
        "prediction_intent_analysis",
        PredictionIntentWiringNode(
            IntentAnalysisNode(
                dependencies.llm,
                model_id=dependencies.model_id,
                belief_history=dependencies.belief_history,
                snapshot_provider=intent_situation_provider,
            ),
            prediction_intent_monitor,
            intent_situation_provider,
            dependencies.intent_change_confirmation,
        ),
    )
    builder.add_node(
        "regional_generation",
        _build_live_regional_generation(dependencies, planning_provider),
    )
    builder.add_node(
        "regional_strategy",
        RegionalStrategyWiringNode(
            RegionalStrategyGenerationNode(
                dependencies.llm,
                model_id=dependencies.model_id,
                snapshot_provider=planning_provider,
                uuv_only=dependencies.uuv_only,
                batch_size=dependencies.regional_batch_size,
                max_concurrency=dependencies.regional_max_concurrency,
                semantic_correction_attempts=dependencies.semantic_correction_attempts,
            )
        ),
    )
    builder.add_node("regional_strategy_adapter", RegionalStrategyToStrategySetNode())
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
            uuv_only=dependencies.uuv_only,
        ),
    )
    builder.add_node(
        "verify_plan",
        VerifyPlanNode(
            planning_provider,
            store,
            dependencies.optimizer,
            uuv_only=dependencies.uuv_only,
        ),
    )
    builder.add_node(
        "commit_plan",
        CommitPlanNode(
            CommitNode(
                repository=dependencies.plans,
                snapshot_provider=planning_provider,
                config=dependencies.optimizer,
                current_snapshot_revision=dependencies.current_snapshot_revision,
            ),
            store,
            repository=dependencies.plans,
            snapshot_provider=planning_provider,
            uuv_only=dependencies.uuv_only,
            planning_epoch_provider=dependencies.planning_epoch_provider,
            epoch_commit_port=dependencies.epoch_commit_port,
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
    builder.add_node("finalize_epoch", FinalizeEpochNode())
    builder.add_node("directive_branch", DirectiveNode(dependencies.ledger))
    builder.add_node("question_branch", QuestionBranchNode(dependencies.ledger))
    builder.add_node(
        "active_verification",
        ActiveVerificationNode(
            dependencies.reservations, dependencies.situation_provider
        ),
    )

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "event_monitor")
    builder.add_edge("event_monitor", "build_snapshot")
    builder.add_edge("build_snapshot", "directive_branch")
    builder.add_conditional_edges(
        "directive_branch",
        _route_directive_branch,
        {
            "strategic": "question_branch",
            "tactical": "question_branch",
            "informational": "question_branch",
            "error": "handle_error",
        },
    )
    builder.add_conditional_edges(
        "question_branch",
        _route_question_branch,
        {
            "strategic": "intent_analysis",
            "strategic_prediction": "trajectory_prediction",
            "tactical": "trajectory_prediction",
            "informational": "active_verification",
            "error": "handle_error",
        },
    )
    builder.add_conditional_edges(
        "active_verification",
        _route_after_verification,
        {
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
        {
            "intent_verification": "prediction_intent_analysis",
            "strategic": "regional_generation",
            "tactical": "resource_optimizer",
        },
    )
    builder.add_conditional_edges(
        "prediction_intent_analysis",
        _route_after_prediction_intent,
        {
            "strategic": "regional_generation",
            "tactical": "resource_optimizer",
            "error": "handle_error",
        },
    )
    builder.add_conditional_edges(
        "regional_generation",
        _route_error,
        {"continue": "regional_strategy", "error": "handle_error"},
    )
    builder.add_conditional_edges(
        "regional_strategy",
        _route_error,
        {"continue": "regional_strategy_adapter", "error": "handle_error"},
    )
    builder.add_conditional_edges(
        "regional_strategy_adapter",
        _route_error,
        {"continue": "verify_strategy", "error": "handle_error"},
    )
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
    builder.add_edge("commit_plan", "finalize_epoch")
    builder.add_conditional_edges(
        "finalize_epoch",
        _route_after_epoch_finalization,
        {"record": "record_decision", "end": END},
    )
    builder.add_edge("record_decision", "progress_report")
    builder.add_edge("progress_report", END)
    builder.add_edge("handle_error", "finalize_epoch")
    return builder.compile(checkpointer=checkpointer)
