# src/underwater_tracking/agent/state.py
"""Carrier-state contract for the persistent scenario graph (spec 8.1, 8.2).

``CarrierState`` is the LangGraph state of the carrier central graph. It
holds identifiers and references — scenario id, snapshot revision and
storage reference, pending/coalesced events, route, intent hypotheses,
predictions, the strategy set, validation attempts, candidate plan
references, the selected plan reference, the latest directive and
question, history summary references, errors and output messages — rather
than raw observation histories (spec 8.4: high-frequency raw observations
live in the EventStore, not the graph checkpoint).

The state is ``total=False`` so each graph node may return only the fields
it updates; append semantics for events, summaries and messages are added
by LangGraph reducers in the graph task (Task 8+).
"""

from __future__ import annotations

from typing import Literal, TypedDict

from underwater_tracking.agent.llm import LLMCallMetadata
from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    IntentHypothesis,
    PlanAdjustmentSuggestion,
    PredictedTrackRef,
    RegionalPlanMetrics,
    StrategySet,
    VerificationCommand,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.domain.planning_epoch_models import EpochCommitResult, PlanningEpoch
from underwater_tracking.domain.regional_models import (
    RegionTask,
    RegionalMissionCandidate,
    RegionalStrategySet,
    TargetRegionPlan,
    UUVRegionalStrategySet,
)


RegionalReplanReason = Literal[
    "regional_feedback",
    "endurance",
    "communication_link",
    "covariance",
    "target_reacquired",
]


class CarrierState(TypedDict, total=False):
    """Persistent scenario graph state; references, never raw histories."""

    scenario_id: str
    uuv_only: bool
    snapshot_revision: int
    snapshot_sim_time_s: int
    # Storage reference (e.g. snapshot id or hash) to the immutable
    # SituationSnapshot kept outside the checkpoint.
    snapshot_ref: str | None
    pending_events: tuple[RuntimeEvent, ...]
    coalesced_events: tuple[RuntimeEvent, ...]
    # Current three-tier routing decision (spec 8.2).
    route: EventLevel | None
    # Why the current cycle escalated to the regional strategic branch.
    strategic_replan_reasons: tuple[RegionalReplanReason, ...]
    # Target ids observed in prior cycles, used by deterministic loss detection.
    known_target_ids: tuple[str, ...]
    # Targets awaiting a later observation before they are marked reacquired.
    lost_target_ids: tuple[str, ...]
    intent_hypotheses: dict[str, IntentHypothesis]
    predictions: dict[str, PredictedTrackRef]
    regional_plans: dict[str, TargetRegionPlan]
    regional_candidates: dict[str, tuple[RegionalMissionCandidate, ...]]
    regional_policies: dict[str, RegionalStrategySet | UUVRegionalStrategySet]
    region_tasks: dict[str, RegionTask]
    regional_metrics: RegionalPlanMetrics | None
    executable_mission_plan: ExecutableMissionPlan | None
    planning_epoch: PlanningEpoch | None
    epoch_commit_result: EpochCommitResult | None
    strategy_set: StrategySet | None
    # Provenance of the latest semantic LLM calls (spec 16): per-call key
    # (e.g. "intent:T1", "strategy:quality_first") -> metadata with model and
    # prompt versions plus request/response hashes. Payloads are never stored.
    llm_provenance: dict[str, LLMCallMetadata]
    # Ontology query ids used as external expert evidence for the latest plan
    # adjustment. Full responses live in the SQLite audit table.
    knowledge_query_ids: tuple[str, ...]
    plan_adjustment_suggestions: tuple[PlanAdjustmentSuggestion, ...]
    validation_attempts: int
    candidate_plan_refs: tuple[str, ...]
    selected_plan_ref: str | None
    latest_directive: ExpertDirective | None
    latest_question: str | None
    history_summaries: tuple[str, ...]
    errors: tuple[str, ...]
    output_messages: tuple[str, ...]
    # Active-sonar verification protocol (spec 17.3): per-contact protocol
    # state, the UUV id pinging each contact, and the engine commands.
    verification_states: dict[str, str]
    verification_pingers: dict[str, str]
    verification_commands: tuple[VerificationCommand, ...]
