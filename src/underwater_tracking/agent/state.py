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

from typing import TypedDict

from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    IntentHypothesis,
    PredictedTrackRef,
    StrategySet,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent


class CarrierState(TypedDict, total=False):
    """Persistent scenario graph state; references, never raw histories."""

    scenario_id: str
    snapshot_revision: int
    # Storage reference (e.g. snapshot id or hash) to the immutable
    # SituationSnapshot kept outside the checkpoint.
    snapshot_ref: str | None
    pending_events: tuple[RuntimeEvent, ...]
    coalesced_events: tuple[RuntimeEvent, ...]
    # Current three-tier routing decision (spec 8.2).
    route: EventLevel | None
    intent_hypotheses: dict[str, IntentHypothesis]
    predictions: dict[str, PredictedTrackRef]
    strategy_set: StrategySet | None
    validation_attempts: int
    candidate_plan_refs: tuple[str, ...]
    selected_plan_ref: str | None
    latest_directive: ExpertDirective | None
    latest_question: str | None
    history_summaries: tuple[str, ...]
    errors: tuple[str, ...]
    output_messages: tuple[str, ...]
