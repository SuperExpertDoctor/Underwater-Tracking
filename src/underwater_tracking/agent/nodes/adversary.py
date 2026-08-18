"""Nodes for the real-LLM submarine adversary escape graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, isfinite, sin
from typing import TypedDict

from underwater_tracking.agent.llm import LLMContentError, StructuredLLM
from underwater_tracking.domain.adversary_models import (
    AdversaryEscapeDecision,
    AdversaryEscapeInput,
)

ADVERSARY_PROMPT_VERSION = "adversary-v1"

ADVERSARY_SYSTEM_PROMPT = (
    "You are the adversary controller for one underwater target. "
    "Choose one escape decision using only the supplied target-owned belief, "
    "target-owned observations, summarized hostile platform threats, "
    "communications and acoustic exposure, previous decisions, kinematic "
    "limits, and the operating boundary. The simulator's private state is "
    "unavailable and must not be requested, inferred, or claimed.\n"
    "Account for partial observability, observation age and confidence, "
    "UUV passive/active sonar risk, USV surface relay and active-sonar risk, "
    "distance and bearing of each platform, relay detection risk, acoustic "
    "clutter, emission discipline, decoy inventory, and the continuity of "
    "the current segment. Prefer a maneuver that is feasible within the "
    "provided speed, turn-rate, horizon, inventory, and boundary limits. "
    "The waypoint must stay inside the supplied operating boundary.\n"
    "Use trigger_events as explicit change points: retain the current intent "
    "when evidence is stable, but dynamically adjust when a new detection, "
    "active ping, observability alert, or contact-loss event changes the risk. "
    "Return exactly one JSON object matching the AdversaryEscapeDecision "
    "schema. Rationale must cite only the supplied evidence categories and "
    "must not assert unavailable state."
)


@dataclass(slots=True)
class AdversaryDecisionGate:
    """Rate-limit stable target evidence before invoking the adversary graph.

    A fresh target and a newly observed trigger bypass the cool-down.  Ordinary
    belief revisions must cross a material threshold twice, which prevents
    noisy observations from producing a full LLM decision on every cycle.
    """

    cooldown_s: int = 60
    heading_revision_rad: float = 0.12
    speed_revision_mps: float = 0.75
    _last_decision_s: dict[str, int] = field(default_factory=dict, init=False)
    _last_signature: dict[str, tuple[float, float]] = field(default_factory=dict, init=False)
    _last_trigger_ids: dict[str, frozenset[str]] = field(default_factory=dict, init=False)
    _revision_streaks: dict[str, int] = field(default_factory=dict, init=False)

    def should_request(self, context: AdversaryEscapeInput) -> bool:
        target_id = context.target_id
        if target_id not in self._last_decision_s:
            return True
        trigger_ids = frozenset(
            trigger.trigger_id
            for trigger in context.trigger_events
            if trigger.event_type != "active_ping" and trigger.severity == "strategic"
        )
        if trigger_ids - self._last_trigger_ids.get(target_id, frozenset()):
            return True
        if context.sim_time_s - self._last_decision_s[target_id] >= self.cooldown_s:
            return True
        last_heading, last_speed = self._last_signature[target_id]
        heading_delta = _angular_distance(last_heading, context.belief.estimated_heading)
        speed_delta = abs(last_speed - context.belief.estimated_speed_mps)
        if heading_delta < self.heading_revision_rad and speed_delta < self.speed_revision_mps:
            self._revision_streaks.pop(target_id, None)
            return False
        streak = self._revision_streaks.get(target_id, 0) + 1
        self._revision_streaks[target_id] = streak
        return streak >= 2

    def record_decision(self, context: AdversaryEscapeInput) -> None:
        self._last_decision_s[context.target_id] = context.sim_time_s
        self._last_signature[context.target_id] = (
            context.belief.estimated_heading,
            context.belief.estimated_speed_mps,
        )
        self._last_trigger_ids[context.target_id] = frozenset(
            trigger.trigger_id
            for trigger in context.trigger_events
            if trigger.event_type != "active_ping" and trigger.severity == "strategic"
        )
        self._revision_streaks.pop(context.target_id, None)


class AdversaryState(TypedDict, total=False):
    """LangGraph state; raw evidence is kept only for the current invocation."""

    context: AdversaryEscapeInput
    payload: dict[str, object]
    decision: AdversaryEscapeDecision
    repair_attempted: bool


def build_adversary_payload(context: AdversaryEscapeInput) -> dict[str, object]:
    """Serialize only the target-maintained evidence packet for the LLM."""
    return {
        "prompt_version": ADVERSARY_PROMPT_VERSION,
        "system_prompt": ADVERSARY_SYSTEM_PROMPT,
        "target_id": context.target_id,
        "sim_time_s": context.sim_time_s,
        "belief": context.belief.model_dump(mode="json"),
        "observations": [
            observation.model_dump(mode="json") for observation in context.observations
        ],
        "platform_threats": [threat.model_dump(mode="json") for threat in context.platform_threats],
        "trigger_events": [
            trigger.model_dump(mode="json") for trigger in context.trigger_events
        ],
        "communications_acoustic_exposure": context.communications_acoustic_exposure.model_dump(
            mode="json"
        ),
        "decision_history": [record.model_dump(mode="json") for record in context.decision_history],
        "kinematic_limits": context.kinematic_limits.model_dump(mode="json"),
        "operating_boundary": context.operating_boundary.model_dump(mode="json"),
    }


def _angular_distance(first: float, second: float) -> float:
    return abs(atan2(sin(second - first), cos(second - first)))


def validate_adversary_decision(
    decision: AdversaryEscapeDecision,
    context: AdversaryEscapeInput,
) -> AdversaryEscapeDecision:
    """Apply hard feasibility guards after the structured LLM response."""
    if decision.target_id != context.target_id:
        raise ValueError("adversary decision target_id does not match input")
    values = (*decision.waypoint, decision.speed, decision.heading, decision.confidence)
    if not all(isfinite(value) for value in values):
        raise ValueError("adversary decision contains a non-finite numeric value")

    limits = context.kinematic_limits
    if decision.speed > limits.max_speed_mps:
        raise ValueError("adversary decision speed exceeds max_speed_mps")
    allowed_turn = limits.max_turn_rate_rad_s * limits.decision_horizon_s
    turn = _angular_distance(context.belief.estimated_heading, decision.heading)
    if turn > allowed_turn + 1e-9:
        raise ValueError("adversary decision heading exceeds turn-rate limit")
    boundary = context.operating_boundary
    x, y = decision.waypoint
    if not (boundary.min_x <= x <= boundary.max_x):
        raise ValueError("adversary waypoint is outside the x boundary")
    if not (boundary.min_y <= y <= boundary.max_y):
        raise ValueError("adversary waypoint is outside the y boundary")
    if decision.decoy_count > limits.max_decoy_count:
        raise ValueError("adversary decision exceeds per-decision decoy limit")
    if decision.decoy_count > limits.decoy_inventory:
        raise ValueError("adversary decision exceeds decoy inventory")
    if decision.decoy_action == "none" and decision.decoy_count != 0:
        raise ValueError("decoy_action=none requires decoy_count=0")
    if decision.decoy_action == "deploy" and decision.decoy_count == 0:
        raise ValueError("decoy_action=deploy requires a positive decoy_count")
    if not decision.trigger_event_ids and context.trigger_events:
        return decision.model_copy(
            update={
                "trigger_event_ids": tuple(
                    trigger.trigger_id for trigger in context.trigger_events
                )
            }
        )
    return decision


class BuildAdversaryPayloadNode:
    def __call__(self, state: AdversaryState) -> AdversaryState:
        context = state.get("context")
        if context is None:
            raise ValueError("adversary graph requires context")
        return {"payload": build_adversary_payload(context)}


class AdversaryDecisionNode:
    def __init__(
        self,
        llm: StructuredLLM[AdversaryEscapeDecision],
        *,
        operation: str = "adversary_escape",
        prompt_version: str = ADVERSARY_PROMPT_VERSION,
    ) -> None:
        self._llm = llm
        self._operation = operation
        self._prompt_version = prompt_version

    def __call__(self, state: AdversaryState) -> AdversaryState:
        payload = state.get("payload")
        if payload is None:
            raise ValueError("adversary graph payload was not built")
        try:
            decision = self._llm.invoke_structured(
                self._operation,
                payload,
                AdversaryEscapeDecision,
                prompt_version=self._prompt_version,
            )
        except LLMContentError as exc:
            decision = self._llm.invoke_structured(
                self._operation,
                {
                    **payload,
                    "correction_feedback": (
                        "The previous response was not valid for the supplied schema. "
                        f"Return one complete JSON object only: {exc}"
                    ),
                },
                AdversaryEscapeDecision,
                prompt_version=self._prompt_version,
            )
        if not isinstance(decision, AdversaryEscapeDecision):
            raise TypeError("structured adversary LLM returned the wrong model")
        return {"decision": decision}


class ValidateAdversaryDecisionNode:
    def __init__(
        self,
        llm: StructuredLLM[AdversaryEscapeDecision] | None = None,
        *,
        operation: str = "adversary_escape",
        prompt_version: str = ADVERSARY_PROMPT_VERSION,
    ) -> None:
        self._llm = llm
        self._operation = operation
        self._prompt_version = prompt_version

    def __call__(self, state: AdversaryState) -> AdversaryState:
        context = state.get("context")
        decision = state.get("decision")
        if context is None or decision is None:
            raise ValueError("adversary graph is missing context or decision")
        try:
            return {"decision": validate_adversary_decision(decision, context)}
        except Exception as exc:
            if self._llm is None or state.get("repair_attempted", False):
                raise
            payload = state.get("payload")
            if payload is None:
                raise
            repaired = self._llm.invoke_structured(
                self._operation,
                {
                    **payload,
                    "correction_feedback": (
                        "The previous escape decision violated the supplied hard boundary: "
                        f"{exc}. Return a newly feasible JSON decision only."
                    ),
                },
                AdversaryEscapeDecision,
                prompt_version=self._prompt_version,
            )
            if not isinstance(repaired, AdversaryEscapeDecision):
                raise TypeError("structured adversary repair returned the wrong model")
            return {
                "decision": validate_adversary_decision(repaired, context),
                "repair_attempted": True,
            }


__all__ = [
    "ADVERSARY_PROMPT_VERSION",
    "ADVERSARY_SYSTEM_PROMPT",
    "AdversaryDecisionGate",
    "AdversaryDecisionNode",
    "AdversaryState",
    "BuildAdversaryPayloadNode",
    "ValidateAdversaryDecisionNode",
    "build_adversary_payload",
    "validate_adversary_decision",
]
