"""Nodes for the real-LLM submarine adversary escape graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, isfinite, sin
from typing import Any, TypedDict, cast

from underwater_tracking.agent.llm import LLMContentError, StructuredLLM
from underwater_tracking.domain.adversary_models import (
    AdversaryEscapeDecision,
    AdversaryEscapeInput,
    AdversaryIntentDecision,
)

ADVERSARY_PROMPT_VERSION = "adversary-v5"
_DECISION_TRIGGER_TYPES = {
    "target_detection",
    "target_detection_acquired",
    "target_detection_lost",
    "target_contact_range_changed",
    "target_contact_threat_changed",
    "target_active_emitter_acquired",
    "target_sensor_mode_changed",
    "target_mission_initialized",
    "target_route_invalidated",
    "target_mission_stage_changed",
}

ADVERSARY_SYSTEM_PROMPT = (
    "You are the adversary controller for one underwater target. "
    "Choose one high-level mission intent using only the supplied target-owned "
    "mission state, own navigation estimate, local noisy contacts, exposure, "
    "previous decisions, kinematic limits, and operating boundary. The "
    "simulator's private state is unavailable and must not be requested, "
    "inferred, or claimed.\n"
    "Sensor doctrine is explicit: passive tracking is the cooperative UUV role "
    "and has no deliberate emission; active scan is the regional-coverage UUV "
    "role and is a deliberate active-sonar emission. Never treat a passive "
    "tracker as an active emitter or claim an active ping without supplied "
    "active_sonar evidence. When active scan exposure appears, prefer a measured "
    "avoid_contact, break_contact, or escape_to_region response that changes the "
    "target cell or tracking geometry; when only passive tracking is observed, "
    "preserve mission progress while reducing correlation and avoiding predictable "
    "routes.\n"
    "The only valid intents are continue_mission, avoid_contact, break_contact, "
    "escape_to_region, and hold_position. Select an escape_region_id only for "
    "escape_to_region and use one of the configured IDs. Select target_cell_xy as "
    "the center of one feasible 1 km global-grid cell, using local UUV tracking "
    "threats, exposure, mission progress, and kinematic limits. Do not emit a "
    "waypoint other than target_cell_xy, speed, heading, depth change, decoy action, "
    "or communications action; deterministic target guidance owns those physical choices.\n"
    "Use uuv_trajectory_cache as the target's observed per-UUV history and "
    "uuv_tracking_patterns as its high-level semantic assessment. Account for "
    "tracking approach, stable trailing, accompanying tracking, intercept tracking, "
    "intermittent tracking, reacquisition, multi-UUV coordination, relay tracking, "
    "flank envelope tracking, and tracking disengagement when selecting the next cell. "
    "These estimates are observation-derived and must not be treated as simulator truth.\n"
    "Use trigger_events as explicit change points: retain the current intent "
    "when evidence is stable, but dynamically adjust when a new detection, "
    "active ping, observability alert, or contact-loss event changes the risk. "
    "Preserve mission progress when risk is low: continue_mission follows the "
    "private mission route; avoid_contact creates a measured separation response; "
    "break_contact prioritizes rapid loss of a credible local contact; "
    "escape_to_region commits to one configured escape region; and hold_position "
    "is only for an immediate safety or uncertainty hold. Select depth_intent "
    "only when the exposure and maneuver objective justify it. Compare previous "
    "decision outcomes and current local-contact evolution before repeating a "
    "maneuver; do not claim that an unknown outcome succeeded. "
    "Return exactly one JSON object matching the AdversaryIntentDecision "
    "schema. Rationale must cite only the supplied evidence categories and "
    "must not assert unavailable state."
)


@dataclass(slots=True)
class AdversaryDecisionGate:
    """Rate-limit stable target evidence before invoking the adversary graph.

    A fresh target and a newly observed local detection transition bypass the
    cool-down. Ordinary belief revisions must cross a material threshold twice,
    which prevents noisy observations from producing a full LLM decision on
    every cycle.
    """

    cooldown_s: int = 60
    heading_revision_rad: float = 0.12
    speed_revision_mps: float = 0.75
    _last_decision_s: dict[str, int] = field(default_factory=dict, init=False)
    _last_signature: dict[str, tuple[float, float]] = field(default_factory=dict, init=False)
    _last_local_signature: dict[str, tuple[tuple[str, str, str, str], ...]] = field(
        default_factory=dict,
        init=False,
    )
    _last_trigger_ids: dict[str, frozenset[str]] = field(default_factory=dict, init=False)
    _revision_streaks: dict[str, int] = field(default_factory=dict, init=False)

    def should_request(self, context: AdversaryEscapeInput) -> bool:
        target_id = context.target_id
        if not _has_local_evidence(context) and not _has_mission_trigger(context):
            return False
        if target_id not in self._last_decision_s:
            return True
        trigger_ids = frozenset(
            trigger.trigger_id
            for trigger in context.trigger_events
            if trigger.event_type
            in _DECISION_TRIGGER_TYPES
        )
        if trigger_ids - self._last_trigger_ids.get(target_id, frozenset()):
            return True
        if context.sim_time_s - self._last_decision_s[target_id] < self.cooldown_s:
            return False
        if _local_signature(context) != self._last_local_signature.get(target_id, ()):
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
        self._last_local_signature[context.target_id] = _local_signature(context)
        self._last_trigger_ids[context.target_id] = frozenset(
            trigger.trigger_id
            for trigger in context.trigger_events
            if trigger.event_type
            in _DECISION_TRIGGER_TYPES
        )
        self._revision_streaks.pop(context.target_id, None)


def _has_local_evidence(context: AdversaryEscapeInput) -> bool:
    """Return whether the target has evidence admitted by local sensing."""
    return bool(
        context.observations
        or context.platform_threats
        or any(
            trigger.event_type
            in {"active_ping", *_DECISION_TRIGGER_TYPES}
            for trigger in context.trigger_events
        )
        or context.communications_acoustic_exposure.active_emitter_exposure > 0.0
    )


def _has_mission_trigger(context: AdversaryEscapeInput) -> bool:
    return any(
        trigger.event_type in _DECISION_TRIGGER_TYPES
        for trigger in context.trigger_events
    )


def _local_signature(context: AdversaryEscapeInput) -> tuple[tuple[str, str, str, str], ...]:
    """Bucket local range/risk changes so noisy estimates do not thrash the gate."""
    return tuple(
        sorted(
            [
                (
                    "threat",
                    threat.platform_id,
                    str(int(threat.estimated_range_m // 250.0)),
                    f"{threat.threat_level}:{threat.sensor_mode}:{threat.uuv_status or 'none'}",
                )
                for threat in context.platform_threats
            ]
            + [
                (
                    "pattern",
                    pattern.pattern_type,
                    ",".join(pattern.uuv_ids),
                    str(int(pattern.confidence * 10.0)),
                )
                for pattern in context.uuv_tracking_patterns
            ]
        )
    )


class AdversaryState(TypedDict, total=False):
    """LangGraph state; raw evidence is kept only for the current invocation."""

    context: AdversaryEscapeInput
    payload: dict[str, object]
    decision: AdversaryIntentDecision | AdversaryEscapeDecision
    repair_attempted: bool


def build_adversary_payload(context: AdversaryEscapeInput) -> dict[str, object]:
    """Serialize only the target-maintained evidence packet for the LLM."""
    return {
        "prompt_version": ADVERSARY_PROMPT_VERSION,
        "system_prompt": ADVERSARY_SYSTEM_PROMPT,
        "target_id": context.target_id,
        "sim_time_s": context.sim_time_s,
        "decision_policy": {
            "objective": "reduce_detectability_while_preserving_mission_feasibility",
            "short_term_navigation": {
                "target_cell_required": True,
                "coordinate_system": "global_xy_m",
                "cell_size_m": 1000.0,
                "instruction": "Select target_cell_xy as the center of one feasible 1 km cell using local threat, UUV tracking, mission progress, and kinematic limits.",
            },
            "intent_semantics": {
                "continue_mission": "continue the private mission route when local risk is low or unchanged",
                "avoid_contact": "make a measured separation maneuver from credible local contacts",
                "break_contact": "prioritize rapid separation after a high-confidence or active-emitter threat",
                "escape_to_region": "move toward one configured escape region when route continuation is no longer prudent",
                "hold_position": "temporarily hold only for immediate safety or unresolved local uncertainty",
            },
            "selection_order": (
                "local_contact_and_active_emitter_risk",
                "sensor_role_and_emission_exposure",
                "mission_progress_and_escape_options",
                "kinematic_and_boundary_feasibility",
                "communications_acoustic_exposure",
                "previous_decision_outcomes",
            ),
            "sensor_doctrine": {
                "passive_track": {
                    "mission_role": "cooperative tracking",
                    "emission": "none",
                    "counter_tracking": (
                        "reduce motion correlation and avoid predictable routes"
                    ),
                },
                "active_scan": {
                    "mission_role": "regional coverage",
                    "emission": "deliberate",
                    "counter_tracking": (
                        "break or avoid contact and change tracking geometry"
                    ),
                },
            },
        },
        "own_position_xy": context.belief.estimated_position_xy,
        "mission_state": context.mission_state.model_dump(mode="json"),
        "local_contacts": [
            contact.model_dump(mode="json") for contact in context.local_contacts
        ],
        "belief": context.belief.model_dump(mode="json"),
        "observations": [
            observation.model_dump(mode="json") for observation in context.observations
        ],
        "platform_threats": [threat.model_dump(mode="json") for threat in context.platform_threats],
        "uuv_trajectory_cache": {
            uuv_id: [point.model_dump(mode="json") for point in points]
            for uuv_id, points in sorted(context.uuv_trajectory_cache.items())
        },
        "uuv_tracking_patterns": [
            pattern.model_dump(mode="json") for pattern in context.uuv_tracking_patterns
        ],
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
    decision: AdversaryIntentDecision | AdversaryEscapeDecision,
    context: AdversaryEscapeInput,
) -> AdversaryIntentDecision | AdversaryEscapeDecision:
    """Apply hard feasibility guards after the structured LLM response."""
    if decision.target_id != context.target_id:
        raise ValueError("adversary decision target_id does not match input")
    if isinstance(decision, AdversaryIntentDecision):
        if (
            decision.escape_region_id is not None
            and decision.escape_region_id not in context.mission_state.escape_regions
        ):
            raise ValueError("escape_region_id is not a configured escape region")
        if decision.target_cell_xy is not None:
            x, y = decision.target_cell_xy
            boundary = context.operating_boundary
            if not (boundary.min_x <= x <= boundary.max_x and boundary.min_y <= y <= boundary.max_y):
                raise ValueError("target_cell_xy is outside the operating boundary")
            if any(
                abs(((coordinate - 500.0) / 1000.0) - round((coordinate - 500.0) / 1000.0)) > 1e-6
                for coordinate in (x, y)
            ):
                raise ValueError("target_cell_xy must be the center of a 1 km global-grid cell")
        trigger_event_ids = _merge_trigger_event_ids(
            decision.trigger_event_ids,
            context.trigger_events,
        )
        return (
            decision.model_copy(update={"trigger_event_ids": trigger_event_ids})
            if trigger_event_ids != decision.trigger_event_ids
            else decision
        )
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
    trigger_event_ids = _merge_trigger_event_ids(
        decision.trigger_event_ids,
        context.trigger_events,
    )
    return (
        decision.model_copy(update={"trigger_event_ids": trigger_event_ids})
        if trigger_event_ids != decision.trigger_event_ids
        else decision
    )


def _merge_trigger_event_ids(
    decision_ids: tuple[str, ...],
    trigger_events: tuple[Any, ...],
) -> tuple[str, ...]:
    """Keep only evidence IDs present in the current target-local context."""
    allowed = tuple(event.trigger_id for event in trigger_events)
    allowed_set = set(allowed)
    return tuple(
        dict.fromkeys(
            (*(
                event_id
                for event_id in decision_ids
                if event_id in allowed_set
            ), *allowed)
        )
    )[-16:]


class BuildAdversaryPayloadNode:
    def __call__(self, state: AdversaryState) -> AdversaryState:
        context = state.get("context")
        if context is None:
            raise ValueError("adversary graph requires context")
        return {"payload": build_adversary_payload(context)}


class AdversaryDecisionNode:
    def __init__(
        self,
        llm: StructuredLLM[AdversaryIntentDecision] | StructuredLLM[AdversaryEscapeDecision],
        *,
        operation: str = "adversary_mission_decision",
        prompt_version: str = ADVERSARY_PROMPT_VERSION,
    ) -> None:
        self._llm = llm
        self._operation = operation
        self._prompt_version = prompt_version

    def __call__(self, state: AdversaryState) -> AdversaryState:
        payload = state.get("payload")
        if payload is None:
            raise ValueError("adversary graph payload was not built")
        response_model: type[Any] = (
            AdversaryIntentDecision
            if self._operation == "adversary_mission_decision"
            else AdversaryEscapeDecision
        )
        try:
            decision = self._llm.invoke_structured(
                self._operation,
                payload,
                cast(Any, response_model),
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
                cast(Any, response_model),
                prompt_version=self._prompt_version,
            )
        if not isinstance(decision, response_model):
            raise TypeError("structured adversary LLM returned the wrong model")
        return {"decision": decision}


class ValidateAdversaryDecisionNode:
    def __init__(
        self,
        llm: StructuredLLM[AdversaryIntentDecision] | StructuredLLM[AdversaryEscapeDecision] | None = None,
        *,
        operation: str = "adversary_mission_decision",
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
            response_model: type[Any] = (
                AdversaryIntentDecision
                if self._operation == "adversary_mission_decision"
                else AdversaryEscapeDecision
            )
            repaired = self._llm.invoke_structured(
                self._operation,
                {
                    **payload,
                    "correction_feedback": (
                        "The previous escape decision violated the supplied hard boundary: "
                        f"{exc}. Return a newly feasible JSON decision only."
                    ),
                },
                cast(Any, response_model),
                prompt_version=self._prompt_version,
            )
            if not isinstance(repaired, response_model):
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
