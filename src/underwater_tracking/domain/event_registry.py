"""Authoritative metadata for runtime events crossing subsystem boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from underwater_tracking.domain.models import EventAudience, EventLevel, StrictModel


PlanImpactPolicy = Literal["always", "evidence_required", "never"]
MemoryRetentionPolicy = Literal["always", "evidence_required", "never"]


class EventDefinition(StrictModel):
    """Routing, audience, and coalescing metadata for one event type."""

    event_type: str = Field(min_length=1)
    default_level: EventLevel
    audiences: frozenset[EventAudience] = Field(min_length=1)
    plan_impact_policy: PlanImpactPolicy
    memory_policy: MemoryRetentionPolicy
    coalescing_family: str | None = None


PUBLIC_AUDIENCES = frozenset(
    {
        EventAudience.BLUE_PLANNING,
        EventAudience.OPERATOR_AUDIT,
        EventAudience.MEMORY_SOURCE,
    }
)
PRIVATE_AUDIENCES = frozenset(
    {
        EventAudience.ADVERSARY_PRIVATE,
        EventAudience.OPERATOR_AUDIT,
        EventAudience.MEMORY_SOURCE,
    }
)


def _definition(
    event_type: str,
    level: EventLevel,
    policy: PlanImpactPolicy,
    *,
    audiences: frozenset[EventAudience] = PUBLIC_AUDIENCES,
    family: str | None = None,
    memory_policy: MemoryRetentionPolicy = "never",
) -> EventDefinition:
    return EventDefinition(
        event_type=event_type,
        default_level=level,
        audiences=audiences,
        plan_impact_policy=policy,
        memory_policy=memory_policy,
        coalescing_family=family,
    )


EVENT_REGISTRY: dict[str, EventDefinition] = {}


def _register(
    event_types: tuple[str, ...],
    level: EventLevel,
    policy: PlanImpactPolicy,
    *,
    family: str | None = None,
    audiences: frozenset[EventAudience] = PUBLIC_AUDIENCES,
    memory_policy: MemoryRetentionPolicy = "never",
) -> None:
    for event_type in event_types:
        EVENT_REGISTRY[event_type] = _definition(
            event_type,
            level,
            policy,
            audiences=audiences,
            family=family,
            memory_policy=memory_policy,
        )


_register(
    (
        "initialization",
        "target_added",
        "target_removed",
        "target_lost",
        "target_reacquired",
        "major_failure",
        "repair_infeasible",
        "directive_applied",
        "operational_scheme_updated",
        "uuv_range_exhausted",
        "uuv_energy_depleted",
        "uuv_failed",
        "uuv_capability_lost",
        "carrier_task_window_missed",
        "strategic_review",
    ),
    EventLevel.STRATEGIC,
    "always",
    memory_policy="always",
)
_register(
    (
        "group_quality_critical",
        "handoff_blocked",
        "carrier_rendezvous_infeasible",
        "carrier_recovery_blocked",
    ),
    EventLevel.STRATEGIC,
    "evidence_required",
    family="quality",
    memory_policy="evidence_required",
)
_register(
    (
        "region_coverage_degraded",
        "regional_feedback_received",
        "communication_link_lost",
        "covariance_threshold_exceeded",
        "intent_change_confirmed",
        "imm_confidence_shifted",
        "target_exit_predicted",
        "endurance_threshold_crossed",
        "intelligence_report_received",
    ),
    EventLevel.INFORMATIONAL,
    "evidence_required",
    family="quality",
    memory_policy="evidence_required",
)
_register(
    (
        "group_quality_warning",
        "geometry_degradation",
        "battery_rotation",
        "target_maneuver",
        "target_maneuver_observed",
        "target_speed_regime_changed",
        "target_depth_regime_changed",
        "target_estimate_updated",
        "target_detection_acquired",
        "target_detection_lost",
        "carrier_recovery_health_check_pending",
        "member_failed",
        "carrier_plan_degraded",
    ),
    EventLevel.TACTICAL,
    "evidence_required",
    memory_policy="evidence_required",
)
_register(
    (
        "target_prior_expired",
        "carrier_recovery_health_check_pending",
        "llm_degraded",
        "progress_report",
        "question",
        "state_changed",
        "repair_applied",
        "active_ping",
        "contact_classified",
        "group_report_published",
        "manual_sensor_mode",
        "bearing",
        "bearing_observation",
        "carrier_returned_to_fleet",
        "periodic_situation_summary",
        "periodic_summary_backlog_overflow",
        "prediction_revision",
        "quality_warning",
        "target_contact_threat_changed",
        "target_decoy_deployed",
        "uuv_health",
        "contact",
        "target_public_belief",
        "test_event",
    ),
    EventLevel.INFORMATIONAL,
    "never",
)

_register(
    (
        "target_entered_region",
        "handoff_completed",
        "carrier_dispatch_completed",
        "carrier_recovery_completed",
        "uuv_recovery_requested",
        "uuv_deployed",
        "uuv_recovered",
        "carrier_returned_to_fleet",
        "plan_commit",
        "regional_replan",
        "replan",
        "resource_low",
        "uuv_rotation",
    ),
    EventLevel.INFORMATIONAL,
    "never",
    memory_policy="always",
)
_register(
    (
        "target_mission_decision",
        "target_mission_initialized",
        "target_mission_degraded",
        "target_navigation_guard_failed",
        "target_contact_threat_changed",
    ),
    EventLevel.INFORMATIONAL,
    "never",
    audiences=PRIVATE_AUDIENCES,
)

# Coalescing metadata belongs to the registry rather than to a second routing
# table in the carrier graph.
for _event_type in ("group_quality_critical",):
    EVENT_REGISTRY[_event_type] = _definition(
        _event_type,
        EventLevel.STRATEGIC,
        "evidence_required",
        family="quality",
        memory_policy="evidence_required",
    )
for _event_type in ("intent_change_confirmed",):
    EVENT_REGISTRY[_event_type] = _definition(
        _event_type,
        EventLevel.INFORMATIONAL,
        "evidence_required",
        family="intent",
        memory_policy="evidence_required",
    )
EVENT_REGISTRY["target_intent_changed"] = _definition(
    "target_intent_changed",
    EventLevel.STRATEGIC,
    "always",
    family="intent",
    memory_policy="always",
)
EVENT_REGISTRY["target_intent_change_suspected"] = _definition(
    "target_intent_change_suspected",
    EventLevel.TACTICAL,
    "evidence_required",
    family="prediction_diff",
    memory_policy="evidence_required",
)
for _event_type in ("imm_motion_mode_changed", "imm_confidence_shifted"):
    EVENT_REGISTRY[_event_type] = _definition(
        _event_type,
        EventLevel.INFORMATIONAL,
        "evidence_required",
        family="imm_motion",
        memory_policy="evidence_required",
    )
for _event_type in (
    "region_coverage_degraded",
    "regional_feedback_received",
):
    EVENT_REGISTRY[_event_type] = _definition(
        _event_type,
        EventLevel.INFORMATIONAL,
        "evidence_required",
        family="quality",
        memory_policy="evidence_required",
    )
EVENT_REGISTRY["periodic_situation_summary"] = _definition(
    "periodic_situation_summary",
    EventLevel.INFORMATIONAL,
    "never",
    memory_policy="evidence_required",
)


def event_definition(event_type: str) -> EventDefinition:
    """Resolve exact and supported family event types, rejecting unknowns."""
    definition = EVENT_REGISTRY.get(event_type)
    if definition is not None:
        return definition
    if event_type.startswith("quality_guard:"):
        return _definition(
            event_type,
            EventLevel.TACTICAL,
            "evidence_required",
            family="quality",
            memory_policy="evidence_required",
        )
    if event_type.startswith("observability_"):
        return _definition(event_type, EventLevel.INFORMATIONAL, "never")
    raise ValueError(f"unknown event type: {event_type!r}")


def event_audiences(event_type: str) -> frozenset[EventAudience]:
    """Return the durable audience set for an event producer."""
    return event_definition(event_type).audiences


def is_memory_source_event(event_type: str, payload: Mapping[str, object]) -> bool:
    """Return whether an event should create semantic-memory work.

    Audit visibility and durable-memory retention are intentionally separate:
    high-rate events can remain replayable without invoking the memory LLM.
    """

    try:
        policy = event_definition(event_type).memory_policy
    except ValueError:
        return False
    if policy == "always":
        return True
    if policy == "never":
        return False
    if event_type == "periodic_situation_summary":
        marker = payload.get("memory_eligible")
        if isinstance(marker, bool):
            return marker
        changes = payload.get("changes_since_previous")
        return bool(changes) if changes is not None else True
    if event_type == "target_estimate_updated":
        return payload.get("plan_impact") is True
    if event_type in {"imm_motion_mode_changed", "imm_confidence_shifted"}:
        return payload.get("confirmed") is True
    return bool(
        payload.get("observation_ids")
        or payload.get("evidence_ids")
        or payload.get("summary")
        or payload
    )


def validate_event_payload(event_type: str, payload: dict[str, object]) -> None:
    """Validate public evidence required by observable target events."""
    if event_type == "target_intent_change_suspected":
        required = {
            "diff_id",
            "previous_prediction_id",
            "current_prediction_id",
            "observation_ids",
            "absolute_rms_m",
            "normalized_rms",
            "absolute_floor_m",
            "normalized_threshold",
            "consecutive_count",
            "source",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"{event_type} requires payload keys: {', '.join(sorted(missing))}")
        observation_ids = payload.get("observation_ids")
        if not isinstance(observation_ids, (list, tuple, frozenset)) or not observation_ids:
            raise ValueError(f"{event_type} requires non-empty observation_ids")
        if any(not isinstance(value, str) or not value for value in observation_ids):
            raise ValueError(f"{event_type} requires non-empty observation_ids")
    if event_type not in {
        "target_maneuver_observed",
        "target_speed_regime_changed",
        "target_depth_regime_changed",
        "target_estimate_updated",
    }:
        return
    observation_ids = payload.get("observation_ids")
    if not isinstance(observation_ids, (list, tuple, frozenset)):
        raise ValueError(f"{event_type} requires non-empty observation_ids")
    if not observation_ids or any(
        not isinstance(value, str) or not value for value in observation_ids
    ):
        raise ValueError(f"{event_type} requires non-empty observation_ids")


def is_blue_public(event_type: str, audiences: frozenset[EventAudience]) -> bool:
    """Return whether a concrete event is admissible to blue planning."""
    return (
        EventAudience.BLUE_PLANNING in audiences
        and EventAudience.BLUE_PLANNING in event_audiences(event_type)
    )


__all__ = [
    "EVENT_REGISTRY",
    "PRIVATE_AUDIENCES",
    "PUBLIC_AUDIENCES",
    "EventAudience",
    "EventDefinition",
    "MemoryRetentionPolicy",
    "event_audiences",
    "event_definition",
    "is_blue_public",
    "is_memory_source_event",
    "validate_event_payload",
]
