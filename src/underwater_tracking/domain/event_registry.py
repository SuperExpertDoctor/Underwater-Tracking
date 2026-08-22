"""Authoritative metadata for runtime events crossing subsystem boundaries."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from underwater_tracking.domain.models import EventAudience, EventLevel, StrictModel


PlanImpactPolicy = Literal["always", "evidence_required", "never"]


class EventDefinition(StrictModel):
    """Routing, audience, and coalescing metadata for one event type."""

    event_type: str = Field(min_length=1)
    default_level: EventLevel
    audiences: frozenset[EventAudience] = Field(min_length=1)
    plan_impact_policy: PlanImpactPolicy
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
) -> EventDefinition:
    return EventDefinition(
        event_type=event_type,
        default_level=level,
        audiences=audiences,
        plan_impact_policy=policy,
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
) -> None:
    for event_type in event_types:
        EVENT_REGISTRY[event_type] = _definition(
            event_type,
            level,
            policy,
            audiences=audiences,
            family=family,
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
)
_register(
    (
        "group_quality_critical",
        "region_coverage_degraded",
        "regional_feedback_received",
        "communication_link_lost",
        "covariance_threshold_exceeded",
        "intent_change_confirmed",
        "target_intent_changed",
        "imm_confidence_shifted",
        "target_exit_predicted",
        "endurance_threshold_crossed",
        "intelligence_report_received",
        "handoff_blocked",
        "carrier_rendezvous_infeasible",
        "carrier_recovery_blocked",
    ),
    EventLevel.STRATEGIC,
    "evidence_required",
    family="quality",
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
        "target_detection_acquired",
        "target_detection_lost",
        "carrier_recovery_health_check_pending",
        "member_failed",
    ),
    EventLevel.TACTICAL,
    "evidence_required",
)
_register(
    (
        "intent_change_confirmed",
        "intelligence_report_received",
        "target_intent_changed",
        "imm_confidence_shifted",
        "target_entered_region",
        "target_exit_predicted",
        "handoff_completed",
        "region_coverage_degraded",
        "regional_feedback_received",
        "endurance_threshold_crossed",
        "communication_link_lost",
        "covariance_threshold_exceeded",
        "carrier_dispatch_completed",
        "carrier_recovery_completed",
        "target_prior_expired",
        "carrier_recovery_health_check_pending",
        "llm_degraded",
        "progress_report",
        "question",
        "state_changed",
        "repair_applied",
        "active_ping",
        "contact_classified",
        "uuv_recovery_requested",
        "uuv_deployed",
        "uuv_recovered",
        "group_report_published",
        "manual_sensor_mode",
        "bearing",
        "bearing_observation",
        "carrier_returned_to_fleet",
        "periodic_situation_summary",
        "periodic_summary_backlog_overflow",
        "plan_commit",
        "prediction_revision",
        "quality_warning",
        "regional_replan",
        "replan",
        "resource_low",
        "target_contact_threat_changed",
        "target_decoy_deployed",
        "uuv_rotation",
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
    )
for _event_type in (
    "target_intent_changed",
    "imm_confidence_shifted",
    "intent_change_confirmed",
):
    EVENT_REGISTRY[_event_type] = _definition(
        _event_type,
        EventLevel.INFORMATIONAL,
        "evidence_required",
        family="intent",
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
        )
    if event_type.startswith("observability_"):
        return _definition(event_type, EventLevel.INFORMATIONAL, "never")
    raise ValueError(f"unknown event type: {event_type!r}")


def event_audiences(event_type: str) -> frozenset[EventAudience]:
    """Return the durable audience set for an event producer."""
    return event_definition(event_type).audiences


def validate_event_payload(event_type: str, payload: dict[str, object]) -> None:
    """Validate public evidence required by observable target events."""
    if event_type not in {
        "target_maneuver_observed",
        "target_speed_regime_changed",
        "target_depth_regime_changed",
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
    "EventAudience",
    "EventDefinition",
    "PRIVATE_AUDIENCES",
    "PUBLIC_AUDIENCES",
    "event_audiences",
    "event_definition",
    "is_blue_public",
    "validate_event_payload",
]
