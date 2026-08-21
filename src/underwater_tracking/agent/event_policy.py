"""Plan-impact event policy shared by the runtime and live publisher."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from underwater_tracking.domain.models import RuntimeEvent


class EventDisposition(StrEnum):
    """The control consequence of an observed event."""

    AUDIT_ONLY = "audit_only"
    TACTICAL = "tactical"
    CANDIDATE = "candidate"
    KEY = "key"


@dataclass(frozen=True)
class PlanImpactAssessment:
    """Evidence-backed classification used by the planning boundary."""

    disposition: EventDisposition
    plan_impact: bool
    severity: int
    affected_target_ids: tuple[str, ...] = ()
    affected_region_ids: tuple[str, ...] = ()
    reason: str = ""


@dataclass
class _Episode:
    started_s: int
    confirmations: int = 0
    emitted_severity: int = 0


class EventEpisodeGate:
    """Emit on abnormal entry/escalation and release on recovery."""

    def __init__(self) -> None:
        self._episodes: dict[str, _Episode] = {}

    def observe(
        self,
        key: str,
        sim_time_s: int,
        *,
        active: bool,
        hold_s: int = 0,
        confirmations: int = 1,
        severity: int = 1,
    ) -> bool:
        if not active:
            self._episodes.pop(key, None)
            return False
        if hold_s < 0:
            raise ValueError("hold_s must be non-negative")
        if confirmations < 1:
            raise ValueError("confirmations must be positive")
        episode = self._episodes.get(key)
        if episode is None:
            episode = _Episode(started_s=sim_time_s)
            self._episodes[key] = episode
        episode.confirmations += 1
        ready = (
            sim_time_s - episode.started_s >= hold_s
            and episode.confirmations >= confirmations
        )
        if not ready:
            return False
        if episode.emitted_severity == 0:
            episode.emitted_severity = severity
            return True
        if severity > episode.emitted_severity:
            episode.emitted_severity = severity
            return True
        return False


_DIRECT_KEY_TYPES = frozenset(
    {
        "initialization",
        "target_added",
        "target_removed",
        "target_lost",
        "target_reacquired",
        "major_failure",
        "repair_infeasible",
        "uuv_failed",
        "uuv_capability_lost",
        "uuv_range_exhausted",
        "uuv_energy_depleted",
        "carrier_task_window_missed",
        "directive_applied",
        "operational_scheme_updated",
        "strategic_review",
    }
)
_IMPACT_CANDIDATE_TYPES = frozenset(
    {
        "group_quality_critical",
        "region_coverage_degraded",
        "communication_link_lost",
        "covariance_threshold_exceeded",
        "intent_change_confirmed",
        "target_intent_changed",
        "imm_confidence_shifted",
        "target_exit_predicted",
        "regional_feedback_received",
        "endurance_threshold_crossed",
        "intelligence_report_received",
        "handoff_blocked",
        "carrier_rendezvous_infeasible",
    }
)
_TACTICAL_TYPES = frozenset(
    {
        "group_quality_warning",
        "battery_rotation",
        "target_maneuver",
        "target_detection_acquired",
        "target_detection_lost",
        "carrier_recovery_health_check_pending",
    }
)
_AUDIT_ONLY_TYPES = frozenset(
    {
        "active_ping",
        "group_report_published",
        "progress_report",
        "state_changed",
        "strategic_review",
        "llm_degraded",
        "carrier_dispatch_completed",
        "carrier_recovery_completed",
        "periodic_situation_summary",
        "target_entered_region",
        "handoff_completed",
        "uuv_deployed",
        "uuv_recovered",
        "manual_sensor_mode",
        "question",
        "repair_applied",
        "contact_classified",
        "uuv_recovery_requested",
    }
)

_QUALITY_IMPACT_TYPES = frozenset(
    {
        "group_quality_critical",
        "region_coverage_degraded",
        "regional_feedback_received",
    }
)


def evaluate_plan_impact(
    event: RuntimeEvent,
    *,
    active_target_ids: Sequence[str] = (),
    active_region_ids: Sequence[str] = (),
    active_uuv_ids: Sequence[str] = (),
    quality_by_target: Mapping[str, float] | None = None,
    required_quality_by_target: Mapping[str, float] | None = None,
    target_corridor_changed: bool = False,
    resource_feasible: bool = True,
    communication_healthy: bool = True,
    time_window_feasible: bool = True,
) -> PlanImpactAssessment:
    """Classify an event only after comparing it with the active plan."""
    payload = event.payload
    event_type = event.event_type
    target_ids = set(active_target_ids)
    region_ids = set(active_region_ids)
    uuv_ids = set(active_uuv_ids)
    affected_target = _string_value(payload.get("target_id"))
    affected_region = _string_value(payload.get("region_id"))
    if affected_region is None and event_type == "region_coverage_degraded":
        affected_region = event.entity_id
    if affected_target is None and event.entity_id in target_ids:
        affected_target = event.entity_id
    affected_targets = (affected_target,) if affected_target in target_ids else ()
    affected_regions = (affected_region,) if affected_region in region_ids else ()

    quality_by_target = quality_by_target or {}
    required_quality_by_target = required_quality_by_target or {}
    quality_breach = any(
        target_id in required_quality_by_target
        and quality_by_target[target_id] < required_quality_by_target[target_id]
        for target_id in target_ids
        if target_id in quality_by_target
    )
    explicit_impact = payload.get("plan_impact") is True
    target_quality_breach = (
        affected_target is not None
        and affected_target in quality_by_target
        and affected_target in required_quality_by_target
        and quality_by_target[affected_target]
        < required_quality_by_target[affected_target]
    )
    affected_quality_breach = (
        event_type in _QUALITY_IMPACT_TYPES
        and (
            (bool(affected_targets) and target_quality_breach)
            or (not affected_targets and quality_breach)
        )
    )
    structural_impact = (
        explicit_impact
        or bool(
            affected_regions
            and event_type
            in {"region_coverage_degraded", "communication_link_lost"}
        )
        or (
            event_type in {
                "target_intent_changed",
                "imm_confidence_shifted",
                "intent_change_confirmed",
            }
            and bool(affected_targets)
            and (target_corridor_changed or payload.get("confirmed") is True)
        )
        or (event_type == "target_exit_predicted" and not bool(payload.get("successor_available", True)))
        or (event_type == "endurance_threshold_crossed" and event.entity_id in uuv_ids)
        or (
            event_type in _DIRECT_KEY_TYPES
            and (
                not resource_feasible
                or not communication_healthy
                or not time_window_feasible
            )
        )
        or affected_quality_breach
        or (event_type in _DIRECT_KEY_TYPES and event.entity_id in uuv_ids)
    )

    if event_type in _DIRECT_KEY_TYPES:
        return PlanImpactAssessment(
            disposition=EventDisposition.KEY,
            plan_impact=structural_impact,
            severity=3,
            affected_target_ids=affected_targets,
            affected_region_ids=affected_regions,
            reason="direct resource or target lifecycle transition",
        )
    if event_type in _IMPACT_CANDIDATE_TYPES:
        return PlanImpactAssessment(
            disposition=EventDisposition.KEY if structural_impact else EventDisposition.CANDIDATE,
            plan_impact=structural_impact,
            severity=3 if structural_impact else 2,
            affected_target_ids=affected_targets,
            affected_region_ids=affected_regions,
            reason="active plan quality, feasibility, or corridor impact" if structural_impact else "awaiting active-plan impact",
        )
    if event_type in _TACTICAL_TYPES:
        return PlanImpactAssessment(
            disposition=EventDisposition.TACTICAL,
            plan_impact=structural_impact,
            severity=2,
            affected_target_ids=affected_targets,
            affected_region_ids=affected_regions,
            reason="tactical observation or resource warning",
        )
    if event_type in _AUDIT_ONLY_TYPES:
        return PlanImpactAssessment(
            disposition=EventDisposition.AUDIT_ONLY,
            plan_impact=False,
            severity=1,
            affected_target_ids=affected_targets,
            affected_region_ids=affected_regions,
            reason="audit and memory input only",
        )
    return PlanImpactAssessment(
        disposition=EventDisposition.CANDIDATE,
        plan_impact=explicit_impact,
        severity=2 if explicit_impact else 1,
        affected_target_ids=affected_targets,
        affected_region_ids=affected_regions,
        reason="unknown event requires explicit plan_impact evidence",
    )


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
