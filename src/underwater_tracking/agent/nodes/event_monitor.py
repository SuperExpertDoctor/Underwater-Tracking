# src/underwater_tracking/agent/nodes/event_monitor.py
"""Tiered event classification and coalescing (spec 8.2).

The monitor routes each observed situation onto one of three tiers:

* STRATEGIC — initialization, target add/remove/lost, confirmed intent
  change, major failure, infeasible repair, applied expert directives;
* TACTICAL — quality warning, geometry degradation, battery rotation,
  replaceable single-UUV failure;
* INFORMATIONAL — progress timers, questions, ordinary state changes.

Quality carries hysteresis: an EWMA below the warning threshold must hold
for ``warning_hold_s`` before a warning fires, while critical quality must
persist for ``critical_hold_s`` seconds (``0`` keeps the immediate
escalation of the plain monitor) unless a hard protection trigger
(non-empty ``hard_guard_reasons``) fires immediately. Target loss is gated
on both an ungated-bearing gap of at least ``target_lost_gap_s`` and a
position-covariance trace above ``covariance_cap_m2``. Intent changes need
two consecutive analyses
passing the configured confidence/margin gates. Same-type events of one
entity merge inside the ``cooldown_s`` window — the latest payload is
retained and no duplicate is emitted; escalations always break through.
All decisions are pure functions of the observations: no randomness, no
wall clock.
"""

from __future__ import annotations

from typing import Any

from underwater_tracking.config.models import IntentChangeConfirmation
from underwater_tracking.domain.models import EventLevel, RuntimeEvent

# Spec 8.2 default routing tables for context-free event types.
# ``member_failed`` is context-dependent (group size) and handled
# separately in ``EventMonitor.classify`` and ``observe_member_failed``.
_STRATEGIC_TYPES: frozenset[str] = frozenset({
    "initialization",
    "target_added",
    "target_removed",
    "target_lost",
    "intent_change_confirmed",
    "major_failure",
    "repair_infeasible",
    "directive_applied",
    "strategic_review",
    "operational_scheme_updated",
    "intelligence_report_received",
})

_TACTICAL_TYPES: frozenset[str] = frozenset({
    "group_quality_warning",
    "geometry_degradation",
    "battery_rotation",
})

_INFORMATIONAL_TYPES: frozenset[str] = frozenset({
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
})


class EventMonitor:
    """Deterministic classifier and coalescer for carrier events."""

    def __init__(
        self,
        *,
        warning_threshold: float = 0.65,
        warning_hold_s: int = 120,
        critical_threshold: float = 0.40,
        cooldown_s: int = 300,
        scenario_id: str = "S1",
        group_min_size: int = 2,
        intent_confirmation: IntentChangeConfirmation | None = None,
        critical_hold_s: int = 0,
        target_lost_gap_s: int = 300,
        covariance_cap_m2: float = 50_000.0,
    ) -> None:
        self._warning_threshold = warning_threshold
        self._warning_hold_s = warning_hold_s
        self._critical_threshold = critical_threshold
        self._cooldown_s = cooldown_s
        self._scenario_id = scenario_id
        self._group_min_size = group_min_size
        self._confirmation = intent_confirmation or IntentChangeConfirmation()
        self._critical_hold_s = critical_hold_s
        self._target_lost_gap_s = target_lost_gap_s
        self._covariance_cap_m2 = covariance_cap_m2
        # entity id -> sim_time_s when the current below-warning streak began
        self._quality_streaks: dict[str, int] = {}
        # entity id -> sim_time_s when the current below-critical streak began
        self._critical_streaks: dict[str, int] = {}
        # entity id -> (leading label, consecutive analyses passing the gates)
        self._intent_track: dict[str, tuple[str, int]] = {}
        # (entity id, event type) -> sim_time_s of the last emission
        self._last_emitted: dict[tuple[str, str], int] = {}
        # (entity id, event type) -> latest payload retained after coalescing
        self._latest_payloads: dict[tuple[str, str], dict[str, Any]] = {}

    def observe_quality(
        self,
        entity_id: str,
        sim_time_s: int,
        quality: float,
        *,
        hard_guard_reasons: tuple[str, ...] = (),
    ) -> tuple[RuntimeEvent, ...]:
        """Route one group quality sample (spec 8.2, 13).

        A hard protection trigger (non-empty ``hard_guard_reasons``, e.g.
        covariance out of bounds) emits ``group_quality_critical``
        immediately regardless of the EWMA. Otherwise critical quality
        (below ``critical_threshold``) must persist for ``critical_hold_s``
        seconds (``0`` escalates immediately). Warning-band quality fires
        ``group_quality_warning`` (TACTICAL) only after it has stayed below
        ``warning_threshold`` for ``warning_hold_s`` seconds; recovery
        above the warning threshold resets both the warning and the
        critical streak.
        """
        if hard_guard_reasons:
            self._quality_streaks.pop(entity_id, None)
            self._critical_streaks.pop(entity_id, None)
            payload = {
                "quality": quality,
                "threshold": self._critical_threshold,
                "hard_guard_reasons": list(hard_guard_reasons),
            }
            return self._emit(
                "group_quality_critical", entity_id, sim_time_s, EventLevel.STRATEGIC, payload
            )
        if quality < self._critical_threshold:
            self._quality_streaks.pop(entity_id, None)
            streak_start = self._critical_streaks.get(entity_id)
            if streak_start is None:
                self._critical_streaks[entity_id] = sim_time_s
                streak_start = sim_time_s
            if sim_time_s - streak_start < self._critical_hold_s:
                return ()
            self._critical_streaks.pop(entity_id, None)
            payload = {"quality": quality, "threshold": self._critical_threshold}
            return self._emit(
                "group_quality_critical", entity_id, sim_time_s, EventLevel.STRATEGIC, payload
            )
        if quality >= self._warning_threshold:
            self._quality_streaks.pop(entity_id, None)
            self._critical_streaks.pop(entity_id, None)
            return ()
        self._critical_streaks.pop(entity_id, None)
        streak_start = self._quality_streaks.get(entity_id)
        if streak_start is None:
            self._quality_streaks[entity_id] = sim_time_s
            streak_start = sim_time_s
        if sim_time_s - streak_start < self._warning_hold_s:
            return ()
        payload = {"quality": quality, "threshold": self._warning_threshold}
        return self._emit("group_quality_warning", entity_id, sim_time_s, EventLevel.TACTICAL, payload)

    def observe_bearing_gap(
        self,
        entity_id: str,
        sim_time_s: int,
        *,
        last_gated_bearing_s: int,
        position_covariance_trace: float,
    ) -> tuple[RuntimeEvent, ...]:
        """Gate target loss on gap duration and covariance (spec 8.2).

        A target is confirmed lost only when no bearing has been gated for
        at least ``target_lost_gap_s`` seconds AND the position-covariance
        trace exceeds ``covariance_cap_m2`` — a long gap with small
        covariance (e.g. deliberate acoustic silence) stays tracked.
        """
        gap_s = sim_time_s - last_gated_bearing_s
        if gap_s < self._target_lost_gap_s:
            return ()
        if position_covariance_trace <= self._covariance_cap_m2:
            return ()
        payload = {
            "last_gated_bearing_s": last_gated_bearing_s,
            "gap_s": gap_s,
            "position_covariance_trace": position_covariance_trace,
            "gap_threshold_s": self._target_lost_gap_s,
            "covariance_cap_m2": self._covariance_cap_m2,
        }
        return self._emit("target_lost", entity_id, sim_time_s, EventLevel.STRATEGIC, payload)

    def observe_intent_analysis(
        self,
        entity_id: str,
        sim_time_s: int,
        *,
        leading_label: str,
        confidence: float,
        runner_up_confidence: float,
    ) -> tuple[RuntimeEvent, ...]:
        """Confirm an intent-label change after consecutive gated analyses.

        The leading hypothesis must reach ``confidence`` while leading the
        runner-up by at least ``margin``, for ``consecutive`` analyses in a
        row (spec 8.2). A failed gate or a different leading label resets
        the streak; confirmation emits ``intent_change_confirmed`` as
        STRATEGIC.
        """
        passed = (
            confidence >= self._confirmation.confidence
            and confidence - runner_up_confidence >= self._confirmation.margin
        )
        if not passed:
            self._intent_track.pop(entity_id, None)
            return ()
        tracked_label, passes = self._intent_track.get(entity_id, ("", 0))
        if tracked_label != leading_label:
            passes = 0
        passes += 1
        self._intent_track[entity_id] = (leading_label, passes)
        if passes < self._confirmation.consecutive:
            return ()
        self._intent_track.pop(entity_id, None)
        payload = {
            "label": leading_label,
            "confidence": confidence,
            "runner_up_confidence": runner_up_confidence,
        }
        return self._emit(
            "intent_change_confirmed", entity_id, sim_time_s, EventLevel.STRATEGIC, payload
        )

    def observe_member_failed(
        self,
        entity_id: str,
        sim_time_s: int,
        *,
        target_id: str,
        remaining_members: int,
    ) -> tuple[RuntimeEvent, ...]:
        """Route a UUV crash or permanent failure (spec 8.2).

        The failed member is removed immediately; while the group still
        meets the minimum size the failure is a replaceable single-UUV
        failure (TACTICAL), and once any target drops below it the event
        escalates to STRATEGIC.
        """
        payload = {
            "target_id": target_id,
            "remaining_members": remaining_members,
            "group_min_size": self._group_min_size,
        }
        level = (
            EventLevel.STRATEGIC
            if remaining_members < self._group_min_size
            else EventLevel.TACTICAL
        )
        return self._emit("member_failed", entity_id, sim_time_s, level, payload)

    def observe_repair(
        self,
        entity_id: str,
        sim_time_s: int,
        *,
        feasible: bool,
        target_id: str,
    ) -> tuple[RuntimeEvent, ...]:
        """Route the outcome of a tactical repair (spec 8.2).

        A feasible repair is an ordinary state change (INFORMATIONAL); an
        infeasible repair escalates to STRATEGIC.
        """
        payload = {"target_id": target_id, "feasible": feasible}
        if feasible:
            return self._emit(
                "repair_applied", entity_id, sim_time_s, EventLevel.INFORMATIONAL, payload
            )
        return self._emit("repair_infeasible", entity_id, sim_time_s, EventLevel.STRATEGIC, payload)

    def classify(self, event_type: str, *, payload: dict[str, Any] | None = None) -> EventLevel:
        """Route an event type onto its three-tier branch (spec 8.2).

        ``member_failed`` is context-dependent: it stays TACTICAL while the
        group keeps at least ``group_min_size`` members and escalates to
        STRATEGIC otherwise. Unknown types raise so unclassified events
        never slip through silently.
        """
        if event_type == "member_failed":
            remaining = (payload or {}).get("remaining_members")
            if not isinstance(remaining, int):
                raise ValueError(
                    "member_failed classification requires an int payload['remaining_members']"
                )
            return (
                EventLevel.STRATEGIC
                if remaining < self._group_min_size
                else EventLevel.TACTICAL
            )
        if event_type.startswith("quality_guard:"):
            return EventLevel.TACTICAL
        if event_type in _STRATEGIC_TYPES:
            return EventLevel.STRATEGIC
        if event_type in _TACTICAL_TYPES:
            return EventLevel.TACTICAL
        if event_type in _INFORMATIONAL_TYPES:
            return EventLevel.INFORMATIONAL
        raise ValueError(f"unknown event type: {event_type!r}")

    def coalesced_payload(self, entity_id: str, event_type: str) -> dict[str, Any] | None:
        """Latest payload retained for a coalesced entity/type pair."""
        return self._latest_payloads.get((entity_id, event_type))

    def _emit(
        self,
        event_type: str,
        entity_id: str,
        sim_time_s: int,
        level: EventLevel,
        payload: dict[str, Any],
    ) -> tuple[RuntimeEvent, ...]:
        """Emit one event unless a same-type event of this entity is still
        inside its cooldown window; the latest payload is always retained.
        """
        key = (entity_id, event_type)
        last_emitted_s = self._last_emitted.get(key)
        self._latest_payloads[key] = payload
        if last_emitted_s is not None and sim_time_s - last_emitted_s < self._cooldown_s:
            return ()
        self._last_emitted[key] = sim_time_s
        return (
            RuntimeEvent(
                event_id=f"{entity_id}:{event_type}:{sim_time_s}",
                scenario_id=self._scenario_id,
                sim_time_s=sim_time_s,
                event_type=event_type,
                entity_id=entity_id,
                level=level,
                payload=payload,
            ),
        )
