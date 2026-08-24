"""Adversary-private decisions never enter the blue planning projection."""

from __future__ import annotations

from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.domain.event_registry import (
    EVENT_REGISTRY,
    EventAudience,
    event_definition,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent, SituationSnapshot


def test_private_decision_definition_excludes_blue_planning() -> None:
    definition = event_definition("target_mission_decision")
    assert EventAudience.OPERATOR_AUDIT in definition.audiences
    assert EventAudience.MEMORY_SOURCE in definition.audiences
    assert EventAudience.BLUE_PLANNING not in definition.audiences


def test_private_events_are_filtered_before_event_monitor_and_public_observation_is_classified() -> None:
    private = RuntimeEvent(
        event_id="decision:T1:D1",
        scenario_id="S1",
        sim_time_s=30,
        event_type="target_mission_decision",
        entity_id="T1",
        level=EventLevel.STRATEGIC,
        audiences=frozenset(
            {
                EventAudience.ADVERSARY_PRIVATE,
                EventAudience.OPERATOR_AUDIT,
                EventAudience.MEMORY_SOURCE,
            }
        ),
        payload={"guidance_waypoint_xy": (1.0, 2.0)},
    )
    observed = RuntimeEvent(
        event_id="observed:T1:30",
        scenario_id="S1",
        sim_time_s=30,
        event_type="target_maneuver_observed",
        entity_id="T1",
        level=EventLevel.TACTICAL,
        payload={"observation_ids": ["obs-1"], "speed_regime": "high"},
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=30,
        uuvs=(),
        group_reports=(),
        pending_events=(private, observed),
    )
    blue_events = tuple(
        event
        for event in snapshot.pending_events
        if EventAudience.BLUE_PLANNING in event.audiences
    )
    assert all(event.event_type != "target_mission_decision" for event in blue_events)
    assert "guidance_waypoint_xy" not in snapshot.model_copy(
        update={"pending_events": blue_events}
    ).model_dump_json()
    assert EventMonitor().classify(observed.event_type) is EventLevel.TACTICAL
    assert "target_maneuver_observed" in EVENT_REGISTRY


def test_public_target_estimate_update_is_blue_planning_evidence() -> None:
    definition = event_definition("target_estimate_updated")
    assert EventAudience.BLUE_PLANNING in definition.audiences
    assert definition.plan_impact_policy == "evidence_required"
    event = RuntimeEvent(
        event_id="estimate:T1:30",
        scenario_id="S1",
        sim_time_s=30,
        event_type="target_estimate_updated",
        entity_id="T1",
        level=EventLevel.TACTICAL,
        payload={"observation_ids": ("obs-1",), "source": "fused_public_estimate"},
    )
    assert EventAudience.BLUE_PLANNING in event.audiences
