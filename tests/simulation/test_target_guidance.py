from __future__ import annotations

from underwater_tracking.domain.adversary_models import (
    AdversaryIntentDecision,
    AdversaryMissionState,
    AdversaryOperatingBoundary,
    TargetLocalContact,
)
from underwater_tracking.domain.platforms import MotionLimits
from underwater_tracking.simulation.kinematics import MotionState
from underwater_tracking.simulation.target_guidance import resolve_target_guidance


def _mission() -> AdversaryMissionState:
    return AdversaryMissionState(
        target_id="target_00",
        task_region_id="task",
        task_region_polygon_xy=((-5000.0, -5000.0), (5000.0, -5000.0), (5000.0, 5000.0)),
        mission_route_xy=((0.0, 0.0), (1000.0, 0.0), (2000.0, 1000.0)),
        escape_regions={
            "north": ((100.0, 2000.0), (900.0, 2000.0), (500.0, 3000.0)),
        },
        current_route_index=0,
    )


def _limits() -> MotionLimits:
    return MotionLimits(
        min_speed_mps=0.0,
        max_speed_mps=14.0,
        max_acceleration_mps2=0.1,
        max_deceleration_mps2=0.1,
        max_turn_rate_rad_s=0.1,
    )


def _boundary() -> AdversaryOperatingBoundary:
    return AdversaryOperatingBoundary(min_x=-5000.0, max_x=5000.0, min_y=-5000.0, max_y=5000.0)


def test_no_contact_guidance_follows_the_next_route_point() -> None:
    result = resolve_target_guidance(
        decision=None,
        mission=_mission(),
        contacts=(),
        state=MotionState((0.0, 0.0), 0.0, 8.0),
        limits=_limits(),
        operating_boundary=_boundary(),
        exclusion_regions=(),
        sim_time_s=0,
        previous_guidance=None,
    )

    assert result.command.intent == "continue_mission"
    assert result.command.waypoint_xy == (1000.0, 0.0)
    assert result.next_route_index == 0
    assert result.command.source == "mission_route"


def test_reaching_route_point_advances_index_without_mutating_mission() -> None:
    mission = _mission()
    result = resolve_target_guidance(
        decision=None,
        mission=mission,
        contacts=(),
        state=MotionState((995.0, 0.0), 0.0, 8.0),
        limits=_limits(),
        operating_boundary=_boundary(),
        exclusion_regions=(),
        sim_time_s=30,
        previous_guidance=None,
    )

    assert result.next_route_index == 1
    assert mission.current_route_index == 0


def test_avoid_contact_uses_weighted_local_bearing_and_escape_stays_in_region() -> None:
    contact = TargetLocalContact(
        platform_id="uuv_00",
        platform_kind="uuv",
        first_seen_s=0,
        last_seen_s=0,
        estimated_range_m=500.0,
        relative_bearing_rad=0.0,
        threat_level="critical",
        status="active",
    )
    avoid = resolve_target_guidance(
        decision=AdversaryIntentDecision(
            decision_id="avoid-1",
            target_id="target_00",
            intent="avoid_contact",
            confidence=0.9,
            rationale="A critical local contact requires separation.",
        ),
        mission=_mission(),
        contacts=(contact,),
        state=MotionState((0.0, 0.0), 0.0, 8.0),
        limits=_limits(),
        operating_boundary=_boundary(),
        exclusion_regions=(),
        sim_time_s=30,
        previous_guidance=None,
    )
    escape = resolve_target_guidance(
        decision=AdversaryIntentDecision(
            decision_id="escape-1",
            target_id="target_00",
            intent="escape_to_region",
            escape_region_id="north",
            confidence=0.9,
            rationale="The local contact requires the configured northern escape.",
        ),
        mission=_mission(),
        contacts=(),
        state=MotionState((0.0, 0.0), 0.0, 8.0),
        limits=_limits(),
        operating_boundary=_boundary(),
        exclusion_regions=(),
        sim_time_s=30,
        previous_guidance=None,
    )

    assert avoid.command.waypoint_xy[0] < 0.0
    assert 2000.0 <= escape.command.waypoint_xy[1] <= 3000.0
    assert escape.command.source == "llm"


def test_llm_target_cell_is_the_short_term_submarine_guidance_goal() -> None:
    result = resolve_target_guidance(
        decision=AdversaryIntentDecision(
            decision_id="cell-1",
            target_id="target_00",
            intent="avoid_contact",
            target_cell_xy=(1500.0, -500.0),
            confidence=0.9,
            rationale="The selected cell increases separation from the local contact.",
        ),
        mission=_mission(),
        contacts=(),
        state=MotionState((0.0, 0.0), 0.0, 8.0),
        limits=_limits(),
        operating_boundary=_boundary(),
        exclusion_regions=(),
        sim_time_s=30,
        previous_guidance=None,
    )

    assert result.command.waypoint_xy == (1500.0, -500.0)
    assert result.command.source == "llm"


def test_expired_or_blocked_guidance_falls_back_to_a_safe_route() -> None:
    previous = resolve_target_guidance(
        decision=None,
        mission=_mission(),
        contacts=(),
        state=MotionState((0.0, 0.0), 0.0, 8.0),
        limits=_limits(),
        operating_boundary=_boundary(),
        exclusion_regions=(),
        sim_time_s=0,
        previous_guidance=None,
    ).command
    expired = resolve_target_guidance(
        decision=None,
        mission=_mission(),
        contacts=(),
        state=MotionState((0.0, 0.0), 0.0, 8.0),
        limits=_limits(),
        operating_boundary=_boundary(),
        exclusion_regions=(),
        sim_time_s=previous.valid_until_s,
        previous_guidance=previous,
    )

    assert expired.command.source == "mission_route"
    assert expired.command.waypoint_xy == (1000.0, 0.0)
