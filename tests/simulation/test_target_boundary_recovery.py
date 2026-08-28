"""Recoverable target navigation at map boundaries."""

from __future__ import annotations

import math
import random
from itertools import pairwise

import pytest

from underwater_tracking.simulation.kinematics import (
    NavigationBoundary,
    navigation_segment_is_legal,
)
from underwater_tracking.simulation.target import HiddenIntent, TargetEntity


BOUNDS = (-10_000.0, 10_000.0, -10_000.0, 10_000.0)


def boundary_target(*, recovery_timeout_s: float = 300.0) -> TargetEntity:
    return TargetEntity(
        target_id="T1",
        position_xy=(9_700.0, 0.0),
        velocity_xy=(12.0, 0.0),
        intent=HiddenIntent.TRANSIT,
        bounds_xy=BOUNDS,
        max_acceleration_mps2=0.5,
        max_deceleration_mps2=0.8,
        max_turn_rate_rad_s=0.05,
        boundary_recovery_timeout_s=recovery_timeout_s,
    )


def test_target_recovers_from_an_outward_boundary_approach() -> None:
    target = boundary_target()
    states: list[str] = []
    positions = [target.position_xy]
    speeds = [math.hypot(*target.velocity_xy)]

    for _ in range(240):
        target.step(1.0, random.Random(9))
        states.append(target.navigation_state)
        positions.append(target.position_xy)
        speeds.append(math.hypot(*target.velocity_xy))
        if target.navigation_state == "NORMAL" and "BOUNDARY_RECOVERING" in states:
            break

    assert "BOUNDARY_DECELERATING" in states
    assert "BOUNDARY_TURNING" in states
    assert "BOUNDARY_RECOVERING" in states
    assert states[-1] == "NORMAL"
    assert min(speeds) < speeds[0]
    assert all(
        BOUNDS[0] <= x <= BOUNDS[1] and BOUNDS[2] <= y <= BOUNDS[3]
        for x, y in positions
    )
    boundary = NavigationBoundary(BOUNDS)
    assert all(
        navigation_segment_is_legal(start, end, boundary)
        for start, end in pairwise(positions)
    )
    final_x, final_y = target.position_xy
    final_margin = min(
        final_x - BOUNDS[0],
        BOUNDS[1] - final_x,
        final_y - BOUNDS[2],
        BOUNDS[3] - final_y,
    )
    assert final_margin > target.navigation_guard_distance_m
    assert target.last_navigation_error is None
    assert target.navigation_guard_failed is False


def test_recovery_waypoint_retains_the_computed_guard_margin() -> None:
    target = boundary_target()

    target.step(1.0, random.Random(9))

    waypoint = target.navigation_recovery_waypoint_xy
    assert waypoint is not None
    x, y = waypoint
    waypoint_margin = min(
        x - BOUNDS[0], BOUNDS[1] - x, y - BOUNDS[2], BOUNDS[3] - y
    )
    assert target.navigation_guard_distance_m == pytest.approx(380.0)
    assert waypoint_margin >= target.navigation_guard_distance_m


def test_recovery_requires_two_consecutive_legal_physical_substeps() -> None:
    target = boundary_target()
    boundary = NavigationBoundary(BOUNDS)

    for _ in range(600):
        target.step(0.5, random.Random(9))
        if target.navigation_state == "BOUNDARY_RECOVERING":
            break

    assert target.navigation_state == "BOUNDARY_RECOVERING"
    first_qualifying_step_seen = False
    for _ in range(600):
        start = target.position_xy
        target.step(0.5, random.Random(9))
        end = target.position_xy
        margin = min(
            end[0] - BOUNDS[0],
            BOUNDS[1] - end[0],
            end[1] - BOUNDS[2],
            BOUNDS[3] - end[1],
        )
        if margin <= target.navigation_guard_distance_m:
            continue

        first_qualifying_step_seen = True
        assert target.navigation_state == "BOUNDARY_RECOVERING"
        assert navigation_segment_is_legal(start, end, boundary)

        second_start = target.position_xy
        target.step(0.5, random.Random(9))
        second_end = target.position_xy
        assert navigation_segment_is_legal(second_start, second_end, boundary)
        second_margin = min(
            second_end[0] - BOUNDS[0],
            BOUNDS[1] - second_end[0],
            second_end[1] - BOUNDS[2],
            BOUNDS[3] - second_end[1],
        )
        assert second_margin > target.navigation_guard_distance_m
        assert target.navigation_state == "NORMAL"
        break

    assert first_qualifying_step_seen


def test_boundary_deceleration_can_stop_without_losing_body_heading() -> None:
    target = TargetEntity(
        target_id="T1",
        position_xy=(140.0, 0.0),
        velocity_xy=(4.0, 0.0),
        intent=HiddenIntent.TRANSIT,
        bounds_xy=(-200.0, 200.0, -200.0, 200.0),
        max_acceleration_mps2=0.5,
        max_deceleration_mps2=2.0,
        max_turn_rate_rad_s=0.05,
    )
    initial_heading = target.heading_rad
    decelerating_headings: list[float] = []
    decelerating_speeds: list[float] = []

    for _ in range(12):
        target.step(0.5, random.Random(2))
        if target.navigation_state == "BOUNDARY_DECELERATING":
            decelerating_headings.append(target.heading_rad)
            decelerating_speeds.append(math.hypot(*target.velocity_xy))
        if target.navigation_state == "BOUNDARY_TURNING":
            break

    assert decelerating_headings
    assert all(heading == pytest.approx(initial_heading) for heading in decelerating_headings)
    assert min(decelerating_speeds) < 4.0
    assert math.hypot(*target.velocity_xy) == pytest.approx(0.0)
    assert target.heading_rad == pytest.approx(initial_heading)
    assert target.navigation_state == "BOUNDARY_TURNING"


def test_zero_speed_boundary_turn_respects_max_turn_rate() -> None:
    target = TargetEntity(
        target_id="T1",
        position_xy=(140.0, 0.0),
        velocity_xy=(4.0, 0.0),
        intent=HiddenIntent.TRANSIT,
        bounds_xy=(-200.0, 200.0, -200.0, 200.0),
        max_acceleration_mps2=0.5,
        max_deceleration_mps2=2.0,
        max_turn_rate_rad_s=0.05,
    )

    for _ in range(20):
        target.step(0.5, random.Random(2))
        if target.navigation_state == "BOUNDARY_TURNING":
            break

    assert target.navigation_state == "BOUNDARY_TURNING"
    increments: list[float] = []
    while target.navigation_state == "BOUNDARY_TURNING":
        previous_heading = target.heading_rad
        assert math.hypot(*target.velocity_xy) == pytest.approx(0.0)
        target.step(0.5, random.Random(2))
        increments.append(
            abs(
                math.atan2(
                    math.sin(target.heading_rad - previous_heading),
                    math.cos(target.heading_rad - previous_heading),
                )
            )
        )

    assert increments
    assert all(increment <= target.max_turn_rate_rad_s * 0.5 + 1e-9 for increment in increments)


def test_boundary_recovery_times_out_at_exact_configured_age() -> None:
    target = boundary_target(recovery_timeout_s=2.0)

    target.step(1.0, random.Random(1))
    assert target.navigation_state == "BOUNDARY_DECELERATING"

    target.step(1.0, random.Random(1))

    assert target.navigation_state == "FAILED"
    assert target.last_navigation_error == "boundary_recovery_timeout"
    assert target.navigation_guard_failed is True
    failure = target.consume_navigation_transitions()[-1]
    assert failure.old_state == "BOUNDARY_DECELERATING"
    assert failure.new_state == "FAILED"
    assert failure.error_reason == "boundary_recovery_timeout"
    assert failure.state_age_s == pytest.approx(2.0)

    failed_position = target.position_xy
    target.step(0.5, random.Random(1))
    assert target.position_xy == pytest.approx(failed_position)
