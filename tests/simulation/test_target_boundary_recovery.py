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


def test_boundary_recovery_times_out_explicitly() -> None:
    target = boundary_target(recovery_timeout_s=2.0)

    for _ in range(4):
        target.step(1.0, random.Random(1))

    assert target.navigation_state == "FAILED"
    assert target.last_navigation_error == "boundary_recovery_timeout"
    assert target.navigation_guard_failed is True
