from math import atan2, pi

import pytest

from underwater_tracking.domain.platforms import (
    SubmarineMotionCommand,
    SubmarineMotionLimits,
    SubmarineMotionState,
)
from underwater_tracking.simulation.kinematics import NavigationBoundary
from underwater_tracking.simulation.submarine_kinematics import (
    advance_submarine_motion,
    integrate_submarine_motion,
)


def _limits() -> SubmarineMotionLimits:
    return SubmarineMotionLimits(
        min_speed_mps=0.0,
        max_speed_mps=14.0,
        max_acceleration_mps2=0.5,
        max_deceleration_mps2=0.5,
        max_turn_rate_rad_s=0.2,
        min_depth_m=0.0,
        max_depth_m=900.0,
        max_vertical_speed_mps=2.0,
        max_vertical_acceleration_mps2=0.2,
        max_pitch_rad=pi / 12.0,
    )


def test_depth_integration_respects_speed_acceleration_and_pitch() -> None:
    limits = _limits()
    state = SubmarineMotionState(
        position_xy=(0.0, 0.0),
        depth_m=100.0,
        heading_rad=0.0,
        speed_mps=8.0,
        vertical_speed_mps=0.0,
    )
    command = SubmarineMotionCommand(
        desired_heading_rad=pi / 2.0,
        desired_speed_mps=14.0,
        desired_depth_m=800.0,
    )

    next_state = integrate_submarine_motion(state, command, limits, 5.0)

    assert limits.min_depth_m <= next_state.depth_m <= limits.max_depth_m
    assert abs(next_state.vertical_speed_mps) <= limits.max_vertical_speed_mps + 1e-9
    assert abs(next_state.vertical_speed_mps - state.vertical_speed_mps) <= (
        limits.max_vertical_acceleration_mps2 * 5.0 + 1e-9
    )
    assert abs(atan2(next_state.vertical_speed_mps, next_state.speed_mps)) <= (
        limits.max_pitch_rad + 1e-9
    )
    assert abs(next_state.heading_rad - state.heading_rad) <= (
        limits.max_turn_rate_rad_s * 5.0 + 1e-9
    )


def test_long_step_matches_deterministic_half_second_splits() -> None:
    limits = _limits()
    state = SubmarineMotionState(
        position_xy=(0.0, 0.0),
        depth_m=300.0,
        heading_rad=0.0,
        speed_mps=8.0,
        vertical_speed_mps=0.0,
    )
    command = SubmarineMotionCommand(
        desired_heading_rad=0.3,
        desired_speed_mps=10.0,
        desired_depth_m=600.0,
    )
    integrated = integrate_submarine_motion(state, command, limits, 5.0)
    split = state
    for _ in range(10):
        split = advance_submarine_motion(split, command, limits, 0.5)
    assert integrated == split


def test_horizontal_boundary_crossing_is_rejected() -> None:
    limits = _limits()
    state = SubmarineMotionState(
        position_xy=(9.0, 0.0),
        depth_m=100.0,
        heading_rad=0.0,
        speed_mps=4.0,
        vertical_speed_mps=0.0,
    )
    command = SubmarineMotionCommand(
        desired_heading_rad=0.0,
        desired_speed_mps=4.0,
        desired_depth_m=100.0,
    )
    with pytest.raises(RuntimeError):
        advance_submarine_motion(
            state,
            command,
            limits,
            1.0,
            boundary=NavigationBoundary((0.0, 10.0, -10.0, 10.0), safety_margin_m=0.0),
        )


def test_invalid_depth_envelope_is_rejected() -> None:
    with pytest.raises(ValueError):
        SubmarineMotionLimits(
            min_speed_mps=0.0,
            max_speed_mps=14.0,
            max_acceleration_mps2=0.5,
            max_turn_rate_rad_s=0.2,
            min_depth_m=500.0,
            max_depth_m=500.0,
            max_vertical_speed_mps=2.0,
            max_vertical_acceleration_mps2=0.2,
            max_pitch_rad=pi / 12.0,
        )
