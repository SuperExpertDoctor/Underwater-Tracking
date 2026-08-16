"""Shared bounded continuous two-dimensional platform motion."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

from underwater_tracking.domain.platforms import MotionLimits


def wrap_angle(value: float) -> float:
    """Normalize an angle to the half-open interval [-pi, pi)."""
    if -pi <= value < pi:
        return value
    return (value + pi) % (2.0 * pi) - pi


@dataclass(frozen=True, slots=True)
class MotionState:
    position_xy: tuple[float, float]
    heading_rad: float
    speed_mps: float


@dataclass(frozen=True, slots=True)
class MotionCommand:
    desired_heading_rad: float
    desired_speed_mps: float


def advance_motion(
    state: MotionState,
    command: MotionCommand,
    limits: MotionLimits,
    dt_s: float,
) -> MotionState:
    """Advance one interval while respecting speed, acceleration, and turn bounds."""
    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative")

    desired_speed = min(limits.max_speed_mps, max(0.0, command.desired_speed_mps))
    max_speed_delta = limits.max_acceleration_mps2 * dt_s
    speed_delta = max(
        -max_speed_delta,
        min(max_speed_delta, desired_speed - state.speed_mps),
    )
    speed = min(limits.max_speed_mps, max(0.0, state.speed_mps + speed_delta))

    max_heading_delta = limits.max_turn_rate_rad_s * dt_s
    heading_error = wrap_angle(command.desired_heading_rad - state.heading_rad)
    heading_delta = max(
        -max_heading_delta,
        min(max_heading_delta, heading_error),
    )
    heading = wrap_angle(state.heading_rad + heading_delta)
    distance = speed * dt_s
    return MotionState(
        position_xy=(
            state.position_xy[0] + distance * cos(heading),
            state.position_xy[1] + distance * sin(heading),
        ),
        heading_rad=heading,
        speed_mps=speed,
    )
