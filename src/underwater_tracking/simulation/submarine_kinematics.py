"""Deterministic three-degree-of-freedom submarine integration."""

from __future__ import annotations

import math

from underwater_tracking.domain.platforms import (
    SubmarineMotionCommand,
    SubmarineMotionLimits,
    SubmarineMotionState,
)
from underwater_tracking.simulation.kinematics import (
    MotionCommand,
    MotionState,
    NavigationBoundary,
    NavigationInvariantError,
    advance_motion,
    navigation_segment_is_legal,
)


def advance_submarine_motion(
    state: SubmarineMotionState,
    command: SubmarineMotionCommand,
    limits: SubmarineMotionLimits,
    dt_s: float,
    *,
    boundary: NavigationBoundary | None = None,
) -> SubmarineMotionState:
    """Advance one bounded horizontal/depth interval.

    The caller may split a long physics step into smaller intervals.  No
    external state or model call is consulted here, which makes this function
    suitable for both the live target and invariant tests.
    """
    if dt_s < 0.0 or not math.isfinite(dt_s):
        raise ValueError("dt_s must be finite and non-negative")
    horizontal = advance_motion(
        MotionState(state.position_xy, state.heading_rad, state.speed_mps),
        MotionCommand(command.desired_heading_rad, command.desired_speed_mps),
        limits,
        dt_s,
    )
    if boundary is not None and not navigation_segment_is_legal(
        state.position_xy, horizontal.position_xy, boundary
    ):
        raise NavigationInvariantError(
            "submarine integration would leave navigation boundary"
        )

    depth = min(limits.max_depth_m, max(limits.min_depth_m, state.depth_m))
    desired_depth = min(
        limits.max_depth_m,
        max(limits.min_depth_m, command.desired_depth_m),
    )
    if dt_s == 0.0:
        vertical_speed = state.vertical_speed_mps
        next_depth = depth
    else:
        error = desired_depth - depth
        requested_speed = math.copysign(
            min(limits.max_vertical_speed_mps, abs(error) / dt_s), error
        ) if abs(error) > 1e-12 else 0.0
        max_delta = limits.max_vertical_acceleration_mps2 * dt_s
        vertical_speed = min(
            limits.max_vertical_speed_mps,
            max(
                -limits.max_vertical_speed_mps,
                state.vertical_speed_mps
                + max(-max_delta, min(max_delta, requested_speed - state.vertical_speed_mps)),
            ),
        )
        pitch_speed_limit = max(horizontal.speed_mps, 1e-9) * math.tan(
            limits.max_pitch_rad
        )
        vertical_speed = max(
            -pitch_speed_limit,
            min(pitch_speed_limit, vertical_speed),
        )
        # The pitch clamp is also an acceleration-limited command change.
        vertical_speed = max(
            state.vertical_speed_mps - max_delta,
            min(state.vertical_speed_mps + max_delta, vertical_speed),
        )
        vertical_speed = max(
            -limits.max_vertical_speed_mps,
            min(limits.max_vertical_speed_mps, vertical_speed),
        )
        next_depth = depth + vertical_speed * dt_s
        if next_depth < limits.min_depth_m:
            next_depth = limits.min_depth_m
            vertical_speed = max(0.0, (next_depth - depth) / dt_s)
        elif next_depth > limits.max_depth_m:
            next_depth = limits.max_depth_m
            vertical_speed = min(0.0, (next_depth - depth) / dt_s)

    pitch = math.atan2(vertical_speed, max(horizontal.speed_mps, 1e-9))
    if abs(pitch) > limits.max_pitch_rad + 1e-9:
        raise NavigationInvariantError("submarine pitch limit exceeded")
    return SubmarineMotionState(
        position_xy=horizontal.position_xy,
        depth_m=next_depth,
        heading_rad=horizontal.heading_rad,
        speed_mps=horizontal.speed_mps,
        vertical_speed_mps=vertical_speed,
    )


def integrate_submarine_motion(
    state: SubmarineMotionState,
    command: SubmarineMotionCommand,
    limits: SubmarineMotionLimits,
    dt_s: float,
    *,
    boundary: NavigationBoundary | None = None,
    max_substep_s: float = 0.5,
) -> SubmarineMotionState:
    """Integrate a long interval with deterministic bounded substeps."""
    if max_substep_s <= 0.0 or not math.isfinite(max_substep_s):
        raise ValueError("max_substep_s must be finite and positive")
    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative")
    steps = max(1, math.ceil(dt_s / max_substep_s))
    sub_dt = dt_s / steps
    current = state
    for _ in range(steps):
        current = advance_submarine_motion(
            current,
            command,
            limits,
            sub_dt,
            boundary=boundary,
        )
    return current


__all__ = [
    "advance_submarine_motion",
    "integrate_submarine_motion",
]
