"""Continuous separation checks for synchronously executed route segments."""

from __future__ import annotations

from math import hypot, isfinite

Point = tuple[float, float]

_DISTANCE_TOLERANCE_M = 1.0e-6


def minimum_synchronous_separation_m(
    first_start: Point,
    first_end: Point,
    second_start: Point,
    second_end: Point,
) -> float:
    """Return the closest distance while both segments run over ``t in [0, 1]``.

    The check is time-aware: two geometric segments may intersect safely when
    the UUVs reach the crossing at different normalized times.  Conversely,
    safe-looking endpoints are rejected when the two UUVs cross at the same
    point halfway through the transition.
    """

    values = (*first_start, *first_end, *second_start, *second_end)
    if not all(isfinite(value) for value in values):
        raise ValueError("route coordinates must be finite")
    relative_start = (
        first_start[0] - second_start[0],
        first_start[1] - second_start[1],
    )
    relative_delta = (
        (first_end[0] - first_start[0]) - (second_end[0] - second_start[0]),
        (first_end[1] - first_start[1]) - (second_end[1] - second_start[1]),
    )
    delta_squared = (
        relative_delta[0] * relative_delta[0]
        + relative_delta[1] * relative_delta[1]
    )
    if delta_squared <= 1.0e-18:
        closest_time = 0.0
    else:
        closest_time = max(
            0.0,
            min(
                1.0,
                -(
                    relative_start[0] * relative_delta[0]
                    + relative_start[1] * relative_delta[1]
                )
                / delta_squared,
            ),
        )
    return hypot(
        relative_start[0] + closest_time * relative_delta[0],
        relative_start[1] + closest_time * relative_delta[1],
    )


def transition_separation_is_safe(
    first_start: Point,
    first_end: Point,
    second_start: Point,
    second_end: Point,
    *,
    min_separation_m: float,
) -> bool:
    """Check endpoint and swept-path separation for one synchronous step.

    A pair that already starts inside the requested separation may spread out:
    its distance must be non-decreasing from launch and must reach the requested
    separation at the endpoint.  This permits deterministic deployment from a
    shared carrier boundary while rejecting any initial convergence or crossing.
    """

    if not isfinite(min_separation_m) or min_separation_m < 0.0:
        raise ValueError("min_separation_m must be finite and non-negative")
    start_relative = (
        first_start[0] - second_start[0],
        first_start[1] - second_start[1],
    )
    relative_delta = (
        (first_end[0] - first_start[0]) - (second_end[0] - second_start[0]),
        (first_end[1] - first_start[1]) - (second_end[1] - second_start[1]),
    )
    start_distance = hypot(*start_relative)
    end_distance = hypot(
        first_end[0] - second_end[0],
        first_end[1] - second_end[1],
    )
    if end_distance < min_separation_m - _DISTANCE_TOLERANCE_M:
        return False
    if start_distance >= min_separation_m - _DISTANCE_TOLERANCE_M:
        return (
            minimum_synchronous_separation_m(
                first_start,
                first_end,
                second_start,
                second_end,
            )
            >= min_separation_m - _DISTANCE_TOLERANCE_M
        )
    initial_distance_derivative = (
        start_relative[0] * relative_delta[0]
        + start_relative[1] * relative_delta[1]
    )
    return initial_distance_derivative >= -_DISTANCE_TOLERANCE_M


__all__ = [
    "minimum_synchronous_separation_m",
    "transition_separation_is_safe",
]
