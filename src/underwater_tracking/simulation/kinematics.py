"""Shared bounded continuous two-dimensional platform motion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from math import cos, pi, sin
from collections.abc import Sequence

from underwater_tracking.domain.platforms import MotionLimits


def wrap_angle(value: float) -> float:
    """Normalize an angle to the half-open interval [-pi, pi)."""
    if not math.isfinite(value):
        raise ValueError("angle must be finite")
    wrapped = math.remainder(value, 2.0 * pi)
    return -pi if wrapped >= pi else wrapped


@dataclass(frozen=True, slots=True)
class MotionState:
    position_xy: tuple[float, float]
    heading_rad: float
    speed_mps: float


@dataclass(frozen=True, slots=True)
class MotionCommand:
    desired_heading_rad: float
    desired_speed_mps: float


@dataclass(frozen=True, slots=True)
class NavigationBoundary:
    """Legal two-dimensional operating area for a mobile target."""

    bounds_xy: tuple[float, float, float, float]
    exclusion_polygons: tuple[tuple[tuple[float, float], ...], ...] = ()
    safety_margin_m: float = 50.0

    def __post_init__(self) -> None:
        min_x, max_x, min_y, max_y = self.bounds_xy
        if max_x <= min_x or max_y <= min_y:
            raise ValueError("navigation bounds must have positive area")
        if self.safety_margin_m < 0.0 or not math.isfinite(self.safety_margin_m):
            raise ValueError("navigation safety margin must be finite and non-negative")
        if any(len(polygon) < 3 for polygon in self.exclusion_polygons):
            raise ValueError("navigation exclusion polygons require at least three points")


class NavigationInvariantError(RuntimeError):
    """Raised when an accepted integration step would leave legal navigation."""


def stopping_distance_m(speed_mps: float, deceleration_mps2: float) -> float:
    if deceleration_mps2 <= 0:
        raise ValueError("deceleration_mps2 must be positive")
    return speed_mps * speed_mps / (2.0 * deceleration_mps2)


def minimum_turn_radius_m(speed_mps: float, turn_rate_rad_s: float) -> float:
    if turn_rate_rad_s <= 0:
        raise ValueError("turn_rate_rad_s must be positive")
    return speed_mps / turn_rate_rad_s


def advance_motion(
    state: MotionState,
    command: MotionCommand,
    limits: MotionLimits,
    dt_s: float,
) -> MotionState:
    """Advance one interval while respecting speed, acceleration, and turn bounds."""
    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative")

    desired_speed = min(
        limits.max_speed_mps,
        max(limits.min_speed_mps, command.desired_speed_mps),
    )
    max_speed_delta = (
        limits.max_acceleration_mps2
        if desired_speed >= state.speed_mps
        else limits.max_deceleration_mps2
    ) * dt_s
    speed_delta = max(
        -max_speed_delta,
        min(max_speed_delta, desired_speed - state.speed_mps),
    )
    speed = min(
        limits.max_speed_mps,
        max(limits.min_speed_mps, state.speed_mps + speed_delta),
    )

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


def constrain_navigation_command(
    state: MotionState,
    requested: MotionCommand,
    limits: MotionLimits,
    boundary: NavigationBoundary,
    dt_s: float,
) -> MotionCommand:
    """Make a requested command turn inward before a boundary violation.

    This function only changes the requested command.  The actual turn and
    acceleration remain the responsibility of ``advance_motion`` so every
    caller observes the same physical limits.
    """
    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative")
    min_x, max_x, min_y, max_y = boundary.bounds_xy
    x, y = state.position_xy
    requested_speed = min(limits.max_speed_mps, max(limits.min_speed_mps, requested.desired_speed_mps))
    heading = wrap_angle(requested.desired_heading_rad)
    speed = max(state.speed_mps, requested_speed)
    stopping_distance = stopping_distance_m(speed, limits.max_deceleration_mps2)
    turn_radius = minimum_turn_radius_m(speed, limits.max_turn_rate_rad_s)
    guard_distance = stopping_distance + turn_radius + boundary.safety_margin_m
    distances = (x - min_x, max_x - x, y - min_y, max_y - y)
    near_edge = min(distances) <= guard_distance

    def outward(candidate_heading: float) -> bool:
        vx, vy = cos(candidate_heading), sin(candidate_heading)
        return (
            (x <= min_x + boundary.safety_margin_m and vx < 0.0)
            or (x >= max_x - boundary.safety_margin_m and vx > 0.0)
            or (y <= min_y + boundary.safety_margin_m and vy < 0.0)
            or (y >= max_y - boundary.safety_margin_m and vy > 0.0)
        )

    if near_edge and outward(heading):
        heading = _heading_toward_interior(state.position_xy, boundary.bounds_xy)

    projected = (
        x + max(state.speed_mps, requested_speed) * max(dt_s, 0.0) * cos(heading),
        y + max(state.speed_mps, requested_speed) * max(dt_s, 0.0) * sin(heading),
    )
    if not _inside_bounds(projected, boundary.bounds_xy) or _segment_hits_exclusion(
        state.position_xy, projected, boundary.exclusion_polygons
    ):
        heading = _heading_toward_interior(state.position_xy, boundary.bounds_xy)
        if _point_in_any_exclusion(state.position_xy, boundary.exclusion_polygons):
            heading = _heading_away_from_exclusions(state.position_xy, boundary.exclusion_polygons)
        if not _inside_bounds(projected, boundary.bounds_xy):
            speed = min(requested_speed, max(limits.min_speed_mps, state.speed_mps))
        else:
            speed = requested_speed
        return MotionCommand(desired_heading_rad=heading, desired_speed_mps=speed)
    return MotionCommand(desired_heading_rad=heading, desired_speed_mps=requested_speed)


def navigation_segment_is_legal(
    start: tuple[float, float],
    end: tuple[float, float],
    boundary: NavigationBoundary,
) -> bool:
    """Return whether a straight integration segment remains in legal water."""
    return _inside_bounds(end, boundary.bounds_xy) and not _segment_hits_exclusion(
        start, end, boundary.exclusion_polygons
    )


def _inside_bounds(
    point: tuple[float, float], bounds: tuple[float, float, float, float]
) -> bool:
    min_x, max_x, min_y, max_y = bounds
    return min_x - 1e-9 <= point[0] <= max_x + 1e-9 and min_y - 1e-9 <= point[1] <= max_y + 1e-9


def _heading_toward_interior(
    point: tuple[float, float], bounds: tuple[float, float, float, float]
) -> float:
    min_x, max_x, min_y, max_y = bounds
    return math.atan2((min_y + max_y) * 0.5 - point[1], (min_x + max_x) * 0.5 - point[0])


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    inside = False
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        if _point_on_segment(start, end, point):
            return True
        if (start[1] > point[1]) != (end[1] > point[1]):
            crossing_x = (end[0] - start[0]) * (point[1] - start[1]) / (end[1] - start[1]) + start[0]
            if point[0] < crossing_x:
                inside = not inside
    return inside


def _point_on_segment(
    start: tuple[float, float], end: tuple[float, float], point: tuple[float, float]
) -> bool:
    cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])
    return abs(cross) <= 1e-9 and min(start[0], end[0]) - 1e-9 <= point[0] <= max(start[0], end[0]) + 1e-9 and min(start[1], end[1]) - 1e-9 <= point[1] <= max(start[1], end[1]) + 1e-9


def _segments_intersect(
    first_start: tuple[float, float], first_end: tuple[float, float],
    second_start: tuple[float, float], second_end: tuple[float, float],
) -> bool:
    def orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    first = orientation(first_start, first_end, second_start)
    second = orientation(first_start, first_end, second_end)
    third = orientation(second_start, second_end, first_start)
    fourth = orientation(second_start, second_end, first_end)
    if ((first > 0 > second) or (first < 0 < second)) and ((third > 0 > fourth) or (third < 0 < fourth)):
        return True
    return any(
        abs(value) <= 1e-9 and _point_on_segment(start, end, point)
        for value, start, end, point in (
            (first, first_start, first_end, second_start),
            (second, first_start, first_end, second_end),
            (third, second_start, second_end, first_start),
            (fourth, second_start, second_end, first_end),
        )
    )


def _segment_hits_exclusion(
    start: tuple[float, float],
    end: tuple[float, float],
    polygons: Sequence[Sequence[tuple[float, float]]],
) -> bool:
    return any(
        _point_in_polygon(start, polygon)
        or _point_in_polygon(end, polygon)
        or any(
            _segments_intersect(start, end, edge_start, edge_end)
            for edge_start, edge_end in zip(polygon, (*polygon[1:], polygon[0]))
        )
        for polygon in polygons
    )


def _point_in_any_exclusion(
    point: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]]
) -> bool:
    return any(_point_in_polygon(point, polygon) for polygon in polygons)


def _heading_away_from_exclusions(
    point: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]]
) -> float:
    centers = [
        (
            sum(vertex[0] for vertex in polygon) / len(polygon),
            sum(vertex[1] for vertex in polygon) / len(polygon),
        )
        for polygon in polygons
        if polygon
    ]
    if not centers:
        return 0.0
    center_x = sum(center[0] for center in centers) / len(centers)
    center_y = sum(center[1] for center in centers) / len(centers)
    return math.atan2(point[1] - center_y, point[0] - center_x)


__all__ = [
    "MotionCommand",
    "MotionState",
    "NavigationBoundary",
    "NavigationInvariantError",
    "advance_motion",
    "constrain_navigation_command",
    "minimum_turn_radius_m",
    "navigation_segment_is_legal",
    "stopping_distance_m",
    "wrap_angle",
]
