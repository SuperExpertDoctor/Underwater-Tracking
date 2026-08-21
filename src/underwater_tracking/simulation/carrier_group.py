"""Pure geometry and bounded rendezvous solving for the surface carrier group."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, hypot, sin
from typing import Callable

from underwater_tracking.planning.astar import AStarRoutePlanner, Bounds, RoutePlan

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class RendezvousSolution:
    endpoint_xy: Point
    eta_s: int
    route: RoutePlan
    iterations: int


@dataclass(frozen=True, slots=True)
class CommittedServiceStop:
    point_xy: Point
    earliest_s: int
    latest_s: int


def carrier_slot_position(
    leader_position_xy: Point,
    leader_heading_rad: float,
    slot_offset_xy: Point,
) -> Point:
    """Project a formation slot from the leader pose without mutating state."""
    offset_x, offset_y = slot_offset_xy
    cosine = cos(leader_heading_rad)
    sine = sin(leader_heading_rad)
    projected_x = leader_position_xy[0] + cosine * offset_x - sine * offset_y
    projected_y = leader_position_xy[1] + sine * offset_x + cosine * offset_y
    return (_clean_float(projected_x), _clean_float(projected_y))


def solve_moving_rendezvous(
    *,
    start_xy: Point,
    current_time_s: int,
    committed_stops: tuple[CommittedServiceStop, ...],
    mother_speed_mps: float,
    project_slot_at: Callable[[int], Point],
    route_planner: AStarRoutePlanner,
    forbidden_regions: tuple[Bounds, ...],
    map_bounds: Bounds,
    tolerance_m: float,
    max_iterations: int = 8,
) -> RendezvousSolution | None:
    """Find a route whose endpoint is close to the carrier slot at its ETA.

    The endpoint is updated through a bounded fixed-point iteration.  Service
    stops remain ordered, and their windows include waiting at the stop before
    the next leg begins.
    """
    if current_time_s < 0:
        raise ValueError("current_time_s must be non-negative")
    if mother_speed_mps <= 0.0:
        raise ValueError("mother_speed_mps must be positive")
    if tolerance_m < 0.0:
        raise ValueError("tolerance_m must be non-negative")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    for stop in committed_stops:
        if stop.earliest_s < 0 or stop.latest_s < stop.earliest_s:
            return None

    endpoint = project_slot_at(current_time_s)
    for iteration in range(1, max_iterations + 1):
        route = route_planner.plan(
            start_xy,
            tuple(stop.point_xy for stop in committed_stops),
            endpoint,
            forbidden_regions,
            map_bounds,
        )
        if route is None:
            return None
        eta = _route_eta(route, committed_stops, current_time_s, mother_speed_mps)
        if eta is None:
            return None
        projected_endpoint = project_slot_at(eta)
        if hypot(
            route.points[-1][0] - projected_endpoint[0],
            route.points[-1][1] - projected_endpoint[1],
        ) <= tolerance_m:
            return RendezvousSolution(
                endpoint_xy=route.points[-1],
                eta_s=eta,
                route=route,
                iterations=iteration,
            )
        endpoint = projected_endpoint
    return None


def _route_eta(
    route: RoutePlan,
    stops: tuple[CommittedServiceStop, ...],
    current_time_s: int,
    speed_mps: float,
) -> int | None:
    if len(route.stop_indices) != len(stops):
        return None
    elapsed_s = float(current_time_s)
    previous_index = 0
    for stop, stop_index in zip(stops, route.stop_indices, strict=True):
        if stop_index <= previous_index or stop_index >= len(route.points):
            return None
        elapsed_s += _distance(route.points[previous_index : stop_index + 1]) / speed_mps
        if elapsed_s > stop.latest_s + 1e-9:
            return None
        elapsed_s = max(elapsed_s, float(stop.earliest_s))
        previous_index = stop_index
    elapsed_s += _distance(route.points[previous_index:]) / speed_mps
    return int(ceil(elapsed_s))


def _distance(points: tuple[Point, ...] | list[Point]) -> float:
    return sum(
        hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(points, points[1:])
    )


def _clean_float(value: float) -> float:
    rounded = round(value, 12)
    return 0.0 if abs(rounded) <= 1e-12 else rounded
