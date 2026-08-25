"""Outer patrol geometry for a carrier battle group around task regions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import atan2, cos, hypot, pi, sin

Point = tuple[float, float]
Bounds = tuple[float, float, float, float]

_ORBIT_SIDE_COUNT = 16
_MINIMUM_CLEARANCE_M = 1_000.0


def task_region_bounds(
    region_polygons: Iterable[Sequence[Point]],
) -> tuple[Bounds, ...]:
    """Return rectangular no-go bounds for valid task-region polygons."""
    bounds: list[Bounds] = []
    for polygon in region_polygons:
        if len(polygon) < 3:
            continue
        xs, ys = zip(*polygon, strict=True)
        bounds.append((min(xs), max(xs), min(ys), max(ys)))
    return tuple(bounds)


def point_in_task_region(point: Point, bounds: Bounds) -> bool:
    """Return whether a point lies in the open interior of a task region."""
    return bounds[0] < point[0] < bounds[1] and bounds[2] < point[1] < bounds[3]


def build_outer_task_orbit(
    current_position: Point,
    *,
    current_heading_rad: float,
    region_polygons: Iterable[Sequence[Point]],
    formation_radius_m: float,
) -> tuple[Point, ...]:
    """Build a transit-plus-loop patrol route that encloses all task regions.

    The ring is sized from the task-region bounding circle.  Its apothem
    includes the widest follower slot plus one 1 km grid-cell clearance, so
    the straight legs of the polygonal approximation do not enter a task
    region even while the formation is offset from the leader.
    """
    vertices = tuple(
        point
        for polygon in region_polygons
        if len(polygon) >= 3
        for point in polygon
    )
    if not vertices:
        return ()
    if formation_radius_m < 0.0:
        raise ValueError("formation_radius_m must be non-negative")

    min_x = min(point[0] for point in vertices)
    max_x = max(point[0] for point in vertices)
    min_y = min(point[1] for point in vertices)
    max_y = max(point[1] for point in vertices)
    center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    task_radius = max(hypot(point[0] - center[0], point[1] - center[1]) for point in vertices)
    apothem = task_radius + formation_radius_m + _MINIMUM_CLEARANCE_M
    orbit_radius = apothem / cos(pi / _ORBIT_SIDE_COUNT)
    radial_x = current_position[0] - center[0]
    radial_y = current_position[1] - center[1]
    if hypot(radial_x, radial_y) <= 1e-9:
        entry_angle = current_heading_rad
    else:
        entry_angle = atan2(radial_y, radial_x)
    ring = tuple(
        (
            center[0] + orbit_radius * cos(entry_angle + 2.0 * pi * index / _ORBIT_SIDE_COUNT),
            center[1] + orbit_radius * sin(entry_angle + 2.0 * pi * index / _ORBIT_SIDE_COUNT),
        )
        for index in range(_ORBIT_SIDE_COUNT)
    )
    return (current_position, *ring)
