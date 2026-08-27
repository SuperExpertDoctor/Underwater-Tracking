"""Pure metrics for deterministic UUV tracking and coverage audit traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from itertools import pairwise
import json
from math import hypot, isfinite

import numpy as np

Point = tuple[float, float]


def deterministic_trace_digest(trace: Mapping[str, object]) -> str:
    """Return a SHA-256 digest of the trace's canonical JSON representation."""
    payload = json.dumps(
        trace,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _point(value: object) -> Point | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    try:
        point = float(value[0]), float(value[1])
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(isfinite(coordinate) for coordinate in point):
        raise ValueError("point coordinates must be finite")
    return point


def _required_points(values: object, *, field: str) -> tuple[Point, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a sequence of points")
    result: list[Point] = []
    for value in values:
        point = _point(value)
        if point is None:
            raise ValueError(f"{field} must contain two-coordinate numeric points")
        result.append(point)
    return tuple(result)


def _points_by_id(items: object, *, id_field: str) -> dict[str, Point]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return {}
    result: dict[str, Point] = {}
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        identifier = raw.get(id_field)
        point = _point(raw.get("position_xy"))
        if isinstance(identifier, str) and point is not None:
            result[identifier] = point
    return result


def _deployed_points_by_id(items: object) -> dict[str, Point]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return {}
    return _points_by_id(
        tuple(
            raw
            for raw in items
            if isinstance(raw, Mapping)
            and raw.get("deployment_state") == "deployed"
        ),
        id_field="platform_id",
    )


def target_position_errors_m(
    frames: Sequence[Mapping[str, object]],
    target_id: str,
) -> tuple[float, ...]:
    """Pair same-frame estimates and truth and return position errors in metres."""
    errors: list[float] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        truth = _points_by_id(frame.get("target_truth"), id_field="target_id").get(
            target_id
        )
        if truth is None:
            continue
        tracks = frame.get("tracks")
        if not isinstance(tracks, Sequence) or isinstance(tracks, (str, bytes)):
            continue
        for raw in tracks:
            if not isinstance(raw, Mapping) or raw.get("target_id") != target_id:
                continue
            estimate = _point(raw.get("mean"))
            if estimate is None:
                continue
            errors.append(hypot(estimate[0] - truth[0], estimate[1] - truth[1]))
            break
    return tuple(errors)


def minimum_pairwise_separation_m(
    frames: Sequence[Mapping[str, object]],
) -> float | None:
    """Return the minimum same-frame separation between deployed UUVs."""
    minimum: float | None = None
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        deployed = tuple(_deployed_points_by_id(frame.get("uuvs")).values())
        for index, left in enumerate(deployed):
            for right in deployed[index + 1 :]:
                distance = hypot(left[0] - right[0], left[1] - right[1])
                minimum = distance if minimum is None else min(minimum, distance)
    return minimum


def command_motion_counts(
    frames: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """Count commanded UUV intervals and those with observable motion."""
    commanded = 0
    moved = 0
    for current, following in pairwise(frames):
        if not isinstance(current, Mapping) or not isinstance(following, Mapping):
            continue
        commands = current.get("waypoint_commands")
        if not isinstance(commands, Mapping):
            continue
        commanded_ids = {
            uuv_id
            for by_target in commands.values()
            if isinstance(by_target, Mapping)
            for uuv_id in by_target
            if isinstance(uuv_id, str)
        }
        before = _deployed_points_by_id(current.get("uuvs"))
        after = _deployed_points_by_id(following.get("uuvs"))
        for uuv_id in sorted(commanded_ids & before.keys() & after.keys()):
            commanded += 1
            if hypot(
                after[uuv_id][0] - before[uuv_id][0],
                after[uuv_id][1] - before[uuv_id][1],
            ) > 1.0e-9:
                moved += 1
    return {"commanded_intervals": commanded, "moved_intervals": moved}


def waypoint_visit_fraction(
    trajectory: Sequence[Point],
    route: Sequence[Point],
    *,
    numerical_tolerance_m: float = 1.0e-6,
) -> float | None:
    """Return the fraction of planned waypoints physically visited."""
    if not isfinite(numerical_tolerance_m) or numerical_tolerance_m < 0.0:
        raise ValueError("numerical_tolerance_m must be finite and non-negative")
    trajectory_points = _required_points(trajectory, field="trajectory")
    route_points = _required_points(route, field="route")
    if not route_points:
        return None
    visited = sum(
        any(
            hypot(sample[0] - point[0], sample[1] - point[1])
            <= numerical_tolerance_m
            for sample in trajectory_points
        )
        for point in route_points
    )
    return visited / len(route_points)


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    x, y = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    tolerance = 1.0e-9 * max(1.0, abs(dx), abs(dy))
    cross_product = (x - x1) * dy - (y - y1) * dx
    if abs(cross_product) > tolerance:
        return False
    return (
        min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
        and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
    )


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        if _point_on_segment(point, start, end):
            return True
        x1, y1 = start
        x2, y2 = end
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


def _polygon_area_twice(polygon: Sequence[Point]) -> float:
    return sum(
        start[0] * end[1] - end[0] * start[1]
        for start, end in zip(polygon, (*polygon[1:], polygon[0]))
    )


def sampled_footprint_fraction(
    polygon: Sequence[Point],
    emissions: Sequence[tuple[Point, float]],
    *,
    samples_per_axis: int = 81,
) -> float | None:
    """Estimate the actively insonified polygon fraction on a fixed grid."""
    if samples_per_axis < 2:
        raise ValueError("samples_per_axis must be at least two")
    polygon_points = _required_points(polygon, field="polygon")
    if not isinstance(emissions, Sequence) or isinstance(emissions, (str, bytes)):
        raise TypeError("emissions must be a sequence")
    validated_emissions: list[tuple[Point, float]] = []
    for emission in emissions:
        if not isinstance(emission, Sequence) or isinstance(emission, (str, bytes)):
            raise TypeError("each emission must contain a center and radius")
        if len(emission) < 2:
            raise ValueError("each emission must contain a center and radius")
        center = _point(emission[0])
        if center is None:
            raise ValueError("emission center must contain two numeric coordinates")
        try:
            radius = float(emission[1])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("emission radius must be finite and positive") from error
        if not isfinite(radius) or radius <= 0.0:
            raise ValueError("emission radius must be finite and positive")
        validated_emissions.append((center, radius))
    if not validated_emissions:
        return None
    if len(polygon_points) < 3 or _polygon_area_twice(polygon_points) == 0.0:
        return None
    min_x = min(point[0] for point in polygon_points)
    max_x = max(point[0] for point in polygon_points)
    min_y = min(point[1] for point in polygon_points)
    max_y = max(point[1] for point in polygon_points)
    candidates = [
        (float(x), float(y))
        for x in np.linspace(min_x, max_x, samples_per_axis)
        for y in np.linspace(min_y, max_y, samples_per_axis)
        if _point_in_polygon((float(x), float(y)), polygon_points)
    ]
    if not candidates:
        return None
    covered = sum(
        any(
            hypot(point[0] - center[0], point[1] - center[1]) <= radius
            for center, radius in validated_emissions
        )
        for point in candidates
    )
    return covered / len(candidates)


def percentile_summary(values: Sequence[float]) -> dict[str, float] | None:
    """Summarize a non-empty finite metric series."""
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("values must be finite")
    return {
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }
