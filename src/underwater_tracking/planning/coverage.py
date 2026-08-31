"""Deterministic area-coverage paths for UUV active-sonar search."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import nextafter

Point = tuple[float, float]


def serpentine_coverage_waypoints(
    polygon: Sequence[Point],
    *,
    lane_count: int,
) -> tuple[Point, ...]:
    """Return alternating scan-line endpoints contained by ``polygon``.

    The path is deterministic and works for both rectangular task regions and
    simple polygons.  Horizontal edges are ignored when finding intersections
    so a vertex is not counted twice; the upper boundary is sampled just below
    its exact y-coordinate while retaining the requested boundary in output.
    """
    segments = _coverage_segments(polygon, lane_count=lane_count)
    points: list[Point] = []
    for start, end in segments:
        points.extend((start, end))
    return tuple(points)


def serpentine_coverage_waypoints_by_uuv(
    polygon: Sequence[Point],
    uuv_ids: Iterable[str],
    *,
    start_point: Point | None = None,
) -> dict[str, tuple[Point, ...]]:
    """Assign coverage lanes to a stable set of UUVs.

    When a common deployment point is supplied, every lane is oriented from
    its endpoint nearest that point.  This preserves the scan lanes while
    preventing co-located UUVs from initially steering along opposite ends of
    a serpentine path and delaying the required observation geometry.
    """
    ids = tuple(sorted(dict.fromkeys(str(uuv_id) for uuv_id in uuv_ids)))
    if not ids:
        return {}
    segments = _coverage_segments(polygon, lane_count=len(ids))
    paths: dict[str, list[Point]] = {uuv_id: [] for uuv_id in ids}
    for index, segment in enumerate(segments):
        oriented = segment
        if start_point is not None:
            start, end = segment
            start_distance = (start[0] - start_point[0]) ** 2 + (start[1] - start_point[1]) ** 2
            end_distance = (end[0] - start_point[0]) ** 2 + (end[1] - start_point[1]) ** 2
            if end_distance < start_distance:
                oriented = (end, start)
        paths[ids[index % len(ids)]].extend(oriented)
    return {uuv_id: tuple(paths[uuv_id]) for uuv_id in ids}


def _coverage_segments(
    polygon: Sequence[Point],
    *,
    lane_count: int,
) -> tuple[tuple[Point, Point], ...]:
    if lane_count < 1:
        raise ValueError("lane_count must be positive")
    points = tuple((float(x), float(y)) for x, y in polygon)
    if len(points) < 3 or len(set(points)) < 3:
        raise ValueError("coverage polygon requires at least three unique points")
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    if max_y <= min_y:
        raise ValueError("coverage polygon must have positive y extent")
    lane_ys: tuple[float, ...]
    if lane_count == 1:
        lane_ys = ((min_y + max_y) / 2.0,)
    else:
        step = (max_y - min_y) / (lane_count - 1)
        lane_ys = tuple(min_y + index * step for index in range(lane_count))

    segments: list[tuple[Point, Point]] = []
    for lane_index, requested_y in enumerate(lane_ys):
        sample_y = (
            nextafter(requested_y, min_y)
            if requested_y == max_y
            else requested_y
        )
        x_intersections = _scanline_intersections(points, sample_y)
        if len(x_intersections) < 2:
            continue
        for pair_index in range(0, len(x_intersections) - 1, 2):
            left = x_intersections[pair_index]
            right = x_intersections[pair_index + 1]
            if right - left <= 1e-9:
                continue
            endpoints = ((left, requested_y), (right, requested_y))
            segments.append(endpoints if lane_index % 2 == 0 else endpoints[::-1])
    if not segments and lane_count > 1:
        # A polygon whose vertices lie on the requested lanes can produce
        # duplicate intersections at every boundary. Retry with lane centers
        # so valid interior coverage is still available for multi-UUV scans.
        lane_step = (max_y - min_y) / lane_count
        for lane_index in range(lane_count):
            requested_y = min_y + (lane_index + 0.5) * lane_step
            x_intersections = _scanline_intersections(points, requested_y)
            for pair_index in range(0, len(x_intersections) - 1, 2):
                left = x_intersections[pair_index]
                right = x_intersections[pair_index + 1]
                if right - left <= 1e-9:
                    continue
                endpoints = ((left, requested_y), (right, requested_y))
                segments.append(endpoints if lane_index % 2 == 0 else endpoints[::-1])
    if not segments:
        raise ValueError("coverage polygon has no usable scan lanes")
    return tuple(segments)


def _scanline_intersections(polygon: Sequence[Point], y: float) -> tuple[float, ...]:
    intersections: list[float] = []
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        x1, y1 = start
        x2, y2 = end
        if abs(y2 - y1) <= 1e-12:
            continue
        lower, upper = sorted((y1, y2))
        if not lower <= y < upper:
            continue
        ratio = (y - y1) / (y2 - y1)
        intersections.append(x1 + ratio * (x2 - x1))
    return tuple(sorted(intersections))
