"""Deterministic four-region geometry for live execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise, product
from math import atan2, cos, degrees, hypot, radians, sin, sqrt
from typing import Literal

from shapely import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from underwater_tracking.domain.execution_models import ExecutionRegion
from underwater_tracking.domain.prediction_models import AcceptedPrediction


RegionGenerationMode = Literal[
    "imm",
    "degraded_prediction",
    "boundary_recovery",
    "reprojected_previous",
]

WINDOW_OFFSETS_S = ((0.0, 540.0), (450.0, 990.0), (900.0, 1_440.0), (1_350.0, 1_800.0))
_MIN_REGION_AREA_M2 = 62_500.0
_MIN_REGION_WIDTH_M = 250.0


@dataclass(frozen=True, slots=True)
class FourRegionBaseline:
    regions: tuple[ExecutionRegion, ExecutionRegion, ExecutionRegion, ExecutionRegion]
    mode: RegionGenerationMode
    reason_codes: tuple[str, ...]


def build_four_region_baseline(
    accepted: AcceptedPrediction,
    *,
    target_id: str,
    execution_revision: int,
    origin_sim_time_s: float,
    map_bounds_xy: tuple[float, float, float, float],
    prior_regions: Sequence[ExecutionRegion] = (),
) -> FourRegionBaseline:
    """Build the immutable four-slot geometry baseline for one planning cycle."""
    _validate_inputs(target_id, execution_revision, origin_sim_time_s, map_bounds_xy)
    prediction = accepted.prediction
    if prediction is None:
        return _reproject_previous(
            prior_regions,
            target_id=target_id,
            execution_revision=execution_revision,
            origin_sim_time_s=origin_sim_time_s,
            map_bounds_xy=map_bounds_xy,
            reason_codes=accepted.health.reason_codes,
        )
    if prediction.target_id != target_id:
        raise ValueError("accepted prediction target does not match baseline target")

    points = tuple((float(x), float(y)) for x, y in prediction.points_xy)
    if not points:
        return _reproject_previous(
            prior_regions,
            target_id=target_id,
            execution_revision=execution_revision,
            origin_sim_time_s=origin_sim_time_s,
            map_bounds_xy=map_bounds_xy,
            reason_codes=(*accepted.health.reason_codes, "accepted_prediction_has_no_geometry"),
        )
    times = tuple(float(value) for value in prediction.times_s)
    requested_end_s = origin_sim_time_s + WINDOW_OFFSETS_S[-1][1]
    if (
        len(times) != len(points)
        or any(right <= left for left, right in pairwise(times))
        or times[0] > origin_sim_time_s
        or times[-1] < requested_end_s
    ):
        times = tuple(
            origin_sim_time_s + index * prediction.sample_step_s
            for index in range(len(points))
        )
    radii = tuple(float(value) for value in prediction.corridor_radius_m)
    if len(radii) != len(points):
        radii = (0.0,) * len(points)

    absolute_windows = tuple(
        (origin_sim_time_s + start, origin_sim_time_s + end)
        for start, end in WINDOW_OFFSETS_S
    )
    index_groups = tuple(
        _window_indices(times, start_s, end_s)
        for start_s, end_s in absolute_windows
    )
    sample_groups = tuple(
        _window_samples(points, radii, times, start_s, end_s)
        for start_s, end_s in absolute_windows
    )
    geometries = _bounded_slot_polygons(sample_groups, map_bounds_xy)
    mode = _generation_mode(accepted)
    geometry_revision = _next_geometry_revision(prior_regions, target_id, geometries)
    regions = _build_regions(
        target_id=target_id,
        prediction_id=prediction.prediction_id,
        execution_revision=execution_revision,
        geometry_revision=geometry_revision,
        origin_sim_time_s=origin_sim_time_s,
        geometries=geometries,
        index_groups=index_groups,
        evidence_ids=tuple(
            dict.fromkeys((prediction.prediction_id, *prediction.source_belief_history_ids))
        ),
    )
    return FourRegionBaseline(
        regions=regions,
        mode=mode,
        reason_codes=accepted.health.reason_codes,
    )


def _next_geometry_revision(
    prior_regions: Sequence[ExecutionRegion],
    target_id: str,
    geometries: Sequence[tuple[tuple[float, float], ...]],
) -> int:
    ordered = tuple(sorted(prior_regions, key=lambda region: region.slot_index))
    expected_ids = tuple(f"{target_id}:task:{index:02d}" for index in range(1, 5))
    if len(ordered) != 4 or tuple(region.region_id for region in ordered) != expected_ids:
        return 1
    previous_revision = max(region.geometry_revision for region in ordered)
    if tuple(region.geometry for region in ordered) == tuple(geometries):
        return previous_revision
    return previous_revision + 1


def _generation_mode(accepted: AcceptedPrediction) -> RegionGenerationMode:
    if accepted.health.regime == "boundary_recovery":
        return "boundary_recovery"
    if accepted.health.status == "valid" and accepted.health.regime == "imm":
        return "imm"
    return "degraded_prediction"


def _window_indices(
    times: Sequence[float], start_s: float, end_s: float
) -> tuple[int, ...]:
    selected = tuple(index for index, value in enumerate(times) if start_s <= value <= end_s)
    if selected:
        return selected
    midpoint = (start_s + end_s) / 2.0
    return (min(range(len(times)), key=lambda index: abs(times[index] - midpoint)),)


def _window_samples(
    points: Sequence[tuple[float, float]],
    radii: Sequence[float],
    times: Sequence[float],
    start_s: float,
    end_s: float,
) -> tuple[tuple[float, float, float], ...]:
    samples: list[tuple[float, float, float, float]] = []
    for index, time_s in enumerate(times):
        if start_s <= time_s <= end_s:
            samples.append((time_s, *points[index], max(0.0, radii[index])))
    for boundary_s in (start_s, end_s):
        if any(abs(sample[0] - boundary_s) <= 1e-9 for sample in samples):
            continue
        for index, (left_s, right_s) in enumerate(pairwise(times)):
            if left_s <= boundary_s <= right_s:
                ratio = (boundary_s - left_s) / (right_s - left_s)
                left = points[index]
                right = points[index + 1]
                radius = radii[index] + ratio * (radii[index + 1] - radii[index])
                samples.append(
                    (
                        boundary_s,
                        left[0] + ratio * (right[0] - left[0]),
                        left[1] + ratio * (right[1] - left[1]),
                        max(0.0, radius),
                    )
                )
                break
    if not samples:
        midpoint_s = (start_s + end_s) / 2.0
        nearest = min(range(len(times)), key=lambda index: abs(times[index] - midpoint_s))
        samples.append(
            (times[nearest], *points[nearest], max(0.0, radii[nearest]))
        )
    samples.sort(key=lambda sample: sample[0])
    return tuple((x, y, radius) for _, x, y, radius in samples)


def _bounded_slot_polygons(
    sample_groups: Sequence[Sequence[tuple[float, float, float]]],
    bounds: tuple[float, float, float, float],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    min_x, max_x, min_y, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    if width * height < 4.0 * _MIN_REGION_AREA_M2:
        raise ValueError("map bounds cannot retain four minimum-area execution regions")
    all_points = tuple((x, y) for group in sample_groups for x, y, _ in group)
    span = hypot(
        max(x for x, _ in all_points) - min(x for x, _ in all_points),
        max(y for _, y in all_points) - min(y for _, y in all_points),
    )
    if span < _MIN_REGION_WIDTH_M:
        fans = _bounded_fan_polygons(sample_groups, bounds)
        if _slot_chain_is_valid(fans, sample_groups, bounds):
            return fans
        return _bounded_pathological_polygons(sample_groups, bounds)

    ribbons = tuple(
        _bounded_ribbon(group, bounds, taper_start=slot == 0, taper_end=slot == 3)
        for slot, group in enumerate(sample_groups)
    )
    if _slot_chain_is_valid(ribbons, sample_groups, bounds):
        return ribbons
    return _bounded_pathological_polygons(sample_groups, bounds)


def _bounded_pathological_polygons(
    sample_groups: Sequence[Sequence[tuple[float, float, float]]],
    bounds: tuple[float, float, float, float],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    map_polygon = box(bounds[0], bounds[2], bounds[1], bounds[3])
    all_points = tuple((x, y) for group in sample_groups for x, y, _ in group)
    lines = tuple(
        _pathological_line(tuple((x, y) for x, y, _ in group), all_points, bounds)
        for group in sample_groups
    )
    handoffs = tuple(
        _pathological_line(
            (
                (sample_groups[index + 1][0][0], sample_groups[index + 1][0][1]),
                (sample_groups[index][-1][0], sample_groups[index][-1][1]),
            ),
            all_points,
            bounds,
        )
        for index in range(3)
    )

    for width in (
        _MIN_REGION_WIDTH_M / 2.0,
        _MIN_REGION_WIDTH_M,
        _MIN_REGION_WIDTH_M * 2.0,
    ):
        for side_signs in product((1.0, -1.0), repeat=4):
            candidates = tuple(
                line.buffer(
                    side * width,
                    single_sided=True,
                    join_style="bevel",
                ).intersection(map_polygon)
                for line, side in zip(lines, side_signs, strict=True)
            )
            if not all(
                _candidate_is_usable(candidate, group)
                for candidate, group in zip(candidates, sample_groups, strict=True)
            ):
                continue
            if any(
                _positive_area_overlap(candidates[left], candidates[right])
                for left, right in ((0, 2), (0, 3), (1, 3))
            ):
                continue

            missing_handoffs = tuple(
                index
                for index in range(3)
                if not _positive_area_overlap(candidates[index], candidates[index + 1])
            )
            for handoff_signs in product((1.0, -1.0), repeat=len(missing_handoffs)):
                patched = list(candidates)
                for index, side in zip(missing_handoffs, handoff_signs, strict=True):
                    patch = handoffs[index].buffer(
                        side * width,
                        single_sided=True,
                        join_style="bevel",
                    ).intersection(map_polygon)
                    patched[index] = unary_union((patched[index], patch)).intersection(map_polygon)
                    patched[index + 1] = unary_union(
                        (patched[index + 1], patch)
                    ).intersection(map_polygon)
                polygons = tuple(_polygon_coordinates(candidate) for candidate in patched)
                if _slot_chain_is_valid(polygons, sample_groups, bounds):
                    return polygons

    raise ValueError("map bounds cannot retain a legal four-region partition")


def _pathological_line(
    points: Sequence[tuple[float, float]],
    all_points: Sequence[tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> LineString:
    unique = tuple(dict.fromkeys(points))
    if len(unique) > 1:
        return LineString(unique)

    point = unique[0]
    farthest = max(
        all_points,
        key=lambda candidate: (
            hypot(candidate[0] - point[0], candidate[1] - point[1]),
            candidate,
        ),
    )
    delta_x = point[0] - farthest[0]
    delta_y = point[1] - farthest[1]
    if abs(delta_x) >= abs(delta_y):
        low, high = _fit_support_interval(point[0], bounds[0], bounds[1])
        return LineString(((low, point[1]), (high, point[1])))
    low, high = _fit_support_interval(point[1], bounds[2], bounds[3])
    return LineString(((point[0], low), (point[0], high)))


def _fit_support_interval(value: float, low: float, high: float) -> tuple[float, float]:
    width = _MIN_REGION_WIDTH_M
    if high - low + 1e-6 < width:
        raise ValueError("map bounds cannot retain minimum-width execution regions")
    start = min(max(value - width / 2.0, low), high - width)
    return start, start + width


def _candidate_is_usable(
    candidate: object,
    samples: Sequence[tuple[float, float, float]],
) -> bool:
    if not isinstance(candidate, Polygon) or not candidate.is_valid or candidate.interiors:
        return False
    if candidate.area + 1e-6 < _MIN_REGION_AREA_M2:
        return False
    min_x, min_y, max_x, max_y = candidate.bounds
    if min(max_x - min_x, max_y - min_y) + 1e-6 < _MIN_REGION_WIDTH_M:
        return False
    return all(candidate.distance(Point(x, y)) <= 1e-7 for x, y, _ in samples)


def _polygon_coordinates(candidate: object) -> tuple[tuple[float, float], ...]:
    if not isinstance(candidate, Polygon) or candidate.interiors:
        return ()
    coordinates = tuple(
        (float(x), float(y))
        for x, y in tuple(candidate.exterior.coords)[:-1]
    )
    return _clean_polygon(coordinates)


def _slot_chain_is_valid(
    polygons: Sequence[tuple[tuple[float, float], ...]],
    sample_groups: Sequence[Sequence[tuple[float, float, float]]],
    bounds: tuple[float, float, float, float],
) -> bool:
    if len(polygons) != 4 or any(
        len(polygon) < 3 or len(set(polygon)) < 3
        for polygon in polygons
    ):
        return False
    candidates = tuple(Polygon(polygon) for polygon in polygons)
    if not all(
        _minimum_geometry_is_retained(polygon)
        and _inside_map(polygon, bounds)
        and _candidate_is_usable(candidate, group)
        for polygon, candidate, group in zip(
            polygons, candidates, sample_groups, strict=True
        )
    ):
        return False
    return all(
        _positive_area_overlap(candidates[index], candidates[index + 1])
        for index in range(3)
    ) and all(
        not _positive_area_overlap(candidates[left], candidates[right])
        for left, right in ((0, 2), (0, 3), (1, 3))
    )


def _positive_area_overlap(left: BaseGeometry, right: BaseGeometry) -> bool:
    return left.intersection(right).area > 1e-6


def _bounded_ribbon(
    samples: Sequence[tuple[float, float, float]],
    bounds: tuple[float, float, float, float],
    *,
    taper_start: bool,
    taper_end: bool,
) -> tuple[tuple[float, float], ...]:
    if len(samples) < 2:
        return ()
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for index, (x, y, radius) in enumerate(samples):
        if index == 0:
            neighbor = samples[min(1, len(samples) - 1)]
            delta = (neighbor[0] - x, neighbor[1] - y)
        elif index == len(samples) - 1:
            neighbor = samples[index - 1]
            delta = (x - neighbor[0], y - neighbor[1])
        else:
            delta = (
                samples[index + 1][0] - samples[index - 1][0],
                samples[index + 1][1] - samples[index - 1][1],
            )
        length = hypot(*delta)
        normal = (-delta[1] / length, delta[0] / length) if length > 1e-9 else (0.0, 1.0)
        half_width = max(_MIN_REGION_WIDTH_M / 2.0, min(radius, _MIN_REGION_WIDTH_M))
        if (taper_start and index == 0) or (taper_end and index == len(samples) - 1):
            half_width = 0.0
        left.append((x + normal[0] * half_width, y + normal[1] * half_width))
        right.append((x - normal[0] * half_width, y - normal[1] * half_width))
    raw = tuple(left + list(reversed(right)))
    return _clean_polygon(_clip_polygon(raw, bounds))


def _bounded_fan_polygons(
    sample_groups: Sequence[Sequence[tuple[float, float, float]]],
    bounds: tuple[float, float, float, float],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    min_x, max_x, min_y, max_y = bounds
    all_points = tuple((x, y) for group in sample_groups for x, y, _ in group)
    anchor = (
        sum(x for x, _ in all_points) / len(all_points),
        sum(y for _, y in all_points) / len(all_points),
    )
    track_delta = (
        sample_groups[-1][-1][0] - sample_groups[0][0][0],
        sample_groups[-1][-1][1] - sample_groups[0][0][1],
    )
    x_sign = (
        1.0
        if track_delta[0] > 1e-9
        else -1.0
        if track_delta[0] < -1e-9
        else 1.0
        if max_x - anchor[0] >= anchor[0] - min_x
        else -1.0
    )
    y_sign = (
        1.0
        if track_delta[1] > 1e-9
        else -1.0
        if track_delta[1] < -1e-9
        else 1.0
        if max_y - anchor[1] >= anchor[1] - min_y
        else -1.0
    )
    roots = tuple(
        (
            sum(x for x, _, _ in group) / len(group),
            sum(y for _, y, _ in group) / len(group),
        )
        for group in sample_groups
    )
    x_capacity = min(
        max_x - root[0] if x_sign > 0.0 else root[0] - min_x for root in roots
    )
    y_capacity = min(
        max_y - root[1] if y_sign > 0.0 else root[1] - min_y for root in roots
    )
    required_radius = max(
        _MIN_REGION_WIDTH_M / sin(radians(15.0)),
        sqrt(2.0 * _MIN_REGION_AREA_M2 / sin(radians(15.0))),
    )
    radius = min(x_capacity, y_capacity)
    if radius + 1e-6 < required_radius:
        raise ValueError("map bounds cannot retain four minimum-area execution regions")
    radius = required_radius
    transformed_delta = (x_sign * track_delta[0], y_sign * track_delta[1])
    direction_degrees = (
        45.0
        if hypot(*transformed_delta) <= 1e-9
        else max(
            0.0,
            min(
                90.0,
                _angle_degrees(transformed_delta[0], transformed_delta[1]),
            ),
        )
    )
    if direction_degrees <= 45.0:
        sector_degrees = tuple(
            (direction_degrees + offset, direction_degrees + offset + 15.0)
            for offset in (30.0, 20.0, 10.0, 0.0)
        )
    else:
        sector_degrees = tuple(
            (direction_degrees - offset - 15.0, direction_degrees - offset)
            for offset in (30.0, 20.0, 10.0, 0.0)
        )
    polygons: list[tuple[tuple[float, float], ...]] = []
    for root, group, (start_degrees, end_degrees) in zip(
        roots, sample_groups, sector_degrees, strict=True
    ):
        candidates = [(x, y) for x, y, _ in group]
        for angle_degrees in (start_degrees, end_degrees):
            angle = radians(angle_degrees)
            candidates.append(
                (
                    root[0] + x_sign * radius * cos(angle),
                    root[1] + y_sign * radius * sin(angle),
                )
            )
        polygon = _convex_hull(candidates)
        if not _minimum_geometry_is_retained(polygon) or not _inside_map(polygon, bounds):
            raise ValueError("map bounds cannot retain four minimum-area execution regions")
        polygons.append(polygon)
    return tuple(polygons)


def _angle_degrees(x: float, y: float) -> float:
    return degrees(atan2(y, x))


def _convex_hull(
    points: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    ordered = sorted(set(points))
    if len(ordered) < 3:
        raise ValueError("execution region requires three distinct geometry points")

    def cross(
        origin: tuple[float, float],
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def _clip_polygon(
    polygon: Sequence[tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> tuple[tuple[float, float], ...]:
    result = list(polygon)
    for axis, threshold, keep_greater in (
        (0, bounds[0], True),
        (0, bounds[1], False),
        (1, bounds[2], True),
        (1, bounds[3], False),
    ):
        if not result:
            break
        clipped: list[tuple[float, float]] = []
        for start, end in zip(result, (*result[1:], result[0]), strict=True):
            start_inside = start[axis] >= threshold if keep_greater else start[axis] <= threshold
            end_inside = end[axis] >= threshold if keep_greater else end[axis] <= threshold
            if start_inside != end_inside:
                ratio = (threshold - start[axis]) / (end[axis] - start[axis])
                clipped.append(
                    (
                        start[0] + ratio * (end[0] - start[0]),
                        start[1] + ratio * (end[1] - start[1]),
                    )
                )
            if end_inside:
                clipped.append(end)
        result = clipped
        if not result:
            break
    return tuple(result)


def _clean_polygon(
    polygon: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    cleaned: list[tuple[float, float]] = []
    for point in polygon:
        if not cleaned or hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) > 1e-9:
            cleaned.append(point)
    if len(cleaned) > 1 and hypot(
        cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]
    ) <= 1e-9:
        cleaned.pop()
    while len(cleaned) > 3:
        for index, point in enumerate(cleaned):
            previous = cleaned[index - 1]
            following = cleaned[(index + 1) % len(cleaned)]
            cross = (point[0] - previous[0]) * (following[1] - point[1]) - (
                point[1] - previous[1]
            ) * (following[0] - point[0])
            if abs(cross) <= 1e-7:
                cleaned.pop(index)
                break
        else:
            break
    if cleaned:
        start = min(range(len(cleaned)), key=cleaned.__getitem__)
        cleaned = cleaned[start:] + cleaned[:start]
    return tuple(cleaned)


def _minimum_geometry_is_retained(
    polygon: Sequence[tuple[float, float]],
) -> bool:
    if len(polygon) < 3 or _polygon_area(polygon) + 1e-6 < _MIN_REGION_AREA_M2:
        return False
    x_span = max(x for x, _ in polygon) - min(x for x, _ in polygon)
    y_span = max(y for _, y in polygon) - min(y for _, y in polygon)
    return min(x_span, y_span) + 1e-6 >= _MIN_REGION_WIDTH_M


def _polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    return abs(
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(polygon, (*polygon[1:], polygon[0]), strict=True)
        )
    ) / 2.0


def _build_regions(
    *,
    target_id: str,
    prediction_id: str,
    execution_revision: int,
    geometry_revision: int,
    origin_sim_time_s: float,
    geometries: Sequence[tuple[tuple[float, float], ...]],
    index_groups: Sequence[tuple[int, ...]],
    evidence_ids: tuple[str, ...],
) -> tuple[ExecutionRegion, ExecutionRegion, ExecutionRegion, ExecutionRegion]:
    regions = tuple(
        ExecutionRegion(
            region_id=f"{target_id}:task:{slot + 1:02d}",
            target_id=target_id,
            slot_index=slot + 1,
            execution_revision=execution_revision,
            prediction_id=prediction_id,
            geometry=geometries[slot],
            centerline_indices=index_groups[slot],
            start_s=origin_sim_time_s + WINDOW_OFFSETS_S[slot][0],
            end_s=origin_sim_time_s + WINDOW_OFFSETS_S[slot][1],
            geometry_revision=geometry_revision,
            predecessor_region_id=(f"{target_id}:task:{slot:02d}" if slot else None),
            successor_region_id=(f"{target_id}:task:{slot + 2:02d}" if slot < 3 else None),
            handoff_start_s=(
                origin_sim_time_s + WINDOW_OFFSETS_S[slot + 1][0] if slot < 3 else None
            ),
            handoff_end_s=(
                origin_sim_time_s + WINDOW_OFFSETS_S[slot][1] if slot < 3 else None
            ),
            evidence_ids=evidence_ids,
        )
        for slot in range(4)
    )
    return (regions[0], regions[1], regions[2], regions[3])


def _reproject_previous(
    prior_regions: Sequence[ExecutionRegion],
    *,
    target_id: str,
    execution_revision: int,
    origin_sim_time_s: float,
    map_bounds_xy: tuple[float, float, float, float],
    reason_codes: tuple[str, ...],
) -> FourRegionBaseline:
    ordered = tuple(sorted(prior_regions, key=lambda region: region.slot_index))
    expected_ids = tuple(f"{target_id}:task:{index:02d}" for index in range(1, 5))
    if len(ordered) != 4 or tuple(region.region_id for region in ordered) != expected_ids:
        raise ValueError("no accepted geometry or prior four-region baseline")
    if any(not _inside_map(region.geometry, map_bounds_xy) for region in ordered):
        raise ValueError("prior four-region baseline lies outside current map bounds")
    geometries = tuple(region.geometry for region in ordered)
    index_groups = tuple(region.centerline_indices for region in ordered)
    regions = _build_regions(
        target_id=target_id,
        prediction_id=ordered[0].prediction_id,
        execution_revision=execution_revision,
        geometry_revision=max(region.geometry_revision for region in ordered),
        origin_sim_time_s=origin_sim_time_s,
        geometries=geometries,
        index_groups=index_groups,
        evidence_ids=tuple(dict.fromkeys(item for region in ordered for item in region.evidence_ids)),
    )
    return FourRegionBaseline(
        regions=regions,
        mode="reprojected_previous",
        reason_codes=(*reason_codes, "reprojected_previous_regions"),
    )


def _inside_map(
    geometry: Sequence[tuple[float, float]], bounds: tuple[float, float, float, float]
) -> bool:
    return all(bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3] for x, y in geometry)


def _validate_inputs(
    target_id: str,
    execution_revision: int,
    origin_sim_time_s: float,
    bounds: tuple[float, float, float, float],
) -> None:
    if not target_id:
        raise ValueError("target_id is required")
    if execution_revision < 1:
        raise ValueError("execution_revision must be positive")
    if origin_sim_time_s < 0.0:
        raise ValueError("origin_sim_time_s must be non-negative")
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("map bounds must have positive area")


__all__ = [
    "WINDOW_OFFSETS_S",
    "FourRegionBaseline",
    "RegionGenerationMode",
    "build_four_region_baseline",
]
