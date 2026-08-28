"""Deterministic four-region geometry for live execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

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
    if len(times) != len(points) or any(right <= left for left, right in pairwise(times)):
        times = tuple(
            origin_sim_time_s + index * prediction.sample_step_s
            for index in range(len(points))
        )
    radii = tuple(float(value) for value in prediction.corridor_radius_m)
    if len(radii) != len(points):
        radii = (0.0,) * len(points)

    index_groups = tuple(
        _window_indices(times, origin_sim_time_s + start, origin_sim_time_s + end)
        for start, end in WINDOW_OFFSETS_S
    )
    geometries = _bounded_slot_rectangles(points, radii, index_groups, map_bounds_xy)
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


def _bounded_slot_rectangles(
    points: Sequence[tuple[float, float]],
    radii: Sequence[float],
    index_groups: Sequence[Sequence[int]],
    bounds: tuple[float, float, float, float],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    min_x, max_x, min_y, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    point_x_span = max(x for x, _ in points) - min(x for x, _ in points)
    point_y_span = max(y for _, y in points) - min(y for _, y in points)
    split_x = point_x_span >= point_y_span if max(point_x_span, point_y_span) > 1e-9 else width >= height
    axis_min, axis_max = (min_x, max_x) if split_x else (min_y, max_y)
    cross_min, cross_max = (min_y, max_y) if split_x else (min_x, max_x)
    axis_extent = axis_max - axis_min
    cross_extent = cross_max - cross_min
    if axis_extent * cross_extent < 4.0 * _MIN_REGION_AREA_M2:
        raise ValueError("map bounds cannot retain four minimum-area execution regions")

    projected = tuple((x if split_x else y) for x, y in points)
    cross_values = tuple((y if split_x else x) for x, y in points)
    bounded_radius = min(max(radii, default=0.0), axis_extent / 8.0, cross_extent / 2.0)
    direction = 1.0 if projected[-1] >= projected[0] else -1.0
    oriented = tuple(direction * value for value in projected)
    oriented_min, oriented_max = (
        (axis_min, axis_max) if direction > 0 else (-axis_max, -axis_min)
    )
    slot_centers = tuple(
        sum(oriented[index] for index in indices) / len(indices)
        for indices in index_groups
    )
    moving_in_time_order = (
        max(projected) - min(projected) > 1e-9
        and all(right >= left for left, right in pairwise(slot_centers))
    )
    if moving_in_time_order:
        oriented_intervals = [
            [
                max(oriented_min, min(oriented[index] for index in indices) - bounded_radius),
                min(oriented_max, max(oriented[index] for index in indices) + bounded_radius),
            ]
            for indices in index_groups
        ]
        # Non-neighboring windows have no handoff relationship. Split their
        # uncertainty expansion at a deterministic midpoint while retaining
        # every centerline sample in its own chronological slot.
        for left_slot in range(2):
            right_slot = left_slot + 2
            separator = (
                max(oriented[index] for index in index_groups[left_slot])
                + min(oriented[index] for index in index_groups[right_slot])
            ) / 2.0
            oriented_intervals[left_slot][1] = min(
                oriented_intervals[left_slot][1], separator
            )
            oriented_intervals[right_slot][0] = max(
                oriented_intervals[right_slot][0], separator
            )
        axis_intervals = tuple(
            (low, high) if direction > 0 else (-high, -low)
            for low, high in oriented_intervals
        )
    else:
        raw_low = min(projected) - bounded_radius
        raw_high = max(projected) + bounded_radius
        minimum_span = min(axis_extent, 4.0 * _MIN_REGION_WIDTH_M)
        span = min(max(raw_high - raw_low, minimum_span), axis_extent)
        center = min(
            max((raw_low + raw_high) / 2.0, axis_min + span / 2.0),
            axis_max - span / 2.0,
        )
        envelope_low = center - span / 2.0
        slot_width = span / 4.0
        overlap = min(slot_width * 0.18, _MIN_REGION_WIDTH_M / 2.0)
        axis_intervals = tuple(
            (
                max(
                    axis_min,
                    envelope_low + slot * slot_width - (overlap if slot else 0.0),
                ),
                min(
                    axis_max,
                    envelope_low
                    + (slot + 1) * slot_width
                    + (overlap if slot < 3 else 0.0),
                ),
            )
            for slot in range(4)
        )

    rectangles: list[tuple[tuple[float, float], ...]] = []
    for slot, indices in enumerate(index_groups):
        axis_low, axis_high = axis_intervals[slot]
        required_cross = max(
            _MIN_REGION_WIDTH_M,
            _MIN_REGION_AREA_M2 / max(axis_high - axis_low, 1e-9),
        )
        required_cross = min(required_cross, cross_extent)
        segment_cross = sum(cross_values[index] for index in indices) / len(indices)
        cross_half = max(required_cross / 2.0, min(bounded_radius, cross_extent / 2.0))
        cross_width = min(cross_extent, 2.0 * cross_half)
        cross_center = min(
            max(segment_cross, cross_min + cross_width / 2.0),
            cross_max - cross_width / 2.0,
        )
        low_cross = cross_center - cross_width / 2.0
        high_cross = cross_center + cross_width / 2.0
        if split_x:
            geometry = (
                (axis_low, low_cross),
                (axis_high, low_cross),
                (axis_high, high_cross),
                (axis_low, high_cross),
            )
        else:
            geometry = (
                (low_cross, axis_low),
                (high_cross, axis_low),
                (high_cross, axis_high),
                (low_cross, axis_high),
            )
        rectangles.append(geometry)
    return tuple(rectangles)


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
        geometry_revision=max(region.geometry_revision for region in ordered) + 1,
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
