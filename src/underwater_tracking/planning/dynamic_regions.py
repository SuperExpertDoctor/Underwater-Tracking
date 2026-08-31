"""Four-slot execution regions generated from an IMM centerline.

The legacy planner still produces grid candidates for audit and LLM review.
This module owns the executable region chain: four stable slot IDs, absolute
time windows, centerline membership, local uncertainty width, and adjacent
handoff topology.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from math import hypot, isclose, sqrt
from typing import Any

import numpy as np
from pydantic import Field, model_validator

from underwater_tracking.domain.execution_models import ExecutionModel, ExecutionRegion


class RegionWindowPolicy(ExecutionModel):
    """Policy for the four rolling regional slots."""

    starts_s: tuple[float, float, float, float] = (0.0, 450.0, 900.0, 1_350.0)
    ends_s: tuple[float, float, float, float] = (540.0, 990.0, 1_440.0, 1_800.0)
    handoff_overlap_s: float = Field(default=90.0, ge=0)
    min_width_m: float = Field(default=250.0, gt=0)
    uncertainty_margin_m: float = Field(default=100.0, ge=0)
    min_area_m2: float | None = Field(default=None, gt=0)
    rolling_check_s: float = Field(default=450.0, gt=0)

    @model_validator(mode="after")
    def validate_windows(self) -> RegionWindowPolicy:
        if any(end <= start for start, end in zip(self.starts_s, self.ends_s)):
            raise ValueError("region windows must have positive duration")
        if any(right < left for left, right in zip(self.starts_s, self.starts_s[1:])):
            raise ValueError("region windows must be ordered")
        for index in range(3):
            overlap = self.ends_s[index] - self.starts_s[index + 1]
            if not isclose(overlap, self.handoff_overlap_s, abs_tol=1e-6):
                raise ValueError("adjacent region windows must preserve handoff overlap")
        return self


class DynamicRegionChain(ExecutionModel):
    """The four executable regions for one target and prediction revision."""

    target_id: str = Field(min_length=1)
    prediction_id: str = Field(min_length=1)
    execution_revision: int = Field(ge=1)
    geometry_revision: int = Field(ge=1)
    regions: tuple[ExecutionRegion, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_chain(self) -> DynamicRegionChain:
        expected_ids = tuple(f"{self.target_id}:task:{index:02d}" for index in range(1, 5))
        actual_ids = tuple(region.region_id for region in self.regions)
        if actual_ids != expected_ids:
            raise ValueError("dynamic region chain must contain four ordered stable slots")
        if any(region.target_id != self.target_id for region in self.regions):
            raise ValueError("dynamic region target IDs must agree")
        if any(region.prediction_id != self.prediction_id for region in self.regions):
            raise ValueError("dynamic region prediction IDs must agree")
        if any(region.execution_revision != self.execution_revision for region in self.regions):
            raise ValueError("dynamic region execution revisions must agree")
        if any(region.geometry_revision != self.geometry_revision for region in self.regions):
            raise ValueError("dynamic region geometry revisions must agree")
        return self


def build_dynamic_region_chain(
    prediction: Any,
    *,
    execution_revision: int,
    map_bounds_xy: tuple[float, float, float, float],
    policy: RegionWindowPolicy | None = None,
    previous_chain: DynamicRegionChain | None = None,
) -> DynamicRegionChain:
    """Build legacy four-region geometry for replay and migration callers.

    Live planning uses :func:`build_four_region_baseline`, whose accepted-
    prediction boundary records the generation mode and bounded fallback.
    """
    active_policy = policy or RegionWindowPolicy()
    if execution_revision < 1:
        raise ValueError("execution_revision must be positive")
    _validate_bounds(map_bounds_xy)
    target_id = str(getattr(prediction, "target_id", ""))
    prediction_id = str(getattr(prediction, "prediction_id", ""))
    if not target_id or not prediction_id:
        raise ValueError("prediction target_id and prediction_id are required")
    origin_s = float(
        getattr(
            prediction,
            "origin_sim_time_s",
            getattr(prediction, "sim_time_s", 0.0),
        )
    )
    times, points, covariances, radii = _forecast_arrays(prediction, origin_s)
    geometry_revision = _next_geometry_revision(previous_chain, points, active_policy, map_bounds_xy)
    regions: list[ExecutionRegion] = []
    for slot_index, (relative_start, relative_end) in enumerate(
        zip(active_policy.starts_s, active_policy.ends_s), start=1
    ):
        start_s = origin_s + relative_start
        end_s = origin_s + relative_end
        indices = _window_indices(times, start_s, end_s)
        geometry = _region_geometry(
            points,
            covariances,
            radii,
            indices,
            map_bounds_xy,
            active_policy,
        )
        region_id = f"{target_id}:task:{slot_index:02d}"
        predecessor = f"{target_id}:task:{slot_index - 1:02d}" if slot_index > 1 else None
        successor = f"{target_id}:task:{slot_index + 1:02d}" if slot_index < 4 else None
        handoff_start = (
            origin_s + active_policy.starts_s[slot_index]
            if slot_index < 4
            else None
        )
        handoff_end = end_s if slot_index < 4 else None
        regions.append(
            ExecutionRegion(
                region_id=region_id,
                target_id=target_id,
                slot_index=slot_index,
                execution_revision=execution_revision,
                prediction_id=prediction_id,
                geometry=geometry,
                centerline_indices=indices,
                start_s=start_s,
                end_s=end_s,
                geometry_revision=geometry_revision,
                predecessor_region_id=predecessor,
                successor_region_id=successor,
                handoff_start_s=handoff_start,
                handoff_end_s=handoff_end,
                evidence_ids=(prediction_id,),
            )
        )
    return DynamicRegionChain(
        target_id=target_id,
        prediction_id=prediction_id,
        execution_revision=execution_revision,
        geometry_revision=geometry_revision,
        regions=tuple(regions),
    )


def normalize_region_chain(
    chain: DynamicRegionChain | Mapping[str, Any] | Sequence[Any],
    *,
    prediction: Any | None = None,
    execution_revision: int | None = None,
    map_bounds_xy: tuple[float, float, float, float] | None = None,
    policy: RegionWindowPolicy | None = None,
) -> DynamicRegionChain:
    """Normalize provider output to the executable four-slot contract.

    When the current prediction is available, deterministic geometry is
    regenerated from it so extra slots, reversed order, overlap violations,
    or missing centerline coverage cannot enter execution.
    """
    if prediction is not None:
        if execution_revision is None or map_bounds_xy is None:
            raise ValueError("prediction normalization requires revision and map bounds")
        return build_dynamic_region_chain(
            prediction,
            execution_revision=execution_revision,
            map_bounds_xy=map_bounds_xy,
            policy=policy,
        )
    if isinstance(chain, DynamicRegionChain):
        regions = tuple(sorted(chain.regions, key=lambda region: region.slot_index))
        return chain.model_copy(update={"regions": regions})
    if isinstance(chain, Mapping):
        return normalize_region_chain(
            DynamicRegionChain.model_validate(chain),
            policy=policy,
        )
    raise TypeError("region chain must be a DynamicRegionChain or mapping")


def _forecast_arrays(
    prediction: Any,
    origin_s: float,
) -> tuple[
    tuple[float, ...],
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float, float, float], ...],
    tuple[float, ...],
]:
    points_raw = getattr(prediction, "centerline_xy", None)
    if points_raw is None:
        points_raw = getattr(prediction, "points_xy", ())
    points = tuple((float(point[0]), float(point[1])) for point in points_raw)
    if not points:
        raise ValueError("prediction centerline is required")
    times_raw = tuple(float(value) for value in getattr(prediction, "times_s", ()))
    if len(times_raw) != len(points):
        step = float(getattr(prediction, "sample_step_s", 30.0))
        times_raw = tuple(origin_s + index * step for index in range(len(points)))
    if any(right <= left for left, right in pairwise(times_raw)):
        raise ValueError("prediction times must be strictly increasing")
    raw_covariances = tuple(getattr(prediction, "covariance_xy", ()))
    covariances = (
        tuple(_covariance2(value) for value in raw_covariances)
        if len(raw_covariances) == len(points)
        else tuple((0.0, 0.0, 0.0, 0.0) for _ in points)
    )
    raw_radii = tuple(float(value) for value in getattr(prediction, "corridor_radius_m", ()))
    radii = raw_radii if len(raw_radii) == len(points) else tuple(0.0 for _ in points)
    return times_raw, points, covariances, radii


def _covariance2(value: Any) -> tuple[float, float, float, float]:
    if len(value) == 4 and not isinstance(value[0], (tuple, list)):
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    array = np.asarray(value, dtype=float)
    if array.shape != (2, 2):
        return (0.0, 0.0, 0.0, 0.0)
    return (float(array[0, 0]), float(array[0, 1]), float(array[1, 0]), float(array[1, 1]))


def _window_indices(times: Sequence[float], start_s: float, end_s: float) -> tuple[int, ...]:
    selected = tuple(index for index, time_s in enumerate(times) if start_s <= time_s <= end_s)
    if selected:
        return selected
    nearest = min(range(len(times)), key=lambda index: abs(times[index] - (start_s + end_s) / 2.0))
    return (nearest,)


def _region_geometry(
    points: Sequence[tuple[float, float]],
    covariances: Sequence[tuple[float, float, float, float]],
    radii: Sequence[float],
    indices: Sequence[int],
    bounds: tuple[float, float, float, float],
    policy: RegionWindowPolicy,
) -> tuple[tuple[float, float], ...]:
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for index in indices:
        tangent = _tangent(points, index)
        normal = (-tangent[1], tangent[0])
        covariance = np.asarray(
            [[covariances[index][0], covariances[index][1]],
             [covariances[index][2], covariances[index][3]]],
            dtype=float,
        )
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues = np.linalg.eigvalsh(covariance)
        uncertainty = sqrt(max(float(np.max(eigenvalues)), 0.0)) if eigenvalues.size else 0.0
        half_width = max(
            policy.min_width_m / 2.0,
            float(radii[index]) + policy.uncertainty_margin_m + uncertainty,
        )
        left.append((points[index][0] + normal[0] * half_width, points[index][1] + normal[1] * half_width))
        right.append((points[index][0] - normal[0] * half_width, points[index][1] - normal[1] * half_width))
    if len(left) == 1:
        tangent = _tangent(points, indices[0])
        required_half_length = (
            policy.min_area_m2 / policy.min_width_m
            if policy.min_area_m2 is not None
            else policy.min_width_m / 2.0
        )
        half_length = max(policy.min_width_m / 2.0, required_half_length)
        half_width = policy.min_width_m / 2.0
        center = points[indices[0]]
        normal = (-tangent[1], tangent[0])
        geometry = (
            _add(center, tangent, half_length, normal, half_width),
            _add(center, tangent, -half_length, normal, half_width),
            _add(center, tangent, -half_length, normal, -half_width),
            _add(center, tangent, half_length, normal, -half_width),
        )
    else:
        geometry = tuple(left + list(reversed(right)))
    minimum_area = max(policy.min_width_m**2, policy.min_area_m2 or 0.0)
    clipped = _clip_polygon(geometry, bounds)
    if _polygon_area(clipped) < minimum_area:
        clipped = _boundary_fallback(points[indices[-1]], _tangent(points, indices[-1]), bounds, policy)
    if _polygon_area(clipped) + 1e-6 < minimum_area:
        raise ValueError("map bounds cannot retain minimum dynamic region area")
    return tuple(clipped)


def _boundary_fallback(
    point: tuple[float, float],
    tangent: tuple[float, float],
    bounds: tuple[float, float, float, float],
    policy: RegionWindowPolicy,
) -> tuple[tuple[float, float], ...]:
    center = (
        min(max(point[0], bounds[0]), bounds[1]),
        min(max(point[1], bounds[2]), bounds[3]),
    )
    minimum_area = max(policy.min_width_m**2, policy.min_area_m2 or 0.0)
    length = max(policy.min_width_m, 2.0 * minimum_area / policy.min_width_m)
    half_length = length / 2.0
    half_width = policy.min_width_m / 2.0
    normal = (-tangent[1], tangent[0])
    raw = (
        _add(center, tangent, half_length, normal, half_width),
        _add(center, tangent, -half_length, normal, half_width),
        _add(center, tangent, -half_length, normal, -half_width),
        _add(center, tangent, half_length, normal, -half_width),
    )
    return tuple(_clip_polygon(raw, bounds))


def _add(
    center: tuple[float, float],
    tangent: tuple[float, float],
    along: float,
    normal: tuple[float, float],
    across: float,
) -> tuple[float, float]:
    return (
        center[0] + tangent[0] * along + normal[0] * across,
        center[1] + tangent[1] * along + normal[1] * across,
    )


def _tangent(points: Sequence[tuple[float, float]], index: int) -> tuple[float, float]:
    if len(points) == 1:
        return (1.0, 0.0)
    if index == 0:
        delta = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    elif index == len(points) - 1:
        delta = (points[-1][0] - points[-2][0], points[-1][1] - points[-2][1])
    else:
        delta = (points[index + 1][0] - points[index - 1][0], points[index + 1][1] - points[index - 1][1])
    length = hypot(*delta)
    return (delta[0] / length, delta[1] / length) if length > 1e-12 else (1.0, 0.0)


def _clip_polygon(
    polygon: Sequence[tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    min_x, max_x, min_y, max_y = bounds
    result = list(polygon)
    for axis, threshold, keep_greater in (
        (0, min_x, True),
        (0, max_x, False),
        (1, min_y, True),
        (1, max_y, False),
    ):
        if not result:
            break
        clipped: list[tuple[float, float]] = []
        for start, end in zip(result, (*result[1:], result[0])):
            start_inside = start[axis] >= threshold if keep_greater else start[axis] <= threshold
            end_inside = end[axis] >= threshold if keep_greater else end[axis] <= threshold
            if start_inside != end_inside:
                denominator = end[axis] - start[axis]
                ratio = 0.0 if abs(denominator) <= 1e-12 else (threshold - start[axis]) / denominator
                intersection = (
                    start[0] + ratio * (end[0] - start[0]),
                    start[1] + ratio * (end[1] - start[1]),
                )
                clipped.append(intersection)
            if end_inside:
                clipped.append(end)
        result = clipped
    return result


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(left[0] * right[1] - right[0] * left[1] for left, right in zip(points, (*points[1:], points[0])))
    ) / 2.0


def _next_geometry_revision(
    previous: DynamicRegionChain | None,
    points: Sequence[tuple[float, float]],
    policy: RegionWindowPolicy,
    bounds: tuple[float, float, float, float],
) -> int:
    del points, policy, bounds
    return 1 if previous is None else previous.geometry_revision + 1


def _validate_bounds(bounds: tuple[float, float, float, float]) -> None:
    if len(bounds) != 4 or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("map bounds must have positive area")


__all__ = [
    "DynamicRegionChain",
    "RegionWindowPolicy",
    "build_dynamic_region_chain",
    "normalize_region_chain",
]
