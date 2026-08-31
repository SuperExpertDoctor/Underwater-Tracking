from __future__ import annotations

from collections.abc import Sequence
from math import ceil, floor, hypot, sqrt

from underwater_tracking.domain.agent_models import IntentHypothesis, PredictedTrackRef
from underwater_tracking.domain.models import TargetBelief
from underwater_tracking.domain.regional_models import GridSpec
from underwater_tracking.domain.mission_models import PredictionGrid, PredictionGridCell

Bounds = tuple[float, float, float, float]
_DEFAULT_BOUNDS: Bounds = (0.0, 2_500.0, 0.0, 1_000.0)


def build_prediction_grid(
    belief: TargetBelief,
    prediction: PredictedTrackRef,
    intent: IntentHypothesis,
    revision: int,
    grid_spec: GridSpec,
    map_bounds_xy: Bounds = _DEFAULT_BOUNDS,
) -> PredictionGrid:
    """Project estimator-visible belief and prediction evidence into stable cells."""
    if revision < 1:
        raise ValueError("prediction grid revision must be positive")
    if belief.target_id != prediction.target_id:
        raise ValueError("belief and prediction target IDs must match")
    if not prediction.points_xy:
        raise ValueError("prediction points are required for a prediction grid")
    min_map_x, max_map_x, min_map_y, max_map_y = map_bounds_xy
    if min_map_x >= max_map_x or min_map_y >= max_map_y:
        raise ValueError("prediction grid map bounds must have positive area")

    covariance_radius = _covariance_radius(belief.covariance, intent.confidence)
    corridor_radius = max(prediction.corridor_radius_m, default=0.0)
    radius = max(corridor_radius, covariance_radius)
    points = prediction.points_xy
    envelope_area = max(
        grid_spec.min_cell_size_m**2,
        (max(point[0] for point in points) - min(point[0] for point in points) + 2 * radius)
        * (max(point[1] for point in points) - min(point[1] for point in points) + 2 * radius),
    )
    cell_size = _rounded_cell_size(envelope_area, grid_spec)

    times = prediction.times_s or tuple(
        prediction.sim_time_s + index * prediction.sample_step_s
        for index in range(len(points))
    )
    if len(times) != len(points):
        raise ValueError("prediction times and points must have equal lengths")
    radii = prediction.corridor_radius_m or tuple(0.0 for _ in points)
    keys: set[tuple[int, int]] = set()
    centerline_keys: list[tuple[int, int]] = []
    for index, point in enumerate(points):
        center_key = _grid_key(point, grid_spec, cell_size)
        if _inside_map(center_key, grid_spec, cell_size, map_bounds_xy):
            keys.add(center_key)
            centerline_keys.append(center_key)
        for key in _keys_in_radius(point, max(radii[index], radius), grid_spec, cell_size):
            if _inside_map(key, grid_spec, cell_size, map_bounds_xy):
                keys.add(key)
    if not keys:
        raise ValueError("prediction corridor does not intersect map bounds")

    covariance_summary = _covariance_summary(belief.covariance)
    cells = tuple(
        _make_cell(
            key,
            target_id=prediction.target_id,
            revision=revision,
            grid_spec=grid_spec,
            cell_size=cell_size,
            points=points,
            times=times,
            keys=keys,
            covariance_summary=covariance_summary,
            intent=intent,
        )
        for key in sorted(
            keys,
            key=lambda item: f"{prediction.target_id}:r{revision}:cell:{item[0]}:{item[1]}",
        )
    )
    centerline_ids = tuple(
        dict.fromkeys(
            f"{prediction.target_id}:r{revision}:cell:{key[0]}:{key[1]}"
            for key in centerline_keys
            if key in keys
        )
    )
    return PredictionGrid(
        target_id=prediction.target_id,
        revision=revision,
        origin=grid_spec.origin_xy,
        cell_size_m=cell_size,
        cells=cells,
        centerline_region_ids=centerline_ids,
    )


def _covariance_radius(covariance: Sequence[Sequence[float]], confidence: float) -> float:
    if len(covariance) < 2 or len(covariance[0]) < 2 or len(covariance[1]) < 2:
        return 0.0
    variance = max(0.0, float(covariance[0][0]), float(covariance[1][1]))
    return sqrt(variance) * (1.0 + max(0.0, 1.0 - confidence))


def _covariance_summary(covariance: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    xx = float(covariance[0][0]) if len(covariance) > 0 and covariance[0] else 0.0
    yy = float(covariance[1][1]) if len(covariance) > 1 and len(covariance[1]) > 1 else 0.0
    xy = float(covariance[0][1]) if len(covariance) > 0 and len(covariance[0]) > 1 else 0.0
    return xx, yy, xy


def _rounded_cell_size(envelope_area: float, grid_spec: GridSpec) -> float:
    raw = sqrt(envelope_area / grid_spec.target_grid_cells)
    bounded = min(max(raw, grid_spec.min_cell_size_m), grid_spec.max_cell_size_m)
    rounded = floor(bounded / grid_spec.cell_size_rounding_m + 0.5) * grid_spec.cell_size_rounding_m
    return min(max(rounded, grid_spec.min_cell_size_m), grid_spec.max_cell_size_m)


def _grid_key(
    point: tuple[float, float], grid_spec: GridSpec, cell_size: float
) -> tuple[int, int]:
    return (
        floor((point[0] - grid_spec.origin_xy[0]) / cell_size),
        floor((point[1] - grid_spec.origin_xy[1]) / cell_size),
    )


def _cell_bounds(
    key: tuple[int, int], grid_spec: GridSpec, cell_size: float
) -> tuple[float, float, float, float]:
    min_x = grid_spec.origin_xy[0] + key[0] * cell_size
    min_y = grid_spec.origin_xy[1] + key[1] * cell_size
    return min_x, min_x + cell_size, min_y, min_y + cell_size


def _inside_map(
    key: tuple[int, int], grid_spec: GridSpec, cell_size: float, bounds: Bounds
) -> bool:
    min_x, max_x, min_y, max_y = _cell_bounds(key, grid_spec, cell_size)
    return min_x >= bounds[0] and max_x <= bounds[1] and min_y >= bounds[2] and max_y <= bounds[3]


def _keys_in_radius(
    point: tuple[float, float], radius: float, grid_spec: GridSpec, cell_size: float
) -> set[tuple[int, int]]:
    min_x = floor((point[0] - radius - grid_spec.origin_xy[0]) / cell_size)
    max_x = ceil((point[0] + radius - grid_spec.origin_xy[0]) / cell_size)
    min_y = floor((point[1] - radius - grid_spec.origin_xy[1]) / cell_size)
    max_y = ceil((point[1] + radius - grid_spec.origin_xy[1]) / cell_size)
    result: set[tuple[int, int]] = set()
    for grid_x in range(min_x, max_x + 1):
        for grid_y in range(min_y, max_y + 1):
            bounds = _cell_bounds((grid_x, grid_y), grid_spec, cell_size)
            nearest = (
                min(max(point[0], bounds[0]), bounds[1]),
                min(max(point[1], bounds[2]), bounds[3]),
            )
            if hypot(point[0] - nearest[0], point[1] - nearest[1]) <= radius:
                result.add((grid_x, grid_y))
    return result


def _make_cell(
    key: tuple[int, int],
    *,
    target_id: str,
    revision: int,
    grid_spec: GridSpec,
    cell_size: float,
    points: tuple[tuple[float, float], ...],
    times: tuple[float, ...],
    keys: set[tuple[int, int]],
    covariance_summary: tuple[float, float, float],
    intent: IntentHypothesis,
) -> PredictionGridCell:
    min_x, max_x, min_y, max_y = _cell_bounds(key, grid_spec, cell_size)
    nearest_index = min(
        range(len(points)),
        key=lambda index: (
            (points[index][0] - (min_x + max_x) / 2.0) ** 2
            + (points[index][1] - (min_y + max_y) / 2.0) ** 2,
            index,
        ),
    )
    probability = min(1.0, max(0.0, len(keys) / max(1, len(points) * 4)))
    return PredictionGridCell(
        target_id=target_id,
        revision=revision,
        grid_x=key[0],
        grid_y=key[1],
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        cell_size_m=cell_size,
        probability=probability,
        first_entry_s=max(0, round(times[nearest_index] - 0.5 * (times[1] - times[0] if len(times) > 1 else 1.0))),
        last_exit_s=max(0, round(times[nearest_index] + (times[1] - times[0] if len(times) > 1 else 1.0))),
        imm_model_probabilities={"evidence": probability},
        covariance_summary=covariance_summary,
        intent_label=intent.label,
        intent_confidence=intent.confidence,
    )
