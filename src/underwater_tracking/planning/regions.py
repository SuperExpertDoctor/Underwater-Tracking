from __future__ import annotations

from collections import defaultdict
from math import ceil, floor, hypot, isfinite, sqrt

from underwater_tracking.domain.agent_models import IntentHypothesis, PredictedTrackRef
from underwater_tracking.domain.regional_models import (
    GridSpec,
    RegionCell,
    RegionTask,
    SonarPolicy,
    TargetRegionPlan,
    TimeWindow,
)


def compute_cell_size(envelope_area_m2: float, grid_spec: GridSpec) -> float:
    """Return the deterministic engineering-rounded side length for an envelope."""
    if not isfinite(envelope_area_m2) or envelope_area_m2 <= 0.0:
        raise ValueError("envelope_area_m2 must be finite and positive")
    initial = sqrt(envelope_area_m2 / grid_spec.target_grid_cells)
    bounded = min(max(initial, grid_spec.min_cell_size_m), grid_spec.max_cell_size_m)
    rounded = floor(bounded / grid_spec.cell_size_rounding_m + 0.5) * grid_spec.cell_size_rounding_m
    rounded = min(max(rounded, grid_spec.min_cell_size_m), grid_spec.max_cell_size_m)
    return round(float(rounded), 9)


def rectangles_overlap(left: RegionCell, right: RegionCell) -> bool:
    """Return true only when two axis-aligned cells share positive area."""
    return (
        min(left.max_x, right.max_x) > max(left.min_x, right.min_x)
        and min(left.max_y, right.max_y) > max(left.min_y, right.min_y)
    )


def generate_target_region_plan(
    prediction: PredictedTrackRef,
    intent: IntentHypothesis,
    map_bounds_xy: tuple[float, float, float, float],
    grid_spec: GridSpec,
) -> TargetRegionPlan:
    """Rasterize an estimated prediction corridor into ordered square tasks."""
    min_map_x, max_map_x, min_map_y, max_map_y = map_bounds_xy
    if not min_map_x < max_map_x or not min_map_y < max_map_y:
        raise ValueError("map bounds must have positive area")
    points = tuple(prediction.points_xy)
    if not points:
        raise ValueError("prediction points are required for regionalization")
    times = tuple(prediction.times_s)
    if len(times) != len(points):
        times = tuple(
            prediction.sim_time_s + index * prediction.sample_step_s
            for index in range(len(points))
        )
    radii = tuple(prediction.corridor_radius_m)
    if len(radii) != len(points):
        radii = tuple(0.0 for _ in points)
    if any(radius < 0 for radius in radii):
        raise ValueError("prediction corridor radius must be non-negative")

    corridor_radius = max(radii, default=0.0)
    envelope_min_x = max(min_map_x, min(point[0] for point in points) - corridor_radius)
    envelope_max_x = min(max_map_x, max(point[0] for point in points) + corridor_radius)
    envelope_min_y = max(min_map_y, min(point[1] for point in points) - corridor_radius)
    envelope_max_y = min(max_map_y, max(point[1] for point in points) + corridor_radius)
    envelope_area = max(
        (envelope_max_x - envelope_min_x) * (envelope_max_y - envelope_min_y),
        grid_spec.min_cell_size_m**2,
    )
    cell_size = compute_cell_size(envelope_area, grid_spec)
    keys_by_sample: dict[tuple[int, int], set[int]] = defaultdict(set)

    for index, point in enumerate(points):
        for offset in range(
            -grid_spec.lateral_half_width_cells,
            grid_spec.lateral_half_width_cells + 1,
        ):
            tangent_x, tangent_y = _tangent(points, index)
            normal_x, normal_y = -tangent_y, tangent_x
            candidate = (
                point[0] + normal_x * offset * cell_size,
                point[1] + normal_y * offset * cell_size,
            )
            key = _grid_key(candidate, grid_spec, cell_size)
            if _cell_is_inside_map(key, grid_spec, cell_size, map_bounds_xy):
                keys_by_sample[key].add(index)
        uncertainty_radius = radii[index] + grid_spec.max_uncertainty_margin_cells * cell_size
        for key in _corridor_keys(point, uncertainty_radius, grid_spec, cell_size):
            if _cell_is_inside_map(key, grid_spec, cell_size, map_bounds_xy):
                keys_by_sample[key].add(index)

    if not keys_by_sample:
        raise ValueError("prediction corridor does not intersect map bounds")
    ordered_keys = sorted(
        keys_by_sample,
        key=lambda key: (_first_time(keys_by_sample[key], times), key[0], key[1]),
    )
    base_cells: list[RegionCell] = []
    evidence_ids = set(prediction.source_belief_history_ids) | set(intent.evidence_ids)
    evidence_ids.add(prediction.prediction_id)
    if prediction.fallback_used:
        evidence_ids.add("prediction:fallback")
    for key in ordered_keys:
        windows = _visit_windows(keys_by_sample[key], times, prediction.sample_step_s)
        min_x, max_x, min_y, max_y = _cell_bounds(key, grid_spec, cell_size)
        base_cells.append(
            RegionCell(
                region_id=f"{prediction.target_id}:cell:{key[0]}:{key[1]}",
                target_id=prediction.target_id,
                grid_x=key[0],
                grid_y=key[1],
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                center_xy=((min_x + max_x) / 2.0, (min_y + max_y) / 2.0),
                cell_size_m=cell_size,
                first_entry_s=windows[0].start_s,
                last_exit_s=windows[-1].end_s,
                visit_windows=windows,
                occupancy_likelihood=min(1.0, len(keys_by_sample[key]) / len(points)),
                intent_labels=(intent.label,),
                evidence_ids=tuple(sorted(evidence_ids)),
            )
        )
    cells: list[RegionCell] = []
    for index, cell in enumerate(base_cells):
        cells.append(
            cell.model_copy(
                update={
                    "predecessor_region_ids": (
                        () if index == 0 else (base_cells[index - 1].region_id,)
                    ),
                    "successor_region_ids": (
                        ()
                        if index == len(base_cells) - 1
                        else (base_cells[index + 1].region_id,)
                    ),
                }
            )
        )
    uuv_count = 1 if grid_spec.require_uuv_per_region else 0
    usv_count = 1 if grid_spec.require_usv_per_region else 0
    tasks = tuple(
        RegionTask(
            region_id=cell.region_id,
            target_id=cell.target_id,
            active_window=TimeWindow(start_s=cell.first_entry_s, end_s=cell.last_exit_s),
            required_quality=0.0,
            required_uuv_count=uuv_count,
            required_usv_count=usv_count,
            uuv_roles=("passive_tracker",) if uuv_count else (),
            usv_role="surface_relay" if usv_count else None,
            sonar_policy=SonarPolicy(passive_required=True, active_allowed=False),
            communication=cell_communication(grid_spec),
            predecessor_region_id=(
                cell.predecessor_region_ids[0] if cell.predecessor_region_ids else None
            ),
            successor_region_id=(
                cell.successor_region_ids[0] if cell.successor_region_ids else None
            ),
            evidence_ids=tuple(sorted(evidence_ids)),
        )
        for cell in cells
    )
    return TargetRegionPlan(
        target_id=prediction.target_id,
        grid_spec=grid_spec,
        cell_size_m=cell_size,
        cells=tuple(cells),
        tasks=tasks,
        prediction_id=prediction.prediction_id,
        intent_label=intent.label,
        intent_confidence=intent.confidence,
        evidence_ids=tuple(sorted(evidence_ids)),
        fallback_used=prediction.fallback_used,
        fallback_reason=prediction.fallback_reason,
    )


def cell_communication(grid_spec: GridSpec):
    from underwater_tracking.domain.regional_models import CommunicationRequirement

    return CommunicationRequirement(relay_overlap_policy=grid_spec.relay_overlap_policy)


def _grid_key(point: tuple[float, float], grid_spec: GridSpec, cell_size: float) -> tuple[int, int]:
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


def _cell_is_inside_map(
    key: tuple[int, int], grid_spec: GridSpec, cell_size: float,
    map_bounds_xy: tuple[float, float, float, float],
) -> bool:
    min_x, max_x, min_y, max_y = _cell_bounds(key, grid_spec, cell_size)
    return (
        min_x >= map_bounds_xy[0]
        and max_x <= map_bounds_xy[1]
        and min_y >= map_bounds_xy[2]
        and max_y <= map_bounds_xy[3]
    )


def _tangent(points: tuple[tuple[float, float], ...], index: int) -> tuple[float, float]:
    if len(points) == 1:
        return 1.0, 0.0
    if index == 0:
        delta = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    elif index == len(points) - 1:
        delta = (points[-1][0] - points[-2][0], points[-1][1] - points[-2][1])
    else:
        delta = (
            points[index + 1][0] - points[index - 1][0],
            points[index + 1][1] - points[index - 1][1],
        )
    length = hypot(*delta)
    return (delta[0] / length, delta[1] / length) if length else (1.0, 0.0)


def _corridor_keys(
    point: tuple[float, float], radius: float, grid_spec: GridSpec, cell_size: float
) -> set[tuple[int, int]]:
    min_x = floor((point[0] - radius - grid_spec.origin_xy[0]) / cell_size)
    max_x = ceil((point[0] + radius - grid_spec.origin_xy[0]) / cell_size)
    min_y = floor((point[1] - radius - grid_spec.origin_xy[1]) / cell_size)
    max_y = ceil((point[1] + radius - grid_spec.origin_xy[1]) / cell_size)
    keys: set[tuple[int, int]] = set()
    for grid_x in range(min_x, max_x + 1):
        for grid_y in range(min_y, max_y + 1):
            bounds = _cell_bounds((grid_x, grid_y), grid_spec, cell_size)
            nearest_x = min(max(point[0], bounds[0]), bounds[1])
            nearest_y = min(max(point[1], bounds[2]), bounds[3])
            if hypot(point[0] - nearest_x, point[1] - nearest_y) <= radius:
                keys.add((grid_x, grid_y))
    return keys


def _first_time(indices: set[int], times: tuple[float, ...]) -> float:
    return min(times[index] for index in indices)


def _visit_windows(
    indices: set[int], times: tuple[float, ...], sample_step_s: float
) -> tuple[TimeWindow, ...]:
    ordered = sorted(indices)
    groups: list[list[int]] = [[ordered[0]]]
    for index in ordered[1:]:
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    windows: list[TimeWindow] = []
    for group in groups:
        start = round(times[group[0]])
        next_time = (
            times[group[-1] + 1]
            if group[-1] + 1 < len(times)
            else times[group[-1]] + sample_step_s
        )
        end = max(start + 1, round(next_time))
        windows.append(TimeWindow(start_s=start, end_s=end))
    return tuple(windows)
