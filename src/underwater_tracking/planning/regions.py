from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from math import ceil, floor, hypot, isfinite, pi, sqrt

from underwater_tracking.domain.agent_models import IntentHypothesis, PredictedTrackRef
from underwater_tracking.domain.regional_models import (
    GridSpec,
    RegionCell,
    RegionTask,
    SonarPolicy,
    TaskRegion,
    TaskRegionProposal,
    TaskRegionProposalSet,
    TargetRegionPlan,
    TimeWindow,
)

TASK_REGION_CELL_SIZE_M = 1_000.0
TASK_REGION_MIN_EXTENT_M = 3_000.0
TASK_REGION_MAX_ADJACENT_OVERLAP_RATIO = 0.35


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
    *,
    required_quality: float = 0.0,
) -> TargetRegionPlan:
    """Rasterize an estimated prediction corridor into ordered square tasks."""
    if not 0.0 <= required_quality <= 1.0:
        raise ValueError("required_quality must be between 0 and 1")
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
    centerline_keys_by_sample: dict[tuple[int, int], set[int]] = defaultdict(set)

    for index, point in enumerate(points):
        centerline_key = _grid_key(point, grid_spec, cell_size)
        if _cell_is_inside_map(centerline_key, grid_spec, cell_size, map_bounds_xy):
            centerline_keys_by_sample[centerline_key].add(index)
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
        # Keep loop-backs visible even when uncertainty expansion bridges the
        # gap between two centerline visits.
        visit_indices = centerline_keys_by_sample.get(key) or keys_by_sample[key]
        windows = _visit_windows(visit_indices, times, prediction.sample_step_s)
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
                predicted_target_xy=points[min(visit_indices)],
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
    tasks = tuple(
        RegionTask(
            region_id=cell.region_id,
            target_id=cell.target_id,
            active_window=TimeWindow(start_s=cell.first_entry_s, end_s=cell.last_exit_s),
            required_quality=required_quality,
            required_uuv_count=uuv_count,
            uuv_roles=("passive_tracker",) if uuv_count else (),
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

    return CommunicationRequirement()


def build_dynamic_region_chain(*args, **kwargs):
    """Compatibility entrypoint for the executable four-region planner."""
    from underwater_tracking.planning.dynamic_regions import (
        build_dynamic_region_chain as _build_dynamic_region_chain,
    )

    return _build_dynamic_region_chain(*args, **kwargs)


def build_llm_task_region_plan(
    prediction: PredictedTrackRef,
    intent: IntentHypothesis,
    proposal_set: TaskRegionProposalSet,
    map_bounds_xy: tuple[float, float, float, float],
    grid_spec: GridSpec,
    *,
    required_quality: float = 0.0,
    uuv_scan_range_m: float = 3_500.0,
) -> TargetRegionPlan:
    """Materialize LLM coordinate regions into a shared 1 km cell grid."""
    if not 0.0 <= required_quality <= 1.0:
        raise ValueError("required_quality must be between 0 and 1")
    if not isfinite(uuv_scan_range_m) or uuv_scan_range_m <= 0.0:
        raise ValueError("uuv_scan_range_m must be finite and positive")
    if len(proposal_set.regions) != 4:
        raise ValueError("exactly four task regions are required")
    points = tuple(prediction.points_xy)
    if not points:
        raise ValueError("prediction points are required for task regions")
    times = tuple(prediction.times_s)
    if len(times) != len(points):
        times = tuple(
            prediction.sim_time_s + index * prediction.sample_step_s
            for index in range(len(points))
        )
    task_grid = grid_spec.model_copy(
        update={
            "min_cell_size_m": TASK_REGION_CELL_SIZE_M,
            "max_cell_size_m": TASK_REGION_CELL_SIZE_M,
            "cell_size_rounding_m": TASK_REGION_CELL_SIZE_M,
        }
    )
    regions: list[TaskRegion] = []
    cells_by_id: dict[str, RegionCell] = {}
    tasks_by_id: dict[str, RegionTask] = {}
    evidence_ids = tuple(sorted({*prediction.source_belief_history_ids, *intent.evidence_ids, prediction.prediction_id}))
    aligned_proposals: list[
        tuple[
            int,
            TaskRegionProposal,
            tuple[float, float, float, float],
        ]
    ] = []
    for provider_index, proposal in enumerate(proposal_set.regions):
        aligned_bounds = _grid_aligned_task_region_bounds(
            proposal.lower_left_xy,
            proposal.upper_right_xy,
            task_grid.origin_xy,
        )
        if (
            aligned_bounds[1] - aligned_bounds[0] < TASK_REGION_MIN_EXTENT_M
            or aligned_bounds[3] - aligned_bounds[2] < TASK_REGION_MIN_EXTENT_M
        ):
            raise ValueError("task regions must be at least 3000 m wide and high")
        bounds = _clip_task_region_bounds(aligned_bounds, map_bounds_xy)
        aligned_proposals.append((provider_index, proposal, bounds))
    ordered_proposals = _normalize_task_region_proposals(
        tuple(aligned_proposals),
        points,
        map_bounds_xy,
        task_grid.origin_xy,
    )

    for index, (_, _, proposal, bounds, sample_indices) in enumerate(
        ordered_proposals, start=1
    ):
        min_x, max_x, min_y, max_y = bounds
        first_index = sample_indices[0]
        last_index = sample_indices[-1]
        entry_s = max(0, round(times[first_index]))
        exit_s = max(
            entry_s + 1,
            round(
                times[last_index + 1]
                if last_index + 1 < len(times)
                else times[last_index] + prediction.sample_step_s
            ),
        )
        window = TimeWindow(start_s=entry_s, end_s=exit_s)
        region_id = f"{prediction.target_id}:task:{index:02d}"
        region_cells = _task_region_cells(
            prediction.target_id,
            min_x,
            max_x,
            min_y,
            max_y,
            task_grid.origin_xy,
        )
        cell_ids = tuple(cell.region_id for cell in region_cells)
        region_area_m2 = (max_x - min_x) * (max_y - min_y)
        uuv_demand = _required_uuv_count(region_area_m2, uuv_scan_range_m)
        regions.append(
            TaskRegion(
                region_id=region_id,
                lower_left_xy=(min_x, min_y),
                upper_right_xy=(max_x, max_y),
                cell_ids=cell_ids,
                active_window=window,
                required_uuv_count=uuv_demand,
                rationale=proposal.rationale,
            )
        )
        for cell in region_cells:
            updated_cell = cell.model_copy(
                update={
                    "first_entry_s": entry_s,
                    "last_exit_s": exit_s,
                    "visit_windows": (window,),
                    "occupancy_likelihood": len(sample_indices) / len(points),
                    "intent_labels": (intent.label,),
                    "evidence_ids": evidence_ids,
                    "predicted_target_xy": points[first_index],
                }
            )
            existing_cell = cells_by_id.get(cell.region_id)
            if existing_cell is None:
                cells_by_id[cell.region_id] = updated_cell
            else:
                cells_by_id[cell.region_id] = existing_cell.model_copy(
                    update={
                        "first_entry_s": min(existing_cell.first_entry_s, entry_s),
                        "last_exit_s": max(existing_cell.last_exit_s, exit_s),
                        "visit_windows": tuple(
                            {
                                (item.start_s, item.end_s): item
                                for item in (*existing_cell.visit_windows, window)
                            }[key]
                            for key in sorted(
                                {
                                    (item.start_s, item.end_s)
                                    for item in (*existing_cell.visit_windows, window)
                                }
                            )
                        ),
                        "occupancy_likelihood": max(
                            existing_cell.occupancy_likelihood,
                            len(sample_indices) / len(points),
                        ),
                    }
                )
            task = RegionTask(
                region_id=cell.region_id,
                target_id=prediction.target_id,
                active_window=window,
                required_quality=required_quality,
                required_uuv_count=uuv_demand,
                uuv_roles=("passive_tracker",) * uuv_demand,
                sonar_policy=SonarPolicy(passive_required=True, active_allowed=False),
                communication=cell_communication(task_grid),
                evidence_ids=evidence_ids,
            )
            existing_task = tasks_by_id.get(cell.region_id)
            if existing_task is None:
                tasks_by_id[cell.region_id] = task
            else:
                merged_count = max(existing_task.required_uuv_count, uuv_demand)
                tasks_by_id[cell.region_id] = existing_task.model_copy(
                    update={
                        "active_window": TimeWindow(
                            start_s=min(existing_task.active_window.start_s, entry_s),
                            end_s=max(existing_task.active_window.end_s, exit_s),
                        ),
                        "required_uuv_count": merged_count,
                        "uuv_roles": ("passive_tracker",) * merged_count,
                    }
                )
    return TargetRegionPlan(
        target_id=prediction.target_id,
        grid_spec=task_grid,
        cell_size_m=TASK_REGION_CELL_SIZE_M,
        cells=tuple(cells_by_id.values()),
        tasks=tuple(tasks_by_id.values()),
        task_regions=tuple(regions),
        prediction_id=prediction.prediction_id,
        intent_label=intent.label,
        intent_confidence=intent.confidence,
        evidence_ids=evidence_ids,
        fallback_used=prediction.fallback_used,
        fallback_reason=prediction.fallback_reason,
    )


def _corridor_sample_indices(
    points: tuple[tuple[float, float], ...],
    bounds: tuple[float, float, float, float],
) -> tuple[int, ...]:
    """Return prediction-centerline samples contained by a task region."""
    min_x, max_x, min_y, max_y = bounds
    return tuple(
        index
        for index, point in enumerate(points)
        if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y
    )


def _normalize_task_region_proposals(
    aligned_proposals: tuple[
        tuple[int, TaskRegionProposal, tuple[float, float, float, float]], ...
    ],
    points: tuple[tuple[float, float], ...],
    map_bounds: tuple[float, float, float, float],
    origin: tuple[float, float],
) -> tuple[
    tuple[
        int,
        int,
        TaskRegionProposal,
        tuple[float, float, float, float],
        tuple[int, ...],
    ],
    ...,
]:
    """Keep valid provider geometry or repair its forecast-chain invariants."""
    ordered: list[
        tuple[
            int,
            int,
            TaskRegionProposal,
            tuple[float, float, float, float],
            tuple[int, ...],
        ]
    ] = []
    missing_centerline = False
    for provider_index, proposal, bounds in aligned_proposals:
        sample_indices = _corridor_sample_indices(points, bounds)
        if (
            not sample_indices
            or bounds[1] - bounds[0] < TASK_REGION_MIN_EXTENT_M
            or bounds[3] - bounds[2] < TASK_REGION_MIN_EXTENT_M
        ):
            missing_centerline = True
            break
        ordered.append(
            (sample_indices[0], provider_index, proposal, bounds, sample_indices)
        )
    if not missing_centerline:
        ordered.sort(key=lambda item: (item[0], item[1]))
        if _centerline_is_stationary(points):
            # A stationary forecast legitimately maps all four chronological
            # slots to the same search surface.  Spatial non-overlap is
            # impossible in that case; retain the LLM geometry and keep the
            # distinct time windows for execution handoff.
            return tuple(ordered)
        try:
            _validate_task_region_overlap_sequence(
                tuple(item[3] for item in ordered)
            )
        except ValueError:
            pass
        else:
            return tuple(ordered)

    if _centerline_is_stationary(points):
        anchor = points[0]
        stationary: list[
            tuple[int, TaskRegionProposal, tuple[float, float, float, float]]
        ] = []
        for provider_index, proposal, bounds in aligned_proposals:
            x_interval = _fit_aligned_interval(
                (bounds[0], bounds[1]),
                anchor[0],
                map_bounds[0],
                map_bounds[1],
                origin[0],
            )
            y_interval = _fit_aligned_interval(
                (bounds[2], bounds[3]),
                anchor[1],
                map_bounds[2],
                map_bounds[3],
                origin[1],
            )
            stationary.append(
                (provider_index, proposal, (*x_interval, *y_interval))
            )
        ordered_stationary = sorted(stationary, key=lambda item: item[0])
        slot_indices = tuple(
            round(index * (len(points) - 1) / (len(ordered_stationary) - 1))
            if len(ordered_stationary) > 1
            else 0
            for index in range(len(ordered_stationary))
        )
        return tuple(
            (
                slot_index,
                provider_index,
                proposal,
                bounds,
                (slot_index,),
            )
            for slot_index, (provider_index, proposal, bounds) in zip(
                slot_indices,
                ordered_stationary,
                strict=True,
            )
        )

    try:
        normalized_bounds = _prediction_partition_bounds(
            aligned_proposals,
            points,
            map_bounds,
            origin,
        )
    except ValueError as exc:
        if "prediction centerline is too short" not in str(exc):
            raise
        # Four minimum-size spatial cells cannot be separated when the
        # forecast only covers a compact movement.  Preserve the LLM's four
        # proposals as chronological time slots and fit each slot around its
        # own forecast sample; this is still an LLM-authored plan, not a
        # deterministic mission replacement.
        return _compact_temporal_region_proposals(
            aligned_proposals,
            points,
            map_bounds,
            origin,
        )
    repaired: list[
        tuple[
            int,
            int,
            TaskRegionProposal,
            tuple[float, float, float, float],
            tuple[int, ...],
        ]
    ] = []
    for provider_index, proposal, bounds in normalized_bounds:
        sample_indices = _corridor_sample_indices(points, bounds)
        if not sample_indices:
            raise ValueError(
                "LLM task region normalization could not supply a prediction centerline sample"
            )
        repaired.append(
            (sample_indices[0], provider_index, proposal, bounds, sample_indices)
        )
    repaired.sort(key=lambda item: (item[0], item[1]))
    _validate_task_region_overlap_sequence(tuple(item[3] for item in repaired))
    return tuple(repaired)


def _compact_temporal_region_proposals(
    aligned_proposals: tuple[
        tuple[int, TaskRegionProposal, tuple[float, float, float, float]], ...
    ],
    points: tuple[tuple[float, float], ...],
    map_bounds: tuple[float, float, float, float],
    origin: tuple[float, float],
) -> tuple[
    tuple[
        int,
        int,
        TaskRegionProposal,
        tuple[float, float, float, float],
        tuple[int, ...],
    ],
    ...,
]:
    """Keep four LLM slots when compact motion cannot form four spatial cells."""
    ranked = sorted(aligned_proposals, key=lambda item: item[0])
    anchor_indices = tuple(
        round(index * (len(points) - 1) / (len(ranked) - 1))
        for index in range(len(ranked))
    )
    compact: list[
        tuple[
            int,
            int,
            TaskRegionProposal,
            tuple[float, float, float, float],
            tuple[int, ...],
        ]
    ] = []
    for (provider_index, proposal, raw_bounds), anchor_index in zip(
        ranked,
        anchor_indices,
        strict=True,
    ):
        anchor = points[anchor_index]
        x_interval = _fit_aligned_interval(
            (raw_bounds[0], raw_bounds[1]),
            anchor[0],
            map_bounds[0],
            map_bounds[1],
            origin[0],
        )
        y_interval = _fit_aligned_interval(
            (raw_bounds[2], raw_bounds[3]),
            anchor[1],
            map_bounds[2],
            map_bounds[3],
            origin[1],
        )
        compact.append(
            (
                anchor_index,
                provider_index,
                proposal,
                (*x_interval, *y_interval),
                (anchor_index,),
            )
        )
    return tuple(compact)


def _centerline_is_stationary(
    points: tuple[tuple[float, float], ...],
) -> bool:
    """Treat sub-cell motion as one spatial surface with ordered time slots."""
    if not points:
        return False
    return max(
        max(point[0] for point in points) - min(point[0] for point in points),
        max(point[1] for point in points) - min(point[1] for point in points),
    ) < TASK_REGION_CELL_SIZE_M


def _prediction_partition_bounds(
    aligned_proposals: tuple[
        tuple[int, TaskRegionProposal, tuple[float, float, float, float]], ...
    ],
    points: tuple[tuple[float, float], ...],
    map_bounds: tuple[float, float, float, float],
    origin: tuple[float, float],
) -> tuple[
    tuple[int, TaskRegionProposal, tuple[float, float, float, float]], ...
]:
    """Partition provider rectangles along the prediction's dominant axis."""
    x_span = max(point[0] for point in points) - min(point[0] for point in points)
    y_span = max(point[1] for point in points) - min(point[1] for point in points)
    axis = 0 if x_span >= y_span else 1
    axis_start = points[0][axis]
    axis_end = points[-1][axis]
    direction = 1.0 if axis_end >= axis_start else -1.0
    axis_map_min, axis_map_max = (
        (map_bounds[0], map_bounds[1])
        if axis == 0
        else (map_bounds[2], map_bounds[3])
    )

    def projected(value: float) -> float:
        return value - axis_map_min if direction > 0 else axis_map_max - value

    def unprojected_interval(start: float, end: float) -> tuple[float, float]:
        if direction > 0:
            return axis_map_min + start, axis_map_min + end
        return axis_map_max - end, axis_map_max - start

    ranked = sorted(
        aligned_proposals,
        key=lambda item: (
            projected((item[2][axis * 2] + item[2][axis * 2 + 1]) / 2.0),
            item[0],
        ),
    )
    anchor_indices = tuple(
        round(index * (len(points) - 1) / (len(ranked) - 1))
        for index in range(len(ranked))
    )
    anchors = tuple(projected(points[index][axis]) for index in anchor_indices)
    if any(right <= left for left, right in pairwise(anchors)):
        raise ValueError(
            "prediction centerline is too short to normalize four task regions"
        )
    handoffs = tuple(
        floor(((left + right) / 2.0) / TASK_REGION_CELL_SIZE_M)
        * TASK_REGION_CELL_SIZE_M
        for left, right in pairwise(anchors)
    )
    axis_length = axis_map_max - axis_map_min
    axis_intervals: list[tuple[float, float]] = []
    for index, anchor in enumerate(anchors):
        start = (
            max(0.0, floor((anchor - 2_000.0) / 1_000.0) * 1_000.0)
            if index == 0
            else handoffs[index - 1]
        )
        end = (
            min(axis_length, ceil((anchor + 2_000.0) / 1_000.0) * 1_000.0)
            if index == len(anchors) - 1
            else handoffs[index] + TASK_REGION_CELL_SIZE_M
        )
        if end - start < TASK_REGION_MIN_EXTENT_M:
            raise ValueError(
                "prediction centerline is too short to normalize four task regions"
            )
        axis_intervals.append((start, end))

    result: list[
        tuple[int, TaskRegionProposal, tuple[float, float, float, float]]
    ] = []
    cross_axis = 1 - axis
    cross_map_min, cross_map_max = (
        (map_bounds[2], map_bounds[3])
        if cross_axis == 1
        else (map_bounds[0], map_bounds[1])
    )
    cross_origin = origin[cross_axis]
    for index, (provider_index, proposal, raw_bounds) in enumerate(ranked):
        anchor_point = points[anchor_indices[index]]
        cross_raw = (
            (raw_bounds[2], raw_bounds[3])
            if cross_axis == 1
            else (raw_bounds[0], raw_bounds[1])
        )
        cross_interval = _fit_aligned_interval(
            cross_raw,
            anchor_point[cross_axis],
            cross_map_min,
            cross_map_max,
            cross_origin,
        )
        axis_interval = unprojected_interval(*axis_intervals[index])
        bounds = (
            (axis_interval[0], axis_interval[1], cross_interval[0], cross_interval[1])
            if axis == 0
            else (cross_interval[0], cross_interval[1], axis_interval[0], axis_interval[1])
        )
        result.append((provider_index, proposal, bounds))
    return tuple(result)


def _fit_aligned_interval(
    raw: tuple[float, float],
    anchor: float,
    map_min: float,
    map_max: float,
    origin: float,
) -> tuple[float, float]:
    """Shift one grid-aligned interval just enough to contain an anchor."""
    if raw[0] <= anchor <= raw[1]:
        return raw
    extent = max(TASK_REGION_MIN_EXTENT_M, raw[1] - raw[0])
    extent = min(extent, map_max - map_min)
    start = origin + floor((anchor - extent / 2.0 - origin) / 1_000.0) * 1_000.0
    start = min(max(start, map_min), map_max - extent)
    end = start + extent
    if not start <= anchor <= end:
        raise ValueError("task region cannot be aligned around prediction centerline")
    return start, end


def _validate_task_region_overlap_sequence(
    bounds_by_region: tuple[tuple[float, float, float, float], ...],
) -> None:
    """Require a small handoff overlap only between consecutive regions."""
    for left_index, left in enumerate(bounds_by_region):
        left_area = (left[1] - left[0]) * (left[3] - left[2])
        for right_index in range(left_index + 1, len(bounds_by_region)):
            right = bounds_by_region[right_index]
            overlap_width = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
            overlap_height = max(0.0, min(left[3], right[3]) - max(left[2], right[2]))
            overlap_area = overlap_width * overlap_height
            if right_index == left_index + 1:
                if overlap_area <= 0.0:
                    raise ValueError("adjacent task regions require a handoff overlap")
                right_area = (right[1] - right[0]) * (right[3] - right[2])
                if overlap_area / min(left_area, right_area) > TASK_REGION_MAX_ADJACENT_OVERLAP_RATIO:
                    raise ValueError("adjacent task region overlap is too large")
            elif overlap_area > 0.0:
                raise ValueError("non-adjacent task regions must not overlap")


def _required_uuv_count(region_area_m2: float, uuv_scan_range_m: float) -> int:
    """Size a batch from area and one UUV's circular active-scan footprint."""
    scan_footprint_m2 = pi * uuv_scan_range_m**2
    return min(4, max(2, ceil(region_area_m2 / scan_footprint_m2)))


def _grid_aligned_task_region_bounds(
    lower_left: tuple[float, float],
    upper_right: tuple[float, float],
    origin: tuple[float, float],
) -> tuple[float, float, float, float]:
    min_x = origin[0] + floor((lower_left[0] - origin[0]) / TASK_REGION_CELL_SIZE_M) * TASK_REGION_CELL_SIZE_M
    max_x = origin[0] + ceil((upper_right[0] - origin[0]) / TASK_REGION_CELL_SIZE_M) * TASK_REGION_CELL_SIZE_M
    min_y = origin[1] + floor((lower_left[1] - origin[1]) / TASK_REGION_CELL_SIZE_M) * TASK_REGION_CELL_SIZE_M
    max_y = origin[1] + ceil((upper_right[1] - origin[1]) / TASK_REGION_CELL_SIZE_M) * TASK_REGION_CELL_SIZE_M
    return min_x, max_x, min_y, max_y


def _clip_task_region_bounds(
    bounds: tuple[float, float, float, float],
    map_bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Clip provider geometry before deterministic centerline normalization."""
    return (
        min(max(bounds[0], map_bounds[0]), map_bounds[1]),
        min(max(bounds[1], map_bounds[0]), map_bounds[1]),
        min(max(bounds[2], map_bounds[2]), map_bounds[3]),
        min(max(bounds[3], map_bounds[2]), map_bounds[3]),
    )


def _task_region_cells(
    target_id: str,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    origin: tuple[float, float],
) -> tuple[RegionCell, ...]:
    start_x = round((min_x - origin[0]) / TASK_REGION_CELL_SIZE_M)
    end_x = round((max_x - origin[0]) / TASK_REGION_CELL_SIZE_M)
    start_y = round((min_y - origin[1]) / TASK_REGION_CELL_SIZE_M)
    end_y = round((max_y - origin[1]) / TASK_REGION_CELL_SIZE_M)
    return tuple(
        RegionCell(
            region_id=f"{target_id}:cell:{grid_x}:{grid_y}",
            target_id=target_id,
            grid_x=grid_x,
            grid_y=grid_y,
            min_x=origin[0] + grid_x * TASK_REGION_CELL_SIZE_M,
            max_x=origin[0] + (grid_x + 1) * TASK_REGION_CELL_SIZE_M,
            min_y=origin[1] + grid_y * TASK_REGION_CELL_SIZE_M,
            max_y=origin[1] + (grid_y + 1) * TASK_REGION_CELL_SIZE_M,
            center_xy=(
                origin[0] + (grid_x + 0.5) * TASK_REGION_CELL_SIZE_M,
                origin[1] + (grid_y + 0.5) * TASK_REGION_CELL_SIZE_M,
            ),
            cell_size_m=TASK_REGION_CELL_SIZE_M,
            first_entry_s=0,
            last_exit_s=1,
        )
        for grid_x in range(start_x, end_x)
        for grid_y in range(start_y, end_y)
    )


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
