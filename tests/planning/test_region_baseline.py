from __future__ import annotations

from itertools import combinations
from math import cos, pi, sin

import pytest
from shapely import Polygon

from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.execution_models import ExecutionRegion
from underwater_tracking.domain.prediction_models import AcceptedPrediction, PredictionHealth
from underwater_tracking.planning.region_baseline import (
    FourRegionBaseline,
    build_four_region_baseline,
)


MAP_BOUNDS = (0.0, 8_000.0, 0.0, 6_000.0)
WINDOWS = [(1_000.0, 1_540.0), (1_450.0, 1_990.0), (1_900.0, 2_440.0), (2_350.0, 2_800.0)]
MIN_REGION_AREA_M2 = 62_500.0
MIN_REGION_WIDTH_M = 250.0


def _area(polygon: tuple[tuple[float, float], ...]) -> float:
    return abs(
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(polygon, (*polygon[1:], polygon[0]), strict=True)
        )
    ) / 2.0


def _bounds(region: ExecutionRegion) -> tuple[float, float, float, float]:
    return (
        min(point[0] for point in region.geometry),
        max(point[0] for point in region.geometry),
        min(point[1] for point in region.geometry),
        max(point[1] for point in region.geometry),
    )


def _cross(
    start: tuple[float, float],
    middle: tuple[float, float],
    end: tuple[float, float],
) -> float:
    return (middle[0] - start[0]) * (end[1] - start[1]) - (
        middle[1] - start[1]
    ) * (end[0] - start[0])


def _triangulate(
    polygon: tuple[tuple[float, float], ...],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    points = list(polygon)
    if points[0] == points[-1]:
        points.pop()
    signed_area = sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(points, (*points[1:], points[0]), strict=True)
    )
    if signed_area < 0.0:
        points.reverse()
    indices = list(range(len(points)))
    triangles: list[tuple[tuple[float, float], ...]] = []
    while len(indices) > 3:
        for offset, current in enumerate(indices):
            previous = indices[offset - 1]
            following = indices[(offset + 1) % len(indices)]
            triangle = (points[previous], points[current], points[following])
            if _cross(*triangle) <= 1e-7:
                continue
            if any(
                candidate not in (previous, current, following)
                and all(
                    _cross(triangle[edge], triangle[(edge + 1) % 3], points[candidate])
                    > 1e-7
                    for edge in range(3)
                )
                for candidate in indices
            ):
                continue
            triangles.append(triangle)
            indices.pop(offset)
            break
        else:
            raise AssertionError("test polygon must be simple and triangulatable")
    triangles.append(tuple(points[index] for index in indices))
    return tuple(triangles)


def _clip_to_triangle(
    subject: tuple[tuple[float, float], ...],
    clip: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    result = list(subject)
    for clip_start, clip_end in zip(clip, (*clip[1:], clip[0]), strict=True):
        source = result
        result = []
        if not source:
            break
        for start, end in zip(source, (*source[1:], source[0]), strict=True):
            start_cross = _cross(clip_start, clip_end, start)
            end_cross = _cross(clip_start, clip_end, end)
            if (start_cross >= -1e-7) != (end_cross >= -1e-7):
                ratio = start_cross / (start_cross - end_cross)
                result.append(
                    (
                        start[0] + ratio * (end[0] - start[0]),
                        start[1] + ratio * (end[1] - start[1]),
                    )
                )
            if end_cross >= -1e-7:
                result.append(end)
    return tuple(result)


def _overlap(left: ExecutionRegion, right: ExecutionRegion) -> bool:
    intersection_area = sum(
        _area(clipped)
        for left_triangle in _triangulate(left.geometry)
        for right_triangle in _triangulate(right.geometry)
        if len(clipped := _clip_to_triangle(left_triangle, right_triangle)) >= 3
    )
    return intersection_area > 1e-6


def _point_in_or_on_polygon(
    point: tuple[float, float], polygon: tuple[tuple[float, float], ...]
) -> bool:
    x, y = point
    inside = False
    for start, end in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
        cross = (end[0] - start[0]) * (y - start[1]) - (end[1] - start[1]) * (
            x - start[0]
        )
        if (
            abs(cross) <= 1e-7
            and min(start[0], end[0]) - 1e-7 <= x <= max(start[0], end[0]) + 1e-7
            and min(start[1], end[1]) - 1e-7 <= y <= max(start[1], end[1]) + 1e-7
        ):
            return True
        if (start[1] > y) != (end[1] > y):
            intersection_x = start[0] + (y - start[1]) * (end[0] - start[0]) / (
                end[1] - start[1]
            )
            if intersection_x > x:
                inside = not inside
    return inside


def _window_points(
    accepted: AcceptedPrediction, start_s: float, end_s: float
) -> tuple[tuple[float, float], ...]:
    assert accepted.prediction is not None
    prediction = accepted.prediction
    samples: list[tuple[float, float]] = []
    for boundary_s in (start_s, end_s):
        for index in range(len(prediction.times_s) - 1):
            left_t = prediction.times_s[index]
            right_t = prediction.times_s[index + 1]
            if left_t <= boundary_s <= right_t:
                ratio = (boundary_s - left_t) / (right_t - left_t)
                left = prediction.points_xy[index]
                right = prediction.points_xy[index + 1]
                samples.append(
                    (
                        left[0] + ratio * (right[0] - left[0]),
                        left[1] + ratio * (right[1] - left[1]),
                    )
                )
                break
    samples.extend(
        point
        for time_s, point in zip(
            prediction.times_s, prediction.points_xy, strict=True
        )
        if start_s <= time_s <= end_s
    )
    return tuple(dict.fromkeys(samples))


def assert_four_region_invariants(result: FourRegionBaseline) -> None:
    regions = result.regions
    assert tuple(region.region_id for region in regions) == (
        "T1:task:01",
        "T1:task:02",
        "T1:task:03",
        "T1:task:04",
    )
    assert [(region.start_s, region.end_s) for region in regions] == WINDOWS
    assert all(
        all(MAP_BOUNDS[0] <= x <= MAP_BOUNDS[1] and MAP_BOUNDS[2] <= y <= MAP_BOUNDS[3] for x, y in region.geometry)
        for region in regions
    )
    assert all(Polygon(region.geometry).is_valid for region in regions)
    assert all(_area(region.geometry) >= MIN_REGION_AREA_M2 for region in regions)
    assert all(
        min(max_x - min_x, max_y - min_y) + 1e-6 >= MIN_REGION_WIDTH_M
        for min_x, max_x, min_y, max_y in map(_bounds, regions)
    )
    assert all(
        not _overlap(regions[left], regions[right])
        for left, right in combinations(range(4), 2)
        if right - left > 1
    )
    assert all(_overlap(regions[index], regions[index + 1]) for index in range(3))
    assert [region.predecessor_region_id for region in regions] == [
        None,
        "T1:task:01",
        "T1:task:02",
        "T1:task:03",
    ]
    assert [region.successor_region_id for region in regions] == [
        "T1:task:02",
        "T1:task:03",
        "T1:task:04",
        None,
    ]


def _accepted(
    *,
    status: str = "valid",
    regime: str = "imm",
    points: tuple[tuple[float, float], ...] | None = None,
    radii: tuple[float, ...] | None = None,
) -> AcceptedPrediction:
    track_points = points or tuple((1_000.0 + index * 300.0, 2_000.0) for index in range(19))
    track_radii = radii or tuple(150.0 for _ in track_points)
    prediction = PredictedTrackRef(
        prediction_id=f"pred:T1:{regime}",
        target_id="T1",
        sim_time_s=1_000,
        horizon_s=1_800.0,
        sample_step_s=100.0,
        times_s=tuple(1_000.0 + index * 100.0 for index in range(len(track_points))),
        points_xy=track_points,
        corridor_radius_m=track_radii,
        source_belief_history_ids=("belief:T1:1",),
        prediction_regime=regime,
    )
    return AcceptedPrediction(
        prediction=prediction,
        health=PredictionHealth(
            status=status,
            regime=regime,
            source_track_age_s=0.0,
            clipped_point_fraction=0.0,
            maximum_radius_m=max(track_radii),
            raw_prediction_id=prediction.prediction_id,
        ),
    )


@pytest.mark.parametrize(
    ("status", "regime", "expected_mode"),
    [
        ("valid", "imm", "imm"),
        ("degraded", "bspline", "degraded_prediction"),
        ("degraded", "boundary_recovery", "boundary_recovery"),
    ],
)
def test_baseline_preserves_four_region_invariants_for_prediction_modes(
    status: str, regime: str, expected_mode: str
) -> None:
    result = build_four_region_baseline(
        _accepted(status=status, regime=regime),
        target_id="T1",
        execution_revision=7,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=MAP_BOUNDS,
    )

    assert result.mode == expected_mode
    assert_four_region_invariants(result)


def test_stationary_corner_with_map_sized_uncertainty_still_returns_four_regions() -> None:
    points = tuple((0.0, 0.0) for _ in range(19))
    result = build_four_region_baseline(
        _accepted(status="degraded", regime="short_history", points=points, radii=(50_000.0,) * 19),
        target_id="T1",
        execution_revision=8,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=MAP_BOUNDS,
    )

    assert result.mode == "degraded_prediction"
    assert_four_region_invariants(result)


@pytest.mark.parametrize("reverse", [False, True])
def test_moving_slot_geometry_contains_its_time_segment_centerline(reverse: bool) -> None:
    points = tuple((1_000.0 + index * 300.0, 2_000.0) for index in range(19))
    if reverse:
        points = tuple(reversed(points))
    accepted = _accepted(points=points)

    result = build_four_region_baseline(
        accepted,
        target_id="T1",
        execution_revision=7,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=MAP_BOUNDS,
    )

    assert accepted.prediction is not None
    for region in result.regions:
        min_x, max_x, min_y, max_y = _bounds(region)
        assert all(
            min_x <= accepted.prediction.points_xy[index][0] <= max_x
            and min_y <= accepted.prediction.points_xy[index][1] <= max_y
            for index in region.centerline_indices
        )


@pytest.mark.parametrize(
    "points",
    [
        tuple((1_000.0 + index * 180.0, 700.0 + index * 140.0) for index in range(19)),
        tuple((1_000.0 + index / 18.0, 1_000.0 + index / 18.0) for index in range(19)),
        tuple((index / 18.0, index / 18.0) for index in range(19)),
    ],
    ids=("diagonal", "one-meter-displacement", "corner-one-meter-displacement"),
)
def test_adversarial_moving_geometry_keeps_centerline_and_overlap_invariants(
    points: tuple[tuple[float, float], ...],
) -> None:
    accepted = _accepted(regime="short_history", points=points, radii=(0.0,) * 19)

    result = build_four_region_baseline(
        accepted,
        target_id="T1",
        execution_revision=7,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=MAP_BOUNDS,
    )

    assert accepted.prediction is not None
    assert_four_region_invariants(result)
    for region in result.regions:
        assert all(
            _point_in_or_on_polygon(point, region.geometry)
            for point in _window_points(accepted, region.start_s, region.end_s)
        )


def test_looping_centerline_regions_contain_every_fixed_window_sample() -> None:
    points = tuple(
        (
            4_000.0 + 1_000.0 * cos(2.0 * pi * index / 18.0),
            3_000.0 + 1_000.0 * sin(2.0 * pi * index / 18.0),
        )
        for index in range(19)
    )
    accepted = _accepted(regime="short_history", points=points, radii=(0.0,) * 19)

    result = build_four_region_baseline(
        accepted,
        target_id="T1",
        execution_revision=7,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=MAP_BOUNDS,
    )

    assert_four_region_invariants(result)
    for region in result.regions:
        assert all(
            _point_in_or_on_polygon(point, region.geometry)
            for point in _window_points(accepted, region.start_s, region.end_s)
        )


def test_double_loop_keeps_positive_area_overlap_to_adjacent_regions_only() -> None:
    points = tuple(
        (
            4_000.0 + 1_000.0 * cos(4.0 * pi * index / 18.0),
            3_000.0 + 1_000.0 * sin(4.0 * pi * index / 18.0),
        )
        for index in range(19)
    )
    accepted = _accepted(regime="short_history", points=points, radii=(0.0,) * 19)

    result = build_four_region_baseline(
        accepted,
        target_id="T1",
        execution_revision=7,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=MAP_BOUNDS,
    )

    assert_four_region_invariants(result)
    for region in result.regions:
        assert all(
            _point_in_or_on_polygon(point, region.geometry)
            for point in _window_points(accepted, region.start_s, region.end_s)
        )


def test_horizontal_out_and_back_returns_simple_regions_containing_window_samples() -> None:
    points = tuple(
        (1_000.0 + 500.0 * (index if index <= 9 else 18 - index), 3_000.0)
        for index in range(19)
    )
    accepted = _accepted(regime="short_history", points=points, radii=(0.0,) * 19)

    result = build_four_region_baseline(
        accepted,
        target_id="T1",
        execution_revision=7,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=MAP_BOUNDS,
    )

    assert_four_region_invariants(result)
    for region in result.regions:
        assert all(
            _point_in_or_on_polygon(point, region.geometry)
            for point in _window_points(accepted, region.start_s, region.end_s)
        )


def test_unavailable_prediction_reprojects_prior_regions_with_new_windows() -> None:
    prior = build_four_region_baseline(
        _accepted(),
        target_id="T1",
        execution_revision=6,
        origin_sim_time_s=0.0,
        map_bounds_xy=MAP_BOUNDS,
    )
    unavailable = AcceptedPrediction(
        prediction=None,
        health=PredictionHealth(
            status="unavailable",
            regime="short_history",
            reason_codes=("all_candidates_rejected",),
            source_track_age_s=120.0,
            clipped_point_fraction=1.0,
            maximum_radius_m=0.0,
        ),
    )

    result = build_four_region_baseline(
        unavailable,
        target_id="T1",
        execution_revision=7,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=MAP_BOUNDS,
        prior_regions=prior.regions,
    )

    assert result.mode == "reprojected_previous"
    assert tuple(region.geometry for region in result.regions) == tuple(
        region.geometry for region in prior.regions
    )
    assert tuple(region.geometry_revision for region in result.regions) == tuple(
        region.geometry_revision for region in prior.regions
    )
    assert_four_region_invariants(result)


def test_geometry_revision_changes_only_when_geometry_changes() -> None:
    initial = build_four_region_baseline(
        _accepted(),
        target_id="T1",
        execution_revision=6,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=MAP_BOUNDS,
    )
    unchanged = build_four_region_baseline(
        _accepted(),
        target_id="T1",
        execution_revision=7,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=MAP_BOUNDS,
        prior_regions=initial.regions,
    )
    shifted = build_four_region_baseline(
        _accepted(points=tuple((1_500.0 + index * 300.0, 2_500.0) for index in range(19))),
        target_id="T1",
        execution_revision=8,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=MAP_BOUNDS,
        prior_regions=unchanged.regions,
    )

    assert {region.geometry_revision for region in initial.regions} == {1}
    assert {region.geometry_revision for region in unchanged.regions} == {1}
    assert {region.geometry_revision for region in shifted.regions} == {2}


def test_unavailable_prediction_without_prior_regions_fails_explicitly() -> None:
    unavailable = AcceptedPrediction(
        prediction=None,
        health=PredictionHealth(
            status="unavailable",
            regime="short_history",
            source_track_age_s=120.0,
            clipped_point_fraction=1.0,
            maximum_radius_m=0.0,
        ),
    )

    with pytest.raises(ValueError, match="no accepted geometry or prior four-region baseline"):
        build_four_region_baseline(
            unavailable,
            target_id="T1",
            execution_revision=7,
            origin_sim_time_s=1_000.0,
            map_bounds_xy=MAP_BOUNDS,
        )
