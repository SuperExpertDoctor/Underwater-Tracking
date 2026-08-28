from __future__ import annotations

from itertools import combinations

import pytest

from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.execution_models import ExecutionRegion
from underwater_tracking.domain.prediction_models import AcceptedPrediction, PredictionHealth
from underwater_tracking.planning.region_baseline import (
    FourRegionBaseline,
    build_four_region_baseline,
)


MAP_BOUNDS = (0.0, 8_000.0, 0.0, 6_000.0)
WINDOWS = [(1_000.0, 1_540.0), (1_450.0, 1_990.0), (1_900.0, 2_440.0), (2_350.0, 2_800.0)]


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


def _overlap(left: ExecutionRegion, right: ExecutionRegion) -> bool:
    left_bounds = _bounds(left)
    right_bounds = _bounds(right)
    return (
        min(left_bounds[1], right_bounds[1]) > max(left_bounds[0], right_bounds[0])
        and min(left_bounds[3], right_bounds[3]) > max(left_bounds[2], right_bounds[2])
    )


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
    assert all(_area(region.geometry) > 0.0 for region in regions)
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
    ],
    ids=("diagonal", "one-meter-displacement"),
)
def test_adversarial_moving_geometry_keeps_centerline_and_overlap_invariants(
    points: tuple[tuple[float, float], ...],
) -> None:
    accepted = _accepted(points=points, radii=(10.0,) * 19)

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
        min_x, max_x, min_y, max_y = _bounds(region)
        assert all(
            min_x <= accepted.prediction.points_xy[index][0] <= max_x
            and min_y <= accepted.prediction.points_xy[index][1] <= max_y
            for index in region.centerline_indices
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
