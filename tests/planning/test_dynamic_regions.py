from __future__ import annotations

from types import SimpleNamespace

from underwater_tracking.planning.dynamic_regions import (
    RegionWindowPolicy,
    build_dynamic_region_chain,
    normalize_region_chain,
)
from underwater_tracking.planning.regions import build_dynamic_region_chain as legacy_region_entrypoint


def _prediction() -> SimpleNamespace:
    points = tuple(
        (
            100.0 + 0.8 * time,
            500.0 + 300.0 * (time / 1800.0) ** 2,
        )
        for time in range(0, 1801, 100)
    )
    return SimpleNamespace(
        prediction_id="pred:T1:7",
        target_id="T1",
        origin_sim_time_s=0.0,
        sim_time_s=0,
        times_s=tuple(float(time) for time in range(0, 1801, 100)),
        centerline_xy=points,
        points_xy=points,
        covariance_xy=tuple((25.0, 0.0, 0.0, 25.0) for _ in points),
        corridor_radius_m=tuple(5.0 for _ in points),
    )


def test_build_dynamic_region_chain_has_four_stable_windows_and_ids() -> None:
    chain = build_dynamic_region_chain(
        _prediction(),
        execution_revision=7,
        map_bounds_xy=(-500.0, 2_000.0, -500.0, 2_000.0),
    )

    assert [region.region_id for region in chain.regions] == [
        "T1:task:01",
        "T1:task:02",
        "T1:task:03",
        "T1:task:04",
    ]
    assert [(region.start_s, region.end_s) for region in chain.regions] == [
        (0.0, 540.0),
        (450.0, 990.0),
        (900.0, 1440.0),
        (1350.0, 1800.0),
    ]
    assert all(region.execution_revision == 7 for region in chain.regions)
    assert all(region.prediction_id == "pred:T1:7" for region in chain.regions)
    assert all(region.centerline_indices for region in chain.regions)
    assert all(
        set(region.centerline_indices)
        <= set(range(len(_prediction().centerline_xy)))
        for region in chain.regions
    )


def test_dynamic_regions_follow_a_turning_centerline_and_only_neighbors_overlap() -> None:
    prediction = _prediction()
    chain = build_dynamic_region_chain(
        prediction,
        execution_revision=1,
        map_bounds_xy=(-500.0, 2_000.0, -500.0, 2_000.0),
        policy=RegionWindowPolicy(min_width_m=35.0, uncertainty_margin_m=10.0),
    )

    assert any(
        left[0] != right[0] and left[1] != right[1]
        for region in chain.regions
        for left, right in zip(region.geometry, (*region.geometry[1:], region.geometry[0]))
    )
    assert all(
        region.handoff_start_s == next_region.start_s
        and region.handoff_end_s == region.end_s
        for region, next_region in zip(chain.regions, chain.regions[1:])
    )
    assert all(
        not _bbox_overlap(chain.regions[left].geometry, chain.regions[right].geometry)
        for left in range(4)
        for right in range(left + 2, 4)
    )


def test_dynamic_region_geometry_is_clipped_but_retains_width_and_area() -> None:
    chain = build_dynamic_region_chain(
        _prediction(),
        execution_revision=2,
        map_bounds_xy=(0.0, 1_000.0, 0.0, 1_000.0),
        policy=RegionWindowPolicy(min_width_m=40.0, uncertainty_margin_m=20.0),
    )

    for region in chain.regions:
        assert all(0.0 <= x <= 1_000.0 and 0.0 <= y <= 1_000.0 for x, y in region.geometry)
        assert _area(region.geometry) >= 40.0 * 40.0


def test_normalize_region_chain_rebuilds_invalid_slot_order_from_prediction() -> None:
    chain = build_dynamic_region_chain(
        _prediction(),
        execution_revision=3,
        map_bounds_xy=(-500.0, 2_000.0, -500.0, 2_000.0),
    )
    malformed = chain.model_copy(
        update={"regions": tuple(reversed(chain.regions))}
    )

    normalized = normalize_region_chain(
        malformed,
        prediction=_prediction(),
        execution_revision=3,
        map_bounds_xy=(-500.0, 2_000.0, -500.0, 2_000.0),
    )

    assert tuple(region.slot_index for region in normalized.regions) == (1, 2, 3, 4)
    assert tuple(region.region_id for region in normalized.regions) == (
        "T1:task:01",
        "T1:task:02",
        "T1:task:03",
        "T1:task:04",
    )


def test_legacy_region_entrypoint_exposes_the_execution_chain() -> None:
    chain = legacy_region_entrypoint(
        _prediction(),
        execution_revision=5,
        map_bounds_xy=(-500.0, 2_000.0, -500.0, 2_000.0),
    )

    assert len(chain.regions) == 4


def _bbox_overlap(left, right) -> bool:
    left_x = [point[0] for point in left]
    left_y = [point[1] for point in left]
    right_x = [point[0] for point in right]
    right_y = [point[1] for point in right]
    return (
        min(left_x) < max(right_x)
        and min(right_x) < max(left_x)
        and min(left_y) < max(right_y)
        and min(right_y) < max(left_y)
    )


def _area(points) -> float:
    return abs(
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(points, (*points[1:], points[0]))
        )
    ) / 2.0
