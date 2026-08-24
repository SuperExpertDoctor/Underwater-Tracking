from underwater_tracking.planning.coverage import (
    serpentine_coverage_waypoints,
    serpentine_coverage_waypoints_by_uuv,
)


def test_serpentine_coverage_stays_inside_rectangle_and_alternates_direction() -> None:
    points = serpentine_coverage_waypoints(
        ((0.0, 0.0), (100.0, 0.0), (100.0, 60.0), (0.0, 60.0)),
        lane_count=3,
    )

    assert len(points) == 6
    assert points == (
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 30.0),
        (0.0, 30.0),
        (0.0, 60.0),
        (100.0, 60.0),
    )


def test_multi_uuv_coverage_assigns_distinct_serpentine_lanes() -> None:
    paths = serpentine_coverage_waypoints_by_uuv(
        ((0.0, 0.0), (100.0, 0.0), (100.0, 60.0), (0.0, 60.0)),
        ("U02", "U01"),
    )

    assert tuple(paths) == ("U01", "U02")
    assert paths["U01"] == ((0.0, 0.0), (100.0, 0.0))
    assert paths["U02"] == ((100.0, 60.0), (0.0, 60.0))


def test_multi_uuv_coverage_orients_each_lane_from_the_deployment_point() -> None:
    paths = serpentine_coverage_waypoints_by_uuv(
        ((0.0, 0.0), (100.0, 0.0), (100.0, 60.0), (0.0, 60.0)),
        ("U01", "U02"),
        start_point=(0.0, 0.0),
    )

    assert paths["U01"] == ((0.0, 0.0), (100.0, 0.0))
    assert paths["U02"] == ((0.0, 60.0), (100.0, 60.0))


def test_multi_uuv_coverage_falls_back_to_interior_lanes_for_vertex_boundaries() -> None:
    paths = serpentine_coverage_waypoints_by_uuv(
        ((0.0, 0.0), (0.0, 100.0), (100.0, 0.0), (100.0, 100.0)),
        ("U01", "U02"),
    )

    assert all(paths.values())
    assert paths["U01"] != paths["U02"]
