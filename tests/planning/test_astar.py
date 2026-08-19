from underwater_tracking.planning.astar import AStarRoutePlanner, RoutePlan


def test_route_starts_at_current_position_and_returns_home_around_region() -> None:
    planner = AStarRoutePlanner(grid_size_m=1.0)

    route = planner.plan(
        (0.0, 3.0),
        ((6.0, 3.0),),
        (0.0, 3.0),
        forbidden_regions=((2.0, 4.0, 2.0, 4.0),),
        map_bounds=(0.0, 8.0, 0.0, 8.0),
    )

    assert isinstance(route, RoutePlan)
    assert route.points[0] == (0.0, 3.0)
    assert route.points[-1] == (0.0, 3.0)
    assert route.distance_m > 0.0
    assert not any(
        2.0 < x < 4.0 and 2.0 < y < 4.0 for x, y in route.points
    )


def test_route_revalidates_every_inserted_stop_and_rejects_interior_stop() -> None:
    planner = AStarRoutePlanner(grid_size_m=1.0)

    route = planner.plan(
        (0.0, 0.0),
        ((4.0, 4.0),),
        (0.0, 0.0),
        forbidden_regions=((2.0, 3.0, 2.0, 3.0),),
        map_bounds=(0.0, 6.0, 0.0, 6.0),
    )
    impossible = planner.plan(
        (0.0, 0.0),
        ((2.5, 2.5),),
        (0.0, 0.0),
        forbidden_regions=((2.0, 3.0, 2.0, 3.0),),
        map_bounds=(0.0, 6.0, 0.0, 6.0),
    )

    assert route is not None
    assert route.points[-1] == (0.0, 0.0)
    assert impossible is None


def test_every_route_forces_home_even_when_home_is_not_last_requested_stop() -> None:
    planner = AStarRoutePlanner(grid_size_m=1.0)

    route = planner.plan(
        (1.0, 1.0),
        ((6.0, 1.0), (6.0, 6.0), (1.0, 6.0)),
        (1.0, 1.0),
        forbidden_regions=(),
        map_bounds=(0.0, 8.0, 0.0, 8.0),
    )

    assert route is not None
    assert route.points[-1] == (1.0, 1.0)
    assert route.stop_points == ((6.0, 1.0), (6.0, 6.0), (1.0, 6.0))
