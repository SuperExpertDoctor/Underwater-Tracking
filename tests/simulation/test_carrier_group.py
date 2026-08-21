from __future__ import annotations

from underwater_tracking.planning.astar import AStarRoutePlanner
from underwater_tracking.simulation.carrier_group import (
    CommittedServiceStop,
    carrier_slot_position,
    solve_moving_rendezvous,
)


def test_carrier_slot_position_rotates_with_leader_heading() -> None:
    assert carrier_slot_position((10.0, 20.0), 0.0, (3.0, 4.0)) == (13.0, 24.0)
    assert carrier_slot_position((10.0, 20.0), 1.5707963267948966, (3.0, 4.0)) == (
        6.0,
        23.0,
    )
    assert carrier_slot_position((10.0, 20.0), 3.141592653589793, (3.0, 4.0)) == (
        7.0,
        16.0,
    )


def test_moving_rendezvous_converges_and_is_repeatable() -> None:
    kwargs = {
        "start_xy": (0.0, 0.0),
        "current_time_s": 0,
        "committed_stops": (
            CommittedServiceStop(point_xy=(2.0, 0.0), earliest_s=0, latest_s=20),
        ),
        "mother_speed_mps": 1.0,
        "project_slot_at": lambda eta_s: (8.0 + min(2.0, eta_s * 0.1), 0.0),
        "route_planner": AStarRoutePlanner(grid_size_m=1.0),
        "forbidden_regions": (),
        "map_bounds": (-1.0, 20.0, -1.0, 5.0),
        "tolerance_m": 0.25,
    }

    first = solve_moving_rendezvous(**kwargs)
    second = solve_moving_rendezvous(**kwargs)

    assert first is not None
    assert first == second
    assert first.route.points[0] == (0.0, 0.0)
    assert first.route.points[-1][0] <= 10.0
    projected = kwargs["project_slot_at"](first.eta_s)
    assert abs(first.route.points[-1][0] - projected[0]) <= 0.25
    assert first.route.stop_points == ((2.0, 0.0),)


def test_moving_rendezvous_waits_for_earliest_and_rejects_latest() -> None:
    common = {
        "start_xy": (0.0, 0.0),
        "current_time_s": 0,
        "mother_speed_mps": 1.0,
        "project_slot_at": lambda _eta_s: (5.0, 0.0),
        "route_planner": AStarRoutePlanner(grid_size_m=1.0),
        "forbidden_regions": (),
        "map_bounds": (-1.0, 10.0, -1.0, 5.0),
        "tolerance_m": 0.1,
    }
    waiting = solve_moving_rendezvous(
        **common,
        committed_stops=(
            CommittedServiceStop(point_xy=(2.0, 0.0), earliest_s=5, latest_s=10),
        ),
    )
    rejected = solve_moving_rendezvous(
        **common,
        committed_stops=(
            CommittedServiceStop(point_xy=(2.0, 0.0), earliest_s=5, latest_s=1),
        ),
    )

    assert waiting is not None
    assert waiting.eta_s >= 8
    assert rejected is None


def test_moving_rendezvous_honors_map_and_forbidden_regions() -> None:
    planner = AStarRoutePlanner(grid_size_m=1.0)
    blocked = solve_moving_rendezvous(
        start_xy=(0.0, 0.0),
        current_time_s=0,
        committed_stops=(),
        mother_speed_mps=1.0,
        project_slot_at=lambda _eta_s: (5.0, 0.0),
        route_planner=planner,
        forbidden_regions=((-0.5, 5.5, -2.0, 2.0),),
        map_bounds=(-1.0, 6.0, -2.0, 2.0),
        tolerance_m=0.1,
    )
    outside_map = solve_moving_rendezvous(
        start_xy=(0.0, 0.0),
        current_time_s=0,
        committed_stops=(),
        mother_speed_mps=1.0,
        project_slot_at=lambda _eta_s: (50.0, 0.0),
        route_planner=planner,
        forbidden_regions=(),
        map_bounds=(-1.0, 6.0, -2.0, 2.0),
        tolerance_m=0.1,
    )

    assert blocked is None
    assert outside_map is None


def test_moving_rendezvous_returns_none_after_bounded_non_convergence() -> None:
    projections = iter(((10.0, 0.0), (0.0, 0.0), (10.0, 0.0)))
    result = solve_moving_rendezvous(
        start_xy=(0.0, 0.0),
        current_time_s=0,
        committed_stops=(),
        mother_speed_mps=1.0,
        project_slot_at=lambda _eta_s: next(projections),
        route_planner=AStarRoutePlanner(grid_size_m=1.0),
        forbidden_regions=(),
        map_bounds=(-1.0, 20.0, -1.0, 2.0),
        tolerance_m=0.01,
        max_iterations=2,
    )

    assert result is None
