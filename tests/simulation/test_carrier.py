from math import atan2, pi

import pytest

from underwater_tracking.domain.models import CarrierStatus, UUVState
from underwater_tracking.simulation.carrier import CarrierEntity


def _uuv(uuv_id: str, deployment_state: str) -> UUVState:
    return UUVState(
        uuv_id=uuv_id,
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
        speed_mps=1.0,
        energy_fraction=0.9,
        status=(
            "returning" if deployment_state == "returning"
            else "failed" if deployment_state == "failed"
            else "available"
        ),
        deployment_state=deployment_state,
    )


def test_carrier_patrol_is_deterministic_and_moves() -> None:
    left = CarrierEntity()
    right = CarrierEntity()
    left.step(30.0)
    right.step(30.0)
    assert left.state_for(()) == right.state_for(())
    assert left.state_for(()).position_xy != (-3000.0, -3000.0)


def test_carrier_clamps_negative_elapsed_time() -> None:
    carrier = CarrierEntity()
    before = carrier.state_for(())

    carrier.step(-30.0)

    assert carrier.state_for(()) == before


def test_carrier_reflects_onto_next_leg_after_crossing_a_corner() -> None:
    carrier = CarrierEntity()

    carrier.step(1201.0)

    assert carrier.position_xy[0] > 3000.0
    assert carrier.position_xy[1] > -3000.0
    assert carrier.heading_rad == pytest.approx(0.25)


def test_carrier_heading_change_is_bounded_at_a_corner() -> None:
    carrier = CarrierEntity(max_turn_rate_rad_s=0.1)
    carrier.step(1201.0)

    assert abs(carrier.heading_rad) <= 0.1 + 1e-9


def test_carrier_displacement_obeys_its_turn_limited_heading() -> None:
    carrier = CarrierEntity(
        position_xy=(0.0, 0.0),
        speed_mps=10.0,
        patrol_route_xy=((0.0, 0.0), (100.0, 0.0)),
        heading_rad=pi / 2.0,
        max_turn_rate_rad_s=0.1,
    )

    carrier.step(1.0)

    displacement_heading = atan2(carrier.position_xy[1], carrier.position_xy[0])
    assert displacement_heading == pytest.approx(carrier.heading_rad)


def test_carrier_route_installation_does_not_jump_heading() -> None:
    carrier = CarrierEntity(
        position_xy=(0.0, 0.0),
        speed_mps=10.0,
        patrol_route_xy=((0.0, 0.0), (100.0, 0.0)),
        max_turn_rate_rad_s=0.1,
        heading_rad=0.0,
    )

    carrier.set_mission_route(((0.0, 0.0), (0.0, 100.0), (0.0, 0.0)))

    assert carrier.heading_rad == pytest.approx(0.0)
    carrier.step(1.0)
    assert carrier.heading_rad == pytest.approx(0.1)


def test_carrier_patrol_projection_is_pure_across_corner_and_route_wrap() -> None:
    carrier = CarrierEntity()
    before = carrier.state_for(())

    projected_position, projected_heading = carrier.project_patrol_state(4801.0)

    assert carrier.state_for(()) == before
    carrier.step(4801.0)
    assert carrier.position_xy == projected_position
    assert carrier.heading_rad == projected_heading


def test_carrier_status_and_uuv_lists_follow_deployment_state() -> None:
    uuvs = (
        _uuv("uuv_01", "onboard"),
        _uuv("uuv_02", "deployed"),
        _uuv("uuv_03", "returning"),
        _uuv("uuv_04", "failed"),
    )
    state = CarrierEntity().state_for(uuvs)
    assert state.status == CarrierStatus.RECOVERING
    assert state.onboard_uuv_ids == ("uuv_01",)
    assert state.deployed_uuv_ids == ("uuv_02",)
    assert state.returning_uuv_ids == ("uuv_03",)


def test_carrier_reports_recovering_at_home_while_uuv_is_returning() -> None:
    carrier = CarrierEntity(
        position_xy=(0.0, 0.0),
        speed_mps=10.0,
        patrol_route_xy=((0.0, 0.0), (1.0, 1.0)),
        heading_rad=0.0,
    )
    carrier.set_mission_route(((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)))
    for _ in range(20):
        carrier.step(1.0)

    assert carrier.mission_route_complete is True
    assert carrier.state_for((_uuv("uuv_returning", "returning"),)).status is CarrierStatus.RECOVERING


def test_carrier_sorts_uuv_ids_within_each_deployment_state() -> None:
    state = CarrierEntity().state_for(
        (
            _uuv("uuv_09", "deployed"),
            _uuv("uuv_07", "returning"),
            _uuv("uuv_03", "onboard"),
            _uuv("uuv_01", "returning"),
            _uuv("uuv_04", "deployed"),
            _uuv("uuv_02", "onboard"),
        )
    )

    assert state.onboard_uuv_ids == ("uuv_02", "uuv_03")
    assert state.deployed_uuv_ids == ("uuv_04", "uuv_09")
    assert state.returning_uuv_ids == ("uuv_01", "uuv_07")


def test_carrier_can_follow_multi_stop_route_and_return_home() -> None:
    carrier = CarrierEntity(
        position_xy=(0.0, 0.0),
        speed_mps=10.0,
        patrol_route_xy=((0.0, 0.0), (1.0, 1.0)),
    )
    carrier.set_mission_route(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)))

    for _ in range(40):
        carrier.step(2.0)

    assert carrier.mission_route_xy[-1] == (0.0, 0.0)
    assert carrier.position_xy == (0.0, 0.0)
    assert carrier.mission_route_complete is True


def test_carrier_rejects_mission_route_that_does_not_return_home() -> None:
    carrier = CarrierEntity(
        position_xy=(0.0, 0.0),
        patrol_route_xy=((0.0, 0.0), (1.0, 1.0)),
    )

    try:
        carrier.set_mission_route(((0.0, 0.0), (10.0, 0.0)))
    except ValueError as exc:
        assert "home" in str(exc)
    else:
        raise AssertionError("expected mission route without home to be rejected")


def test_carrier_can_return_to_fixed_home_from_a_moving_start() -> None:
    carrier = CarrierEntity(
        position_xy=(10.0, 0.0),
        speed_mps=10.0,
        patrol_route_xy=((10.0, 0.0), (11.0, 1.0)),
    )
    carrier.set_mission_route(
        ((10.0, 0.0), (20.0, 0.0), (0.0, 0.0)),
        home_xy=(0.0, 0.0),
    )

    for _ in range(40):
        carrier.step(2.0)

    assert carrier.position_xy == (0.0, 0.0)
    assert carrier.mission_route_complete is True


def test_carrier_waits_for_service_window_before_releasing_stop() -> None:
    carrier = CarrierEntity(
        position_xy=(0.0, 0.0),
        speed_mps=10.0,
        patrol_route_xy=((0.0, 0.0), (1.0, 1.0)),
        heading_rad=0.0,
    )
    carrier.set_mission_route(
        ((0.0, 0.0), (10.0, 0.0), (0.0, 0.0)),
        stop_windows={1: (5, 20)},
    )

    carrier.step(1.0, sim_time_s=1)
    assert carrier.consume_arrived_mission_stop_indices() == ()
    assert carrier.position_xy == (10.0, 0.0)

    carrier.step(4.0, sim_time_s=5)
    assert carrier.consume_arrived_mission_stop_indices() == (1,)


def test_carrier_holds_external_stop_until_engine_releases_it() -> None:
    carrier = CarrierEntity(
        position_xy=(0.0, 0.0),
        speed_mps=10.0,
        patrol_route_xy=((0.0, 0.0), (1.0, 1.0)),
        heading_rad=0.0,
    )
    carrier.set_mission_route(
        ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0)),
        externally_released_stop_indices=frozenset({1}),
        rendezvous_xy=(20.0, 0.0),
    )

    carrier.step(1.0, sim_time_s=1)
    position = carrier.position_xy
    assert carrier.awaiting_release_stop_index == 1
    assert carrier.consume_arrived_mission_stop_indices() == (1,)

    carrier.step(5.0, sim_time_s=6)
    assert carrier.position_xy == position
    assert carrier.consume_arrived_mission_stop_indices() == ()

    carrier.release_mission_stop(1)
    assert carrier.awaiting_release_stop_index is None
    carrier.step(2.0, sim_time_s=8)
    assert carrier.position_xy == (20.0, 0.0)
    assert carrier.mission_route_complete is True


def test_carrier_rejects_wrong_external_release_index_and_invalid_route_tail() -> None:
    carrier = CarrierEntity(
        position_xy=(0.0, 0.0),
        speed_mps=1.0,
        patrol_route_xy=((0.0, 0.0), (1.0, 1.0)),
        heading_rad=0.0,
    )
    carrier.set_mission_route(
        ((0.0, 0.0), (5.0, 0.0), (10.0, 0.0)),
        externally_released_stop_indices=frozenset({1}),
        rendezvous_xy=(10.0, 0.0),
    )

    with pytest.raises(ValueError, match="externally released"):
        carrier.release_mission_stop(2)
    with pytest.raises(ValueError, match="current position"):
        carrier.replace_unfinished_return_segment(
            ((1.0, 0.0), (5.0, 0.0), (10.0, 0.0))
        )
    with pytest.raises(ValueError, match="omits"):
        carrier.replace_unfinished_return_segment(((0.0, 0.0), (10.0, 0.0)))


def test_carrier_route_tail_replacement_preserves_committed_stops() -> None:
    carrier = CarrierEntity(
        position_xy=(0.0, 0.0),
        speed_mps=1.0,
        patrol_route_xy=((0.0, 0.0), (1.0, 1.0)),
        heading_rad=0.0,
    )
    carrier.set_mission_route(
        ((0.0, 0.0), (5.0, 0.0), (10.0, 0.0)),
        externally_released_stop_indices=frozenset({1}),
        rendezvous_xy=(10.0, 0.0),
    )
    carrier.step(5.0, sim_time_s=5)
    before = carrier.position_xy

    carrier.replace_unfinished_return_segment(
        ((5.0, 0.0), (5.0, 2.0), (10.0, 0.0))
    )

    assert carrier.position_xy == before
    assert carrier.remaining_committed_stops() == ((5.0, 0.0),)
