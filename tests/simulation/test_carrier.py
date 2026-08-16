from math import pi

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
        status="available",
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

    assert carrier.position_xy == pytest.approx((3000.0, -2995.0))
    assert carrier.heading_rad == pytest.approx(pi / 2.0)


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
