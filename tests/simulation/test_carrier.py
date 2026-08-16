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
