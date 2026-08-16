import pytest
from pydantic import ValidationError

from underwater_tracking.domain.observations import (
    ActiveTransmission,
    MultistaticObservation,
    PassiveSonarObservation,
)


def test_observation_contracts_use_generic_platform_ids() -> None:
    passive = PassiveSonarObservation(
        observation_id="passive:usv_00:target_00:30",
        scenario_id="single-target-relay",
        sim_time_s=30,
        observer_id="usv_00",
        target_id="target_00",
        azimuth_rad=0.2,
        variance_rad2=0.01,
        detection_confidence=0.8,
        snr_db=6.0,
    )
    transmission = ActiveTransmission(
        transmission_id="ping:usv_00:target_00:30",
        scenario_id="single-target-relay",
        sim_time_s=30,
        emitter_id="usv_00",
        target_id="target_00",
    )
    active = MultistaticObservation(
        observation_id="active:usv_00:uuv_00:target_00:30",
        transmission_id=transmission.transmission_id,
        scenario_id="single-target-relay",
        sim_time_s=30,
        emitter_id="usv_00",
        receiver_id="uuv_00",
        target_id="target_00",
        bistatic_range_m=3000.0,
        receiver_azimuth_rad=0.3,
        range_variance_m2=225.0,
        bearing_variance_rad2=9e-6,
        detection_confidence=0.9,
    )

    assert passive.observer_id == "usv_00"
    assert active.receiver_id == "uuv_00"
    assert "position" not in active.model_dump()


def test_observations_reject_non_finite_measurements() -> None:
    with pytest.raises(ValidationError):
        PassiveSonarObservation(
            observation_id="bad",
            scenario_id="scenario",
            sim_time_s=0,
            observer_id="uuv_00",
            target_id="target_00",
            azimuth_rad=float("nan"),
            variance_rad2=0.01,
            detection_confidence=1.0,
            snr_db=0.0,
        )
