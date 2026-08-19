from math import inf

import pytest
from pydantic import ValidationError

from underwater_tracking.domain.observations import PassiveSonarObservation


def _passive_payload() -> dict[str, object]:
    return {
        "observation_id": "passive:usv_00:target_00:30",
        "scenario_id": "single-target-relay",
        "sim_time_s": 30,
        "observer_id": "usv_00",
        "target_id": "target_00",
        "azimuth_rad": 0.2,
        "variance_rad2": 0.01,
        "detection_confidence": 0.8,
        "snr_db": 6.0,
    }


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

    assert passive.observer_id == "usv_00"
    assert passive.is_false_alarm is False


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


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        (PassiveSonarObservation, "sim_time_s", "30"),
        (PassiveSonarObservation, "variance_rad2", "0.01"),
        (PassiveSonarObservation, "snr_db", "6.0"),
    ],
)
def test_observations_reject_coercible_numeric_strings(
    model: type[object], field: str, value: str
) -> None:
    payload = _passive_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        model(**payload)


@pytest.mark.parametrize("model", [PassiveSonarObservation])
def test_observations_reject_infinite_numeric_measurements(model: type[object]) -> None:
    payload = _passive_payload()
    payload["snr_db"] = inf

    with pytest.raises(ValidationError):
        model(**payload)


def test_observations_reject_unknown_fields_and_are_immutable() -> None:
    payload = _passive_payload()
    with pytest.raises(ValidationError, match="extra"):
        PassiveSonarObservation(**payload, unexpected="forbidden")

    observation = PassiveSonarObservation(**payload)
    with pytest.raises(ValidationError, match="frozen"):
        observation.snr_db = 7.0
