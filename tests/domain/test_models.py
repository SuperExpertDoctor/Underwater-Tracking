import pytest
from pydantic import ValidationError
from underwater_tracking.domain.models import (
    BearingObservation,
    CarrierState,
    DeploymentState,
    SituationSnapshot,
    UUVState,
)


def test_bearing_rejects_unknown_fields_and_normalizes_angle():
    observation = BearingObservation(
        observation_id="O1", scenario_id="S1", sim_time_s=30,
        uuv_id="U1", target_id="T1", azimuth_rad=7.0,
        variance_rad2=0.01, detection_confidence=0.9,
    )
    assert -3.141592653589793 <= observation.azimuth_rad < 3.141592653589793
    with pytest.raises(ValidationError):
        BearingObservation(**observation.model_dump(), target_truth=[1.0, 2.0])


def test_operational_snapshot_has_no_truth_field():
    assert "truth" not in SituationSnapshot.model_fields
    assert "true_targets" not in SituationSnapshot.model_fields


def test_carrier_and_deployment_state_round_trip():
    carrier = CarrierState(
        carrier_id="carrier-01",
        position_xy=(-3000.0, -3000.0),
        heading_rad=0.25,
        speed_mps=1.5,
        status="recovering",
        onboard_uuv_ids=("uuv_03",),
        deployed_uuv_ids=("uuv_01",),
        returning_uuv_ids=("uuv_02",),
    )
    restored = CarrierState.model_validate_json(carrier.model_dump_json())
    assert restored == carrier


def test_old_uuv_and_snapshot_payloads_get_compatible_defaults():
    uuv = UUVState.model_validate({
        "uuv_id": "uuv_01",
        "position_xy": [0.0, 0.0],
        "heading_rad": 0.0,
        "speed_mps": 1.0,
        "energy_fraction": 0.9,
        "status": "available",
    })
    assert uuv.deployment_state == DeploymentState.DEPLOYED
    assert SituationSnapshot.model_validate({
        "scenario_id": "scenario-1",
        "snapshot_revision": 1,
        "sim_time_s": 30,
        "uuvs": [uuv.model_dump()],
        "group_reports": [],
        "pending_events": [],
    }).carrier is None
