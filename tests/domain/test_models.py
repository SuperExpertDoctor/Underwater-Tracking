import pytest
from pydantic import ValidationError
from underwater_tracking.domain.models import BearingObservation, SituationSnapshot


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
