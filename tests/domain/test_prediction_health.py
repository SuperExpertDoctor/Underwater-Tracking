import pytest
from pydantic import ValidationError

from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.prediction_models import (
    AcceptedPrediction,
    PredictionHealth,
)


def test_prediction_health_accepts_a_valid_imm_result() -> None:
    health = PredictionHealth(
        status="valid",
        regime="imm",
        reason_codes=(),
        source_track_age_s=10.0,
        clipped_point_fraction=0.0,
        maximum_radius_m=900.0,
        raw_prediction_id="prediction-7",
    )
    assert health.status == "valid"


def test_prediction_health_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        PredictionHealth(
            status="unknown",
            regime="imm",
            reason_codes=(),
            source_track_age_s=10.0,
            clipped_point_fraction=0.0,
            maximum_radius_m=900.0,
        )


def test_prediction_health_rejects_out_of_bounds_metrics() -> None:
    with pytest.raises(ValidationError):
        PredictionHealth(
            status="degraded",
            regime="short_history",
            reason_codes=(),
            source_track_age_s=-1.0,
            clipped_point_fraction=1.01,
            maximum_radius_m=-1.0,
        )


def test_unavailable_prediction_can_omit_a_payload() -> None:
    health = PredictionHealth(
        status="unavailable",
        regime="boundary_recovery",
        reason_codes=("outside_boundary",),
        source_track_age_s=10.0,
        clipped_point_fraction=0.0,
        maximum_radius_m=900.0,
    )

    accepted = AcceptedPrediction(prediction=None, health=health)

    assert accepted.prediction is None


def test_unavailable_prediction_rejects_a_payload() -> None:
    health = PredictionHealth(
        status="unavailable",
        regime="boundary_recovery",
        reason_codes=("outside_boundary",),
        source_track_age_s=10.0,
        clipped_point_fraction=0.0,
        maximum_radius_m=900.0,
    )
    prediction = PredictedTrackRef(
        prediction_id="prediction-7",
        target_id="target-7",
        sim_time_s=10,
        horizon_s=60.0,
        sample_step_s=30.0,
    )

    with pytest.raises(ValueError, match="unavailable prediction cannot carry a payload"):
        AcceptedPrediction(prediction=prediction, health=health)


def test_accepted_prediction_requires_payload_unless_unavailable() -> None:
    health = PredictionHealth(
        status="valid",
        regime="imm",
        reason_codes=(),
        source_track_age_s=10.0,
        clipped_point_fraction=0.0,
        maximum_radius_m=900.0,
        raw_prediction_id="prediction-7",
    )
    with pytest.raises(ValueError, match="valid prediction requires"):
        AcceptedPrediction(prediction=None, health=health)
