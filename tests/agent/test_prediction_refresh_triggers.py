import pytest

from underwater_tracking.cli import _event_requests_planning_epoch
from underwater_tracking.domain.models import EventLevel, RuntimeEvent


@pytest.mark.parametrize(
    "event_type",
    (
        "target_estimate_updated",
        "target_maneuver_observed",
        "target_speed_regime_changed",
        "target_depth_regime_changed",
        "imm_confidence_shifted",
        "target_intent_change_suspected",
    ),
)
def test_public_target_observation_requests_prediction_refresh_without_plan_impact(
    event_type: str,
) -> None:
    payload = {
        "observation_ids": ("obs-1",),
        "source": "fused_public_estimate",
        "plan_impact": False,
    }
    if event_type == "target_intent_change_suspected":
        payload.update(
            {
                "diff_id": "diff-1",
                "previous_prediction_id": "prediction-0",
                "current_prediction_id": "prediction-1",
                "absolute_rms_m": 300.0,
                "normalized_rms": 3.0,
                "absolute_floor_m": 250.0,
                "normalized_threshold": 2.45,
                "consecutive_count": 2,
                "exceeded": True,
            }
        )
    event = RuntimeEvent(
        event_id=f"{event_type}:refresh:1",
        scenario_id="S1",
        sim_time_s=60,
        event_type=event_type,
        entity_id="T1",
        level=EventLevel.TACTICAL,
        payload=payload,
    )

    assert _event_requests_planning_epoch(event) is True
