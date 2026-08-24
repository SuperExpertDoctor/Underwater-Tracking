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
    ),
)
def test_public_target_observation_requests_prediction_refresh_without_plan_impact(
    event_type: str,
) -> None:
    event = RuntimeEvent(
        event_id=f"{event_type}:refresh:1",
        scenario_id="S1",
        sim_time_s=60,
        event_type=event_type,
        entity_id="T1",
        level=EventLevel.TACTICAL,
        payload={
            "observation_ids": ("obs-1",),
            "source": "fused_public_estimate",
            "plan_impact": False,
        },
    )

    assert _event_requests_planning_epoch(event) is True
