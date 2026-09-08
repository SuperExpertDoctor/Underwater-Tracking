import pytest

from underwater_tracking.domain.models import TargetBelief
from underwater_tracking.tracking.public_estimate import assess_public_estimate
from underwater_tracking.tracking.region_probability import public_region_probability


def belief(**updates):
    return TargetBelief(
        target_id="T",
        sim_time_s=30,
        mean=(0.0, 0.0, 0.0, 0.0, 0.0),
        covariance=((100.0, 0.0), (0.0, 100.0)),
        model_probabilities={"cv": 1.0},
        source_observation_ids=("O1",),
        track_revision=1,
        last_observed_at_s=30,
        valid_until_s=930,
        accepted_observation_ids_this_cycle=("O1",),
    ).model_copy(update=updates)


def probability(value, now=30):
    return public_region_probability(
        belief=value,
        now_s=now,
        polygon_xy=((-100.0, -100.0), (100.0, -100.0), (100.0, 100.0), (-100.0, 100.0)),
    )


def test_probability_retains_auditable_source_and_ages_without_restamping():
    fresh = probability(belief())
    old = probability(belief(sim_time_s=600, accepted_observation_ids_this_cycle=()), 600)
    assert fresh["probability"] > 0.99 and fresh["eligible_for_confirmation"]
    assert old["status"] == "degraded" and old["source_age_s"] == 570
    assert old["source_track_revision"] == 1 and not old["eligible_for_confirmation"]
    expired = probability(belief(), 930)
    assert expired["status"] == "expired" and expired["probability"] is None


@pytest.mark.parametrize(
    "updates",
    [
        {"covariance": ((1.0, 2.0), (2.0, 1.0))},
        {"covariance": ((1.0, 0.0), (2.0, 1.0))},
        {"covariance": ((float("nan"), 0.0), (0.0, 1.0))},
        {"mean": (float("inf"), 0.0)},
        {"last_observed_at_s": None},
        {"track_revision": 0},
        {"source_observation_ids": ()},
        {"last_observed_at_s": 40},
        {"valid_until_s": 10},
    ],
)
def test_invalid_source_is_unavailable_not_zero(updates):
    result = probability(belief(**updates))
    assert result["status"] == "unavailable"
    assert result["probability"] is None and result["reason_codes"]


def test_real_outside_region_zero_is_not_unavailable():
    result = probability(belief(mean=(1000.0, 1000.0, 0.0, 0.0, 0.0)))
    assert result["probability"] < 1e-10 and result["status"] == "current"


def test_future_publication_time_is_invalid():
    assert assess_public_estimate(belief(), 10).status == "unavailable"
