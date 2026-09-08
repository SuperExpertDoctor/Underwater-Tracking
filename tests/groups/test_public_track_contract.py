from math import atan2
import numpy as np
import pytest

from underwater_tracking.domain.models import BearingObservation
from underwater_tracking.groups.manager import GroupManager
from underwater_tracking.tracking.initialization import initialize_from_bearings


POSITIONS = {"U1": (0.0, 0.0), "U2": (1000.0, 0.0)}


def observations(time=30, suffix=""):
    return tuple(
        BearingObservation(
            observation_id=f"O{i}{suffix}",
            scenario_id="S",
            target_id="T",
            uuv_id=u,
            sim_time_s=time,
            azimuth_rad=atan2(500.0, 500.0 - xy[0]),
            variance_rad2=1e-3,
            detection_confidence=1.0,
        )
        for i, (u, xy) in enumerate(POSITIONS.items())
    )


def manager():
    value = GroupManager()
    value.create(
        "T",
        scenario_id="S",
        group_id="G",
        member_ids=tuple(POSITIONS),
        coarse_prior=(500.0, 500.0),
        member_positions=POSITIONS,
    )
    return value


def test_predict_only_keeps_source_identity_not_freshness():
    value = manager()
    fresh = value.invoke("T", observations=observations(), sim_time_s=30)
    aged = value.invoke("T", sim_time_s=600)
    assert fresh.belief.track_revision == 1
    assert aged.belief.track_revision == 1
    assert aged.belief.last_observed_at_s == 30
    assert aged.belief.source_observation_ids == fresh.belief.source_observation_ids
    assert aged.belief.accepted_observation_ids_this_cycle == ()
    assert aged.belief.sim_time_s == 600


def test_duplicate_or_out_of_sequence_observations_do_not_refresh_track():
    value = manager()
    fresh = value.invoke("T", observations=observations(), sim_time_s=30)
    value.invoke("T", sim_time_s=600)
    replay = value.invoke("T", observations=observations(), sim_time_s=630)
    assert replay.belief.track_revision == fresh.belief.track_revision
    assert replay.belief.last_observed_at_s == 30
    assert replay.belief.accepted_observation_ids_this_cycle == ()
    older = value.invoke("T", observations=observations(20, "late"), sim_time_s=660)
    assert older.belief.last_observed_at_s == 30
    assert "out_of_sequence_observation" in older.belief.reason_codes


def test_new_observation_recovers_and_advances_revision():
    value = manager()
    value.invoke("T", observations=observations(), sim_time_s=30)
    value.invoke("T", sim_time_s=960)
    recovered = value.invoke("T", observations=observations(990, "new"), sim_time_s=990)
    assert recovered.belief.track_revision == 2
    assert recovered.belief.last_observed_at_s == 990
    assert recovered.belief.valid_until_s > 990


def test_same_observation_sequence_is_deterministic():
    results = []
    for _ in range(2):
        value = manager()
        value.invoke("T", observations=observations(), sim_time_s=30)
        results.append(value.invoke("T", sim_time_s=60).belief.model_dump_json())
    assert results[0] == results[1]


def test_triangulation_inputs_are_not_applied_a_second_time():
    obs = observations()
    initialized = initialize_from_bearings(
        np.asarray(tuple(POSITIONS.values())),
        np.asarray([o.azimuth_rad for o in obs]),
        np.asarray([o.variance_rad2 for o in obs]),
        prior=np.asarray((500.0, 500.0)),
    )
    report = _initialize_with_observations(obs, 30)
    np.testing.assert_allclose(
        np.asarray(report.belief.covariance)[:2, :2], initialized.covariance_xy, atol=1e-10
    )
    assert report.belief.accepted_observation_ids_this_cycle == tuple(o.observation_id for o in obs)


def test_delayed_bearings_use_observer_position_at_measurement_time():
    old_positions = {key: (x + 250.0, y - 250.0) for key, (x, y) in POSITIONS.items()}
    obs = tuple(
        o.model_copy(
            update={
                "observer_position_xy": old_positions[o.uuv_id],
                "azimuth_rad": atan2(
                    500.0 - old_positions[o.uuv_id][1], 500.0 - old_positions[o.uuv_id][0]
                ),
            }
        )
        for o in observations()
    )
    report = _initialize_with_observations(obs, 60)
    assert report.belief.mean[:2] == pytest.approx((500.0, 500.0), abs=1e-6)
    assert report.belief.last_observed_at_s == 30 and report.belief.sim_time_s == 60


def test_non_finite_observer_position_is_rejected_without_refreshing_source():
    value = manager()
    value.invoke("T", observations=observations(), sim_time_s=30)
    invalid = tuple(
        o.model_copy(update={"observer_position_xy": (float("nan"), 0.0)})
        for o in observations(60, "invalid")
    )
    report = value.invoke("T", observations=invalid, sim_time_s=60)
    assert report.belief.track_revision == 1 and report.belief.last_observed_at_s == 30
    assert "invalid_observation" in report.belief.reason_codes


def _initialize_with_observations(obs, now_s):
    return GroupManager().create(
        "T",
        scenario_id="S",
        group_id="G",
        member_ids=tuple(POSITIONS),
        coarse_prior=(500.0, 500.0),
        member_positions=POSITIONS,
        initial_observations=obs,
        initial_sim_time_s=now_s,
    )
