import random
from math import hypot

import pytest

from underwater_tracking.domain.platforms import SonarCapability
from underwater_tracking.simulation.sonar import (
    SonarNode,
    make_multistatic_observations,
    make_passive_observation,
)


CAPABILITY = SonarCapability(
    passive_range_m=5000.0,
    passive_bearing_variance_rad2=0.01,
    active_source_range_m=4000.0,
    active_receive_range_m=5000.0,
    active_range_sigma_m=15.0,
    active_bearing_sigma_rad=0.003,
    active_capable=True,
    ping_cooldown_s=30,
    ping_energy_cost_fraction=0.001,
    clutter_sensitivity=0.2,
    exposure_cost=0.4,
)


def test_passive_observation_respects_detection_range() -> None:
    observer = SonarNode("uuv_00", (0.0, 0.0), CAPABILITY)
    near = make_passive_observation(
        scenario_id="scenario",
        sim_time_s=30,
        observer=observer,
        target_id="target_00",
        target_xy=(3000.0, 0.0),
        rng=random.Random(1),
    )
    far = make_passive_observation(
        scenario_id="scenario",
        sim_time_s=30,
        observer=observer,
        target_id="target_00",
        target_xy=(6000.0, 0.0),
        rng=random.Random(1),
    )

    assert near is not None
    assert far is None


def test_passive_quality_is_seeded_and_varies_between_seeds() -> None:
    observer = SonarNode("uuv_00", (0.0, 0.0), CAPABILITY)
    kwargs = {
        "scenario_id": "scenario",
        "sim_time_s": 30,
        "observer": observer,
        "target_id": "target_00",
        "target_xy": (100.0, 0.0),
    }

    first = make_passive_observation(**kwargs, rng=random.Random(1))
    repeated = make_passive_observation(**kwargs, rng=random.Random(1))
    different_seed = make_passive_observation(**kwargs, rng=random.Random(2))

    assert first is not None
    assert repeated == first
    assert different_seed is not None
    assert (
        different_seed.detection_confidence,
        different_seed.snr_db,
    ) != (
        first.detection_confidence,
        first.snr_db,
    )


def test_passive_public_quality_cannot_recover_exact_target_position() -> None:
    observers = (
        SonarNode("uuv_00", (0.0, 0.0), CAPABILITY),
        SonarNode("uuv_01", (1000.0, 0.0), CAPABILITY),
        SonarNode("uuv_02", (0.0, 1000.0), CAPABILITY),
    )
    target_xy = (1600.0, 2100.0)
    observations = tuple(
        make_passive_observation(
            scenario_id="scenario",
            sim_time_s=30,
            observer=observer,
            target_id="target_00",
            target_xy=target_xy,
            rng=random.Random(seed),
        )
        for observer, seed in zip(observers, (4, 5, 6), strict=True)
    )

    assert all(observation is not None for observation in observations)
    public_observations = tuple(
        observation for observation in observations if observation is not None
    )
    assert all(
        observation.snr_db
        != pytest.approx(20.0 * observation.detection_confidence - 10.0)
        for observation in public_observations
    )
    recovered_ranges = tuple(
        CAPABILITY.passive_range_m * (1.0 - observation.detection_confidence)
        for observation in public_observations
    )
    recovered_x = (recovered_ranges[0] ** 2 - recovered_ranges[1] ** 2 + 1000.0**2) / (
        2.0 * 1000.0
    )
    recovered_y = (recovered_ranges[0] ** 2 - recovered_ranges[2] ** 2 + 1000.0**2) / (
        2.0 * 1000.0
    )

    assert hypot(recovered_x - target_xy[0], recovered_y - target_xy[1]) > 1.0


def test_passive_quality_degrades_in_expectation_with_range() -> None:
    observer = SonarNode("uuv_00", (0.0, 0.0), CAPABILITY)
    near_observations = []
    far_observations = []
    for seed in range(64):
        near = make_passive_observation(
            scenario_id="scenario",
            sim_time_s=30,
            observer=observer,
            target_id="target_00",
            target_xy=(500.0, 0.0),
            rng=random.Random(seed),
        )
        far = make_passive_observation(
            scenario_id="scenario",
            sim_time_s=30,
            observer=observer,
            target_id="target_00",
            target_xy=(4500.0, 0.0),
            rng=random.Random(seed),
        )
        assert near is not None
        assert far is not None
        near_observations.append(near)
        far_observations.append(far)

    assert all(0.0 <= observation.detection_confidence <= 1.0 for observation in near_observations)
    assert all(0.0 <= observation.detection_confidence <= 1.0 for observation in far_observations)
    assert sum(observation.detection_confidence for observation in near_observations) > sum(
        observation.detection_confidence for observation in far_observations
    )
    assert sum(observation.snr_db for observation in near_observations) > sum(
        observation.snr_db for observation in far_observations
    )


def test_one_emitter_produces_observations_for_all_in_range_receivers() -> None:
    emitter = SonarNode("usv_00", (0.0, 0.0), CAPABILITY)
    receivers = (
        SonarNode("uuv_00", (2000.0, 0.0), CAPABILITY),
        SonarNode("uuv_01", (0.0, 2000.0), CAPABILITY),
        SonarNode("uuv_far", (9000.0, 0.0), CAPABILITY),
    )

    transmission, observations = make_multistatic_observations(
        scenario_id="scenario",
        sim_time_s=60,
        emitter=emitter,
        receivers=receivers,
        target_id="target_00",
        target_xy=(1000.0, 1000.0),
        rng=random.Random(5),
    )

    assert transmission.emitter_id == "usv_00"
    assert [observation.receiver_id for observation in observations] == ["uuv_00", "uuv_01"]
    assert all(observation.bistatic_range_m > 0.0 for observation in observations)


def test_multistatic_confidence_is_not_an_exact_receiver_range_formula() -> None:
    emitter = SonarNode("usv_00", (0.0, 0.0), CAPABILITY)
    receiver = SonarNode("uuv_00", (2000.0, 0.0), CAPABILITY)

    _, observations = make_multistatic_observations(
        scenario_id="scenario",
        sim_time_s=60,
        emitter=emitter,
        receivers=(receiver,),
        target_id="target_00",
        target_xy=(1000.0, 0.0),
        rng=random.Random(7),
    )

    assert len(observations) == 1
    assert observations[0].detection_confidence != pytest.approx(0.8)
