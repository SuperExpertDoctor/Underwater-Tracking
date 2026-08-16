import random

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
