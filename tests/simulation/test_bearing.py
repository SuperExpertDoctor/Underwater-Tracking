import random
from math import pi

from underwater_tracking.simulation.bearing import make_bearing_observation


def _observe(rng):
    return make_bearing_observation(
        scenario_id="s1", sim_time_s=30, uuv_id="u1", uuv_xy=(0.0, 0.0),
        target_id="t1", target_xy=(-1000.0, -1e-6), variance_rad2=1e-4, rng=rng,
    )


def test_identically_seeded_rngs_produce_equal_azimuths():
    first = _observe(random.Random(42))
    second = _observe(random.Random(42))
    assert first.azimuth_rad == second.azimuth_rad


def test_azimuth_is_wrapped_to_unit_circle():
    obs = _observe(random.Random(42))
    assert -pi <= obs.azimuth_rad < pi
