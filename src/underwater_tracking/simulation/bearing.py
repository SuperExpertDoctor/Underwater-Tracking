# src/underwater_tracking/simulation/bearing.py
from math import atan2, pi
from underwater_tracking.domain.models import BearingObservation


def make_bearing_observation(*, scenario_id, sim_time_s, uuv_id, uuv_xy, target_id,
                             target_xy, variance_rad2, rng):
    truth = atan2(target_xy[1] - uuv_xy[1], target_xy[0] - uuv_xy[0])
    measured = (truth + rng.normalvariate(0.0, variance_rad2 ** 0.5) + pi) % (2 * pi) - pi
    return BearingObservation(
        observation_id=f"{target_id}:{uuv_id}:{sim_time_s}", scenario_id=scenario_id,
        sim_time_s=sim_time_s, uuv_id=uuv_id, target_id=target_id,
        azimuth_rad=measured, variance_rad2=variance_rad2, detection_confidence=1.0,
    )
