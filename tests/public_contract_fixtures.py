"""Sensor fixtures derived from public search priors, never simulator truth."""

from math import atan2


def prior_bearing(engine, uuv_id, sim_time_s=0):
    prior = engine._active_target_search_priors(sim_time_s)[0]
    observer = engine._uuvs[uuv_id].position_xy
    return atan2(prior.center_xy[1] - observer[1], prior.center_xy[0] - observer[0])
