from __future__ import annotations

from math import hypot

import pytest

from underwater_tracking.agent.graphs.central import _prior_seeded_planning_inputs
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.simulation.engine import SimulationEngine


def test_public_prior_becomes_a_bounded_temporal_search_envelope() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(
        config,
        seed=7,
    )
    situation = engine.publication_situation()
    prior = situation.target_search_priors[0]

    state = _prior_seeded_planning_inputs(situation)
    prediction = state["predictions"][prior.target_id]

    assert prediction.fallback_used is True
    assert prediction.fallback_reason == "public_target_search_envelope"
    assert prediction.prediction_regime == "public_prior"
    assert prediction.imm_model_probabilities == {}
    assert prediction.source_belief_history_ids == ()
    assert prediction.times_s == tuple(sorted(prediction.times_s))
    assert len(set(prediction.points_xy)) > 1
    assert prediction.corridor_radius_m[0] >= max(
        prior.covariance_xy[0][0] ** 0.5,
        prior.covariance_xy[1][1] ** 0.5,
    )
    assert prediction.corridor_radius_m[-1] == pytest.approx(
        prediction.corridor_radius_m[0]
        + config.tracking.submarine_sprint_speed_mps
        * (prediction.times_s[-1] - prediction.times_s[0])
    )

    # The fallback may search outward, but it may not outrun the configured
    # public target-speed bound or smuggle a private target trajectory in.
    for time_s, point in zip(prediction.times_s, prediction.points_xy):
        elapsed_s = time_s - situation.sim_time_s
        displacement = hypot(
            point[0] - prior.center_xy[0],
            point[1] - prior.center_xy[1],
        )
        assert displacement <= 14.0 * elapsed_s + 1e-6
