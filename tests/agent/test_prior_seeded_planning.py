from __future__ import annotations

from math import floor, hypot

import pytest

from underwater_tracking.agent.graphs.central import _known_submarine_planning_inputs
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.regional_models import GridSpec
from underwater_tracking.simulation.engine import SimulationEngine


def test_known_submarine_contact_becomes_a_bounded_tracking_envelope() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(
        config,
        seed=7,
    )
    situation = engine.publication_situation()
    contact = next(
        contact
        for contact in situation.contacts
        if contact.classification.value == "submarine"
    )
    assert contact.estimated_position_xy is not None

    state = _known_submarine_planning_inputs(situation)
    prediction = state["predictions"][contact.contact_id]

    assert prediction.fallback_used is True
    assert prediction.fallback_reason == "known_submarine_contact"
    assert prediction.prediction_regime == "known_submarine"
    assert prediction.imm_model_probabilities == {}
    assert prediction.source_belief_history_ids == ()
    assert prediction.times_s == tuple(sorted(prediction.times_s))
    assert prediction.horizon_s == config.timing.prediction_horizon_s == 1_800
    assert len(set(prediction.points_xy)) > 1
    assert prediction.points_xy[0] == contact.estimated_position_xy
    assert hypot(
        prediction.points_xy[-1][0] - prediction.points_xy[0][0],
        prediction.points_xy[-1][1] - prediction.points_xy[0][1],
    ) >= 5_000.0
    assert prediction.corridor_radius_m[0] >= 250.0
    assert prediction.corridor_radius_m[-1] == pytest.approx(
        prediction.corridor_radius_m[0]
        + config.tracking.submarine_sprint_speed_mps
        * (prediction.times_s[-1] - prediction.times_s[0])
    )

    # The initial route is a bounded operational envelope from the public
    # contact, not a copy of the private target trajectory.
    for time_s, point in zip(prediction.times_s, prediction.points_xy):
        elapsed_s = time_s - situation.sim_time_s
        displacement = hypot(
            point[0] - contact.estimated_position_xy[0],
            point[1] - contact.estimated_position_xy[1],
        )
        assert displacement <= 4.0 * elapsed_s + 1e-6


def test_known_submarine_envelope_supports_four_in_bounds_task_cells() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    situation = SimulationEngine(config, seed=7).publication_situation()
    prediction = next(
        iter(_known_submarine_planning_inputs(situation)["predictions"].values())
    )
    origin_x, origin_y = GridSpec().origin_xy
    map_min_x, map_max_x, map_min_y, map_max_y = config.environment.map_bounds_xy
    task_cells = {
        (
            floor((point[0] - origin_x) / 1_000.0),
            floor((point[1] - origin_y) / 1_000.0),
        )
        for point in prediction.points_xy
        if map_min_x <= point[0] <= map_max_x and map_min_y <= point[1] <= map_max_y
    }

    assert len(task_cells) >= 4
