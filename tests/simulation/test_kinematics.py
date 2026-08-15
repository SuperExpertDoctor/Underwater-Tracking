import random
from math import pi

from underwater_tracking.simulation.target import HiddenIntent, TargetEntity
from underwater_tracking.simulation.uuv import UUVEntity


def test_uuv_respects_turn_rate_and_energy_monotonicity():
    uuv = UUVEntity("U1", (0.0, 0.0), 0.0, energy_fraction=1.0)
    uuv.set_waypoints([(0.0, 1000.0)])
    previous_energy = uuv.energy_fraction
    uuv.step(dt_s=10, max_speed_mps=3.0, max_turn_rate_rad_s=pi / 60)
    assert 0.0 < uuv.heading_rad <= pi / 6
    assert uuv.energy_fraction < previous_energy


def test_hidden_intent_is_not_exposed_by_public_state():
    target = TargetEntity("T1", (0.0, 0.0), (2.0, 0.0), HiddenIntent.TRANSIT)
    assert "intent" not in target.public_kinematics()


def test_seeded_targets_produce_identical_transitions():
    first = TargetEntity("T1", (0.0, 0.0), (2.0, 0.0), HiddenIntent.TRANSIT)
    second = TargetEntity("T1", (0.0, 0.0), (2.0, 0.0), HiddenIntent.TRANSIT)
    first_rng = random.Random(42)
    second_rng = random.Random(42)
    transitioned = False
    for _ in range(100):
        first.step(10.0, first_rng)
        second.step(10.0, second_rng)
        assert first.position_xy == second.position_xy
        assert first.velocity_xy == second.velocity_xy
        assert first.intent is second.intent
        transitioned = transitioned or first.intent is not HiddenIntent.TRANSIT
    assert transitioned


def test_target_positions_stay_inside_bounds_over_1000_seconds():
    bounds = (-100.0, 100.0, -100.0, 100.0)
    target = TargetEntity("T1", (0.0, 0.0), (2.0, 0.0), HiddenIntent.TRANSIT, bounds_xy=bounds)
    rng = random.Random(42)
    closest_to_bound = float("inf")
    for _ in range(100):
        target.step(10.0, rng)
        x, y = target.position_xy
        assert bounds[0] <= x <= bounds[1]
        assert bounds[2] <= y <= bounds[3]
        closest_to_bound = min(closest_to_bound, x - bounds[0], bounds[1] - x, y - bounds[2], bounds[3] - y)
    assert closest_to_bound < 30.0


import math

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.simulation.engine import SimulationEngine
from tests.conftest import CONFIG_PATH


def test_uuv_motion_respects_configured_max_speed(tmp_path):
    """R2: the configured UUV maximum speed caps actual motion."""
    base = load_app_config(CONFIG_PATH)
    config = base.model_copy(
        update={
            "tracking": base.tracking.model_copy(
                update={"uuv_max_speed_mps": 1.0}
            )
        }
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    speeds: list[float] = []
    for _ in range(12):
        frame = engine.step()
        speeds.extend(float(uuv["speed_mps"]) for uuv in frame["uuvs"])
    assert max(speeds) <= 1.0 + 1e-6


def test_submarine_cruise_outruns_uuv_max_speed(tmp_path):
    """R2: a cruising submarine is faster than the fastest UUV."""
    config = load_app_config(CONFIG_PATH)
    truths: list[dict[str, object]] = []
    engine = SimulationEngine(
        config, seed=42, output_dir=tmp_path, evaluation_sink=truths.append
    )
    for _ in range(3):
        engine.step()
    assert truths
    target = truths[-1]["targets"][0]
    vx, vy = target["velocity_xy"]
    assert math.hypot(vx, vy) >= config.tracking.submarine_cruise_speed_mps - 1e-6
    assert config.tracking.uuv_max_speed_mps < config.tracking.submarine_cruise_speed_mps
