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
