from __future__ import annotations

import random
from pathlib import Path

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AppConfig
from underwater_tracking.domain.platforms import SonarCapability
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.simulation.sonar import (
    SonarNode,
    default_pd_curve,
    make_passive_observation,
    make_passive_observations,
)

SCENARIO_PATH = Path("configs/scenario/segmented_single_target.yaml")


CAPABILITY = SonarCapability(
    passive_range_m=5_000.0,
    passive_bearing_variance_rad2=0.01,
    active_source_range_m=4_000.0,
    active_receive_range_m=5_000.0,
    active_range_sigma_m=15.0,
    active_bearing_sigma_rad=0.003,
    active_capable=True,
    ping_cooldown_s=30,
    ping_energy_cost_fraction=0.001,
    clutter_sensitivity=0.4,
    exposure_cost=0.4,
)


def test_default_pd_curve_decreases_with_range() -> None:
    values = [default_pd_curve(fraction) for fraction in (0.0, 0.25, 0.5, 1.0)]

    assert values == sorted(values, reverse=True)
    assert values[0] == 0.95
    assert values[-1] == 0.40


def test_detection_gate_is_deterministic_and_does_not_consume_measurement_rng() -> None:
    observer = SonarNode("uuv_00", (0.0, 0.0), CAPABILITY)
    measurement_rng = random.Random(17)
    expected_measurement_rng = random.Random(17)
    detection_rng = random.Random(3)
    expected_detection_rng = random.Random(3)

    observation = make_passive_observation(
        scenario_id="scenario",
        sim_time_s=30,
        observer=observer,
        target_id="target_00",
        target_xy=(1_000.0, 0.0),
        rng=measurement_rng,
        detection_rng=detection_rng,
        pd_curve=lambda _: 1.0,
    )

    assert observation is not None
    expected_measurement_rng.gauss(0.0, CAPABILITY.passive_bearing_variance_rad2**0.5)
    assert measurement_rng.getstate() == expected_measurement_rng.getstate()
    expected_detection_rng.random()
    assert detection_rng.getstate() == expected_detection_rng.getstate()


def test_pd_curve_can_force_a_miss_without_consuming_measurement_rng() -> None:
    observer = SonarNode("uuv_00", (0.0, 0.0), CAPABILITY)
    measurement_rng = random.Random(17)
    before = measurement_rng.getstate()

    observation = make_passive_observation(
        scenario_id="scenario",
        sim_time_s=30,
        observer=observer,
        target_id="target_00",
        target_xy=(1_000.0, 0.0),
        rng=measurement_rng,
        detection_rng=random.Random(1),
        pd_curve=lambda _: 0.0,
    )

    assert observation is None
    assert measurement_rng.getstate() == before


def test_clutter_sensitivity_controls_false_alarm_rate_and_marks_observations() -> None:
    observer = SonarNode("uuv_00", (0.0, 0.0), CAPABILITY)
    false_alarm_count = 0
    total = 1_000
    for seed in range(total):
        observations = make_passive_observations(
            scenario_id="scenario",
            sim_time_s=30,
            observer=observer,
            target_id="target_00",
            target_xy=(1_000.0, 0.0),
            rng=random.Random(seed),
            detection_rng=random.Random(seed + 10_000),
            clutter_rng=random.Random(seed + 20_000),
            clutter_sensitivity=CAPABILITY.clutter_sensitivity,
            pd_curve=lambda _: 1.0,
        )
        false_alarm_count += sum(item.is_false_alarm for item in observations)

    observed_rate = false_alarm_count / total
    assert 0.35 <= observed_rate <= 0.45


def test_false_alarm_is_not_forwarded_to_estimator_input() -> None:
    observer = SonarNode("uuv_00", (0.0, 0.0), CAPABILITY)
    observations = make_passive_observations(
        scenario_id="scenario",
        sim_time_s=30,
        observer=observer,
        target_id="target_00",
        target_xy=(1_000.0, 0.0),
        rng=random.Random(1),
        detection_rng=random.Random(2),
        clutter_rng=random.Random(3),
        clutter_sensitivity=1.0,
        pd_curve=lambda _: 0.0,
    )

    false_alarms = tuple(item for item in observations if item.is_false_alarm)
    estimator_inputs = tuple(item for item in observations if not item.is_false_alarm)
    assert false_alarms
    assert all(item.target_id.startswith("clutter:") for item in false_alarms)
    assert estimator_inputs == ()


def test_engine_keeps_false_alarms_out_of_target_filter_updates(tmp_path) -> None:
    data = load_app_config(SCENARIO_PATH).model_dump()
    for profile in data["sensors"]["profiles"].values():
        profile["clutter_sensitivity"] = 1.0
    config = AppConfig.model_validate(data)
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)

    frame = {}
    for _ in range(6):
        frame = engine.step()

    false_alarms = [
        item for item in frame["sonar_observations"] if item["is_false_alarm"]
    ]
    target_contact = next(
        item for item in frame["contacts"] if item["contact_id"] == "target_00"
    )
    assert false_alarms
    assert all(not item["is_false_alarm"] for item in target_contact["bearing_rays"])
