"""Command-line showcase for the rule-based future-event predictor.

Run from the repository root with ``PYTHONPATH=src``::

    python -m underwater_tracking.world_model.demo --scenario left_turn --pretty

The scenarios are deterministic operational inputs, not simulator truth.
They make the code-stage deliverable reviewable before a front end exists.
"""

from __future__ import annotations

import argparse
import json
from math import cos, sin
from pathlib import Path
from typing import Literal, cast

from underwater_tracking.world_model.config import (
    DEFAULT_WORLD_MODEL_CONFIG,
    load_world_model_config,
)
from underwater_tracking.world_model.models import (
    ImmBeliefInput,
    RuleWorldModelConfig,
    RuleWorldModelInput,
    TrackingContextInput,
    TrajectoryForecastInput,
    UuvForecastInput,
)
from underwater_tracking.world_model.rules import predict_future_events


ScenarioName = Literal[
    "normal",
    "left_turn",
    "sprint",
    "area_exit",
    "decoy",
    "geometry_bad",
    "coverage_gap",
    "track_loss",
    "stop",
]
SCENARIOS: tuple[ScenarioName, ...] = (
    "normal",
    "left_turn",
    "sprint",
    "area_exit",
    "decoy",
    "geometry_bad",
    "coverage_gap",
    "track_loss",
    "stop",
)


def build_demo_input(scenario: ScenarioName) -> RuleWorldModelInput:
    """Return one deterministic input that isolates a showcase event."""

    as_of_s = 300.0
    step_s = 30.0
    offsets = tuple(step_s * index for index in range(1, 61))
    times = tuple(as_of_s + offset for offset in offsets)
    speed = 8.0
    model_probabilities = {"cv": 0.80, "left_turn": 0.10, "right_turn": 0.10}
    velocity = (speed, 0.0)
    if scenario == "left_turn":
        omega = 0.0105
        radius = speed / omega
        points = tuple(
            (radius * sin(omega * offset), radius * (1.0 - cos(omega * offset)))
            for offset in offsets
        )
        model_probabilities = {"cv": 0.17, "left_turn": 0.78, "right_turn": 0.05}
    elif scenario == "sprint":
        speed = 12.5
        velocity = (speed, 0.0)
        points = tuple((speed * offset, 0.0) for offset in offsets)
    elif scenario == "stop":
        velocity = (0.0, 0.0)
        points = tuple((0.0, 0.0) for _ in offsets)
    else:
        points = tuple((speed * offset, 0.0) for offset in offsets)
    corridor = tuple(100.0 + 0.10 * offset for offset in offsets)
    map_bounds = (
        (-5000.0, 3200.0, -5000.0, 5000.0)
        if scenario == "area_exit"
        else (-30000.0, 30000.0, -30000.0, 30000.0)
    )
    uuvs = _formation_uuvs(times, points)
    if scenario == "geometry_bad":
        uuvs = _collinear_uuvs(times, points)
    elif scenario == "coverage_gap":
        uuvs = uuvs[:1]
    elif scenario == "track_loss":
        uuvs = tuple(uuv.model_copy(update={"communication_ok": False}) for uuv in uuvs)
    tracking = TrackingContextInput(
        quality_ewma=0.85 if scenario != "track_loss" else 0.30,
        current_contact_count=2 if scenario == "decoy" else 1,
        previous_contact_count=1 if scenario == "decoy" else None,
        association_confidence=0.55 if scenario == "decoy" else None,
        previous_association_confidence=0.90 if scenario == "decoy" else None,
        association_entropy=0.55 if scenario == "decoy" else None,
        previous_association_entropy=0.20 if scenario == "decoy" else None,
    )
    return RuleWorldModelInput(
        scenario_id=f"demo-{scenario}",
        target_id="submarine_01",
        as_of_s=as_of_s,
        belief=ImmBeliefInput(
            position_xy=(0.0, 0.0),
            velocity_xy_mps=velocity,
            turn_rate_rad_s=0.0105 if scenario == "left_turn" else 0.0,
            covariance_trace_m2=400.0,
            model_probabilities=model_probabilities,
        ),
        trajectory=TrajectoryForecastInput(
            prediction_id=f"demo-{scenario}-bspline",
            times_s=times,
            points_xy=points,
            corridor_radius_m=corridor,
        ),
        uuvs=uuvs,
        tracking=tracking,
        map_bounds_xy=map_bounds,
        source_observation_ids=("demo-observation-01",),
    )


def run_demo(
    scenario: ScenarioName,
    config: RuleWorldModelConfig = DEFAULT_WORLD_MODEL_CONFIG,
) -> dict[str, object]:
    """Run one scenario and return a JSON-compatible forecast mapping."""

    forecast = predict_future_events(build_demo_input(scenario), config)
    return cast(dict[str, object], forecast.model_dump(mode="json"))


def _formation_uuvs(
    times: tuple[float, ...],
    target_points: tuple[tuple[float, float], ...],
) -> tuple[UuvForecastInput, ...]:
    offsets = ((-1000.0, 0.0), (500.0, 866.0), (500.0, -866.0))
    return tuple(
        UuvForecastInput(
            uuv_id=f"uuv_{index:02d}",
            position_xy=offset,
            passive_range_m=4500.0,
            bearing_variance_rad2=0.008,
            energy_fraction=0.80,
            healthy=True,
            communication_ok=True,
            planned_times_s=times,
            planned_points_xy=tuple(
                (point[0] + offset[0], point[1] + offset[1])
                for point in target_points
            ),
        )
        for index, offset in enumerate(offsets)
    )


def _collinear_uuvs(
    times: tuple[float, ...],
    target_points: tuple[tuple[float, float], ...],
) -> tuple[UuvForecastInput, ...]:
    offsets = ((-1500.0, 0.0), (1000.0, 0.0), (2000.0, 0.0))
    return tuple(
        UuvForecastInput(
            uuv_id=f"uuv_{index:02d}",
            position_xy=offset,
            passive_range_m=4500.0,
            bearing_variance_rad2=0.008,
            energy_fraction=0.80,
            planned_times_s=times,
            planned_points_xy=tuple(
                (point[0] + offset[0], point[1] + offset[1])
                for point in target_points
            ),
        )
        for index, offset in enumerate(offsets)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "all"),
        default="all",
        help="deterministic event scenario to run",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="optional world_model_rules.yaml path",
    )
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    args = parser.parse_args()
    config = (
        load_world_model_config(args.config)
        if args.config is not None
        else DEFAULT_WORLD_MODEL_CONFIG
    )
    if args.scenario == "all":
        payload: object = {name: run_demo(name, config) for name in SCENARIOS}
    else:
        payload = run_demo(cast(ScenarioName, args.scenario), config)
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
