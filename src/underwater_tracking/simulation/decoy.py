# src/underwater_tracking/simulation/decoy.py
"""Passive-sonar decoy entity (spec 5.1 amendment, R5).

A decoy drifts slowly with a heading random walk and is indistinguishable
from a submarine to passive sonar: the engine emits the same bearing
observations for it. Its true nature stays truth-side (``_truth`` only);
the operational ``ContactClassification`` of a contact comes exclusively
from active-sonar pings, never from the truth.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from underwater_tracking.domain.platforms import MotionLimits
from underwater_tracking.simulation.kinematics import (
    MotionCommand,
    MotionState,
    NavigationBoundary,
    advance_motion,
    constrain_navigation_command,
    wrap_angle,
)


_DECOY_MAX_TURN_RATE_RAD_S = 0.25


@dataclass(slots=True)
class DecoyEntity:
    decoy_id: str
    position_xy: tuple[float, float]
    heading_rad: float
    drift_speed_mps: float
    heading_noise_rad_per_s: float
    max_turn_rate_rad_s: float = _DECOY_MAX_TURN_RATE_RAD_S
    boundary: NavigationBoundary | None = None

    def step(self, dt_s: float, rng: random.Random) -> None:
        state = MotionState(
            position_xy=self.position_xy,
            heading_rad=self.heading_rad,
            speed_mps=self.drift_speed_mps,
        )
        limits = MotionLimits(
            max_speed_mps=self.drift_speed_mps,
            max_acceleration_mps2=self.drift_speed_mps,
            max_deceleration_mps2=self.drift_speed_mps,
            max_turn_rate_rad_s=self.max_turn_rate_rad_s,
        )
        command = MotionCommand(
            desired_heading_rad=wrap_angle(
                self.heading_rad + rng.gauss(0.0, self.heading_noise_rad_per_s) * dt_s
            ),
            desired_speed_mps=self.drift_speed_mps,
        )
        if self.boundary is not None:
            command = constrain_navigation_command(
                state, command, limits, self.boundary, dt_s
            )
        end = advance_motion(state, command, limits, dt_s)
        self.position_xy = end.position_xy
        self.heading_rad = end.heading_rad
