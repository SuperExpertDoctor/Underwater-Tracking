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
from math import cos, sin


@dataclass(slots=True)
class DecoyEntity:
    decoy_id: str
    position_xy: tuple[float, float]
    heading_rad: float
    drift_speed_mps: float
    heading_noise_rad_per_s: float

    def step(self, dt_s: float, rng: random.Random) -> None:
        self.heading_rad = self.heading_rad + rng.gauss(
            0.0, self.heading_noise_rad_per_s
        ) * dt_s
        distance = self.drift_speed_mps * dt_s
        self.position_xy = (
            self.position_xy[0] + distance * cos(self.heading_rad),
            self.position_xy[1] + distance * sin(self.heading_rad),
        )
