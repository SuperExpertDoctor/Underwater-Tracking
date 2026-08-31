from __future__ import annotations

from math import hypot

from underwater_tracking.simulation.decoy import DecoyEntity
from underwater_tracking.simulation.kinematics import wrap_angle


class _FixedNoise:
    def gauss(self, _mean: float, _sigma: float) -> float:
        return 10.0


def test_decoy_drift_is_turn_rate_limited() -> None:
    decoy = DecoyEntity(
        decoy_id="decoy-01",
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
        drift_speed_mps=0.5,
        heading_noise_rad_per_s=1.0,
    )

    decoy.step(1.0, _FixedNoise())

    assert abs(wrap_angle(decoy.heading_rad)) <= 0.25 + 1e-9
    assert hypot(*decoy.position_xy) <= 0.5 + 1e-9
