from dataclasses import dataclass, field
from math import atan2, hypot

from underwater_tracking.domain.models import SurveillanceCapability
from underwater_tracking.domain.platforms import MotionLimits
from underwater_tracking.simulation.kinematics import (
    MotionCommand,
    MotionState,
    advance_motion,
    wrap_angle,
)


def wrap(value: float) -> float:
    """Compatibility alias for the shared angle-normalization helper."""
    return wrap_angle(value)


@dataclass(slots=True)
class UUVEntity:
    uuv_id: str
    position_xy: tuple[float, float]
    heading_rad: float
    energy_fraction: float
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    capability: SurveillanceCapability = field(default_factory=SurveillanceCapability)
    platform_index: int = 0
    speed_mps: float = 0.0

    def set_waypoints(self, points: list[tuple[float, float]]) -> None:
        self.waypoints = list(points)

    def step(
        self,
        dt_s: float,
        max_speed_mps: float,
        max_turn_rate_rad_s: float,
        max_acceleration_mps2: float | None = None,
    ) -> None:
        if not self.waypoints or self.energy_fraction <= 0:
            self.speed_mps = 0.0
            return
        wx, wy = self.waypoints[0]
        desired = atan2(wy - self.position_xy[1], wx - self.position_xy[0])
        limits = MotionLimits(
            max_speed_mps=max_speed_mps,
            max_acceleration_mps2=(
                max_acceleration_mps2
                if max_acceleration_mps2 is not None
                else max_speed_mps
            ),
            max_turn_rate_rad_s=max_turn_rate_rad_s,
        )
        end = advance_motion(
            MotionState(self.position_xy, self.heading_rad, self.speed_mps),
            MotionCommand(desired, max_speed_mps),
            limits,
            dt_s,
        )
        distance = hypot(
            end.position_xy[0] - self.position_xy[0],
            end.position_xy[1] - self.position_xy[1],
        )
        self.position_xy = end.position_xy
        self.heading_rad = end.heading_rad
        self.speed_mps = end.speed_mps
        self.energy_fraction = max(0.0, self.energy_fraction - distance * 2e-6 - dt_s * 1e-7)
        if hypot(wx - self.position_xy[0], wy - self.position_xy[1]) < 1.0:
            self.waypoints.pop(0)
