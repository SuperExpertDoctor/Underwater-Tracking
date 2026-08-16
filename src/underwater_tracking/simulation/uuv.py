from dataclasses import dataclass, field
from math import atan2, cos, hypot, pi, sin

from underwater_tracking.domain.models import SurveillanceCapability


def wrap(value: float) -> float:
    return (value + pi) % (2 * pi) - pi


@dataclass(slots=True)
class UUVEntity:
    uuv_id: str
    position_xy: tuple[float, float]
    heading_rad: float
    energy_fraction: float
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    capability: SurveillanceCapability = field(default_factory=SurveillanceCapability)

    def set_waypoints(self, points: list[tuple[float, float]]) -> None:
        self.waypoints = list(points)

    def step(self, dt_s: float, max_speed_mps: float, max_turn_rate_rad_s: float) -> None:
        if not self.waypoints or self.energy_fraction <= 0:
            return
        x, y = self.position_xy
        wx, wy = self.waypoints[0]
        desired = atan2(wy - y, wx - x)
        turn = max(-max_turn_rate_rad_s * dt_s, min(max_turn_rate_rad_s * dt_s, wrap(desired - self.heading_rad)))
        self.heading_rad = wrap(self.heading_rad + turn)
        distance = min(max_speed_mps * dt_s, hypot(wx - x, wy - y))
        self.position_xy = (x + distance * cos(self.heading_rad), y + distance * sin(self.heading_rad))
        self.energy_fraction = max(0.0, self.energy_fraction - distance * 2e-6 - dt_s * 1e-7)
        if hypot(wx - self.position_xy[0], wy - self.position_xy[1]) < 1.0:
            self.waypoints.pop(0)
