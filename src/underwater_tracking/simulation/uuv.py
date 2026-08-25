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

_WAYPOINT_CROSS_TRACK_TOLERANCE_M = 1e-6


def wrap(value: float) -> float:
    """Compatibility alias for the shared angle-normalization helper."""
    return wrap_angle(value)


def _segment_reaches_waypoint(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    waypoint_xy: tuple[float, float],
) -> bool:
    """Return whether the bounded displacement segment reaches a waypoint."""
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    segment_length_squared = dx * dx + dy * dy
    if segment_length_squared == 0.0:
        return hypot(waypoint_xy[0] - start_xy[0], waypoint_xy[1] - start_xy[1]) <= (
            _WAYPOINT_CROSS_TRACK_TOLERANCE_M
        )

    projection = (
        (waypoint_xy[0] - start_xy[0]) * dx
        + (waypoint_xy[1] - start_xy[1]) * dy
    ) / segment_length_squared
    if not 0.0 <= projection <= 1.0:
        return False

    closest_x = start_xy[0] + projection * dx
    closest_y = start_xy[1] + projection * dy
    return hypot(waypoint_xy[0] - closest_x, waypoint_xy[1] - closest_y) <= (
        _WAYPOINT_CROSS_TRACK_TOLERANCE_M
    )


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
    transit_energy_per_m: float = 2e-6
    hotel_energy_per_s: float = 1e-7

    def set_waypoints(self, points: list[tuple[float, float]]) -> None:
        self.waypoints = list(points)

    def step(
        self,
        dt_s: float,
        max_speed_mps: float,
        max_turn_rate_rad_s: float,
        max_acceleration_mps2: float | None = None,
        max_deceleration_mps2: float | None = None,
    ) -> None:
        if self.energy_fraction <= 0:
            self.speed_mps = 0.0
            return
        limits = MotionLimits(
            max_speed_mps=max_speed_mps,
            max_acceleration_mps2=(
                max_acceleration_mps2
                if max_acceleration_mps2 is not None
                else max_speed_mps
            ),
            max_deceleration_mps2=(
                max_deceleration_mps2
                if max_deceleration_mps2 is not None
                else (
                    max_acceleration_mps2
                    if max_acceleration_mps2 is not None
                    else max_speed_mps
                )
            ),
            max_turn_rate_rad_s=max_turn_rate_rad_s,
        )
        start = MotionState(self.position_xy, self.heading_rad, self.speed_mps)
        if not self.waypoints:
            end = advance_motion(
                start,
                MotionCommand(self.heading_rad, 0.0),
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
            self.energy_fraction = max(
                0.0,
                self.energy_fraction
                - distance * self.transit_energy_per_m
                - dt_s * self.hotel_energy_per_s,
            )
            return

        wx, wy = self.waypoints[0]
        desired = atan2(wy - self.position_xy[1], wx - self.position_xy[0])
        distance_to_waypoint = hypot(wx - self.position_xy[0], wy - self.position_xy[1])
        heading_error = abs(wrap_angle(desired - self.heading_rad))
        desired_speed = min(max_speed_mps, distance_to_waypoint / max(dt_s, 1e-9))
        if heading_error > 1e-3:
            desired_speed = min(
                desired_speed,
                distance_to_waypoint * max_turn_rate_rad_s * 0.5,
            )
        end = advance_motion(
            start,
            MotionCommand(desired, desired_speed),
            limits,
            dt_s,
        )
        distance = hypot(
            end.position_xy[0] - self.position_xy[0],
            end.position_xy[1] - self.position_xy[1],
        )
        reached_waypoint = _segment_reaches_waypoint(
            self.position_xy,
            end.position_xy,
            (wx, wy),
        )
        if reached_waypoint:
            self.position_xy = (wx, wy)
            self.heading_rad = end.heading_rad
            self.speed_mps = end.speed_mps
            distance = hypot(wx - start.position_xy[0], wy - start.position_xy[1])
            self.waypoints.pop(0)
        else:
            self.position_xy = end.position_xy
            self.heading_rad = end.heading_rad
            self.speed_mps = end.speed_mps
        self.energy_fraction = max(
            0.0,
            self.energy_fraction
            - distance * self.transit_energy_per_m
            - dt_s * self.hotel_energy_per_s,
        )
