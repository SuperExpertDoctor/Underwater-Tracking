"""Deterministic carrier patrol kinematics."""

from __future__ import annotations

from collections.abc import Sequence
from math import atan2, hypot, pi

from underwater_tracking.domain.models import (
    CarrierState,
    CarrierStatus,
    DeploymentState,
    UUVState,
)

_CARRIER_ID = "carrier_01"
_PATROL_SPEED_MPS = 5.0
_MAX_TURN_RATE_RAD_S = 0.25
# This outer patrol lane remains inside the visible +/-4000 m map so the
# carrier sprite and recovery links stay on-screen.
_PATROL_CORNERS = (
    (-3000.0, -3000.0),
    (3000.0, -3000.0),
    (3000.0, 3000.0),
    (-3000.0, 3000.0),
)


class CarrierEntity:
    """A carrier moving at constant speed around a fixed outer patrol route."""

    def __init__(
        self,
        *,
        carrier_id: str = _CARRIER_ID,
        position_xy: tuple[float, float] = _PATROL_CORNERS[0],
        speed_mps: float = _PATROL_SPEED_MPS,
        patrol_route_xy: tuple[tuple[float, float], ...] = _PATROL_CORNERS,
        support_radius_m: float = 16000.0,
        max_turn_rate_rad_s: float = _MAX_TURN_RATE_RAD_S,
        heading_rad: float | None = None,
    ) -> None:
        if len(patrol_route_xy) < 2:
            raise ValueError("carrier patrol route requires at least two points")
        self.carrier_id = carrier_id
        self.position_xy = position_xy
        self.speed_mps = speed_mps
        self.support_radius_m = support_radius_m
        if max_turn_rate_rad_s <= 0.0:
            raise ValueError("max_turn_rate_rad_s must be positive")
        self.max_turn_rate_rad_s = max_turn_rate_rad_s
        self._patrol_route_xy = patrol_route_xy
        self._next_corner_index = 1
        self._mission_route_xy: tuple[tuple[float, float], ...] | None = None
        self._mission_route_index = 1
        self.heading_rad = (
            self._heading_to_next_corner() if heading_rad is None else heading_rad
        )

    def step(self, dt_s: float) -> None:
        """Advance the route while limiting heading change at each turn."""
        if self._mission_route_xy is not None:
            self._step_mission_route(dt_s)
            return
        remaining_s = max(0.0, dt_s)
        while remaining_s > 0.0 and self.speed_mps > 0.0:
            target = self._patrol_route_xy[self._next_corner_index]
            distance = hypot(target[0] - self.position_xy[0], target[1] - self.position_xy[1])
            if distance <= 1e-9:
                self._next_corner_index = (self._next_corner_index + 1) % len(self._patrol_route_xy)
                continue
            segment_heading = self._heading_to_next_corner()
            segment_s = min(remaining_s, distance / self.speed_mps)
            max_heading_delta = self.max_turn_rate_rad_s * segment_s
            heading_error = (segment_heading - self.heading_rad + pi) % (2.0 * pi) - pi
            self.heading_rad += max(-max_heading_delta, min(max_heading_delta, heading_error))
            distance_travelled = self.speed_mps * segment_s
            self.position_xy = (
                self.position_xy[0] + distance_travelled * (target[0] - self.position_xy[0]) / distance,
                self.position_xy[1] + distance_travelled * (target[1] - self.position_xy[1]) / distance,
            )
            remaining_s -= segment_s
            if segment_s < distance / self.speed_mps - 1e-9:
                return
            self.position_xy = target
            self._next_corner_index = (self._next_corner_index + 1) % len(self._patrol_route_xy)

    def set_mission_route(
        self,
        route_xy: tuple[tuple[float, float], ...],
    ) -> None:
        """Install a finite multi-stop route that must end at its home point."""
        if len(route_xy) < 2:
            raise ValueError("mission route requires at least one stop and home")
        if route_xy[0] != self.position_xy:
            raise ValueError("mission route must start at the current position")
        if route_xy[-1] != route_xy[0]:
            raise ValueError("mission route must return to home")
        self._mission_route_xy = route_xy
        self._mission_route_index = 1
        self.heading_rad = self._heading_to_mission_stop()

    @property
    def mission_route_xy(self) -> tuple[tuple[float, float], ...]:
        """Return the installed finite route, or an empty tuple for patrol mode."""
        return self._mission_route_xy or ()

    @property
    def mission_route_complete(self) -> bool:
        """Whether the finite mission route has reached its home point."""
        return self._mission_route_xy is not None and self._mission_route_index >= len(
            self._mission_route_xy
        )

    def _step_mission_route(self, dt_s: float) -> None:
        route = self._mission_route_xy
        if route is None or self.mission_route_complete:
            return
        remaining_s = max(0.0, dt_s)
        while remaining_s > 0.0 and self.speed_mps > 0.0:
            target = route[self._mission_route_index]
            distance = hypot(target[0] - self.position_xy[0], target[1] - self.position_xy[1])
            if distance <= 1e-9:
                self._mission_route_index += 1
                if self.mission_route_complete:
                    self.position_xy = route[-1]
                    return
                continue
            segment_heading = self._heading_to_mission_stop()
            segment_s = min(remaining_s, distance / self.speed_mps)
            max_heading_delta = self.max_turn_rate_rad_s * segment_s
            heading_error = (segment_heading - self.heading_rad + pi) % (2.0 * pi) - pi
            self.heading_rad += max(-max_heading_delta, min(max_heading_delta, heading_error))
            distance_travelled = self.speed_mps * segment_s
            self.position_xy = (
                self.position_xy[0] + distance_travelled * (target[0] - self.position_xy[0]) / distance,
                self.position_xy[1] + distance_travelled * (target[1] - self.position_xy[1]) / distance,
            )
            remaining_s -= segment_s
            if segment_s < distance / self.speed_mps - 1e-9:
                return
            self.position_xy = target
            self._mission_route_index += 1
            if self.mission_route_complete:
                return

    def _heading_to_mission_stop(self) -> float:
        assert self._mission_route_xy is not None
        target = self._mission_route_xy[self._mission_route_index]
        return atan2(target[1] - self.position_xy[1], target[0] - self.position_xy[0])

    def state_for(self, uuvs: Sequence[UUVState]) -> CarrierState:
        """Return the carrier state and sorted UUV deployment relationships."""
        returning = tuple(sorted(u.uuv_id for u in uuvs if u.deployment_state is DeploymentState.RETURNING))
        onboard = tuple(sorted(u.uuv_id for u in uuvs if u.deployment_state is DeploymentState.ONBOARD))
        deployed = tuple(sorted(u.uuv_id for u in uuvs if u.deployment_state is DeploymentState.DEPLOYED))
        status = (
            CarrierStatus.STANDBY
            if self.speed_mps == 0.0
            else CarrierStatus.RECOVERING
            if returning
            else CarrierStatus.DEPLOYING
            if onboard and deployed
            else CarrierStatus.TRANSIT
        )
        return CarrierState(
            carrier_id=self.carrier_id,
            position_xy=self.position_xy,
            heading_rad=self.heading_rad,
            speed_mps=self.speed_mps,
            status=status,
            onboard_uuv_ids=onboard,
            deployed_uuv_ids=deployed,
            returning_uuv_ids=returning,
        )

    def _heading_to_next_corner(self) -> float:
        target = self._patrol_route_xy[self._next_corner_index]
        return atan2(target[1] - self.position_xy[1], target[0] - self.position_xy[0])
