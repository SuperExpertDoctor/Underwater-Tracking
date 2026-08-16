"""Deterministic carrier patrol kinematics."""

from __future__ import annotations

from collections.abc import Sequence
from math import atan2, hypot

from underwater_tracking.domain.models import (
    CarrierState,
    CarrierStatus,
    DeploymentState,
    UUVState,
)

_CARRIER_ID = "carrier_01"
_PATROL_SPEED_MPS = 5.0
_PATROL_CORNERS = (
    (-3000.0, -3000.0),
    (3000.0, -3000.0),
    (3000.0, 3000.0),
    (-3000.0, 3000.0),
)


class CarrierEntity:
    """A carrier moving at constant speed around a fixed outer patrol route."""

    def __init__(self) -> None:
        self.position_xy = _PATROL_CORNERS[0]
        self.speed_mps = _PATROL_SPEED_MPS
        self._next_corner_index = 1
        self.heading_rad = self._heading_to_next_corner()

    def step(self, dt_s: float) -> None:
        """Advance along the patrol route, reflecting onto the next leg at corners."""
        remaining = max(0.0, dt_s) * self.speed_mps
        while remaining > 0.0:
            target = _PATROL_CORNERS[self._next_corner_index]
            distance = hypot(target[0] - self.position_xy[0], target[1] - self.position_xy[1])
            if remaining < distance:
                self.heading_rad = self._heading_to_next_corner()
                self.position_xy = (
                    self.position_xy[0] + remaining * (target[0] - self.position_xy[0]) / distance,
                    self.position_xy[1] + remaining * (target[1] - self.position_xy[1]) / distance,
                )
                return
            self.position_xy = target
            remaining -= distance
            self._next_corner_index = (self._next_corner_index + 1) % len(_PATROL_CORNERS)
            self.heading_rad = self._heading_to_next_corner()

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
            carrier_id=_CARRIER_ID,
            position_xy=self.position_xy,
            heading_rad=self.heading_rad,
            speed_mps=self.speed_mps,
            status=status,
            onboard_uuv_ids=onboard,
            deployed_uuv_ids=deployed,
            returning_uuv_ids=returning,
        )

    def _heading_to_next_corner(self) -> float:
        target = _PATROL_CORNERS[self._next_corner_index]
        return atan2(target[1] - self.position_xy[1], target[0] - self.position_xy[0])
