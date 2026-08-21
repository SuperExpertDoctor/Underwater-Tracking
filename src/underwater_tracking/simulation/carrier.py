"""Deterministic carrier patrol kinematics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import atan2, hypot, pi
from typing import Literal

from underwater_tracking.domain.models import (
    CarrierState,
    CarrierStatus,
    DeploymentState,
    UUVState,
)
from underwater_tracking.simulation.kinematics import wrap_angle

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
        role: Literal["carrier", "mother_ship"] = "carrier",
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
        self.role = role
        self.position_xy = position_xy
        self.speed_mps = speed_mps
        self.support_radius_m = support_radius_m
        if max_turn_rate_rad_s <= 0.0:
            raise ValueError("max_turn_rate_rad_s must be positive")
        self.max_turn_rate_rad_s = max_turn_rate_rad_s
        self._patrol_route_xy = patrol_route_xy
        self._next_corner_index = 1
        self._mission_route_xy: tuple[tuple[float, float], ...] | None = None
        self._mission_home_xy: tuple[float, float] | None = None
        self._mission_route_index = 1
        self._mission_stop_windows: dict[int, tuple[int, int]] = {}
        self._externally_released_stop_indices: frozenset[int] = frozenset()
        self._externally_arrived_stop_indices: set[int] = set()
        self._arrived_mission_stop_indices: list[int] = []
        self.heading_rad = (
            self._heading_to_next_corner() if heading_rad is None else heading_rad
        )

    def step(self, dt_s: float, *, sim_time_s: int | None = None) -> None:
        """Advance the route while limiting heading change at each turn."""
        if self._mission_route_xy is not None:
            self._step_mission_route(dt_s, sim_time_s=sim_time_s)
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
            self.heading_rad = wrap_angle(
                self.heading_rad
                + max(-max_heading_delta, min(max_heading_delta, heading_error))
            )
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
        *,
        stop_windows: Mapping[int, tuple[int, int]] | None = None,
        externally_released_stop_indices: frozenset[int] = frozenset(),
        rendezvous_xy: tuple[float, float] | None = None,
        home_xy: tuple[float, float] | None = None,
    ) -> None:
        """Install a finite multi-stop route ending at a rendezvous point."""
        if len(route_xy) < 2:
            raise ValueError("mission route requires at least one leg")
        if route_xy[0] != self.position_xy:
            raise ValueError("mission route must start at the current position")
        if home_xy is not None and rendezvous_xy is not None and home_xy != rendezvous_xy:
            raise ValueError("home_xy and rendezvous_xy must agree")
        expected_endpoint = rendezvous_xy if rendezvous_xy is not None else home_xy
        if expected_endpoint is None:
            expected_endpoint = route_xy[0]
        if route_xy[-1] != expected_endpoint:
            if rendezvous_xy is None and home_xy is None:
                raise ValueError("mission route must return to home")
            raise ValueError("mission route must end at its rendezvous point")
        windows = dict(stop_windows or {})
        external_indices = frozenset(externally_released_stop_indices)
        if any(
            index <= 0 or index >= len(route_xy) - 1 for index in windows
        ):
            raise ValueError("mission stop window must identify an interior route point")
        if any(
            index <= 0 or index >= len(route_xy) - 1 for index in external_indices
        ):
            raise ValueError(
                "externally released stop must identify an interior route point"
            )
        if any(
            entry_s < 0 or exit_s <= entry_s
            for entry_s, exit_s in windows.values()
        ):
            raise ValueError("mission stop windows must be ordered")
        self._mission_route_xy = route_xy
        self._mission_home_xy = expected_endpoint
        self._mission_route_index = 1
        self._mission_stop_windows = windows
        self._externally_released_stop_indices = external_indices
        self._externally_arrived_stop_indices.clear()
        self._arrived_mission_stop_indices.clear()
        self.heading_rad = self._heading_to_mission_stop()

    @property
    def awaiting_release_stop_index(self) -> int | None:
        """Return the externally controlled stop currently holding the carrier."""
        route_index = self._mission_route_index
        if route_index in self._externally_arrived_stop_indices:
            return route_index
        return None

    def release_mission_stop(self, route_index: int) -> None:
        """Release the currently held externally controlled stop."""
        if route_index not in self._externally_released_stop_indices:
            raise ValueError(f"route index {route_index} is not externally released")
        if route_index != self._mission_route_index:
            raise ValueError(f"route index {route_index} is not the current mission stop")
        if route_index not in self._externally_arrived_stop_indices:
            raise ValueError(f"route index {route_index} has not been reached")
        self._externally_arrived_stop_indices.remove(route_index)
        self._mission_route_index += 1
        if not self.mission_route_complete:
            self.heading_rad = self._heading_to_mission_stop()

    def remaining_committed_stops(self) -> tuple[tuple[float, float], ...]:
        """Return unfinished service stops in their installed route order."""
        route = self._mission_route_xy
        if route is None:
            return ()
        committed_indices = sorted(
            index
            for index in set(self._mission_stop_windows) |
            self._externally_released_stop_indices
            if index >= self._mission_route_index
        )
        return tuple(route[index] for index in committed_indices)

    def replace_unfinished_return_segment(
        self,
        route_xy: tuple[tuple[float, float], ...],
    ) -> None:
        """Replace the unfinished tail while retaining every committed stop."""
        current_route = self._mission_route_xy
        if current_route is None or self.mission_route_complete:
            raise ValueError("cannot replace an inactive or completed mission")
        if len(route_xy) < 2:
            raise ValueError("replacement route requires at least one leg")
        if route_xy[0] != self.position_xy:
            raise ValueError("replacement route must start at the current position")

        committed_indices = sorted(
            index
            for index in set(self._mission_stop_windows) |
            self._externally_released_stop_indices
            if index >= self._mission_route_index
        )
        committed_points = tuple(current_route[index] for index in committed_indices)
        replacement = list(route_xy)
        # A stop being held at the current position must remain an interior
        # route point so that release_mission_stop() can advance the route.
        if committed_points and committed_points[0] == self.position_xy:
            if len(replacement) == 1 or replacement[1] != self.position_xy:
                replacement.insert(1, self.position_xy)

        new_indices: list[int] = []
        search_from = 1
        for point in committed_points:
            try:
                index = replacement.index(point, search_from)
            except ValueError as exc:
                raise ValueError("replacement route omits a committed stop") from exc
            if index == 0 or index == len(replacement) - 1:
                raise ValueError("committed stop must remain an interior route point")
            new_indices.append(index)
            search_from = index + 1

        old_window_by_point = {
            current_route[index]: self._mission_stop_windows[index]
            for index in committed_indices
            if index in self._mission_stop_windows
        }
        old_external_by_point = {
            current_route[index]
            for index in committed_indices
            if index in self._externally_released_stop_indices
        }
        old_arrived_by_point = {
            current_route[index]
            for index in committed_indices
            if index in self._externally_arrived_stop_indices
        }
        self._mission_route_xy = tuple(replacement)
        self._mission_home_xy = self._mission_route_xy[-1]
        self._mission_route_index = 1
        self._mission_stop_windows = {
            index: old_window_by_point[point]
            for index, point in zip(new_indices, committed_points, strict=True)
            if point in old_window_by_point
        }
        self._externally_released_stop_indices = frozenset(
            index
            for index, point in zip(new_indices, committed_points, strict=True)
            if point in old_external_by_point
        )
        self._externally_arrived_stop_indices = {
            index
            for index, point in zip(new_indices, committed_points, strict=True)
            if point in old_arrived_by_point
        }
        self._arrived_mission_stop_indices.clear()
        self.heading_rad = self._heading_to_mission_stop()

    def clear_completed_mission(self) -> None:
        """Return the entity to patrol mode after a completed finite mission."""
        if not self.mission_route_complete:
            raise ValueError("mission route is not complete")
        self._mission_route_xy = None
        self._mission_home_xy = None
        self._mission_route_index = 1
        self._mission_stop_windows.clear()
        self._externally_released_stop_indices = frozenset()
        self._externally_arrived_stop_indices.clear()
        self._arrived_mission_stop_indices.clear()
        self.heading_rad = self._heading_to_next_corner()

    def project_patrol_state(self, delta_s: float) -> tuple[tuple[float, float], float]:
        """Project patrol position and heading without mutating this entity."""
        position = self.position_xy
        heading = self.heading_rad
        next_corner_index = self._next_corner_index
        remaining_s = max(0.0, delta_s)
        while remaining_s > 0.0 and self.speed_mps > 0.0:
            target = self._patrol_route_xy[next_corner_index]
            distance = hypot(target[0] - position[0], target[1] - position[1])
            if distance <= 1e-9:
                next_corner_index = (next_corner_index + 1) % len(self._patrol_route_xy)
                continue
            segment_heading = wrap_angle(atan2(target[1] - position[1], target[0] - position[0]))
            segment_s = min(remaining_s, distance / self.speed_mps)
            max_heading_delta = self.max_turn_rate_rad_s * segment_s
            heading_error = (segment_heading - heading + pi) % (2.0 * pi) - pi
            heading = wrap_angle(
                heading
                + max(-max_heading_delta, min(max_heading_delta, heading_error))
            )
            distance_travelled = self.speed_mps * segment_s
            position = (
                position[0] + distance_travelled * (target[0] - position[0]) / distance,
                position[1] + distance_travelled * (target[1] - position[1]) / distance,
            )
            remaining_s -= segment_s
            if segment_s < distance / self.speed_mps - 1e-9:
                break
            position = target
            next_corner_index = (next_corner_index + 1) % len(self._patrol_route_xy)
        return position, heading

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

    def _step_mission_route(self, dt_s: float, *, sim_time_s: int | None = None) -> None:
        route = self._mission_route_xy
        if route is None or self.mission_route_complete:
            return
        self._arrived_mission_stop_indices.clear()
        remaining_s = max(0.0, dt_s)
        end_time_s = float(sim_time_s) if sim_time_s is not None else remaining_s
        while remaining_s > 0.0 and self.speed_mps > 0.0:
            target = route[self._mission_route_index]
            distance = hypot(target[0] - self.position_xy[0], target[1] - self.position_xy[1])
            if distance <= 1e-9:
                released, remaining_s = self._release_mission_stop(
                    self._mission_route_index,
                    remaining_s,
                    end_time_s,
                )
                if not released:
                    return
                if self.mission_route_complete:
                    self.position_xy = route[-1]
                    return
                continue
            segment_heading = self._heading_to_mission_stop()
            segment_s = min(remaining_s, distance / self.speed_mps)
            max_heading_delta = self.max_turn_rate_rad_s * segment_s
            heading_error = (segment_heading - self.heading_rad + pi) % (2.0 * pi) - pi
            self.heading_rad = wrap_angle(
                self.heading_rad
                + max(-max_heading_delta, min(max_heading_delta, heading_error))
            )
            distance_travelled = self.speed_mps * segment_s
            self.position_xy = (
                self.position_xy[0] + distance_travelled * (target[0] - self.position_xy[0]) / distance,
                self.position_xy[1] + distance_travelled * (target[1] - self.position_xy[1]) / distance,
            )
            remaining_s -= segment_s
            if segment_s < distance / self.speed_mps - 1e-9:
                return
            self.position_xy = target
            released, remaining_s = self._release_mission_stop(
                self._mission_route_index,
                remaining_s,
                end_time_s,
            )
            if not released:
                return
            if self.mission_route_complete:
                return

    def _release_mission_stop(
        self,
        route_index: int,
        remaining_s: float,
        end_time_s: float,
    ) -> tuple[bool, float]:
        """Release a reached stop after waiting for its earliest service time."""
        window = self._mission_stop_windows.get(route_index)
        arrival_time_s = end_time_s - remaining_s
        if window is not None:
            earliest_s, _ = window
            wait_s = max(0.0, earliest_s - arrival_time_s)
            if wait_s > remaining_s + 1e-9:
                return False, remaining_s
            remaining_s = max(0.0, remaining_s - wait_s)
        if route_index in self._externally_released_stop_indices:
            if route_index in self._externally_arrived_stop_indices:
                return False, remaining_s
            self._externally_arrived_stop_indices.add(route_index)
            self._arrived_mission_stop_indices.append(route_index)
            return False, remaining_s
        self._arrived_mission_stop_indices.append(route_index)
        self._mission_route_index += 1
        return True, remaining_s

    def consume_arrived_mission_stop_indices(self) -> tuple[int, ...]:
        """Return and clear route-point indices reached during the last step."""
        arrived = tuple(self._arrived_mission_stop_indices)
        self._arrived_mission_stop_indices.clear()
        return arrived

    def _heading_to_mission_stop(self) -> float:
        assert self._mission_route_xy is not None
        target = self._mission_route_xy[self._mission_route_index]
        return wrap_angle(atan2(target[1] - self.position_xy[1], target[0] - self.position_xy[0]))

    def state_for(
        self,
        uuvs: Sequence[UUVState],
        assigned_uuv_ids: Sequence[str] | None = None,
    ) -> CarrierState:
        """Return the carrier state and sorted UUV deployment relationships."""
        selected = (
            tuple(u for u in uuvs if u.uuv_id in set(assigned_uuv_ids))
            if assigned_uuv_ids is not None
            else tuple(uuvs)
        )
        returning = tuple(sorted(u.uuv_id for u in selected if u.deployment_state is DeploymentState.RETURNING))
        onboard = tuple(sorted(u.uuv_id for u in selected if u.deployment_state is DeploymentState.ONBOARD))
        deployed = tuple(sorted(u.uuv_id for u in selected if u.deployment_state is DeploymentState.DEPLOYED))
        reported_speed_mps = 0.0 if self.mission_route_complete else self.speed_mps
        status = (
            CarrierStatus.RECOVERING
            if returning
            else CarrierStatus.STANDBY
            if reported_speed_mps == 0.0
            else CarrierStatus.DEPLOYING
            if onboard and deployed
            else CarrierStatus.TRANSIT
        )
        return CarrierState(
            carrier_id=self.carrier_id,
            role=self.role,
            position_xy=self.position_xy,
            heading_rad=self.heading_rad,
            speed_mps=reported_speed_mps,
            status=status,
            onboard_uuv_ids=onboard,
            deployed_uuv_ids=deployed,
            returning_uuv_ids=returning,
        )

    def _heading_to_next_corner(self) -> float:
        target = self._patrol_route_xy[self._next_corner_index]
        return wrap_angle(atan2(target[1] - self.position_xy[1], target[0] - self.position_xy[0]))
