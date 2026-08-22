"""Mission-directed submarine entity with target-owned state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, cast

from underwater_tracking.domain.adversary_models import (
    AdversaryBelief,
    AdversaryEscapeDecision,
    AdversaryIntentDecision,
    AdversaryMissionState,
    AdversaryOperatingBoundary,
    TargetLocalContact,
)
from underwater_tracking.domain.platforms import MotionLimits
from underwater_tracking.simulation.kinematics import (
    MotionCommand,
    MotionState,
    NavigationBoundary,
    NavigationInvariantError,
    advance_motion,
    constrain_navigation_command,
    navigation_segment_is_legal,
    wrap_angle,
)
from underwater_tracking.simulation.target_guidance import (
    TargetGuidanceCommand,
    TargetGuidanceResult,
    resolve_target_guidance,
)


class HiddenIntent(StrEnum):
    TRANSIT = "transit"
    PATROL = "patrol"
    LOITER = "loiter"
    EVADE = "evade"
    APPROACH = "approach"
    WITHDRAW = "withdraw"


# Kept as stable replay labels for older serialized scenarios.  Live target
# motion does not sample this table or any other random direction.
INTENT_VELOCITIES: dict[HiddenIntent, tuple[float, float]] = {
    HiddenIntent.TRANSIT: (2.0, 0.0),
    HiddenIntent.PATROL: (1.5, 1.5),
    HiddenIntent.LOITER: (0.5, 0.5),
    HiddenIntent.EVADE: (-3.0, 2.0),
    HiddenIntent.APPROACH: (-2.0, -1.0),
    HiddenIntent.WITHDRAW: (3.0, -1.5),
}
INTENT_SPEED_MPS: dict[HiddenIntent, float] = {
    HiddenIntent.TRANSIT: 8.0,
    HiddenIntent.PATROL: 8.0,
    HiddenIntent.LOITER: 8.0,
    HiddenIntent.EVADE: 14.0,
    HiddenIntent.APPROACH: 8.0,
    HiddenIntent.WITHDRAW: 8.0,
}
# Compatibility surface for legacy active-sonar tests and serialized callers.
# The live target no longer samples this table; physical motion is resolved by
# the mission guidance command and deterministic kinematics below.
TRANSITION_PROBABILITIES: dict[HiddenIntent, dict[HiddenIntent, float]] = {
    intent: {intent: 1.0} for intent in HiddenIntent
}
DEFAULT_BOUNDS_XY: tuple[float, float, float, float] = (-5000.0, 5000.0, -5000.0, 5000.0)


@dataclass(frozen=True, slots=True)
class TargetManeuverCommand:
    """Inspectable bounded command currently being interpolated by the target."""

    desired_heading_rad: float
    desired_speed_mps: float
    max_turn_rate_rad_s: float
    max_acceleration_mps2: float
    remaining_steps: int


@dataclass(slots=True)
class TargetEntity:
    target_id: str
    position_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    intent: HiddenIntent
    bounds_xy: tuple[float, float, float, float] = DEFAULT_BOUNDS_XY
    detection_range_m: float = 1200.0
    intent_speed_mps: dict[HiddenIntent, float] | None = None
    max_speed_mps: float = 14.0
    max_acceleration_mps2: float = 0.08
    max_turn_rate_rad_s: float = math.pi / 300.0
    decoy_inventory: int = 2
    evasion_hold_steps: int = 12
    evasion_weave_amplitude_rad: float = 0.12
    evasion_weave_period_s: float = 18.0
    mission_state: AdversaryMissionState | None = None
    min_speed_mps: float = 0.0
    max_deceleration_mps2: float = 0.25
    exclusion_regions: tuple[tuple[tuple[float, float], ...], ...] = ()
    _desired_heading_rad: float = field(init=False, repr=False)
    _desired_speed_mps: float = field(init=False, repr=False)
    _belief_position_xy: tuple[float, float] = field(init=False, repr=False)
    _belief_velocity_xy: tuple[float, float] = field(init=False, repr=False)
    _belief_uncertainty_m: float = field(init=False, repr=False)
    _pending_decoy_count: int = field(init=False, default=0, repr=False)
    _desired_waypoint: tuple[float, float] | None = field(init=False, default=None, repr=False)
    _adversary_hold_steps: int = field(init=False, default=0, repr=False)
    _guidance: TargetGuidanceCommand | None = field(init=False, default=None, repr=False)
    _decision: AdversaryIntentDecision | None = field(init=False, default=None, repr=False)
    _local_contacts: tuple[TargetLocalContact, ...] = field(
        init=False, default=(), repr=False
    )
    _navigation_guard_failed: bool = field(init=False, default=False, repr=False)
    _last_navigation_error: str | None = field(init=False, default=None, repr=False)
    _legacy_command_mode: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.detection_range_m <= 0.0 or not math.isfinite(self.detection_range_m):
            raise ValueError("target detection_range_m must be finite and positive")
        if self.min_speed_mps < 0.0 or self.min_speed_mps >= self.max_speed_mps:
            raise ValueError("target min_speed_mps must be below max_speed_mps")
        if self.max_deceleration_mps2 <= 0.0:
            raise ValueError("target max_deceleration_mps2 must be positive")
        if self.mission_state is None:
            self.mission_state = self._legacy_mission_state()
        self._desired_heading_rad = math.atan2(self.velocity_xy[1], self.velocity_xy[0])
        self._desired_speed_mps = math.hypot(*self.velocity_xy)
        self._belief_position_xy = (float(self.position_xy[0]), float(self.position_xy[1]))
        self._belief_velocity_xy = (float(self.velocity_xy[0]), float(self.velocity_xy[1]))
        self._belief_uncertainty_m = 50.0

    def step(self, dt_s: float, rng: object | None = None, *, sim_time_s: int = 0) -> None:
        """Advance deterministic mission guidance through bounded kinematics.

        ``rng`` is accepted but intentionally ignored for compatibility with
        old callers.  The live engine passes only ``sim_time_s``.
        """
        del rng
        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        legacy_guidance = (
            self._guidance
            if self._legacy_command_mode and self._adversary_hold_steps > 0
            else None
        )
        if self._adversary_hold_steps <= 0 and self._decision is None:
            self._desired_waypoint = None
            if self._guidance is not None and self._guidance.source == "llm":
                self._guidance = None
        if self._decision is not None and self._guidance is not None:
            if self._guidance.valid_until_s <= sim_time_s:
                self._decision = None
                self._guidance = None
        mission = self.mission_state
        assert mission is not None
        limits = self._motion_limits()
        operating_boundary = self._operating_boundary()
        if legacy_guidance is not None:
            guidance = TargetGuidanceResult(
                command=legacy_guidance,
                next_route_index=mission.current_route_index,
            )
        else:
            if (
                self._decision is None
                and not self._local_contacts
                and mission.current_intent == "continue_mission"
                and self.intent is not HiddenIntent.TRANSIT
            ):
                mission = mission.model_copy(
                    update={"current_intent": _mission_intent(self.intent)}
                )
            guidance = resolve_target_guidance(
                decision=self._decision,
                mission=mission,
                contacts=self._local_contacts,
                state=self._motion_state(),
                limits=limits,
                operating_boundary=operating_boundary,
                exclusion_regions=self.exclusion_regions,
                sim_time_s=sim_time_s,
                previous_guidance=self._guidance,
            )
        self._guidance = guidance.command
        self._desired_waypoint = guidance.command.waypoint_xy
        if guidance.next_route_index != mission.current_route_index:
            self.mission_state = mission.model_copy(
                update={"current_route_index": guidance.next_route_index}
            )
        self.intent = _hidden_intent(guidance.command.intent)
        self._desired_heading_rad = guidance.command.desired_heading_rad
        self._desired_speed_mps = guidance.command.desired_speed_mps
        if self._adversary_hold_steps > 0:
            self._adversary_hold_steps -= 1
        if legacy_guidance is not None and self._adversary_hold_steps <= 0:
            self._desired_waypoint = None
            self._guidance = None
        if self._legacy_command_mode and self._adversary_hold_steps <= 0:
            self._desired_waypoint = None
        previous = self._motion_state()
        try:
            end = self._advance_bounded(previous, guidance.command, dt_s)
        except NavigationInvariantError as exc:
            self._navigation_guard_failed = True
            self._last_navigation_error = str(exc)
            end = previous
            self._guidance = TargetGuidanceCommand(
                decision_id=self._guidance.decision_id,
                intent="hold_position",
                waypoint_xy=previous.position_xy,
                desired_heading_rad=previous.heading_rad,
                desired_speed_mps=self.min_speed_mps,
                valid_until_s=sim_time_s + 30,
                source="safe_hold",
            )
        self.position_xy = end.position_xy
        self.velocity_xy = (
            end.speed_mps * math.cos(end.heading_rad),
            end.speed_mps * math.sin(end.heading_rad),
        )
        self._belief_position_xy = (float(self.position_xy[0]), float(self.position_xy[1]))
        self._belief_velocity_xy = (float(self.velocity_xy[0]), float(self.velocity_xy[1]))
        self._belief_uncertainty_m = min(2_000.0, self._belief_uncertainty_m + 1.0)

    def _advance_bounded(
        self,
        state: MotionState,
        guidance: TargetGuidanceCommand,
        dt_s: float,
    ) -> MotionState:
        if dt_s == 0.0:
            return state
        limits = self._motion_limits()
        boundary = NavigationBoundary(self.bounds_xy, self.exclusion_regions, 50.0)
        substeps = max(1, math.ceil(dt_s / 0.5))
        sub_dt = dt_s / substeps
        current = state
        for _ in range(substeps):
            requested = MotionCommand(
                desired_heading_rad=guidance.desired_heading_rad,
                desired_speed_mps=guidance.desired_speed_mps,
            )
            constrained = constrain_navigation_command(
                current,
                requested,
                limits,
                boundary,
                sub_dt,
            )
            candidate = advance_motion(current, constrained, limits, sub_dt)
            if not navigation_segment_is_legal(
                current.position_xy, candidate.position_xy, boundary
            ):
                raise NavigationInvariantError("target integration would leave navigation boundary")
            current = candidate
        return current

    def set_local_contacts(self, contacts: tuple[TargetLocalContact, ...]) -> None:
        self._local_contacts = tuple(contacts)

    def apply_evasive_maneuver(self, turn_angle_rad: float) -> None:
        """Compatibility path for active-ping tests; still uses bounded motion."""
        heading = math.atan2(self.velocity_xy[1], self.velocity_xy[0])
        self._desired_heading_rad = wrap_angle(heading + turn_angle_rad)
        self._desired_speed_mps = min(self.max_speed_mps, self._intent_speed(HiddenIntent.EVADE))
        self._desired_waypoint = None
        self._adversary_hold_steps = max(self._adversary_hold_steps, self.evasion_hold_steps)
        self.intent = HiddenIntent.EVADE
        self._guidance = TargetGuidanceCommand(
            decision_id=None,
            intent="avoid_contact",
            waypoint_xy=(
                self.position_xy[0] + 1000.0 * math.cos(self._desired_heading_rad),
                self.position_xy[1] + 1000.0 * math.sin(self._desired_heading_rad),
            ),
            desired_heading_rad=self._desired_heading_rad,
            desired_speed_mps=self._desired_speed_mps,
            valid_until_s=self.evasion_hold_steps,
            source="llm",
        )
        self._decision = None

    def apply_adversary_intent(
        self,
        decision: AdversaryIntentDecision,
        *,
        sim_time_s: int,
    ) -> TargetGuidanceCommand:
        """Accept a high-level decision and resolve its physical command."""
        if decision.target_id != self.target_id:
            raise ValueError("adversary decision target_id does not match target")
        mission = self.mission_state
        assert mission is not None
        updated_mission = mission.model_copy(
            update={
                "current_intent": decision.intent,
                "last_decision_id": decision.decision_id,
            }
        )
        if decision.intent == "escape_to_region" and decision.escape_region_id not in mission.escape_regions:
            raise ValueError("escape_region_id is not a configured escape region")
        self.mission_state = updated_mission
        self._decision = decision
        self._legacy_command_mode = False
        result = resolve_target_guidance(
            decision=decision,
            mission=updated_mission,
            contacts=self._local_contacts,
            state=self._motion_state(),
            limits=self._motion_limits(),
            operating_boundary=self._operating_boundary(),
            exclusion_regions=self.exclusion_regions,
            sim_time_s=sim_time_s,
            previous_guidance=self._guidance,
        )
        self._guidance = result.command
        self._desired_waypoint = result.command.waypoint_xy
        self._desired_heading_rad = result.command.desired_heading_rad
        self._desired_speed_mps = result.command.desired_speed_mps
        if result.next_route_index != updated_mission.current_route_index:
            self.mission_state = updated_mission.model_copy(
                update={"current_route_index": result.next_route_index}
            )
        self.intent = _hidden_intent(result.command.intent)
        self._adversary_hold_steps = max(1, self.evasion_hold_steps)
        self._belief_uncertainty_m = min(2_000.0, self._belief_uncertainty_m + 25.0)
        return result.command

    def adversary_belief(self, sim_time_s: int) -> AdversaryBelief:
        """Return the target-owned belief admitted to the adversary graph."""
        intent_hypothesis = cast(
            Literal[
                "break_contact",
                "reposition",
                "deception",
                "silent_transit",
                "evade",
                "hold_course",
            ],
            {
                HiddenIntent.TRANSIT: "silent_transit",
                HiddenIntent.PATROL: "reposition",
                HiddenIntent.LOITER: "hold_course",
                HiddenIntent.EVADE: "evade",
                HiddenIntent.APPROACH: "reposition",
                HiddenIntent.WITHDRAW: "break_contact",
            }[self.intent],
        )
        return AdversaryBelief(
            target_id=self.target_id,
            as_of_s=sim_time_s,
            estimated_position_xy=self._belief_position_xy,
            estimated_velocity_xy=self._belief_velocity_xy,
            position_uncertainty_m=self._belief_uncertainty_m,
            velocity_uncertainty_mps=0.5,
            estimated_heading=math.atan2(self._belief_velocity_xy[1], self._belief_velocity_xy[0]),
            estimated_speed_mps=math.hypot(*self._belief_velocity_xy),
            intent_hypothesis=intent_hypothesis,
            intent_confidence=0.65,
        )

    def public_kinematics(self) -> dict[str, object]:
        """Expose only the non-sensitive identity projection."""
        return {"target_id": self.target_id}

    def apply_adversary_decision(
        self, decision: AdversaryEscapeDecision, *, hold_steps: int = 1
    ) -> None:
        """Compatibility adapter for legacy serialized physical decisions."""
        if decision.target_id != self.target_id:
            raise ValueError("adversary decision target_id does not match target")
        if decision.speed > self.max_speed_mps:
            raise ValueError("adversary decision speed exceeds target limit")
        x, y = decision.waypoint
        if not _legal_position((x, y), self._operating_boundary()):
            raise ValueError("adversary waypoint is outside target boundary")
        self._desired_heading_rad = decision.heading
        self._desired_speed_mps = decision.speed
        self._desired_waypoint = (float(x), float(y))
        self._adversary_hold_steps = max(1, hold_steps)
        self._guidance = TargetGuidanceCommand(
            decision_id=f"legacy:{self.target_id}",
            intent="avoid_contact" if decision.intent in {"evade", "deception"} else "break_contact",
            waypoint_xy=(float(x), float(y)),
            desired_heading_rad=decision.heading,
            desired_speed_mps=decision.speed,
            valid_until_s=max(1, hold_steps),
            source="llm",
        )
        self._decision = None
        self._legacy_command_mode = True
        self.intent = {
            "break_contact": HiddenIntent.WITHDRAW,
            "reposition": HiddenIntent.TRANSIT,
            "deception": HiddenIntent.EVADE,
            "silent_transit": HiddenIntent.TRANSIT,
            "evade": HiddenIntent.EVADE,
            "hold_course": HiddenIntent.PATROL,
        }[decision.intent]
        self._pending_decoy_count = min(decision.decoy_count, self.decoy_inventory)
        self.decoy_inventory -= self._pending_decoy_count

    def consume_decoy_request(self) -> int:
        count = self._pending_decoy_count
        self._pending_decoy_count = 0
        return count

    @property
    def maneuver_command(self) -> TargetManeuverCommand | None:
        if self._adversary_hold_steps <= 0:
            return None
        return TargetManeuverCommand(
            desired_heading_rad=self._desired_heading_rad,
            desired_speed_mps=self._desired_speed_mps,
            max_turn_rate_rad_s=self.max_turn_rate_rad_s,
            max_acceleration_mps2=self.max_acceleration_mps2,
            remaining_steps=self._adversary_hold_steps,
        )

    @property
    def guidance_command(self) -> TargetGuidanceCommand | None:
        return self._guidance

    @property
    def navigation_guard_failed(self) -> bool:
        return self._navigation_guard_failed

    @property
    def last_navigation_error(self) -> str | None:
        return self._last_navigation_error

    def _motion_state(self) -> MotionState:
        return MotionState(
            self.position_xy,
            math.atan2(self.velocity_xy[1], self.velocity_xy[0]),
            math.hypot(*self.velocity_xy),
        )

    def _motion_limits(self) -> MotionLimits:
        return MotionLimits(
            min_speed_mps=self.min_speed_mps,
            max_speed_mps=self.max_speed_mps,
            max_acceleration_mps2=self.max_acceleration_mps2,
            max_deceleration_mps2=self.max_deceleration_mps2,
            max_turn_rate_rad_s=self.max_turn_rate_rad_s,
        )

    def _operating_boundary(self) -> AdversaryOperatingBoundary:
        return AdversaryOperatingBoundary(
            min_x=self.bounds_xy[0],
            max_x=self.bounds_xy[1],
            min_y=self.bounds_xy[2],
            max_y=self.bounds_xy[3],
        )

    def _legacy_mission_state(self) -> AdversaryMissionState:
        heading = math.atan2(self.velocity_xy[1], self.velocity_xy[0])
        next_point = (
            self.position_xy[0] + 10_000.0 * math.cos(heading),
            self.position_xy[1] + 10_000.0 * math.sin(heading),
        )
        next_point = (
            min(self.bounds_xy[1], max(self.bounds_xy[0], next_point[0])),
            min(self.bounds_xy[3], max(self.bounds_xy[2], next_point[1])),
        )
        return AdversaryMissionState(
            target_id=self.target_id,
            task_region_id="legacy_task",
            task_region_polygon_xy=(
                (self.bounds_xy[0], self.bounds_xy[2]),
                (self.bounds_xy[1], self.bounds_xy[2]),
                (self.bounds_xy[1], self.bounds_xy[3]),
                (self.bounds_xy[0], self.bounds_xy[3]),
            ),
            mission_route_xy=(self.position_xy, next_point),
            escape_regions={"legacy_escape": self._legacy_escape_polygon()},
            current_intent="continue_mission",
            current_route_index=0,
        )

    def _legacy_escape_polygon(self) -> tuple[tuple[float, float], ...]:
        min_x, max_x, min_y, max_y = self.bounds_xy
        return (
            (min_x + 0.25 * (max_x - min_x), min_y + 0.25 * (max_y - min_y)),
            (max_x - 0.25 * (max_x - min_x), min_y + 0.25 * (max_y - min_y)),
            (max_x - 0.25 * (max_x - min_x), max_y - 0.25 * (max_y - min_y)),
            (min_x + 0.25 * (max_x - min_x), max_y - 0.25 * (max_y - min_y)),
        )

    def _intent_speed(self, intent: HiddenIntent) -> float:
        if self.intent_speed_mps is None:
            return min(self.max_speed_mps, INTENT_SPEED_MPS[intent])
        return min(self.max_speed_mps, self.intent_speed_mps.get(intent, self.max_speed_mps))


def _hidden_intent(intent: str) -> HiddenIntent:
    mapping: dict[str, HiddenIntent] = {
        "continue_mission": HiddenIntent.TRANSIT,
        "avoid_contact": HiddenIntent.EVADE,
        "break_contact": HiddenIntent.WITHDRAW,
        "escape_to_region": HiddenIntent.WITHDRAW,
        "hold_position": HiddenIntent.LOITER,
    }
    return mapping[intent]


def _mission_intent(intent: HiddenIntent) -> Literal[
    "continue_mission",
    "avoid_contact",
    "break_contact",
    "escape_to_region",
    "hold_position",
]:
    return {
        HiddenIntent.TRANSIT: "continue_mission",
        HiddenIntent.PATROL: "continue_mission",
        HiddenIntent.LOITER: "hold_position",
        HiddenIntent.EVADE: "avoid_contact",
        HiddenIntent.APPROACH: "continue_mission",
        HiddenIntent.WITHDRAW: "break_contact",
    }[intent]


def _legal_position(
    position: tuple[float, float], boundary: NavigationBoundary | AdversaryOperatingBoundary
) -> bool:
    if isinstance(boundary, NavigationBoundary):
        bounds = boundary.bounds_xy
        exclusions = boundary.exclusion_polygons
    else:
        bounds = (boundary.min_x, boundary.max_x, boundary.min_y, boundary.max_y)
        exclusions = ()
    min_x, max_x, min_y, max_y = bounds
    return min_x - 1e-9 <= position[0] <= max_x + 1e-9 and min_y - 1e-9 <= position[1] <= max_y + 1e-9 and not any(
        _point_in_polygon(position, polygon) for polygon in exclusions
    )


def _point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        if (start[1] > point[1]) != (end[1] > point[1]):
            crossing_x = (end[0] - start[0]) * (point[1] - start[1]) / (end[1] - start[1]) + start[0]
            if point[0] < crossing_x:
                inside = not inside
    return inside


__all__ = [
    "HiddenIntent",
    "INTENT_SPEED_MPS",
    "INTENT_VELOCITIES",
    "TRANSITION_PROBABILITIES",
    "TargetEntity",
    "TargetManeuverCommand",
]
