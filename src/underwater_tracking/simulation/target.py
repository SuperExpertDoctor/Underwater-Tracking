import math
import random
from dataclasses import dataclass, field
from typing import Literal, cast

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value

from underwater_tracking.domain.adversary_models import (
    AdversaryBelief,
    AdversaryEscapeDecision,
)
from underwater_tracking.domain.platforms import MotionLimits
from underwater_tracking.simulation.kinematics import (
    MotionCommand,
    MotionState,
    advance_motion,
    wrap_angle,
)


class HiddenIntent(StrEnum):
    TRANSIT = "transit"
    PATROL = "patrol"
    LOITER = "loiter"
    EVADE = "evade"
    APPROACH = "approach"
    WITHDRAW = "withdraw"


# Nominal velocity adopted when the target transitions into each intent, in
# metres per second. Deterministic: the only randomness in target motion is
# the explicit RNG passed to step().
INTENT_VELOCITIES: dict[HiddenIntent, tuple[float, float]] = {
    HiddenIntent.TRANSIT: (2.0, 0.0),
    HiddenIntent.PATROL: (1.5, 1.5),
    HiddenIntent.LOITER: (0.5, 0.5),
    HiddenIntent.EVADE: (-3.0, 2.0),
    HiddenIntent.APPROACH: (-2.0, -1.0),
    HiddenIntent.WITHDRAW: (3.0, -1.5),
}

# Cruise/sprint speeds (m/s) adopted when the target transitions into each
# intent (spec 5.1 amendment, R2): submarines cruise at 8 m/s and sprint at
# 14 m/s while evading — significantly faster than the 4 m/s UUV, so intent
# understanding and trajectory prediction are the core of tracking
# feasibility. ``INTENT_VELOCITIES`` above stays as the DIRECTION table.
INTENT_SPEED_MPS: dict[HiddenIntent, float] = {
    HiddenIntent.TRANSIT: 8.0,
    HiddenIntent.PATROL: 8.0,
    HiddenIntent.LOITER: 8.0,
    HiddenIntent.EVADE: 14.0,
    HiddenIntent.APPROACH: 8.0,
    HiddenIntent.WITHDRAW: 8.0,
}

# Per-intent Markov transition probabilities, one row per current intent;
# every row sums to 1.0. Iteration order (enum order) defines the order of
# the cumulative-draw comparison.
TRANSITION_PROBABILITIES: dict[HiddenIntent, dict[HiddenIntent, float]] = {
    HiddenIntent.TRANSIT: {
        HiddenIntent.TRANSIT: 0.90,
        HiddenIntent.PATROL: 0.03,
        HiddenIntent.LOITER: 0.01,
        HiddenIntent.EVADE: 0.02,
        HiddenIntent.APPROACH: 0.02,
        HiddenIntent.WITHDRAW: 0.02,
    },
    HiddenIntent.PATROL: {
        HiddenIntent.PATROL: 0.88,
        HiddenIntent.TRANSIT: 0.04,
        HiddenIntent.LOITER: 0.02,
        HiddenIntent.EVADE: 0.02,
        HiddenIntent.APPROACH: 0.02,
        HiddenIntent.WITHDRAW: 0.02,
    },
    HiddenIntent.LOITER: {
        HiddenIntent.LOITER: 0.90,
        HiddenIntent.TRANSIT: 0.04,
        HiddenIntent.PATROL: 0.02,
        HiddenIntent.EVADE: 0.01,
        HiddenIntent.APPROACH: 0.02,
        HiddenIntent.WITHDRAW: 0.01,
    },
    HiddenIntent.EVADE: {
        HiddenIntent.EVADE: 0.85,
        HiddenIntent.TRANSIT: 0.05,
        HiddenIntent.PATROL: 0.02,
        HiddenIntent.LOITER: 0.02,
        HiddenIntent.APPROACH: 0.03,
        HiddenIntent.WITHDRAW: 0.03,
    },
    HiddenIntent.APPROACH: {
        HiddenIntent.APPROACH: 0.88,
        HiddenIntent.TRANSIT: 0.04,
        HiddenIntent.PATROL: 0.02,
        HiddenIntent.LOITER: 0.02,
        HiddenIntent.EVADE: 0.02,
        HiddenIntent.WITHDRAW: 0.02,
    },
    HiddenIntent.WITHDRAW: {
        HiddenIntent.WITHDRAW: 0.88,
        HiddenIntent.TRANSIT: 0.04,
        HiddenIntent.PATROL: 0.02,
        HiddenIntent.LOITER: 0.02,
        HiddenIntent.EVADE: 0.02,
        HiddenIntent.APPROACH: 0.02,
    },
}

# Default world bounds (x_min, x_max, y_min, y_max) in metres.
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
    _desired_heading_rad: float = field(init=False, repr=False)
    _desired_speed_mps: float = field(init=False, repr=False)
    _belief_position_xy: tuple[float, float] = field(init=False, repr=False)
    _belief_velocity_xy: tuple[float, float] = field(init=False, repr=False)
    _belief_uncertainty_m: float = field(init=False, repr=False)
    _pending_decoy_count: int = field(init=False, default=0, repr=False)
    _desired_waypoint: tuple[float, float] | None = field(init=False, default=None, repr=False)
    _adversary_hold_steps: int = field(init=False, default=0, repr=False)
    _evasion_elapsed_s: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.detection_range_m <= 0.0 or not math.isfinite(self.detection_range_m):
            raise ValueError("target detection_range_m must be finite and positive")
        self._desired_heading_rad = math.atan2(self.velocity_xy[1], self.velocity_xy[0])
        self._desired_speed_mps = math.hypot(*self.velocity_xy)
        self._belief_position_xy = (
            float(self.position_xy[0]),
            float(self.position_xy[1]),
        )
        self._belief_velocity_xy = (
            float(self.velocity_xy[0]),
            float(self.velocity_xy[1]),
        )
        self._belief_uncertainty_m = 50.0
        self._desired_waypoint = None
        self._adversary_hold_steps = 0
        self._evasion_elapsed_s = 0.0

    def step(self, dt_s: float, rng: random.Random) -> None:
        adversary_command_active = self._adversary_hold_steps > 0
        if adversary_command_active:
            self._adversary_hold_steps -= 1
        elif self._desired_waypoint is None:
            next_intent = self._sample_intent(rng)
            if next_intent is not self.intent:
                self.intent = next_intent
                direction = INTENT_VELOCITIES[next_intent]
                self._desired_heading_rad = math.atan2(direction[1], direction[0])
                self._desired_speed_mps = self._intent_speed(next_intent)
                if next_intent is not HiddenIntent.EVADE:
                    self._evasion_elapsed_s = 0.0
        current_speed = math.hypot(*self.velocity_xy)
        current_heading = math.atan2(self.velocity_xy[1], self.velocity_xy[0])
        desired_heading = self._desired_heading_rad
        if self._desired_waypoint is not None:
            if not adversary_command_active:
                self._desired_waypoint = None
            else:
                dx = self._desired_waypoint[0] - self.position_xy[0]
                dy = self._desired_waypoint[1] - self.position_xy[1]
                if math.hypot(dx, dy) > 1.0:
                    desired_heading = math.atan2(dy, dx)
                else:
                    self._desired_waypoint = None
        if self.intent is HiddenIntent.EVADE and self._desired_waypoint is None:
            # Keep evasive motion smooth while varying the commanded heading
            # slowly enough that the shared turn-rate limit remains visible.
            self._evasion_elapsed_s += max(0.0, dt_s)
            weave_phase = (2.0 * math.pi * self._evasion_elapsed_s) / max(
                self.evasion_weave_period_s, 1e-6
            )
            desired_heading = wrap_angle(
                desired_heading + self.evasion_weave_amplitude_rad * math.sin(weave_phase)
            )
        limits = MotionLimits(
            max_speed_mps=self.max_speed_mps,
            max_acceleration_mps2=self.max_acceleration_mps2,
            max_turn_rate_rad_s=self.max_turn_rate_rad_s,
        )
        end = advance_motion(
            MotionState(self.position_xy, current_heading, current_speed),
            MotionCommand(desired_heading, self._desired_speed_mps),
            limits,
            dt_s,
        )
        self.position_xy = end.position_xy
        self.velocity_xy = (
            end.speed_mps * math.cos(end.heading_rad),
            end.speed_mps * math.sin(end.heading_rad),
        )
        self._reflect_into_bounds()
        # The target maintains its own bounded estimate.  It is deliberately
        # exposed only through ``adversary_belief`` and never as simulator
        # truth or as an engine event payload.
        self._belief_position_xy = (
            float(self.position_xy[0]),
            float(self.position_xy[1]),
        )
        self._belief_velocity_xy = (
            float(self.velocity_xy[0]),
            float(self.velocity_xy[1]),
        )
        self._belief_uncertainty_m = min(2_000.0, self._belief_uncertainty_m + 1.0)

    def apply_evasive_maneuver(self, turn_angle_rad: float) -> None:
        """Evasive turn when the target detects an active ping (R2/R5).

        The target switches to EVADE and commands a turn of
        ``turn_angle_rad``. Subsequent steps apply the speed and turn
        changes through the shared bounded-motion integrator.
        """
        heading = math.atan2(self.velocity_xy[1], self.velocity_xy[0])
        self.intent = HiddenIntent.EVADE
        self._desired_heading_rad = wrap_angle(heading + turn_angle_rad)
        self._desired_speed_mps = self._intent_speed(HiddenIntent.EVADE)
        self._desired_waypoint = None
        self._adversary_hold_steps = max(self._adversary_hold_steps, self.evasion_hold_steps)
        self._evasion_elapsed_s = 0.0

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
            estimated_heading=math.atan2(
                self._belief_velocity_xy[1], self._belief_velocity_xy[0]
            ),
            estimated_speed_mps=math.hypot(*self._belief_velocity_xy),
            intent_hypothesis=intent_hypothesis,
            intent_confidence=0.65,
        )

    def apply_adversary_decision(
        self, decision: AdversaryEscapeDecision, *, hold_steps: int = 1
    ) -> None:
        """Apply a validated adversary decision through bounded kinematics."""
        if decision.target_id != self.target_id:
            raise ValueError("adversary decision target_id does not match target")
        if decision.speed > self.max_speed_mps:
            raise ValueError("adversary decision speed exceeds target limit")
        x, y = decision.waypoint
        x_min, x_max, y_min, y_max = self.bounds_xy
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            raise ValueError("adversary waypoint is outside target boundary")
        self.intent = {
            "break_contact": HiddenIntent.WITHDRAW,
            "reposition": HiddenIntent.TRANSIT,
            "deception": HiddenIntent.EVADE,
            "silent_transit": HiddenIntent.TRANSIT,
            "evade": HiddenIntent.EVADE,
            "hold_course": HiddenIntent.PATROL,
        }[decision.intent]
        self._desired_heading_rad = decision.heading
        self._desired_speed_mps = decision.speed
        self._desired_waypoint = (float(x), float(y))
        self._adversary_hold_steps = max(1, hold_steps)
        self._evasion_elapsed_s = 0.0
        self._pending_decoy_count = min(decision.decoy_count, self.decoy_inventory)
        self.decoy_inventory -= self._pending_decoy_count
        self._belief_uncertainty_m = min(2_000.0, self._belief_uncertainty_m + 25.0)

    def consume_decoy_request(self) -> int:
        """Consume the validated decoy request for engine-side deployment."""
        count = self._pending_decoy_count
        self._pending_decoy_count = 0
        return count

    @property
    def maneuver_command(self) -> TargetManeuverCommand | None:
        """Return the active adversary/evasion command until its expiry."""
        if self._adversary_hold_steps <= 0:
            return None
        return TargetManeuverCommand(
            desired_heading_rad=self._desired_heading_rad,
            desired_speed_mps=self._desired_speed_mps,
            max_turn_rate_rad_s=self.max_turn_rate_rad_s,
            max_acceleration_mps2=self.max_acceleration_mps2,
            remaining_steps=self._adversary_hold_steps,
        )

    def _scaled_velocity(self, intent: HiddenIntent, heading: float) -> tuple[float, float]:
        speed = self._intent_speed(intent)
        return (speed * math.cos(heading), speed * math.sin(heading))

    def _intent_velocity(self, intent: HiddenIntent) -> tuple[float, float]:
        """Normalized INTENT_VELOCITIES direction scaled by the intent speed."""
        dx, dy = INTENT_VELOCITIES[intent]
        scale = self._intent_speed(intent) / max(math.hypot(dx, dy), 1e-9)
        return (dx * scale, dy * scale)

    def _intent_speed(self, intent: HiddenIntent) -> float:
        if self.intent_speed_mps is None:
            return math.hypot(*INTENT_VELOCITIES[intent])
        return self.intent_speed_mps.get(intent, 8.0)

    def public_kinematics(self) -> dict[str, object]:
        return {"target_id": self.target_id}

    def _sample_intent(self, rng: random.Random) -> HiddenIntent:
        row = TRANSITION_PROBABILITIES[self.intent]
        draw = rng.random()
        cumulative = 0.0
        for next_intent, probability in row.items():
            cumulative += probability
            if draw < cumulative:
                return next_intent
        return self.intent

    def _reflect_into_bounds(self) -> None:
        x, y = self.position_xy
        vx, vy = self.velocity_xy
        x_min, x_max, y_min, y_max = self.bounds_xy
        if x < x_min:
            x, vx = x_min + (x_min - x), -vx
        elif x > x_max:
            x, vx = x_max - (x - x_max), -vx
        if y < y_min:
            y, vy = y_min + (y_min - y), -vy
        elif y > y_max:
            y, vy = y_max - (y - y_max), -vy
        self.position_xy = (x, y)
        self.velocity_xy = (vx, vy)
        self._desired_heading_rad = math.atan2(vy, vx)
