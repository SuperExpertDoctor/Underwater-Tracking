import math
import random
from dataclasses import dataclass, field
from enum import StrEnum

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


@dataclass(slots=True)
class TargetEntity:
    target_id: str
    position_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    intent: HiddenIntent
    bounds_xy: tuple[float, float, float, float] = DEFAULT_BOUNDS_XY
    intent_speed_mps: dict[HiddenIntent, float] | None = None
    max_speed_mps: float = 14.0
    max_acceleration_mps2: float = 0.08
    max_turn_rate_rad_s: float = math.pi / 300.0
    _desired_heading_rad: float = field(init=False, repr=False)
    _desired_speed_mps: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._desired_heading_rad = math.atan2(self.velocity_xy[1], self.velocity_xy[0])
        self._desired_speed_mps = math.hypot(*self.velocity_xy)

    def step(self, dt_s: float, rng: random.Random) -> None:
        next_intent = self._sample_intent(rng)
        if next_intent is not self.intent:
            self.intent = next_intent
            direction = INTENT_VELOCITIES[next_intent]
            self._desired_heading_rad = math.atan2(direction[1], direction[0])
            self._desired_speed_mps = self._intent_speed(next_intent)
        current_speed = math.hypot(*self.velocity_xy)
        current_heading = math.atan2(self.velocity_xy[1], self.velocity_xy[0])
        limits = MotionLimits(
            max_speed_mps=self.max_speed_mps,
            max_acceleration_mps2=self.max_acceleration_mps2,
            max_turn_rate_rad_s=self.max_turn_rate_rad_s,
        )
        end = advance_motion(
            MotionState(self.position_xy, current_heading, current_speed),
            MotionCommand(self._desired_heading_rad, self._desired_speed_mps),
            limits,
            dt_s,
        )
        self.position_xy = end.position_xy
        self.velocity_xy = (
            end.speed_mps * math.cos(end.heading_rad),
            end.speed_mps * math.sin(end.heading_rad),
        )
        self._reflect_into_bounds()

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
