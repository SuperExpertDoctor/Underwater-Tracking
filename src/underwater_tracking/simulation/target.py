import random
from dataclasses import dataclass
from enum import StrEnum


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

    def step(self, dt_s: float, rng: random.Random) -> None:
        next_intent = self._sample_intent(rng)
        if next_intent is not self.intent:
            self.intent = next_intent
            self.velocity_xy = INTENT_VELOCITIES[next_intent]
        x, y = self.position_xy
        vx, vy = self.velocity_xy
        self.position_xy = (x + vx * dt_s, y + vy * dt_s)
        self._reflect_into_bounds()

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
