# src/underwater_tracking/domain/truth.py
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TargetTruth:
    target_id: str
    position_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    intent_label: str
