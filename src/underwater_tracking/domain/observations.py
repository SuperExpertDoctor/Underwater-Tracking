from __future__ import annotations

from math import pi
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class _ObservationModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class PassiveSonarObservation(_ObservationModel):
    observation_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    observer_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    azimuth_rad: float = Field(allow_inf_nan=False)
    variance_rad2: PositiveFloat
    detection_confidence: UnitFloat
    snr_db: float = Field(allow_inf_nan=False)
    is_false_alarm: bool = False
    observer_position_xy: tuple[float, float] | None = None

    @field_validator("observer_position_xy", mode="before")
    @classmethod
    def normalize_observer_position(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @field_validator("azimuth_rad")
    @classmethod
    def wrap_azimuth(cls, value: float) -> float:
        return (value + pi) % (2.0 * pi) - pi
