from __future__ import annotations

from math import pi
from typing import Annotated

from pydantic import Field, field_validator

from underwater_tracking.domain.models import StrictModel

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class PassiveSonarObservation(StrictModel):
    observation_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    observer_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    azimuth_rad: float = Field(allow_inf_nan=False)
    variance_rad2: PositiveFloat
    detection_confidence: UnitFloat
    snr_db: float = Field(allow_inf_nan=False)

    @field_validator("azimuth_rad")
    @classmethod
    def wrap_azimuth(cls, value: float) -> float:
        return (value + pi) % (2.0 * pi) - pi


class ActiveTransmission(StrictModel):
    transmission_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    emitter_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)


class MultistaticObservation(StrictModel):
    observation_id: str = Field(min_length=1)
    transmission_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    emitter_id: str = Field(min_length=1)
    receiver_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    bistatic_range_m: PositiveFloat
    receiver_azimuth_rad: float = Field(allow_inf_nan=False)
    range_variance_m2: PositiveFloat
    bearing_variance_rad2: PositiveFloat
    detection_confidence: UnitFloat

    @field_validator("receiver_azimuth_rad")
    @classmethod
    def wrap_azimuth(cls, value: float) -> float:
        return (value + pi) % (2.0 * pi) - pi
