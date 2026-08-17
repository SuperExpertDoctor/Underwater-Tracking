"""Explicit operational doctrine shared by the master and slave brains."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


FiniteUnit = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
PositiveSeconds = Annotated[int, Field(gt=0)]


class DoctrineConfig(BaseModel):
    """Hard doctrine knobs; LLMs may choose within these limits only."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    passive_continuous: bool = True
    active_only_on_exception: bool = True
    active_quality_floor: FiniteUnit = 0.40
    active_covariance_growth_factor: float = Field(
        default=1.25, ge=1.0, allow_inf_nan=False
    )
    active_background_noise_db: float = Field(default=6.0, allow_inf_nan=False)
    target_lost_after_s: PositiveSeconds = 300
    handoff_lead_time_s: PositiveSeconds = 600
    rotation_energy_threshold: FiniteUnit = 0.30
    max_active_exposure_cost: FiniteUnit = 0.60
    require_connected_emitter_receiver: bool = True
    usv_support_radius_is_hard_limit: bool = True
    local_autonomy_when_disconnected: bool = True

    @model_validator(mode="after")
    def active_thresholds_are_coherent(self) -> DoctrineConfig:
        if self.active_only_on_exception and not self.passive_continuous:
            raise ValueError("exception-only active doctrine requires continuous passive listening")
        if self.active_quality_floor <= 0.0 and self.active_covariance_growth_factor <= 1.0:
            raise ValueError("active doctrine must retain at least one uncertainty trigger")
        return self


__all__ = ["DoctrineConfig"]
