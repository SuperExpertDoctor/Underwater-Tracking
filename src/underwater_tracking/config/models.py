# src/underwater_tracking/config/models.py
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TimingConfig(StrictModel):
    physics_step_s: int = 10
    observation_step_s: int = 30
    group_report_s: int = 300
    progress_report_s: int = 600
    strategic_review_s: int = 900
    prediction_horizon_s: int = 1800


class ScenarioConfig(StrictModel):
    uuv_count: int = Field(12, ge=2)
    initial_target_count: int = Field(2, ge=1)
    max_target_count: int = Field(4, ge=1)
    duration_s: int = Field(28_800, gt=0)
    seed: int = 42


class TrackingConfig(StrictModel):
    group_min_size: int = 2
    group_max_size: int = 4
    quality_warning: float = 0.65
    quality_critical: float = 0.40
    quality_release: float = 0.75
    quality_window_s: int = 300
    release_hold_s: int = 600
    # Quality normalization reference scales, calibrated to the default
    # scenario: 1 km observer standoff with 1e-3 rad^2 bearing variance.
    covariance_reference_m2: float = Field(default=10_000.0, gt=0)
    fim_min_eigenvalue_reference: float = Field(default=1e-3, gt=0)
    fim_condition_reference: float = Field(default=100.0, gt=1)

    @model_validator(mode="after")
    def validate_group_sizes(self):
        if self.group_min_size > self.group_max_size:
            raise ValueError("group_min_size must not exceed group_max_size")
        return self


class AppConfig(StrictModel):
    scenario: ScenarioConfig
    timing: TimingConfig
    tracking: TrackingConfig
