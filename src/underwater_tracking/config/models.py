# src/underwater_tracking/config/models.py
from math import pi
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, model_validator

from underwater_tracking.config.doctrine import DoctrineConfig
from underwater_tracking.config.platform_core import (
    CommunicationsConfig,
    EnvironmentConfig,
    PlatformCatalogConfig,
    PlatformCoreFiles,
    SensorCatalogConfig,
)
from underwater_tracking.domain.models import OperationalScheme, SurveillanceCapability


_NonEmptyUUVId = Annotated[str, Field(min_length=1)]
LLMRoleName = Literal["master", "slave", "adversary"]
_LLM_ROLE_NAMES = frozenset({"master", "slave", "adversary"})
_LLMNonEmptyString = Annotated[StrictStr, Field(min_length=1)]
_LLMBaseURL = Annotated[StrictStr, Field(min_length=1, pattern=r"^https?://\S+$")]
_LLMTemperature = Annotated[StrictFloat, Field(ge=0, le=2)]
_LLMTimeout = Annotated[StrictFloat, Field(gt=0, le=86_400)]
_LLMMaxTokens = Annotated[StrictInt, Field(ge=1, le=1_000_000)]
_LLMRetries = Annotated[StrictInt, Field(ge=0, le=32)]
_LLMBackoff = Annotated[StrictFloat, Field(gt=0, le=86_400)]


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
    scenario_id: str = "underwater-default"
    uuv_count: int = Field(12, ge=2)
    initial_target_count: int = Field(2, ge=1)
    max_target_count: int = Field(4, ge=1)
    duration_s: int = Field(28_800, gt=0)
    seed: int = 42
    initial_decoy_count: int = Field(default=0, ge=0)
    operational_scheme: OperationalScheme | None = None
    platform_core: PlatformCoreFiles | None = None


class TrackingConfig(StrictModel):
    group_min_size: int = 2
    group_max_size: int = 4
    quality_warning: float = 0.65
    quality_critical: float = 0.40
    quality_release: float = 0.75
    quality_window_s: int = 300
    release_hold_s: int = 600
    # Motion realism (spec 5.1 amendment, R2): the UUV fleet tops out at
    # 4 m/s with a 3 deg/s turn rate, while submarines cruise at 8 m/s and
    # sprint at 14 m/s during evasion — the submarine pulls away from the
    # observer, so intent understanding + trajectory prediction drive
    # tracking feasibility.
    uuv_max_speed_mps: float = Field(default=4.0, gt=0)
    uuv_max_turn_rate_rad_s: float = Field(default=pi / 60.0, gt=0)
    submarine_cruise_speed_mps: float = Field(default=8.0, gt=0)
    submarine_sprint_speed_mps: float = Field(default=14.0, gt=0)
    submarine_turn_rate_rad_s: float = Field(default=pi / 300.0, gt=0)
    # Active-sonar probe model (spec 5.1/11.1 amendment, R5): range, noise,
    # ping cadence, energy cost, ping-heard probability, and the
    # classification probabilities of a contacted submarine vs a decoy.
    sensor_active_range_m: float = Field(default=3000.0, gt=0)
    sensor_active_range_sigma_m: float = Field(default=15.0, gt=0)
    sensor_active_bearing_sigma_rad: float = Field(default=0.003, gt=0)
    sensor_ping_interval_s: int = Field(default=30, gt=0)
    sensor_ping_energy_cost: float = Field(default=2e-4, gt=0)
    sensor_ping_heard_probability: float = Field(default=0.6, ge=0, le=1)
    sensor_active_classify_submarine_prob: float = Field(default=0.95, ge=0, le=1)
    sensor_active_classify_decoy_prob: float = Field(default=0.90, ge=0, le=1)
    # Decoy drift (spec 5.1 amendment, R5): a slow heading random walk.
    decoy_drift_speed_mps: float = Field(default=0.5, gt=0)
    decoy_heading_noise_rad_per_s: float = Field(default=0.02, gt=0)
    # Quality normalization reference scales, calibrated to the default
    # scenario: 1 km observer standoff with 1e-3 rad^2 bearing variance.
    covariance_reference_m2: float = Field(default=10_000.0, gt=0)
    fim_min_eigenvalue_reference: float = Field(default=1e-3, gt=0)
    fim_condition_reference: float = Field(default=100.0, gt=1)
    uuv_capabilities: dict[_NonEmptyUUVId, SurveillanceCapability] | None = None

    @model_validator(mode="after")
    def validate_group_sizes(self) -> "TrackingConfig":
        if self.group_min_size > self.group_max_size:
            raise ValueError("group_min_size must not exceed group_max_size")
        return self


class IntentChangeConfirmation(StrictModel):
    """Gates for confirming an intent-label change (spec 8.2).

    An intent change is confirmed when the leading hypothesis reaches
    ``confidence`` while leading the runner-up by at least ``margin``,
    for ``consecutive`` analyses in a row.
    """

    confidence: float = Field(default=0.70, ge=0, le=1)
    margin: float = Field(default=0.15, ge=0, le=1)
    consecutive: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def margin_must_fit_below_certainty(self) -> "IntentChangeConfirmation":
        if self.confidence + self.margin > 1.0:
            raise ValueError("confidence plus margin must not exceed 1")
        return self


class AgentConfig(StrictModel):
    """Carrier assistant defaults, validated against ``configs/agent.yaml``.

    ``transport_retries`` bounds exponential-backoff API retries (spec 8.3:
    timeout, rate limit, 5xx); ``semantic_repairs`` bounds schema/semantic
    error re-injection (spec 8.3); the persistence thresholds gate
    warning/critical quality events (spec 13); ``event_cooldown_s`` is the
    coalescing window for duplicate events (spec 8.2); the history token
    threshold triggers History compaction (spec 9).
    """

    transport_retries: int = Field(default=3, ge=0)
    semantic_repairs: int = Field(default=2, ge=0)
    quality_warning_persist_s: int = Field(default=120, gt=0)
    quality_critical_persist_s: int = Field(default=30, gt=0)
    event_cooldown_s: int = Field(default=300, gt=0)
    history_token_threshold: int = Field(default=6000, gt=0)
    intent_change_confirmation: IntentChangeConfirmation = Field(
        default_factory=IntentChangeConfirmation
    )


class LLMRoleConfig(StrictModel):
    """Independent client contract for one of the configured LLM roles."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    role: LLMRoleName
    model: _LLMNonEmptyString
    base_url: _LLMBaseURL
    temperature: _LLMTemperature
    request_timeout_s: _LLMTimeout
    connect_timeout_s: _LLMTimeout
    max_tokens: _LLMMaxTokens
    max_retries: _LLMRetries
    backoff_base_s: _LLMBackoff
    backoff_max_s: _LLMBackoff
    prompt_version: _LLMNonEmptyString

    @model_validator(mode="after")
    def validate_timeouts_and_backoff(self) -> "LLMRoleConfig":
        if self.connect_timeout_s > self.request_timeout_s:
            raise ValueError("connect_timeout_s must not exceed request_timeout_s")
        if self.backoff_base_s > self.backoff_max_s:
            raise ValueError("backoff_max_s must not be below backoff_base_s")
        return self


class LLMConfig(StrictModel):
    """Provider-neutral LLM client settings (spec 22, R1).

    ``api_key`` is the explicit LongCat key, resolved by the loader from
    ``configs/.env`` (git-ignored) when present; ``api_key_env`` names the
    environment variable that OVERRIDES it when set (env wins).
    ``max_tokens``, ``max_retries``, ``backoff_base_s`` and
    ``backoff_max_s`` make the client's hidden defaults explicit.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    model: _LLMNonEmptyString = "underwater-assistant-model"
    base_url: _LLMBaseURL = "https://api.example.com/v1"
    api_key: _LLMNonEmptyString | None = None
    api_key_env: _LLMNonEmptyString = "UNDERWATER_TRACKING_API_KEY"
    temperature: _LLMTemperature = 0.2
    request_timeout_s: _LLMTimeout = 60.0
    connect_timeout_s: _LLMTimeout = 10.0
    max_tokens: _LLMMaxTokens = 4096
    max_retries: _LLMRetries = 3
    backoff_base_s: _LLMBackoff = 1.0
    backoff_max_s: _LLMBackoff = 60.0
    roles: dict[LLMRoleName, LLMRoleConfig] | None = None

    @model_validator(mode="after")
    def validate_roles(self) -> "LLMConfig":
        if self.roles is None:
            return self
        missing = _LLM_ROLE_NAMES.difference(self.roles)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"missing required roles: {missing_names}")
        for role_name, role_config in self.roles.items():
            if role_config.role != role_name:
                raise ValueError(
                    f"role {role_config.role!r} does not match mapping key {role_name!r}"
                )
        return self

    def for_role(self, role: str) -> LLMRoleConfig:
        """Return a configured role for a future role-aware client builder."""

        if role not in _LLM_ROLE_NAMES:
            expected = ", ".join(sorted(_LLM_ROLE_NAMES))
            raise ValueError(f"unknown LLM role {role!r}; expected one of: {expected}")
        if self.roles is None:
            raise ValueError("LLM roles are not configured")
        return self.roles[cast(LLMRoleName, role)]


class AppConfig(StrictModel):
    scenario: ScenarioConfig
    timing: TimingConfig
    tracking: TrackingConfig
    # Optional additive sections: present only when ``agent.yaml`` /
    # ``llm.yaml`` exist next to ``tracking.yaml`` (see loader).
    agent: AgentConfig | None = None
    llm: LLMConfig | None = None
    environment: EnvironmentConfig | None = None
    platforms: PlatformCatalogConfig | None = None
    sensors: SensorCatalogConfig | None = None
    communications: CommunicationsConfig | None = None
    doctrine: DoctrineConfig | None = None

    @model_validator(mode="after")
    def platform_core_is_complete(self) -> "AppConfig":
        loaded = (self.environment, self.platforms, self.sensors, self.communications)
        if self.scenario.platform_core is None and all(value is None for value in loaded):
            return self
        if self.scenario.platform_core is None or any(value is None for value in loaded):
            raise ValueError("platform_core references and all loaded sections are required together")
        assert self.environment is not None
        assert self.platforms is not None
        assert self.sensors is not None
        assert self.communications is not None
        if self.scenario.uuv_count != len(self.environment.uuvs):
            raise ValueError("scenario uuv_count must equal explicit UUV roster size")
        if self.scenario.initial_target_count != len(self.environment.submarines):
            raise ValueError("scenario initial_target_count must equal explicit submarine roster size")
        if self.scenario.max_target_count != len(self.environment.submarines):
            raise ValueError("single-target max_target_count must equal explicit submarine roster size")
        for platform in (*self.environment.usvs, *self.environment.uuvs):
            if platform.motion_profile not in self.platforms.motion_profiles:
                raise ValueError(f"unknown motion profile {platform.motion_profile!r}")
            if platform.sensor_profile not in self.sensors.profiles:
                raise ValueError(f"unknown sensor profile {platform.sensor_profile!r}")
            if platform.communication_profile not in self.communications.profiles:
                raise ValueError(
                    f"unknown communication profile {platform.communication_profile!r}"
                )
        for submarine in self.environment.submarines:
            profile = self.platforms.motion_profiles.get(submarine.motion_profile)
            if profile is None:
                raise ValueError(f"unknown submarine motion profile {submarine.motion_profile!r}")
            if submarine.speed_mps > profile.max_speed_mps:
                raise ValueError(
                    f"submarine {submarine.target_id!r} initial speed_mps "
                    f"{submarine.speed_mps} exceeds motion profile "
                    f"{submarine.motion_profile!r} max_speed_mps {profile.max_speed_mps}"
                )
        return self
