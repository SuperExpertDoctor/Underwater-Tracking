# src/underwater_tracking/config/models.py
from math import isfinite, pi
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from underwater_tracking.config.doctrine import DoctrineConfig
from underwater_tracking.config.platform_core import (
    CommunicationsConfig,
    EnvironmentConfig,
    PlatformCatalogConfig,
    PlatformCoreFiles,
    SensorCatalogConfig,
    initial_route_join_distance,
)
from underwater_tracking.domain.models import (
    IntelligenceSource,
    OperationalScheme,
    SurveillanceCapability,
)
from underwater_tracking.domain.regional_models import GridSpec


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
_MemoryDecayHalfLife = Annotated[StrictFloat, Field(gt=0, le=31_536_000)]

# Shared defaults for the quality hysteresis policy. Runtime constructors
# receive values from TrackingConfig; these constants keep offline defaults
# aligned without conflating them with the active-sonar doctrine floor.
DEFAULT_QUALITY_WARNING = 0.65
DEFAULT_QUALITY_CRITICAL = 0.40
DEFAULT_QUALITY_RELEASE = 0.75
DEFAULT_RELEASE_HOLD_S = 600


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TimingConfig(StrictModel):
    physics_step_s: int = 5
    observation_step_s: int = 30
    group_report_s: int = 300
    progress_report_s: int = 600
    strategic_review_s: int = 900
    prediction_horizon_s: int = 1800
    demo_time_scale: float = Field(default=60.0, gt=0)


class TargetSearchPriorConfig(StrictModel):
    """Source-attributed public intelligence, separate from sensor estimates."""

    prior_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    source: IntelligenceSource
    issued_at_s: int = Field(ge=0)
    valid_until_s: int = Field(gt=0)
    center_xy: tuple[float, float]
    covariance_xy: tuple[tuple[float, float], tuple[float, float]]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_geometry_and_validity(self) -> "TargetSearchPriorConfig":
        if self.valid_until_s <= self.issued_at_s:
            raise ValueError("valid_until_s must be after issued_at_s")
        if not all(isfinite(value) for value in self.center_xy):
            raise ValueError("target prior center_xy must contain finite values")
        p00, p01 = self.covariance_xy[0]
        p10, p11 = self.covariance_xy[1]
        if not all(isfinite(value) for value in (p00, p01, p10, p11)):
            raise ValueError("target prior covariance_xy must contain finite values")
        if p01 != p10:
            raise ValueError("target prior covariance_xy must be symmetric")
        if p00 <= 0 or p11 <= 0 or p00 * p11 - p01 * p10 <= 0:
            raise ValueError("target prior covariance_xy must be positive definite")
        return self


class ScenarioConfig(StrictModel):
    scenario_id: str = "underwater-default"
    uuv_count: int = Field(12, ge=2)
    initial_target_count: int = Field(1, ge=1)
    max_target_count: int = Field(4, ge=1)
    uuv_only: bool = False
    uuv_only_carrier_count: int = Field(default=4, ge=1)
    home_battle_group_id: str = Field(default="carrier_battle_group_01", min_length=1)
    region_entry_probability_threshold: float = Field(default=0.70, ge=0, le=1)
    region_transition_confirm_cycles: int = Field(default=2, ge=1)
    duration_s: int = Field(28_800, gt=0)
    seed: int = 42
    initial_decoy_count: int = Field(default=0, ge=0)
    operational_scheme: OperationalScheme | None = None
    platform_core: PlatformCoreFiles | None = None
    target_search_priors: tuple[TargetSearchPriorConfig, ...] = ()
    observability_feedback_config: str = "configs/observability_feedback.yaml"


class TrackingConfig(StrictModel):
    group_min_size: int = 2
    group_max_size: int = 4
    quality_warning: float = DEFAULT_QUALITY_WARNING
    quality_critical: float = DEFAULT_QUALITY_CRITICAL
    quality_release: float = DEFAULT_QUALITY_RELEASE
    quality_window_s: int = 300
    release_hold_s: int = DEFAULT_RELEASE_HOLD_S
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
    # Truth-free formation-slot correction adapted from the pure-Python
    # multi-UUV controller. The carrier planner still owns allocation and
    # safety validation; this only shapes already-generated waypoints.
    formation_enabled: bool = True
    formation_radius_m: float = Field(default=800.0, gt=0)
    formation_horizon_s: float = Field(default=120.0, gt=0)
    formation_max_endpoint_correction_m: float = Field(default=400.0, ge=0)

    grid: GridSpec = Field(default_factory=GridSpec)

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


class RuntimeRetentionConfig(StrictModel):
    """Upper bounds for in-process state that grows during long runs."""

    group_checkpoint_limit: int = Field(default=2, gt=0)
    belief_history_limit: int = Field(default=64, gt=0)
    event_history_limit: int = Field(default=2048, gt=0)
    mission_event_history_limit: int = Field(default=2048, gt=0)
    processed_event_limit: int = Field(default=4096, gt=0)
    payload_cache_limit: int = Field(default=64, gt=0)
    payload_db_limit: int = Field(default=256, gt=0)
    conversation_turn_limit: int = Field(default=128, gt=0)
    directive_job_limit: int = Field(default=256, gt=0)


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
    retention: RuntimeRetentionConfig = Field(default_factory=RuntimeRetentionConfig)


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


class KnowledgeConfig(StrictModel):
    """Ontology knowledge-service settings used during strategic adjustments."""

    enabled: bool = True
    base_url: _LLMBaseURL = "http://172.17.27.172:9642"
    query_path: _LLMNonEmptyString = "/api/query"
    mode: Literal["mix", "hybrid", "local", "global", "naive"] = "mix"
    include_trace: bool = True
    request_timeout_s: _LLMTimeout = 15.0
    max_retries: _LLMRetries = 3
    backoff_base_s: _LLMBackoff = 1.0
    backoff_max_s: _LLMBackoff = 8.0

    @model_validator(mode="after")
    def validate_backoff(self) -> "KnowledgeConfig":
        if self.backoff_base_s > self.backoff_max_s:
            raise ValueError("knowledge backoff_max_s must not be below backoff_base_s")
        return self


class MemoryConfig(StrictModel):
    """Strict configuration for the real asynchronous memory pipeline."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    enabled: StrictBool = True
    poll_interval_s: _LLMTimeout = 2.0
    source_poll_interval_s: _LLMTimeout = 2.0
    maintenance_interval_s: _LLMTimeout = 300.0
    short_term_message_threshold: Annotated[StrictInt, Field(ge=1, le=1024)] = 12
    short_term_token_threshold: _LLMMaxTokens = 6000
    short_term_compress_interval_s: _LLMTimeout = 300.0
    recent_message_limit: Annotated[StrictInt, Field(ge=1, le=1024)] = 12
    context_token_budget: _LLMMaxTokens = 4000
    retrieval_top_k: Annotated[StrictInt, Field(ge=1, le=128)] = 8
    retrieval_candidate_limit: Annotated[StrictInt, Field(ge=1, le=1024)] = 32
    min_importance_score: Annotated[StrictFloat, Field(ge=0, le=1)] = 0.3
    archive_threshold: Annotated[StrictFloat, Field(ge=0, le=1)] = 0.1
    decay_half_life_s: _MemoryDecayHalfLife = 2_592_000.0
    work_lease_timeout_s: _LLMTimeout = 120.0
    max_attempts: _LLMRetries = 3
    retry_backoff_s: _LLMBackoff = 2.0
    # Local SentenceTransformer embeddings are the production default. The
    # HTTP provider remains an explicit compatibility option for migrations
    # and isolated provider-contract tests; it is never an implicit fallback.
    embedding_provider: Literal["sentence_transformers", "http"] = "sentence_transformers"
    embedding_base_url: _LLMBaseURL | None = None
    embedding_model: _LLMNonEmptyString | None = None
    embedding_api_key_env: _LLMNonEmptyString = "UNDERWATER_TRACKING_API_KEY"
    embedding_timeout_s: _LLMTimeout = 30.0
    embedding_vector_version: _LLMNonEmptyString = "v1"
    embedding_local_files_only: StrictBool = True
    embedding_device: _LLMNonEmptyString = "cpu"
    embedding_normalize: StrictBool = True

    @classmethod
    def degraded(cls) -> "MemoryConfig":
        """Return the explicit no-memory configuration for unavailable runtime wiring.

        The disabled contract deliberately contains no embedding endpoint or
        model, so later runtime code can report degradation without invoking a
        local, mock, or substitute embedding implementation.
        """

        return cls(enabled=False)

    @model_validator(mode="after")
    def validate_memory_limits(self) -> "MemoryConfig":
        if self.enabled and self.embedding_model is None:
            raise ValueError(
                "enabled memory config requires embedding_model"
            )
        if self.enabled and self.embedding_provider == "http" and self.embedding_base_url is None:
            raise ValueError(
                "enabled http memory config requires embedding_base_url and embedding_model"
            )
        if self.embedding_provider == "sentence_transformers" and not self.embedding_local_files_only:
            raise ValueError(
                "sentence_transformers provider requires embedding_local_files_only=true"
            )
        if self.retrieval_candidate_limit < self.retrieval_top_k:
            raise ValueError("retrieval_candidate_limit must be at least retrieval_top_k")
        return self


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
    knowledge: KnowledgeConfig | None = None
    memory: MemoryConfig | None = None

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
        if self.scenario.uuv_only and not self.environment.uuv_only:
            raise ValueError("uuv-only scenario requires a uuv-only environment")
        if self.scenario.uuv_only and len(self.environment.carriers) < 1:
            raise ValueError("uuv-only scenario requires at least one carrier")
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
            turn_radius_m = submarine.speed_mps / profile.max_turn_rate_rad_s
            join_distance_m = initial_route_join_distance(
                submarine,
                profile.max_turn_rate_rad_s,
            )
            if join_distance_m > turn_radius_m + 1e-6:
                raise ValueError(
                    f"submarine {submarine.target_id!r} cannot join the first mission "
                    f"route segment within its turn radius {turn_radius_m:.3f} m"
                )
        prior_ids = [prior.prior_id for prior in self.scenario.target_search_priors]
        if len(prior_ids) != len(set(prior_ids)):
            raise ValueError("duplicate prior_id in target_search_priors")
        target_ids = {submarine.target_id for submarine in self.environment.submarines}
        for prior in self.scenario.target_search_priors:
            if prior.target_id not in target_ids:
                raise ValueError(f"unknown target in target_search_priors: {prior.target_id!r}")
            min_x, max_x, min_y, max_y = self.environment.map_bounds_xy
            x, y = prior.center_xy
            if not min_x <= x <= max_x or not min_y <= y <= max_y:
                raise ValueError(
                    f"target prior center for {prior.prior_id!r} is outside map bounds"
                )
        return self
