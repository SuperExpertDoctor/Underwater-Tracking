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


class LLMConfig(StrictModel):
    """Provider-neutral LLM client settings (spec 22).

    ``api_key_env`` names the environment variable holding the API key; the
    key itself is read at call time and never stored in configuration.
    """

    model: str = "underwater-assistant-model"
    base_url: str = "https://api.example.com/v1"
    api_key_env: str = "UNDERWATER_TRACKING_API_KEY"
    temperature: float = Field(default=0.2, ge=0, le=2)
    request_timeout_s: float = Field(default=60.0, gt=0)
    connect_timeout_s: float = Field(default=10.0, gt=0)


class AppConfig(StrictModel):
    scenario: ScenarioConfig
    timing: TimingConfig
    tracking: TrackingConfig
    # Optional additive sections: present only when ``agent.yaml`` /
    # ``llm.yaml`` exist next to ``tracking.yaml`` (see loader).
    agent: AgentConfig | None = None
    llm: LLMConfig | None = None
