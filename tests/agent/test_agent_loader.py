# tests/agent/test_agent_loader.py
"""Regression net for the additive agent/llm config loading path.

``load_app_config`` must keep its original contract (scenario YAML plus the
sibling ``tracking.yaml``) while additively merging ``agent.yaml`` /
``llm.yaml`` when they exist next to ``tracking.yaml``; the new
``AgentConfig`` / ``LLMConfig`` validators must reject invalid input. This
pins the Task 1 loader extension and the brief's Step-4 values without
touching ``tests/config/test_loader.py``.
"""
import pytest
from pydantic import ValidationError

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AgentConfig, IntentChangeConfirmation, TrackingConfig
from underwater_tracking.domain.models import SurveillanceCapability

_SCENARIO_YAML = (
    "scenario:\n"
    "  uuv_count: 12\n"
    "  initial_target_count: 2\n"
    "  max_target_count: 4\n"
    "  duration_s: 28800\n"
    "  seed: 42\n"
    "timing:\n"
    "  physics_step_s: 10\n"
    "  observation_step_s: 30\n"
    "  group_report_s: 300\n"
    "  progress_report_s: 600\n"
    "  strategic_review_s: 900\n"
    "  prediction_horizon_s: 1800\n"
)

_TRACKING_YAML = (
    "group_min_size: 2\n"
    "group_max_size: 4\n"
    "quality_warning: 0.65\n"
    "quality_critical: 0.40\n"
    "quality_release: 0.75\n"
    "quality_window_s: 300\n"
    "release_hold_s: 600\n"
    "covariance_reference_m2: 10000.0\n"
    "fim_min_eigenvalue_reference: 0.001\n"
    "fim_condition_reference: 100.0\n"
)

_AGENT_YAML = (
    "transport_retries: 3\n"
    "semantic_repairs: 2\n"
    "quality_warning_persist_s: 120\n"
    "quality_critical_persist_s: 30\n"
    "event_cooldown_s: 300\n"
    "history_token_threshold: 6000\n"
    "intent_change_confirmation:\n"
    "  confidence: 0.70\n"
    "  margin: 0.15\n"
    "  consecutive: 2\n"
)

_LLM_YAML = (
    'model: "underwater-assistant-model"\n'
    'base_url: "https://api.example.com/v1"\n'
    'api_key_env: "UNDERWATER_TRACKING_API_KEY"\n'
    "temperature: 0.2\n"
    "request_timeout_s: 60.0\n"
    "connect_timeout_s: 10.0\n"
)


def _write_tree(tmp_path, *, agent_yaml=None, llm_yaml=None):
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    (tmp_path / "tracking.yaml").write_text(_TRACKING_YAML, encoding="utf-8")
    (scenario_dir / "default.yaml").write_text(_SCENARIO_YAML, encoding="utf-8")
    if agent_yaml is not None:
        (tmp_path / "agent.yaml").write_text(agent_yaml, encoding="utf-8")
    if llm_yaml is not None:
        (tmp_path / "llm.yaml").write_text(llm_yaml, encoding="utf-8")
    return scenario_dir / "default.yaml"


def test_optional_sections_load_with_brief_step4_values(tmp_path):
    config = load_app_config(
        _write_tree(tmp_path, agent_yaml=_AGENT_YAML, llm_yaml=_LLM_YAML)
    )
    assert config.agent is not None
    assert config.llm is not None
    agent = config.agent
    assert agent.transport_retries == 3
    assert agent.semantic_repairs == 2
    assert agent.quality_warning_persist_s == 120
    assert agent.quality_critical_persist_s == 30
    assert agent.event_cooldown_s == 300
    assert agent.history_token_threshold == 6000
    confirmation = agent.intent_change_confirmation
    assert confirmation.confidence == 0.70
    assert confirmation.margin == 0.15
    assert confirmation.consecutive == 2
    assert config.llm.api_key_env == "UNDERWATER_TRACKING_API_KEY"
    assert config.llm.temperature == 0.2


def test_absent_optional_files_leave_sections_none(tmp_path):
    config = load_app_config(_write_tree(tmp_path))
    assert config.agent is None
    assert config.llm is None


def test_unknown_agent_key_is_rejected(tmp_path):
    config_path = _write_tree(
        tmp_path, agent_yaml="transport_retries: 9\nbogus: 1\n", llm_yaml="model: m\n"
    )
    with pytest.raises(ValidationError):
        load_app_config(config_path)


def test_default_config_values_unchanged_by_loader_extension():
    config = load_app_config("configs/scenario/default.yaml")
    assert config.scenario.uuv_count == 12
    assert config.scenario.initial_target_count == 2
    assert config.scenario.max_target_count == 4
    assert config.scenario.duration_s == 28800
    assert config.scenario.seed == 42
    assert config.timing.physics_step_s == 10
    assert config.timing.observation_step_s == 30
    assert config.timing.group_report_s == 300
    assert config.timing.progress_report_s == 600
    assert config.timing.strategic_review_s == 900
    assert config.timing.prediction_horizon_s == 1800
    assert config.tracking.group_min_size == 2
    assert config.tracking.group_max_size == 4
    assert config.tracking.quality_warning == 0.65
    assert config.tracking.quality_critical == 0.40
    assert config.tracking.quality_release == 0.75
    assert config.tracking.quality_window_s == 300
    assert config.tracking.release_hold_s == 600
    assert config.agent is not None
    assert config.llm is not None


def test_default_config_loads_adaptive_tracking_scheme_and_capability_profiles():
    config = load_app_config("configs/scenario/default.yaml")
    assert config.scenario.operational_scheme is not None
    assert config.scenario.operational_scheme.minimum_quality["target_00"] == 0.75
    assert config.tracking.uuv_capabilities is not None
    assert set(config.tracking.uuv_capabilities) == {"uuv_00", "uuv_01"}
    assert config.tracking.uuv_capabilities["uuv_01"].active_sonar_available is False
    assert config.tracking.uuv_capabilities["uuv_00"].endurance_s == 28_800.0
    assert config.tracking.uuv_capabilities["uuv_01"].availability == 0.85


def test_tracking_config_rejects_empty_uuv_capability_mapping_id():
    with pytest.raises(ValidationError):
        TrackingConfig(uuv_capabilities={"": SurveillanceCapability()})


def test_agent_config_defaults_match_brief_step4_values():
    agent = AgentConfig()
    assert agent.transport_retries == 3
    assert agent.semantic_repairs == 2
    assert agent.quality_warning_persist_s == 120
    assert agent.quality_critical_persist_s == 30
    assert agent.event_cooldown_s == 300
    assert agent.history_token_threshold == 6000
    assert agent.intent_change_confirmation.confidence == 0.70
    assert agent.intent_change_confirmation.margin == 0.15
    assert agent.intent_change_confirmation.consecutive == 2


def test_intent_change_confirmation_rejects_inconsistent_margin():
    with pytest.raises(ValidationError):
        IntentChangeConfirmation(confidence=0.9, margin=0.2, consecutive=2)
    IntentChangeConfirmation(confidence=0.9, margin=0.1, consecutive=2)
