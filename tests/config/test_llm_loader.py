from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from underwater_tracking.config.loader import load_app_config


_SCENARIO_YAML = """
scenario:
  uuv_count: 12
  initial_target_count: 2
  max_target_count: 4
  duration_s: 28800
  seed: 42
timing:
  physics_step_s: 10
  observation_step_s: 30
  group_report_s: 300
  progress_report_s: 600
  strategic_review_s: 900
  prediction_horizon_s: 1800
"""

_TRACKING_YAML = """
group_min_size: 2
group_max_size: 4
quality_warning: 0.65
quality_critical: 0.40
quality_release: 0.75
quality_window_s: 300
release_hold_s: 600
"""


def _role_payload(role: str) -> dict[str, object]:
    return {
        "role": role,
        "model": f"model-{role}",
        "base_url": "https://api.example.com/v1",
        "temperature": 0.2,
        "request_timeout_s": 180.0,
        "connect_timeout_s": 10.0,
        "max_tokens": 4096,
        "max_retries": 3,
        "backoff_base_s": 1.0,
        "backoff_max_s": 60.0,
        "prompt_version": f"{role}-v1",
    }


def _llm_payload() -> dict[str, object]:
    return {
        "model": "legacy-provider-model",
        "base_url": "https://api.example.com/v1",
        "api_key_env": "UNDERWATER_TRACKING_API_KEY",
        "temperature": 0.2,
        "request_timeout_s": 180.0,
        "connect_timeout_s": 10.0,
        "max_tokens": 4096,
        "max_retries": 3,
        "backoff_base_s": 1.0,
        "backoff_max_s": 60.0,
        "roles": {
            role: _role_payload(role) for role in ("master", "slave", "adversary")
        },
    }


def _write_config_tree(tmp_path: Path, llm_payload: dict[str, object]) -> Path:
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    (tmp_path / "tracking.yaml").write_text(_TRACKING_YAML, encoding="utf-8")
    (tmp_path / "llm.yaml").write_text(
        yaml.safe_dump(llm_payload, sort_keys=False), encoding="utf-8"
    )
    scenario_path = scenario_dir / "default.yaml"
    scenario_path.write_text(_SCENARIO_YAML, encoding="utf-8")
    return scenario_path


def test_loader_loads_and_round_trips_explicit_three_role_config(tmp_path):
    config = load_app_config(_write_config_tree(tmp_path, _llm_payload()))

    assert config.llm is not None
    assert config.llm.roles is not None
    assert config.llm.for_role("master").prompt_version == "master-v1"
    assert config.llm.for_role("slave").model == "model-slave"
    assert config.llm.for_role("adversary").base_url == "https://api.example.com/v1"
    assert config.llm.__class__.model_validate(config.llm.model_dump()) == config.llm


def test_loader_preserves_legacy_flat_llm_config(tmp_path):
    payload = _llm_payload()
    del payload["roles"]

    config = load_app_config(_write_config_tree(tmp_path, payload))

    assert config.llm is not None
    assert config.llm.roles is None
    assert config.llm.model == "legacy-provider-model"


def test_loader_rejects_unknown_role_at_startup(tmp_path):
    payload = _llm_payload()
    roles = payload["roles"]
    assert isinstance(roles, dict)
    roles["unknown"] = _role_payload("unknown")

    with pytest.raises(ValidationError):
        load_app_config(_write_config_tree(tmp_path, payload))


def test_loader_rejects_unknown_role_field_at_startup(tmp_path):
    payload = _llm_payload()
    roles = payload["roles"]
    assert isinstance(roles, dict)
    roles["master"]["unknown_field"] = True

    with pytest.raises(ValidationError):
        load_app_config(_write_config_tree(tmp_path, payload))
