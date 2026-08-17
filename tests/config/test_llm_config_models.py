from math import inf, nan

import pytest
from pydantic import ValidationError

from underwater_tracking.config.models import LLMConfig


_ROLE_NAMES = ("master", "slave", "adversary")


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


def _roles_payload() -> dict[str, dict[str, object]]:
    return {role: _role_payload(role) for role in _ROLE_NAMES}


def test_llm_config_exposes_each_strict_role_for_future_client_building():
    config = LLMConfig(roles=_roles_payload())

    assert config.roles is not None
    assert set(config.roles) == set(_ROLE_NAMES)
    assert config.for_role("master") is config.roles["master"]
    assert config.for_role("slave").prompt_version == "slave-v1"
    assert config.for_role("adversary").model == "model-adversary"


def test_llm_config_keeps_legacy_flat_values_without_roles():
    config = LLMConfig(model="legacy-model", temperature=0.7)

    assert config.roles is None
    assert config.model == "legacy-model"
    with pytest.raises(ValueError, match="roles are not configured"):
        config.for_role("master")


@pytest.mark.parametrize("role", ["observer", "", "MASTER"])
def test_llm_config_rejects_unknown_role_name(role: str):
    roles = _roles_payload()
    roles[role] = _role_payload(role)

    with pytest.raises(ValidationError):
        LLMConfig(roles=roles)


def test_llm_config_rejects_missing_required_role():
    roles = _roles_payload()
    del roles["adversary"]

    with pytest.raises(ValidationError, match="missing required roles"):
        LLMConfig(roles=roles)


def test_llm_config_rejects_role_key_and_role_field_mismatch():
    roles = _roles_payload()
    roles["master"] = _role_payload("slave")

    with pytest.raises(ValidationError, match="does not match mapping key"):
        LLMConfig(roles=roles)


def test_llm_config_rejects_unknown_role_field():
    roles = _roles_payload()
    roles["master"]["bogus"] = True

    with pytest.raises(ValidationError):
        LLMConfig(roles=roles)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", "0.2"),
        ("request_timeout_s", nan),
        ("connect_timeout_s", inf),
        ("max_tokens", 0),
        ("max_retries", -1),
        ("backoff_base_s", 0.0),
        ("backoff_max_s", 0.0),
        ("prompt_version", ""),
    ],
)
def test_llm_config_rejects_invalid_role_parameter(field: str, value: object):
    roles = _roles_payload()
    roles["master"][field] = value

    with pytest.raises(ValidationError):
        LLMConfig(roles=roles)


def test_llm_config_rejects_backoff_max_below_backoff_base():
    roles = _roles_payload()
    roles["master"]["backoff_base_s"] = 10.0
    roles["master"]["backoff_max_s"] = 5.0

    with pytest.raises(ValidationError, match="backoff_max_s"):
        LLMConfig(roles=roles)


def test_llm_config_rejects_unknown_role_at_lookup_boundary():
    config = LLMConfig(roles=_roles_payload())

    with pytest.raises(ValueError, match="unknown LLM role"):
        config.for_role("observer")


def test_llm_config_rejects_non_finite_legacy_numeric_value():
    with pytest.raises(ValidationError):
        LLMConfig(request_timeout_s=nan)
