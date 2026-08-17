from __future__ import annotations

import pytest

from underwater_tracking.agent.llm import HTTPStructuredLLM, LLMCallMetadata
from underwater_tracking.agent.llm_factory import (
    RoleHTTPStructuredLLM,
    build_role_llm,
)
from underwater_tracking.config.models import LLMConfig, LLMRoleConfig, LLMRoleName


_ROLES: tuple[LLMRoleName, ...] = ("master", "slave", "adversary")


def _role_payload(role: LLMRoleName) -> dict[str, object]:
    offset = _ROLES.index(role)
    return {
        "role": role,
        "model": f"model-{role}",
        "base_url": f"https://{role}.example.test/v{offset + 1}",
        "temperature": 0.2 + offset / 10,
        "request_timeout_s": 120.0 + offset,
        "connect_timeout_s": 10.0 + offset,
        "max_tokens": 2048 + offset * 1024,
        "max_retries": 2 + offset,
        "backoff_base_s": 1.0 + offset,
        "backoff_max_s": 30.0 + offset * 10,
        "prompt_version": f"{role}-prompt-v{offset + 1}",
    }


def _config() -> LLMConfig:
    return LLMConfig(
        api_key="configured-test-key",
        api_key_env="TEST_ROLE_LLM_KEY",
        roles={
            role: LLMRoleConfig.model_validate(_role_payload(role))
            for role in _ROLES
        },
    )


def test_build_role_llm_binds_every_role_specific_setting() -> None:
    config = _config()

    for role in _ROLES:
        client = build_role_llm(config, role)
        try:
            assert isinstance(client, HTTPStructuredLLM)
            assert isinstance(client, RoleHTTPStructuredLLM)
            role_config = config.for_role(role)

            assert client.role == role
            assert client.prompt_version == role_config.prompt_version
            assert client._model == role_config.model
            assert client._base_url == role_config.base_url
            assert client._temperature == role_config.temperature
            assert client._max_tokens == role_config.max_tokens
            assert client._max_attempts == max(1, role_config.max_retries)
            assert client._backoff_base_s == role_config.backoff_base_s
            assert client._backoff_max_s == role_config.backoff_max_s
            assert client._api_key_env == config.api_key_env
            assert client._api_key == config.api_key
        finally:
            client.close()


def test_build_role_llm_forwards_observability_context_and_hooks() -> None:
    def before(_metadata: LLMCallMetadata) -> None:
        pass

    def after(_metadata: LLMCallMetadata) -> None:
        pass

    client = build_role_llm(
        _config(),
        "slave",
        scenario_id="scenario-17",
        sim_time_s=240,
        before_request=before,
        after_response=after,
    )
    try:
        assert client._scenario_id == "scenario-17"
        assert client._sim_time_s == 240
        assert client._before_request is before
        assert client._after_response is after
    finally:
        client.close()


def test_legacy_flat_config_is_rejected_without_fallback() -> None:
    with pytest.raises(ValueError, match=r"explicit config\.roles"):
        build_role_llm(LLMConfig(), "master")


def test_unknown_role_is_rejected_before_client_creation() -> None:
    with pytest.raises(ValueError, match="unknown LLM role"):
        build_role_llm(_config(), "observer")
