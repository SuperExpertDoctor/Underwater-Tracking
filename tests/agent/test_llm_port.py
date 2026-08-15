# tests/agent/test_llm_port.py
"""Structured LLM port contract tests (spec 22, 8.3).

Deterministic, network-free coverage of the client's configuration and
error contract: the shipped ``configs/llm.yaml`` points at the LongCat
provider, the constructor lands the config defaults, and a missing API key
raises ``LLMConfigError`` before any network attempt. Transport retry
classification (timeout/connection/429/5xx), exponential backoff,
content-failure classification, metadata hooks, and the call-time bearer
token read were previously covered through injected stub transports; per
the user directive (addendum A) no mock may substitute real LLM
functionality, so those paths are now exercised only against the real
provider in ``tests/integration/test_llm_real_api.py`` (live) — the
deterministic failure-injection tests were deleted as an accepted
consequence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from underwater_tracking.agent.llm import HTTPStructuredLLM, LLMConfigError
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import IntentHypothesis
from tests.conftest import make_live_llm

# An environment variable that is never set, so the client must fail at the
# call-time key check; the base URL points at an unroutable local port, so a
# mis-ordered implementation (network first) would surface as a connection
# error instead of the config error.
MISSING_KEY_ENV = "UNDERWATER_TRACKING_API_KEY_MISSING_TEST"


def test_llm_config_points_at_longcat_provider():
    """The shipped llm.yaml wires the OpenAI-compatible LongCat provider.

    Pure config check, no network: the key is referenced by environment
    variable name only and never appears in the config file.
    """
    config_path = (
        Path(__file__).resolve().parents[2] / "configs/scenario/default.yaml"
    )
    config = load_app_config(config_path)
    assert config.llm is not None
    assert config.llm.base_url == "https://api.longcat.chat/openai/v1"
    assert config.llm.model == "LongCat-2.0"
    assert config.llm.api_key_env == "UNDERWATER_TRACKING_API_KEY"


def test_constructor_lands_config_defaults():
    """The shipped llm.yaml defaults land on the client (no network).

    These are constructor-time mirrors of ``configs/llm.yaml`` (the same
    values ``agent-run`` passes); the client is never invoked, so neither
    the API key nor the network is involved.
    """
    client = make_live_llm()
    try:
        assert client._base_url == "https://api.longcat.chat/openai/v1"
        assert client._model == "LongCat-2.0"
        assert client._api_key_env == "UNDERWATER_TRACKING_API_KEY"
        assert client._temperature == 0.2
        assert client._max_tokens == 4096
        assert client._max_attempts == 3
    finally:
        client.close()


def test_missing_api_key_raises_config_error_before_any_network():
    """A missing key fails the call-time check before any network attempt.

    ``LLMConfigError`` names the missing environment variable; a network
    attempt would instead have surfaced as a ``TransientLLMError`` against
    the unroutable base URL, so the error type proves the ordering.
    """
    client = HTTPStructuredLLM(
        base_url="http://127.0.0.1:1/v1/chat/completions",
        model="LongCat-2.0",
        api_key_env=MISSING_KEY_ENV,
        connect_timeout_s=0.2,
        request_timeout_s=0.5,
        max_retries=1,
    )
    try:
        with pytest.raises(LLMConfigError, match=MISSING_KEY_ENV):
            client.invoke_structured("intent", {}, IntentHypothesis)
    finally:
        client.close()
