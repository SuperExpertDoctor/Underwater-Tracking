# tests/conftest.py
"""Shared fixtures for the live LongCat API tests (addendum A).

The client is built from the shipped ``configs/llm.yaml`` via
``load_app_config`` (the same path ``agent-run`` uses). The API key is
referenced by environment-variable name only and read at call time; it is
never printed, logged, or persisted anywhere. Every live test module guards
itself with a module-level ``skipif`` on the key being present.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from underwater_tracking.agent.llm import HTTPStructuredLLM
from underwater_tracking.config.loader import load_app_config

# The reason string every live module reports when no key is available: it
# names the environment variable and the config key, never a value.
REAL_LLM_SKIP_REASON = (
    "neither the UNDERWATER_TRACKING_API_KEY environment variable nor a "
    "configured api_key is set; the live LongCat API tests are skipped"
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/scenario/default.yaml"


def make_live_llm(**kwargs: object) -> HTTPStructuredLLM:
    """A real HTTP client over the shipped LongCat config (key from config or env)."""
    config = load_app_config(CONFIG_PATH)
    assert config.llm is not None, "configs/scenario/default.yaml must load llm.yaml"
    llm_config = config.llm
    return HTTPStructuredLLM(
        base_url=llm_config.base_url,
        model=llm_config.model,
        api_key_env=llm_config.api_key_env,
        api_key=llm_config.api_key,
        request_timeout_s=llm_config.request_timeout_s,
        connect_timeout_s=llm_config.connect_timeout_s,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        max_retries=llm_config.max_retries,
        backoff_base_s=llm_config.backoff_base_s,
        backoff_max_s=llm_config.backoff_max_s,
        **kwargs,
    )


def has_live_api_key() -> bool:
    """True when a key is available: the env var, or the shipped config's key."""
    if os.environ.get("UNDERWATER_TRACKING_API_KEY"):
        return True
    config = load_app_config(CONFIG_PATH)
    return bool(config.llm is not None and config.llm.api_key)


@pytest.fixture
def live_llm() -> Iterator[HTTPStructuredLLM]:
    """One live client with a closed lifecycle, plus the fixtures' extras.

    Fixtures should not use this for clients that must carry a ledger or a
    recording hook; build those with ``make_live_llm`` and close them in a
    ``finally`` block instead.
    """
    client = make_live_llm()
    yield client
    client.close()
