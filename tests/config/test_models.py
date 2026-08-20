from pathlib import Path

import pytest
from pydantic import ValidationError

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import MemoryConfig, TimingConfig


def test_demo_time_scale_defaults_to_sixty_and_rejects_non_positive_values() -> None:
    assert TimingConfig().demo_time_scale == 60.0

    with pytest.raises(ValidationError):
        TimingConfig(demo_time_scale=0.0)


def test_memory_config_validates_limits_and_embedding_settings() -> None:
    config = MemoryConfig(
        embedding_base_url="https://api.longcat.chat/openai/v1",
        embedding_model="embedding-model",
    )
    assert config.retrieval_candidate_limit >= config.retrieval_top_k
    assert config.embedding_api_key_env == "UNDERWATER_TRACKING_API_KEY"

    for change in (
        {"poll_interval_s": 0.0},
        {"max_attempts": -1},
        {"retrieval_top_k": 0},
        {"context_token_budget": 0},
        {"decay_half_life_s": 0.0},
        {"embedding_timeout_s": 0.0},
    ):
        with pytest.raises(ValidationError):
            MemoryConfig(
                embedding_base_url="https://api.longcat.chat/openai/v1",
                embedding_model="embedding-model",
                **change,
            )

    with pytest.raises(ValidationError, match="retrieval_candidate_limit"):
        MemoryConfig(
            embedding_base_url="https://api.longcat.chat/openai/v1",
            embedding_model="embedding-model",
            retrieval_top_k=8,
            retrieval_candidate_limit=7,
        )
    with pytest.raises(ValidationError, match="extra"):
        MemoryConfig(
            embedding_base_url="https://api.longcat.chat/openai/v1",
            embedding_model="embedding-model",
            unexpected_setting=True,
        )


def test_degraded_memory_config_has_no_embedding_fallback() -> None:
    config = MemoryConfig.degraded()

    assert config.enabled is False
    assert config.embedding_base_url is None
    assert config.embedding_model is None


def test_loader_adds_memory_config_from_the_shipped_configuration() -> None:
    config = load_app_config(Path("configs/scenario/default.yaml"))

    assert config.memory is not None
    assert config.memory.embedding_api_key_env == "UNDERWATER_TRACKING_API_KEY"
    assert config.memory.embedding_base_url.startswith("https://")
