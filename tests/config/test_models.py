from pathlib import Path

import pytest
from pydantic import ValidationError

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import (
    MemoryConfig,
    PredictionHealthConfig,
    TimingConfig,
)


CONFIG_PATH = Path("configs/scenario/default.yaml")


def test_demo_time_scale_defaults_to_sixty_and_rejects_non_positive_values() -> None:
    assert TimingConfig().demo_time_scale == 60.0

    with pytest.raises(ValidationError):
        TimingConfig(demo_time_scale=0.0)


def test_memory_config_validates_limits_and_embedding_settings() -> None:
    config = MemoryConfig(
        embedding_provider="http",
        embedding_base_url="https://api.longcat.chat/openai/v1",
        embedding_model="embedding-model",
    )
    assert config.retrieval_candidate_limit >= config.retrieval_top_k
    assert config.embedding_api_key_env == "UNDERWATER_TRACKING_API_KEY"
    assert config.source_poll_interval_s == 2.0

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
                embedding_provider="http",
                embedding_base_url="https://api.longcat.chat/openai/v1",
                embedding_model="embedding-model",
                **change,
            )

    with pytest.raises(ValidationError, match="retrieval_candidate_limit"):
        MemoryConfig(
            embedding_provider="http",
            embedding_base_url="https://api.longcat.chat/openai/v1",
            embedding_model="embedding-model",
            retrieval_top_k=8,
            retrieval_candidate_limit=7,
        )
    with pytest.raises(ValidationError, match="extra"):
        MemoryConfig(
            embedding_provider="http",
            embedding_base_url="https://api.longcat.chat/openai/v1",
            embedding_model="embedding-model",
            unexpected_setting=True,
        )


def test_local_sentence_transformer_config_requires_local_files_only() -> None:
    config = MemoryConfig(
        embedding_provider="sentence_transformers",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    assert config.embedding_base_url is None
    assert config.embedding_local_files_only is True
    assert config.embedding_cache_dir == ".cache/sentence-transformers"
    assert config.embedding_download_on_missing is True
    assert config.embedding_device == "cpu"
    assert config.embedding_normalize is True

    with pytest.raises(ValidationError, match="local_files_only"):
        MemoryConfig(
            embedding_provider="sentence_transformers",
            embedding_model="local-model",
            embedding_local_files_only=False,
        )


def test_degraded_memory_config_has_no_embedding_fallback() -> None:
    config = MemoryConfig.degraded()

    assert config.enabled is False
    assert config.embedding_base_url is None
    assert config.embedding_model is None


def test_loader_constructs_degraded_memory_config_when_memory_yaml_is_absent(
    tmp_path: Path,
) -> None:
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    (tmp_path / "tracking.yaml").write_text(
        "group_min_size: 2\ngroup_max_size: 4\n",
        encoding="utf-8",
    )
    scenario_path = scenario_dir / "default.yaml"
    scenario_path.write_text(
        """scenario:
  uuv_count: 12
  initial_target_count: 1
  max_target_count: 4
  duration_s: 28800
timing:
  physics_step_s: 5
  observation_step_s: 30
  group_report_s: 300
  progress_report_s: 600
  strategic_review_s: 900
  prediction_horizon_s: 1800
""",
        encoding="utf-8",
    )

    config = load_app_config(scenario_path)

    assert config.memory == MemoryConfig.degraded()


def test_loader_adds_memory_config_from_the_shipped_configuration() -> None:
    config = load_app_config(Path("configs/scenario/default.yaml"))

    assert config.memory is not None
    assert config.memory.embedding_api_key_env == "UNDERWATER_TRACKING_API_KEY"
    assert config.memory.embedding_provider == "sentence_transformers"
    assert config.memory.embedding_base_url is None
    assert "paraphrase-multilingual" in str(config.memory.embedding_model)
    assert config.memory.embedding_local_files_only is True


def test_tracking_config_loads_prediction_health_thresholds() -> None:
    config = load_app_config(CONFIG_PATH)

    health = config.tracking.prediction_health
    assert health.refresh_interval_s == 450
    assert health.hard_stale_s == 900
    assert health.max_clipped_point_fraction == 0.20
    assert health.max_corridor_radius_m == 6_000
    assert health.max_corridor_map_fraction == 0.25
    assert health.minimum_point_confidence == 0.02
    assert health.coordinate_tolerance_m == 0.000001
    assert health.boundary_recovery_timeout_s == 300


def test_prediction_health_rejects_stale_window_before_refresh_window() -> None:
    with pytest.raises(ValueError, match="hard_stale_s"):
        PredictionHealthConfig(refresh_interval_s=450, hard_stale_s=449)
