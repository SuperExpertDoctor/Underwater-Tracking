from pathlib import Path

import pytest
from pydantic import ValidationError

from underwater_tracking.config.doctrine import DoctrineConfig
from underwater_tracking.config.loader import load_app_config


SCENARIO = Path("configs/scenario/segmented_single_target.yaml")


def test_shipped_doctrine_is_loaded_and_explicit() -> None:
    config = load_app_config(SCENARIO)

    assert config.doctrine is not None
    assert config.doctrine.passive_continuous is True
    assert config.doctrine.active_only_on_exception is True
    assert config.doctrine.require_connected_emitter_receiver is True
    assert config.doctrine.handoff_lead_time_s == 600


def test_doctrine_rejects_exception_only_without_continuous_passive() -> None:
    with pytest.raises(ValidationError, match="continuous passive"):
        DoctrineConfig(passive_continuous=False, active_only_on_exception=True)


def test_doctrine_rejects_non_increasing_covariance_trigger() -> None:
    with pytest.raises(ValidationError, match="uncertainty trigger"):
        DoctrineConfig(active_quality_floor=0.0, active_covariance_growth_factor=1.0)
