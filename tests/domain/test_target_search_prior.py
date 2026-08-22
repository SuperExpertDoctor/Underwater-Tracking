from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from underwater_tracking.config.loader import load_app_config


CONFIG_PATH = Path("configs/scenario/uuv_only_single_target.yaml")


def _config_data() -> dict[str, object]:
    return load_app_config(CONFIG_PATH).model_dump(mode="python")


def _validate_with_priors(*priors: dict[str, object]) -> None:
    data = _config_data()
    scenario = data["scenario"]
    assert isinstance(scenario, dict)
    scenario["target_search_priors"] = list(priors)
    from underwater_tracking.config.models import AppConfig

    AppConfig.model_validate(data)


def _valid_prior(**updates: object) -> dict[str, object]:
    prior: dict[str, object] = {
        "prior_id": "intel-target-00-initial",
        "target_id": "target_00",
        "source": "technical_reconnaissance",
        "issued_at_s": 0,
        "valid_until_s": 1800,
        "center_xy": (-4200.0, -6200.0),
        "covariance_xy": ((360000.0, 0.0), (0.0, 360000.0)),
        "confidence": 0.45,
    }
    prior.update(updates)
    return prior


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"valid_until_s": 0}, "valid_until_s"),
        ({"issued_at_s": 1800}, "valid_until_s must be after issued_at_s"),
        ({"covariance_xy": ((1.0, 2.0), (0.0, 1.0))}, "symmetric"),
        ({"covariance_xy": ((1.0, 0.0), (0.0, 0.0))}, "positive definite"),
        ({"center_xy": (float("nan"), -6200.0)}, "finite"),
        ({"center_xy": (-13000.0, -6200.0)}, "map bounds"),
        ({"target_id": "missing-target"}, "unknown target"),
    ],
)
def test_target_search_prior_rejects_invalid_configuration(
    updates: dict[str, object], error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        _validate_with_priors(_valid_prior(**updates))


def test_target_search_prior_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="duplicate prior_id"):
        _validate_with_priors(
            _valid_prior(),
            _valid_prior(target_id="target_00"),
        )
