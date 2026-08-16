from pathlib import Path

import pytest
from pydantic import ValidationError

from underwater_tracking.config.loader import load_app_config


SCENARIO = Path("configs/scenario/segmented_single_target.yaml")


def test_explicit_platform_core_roster_loads() -> None:
    config = load_app_config(SCENARIO)

    assert config.scenario.scenario_id == "segmented-single-target"
    assert config.environment is not None
    assert config.platforms is not None
    assert config.sensors is not None
    assert config.communications is not None
    assert config.environment.carrier.platform_id == "carrier_01"
    assert len(config.environment.usvs) == 4
    assert len(config.environment.uuvs) == 12
    assert len(config.environment.submarines) == 1
    assert config.environment.decoys == ()


def test_every_roster_entry_resolves_capability_profiles() -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    assert config.platforms is not None
    assert config.sensors is not None
    assert config.communications is not None

    for platform in (*config.environment.usvs, *config.environment.uuvs):
        assert platform.motion_profile in config.platforms.motion_profiles
        assert platform.sensor_profile in config.sensors.profiles
        assert platform.communication_profile in config.communications.profiles
    for submarine in config.environment.submarines:
        assert submarine.motion_profile in config.platforms.motion_profiles


def test_referenced_config_path_cannot_escape_configs(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    scenario_dir = config_root / "scenario"
    scenario_dir.mkdir(parents=True)
    (config_root / "tracking.yaml").write_text(
        "group_min_size: 2\ngroup_max_size: 4\n", encoding="utf-8"
    )
    scenario = scenario_dir / "bad.yaml"
    scenario.write_text(
        "scenario:\n"
        "  scenario_id: bad\n"
        "  duration_s: 60\n"
        "  seed: 1\n"
        "  platform_core:\n"
        "    environment: ../../outside.yaml\n"
        "    platforms: platforms.yaml\n"
        "    sensors: sensors.yaml\n"
        "    communications: communications.yaml\n"
        "timing:\n"
        "  physics_step_s: 10\n"
        "  observation_step_s: 30\n"
        "  group_report_s: 300\n"
        "  progress_report_s: 600\n"
        "  strategic_review_s: 900\n"
        "  prediction_horizon_s: 1800\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must stay below config root"):
        load_app_config(scenario)


def test_explicit_environment_rejects_duplicate_platform_ids() -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    duplicate = config.environment.model_dump()
    duplicate["uuvs"][0]["platform_id"] = duplicate["usvs"][0]["platform_id"]

    with pytest.raises(ValidationError, match="platform IDs must be unique"):
        type(config.environment).model_validate(duplicate)
