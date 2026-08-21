from __future__ import annotations

from pathlib import Path

from underwater_tracking.config.models import ScenarioConfig
from underwater_tracking.config.platform_core import EnvironmentConfig
from underwater_tracking.config.loader import load_app_config


def test_scenario_declares_uuv_only_switch() -> None:
    assert ScenarioConfig.model_fields["uuv_only"].default is False


def test_environment_declares_multiple_carrier_roster() -> None:
    assert "carriers" in EnvironmentConfig.model_fields


def test_uuv_only_scenario_loads_one_carrier_and_three_mother_ships() -> None:
    scenario = Path("configs/scenario/uuv_only_single_target.yaml")
    config = load_app_config(scenario)
    assert config.scenario.uuv_only is True
    assert config.environment is not None
    assert config.environment.uuv_only is True
    assert config.environment.usvs == ()
    assert len(config.environment.carriers) == 4
    roles = {
        carrier.platform_id: carrier.role
        for carrier in (config.environment.carrier, *config.environment.carriers)
    }
    assert roles == {
        "carrier_01": "carrier",
        "carrier_02": "mother_ship",
        "carrier_03": "mother_ship",
        "carrier_04": "mother_ship",
    }
