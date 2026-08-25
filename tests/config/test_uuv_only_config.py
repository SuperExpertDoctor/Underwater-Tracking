from __future__ import annotations

from math import pi
from pathlib import Path

import pytest
from pydantic import ValidationError

from underwater_tracking.config.models import ScenarioConfig
from underwater_tracking.config.platform_core import EnvironmentConfig
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.models import IntelligenceSource


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
    assert len(config.environment.carriers) == 3
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


def test_uuv_only_roster_is_explicit_and_owned() -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    environment = config.environment
    assert environment is not None
    carriers = (environment.carrier, *environment.carriers)

    assert [(item.platform_id, item.role) for item in carriers] == [
        ("carrier_01", "carrier"),
        ("carrier_02", "mother_ship"),
        ("carrier_03", "mother_ship"),
        ("carrier_04", "mother_ship"),
    ]
    assert environment.usvs == ()
    assert {uuv.deployment_state for uuv in environment.uuvs} == {"onboard"}
    assert [uuv.home_carrier_id for uuv in environment.uuvs[:4]] == ["carrier_02"] * 4
    assert [uuv.home_carrier_id for uuv in environment.uuvs[4:8]] == ["carrier_03"] * 4
    assert [uuv.home_carrier_id for uuv in environment.uuvs[8:]] == ["carrier_04"] * 4
    assert len({carrier.formation_slot_offset_xy for carrier in carriers}) == 4
    assert environment.rendezvous_tolerance_m == 250.0
    assert environment.submarines[0].detection_range_m == 5000.0


def test_carrier_fleet_starts_westbound_and_derives_later_heading_from_motion() -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    environment = config.environment
    assert environment is not None
    carriers = (environment.carrier, *environment.carriers)

    assert all(carrier.heading_rad == -pi for carrier in carriers)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("home_carrier_id", None, "home_carrier_id"),
        ("home_carrier_id", "carrier_01", "carrier cannot own UUVs"),
        ("home_carrier_id", "carrier_missing", "unknown UUV home carrier"),
    ],
)
def test_uuv_only_roster_rejects_invalid_ownership(
    field: str, value: object, error: str
) -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    environment = config.environment
    assert environment is not None
    data = environment.model_dump()
    data["uuvs"][0][field] = value

    with pytest.raises(ValidationError, match=error):
        type(environment).model_validate(data)


def test_uuv_only_roster_rejects_carrier_inventory_imbalance() -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    environment = config.environment
    assert environment is not None
    data = environment.model_dump()
    data["uuvs"][-1]["home_carrier_id"] = "carrier_02"

    with pytest.raises(ValidationError, match="exactly four UUVs"):
        type(environment).model_validate(data)


def test_default_target_prior_is_public_and_not_truth_equal() -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    prior = config.scenario.target_search_priors[0]
    assert prior.prior_id == "intel-target-00-initial"
    assert prior.target_id == "target_00"
    assert prior.source is IntelligenceSource.TECHNICAL_RECONNAISSANCE
    assert config.environment is not None
    assert prior.center_xy != config.environment.submarines[0].position_xy
    assert prior.valid_until_s > prior.issued_at_s


def test_uuv_only_environment_requires_explicit_carrier_roster() -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    environment = config.environment
    assert environment is not None
    data = environment.model_dump()
    data["carriers"] = []

    with pytest.raises(ValidationError, match="three mother ships"):
        type(environment).model_validate(data)
