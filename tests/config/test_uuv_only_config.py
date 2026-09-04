from __future__ import annotations

from math import pi
from pathlib import Path

import pytest
from pydantic import ValidationError

from underwater_tracking.config.models import AppConfig, ScenarioConfig, TrackingPolicyConfig
from underwater_tracking.config.platform_core import EnvironmentConfig
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.models import IntelligenceSource


def test_scenario_declares_uuv_only_switch() -> None:
    assert ScenarioConfig.model_fields["uuv_only"].default is False


def test_uuv_only_tracking_policy_is_single_validated_contract() -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    policy = config.scenario.tracking_policy
    assert policy.region_count == 4
    assert policy.task_group_size == 3
    assert policy.task_region_side_m == 2_000.0
    assert policy.target_detection_radius_m == 1_000.0
    assert policy.uuv_active_detection_radius_m == 600.0
    assert policy.uuv_passive_detection_radius_m == 600.0
    assert policy.region_entry_probability_threshold == 0.70
    assert policy.region_transition_confirm_cycles == 2
    assert policy.max_uuv_mileage_m == 50_000.0
    assert policy.dedicated_release_remaining_mileage_m == 7_000.0


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"region_count": 3}, "region_count must equal 4"),
        ({"task_group_size": 2}, "task_group_size must equal 3"),
        ({"task_region_side_m": 1_000.0}, "region side must exceed target detection"),
        ({"target_detection_radius_m": 600.0}, "target detection must exceed UUV"),
        ({"dedicated_release_remaining_mileage_m": 50_000.0}, "release threshold"),
    ],
)
def test_tracking_policy_rejects_invalid_invariants(update, message) -> None:
    payload = TrackingPolicyConfig().model_dump()
    payload.update(update)
    with pytest.raises(ValueError, match=message):
        TrackingPolicyConfig.model_validate(payload)


def test_uuv_only_target_detection_range_must_match_tracking_policy() -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    payload = config.model_dump()
    payload["environment"]["submarines"][0]["detection_range_m"] = 999.0

    with pytest.raises(ValueError, match="target detection range"):
        AppConfig.model_validate(payload)


@pytest.mark.parametrize(
    "field", ["active_source_range_m", "active_receive_range_m", "passive_range_m"]
)
def test_uuv_only_sonar_ranges_must_match_tracking_policy(field: str) -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    payload = config.model_dump()
    payload["sensors"]["profiles"]["uuv_dual_sonar"][field] = 599.0

    with pytest.raises(ValueError, match="uuv_dual_sonar"):
        AppConfig.model_validate(payload)


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
    assert environment.submarines[0].detection_range_m == 1000.0


def test_uuv_only_demo_uses_reduced_entity_speeds() -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    environment = config.environment
    assert environment is not None
    carriers = (environment.carrier, *environment.carriers)

    assert [carrier.speed_mps for carrier in carriers] == [2.0, 4.0, 4.0, 4.0]
    assert environment.submarines[0].speed_mps == 3.0
    assert all(
        config.platforms.motion_profiles[uuv.motion_profile].max_speed_mps == 4.0
        for uuv in environment.uuvs
    )
    submarine = environment.submarines[0]
    assert config.platforms.motion_profiles[submarine.motion_profile].max_speed_mps == 7.0


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
    assert prior.center_xy == (-6800.0, -6800.0)
    assert prior.covariance_xy == (
        (360000.0, 0.0),
        (0.0, 360000.0),
    )
    assert prior.confidence == 0.45
    assert prior.issued_at_s == 0
    assert prior.valid_until_s == 1800
    assert config.environment is not None
    assert prior.center_xy != config.environment.submarines[0].position_xy


def test_default_carrier_formation_patrols_outside_the_submarine_task_region() -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    environment = config.environment
    assert environment is not None
    task_region = next(
        region
        for region in environment.task_regions
        if region.region_id == environment.submarines[0].task_region_id
    )
    xs = [point[0] for point in task_region.polygon_xy]
    ys = [point[1] for point in task_region.polygon_xy]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def is_inside(point: tuple[float, float]) -> bool:
        return min_x < point[0] < max_x and min_y < point[1] < max_y

    submarine = environment.submarines[0]
    assert all(is_inside(point) for point in submarine.mission_route_xy)
    for carrier in (environment.carrier, *environment.carriers):
        assert not is_inside(carrier.position_xy)
        assert all(not is_inside(point) for point in carrier.patrol_route_xy)


def test_uuv_only_environment_requires_explicit_carrier_roster() -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    environment = config.environment
    assert environment is not None
    data = environment.model_dump()
    data["carriers"] = []

    with pytest.raises(ValidationError, match="three mother ships"):
        type(environment).model_validate(data)
