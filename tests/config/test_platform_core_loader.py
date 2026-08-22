from pathlib import Path
from shutil import copyfile

import pytest
import yaml
from pydantic import ValidationError

from underwater_tracking.config.loader import load_app_config


SCENARIO = Path("configs/scenario/segmented_single_target.yaml")


def _copy_platform_core_config_tree(tmp_path: Path) -> Path:
    config_root = tmp_path / "configs"
    scenario_dir = config_root / "scenario"
    scenario_dir.mkdir(parents=True)
    for relative_path in (
        Path("scenario/segmented_single_target.yaml"),
        Path("tracking.yaml"),
        Path("environment.yaml"),
        Path("platforms.yaml"),
        Path("sensors.yaml"),
        Path("communications.yaml"),
    ):
        copyfile(Path("configs") / relative_path, config_root / relative_path)
    return scenario_dir / "segmented_single_target.yaml"


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


@pytest.mark.parametrize(
    ("platform_kind", "error"),
    [
        ("usvs", "USV platform indices must be unique"),
        ("uuvs", "UUV platform indices must be unique"),
    ],
)
def test_explicit_environment_rejects_duplicate_indices_per_platform_kind(
    platform_kind: str, error: str
) -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    environment = config.environment.model_dump()
    environment[platform_kind][1]["platform_index"] = environment[platform_kind][0][
        "platform_index"
    ]

    with pytest.raises(ValidationError, match=error):
        type(config.environment).model_validate(environment)


def test_explicit_environment_rejects_duplicate_region_ids() -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    environment = config.environment.model_dump()
    environment["escape_regions"][0]["region_id"] = environment["task_regions"][0]["region_id"]

    with pytest.raises(ValidationError, match="region IDs must be unique"):
        type(config.environment).model_validate(environment)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("task_region_id", "missing_task", "unknown task region"),
        ("escape_region_ids", ["escape_north", "missing_escape"], "unknown escape region"),
    ],
)
def test_explicit_environment_requires_declared_submarine_regions(
    field: str, value: object, error: str
) -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    environment = config.environment.model_dump()
    environment["submarines"][0][field] = value

    with pytest.raises(ValidationError, match=error):
        type(config.environment).model_validate(environment)


def test_explicit_environment_rejects_duplicate_escape_ids_and_route_outside_map() -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None

    duplicate = config.environment.model_dump()
    duplicate["submarines"][0]["escape_region_ids"] = ["escape_north", "escape_north"]
    with pytest.raises(ValidationError, match="duplicate escape region IDs"):
        type(config.environment).model_validate(duplicate)

    outside = config.environment.model_dump()
    outside["submarines"][0]["mission_route_xy"] = [
        outside["submarines"][0]["mission_route_xy"][0],
        [50000.0, 0.0],
        *outside["submarines"][0]["mission_route_xy"][2:],
    ]
    with pytest.raises(ValidationError, match="mission route.*outside map bounds"):
        type(config.environment).model_validate(outside)


def test_explicit_environment_rejects_route_through_navigation_exclusion() -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    environment = config.environment.model_dump()
    environment["navigation_exclusion_regions"] = [
        {
            "region_id": "navigation_block",
            "polygon_xy": [[-3500.0, -4300.0], [-2500.0, -4300.0], [-2500.0, -4100.0], [-3500.0, -4100.0]],
        }
    ]
    with pytest.raises(ValidationError, match="intersects navigation exclusion"):
        type(config.environment).model_validate(environment)


def test_loader_rejects_unknown_submarine_region_reference(tmp_path: Path) -> None:
    scenario = _copy_platform_core_config_tree(tmp_path)
    environment_path = scenario.parents[1] / "environment.yaml"
    environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    environment["submarines"][0]["task_region_id"] = "missing_task"
    environment_path.write_text(yaml.safe_dump(environment), encoding="utf-8")

    with pytest.raises(ValidationError, match="unknown task region"):
        load_app_config(scenario)


def test_explicit_environment_rejects_usv_outside_carrier_support_radius() -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    environment = config.environment.model_dump()
    environment["carrier"]["support_radius_m"] = 100.0

    with pytest.raises(ValidationError, match="outside carrier support radius"):
        type(config.environment).model_validate(environment)


def test_explicit_environment_rejects_patrol_route_with_zero_length_segment() -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    environment = config.environment.model_dump()
    route = environment["carrier"]["patrol_route_xy"]
    environment["carrier"]["patrol_route_xy"] = [route[0], route[0], route[1]]

    with pytest.raises(ValidationError, match="zero-length consecutive"):
        type(config.environment).model_validate(environment)


def test_explicit_environment_rejects_patrol_route_without_valid_segment() -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    environment = config.environment.model_dump()
    point = environment["carrier"]["patrol_route_xy"][0]
    environment["carrier"]["patrol_route_xy"] = [point, point]

    with pytest.raises(ValidationError, match="at least one valid"):
        type(config.environment).model_validate(environment)


def test_app_config_rejects_submarine_initial_speed_above_motion_profile() -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    assert config.platforms is not None
    data = config.model_dump()
    submarine = data["environment"]["submarines"][0]
    profile = config.platforms.motion_profiles[submarine["motion_profile"]]
    submarine["speed_mps"] = profile.max_speed_mps + 0.1

    with pytest.raises(ValidationError, match="initial speed_mps.*exceeds.*max_speed_mps"):
        type(config).model_validate(data)


def test_app_config_rejects_initial_submarine_that_cannot_join_route() -> None:
    config = load_app_config(SCENARIO)
    data = config.model_dump()
    data["environment"]["submarines"][0]["position_xy"] = [0.0, 11000.0]
    with pytest.raises(ValidationError, match="cannot join the first mission route"):
        type(config).model_validate(data)


def test_loader_rejects_nonfinite_map_bound(tmp_path: Path) -> None:
    scenario = _copy_platform_core_config_tree(tmp_path)
    environment_path = scenario.parents[1] / "environment.yaml"
    environment_data = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    environment_data["map_bounds_xy"][0] = float("nan")
    environment_path.write_text(yaml.safe_dump(environment_data), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_app_config(scenario)


@pytest.mark.parametrize(
    ("section", "value"),
    [
        ("usv_position", float("inf")),
        ("carrier_route", float("nan")),
    ],
)
def test_environment_rejects_nonfinite_coordinates(section: str, value: float) -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    environment = config.environment.model_dump()
    if section == "usv_position":
        environment["usvs"][0]["position_xy"] = [value, 0.0]
    else:
        environment["carrier"]["patrol_route_xy"] = [[value, 0.0], [0.0, 0.0]]

    with pytest.raises(ValidationError):
        type(config.environment).model_validate(environment)


def test_motion_profile_rejects_nonfinite_value() -> None:
    config = load_app_config(SCENARIO)
    assert config.platforms is not None
    platforms = config.platforms.model_dump()
    platforms["motion_profiles"]["usv_standard"]["max_speed_mps"] = float("inf")

    with pytest.raises(ValidationError):
        type(config.platforms).model_validate(platforms)
