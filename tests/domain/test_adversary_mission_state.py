from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.adversary_models import AdversaryMissionState


def test_default_runtime_carries_the_configured_target_mission() -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    assert config.environment is not None
    submarine = config.environment.submarines[0]

    assert submarine.mission_route_xy[-1] == (8500.0, 0.0)
    assert config.environment.navigation_exclusion_regions == ()


def test_mission_state_requires_a_valid_route_index_and_escape_region() -> None:
    with pytest.raises(ValidationError, match="current_route_index"):
        AdversaryMissionState(
            target_id="target_00",
            task_region_id="task",
            task_region_polygon_xy=((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)),
            mission_route_xy=((0.0, 0.0), (1.0, 1.0)),
            escape_regions={"escape": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))},
            current_route_index=2,
        )

    with pytest.raises(ValidationError, match="escape region"):
        AdversaryMissionState(
            target_id="target_00",
            task_region_id="task",
            task_region_polygon_xy=((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)),
            mission_route_xy=((0.0, 0.0), (1.0, 1.0)),
            escape_regions={},
            current_route_index=0,
        )
