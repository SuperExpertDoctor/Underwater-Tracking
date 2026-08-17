import pytest

from underwater_tracking.agent.nodes.optimize import PlanningConfig


@pytest.mark.parametrize(
    "field",
    [
        "max_range_m",
        "min_range_m",
        "min_separation_m",
        "bearing_variance",
        "replan_period_s",
        "return_reserve",
        "quality_warning",
        "quality_release",
        "release_hold_s",
        "reassignment_penalty",
        "rotation_threshold",
        "improvement_margin",
        "plan_horizon_s",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_planning_config_rejects_non_finite_float_values(field, value) -> None:
    with pytest.raises(ValueError):
        PlanningConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bounds", (0.0, 0.0, -1.0, 1.0)),
        ("max_range_m", 0.0),
        ("min_range_m", -1.0),
        ("min_separation_m", -1.0),
        ("bearing_variance", 0.0),
        ("replan_period_s", 0.0),
        ("return_reserve", 1.1),
        ("quality_warning", -0.1),
        ("quality_release", 1.1),
        ("release_hold_s", -1.0),
        ("reassignment_penalty", -1.0),
        ("rotation_threshold", 1.1),
        ("plan_horizon_s", 0),
        ("improvement_margin", -0.1),
    ],
)
def test_planning_config_rejects_out_of_range_values(field, value) -> None:
    with pytest.raises(ValueError):
        PlanningConfig(**{field: value})


def test_planning_config_requires_warning_below_release() -> None:
    with pytest.raises(ValueError):
        PlanningConfig(quality_warning=0.8, quality_release=0.8)
