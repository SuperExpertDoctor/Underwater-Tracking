from underwater_tracking.config.models import TrackingConfig


def test_tracking_config_loads_grid_spec_defaults() -> None:
    config = TrackingConfig()
    assert config.grid.target_grid_cells == 64
    assert config.grid.lateral_half_width_cells == 2


def test_tracking_config_round_trips_explicit_grid_spec() -> None:
    config = TrackingConfig.model_validate(
        {
            "grid": {
                "target_grid_cells": 16,
                "min_cell_size_m": 100.0,
                "max_cell_size_m": 400.0,
                "cell_size_rounding_m": 50.0,
            }
        }
    )
    assert config.grid.target_grid_cells == 16
    assert not hasattr(config.grid, "relay_overlap_policy")
