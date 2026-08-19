from underwater_tracking.config.loader import load_app_config


def test_default_config_has_confirmed_multirate_defaults():
    config = load_app_config("configs/scenario/default.yaml")
    assert config.scenario.uuv_count == 12
    assert config.scenario.initial_target_count == 2
    assert config.timing.physics_step_s == 5
    assert config.timing.observation_step_s == 30
    assert config.timing.group_report_s == 300
    assert config.timing.strategic_review_s == 900
    assert config.tracking.group_min_size == 2
    assert config.tracking.group_max_size == 4
