from underwater_tracking.api.frame_builder import _build_world_model_forecast
from underwater_tracking.domain.ui_models import MapBounds, OperationalFrame
from underwater_tracking.world_model.demo import build_demo_input
from underwater_tracking.world_model.rules import predict_future_events


def test_world_model_forecast_maps_to_truth_safe_ui_contract() -> None:
    forecast = predict_future_events(build_demo_input("left_turn"))

    view = _build_world_model_forecast(
        forecast,
        MapBounds(min_x=-12_000, min_y=-12_000, max_x=12_000, max_y=12_000),
    )

    assert view is not None
    assert view.control_authority is False
    assert view.events
    assert view.events[0].horizon in {"H1", "H2", "H3", "H4"}
    assert view.events[0].evidence
    assert "ground_truth" not in view.model_dump_json()


def test_legacy_operational_frames_do_not_require_world_model_data() -> None:
    frame = OperationalFrame.model_validate(
        {
            "frame_id": 0,
            "sim_time_s": 0,
            "plan_version": 0,
            "map_bounds": {
                "min_x": -1,
                "min_y": -1,
                "max_x": 1,
                "max_y": 1,
            },
        }
    )

    assert frame.target_estimates == ()
