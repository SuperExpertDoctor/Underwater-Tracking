from __future__ import annotations

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.models import DeploymentState
from underwater_tracking.simulation import adversary_sensing
from underwater_tracking.simulation.engine import SimulationEngine


def test_extracts_approach_and_disengagement_from_observed_uuv_tracks() -> None:
    extractor = getattr(adversary_sensing, "extract_uuv_tracking_patterns", None)
    assert extractor is not None
    cache = {
        "UUV-approach": (
            (0, (4_000.0, 0.0)),
            (30, (3_000.0, 0.0)),
            (60, (2_000.0, 0.0)),
            (90, (1_000.0, 0.0)),
        ),
        "UUV-leave": (
            (0, (500.0, 500.0)),
            (30, (1_200.0, 700.0)),
            (60, (2_100.0, 1_100.0)),
            (90, (3_200.0, 1_800.0)),
        ),
    }

    patterns = extractor(
        cache,
        target_position_xy=(0.0, 0.0),
        target_velocity_xy=(4.0, 0.0),
    )

    by_type = {pattern.pattern_type: pattern for pattern in patterns}
    assert by_type["tracking_approach"].uuv_ids == ("UUV-approach",)
    assert by_type["tracking_disengagement"].uuv_ids == ("UUV-leave",)


def test_extracts_reacquisition_relay_and_flank_envelope() -> None:
    extractor = getattr(adversary_sensing, "extract_uuv_tracking_patterns", None)
    assert extractor is not None
    cache = {
        "UUV-left": (
            (0, (-1_000.0, 1_000.0)),
            (30, (-850.0, 950.0)),
            (180, (-700.0, 800.0)),
            (210, (-450.0, 650.0)),
        ),
        "UUV-right": (
            (0, (600.0, -1_000.0)),
            (30, (900.0, -900.0)),
            (60, (1_300.0, -800.0)),
            (90, (1_800.0, -700.0)),
        ),
    }

    patterns = extractor(
        cache,
        target_position_xy=(0.0, 0.0),
        target_velocity_xy=(4.0, 0.0),
    )

    pattern_types = {pattern.pattern_type for pattern in patterns}
    assert "tracking_reacquisition" in pattern_types
    assert "relay_tracking" in pattern_types
    assert "flank_envelope_tracking" in pattern_types


def test_engine_feeds_target_observed_uuv_history_and_patterns_to_decision_input() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)
    target = engine._targets["target_00"]
    uuv_id = "uuv_00"
    engine._deployment_states[uuv_id] = DeploymentState.DEPLOYED
    engine._waterborne_uuv_ids.add(uuv_id)
    for carrier in engine._carrier_entities.values():
        carrier.position_xy = (
            target.position_xy[0] + target.detection_range_m * 2.0,
            target.position_xy[1],
        )

    engine._uuvs[uuv_id].position_xy = (
        target.position_xy[0] + target.detection_range_m * 0.8,
        target.position_xy[1],
    )
    engine._update_target_detection_events(30)
    engine._uuvs[uuv_id].position_xy = (
        target.position_xy[0] + target.detection_range_m * 0.2,
        target.position_xy[1],
    )
    engine._update_target_detection_events(60)

    contexts = engine.build_adversary_inputs(engine._build_situation(60))

    assert contexts
    points = contexts[0].uuv_trajectory_cache[uuv_id]
    assert [(point.event, point.uuv_status) for point in points] == [
        ("acquired", "scan"),
        ("observed", "scan"),
    ]
    assert "tracking_approach" in {
        pattern.pattern_type for pattern in contexts[0].uuv_tracking_patterns
    }
