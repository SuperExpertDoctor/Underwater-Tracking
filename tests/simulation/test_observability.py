from __future__ import annotations

from pathlib import Path

import pytest

from underwater_tracking.simulation.observability import (
    EVENT_HYPOTHESES,
    InputFrame,
    ObservabilitySupervisor,
    TruthSafetyError,
    load_observability_config,
)


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "observability_feedback.yaml"


def _frame(
    timestamp_sec: float,
    *,
    sequence_id: int,
    active_uuv_count: int = 2,
    speed_mps: float = 4.0,
    velocity_xy_mps: tuple[float, float] = (4.0, 0.0),
    innovation_rad: float = 0.04,
    association_confidence: float = 0.9,
    association_entropy: float = 0.1,
) -> InputFrame:
    uuvs = tuple(
        {
            "uuv_id": f"uuv-{index}",
            "position_xy_m": ((10.0, 0.0) if index == 0 else (0.0, 10.0)),
            "heading_rad": 0.0,
            "state_age_sec": 0.0,
            "communication_age_sec": 0.0,
            "valid": index < active_uuv_count,
        }
        for index in range(2)
    )
    observations = tuple(
        {
            "uuv_id": f"uuv-{index}",
            "candidate_track_id": "target-1",
            "sequence_id": sequence_id,
            "bearing_rad": 0.5 + index,
            "bearing_variance_rad2": 0.02,
            "measurement_age_sec": 0.0,
            "valid": index < active_uuv_count,
        }
        for index in range(2)
    )
    innovations = tuple(
        {
            "track_id": "target-1",
            "uuv_id": f"uuv-{index}",
            "innovation_rad": innovation_rad,
            "innovation_variance_rad2": 0.02,
        }
        for index in range(active_uuv_count)
    )
    return InputFrame.from_mapping(
        {
            "timestamp_sec": timestamp_sec,
            "frame_sequence_id": sequence_id,
            "frame_id": "map",
            "tracks": [
                {
                    "track_id": "target-1",
                    "estimated_position_xy_m": (0.0, 0.0),
                    "estimated_velocity_xy_mps": velocity_xy_mps,
                    "position_covariance_2x2": (4.0, 0.0, 0.0, 9.0),
                    "association_confidence": association_confidence,
                    "association_entropy": association_entropy,
                    "lifecycle_state": "confirmed",
                }
            ],
            "uuvs": list(uuvs),
            "bearing_observations": list(observations),
            "innovations": list(innovations),
        }
    )


def _test_config(**timing: object) -> object:
    config = load_observability_config(CONFIG_PATH)
    return config.with_timing(**timing)


def test_config_declares_external_mvp_metrics_and_event_contract() -> None:
    config = load_observability_config(CONFIG_PATH)

    assert config.metric_names == (
        "geometry_od",
        "fim_min_eigenvalue",
        "fim_logdet",
        "crlb_position_rmse_m",
        "posterior_rmse_m",
        "covariance_area_95_m2",
        "innovation_rms_rad",
        "active_sensor_count",
    )
    assert config.timing.window_duration_sec > 0.0
    assert config.timing.periodic_feedback_sec > 0.0
    assert config.timing.event_cooldown_sec >= 0.0
    assert config.metric_thresholds["geometry_od"].direction.value == "HIGHER_IS_BETTER"
    assert EVENT_HYPOTHESES == (
        "CLOCK_RESET",
        "TARGET_STOPPED",
        "TARGET_SIGNAL_LOST_AFTER_STOP",
        "UUV_SENSOR_OR_COMM_FAILURE",
        "DECOY_OR_NEW_TARGET",
        "TARGET_MANEUVER",
        "GEOMETRY_DEGRADED",
        "ISOLATED_BAD_MEASUREMENT",
        "OBSERVABILITY_CHANGE_UNCLASSIFIED",
    )


def test_input_rejects_truth_and_low_level_control_fields() -> None:
    payload = {
        "timestamp_sec": 0.0,
        "frame_sequence_id": 0,
        "frame_id": "map",
        "tracks": [],
        "uuvs": [],
        "bearing_observations": [],
        "innovations": [],
        "target_truth": {"position_xy_m": (1.0, 2.0)},
    }

    with pytest.raises(TruthSafetyError):
        InputFrame.from_mapping(payload)

    payload.pop("target_truth")
    payload["control_command"] = {"heading_rad": 0.5}
    with pytest.raises(TruthSafetyError):
        InputFrame.from_mapping(payload)


def test_periodic_feedback_contains_all_eight_metrics_and_no_commands() -> None:
    supervisor = ObservabilitySupervisor(
        _test_config(warmup_sec=0.0, periodic_feedback_sec=10.0)
    )

    assert supervisor.process_frame(_frame(0.0, sequence_id=0)) == ()
    reports = supervisor.process_frame(_frame(10.0, sequence_id=1))

    assert len(reports) == 1
    report = reports[0]
    assert report.report_type == "PERIODIC"
    assert report.events == ()
    track = report.tracks[0]
    assert tuple(track.metrics) == load_observability_config(CONFIG_PATH).metric_names
    assert track.metrics["active_sensor_count"].instant == 2
    public = report.to_public_dict()
    assert "control_command" not in str(public).lower()
    assert "waypoint" not in str(public).lower()
    assert "ground_truth" not in str(public).lower()


def test_sensor_failure_emits_evidence_only_event_after_confirmation() -> None:
    supervisor = ObservabilitySupervisor(
        _test_config(
            warmup_sec=0.0,
            periodic_feedback_sec=100.0,
            soft_confirmation_samples=2,
        )
    )
    supervisor.process_frame(_frame(0.0, sequence_id=0))
    assert supervisor.process_frame(_frame(1.0, sequence_id=1, active_uuv_count=1)) == ()
    reports = supervisor.process_frame(_frame(2.0, sequence_id=2, active_uuv_count=1))

    assert len(reports) == 1
    event_report = reports[0]
    assert event_report.report_type == "URGENT"
    assert any(
        event.hypothesis == "UUV_SENSOR_OR_COMM_FAILURE"
        for event in event_report.events
    )
    event = event_report.events[0]
    assert event.evidence
    assert event.confidence <= 1.0
    assert all(
        forbidden not in event.evidence
        for forbidden in ("ground truth", "control command", "waypoint")
    )


def test_event_cooldown_suppresses_repeated_episode_until_recovery() -> None:
    supervisor = ObservabilitySupervisor(
        _test_config(
            warmup_sec=0.0,
            periodic_feedback_sec=100.0,
            soft_confirmation_samples=1,
            event_cooldown_sec=10.0,
            recovery_stable_sec=2.0,
        )
    )
    supervisor.process_frame(_frame(0.0, sequence_id=0))
    first = supervisor.process_frame(_frame(1.0, sequence_id=1, active_uuv_count=1))
    assert any(event.hypothesis == "UUV_SENSOR_OR_COMM_FAILURE" for event in first[0].events)

    supervisor.process_frame(_frame(2.0, sequence_id=2, active_uuv_count=2))
    recovery = supervisor.process_frame(_frame(4.0, sequence_id=3, active_uuv_count=2))
    assert any(report.report_type == "RECOVERY" for report in recovery)

    suppressed = supervisor.process_frame(
        _frame(5.0, sequence_id=4, active_uuv_count=1)
    )
    assert not any(
        event.hypothesis == "UUV_SENSOR_OR_COMM_FAILURE"
        for report in suppressed
        for event in report.events
    )

    emitted = supervisor.process_frame(
        _frame(16.0, sequence_id=5, active_uuv_count=1)
    )
    assert any(
        event.hypothesis == "UUV_SENSOR_OR_COMM_FAILURE"
        for report in emitted
        for event in report.events
    )


def test_target_maneuver_uses_motion_and_innovation_evidence() -> None:
    supervisor = ObservabilitySupervisor(
        _test_config(
            warmup_sec=0.0,
            periodic_feedback_sec=100.0,
            soft_confirmation_samples=1,
        )
    )
    supervisor.process_frame(_frame(0.0, sequence_id=0))
    reports = supervisor.process_frame(
        _frame(
            1.0,
            sequence_id=1,
            speed_mps=5.0,
            velocity_xy_mps=(0.0, 5.0),
            innovation_rad=0.8,
        )
    )

    assert any(
        event.hypothesis == "TARGET_MANEUVER"
        for report in reports
        for event in report.events
    )


def test_clock_reset_is_reported_without_using_previous_truth() -> None:
    supervisor = ObservabilitySupervisor(_test_config(periodic_feedback_sec=100.0))
    supervisor.process_frame(_frame(20.0, sequence_id=20))
    reports = supervisor.process_frame(_frame(10.0, sequence_id=21))

    assert len(reports) == 1
    assert reports[0].report_type == "URGENT"
    assert reports[0].events[0].hypothesis == "CLOCK_RESET"
