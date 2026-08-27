from __future__ import annotations

from math import inf, nan, sqrt

import pytest

from underwater_tracking import verification as verification_api
from underwater_tracking.verification.uuv_tracking_coverage_audit import (
    command_motion_counts,
    deterministic_trace_digest,
    minimum_pairwise_separation_m,
    percentile_summary,
    sampled_footprint_fraction,
    target_position_errors_m,
    waypoint_visit_fraction,
)


def _trace_frames() -> tuple[dict[str, object], ...]:
    return (
        {
            "sim_time_s": 0,
            "uuvs": [
                {
                    "platform_id": "uuv_00",
                    "position_xy": [0.0, 0.0],
                    "deployment_state": "deployed",
                },
                {
                    "platform_id": "uuv_01",
                    "position_xy": [3.0, 4.0],
                    "deployment_state": "deployed",
                },
            ],
            "tracks": [
                {
                    "target_id": "target_00",
                    "sim_time_s": 0,
                    "mean": [1.0, 0.0, 0.0, 0.0],
                }
            ],
            "target_truth": [
                {"target_id": "target_00", "position_xy": [0.0, 0.0]}
            ],
            "waypoint_commands": {"target_00": {"uuv_00": [2.0, 0.0]}},
        },
        {
            "sim_time_s": 5,
            "uuvs": [
                {
                    "platform_id": "uuv_00",
                    "position_xy": [1.0, 0.0],
                    "deployment_state": "deployed",
                },
                {
                    "platform_id": "uuv_01",
                    "position_xy": [3.0, 4.0],
                    "deployment_state": "deployed",
                },
            ],
            "tracks": [
                {
                    "target_id": "target_00",
                    "sim_time_s": 5,
                    "mean": [0.0, 2.0, 0.0, 0.0],
                }
            ],
            "target_truth": [
                {"target_id": "target_00", "position_xy": [0.0, 0.0]}
            ],
            "waypoint_commands": {},
        },
    )


def test_tracking_control_and_separation_metrics_use_the_same_frames() -> None:
    frames = _trace_frames()

    assert target_position_errors_m(frames, "target_00") == pytest.approx((1.0, 2.0))
    assert minimum_pairwise_separation_m(frames) == pytest.approx(sqrt(20.0))
    assert command_motion_counts(frames) == {
        "commanded_intervals": 1,
        "moved_intervals": 1,
    }


def test_position_metrics_reject_non_finite_coordinates() -> None:
    invalid_track_frame = (
        {
            "sim_time_s": 0,
            "tracks": [
                {
                    "target_id": "target_00",
                    "sim_time_s": 0,
                    "mean": [nan, 0.0],
                }
            ],
            "target_truth": [
                {"target_id": "target_00", "position_xy": [0.0, 0.0]}
            ],
        },
    )
    invalid_separation_frame = (
        {
            "uuvs": [
                {
                    "platform_id": "uuv_00",
                    "position_xy": [0.0, 0.0],
                    "deployment_state": "deployed",
                },
                {
                    "platform_id": "uuv_01",
                    "position_xy": [inf, 0.0],
                    "deployment_state": "deployed",
                },
            ]
        },
    )

    with pytest.raises(ValueError, match="finite"):
        target_position_errors_m(invalid_track_frame, "target_00")
    with pytest.raises(ValueError, match="finite"):
        minimum_pairwise_separation_m(invalid_separation_frame)


def test_tracking_errors_only_count_fresh_unique_track_timestamps() -> None:
    frames = (
        {
            "sim_time_s": 10,
            "tracks": [
                {
                    "target_id": "target_00",
                    "sim_time_s": 5,
                    "mean": [100.0, 0.0],
                }
            ],
            "target_truth": [
                {"target_id": "target_00", "position_xy": [0.0, 0.0]}
            ],
        },
        {
            "sim_time_s": 15,
            "tracks": [
                {
                    "target_id": "target_00",
                    "sim_time_s": 15,
                    "mean": [3.0, 4.0],
                }
            ],
            "target_truth": [
                {"target_id": "target_00", "position_xy": [0.0, 0.0]}
            ],
        },
        {
            "sim_time_s": 15,
            "tracks": [
                {
                    "target_id": "target_00",
                    "sim_time_s": 15,
                    "mean": [6.0, 8.0],
                }
            ],
            "target_truth": [
                {"target_id": "target_00", "position_xy": [0.0, 0.0]}
            ],
        },
    )

    assert target_position_errors_m(frames, "target_00") == pytest.approx((5.0,))


@pytest.mark.parametrize(
    ("frame_time", "track_time"),
    ((inf, 0.0), (0.0, inf)),
)
def test_tracking_errors_reject_non_finite_timestamps(
    frame_time: float,
    track_time: float,
) -> None:
    frames = (
        {
            "sim_time_s": frame_time,
            "tracks": [
                {
                    "target_id": "target_00",
                    "sim_time_s": track_time,
                    "mean": [0.0, 0.0],
                }
            ],
            "target_truth": [
                {"target_id": "target_00", "position_xy": [0.0, 0.0]}
            ],
        },
    )

    with pytest.raises(ValueError, match="sim_time_s.*finite"):
        target_position_errors_m(frames, "target_00")


@pytest.mark.parametrize(
    ("frame_time", "track_time"),
    ((True, 1), (1, True)),
)
def test_tracking_errors_skip_boolean_timestamps(
    frame_time: object,
    track_time: object,
) -> None:
    frames = (
        {
            "sim_time_s": frame_time,
            "tracks": [
                {
                    "target_id": "target_00",
                    "sim_time_s": track_time,
                    "mean": [1.0, 0.0],
                }
            ],
            "target_truth": [
                {"target_id": "target_00", "position_xy": [0.0, 0.0]}
            ],
        },
    )

    assert target_position_errors_m(frames, "target_00") == ()


def test_command_motion_ignores_uuvs_not_deployed_in_both_frames() -> None:
    frames = (
        {
            "uuvs": [
                {
                    "platform_id": "uuv_00",
                    "position_xy": [0.0, 0.0],
                    "deployment_state": "onboard",
                }
            ],
            "waypoint_commands": {"target_00": {"uuv_00": [2.0, 0.0]}},
        },
        {
            "uuvs": [
                {
                    "platform_id": "uuv_00",
                    "position_xy": [1.0, 0.0],
                    "deployment_state": "deployed",
                }
            ],
            "waypoint_commands": {"target_00": {"uuv_00": [3.0, 0.0]}},
        },
        {
            "uuvs": [
                {
                    "platform_id": "uuv_00",
                    "position_xy": [2.0, 0.0],
                    "deployment_state": "returning",
                }
            ],
            "waypoint_commands": {},
        },
    )

    assert command_motion_counts(frames) == {
        "commanded_intervals": 0,
        "moved_intervals": 0,
    }


def test_route_progress_requires_physical_waypoint_visits() -> None:
    trajectory = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    route = ((0.0, 0.0), (1.0, 0.0), (3.0, 0.0))

    assert waypoint_visit_fraction(trajectory, route) == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize("tolerance", (-1.0, inf))
def test_route_progress_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="numerical_tolerance_m"):
        waypoint_visit_fraction(((0.0, 0.0),), ((0.0, 0.0),), numerical_tolerance_m=tolerance)


def test_route_progress_rejects_non_finite_coordinates() -> None:
    with pytest.raises(ValueError, match="finite"):
        waypoint_visit_fraction(((0.0, 0.0),), ((nan, 0.0),))


def test_sampled_footprint_is_unavailable_without_emissions_and_complete_for_large_range() -> None:
    polygon = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))

    assert sampled_footprint_fraction(polygon, (), samples_per_axis=11) is None
    assert sampled_footprint_fraction(
        polygon,
        (((5.0, 5.0), 100.0),),
        samples_per_axis=11,
    ) == pytest.approx(1.0)


def test_sampled_footprint_rejects_an_invalid_grid_size() -> None:
    polygon = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))

    with pytest.raises(ValueError, match="samples_per_axis"):
        sampled_footprint_fraction(
            polygon,
            (((5.0, 5.0), 1.0),),
            samples_per_axis=1,
        )


@pytest.mark.parametrize(
    ("polygon", "emissions"),
    (
        (
            ((nan, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
            (((5.0, 5.0), 1.0),),
        ),
        (
            ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
            (((inf, 5.0), 1.0),),
        ),
    ),
)
def test_sampled_footprint_rejects_non_finite_coordinates(
    polygon: tuple[tuple[float, float], ...],
    emissions: tuple[tuple[tuple[float, float], float], ...],
) -> None:
    with pytest.raises(ValueError, match="finite"):
        sampled_footprint_fraction(polygon, emissions, samples_per_axis=3)


@pytest.mark.parametrize("radius", (-1.0, 0.0, inf))
def test_sampled_footprint_rejects_invalid_radius(radius: float) -> None:
    polygon = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))

    with pytest.raises(ValueError, match="radius"):
        sampled_footprint_fraction(
            polygon,
            (((5.0, 5.0), radius),),
            samples_per_axis=3,
        )


@pytest.mark.parametrize(
    "boundary_point",
    (
        (0.0, 0.0),
        (5.0, 0.0),
        (10.0, 0.0),
        (10.0, 5.0),
        (10.0, 10.0),
        (5.0, 10.0),
        (0.0, 10.0),
        (0.0, 5.0),
    ),
)
def test_sampled_footprint_treats_rectangle_boundaries_symmetrically(
    boundary_point: tuple[float, float],
) -> None:
    polygon = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))

    assert sampled_footprint_fraction(
        polygon,
        ((boundary_point, 0.1),),
        samples_per_axis=3,
    ) == pytest.approx(1.0 / 9.0)


def test_sampled_footprint_is_unavailable_for_zero_area_polygon() -> None:
    polygon = ((0.0, 0.0), (5.0, 0.0), (10.0, 0.0))

    assert sampled_footprint_fraction(
        polygon,
        (((5.0, 0.0), 10.0),),
        samples_per_axis=3,
    ) is None


def test_trace_digest_is_canonical_and_seed_sensitive() -> None:
    first = {"seed": 42, "frames": list(_trace_frames())}
    second = {"frames": list(_trace_frames()), "seed": 42}

    assert deterministic_trace_digest(first) == deterministic_trace_digest(second)
    assert deterministic_trace_digest(first) != deterministic_trace_digest(
        {**first, "seed": 43}
    )


def test_percentile_summary_reports_expected_statistics_and_empty_input() -> None:
    assert percentile_summary(()) is None
    assert percentile_summary((0.0, 3.0, 4.0)) == pytest.approx(
        {
            "rmse": sqrt(25.0 / 3.0),
            "median": 3.0,
            "p95": 3.9,
            "maximum": 4.0,
        }
    )


@pytest.mark.parametrize("value", (nan, inf))
def test_percentile_summary_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        percentile_summary((1.0, value))


def test_verification_package_keeps_existing_and_new_stable_exports() -> None:
    expected = {
        "BattleEvidenceChain",
        "EntityMotionAudit",
        "EntityMotionLimits",
        "FullBattleAcceptance",
        "PhysicsInvariantMonitor",
        "command_motion_counts",
        "deterministic_trace_digest",
        "minimum_pairwise_separation_m",
        "percentile_summary",
        "sampled_footprint_fraction",
        "target_position_errors_m",
        "waypoint_visit_fraction",
    }

    assert expected <= set(verification_api.__all__)


def test_trace_metrics_ignore_non_collection_payloads() -> None:
    malformed_frames = (
        {
            "uuvs": "not-a-uuv-sequence",
            "tracks": b"not-a-track-sequence",
            "target_truth": "not-a-truth-sequence",
            "waypoint_commands": "not-a-command-mapping",
        },
        {"uuvs": (), "tracks": (), "target_truth": (), "waypoint_commands": {}},
    )

    assert target_position_errors_m(malformed_frames, "target_00") == ()
    assert minimum_pairwise_separation_m(malformed_frames) is None
    assert command_motion_counts(malformed_frames) == {
        "commanded_intervals": 0,
        "moved_intervals": 0,
    }
