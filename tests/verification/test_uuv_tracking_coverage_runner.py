from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from underwater_tracking.agent.llm import LLMError
from underwater_tracking.verification.uuv_tracking_coverage_runner import (
    NoNetworkLLM,
    REMEDIATION_METRIC_NAMES,
    _deterministic_trace_digest,
    _write_json,
    project_audit_frame,
    run_audit,
    run_duration_audit,
    run_once,
    summarize_trace,
)


def _complete_physics_audit(*, steps: int = 2) -> dict[str, object]:
    entity_ids = ("uuv_00",)
    monitored_frames = steps + 1
    return {
        "entity_count": 1,
        "audits": [
            {
                "entity_id": "uuv_00",
                "limit_violation_count": 0,
                "teleport_count": 0,
                "boundary_violation_count": 0,
            }
        ],
        "coverage": {
            "expected_entity_ids": entity_ids,
            "expected_entity_count": 1,
            "observed_entity_ids": entity_ids,
            "observed_entity_count": 1,
            "observed_frame_count": monitored_frames,
            "observed_frame_observation_count": monitored_frames,
            "first_frame_id": 0,
            "last_frame_id": steps,
            "duplicate_frame_ids": (),
            "duplicate_entity_frame_ids": (),
            "missing_entity_frame_ids": {},
            "frame_id_gaps": (),
            "nonmonotonic_frame_ids": (),
            "nonmonotonic_sim_time_frame_ids": (),
            "inconsistent_sample_frame_ids": (),
            "physics_step_s": 5,
            "sequence_expected_frame_count": monitored_frames,
        },
    }


def _minimal_trace(physics_audit: object) -> dict[str, object]:
    return {
        "scenario": "synthetic",
        "seed": 42,
        "steps": 2,
        "routes": {"R1": {"uuv_00": [[0.0, 0.0], [1.0, 0.0]]}},
        "regions": {
            "R1": {
                "target_id": "target_00",
                "polygon": [
                    [-1.0, -1.0],
                    [2.0, -1.0],
                    [2.0, 1.0],
                    [-1.0, 1.0],
                ],
            }
        },
        "active_ranges_m": {"uuv_00": 100.0},
        "frames": [
            {
                "sim_time_s": 5,
                "uuvs": [
                    {
                        "platform_id": "uuv_00",
                        "position_xy": [0.0, 0.0],
                        "deployment_state": "deployed",
                    }
                ],
                "tracks": [
                    {
                        "target_id": "target_00",
                        "sim_time_s": 5,
                        "mean": [0.0, 0.0, 0.0, 0.0],
                    }
                ],
                "target_truth": [
                    {"target_id": "target_00", "position_xy": [0.0, 0.0]}
                ],
                "events": [],
                "waypoint_commands": {"target_00": {"uuv_00": [1.0, 0.0]}},
            },
            {
                "sim_time_s": 10,
                "uuvs": [
                    {
                        "platform_id": "uuv_00",
                        "position_xy": [1.0, 0.0],
                        "deployment_state": "deployed",
                    }
                ],
                "tracks": [
                    {
                        "target_id": "target_00",
                        "sim_time_s": 10,
                        "mean": [0.0, 0.0, 0.0, 0.0],
                    }
                ],
                "target_truth": [
                    {"target_id": "target_00", "position_xy": [0.0, 0.0]}
                ],
                "events": [],
                "waypoint_commands": {},
            },
        ],
        "physics_audit_scope": "post_deterministic_baseline",
        "physics_audit_initial_conditions": {
            "frame_id": 0,
            "sim_time_s": 0,
            "deployed_uuv_ids": ["uuv_00"],
        },
        "physics_audit": physics_audit,
        "verification_evidence": {"public_observation_ids": []},
    }


def test_no_network_llm_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="network LLM is disabled"):
        try:
            NoNetworkLLM().invoke_structured("strategy", {}, dict)
        except RuntimeError as exc:
            assert isinstance(exc, LLMError)
            raise


def test_trace_summary_reports_all_llm_remediation_metrics() -> None:
    trace = _minimal_trace(_complete_physics_audit())
    trace["event_sequence"] = [
        {
            "event_id": "refresh-1-due",
            "event_type": "execution_refresh_due",
            "sim_time_s": 330,
            "payload": {"reason_code": "deadline_margin"},
        },
        {
            "event_id": "refresh-1-attempted",
            "event_type": "execution_refresh_attempted",
            "sim_time_s": 330,
            "payload": {},
        },
        {
            "event_id": "refresh-1-committed",
            "event_type": "execution_refresh_committed",
            "sim_time_s": 330,
            "payload": {},
        },
        {
            "event_id": "refresh-2-attempted",
            "event_type": "execution_refresh_attempted",
            "sim_time_s": 780,
            "payload": {},
        },
        {
            "event_id": "refresh-2-rejected",
            "event_type": "execution_refresh_rejected",
            "sim_time_s": 780,
            "payload": {"reason_code": "stale_execution_revision"},
        },
        {
            "event_id": "refresh-2-recovered",
            "event_type": "execution_snapshot_recovered",
            "sim_time_s": 810,
            "payload": {},
        },
        {
            "event_id": "planning-stale",
            "event_type": "planning_epoch_failed",
            "sim_time_s": 840,
            "payload": {"failure_category": "stale"},
        },
    ]
    trace["frames"][0]["operational_frame"] = {
        "frame_id": 1,
        "execution": {
            "execution_revision": 1,
            "health_status": "current",
            "data_age_s": 12.0,
        },
        "events": [],
        "uuvs": [],
    }
    trace["frames"][1]["operational_frame"] = {
        "frame_id": 2,
        "execution": {
            "execution_revision": 2,
            "health_status": "expired",
            "data_age_s": 123.5,
        },
        "events": [],
        "uuvs": [],
    }
    trace["frames"][0]["agent_telemetry"] = {
        "llm_failure_count": 1,
        "physics_time_advanced": False,
    }
    trace["assistant_cases"] = [
        {
            "case": "expired",
            "request": {"frame_id": 2, "execution_revision": 2},
            "response": {"frame_id": 2, "execution_revision": 3},
        }
    ]
    trace["memory_episodes"] = [
        {"episode_id": "episode-1", "source_event_ids": ["missing-event"]}
    ]

    summary = summarize_trace(trace)

    assert set(REMEDIATION_METRIC_NAMES) == {
        "execution_refresh_attempt_count",
        "execution_refresh_commit_count",
        "execution_refresh_rejection_count",
        "execution_expired_frame_count",
        "execution_recovery_count",
        "max_execution_data_age_s",
        "stale_llm_result_count",
        "llm_failure_physics_gap_count",
        "assistant_answer_revision_mismatch_count",
        "memory_episode_source_gap_count",
    }
    assert {
        name: summary[name]
        for name in REMEDIATION_METRIC_NAMES
    } == {
        "execution_refresh_attempt_count": 2,
        "execution_refresh_commit_count": 1,
        "execution_refresh_rejection_count": 1,
        "execution_expired_frame_count": 1,
        "execution_recovery_count": 1,
        "max_execution_data_age_s": 123.5,
        "stale_llm_result_count": 2,
        "llm_failure_physics_gap_count": 1,
        "assistant_answer_revision_mismatch_count": 1,
        "memory_episode_source_gap_count": 1,
    }


def test_audit_digest_ignores_wall_clock_planning_deadline() -> None:
    first = {
        "frames": [
            {
                "operational_frame": {
                    "planning": {
                        "deadline_utc_ms": 1000,
                        "attempt": 1,
                        "last_result_status": "failed",
                        "status": "queued",
                    }
                },
                "transport_hash": "hash-a",
                "audiences": ["operator_audit", "memory_source"],
            }
        ]
    }
    second = {
        "frames": [
            {
                "operational_frame": {
                    "planning": {
                        "deadline_utc_ms": 2000,
                        "attempt": 2,
                        "last_result_status": "failed",
                        "status": "degraded",
                    }
                },
                "transport_hash": "hash-b",
                "audiences": ["memory_source", "operator_audit"],
            }
        ]
    }

    assert _deterministic_trace_digest(first) == _deterministic_trace_digest(second)


def test_audit_digest_ignores_scheduler_only_telemetry() -> None:
    first = {
        "frames": [
            {
                "operational_frame": {
                    "planning": {
                        "last_result_status": "failed",
                        "status": "degraded",
                        "queued_event_count": 2,
                    }
                },
                "agent_telemetry": {
                    "llm_failure_count": 2,
                    "carrier_error_count": 1,
                    "physics_time_advanced": True,
                }
            }
        ]
    }
    second = {
        "frames": [
            {
                "operational_frame": {
                    "planning": {
                        "last_result_status": "failed",
                        "status": "degraded",
                        "queued_event_count": 3,
                    }
                },
                "agent_telemetry": {
                    "llm_failure_count": 3,
                    "carrier_error_count": 1,
                    "physics_time_advanced": True,
                }
            }
        ]
    }

    assert _deterministic_trace_digest(first) == _deterministic_trace_digest(second)


def test_projected_audit_events_have_stable_audience_order() -> None:
    projected = project_audit_frame(
        {
            "sim_time_s": 5,
            "events": [
                {"event_type": "observation", "audiences": ["z", "a"]}
            ],
        },
        {"sim_time_s": 5, "targets": []},
        mission_modes={},
        region_lifecycles={},
    )

    assert projected["events"][0]["audiences"] == ["a", "z"]


def test_two_step_runner_records_assistant_and_memory_audit_inputs(
    tmp_path: Path,
) -> None:
    result = run_once(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=2,
        work_dir=tmp_path / "run",
    )

    assert result["event_sequence"]
    assert len(result["assistant_cases"]) == 4
    assert result["memory_episodes"] is not None
    for item in result["assistant_cases"]:
        assert item["request"]["frame_id"] is not None
        assert item["request"]["execution_revision"] is not None
        assert item["response"]["frame_id"] is not None
        assert item["response"]["execution_revision"] is not None


def test_projected_frame_pairs_truth_without_mutating_operational_input() -> None:
    operational = {
        "sim_time_s": 5,
        "uuvs": [],
        "tracks": [],
        "events": [],
        "waypoint_commands": {},
    }
    original = dict(operational)
    truth = {
        "sim_time_s": 5,
        "targets": [{"target_id": "target_00", "position_xy": [1.0, 2.0]}],
    }

    projected = project_audit_frame(
        operational,
        truth,
        mission_modes={"uuv_00": "ACTIVE_SCAN"},
        region_lifecycles={"R1": "ACTIVE_SCAN"},
        region_assignments={
            "R1": {
                "active_scan_uuv_ids": ["uuv_00"],
                "passive_track_uuv_ids": ["uuv_01"],
            }
        },
    )

    assert operational == original
    assert "target_truth" not in operational
    assert projected["target_truth"] == truth["targets"]
    assert projected["mission_modes"] == {"uuv_00": "ACTIVE_SCAN"}
    assert projected["region_lifecycles"] == {"R1": "ACTIVE_SCAN"}
    assert projected["region_assignments"]["R1"]["passive_track_uuv_ids"] == [
        "uuv_01"
    ]


def test_projected_frame_rejects_mismatched_truth_time() -> None:
    with pytest.raises(ValueError, match="share sim_time_s"):
        project_audit_frame(
            {"sim_time_s": 5},
            {"sim_time_s": 10},
            mission_modes={},
            region_lifecycles={},
        )


def test_two_step_runner_uses_repository_baseline_without_network(tmp_path: Path) -> None:
    result = run_once(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=2,
        work_dir=tmp_path / "run",
    )

    assert result["seed"] == 42
    assert result["steps"] == 2
    assert len(result["frames"]) == 2
    assert result["routes"]
    assert result["regions"]
    assert all(frame["target_truth"] for frame in result["frames"])


def test_runner_reports_authoritative_three_uuv_execution_metrics(tmp_path: Path) -> None:
    result = run_once(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=60,
        work_dir=tmp_path / "run",
    )

    summary = summarize_trace(result)
    runtime = summary["runtime_execution"]
    assert summary["status"] == "PASS"
    assert runtime["available"] is True
    assert runtime["valid"] is True
    assert summary["region_side_m"] == 2_000.0
    assert summary["target_detection_radius_m"] == 1_000.0
    assert summary["uuv_detection_radius_m"] == 600.0
    assert summary["task_group_size"] == 3
    assert summary["max_coverage_gap_area_m2"] == pytest.approx(0.0)
    assert summary["active_ping_count_during_passive"] == 0
    assert summary["tracking_owner_gap_frames"] == 0
    assert summary["max_visible_uuv_count"] == 12
    assert summary["hard_checks"]["runtime_execution_contract"] is True


def test_two_step_runner_accepts_constructor_and_step_physics_frames(
    tmp_path: Path,
) -> None:
    result = run_once(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=2,
        work_dir=tmp_path / "run",
    )
    physics = result["physics_audit"]
    assert isinstance(physics, dict)
    coverage = physics["coverage"]
    assert result["physics_audit_scope"] == "post_deterministic_baseline"
    initial_conditions = result["physics_audit_initial_conditions"]
    assert initial_conditions["frame_id"] == 0
    assert initial_conditions["sim_time_s"] == 0
    assert initial_conditions["deployed_uuv_ids"]
    first_deployed = tuple(
        sorted(
            item["platform_id"]
            for item in result["frames"][0]["uuvs"]
            if item["deployment_state"] == "deployed"
        )
    )
    assert tuple(initial_conditions["deployed_uuv_ids"]) == first_deployed

    assert coverage["observed_frame_count"] == 3
    assert coverage["observed_frame_observation_count"] == 3
    assert coverage["sequence_expected_frame_count"] == 3
    assert coverage["first_frame_id"] == 0
    assert coverage["last_frame_id"] == 2
    assert coverage["expected_entity_count"] == physics["entity_count"]
    assert coverage["observed_entity_count"] == physics["entity_count"]
    assert coverage["expected_entity_ids"] == coverage["observed_entity_ids"]
    assert set(coverage["expected_entity_ids"]) == {
        audit["entity_id"] for audit in physics["audits"]
    }
    for field in (
        "duplicate_frame_ids",
        "duplicate_entity_frame_ids",
        "missing_entity_frame_ids",
        "frame_id_gaps",
        "nonmonotonic_frame_ids",
        "nonmonotonic_sim_time_frame_ids",
        "inconsistent_sample_frame_ids",
    ):
        assert not coverage[field], (field, coverage[field])
    summary = summarize_trace(result)
    assert summary["physics_violation_count"] == 0, [
        (
            audit["entity_id"],
            audit["limit_violation_count"],
            audit["teleport_count"],
            audit["boundary_violation_count"],
            audit["violating_frame_ids"],
        )
        for audit in physics["audits"]
        if audit["limit_violation_count"]
    ]
    assert summary["hard_checks"]["configured_physics_invariants"] is True, physics
    assert summary["physics_audit_scope"] == "post_deterministic_baseline"


def test_runner_rejects_non_positive_steps_without_creating_output(tmp_path: Path) -> None:
    work_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="steps must be positive"):
        run_once(
            config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
            seed=42,
            steps=0,
            work_dir=work_dir,
        )

    assert not work_dir.exists()


def test_trace_summary_uses_physical_ping_emitter_and_same_frame_truth() -> None:
    trace = {
        "scenario": "synthetic",
        "seed": 42,
        "steps": 2,
        "routes": {"R1": {"uuv_00": [[0.0, 0.0], [1.0, 0.0]]}},
        "regions": {
            "R1": {
                "target_id": "target_00",
                "polygon": [
                    [-1.0, -1.0],
                    [2.0, -1.0],
                    [2.0, 1.0],
                    [-1.0, 1.0],
                ],
            }
        },
        "active_ranges_m": {"uuv_00": 100.0},
        "frames": [
            {
                "sim_time_s": 5,
                "uuvs": [
                    {
                        "platform_id": "uuv_00",
                        "position_xy": [0.0, 0.0],
                        "deployment_state": "deployed",
                    }
                ],
                "tracks": [
                    {
                        "target_id": "target_00",
                        "sim_time_s": 5,
                        "mean": [0.0, 0.0, 0.0, 0.0],
                    }
                ],
                "target_truth": [
                    {"target_id": "target_00", "position_xy": [0.0, 0.0]}
                ],
                "events": [
                    {
                        "event_type": "active_ping",
                        "entity_id": "target_00",
                        "payload": {"emitter_id": "uuv_00"},
                    }
                ],
                "waypoint_commands": {"target_00": {"uuv_00": [1.0, 0.0]}},
            },
            {
                "sim_time_s": 10,
                "uuvs": [
                    {
                        "platform_id": "uuv_00",
                        "position_xy": [1.0, 0.0],
                        "deployment_state": "deployed",
                    }
                ],
                "tracks": [
                    {
                        "target_id": "target_00",
                        "sim_time_s": 10,
                        "mean": [0.0, 0.0, 0.0, 0.0],
                    }
                ],
                "target_truth": [
                    {"target_id": "target_00", "position_xy": [0.0, 0.0]}
                ],
                "events": [],
                "waypoint_commands": {},
            },
        ],
        "physics_audit_scope": "post_deterministic_baseline",
        "physics_audit_initial_conditions": {
            "frame_id": 0,
            "sim_time_s": 0,
            "deployed_uuv_ids": ["uuv_00"],
        },
        "physics_audit": _complete_physics_audit(),
        "verification_evidence": {"public_observation_ids": ["obs-1"]},
    }

    summary = summarize_trace(trace)

    coverage = summary["coverage"]["R1"]
    assert coverage["active_emission_count"] == 1
    assert coverage["sampled_active_sonar_footprint_fraction"] == pytest.approx(1.0)
    assert summary["evidence"] == {"public_observation_count": 1}
    assert summary["status"] == "PASS"


def test_trace_summary_fails_when_assigned_pair_never_establishes_300_metres() -> None:
    trace = _minimal_trace(_complete_physics_audit())
    trace["regions"]["R1"].update(
        {
            "active_scan_uuv_ids": ["uuv_00"],
            "passive_track_uuv_ids": ["uuv_01"],
        }
    )
    for frame in trace["frames"]:
        frame["uuvs"].append(
            {
                "platform_id": "uuv_01",
                "position_xy": [100.0, 0.0],
                "deployment_state": "deployed",
            }
        )

    summary = summarize_trace(trace)

    assert summary["control_and_motion"]["minimum_pairwise_separation_m"] == 99.0
    pair = summary["control_and_motion"]["assigned_group_separation"][
        "R1:uuv_00|uuv_01"
    ]
    assert pair["separation_established_at_s"] is None
    assert pair["safe_after_establishment"] is False
    assert (
        summary["hard_checks"]["assigned_group_separation_after_establishment"]
        is False
    )
    assert summary["status"] == "FAIL"


def test_trace_summary_allows_shared_deployment_then_requires_safe_spacing() -> None:
    trace = _minimal_trace(_complete_physics_audit())
    trace["regions"]["R1"].update(
        {
            "active_scan_uuv_ids": ["uuv_00"],
            "passive_track_uuv_ids": ["uuv_01"],
        }
    )
    second_positions = ((0.0, 0.0), (401.0, 0.0))
    for frame, position in zip(trace["frames"], second_positions, strict=True):
        frame["uuvs"].append(
            {
                "platform_id": "uuv_01",
                "position_xy": list(position),
                "deployment_state": "deployed",
            }
        )

    summary = summarize_trace(trace)

    pair = summary["control_and_motion"]["assigned_group_separation"][
        "R1:uuv_00|uuv_01"
    ]
    assert pair["minimum_observed_separation_m"] == 0.0
    assert pair["separation_established_at_s"] == 10.0
    assert pair["minimum_after_establishment_m"] == 400.0
    assert pair["safe_after_establishment"] is True
    assert summary["hard_checks"][
        "assigned_group_separation_after_establishment"
    ] is True


def test_trace_summary_follows_dynamic_task_group_membership() -> None:
    trace = _minimal_trace(_complete_physics_audit())
    trace["frames"][0]["region_assignments"] = {
        "R1": {
            "active_scan_uuv_ids": ["uuv_00"],
            "passive_track_uuv_ids": ["uuv_01"],
        }
    }
    trace["frames"][0]["uuvs"].append(
        {
            "platform_id": "uuv_01",
            "position_xy": [400.0, 0.0],
            "deployment_state": "deployed",
        }
    )
    trace["frames"][1]["region_assignments"] = {
        "R1": {
            "active_scan_uuv_ids": ["uuv_00"],
            "passive_track_uuv_ids": ["uuv_02"],
        }
    }
    trace["frames"][1]["uuvs"].extend(
        (
            {
                "platform_id": "uuv_01",
                "position_xy": [2.0, 0.0],
                "deployment_state": "deployed",
            },
            {
                "platform_id": "uuv_02",
                "position_xy": [401.0, 0.0],
                "deployment_state": "deployed",
            },
        )
    )

    summary = summarize_trace(trace)

    pairs = summary["control_and_motion"]["assigned_group_separation"]
    assert set(pairs) == {"R1:uuv_00|uuv_01", "R1:uuv_00|uuv_02"}
    assert all(pair["safe_after_establishment"] for pair in pairs.values())


def test_trace_summary_rejects_zero_area_region_geometry() -> None:
    trace = {
        "scenario": "synthetic",
        "seed": 42,
        "steps": 1,
        "routes": {"R1": {"uuv_00": [[0.0, 0.0], [1.0, 0.0]]}},
        "regions": {
            "R1": {
                "target_id": "target_00",
                "polygon": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            }
        },
        "active_ranges_m": {"uuv_00": 100.0},
        "frames": [
            {
                "sim_time_s": 5,
                "uuvs": [
                    {
                        "platform_id": "uuv_00",
                        "position_xy": [0.0, 0.0],
                        "deployment_state": "deployed",
                    }
                ],
                "tracks": [
                    {
                        "target_id": "target_00",
                        "sim_time_s": 5,
                        "mean": [0.0, 0.0, 0.0, 0.0],
                    }
                ],
                "target_truth": [
                    {"target_id": "target_00", "position_xy": [0.0, 0.0]}
                ],
                "events": [],
                "waypoint_commands": {},
            }
        ],
        "physics_audit_scope": "post_deterministic_baseline",
        "physics_audit_initial_conditions": {
            "frame_id": 0,
            "sim_time_s": 0,
            "deployed_uuv_ids": ["uuv_00"],
        },
        "physics_audit": _complete_physics_audit(steps=1),
        "verification_evidence": {"public_observation_ids": []},
    }

    summary = summarize_trace(trace)

    assert summary["hard_checks"]["assigned_route_geometry_valid"] is False


def test_physics_gate_rejects_missing_or_empty_audits() -> None:
    missing = summarize_trace(_minimal_trace({}))
    empty = summarize_trace(
        _minimal_trace(
            {
                **_complete_physics_audit(),
                "entity_count": 0,
                "audits": [],
            }
        )
    )
    count_mismatch = _complete_physics_audit()
    count_mismatch["audits"] = []

    assert missing["hard_checks"]["configured_physics_invariants"] is False
    assert empty["hard_checks"]["configured_physics_invariants"] is False
    assert (
        summarize_trace(_minimal_trace(count_mismatch))["hard_checks"][
            "configured_physics_invariants"
        ]
        is False
    )


@pytest.mark.parametrize("value", (None, "0", -1, True))
def test_physics_gate_rejects_malformed_limit_violation_count(value: object) -> None:
    physics = _complete_physics_audit()
    physics["audits"] = [
        {"entity_id": "uuv_00", "limit_violation_count": value}
    ]

    summary = summarize_trace(_minimal_trace(physics))

    assert summary["hard_checks"]["configured_physics_invariants"] is False


@pytest.mark.parametrize(
    "derived_field",
    ("teleport_count", "boundary_violation_count"),
)
def test_physics_violation_total_does_not_double_count_derived_categories(
    derived_field: str,
) -> None:
    physics = _complete_physics_audit()
    physics["audits"] = [
        {
            "entity_id": "uuv_00",
            "limit_violation_count": 1,
            derived_field: 1,
        }
    ]

    summary = summarize_trace(_minimal_trace(physics))

    assert summary["physics_violation_count"] == 1
    assert summary["hard_checks"]["configured_physics_invariants"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("duplicate_frame_ids", (1,)),
        ("duplicate_entity_frame_ids", ("uuv_00@1",)),
        ("missing_entity_frame_ids", {"uuv_00": (2,)}),
        ("frame_id_gaps", (2,)),
        ("nonmonotonic_frame_ids", (1,)),
        ("nonmonotonic_sim_time_frame_ids", (1,)),
        ("inconsistent_sample_frame_ids", (1,)),
    ),
)
def test_physics_gate_rejects_coverage_sequence_anomalies(
    field: str,
    value: object,
) -> None:
    physics = _complete_physics_audit()
    coverage = physics["coverage"]
    assert isinstance(coverage, dict)
    coverage[field] = value

    summary = summarize_trace(_minimal_trace(physics))

    assert summary["hard_checks"]["configured_physics_invariants"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("observed_frame_count", 1),
        ("sequence_expected_frame_count", 1),
        ("observed_frame_observation_count", 1),
    ),
)
def test_physics_gate_requires_complete_trace_frame_coverage(
    field: str,
    value: object,
) -> None:
    physics = _complete_physics_audit()
    coverage = physics["coverage"]
    assert isinstance(coverage, dict)
    coverage[field] = value

    summary = summarize_trace(_minimal_trace(physics))

    assert summary["hard_checks"]["configured_physics_invariants"] is False


def test_physics_gate_requires_matching_expected_and_observed_entities() -> None:
    physics = _complete_physics_audit()
    coverage = physics["coverage"]
    assert isinstance(coverage, dict)
    coverage["observed_entity_ids"] = ()
    coverage["observed_entity_count"] = 0

    summary = summarize_trace(_minimal_trace(physics))

    assert summary["hard_checks"]["configured_physics_invariants"] is False


def test_json_serialization_failure_does_not_create_partial_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.json"

    with pytest.raises(TypeError):
        _write_json(path, {"unsupported": object()}, pretty=False)

    assert not path.exists()


def test_final_audit_requires_two_runs_without_creating_output(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    evidence_dir = tmp_path / "evidence"

    with pytest.raises(ValueError, match="exactly two"):
        run_audit(
            config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
            seed=42,
            steps=2,
            repeat=1,
            work_dir=work_dir,
            evidence_dir=evidence_dir,
        )

    assert not work_dir.exists()
    assert not evidence_dir.exists()


def test_final_audit_refuses_existing_evidence_without_creating_runs(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_audit(
            config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
            seed=42,
            steps=2,
            repeat=2,
            work_dir=work_dir,
            evidence_dir=evidence_dir,
        )

    assert not work_dir.exists()


def test_final_audit_writes_equal_fixed_seed_digests(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"

    metrics = run_audit(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=2,
        repeat=2,
        work_dir=tmp_path / "work",
        evidence_dir=evidence_dir,
    )

    digests = metrics["trace_digests"]
    assert digests["run-a"] == digests["run-b"]
    assert metrics["hard_checks"]["deterministic_repeat"] is True
    assert (evidence_dir / "trajectory.json").is_file()
    assert (evidence_dir / "metrics.json").is_file()


def test_duration_audit_writes_one_trace_and_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "duration-audit"

    metrics = run_duration_audit(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        duration_s=10,
        output_dir=output_dir,
    )

    assert metrics["requested_duration_s"] == 10.0
    assert metrics["actual_duration_s"] == 10.0
    assert isinstance(metrics["trace_digest"], str)
    assert (output_dir / "trajectory.json").is_file()
    assert (output_dir / "metrics.json").is_file()
    persisted_metrics = json.loads(
        (output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert persisted_metrics["trace_digest"] == metrics["trace_digest"]


def test_runner_module_exposes_its_command_line_entrypoint() -> None:
    environment = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "underwater_tracking.verification.uuv_tracking_coverage_runner",
            "--help",
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0
    assert "multi-UUV tracking/coverage audit" in result.stdout
