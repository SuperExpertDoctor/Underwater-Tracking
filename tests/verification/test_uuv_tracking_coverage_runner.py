from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from hashlib import sha256
import json

import pytest

from underwater_tracking.agent.llm import LLMError
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.verification.uuv_tracking_coverage_runner import (
    NoNetworkLLM,
    REMEDIATION_METRIC_NAMES,
    _repository_event_sequence,
    _deterministic_trace_digest,
    _trace_event_sequence,
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


def _rehash_operational_frame(trace_frame: dict[str, object]) -> None:
    operational_frame = trace_frame["operational_frame"]
    trace_frame["execution"] = operational_frame["execution"]
    trace_frame["transport_hash"] = sha256(
        json.dumps(
            operational_frame,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


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


@pytest.mark.parametrize(
    ("metric_name", "failure_kind"),
    (
        ("stale_llm_result_count", "stale_llm_result"),
        ("llm_failure_physics_gap_count", "physics_gap"),
        ("assistant_answer_revision_mismatch_count", "assistant_mismatch"),
        ("memory_episode_source_gap_count", "memory_gap"),
    ),
)
def test_trace_summary_fails_when_remediation_closure_metric_is_nonzero(
    metric_name: str,
    failure_kind: str,
) -> None:
    trace = _minimal_trace(_complete_physics_audit())
    if failure_kind == "stale_llm_result":
        trace["event_sequence"] = [
            {
                "event_id": "stale-llm-result",
                "event_type": "planning_epoch_failed",
                "sim_time_s": 10,
                "payload": {"failure_category": "stale"},
            }
        ]
    elif failure_kind == "physics_gap":
        trace["frames"][0]["agent_telemetry"] = {
            "llm_failure_count": 1,
            "physics_time_advanced": False,
        }
    elif failure_kind == "assistant_mismatch":
        trace["assistant_cases"] = [
            {
                "request": {"frame_id": 1, "execution_revision": 4},
                "response": {"frame_id": 1, "execution_revision": 5},
            }
        ]
    else:
        trace["memory_episodes"] = [
            {"episode_id": "episode-1", "source_event_ids": ["missing-event"]}
        ]

    summary = summarize_trace(trace)

    assert summary[metric_name] == 1
    assert summary["hard_checks"]["remediation_closure"] is False
    assert summary["status"] == "FAIL"


def test_run_audit_fails_when_second_repeat_has_remediation_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean_trace = _minimal_trace(_complete_physics_audit())
    failing_trace = _minimal_trace(_complete_physics_audit())
    failing_trace["memory_episodes"] = [
        {"episode_id": "episode-1", "source_event_ids": ["missing-event"]}
    ]
    traces = iter((clean_trace, failing_trace))

    def fake_run_once(*, work_dir: Path, **_: object) -> dict[str, object]:
        work_dir.mkdir(parents=True, exist_ok=False)
        return next(traces)

    monkeypatch.setattr(
        "underwater_tracking.verification.uuv_tracking_coverage_runner.run_once",
        fake_run_once,
    )

    metrics = run_audit(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=2,
        repeat=2,
        work_dir=tmp_path / "work",
        evidence_dir=tmp_path / "evidence",
    )

    assert metrics["memory_episode_source_gap_count"] == 0
    assert metrics["hard_checks"]["repeat_remediation_closure"] is False
    assert metrics["status"] == "FAIL"


def test_repository_event_sequence_preserves_same_timestamp_append_order(
    tmp_path: Path,
) -> None:
    repository = EventRepository(tmp_path / "events.db")
    try:
        for event_id, event_type in (
            ("committed-first", "execution_refresh_committed"),
            ("attempted-second", "execution_refresh_attempted"),
            ("due-third", "execution_refresh_due"),
        ):
            repository.append(
                event_id=event_id,
                event_type=event_type,
                scenario_id="S1",
                sim_time_s=10,
                payload={},
            )

        sequence = _repository_event_sequence(repository, "S1")
    finally:
        repository.close()

    assert [event["event_id"] for event in sequence] == [
        "committed-first",
        "attempted-second",
        "due-third",
    ]


def test_trace_event_sequence_preserves_causal_input_order() -> None:
    events = [
        {"event_id": "committed-first", "event_type": "execution_refresh_committed", "sim_time_s": 10},
        {"event_id": "attempted-second", "event_type": "execution_refresh_attempted", "sim_time_s": 10},
    ]

    sequence = _trace_event_sequence({"event_sequence": events})

    assert [event["event_id"] for event in sequence] == [
        "committed-first",
        "attempted-second",
    ]


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


def test_audit_digest_preserves_llm_failure_presence() -> None:
    first = {
        "frames": [
            {
                "agent_telemetry": {
                    "llm_failure_count": 0,
                    "physics_time_advanced": True,
                }
            }
        ]
    }
    second = {
        "frames": [
            {
                "agent_telemetry": {
                    "llm_failure_count": 1,
                    "physics_time_advanced": True,
                }
            }
        ]
    }

    assert _deterministic_trace_digest(first) != _deterministic_trace_digest(second)


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
        steps=120,
        work_dir=tmp_path / "run",
    )

    summary = summarize_trace(result)
    runtime = summary["runtime_execution"]
    assert summary["status"] == "PASS", summary
    assert runtime["available"] is True
    assert runtime["valid"] is True
    assert summary["region_side_m"] == 2_000.0
    assert summary["target_detection_radius_m"] == 1_000.0
    assert summary["uuv_detection_radius_m"] == 600.0
    assert summary["task_group_size"] == 3
    assert summary["max_coverage_gap_area_m2"] == pytest.approx(0.0)
    assert summary["active_ping_count_during_passive"] == 0
    assert summary["tracking_owner_gap_frames"] == 0
    assert summary["max_visible_uuv_count"] == 24
    assert summary["runtime_scan_regions_with_pings"] == 4
    assert summary["max_runtime_scan_coverage"] > 0.0
    assert summary["runtime_entry_confirmation_count"] == 2
    assert summary["hard_checks"]["runtime_execution_contract"] is True
    assert summary["execution_revision_monotonic"] is True
    assert summary["execution_revision_advanced"] is True
    assert summary["max_runtime_group_count"] == 8
    assert summary["runtime_replacement_pair_count"] == 4
    assert summary["active_ping_cadence_valid"] is True
    assert summary["active_ping_count"] > summary["active_echo_count"] > 0
    assert summary["active_echo_source_valid"] is True
    assert summary["entry_evidence_public"] is True
    assert summary["tracking_owner_observed"] is True
    assert summary["max_tracking_owner_count"] == 1
    assert summary["canonical_transport_valid"] is True
    for check_name in (
        "c01_revision_replacement_contract",
        "c02_public_fusion_tracking_owner",
        "c03_physical_scan_coverage",
        "c04_atomic_handoff_continuity",
        "c05_resource_bounded_replacement",
        "c06_canonical_operational_frame",
    ):
        assert summary["hard_checks"][check_name] is True
    assert max(
        frame["execution"]["execution_revision"]
        for frame in result["frames"]
    ) > 1
    assert any(
        frame["execution"]["tracking_control"]["tracking_owner_group_id"]
        for frame in result["frames"]
    )

    last_trace_frame = result["frames"][-1]
    last_operational_frame = last_trace_frame["operational_frame"]
    last_operational_frame["plan_version"] = 1
    last_execution = last_operational_frame["execution"]
    last_execution["execution_revision"] = 1
    for region in last_execution["regions"]:
        region["execution_revision"] = 1
    last_trace_frame["transport_hash"] = sha256(
        json.dumps(
            last_operational_frame,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    tampered_summary = summarize_trace(result)

    assert tampered_summary["execution_revision_monotonic"] is False
    assert tampered_summary["hard_checks"][
        "c01_revision_replacement_contract"
    ] is False
    assert any(
        "execution_revision_nonmonotonic" in violation
        for violation in tampered_summary["runtime_execution"]["violations"]
    )


def test_runtime_audit_rejects_forged_scan_evidence(tmp_path: Path) -> None:
    result = run_once(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=2,
        work_dir=tmp_path / "run",
    )
    trace_frame = result["frames"][-1]
    operational_frame = trace_frame["operational_frame"]
    region = operational_frame["execution"]["regions"][0]
    region["ping_count"] = 1
    region["scan_evidence_ids"] = ["active_ping:forged"]
    trace_frame["transport_hash"] = sha256(
        json.dumps(
            operational_frame,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    summary = summarize_trace(result)

    runtime = summary["runtime_execution"]
    assert runtime["valid"] is False
    assert any(
        "runtime_scan_evidence_unresolved" in violation
        for violation in runtime["violations"]
    )


def test_runtime_audit_allows_scan_progress_reset_for_new_round(
    tmp_path: Path,
) -> None:
    result = run_once(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=2,
        work_dir=tmp_path / "run",
    )
    first_frame, second_frame = result["frames"]
    first_region = first_frame["operational_frame"]["execution"]["regions"][0]
    second_region = second_frame["operational_frame"]["execution"]["regions"][0]
    second_region.update(
        {
            "scan_round": first_region["scan_round"] + 1,
            "coverage": 0.0,
            "route_progress": 0.0,
            "ping_count": 0,
            "scan_completed": False,
            "scan_evidence_ids": [],
        }
    )
    _rehash_operational_frame(second_frame)

    runtime = summarize_trace(result)["runtime_execution"]

    assert not any(
        "runtime_scan_state_nonmonotonic" in violation
        for violation in runtime["violations"]
    )


def test_runtime_audit_rejects_scan_progress_drop_within_round(
    tmp_path: Path,
) -> None:
    result = run_once(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=2,
        work_dir=tmp_path / "run",
    )
    first_frame, second_frame = result["frames"]
    first_region = first_frame["operational_frame"]["execution"]["regions"][0]
    second_region = second_frame["operational_frame"]["execution"]["regions"][0]
    first_region.update({"coverage": 0.1, "scan_completed": False})
    second_region.update(
        {
            "scan_round": first_region["scan_round"],
            "coverage": 0.0,
            "scan_completed": False,
        }
    )
    _rehash_operational_frame(first_frame)
    _rehash_operational_frame(second_frame)

    runtime = summarize_trace(result)["runtime_execution"]

    assert runtime["valid"] is False
    assert any(
        "runtime_scan_state_nonmonotonic" in violation
        for violation in runtime["violations"]
    )


def test_runtime_audit_rejects_route_progress_drop_within_round(
    tmp_path: Path,
) -> None:
    result = run_once(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=2,
        work_dir=tmp_path / "run",
    )
    first_frame, second_frame = result["frames"]
    first_region = first_frame["operational_frame"]["execution"]["regions"][0]
    second_region = second_frame["operational_frame"]["execution"]["regions"][0]
    first_region["route_progress"] = 0.5
    second_region.update(
        {
            "scan_round": first_region["scan_round"],
            "route_progress": 0.25,
        }
    )
    _rehash_operational_frame(first_frame)
    _rehash_operational_frame(second_frame)

    runtime = summarize_trace(result)["runtime_execution"]

    assert any(
        "runtime_scan_state_nonmonotonic" in violation
        for violation in runtime["violations"]
    )


def test_runtime_audit_allows_ping_count_larger_than_bounded_evidence_window(
    tmp_path: Path,
) -> None:
    result = run_once(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=2,
        work_dir=tmp_path / "run",
    )
    trace_frame = result["frames"][-1]
    region = trace_frame["operational_frame"]["execution"]["regions"][0]
    region["ping_count"] = len(region["scan_evidence_ids"]) + 1
    _rehash_operational_frame(trace_frame)

    runtime = summarize_trace(result)["runtime_execution"]

    assert not any(
        "runtime_scan_ping_count_mismatch" in violation
        for violation in runtime["violations"]
    )


def test_runtime_audit_requires_physical_pings_in_all_four_regions(
    tmp_path: Path,
) -> None:
    result = run_once(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=2,
        work_dir=tmp_path / "run",
    )
    missing_region_id = "target_00:task:04"
    emitters: set[str] = set()
    for trace_frame in result["frames"]:
        execution = trace_frame["operational_frame"]["execution"]
        emitters.update(
            member_id
            for group in execution["task_groups"]
            if group["region_id"] == missing_region_id
            for member_id in group["member_uuv_ids"]
        )
    for trace_frame in result["frames"]:
        operational_frame = trace_frame["operational_frame"]
        region = next(
            item
            for item in operational_frame["execution"]["regions"]
            if item["region_id"] == missing_region_id
        )
        region["coverage"] = 0.0
        region["ping_count"] = 0
        region["scan_evidence_ids"] = []
        trace_frame["events"] = [
            event
            for event in trace_frame["events"]
            if event.get("event_type") != "active_ping"
            or event.get("payload", {}).get("emitter_id") not in emitters
        ]
        trace_frame["transport_hash"] = sha256(
            json.dumps(
                operational_frame,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

    summary = summarize_trace(result)

    runtime = summary["runtime_execution"]
    assert runtime["valid"] is False
    assert any(
        "runtime_active_scan_region_ping_missing" in violation
        and missing_region_id in violation
        for violation in runtime["violations"]
    ), runtime


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


def test_runner_keeps_live_group_routes_separated_after_refresh(tmp_path: Path) -> None:
    result = run_once(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=103,
        work_dir=tmp_path / "run",
    )
    frame = next(frame for frame in result["frames"] if frame["sim_time_s"] == 510)
    group = next(
        group
        for group in frame["operational_frame"]["execution"]["task_groups"]
        if group["region_id"] == "target_00:task:01"
        and group["lifecycle"] in {"active_scan", "passive_track"}
    )
    commands = frame["waypoint_commands"]["target_00"]
    group_commands = tuple(
        tuple(commands[member_id]) for member_id in group["member_uuv_ids"]
    )

    assert len(set(group_commands)) >= 2


def test_runner_deploys_replacement_members_on_independent_slot_routes(
    tmp_path: Path,
) -> None:
    result = run_once(
        config_path=Path("configs/scenario/uuv_only_single_target.yaml"),
        seed=42,
        steps=12,
        work_dir=tmp_path / "run",
    )
    frame = next(frame for frame in result["frames"] if frame["sim_time_s"] == 60)
    group = next(
        group
        for group in frame["operational_frame"]["execution"]["task_groups"]
        if group["region_id"] == "target_00:task:01"
        and group["lifecycle"] == "entering"
    )
    positions = {
        uuv["platform_id"]: tuple(uuv["position_xy"])
        for uuv in frame["uuvs"]
        if uuv["platform_id"] in group["member_uuv_ids"]
    }

    assert tuple(positions) == tuple(group["member_uuv_ids"])
    assert len(set(positions.values())) == len(group["member_uuv_ids"])


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
    assert summary["status"] == "PASS", summary


def test_trace_summary_does_not_attribute_one_target_ping_to_every_region() -> None:
    trace = _minimal_trace(_complete_physics_audit())
    trace["routes"]["R2"] = {"uuv_01": [[10.0, 0.0], [11.0, 0.0]]}
    trace["regions"]["R2"] = {
        "target_id": "target_00",
        "polygon": [[9.0, -1.0], [12.0, -1.0], [12.0, 1.0], [9.0, 1.0]],
    }
    trace["active_ranges_m"]["uuv_01"] = 100.0
    trace["frames"][0]["region_assignments"] = {
        "R1": {
            "active_scan_uuv_ids": ["uuv_00"],
            "passive_track_uuv_ids": [],
        },
        "R2": {
            "active_scan_uuv_ids": ["uuv_01"],
            "passive_track_uuv_ids": [],
        },
    }
    trace["frames"][0]["events"] = [
        {
            "event_type": "active_ping",
            "entity_id": "target_00",
            "payload": {
                "emitter_id": "uuv_00",
                "source_position_xy": [0.0, 0.0],
                "configured_range_m": 100.0,
            },
        }
    ]

    summary = summarize_trace(trace)

    assert summary["coverage"]["R1"]["active_emission_count"] == 1
    assert summary["coverage"]["R2"]["active_emission_count"] == 0
    assert (
        summary["coverage"]["R2"]["sampled_active_sonar_footprint_fraction"]
        is None
    )


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


def test_physics_gate_allows_lifecycle_bounded_gaps_for_dynamic_entities() -> None:
    physics = _complete_physics_audit()
    physics["entity_count"] = 2
    physics["audits"].append(
        {
            "entity_id": "runtime-group:member:01",
            "limit_violation_count": 0,
            "teleport_count": 0,
            "boundary_violation_count": 0,
        }
    )
    coverage = physics["coverage"]
    assert isinstance(coverage, dict)
    coverage.update(
        {
            "expected_entity_ids": ("runtime-group:member:01", "uuv_00"),
            "expected_entity_count": 2,
            "observed_entity_ids": ("runtime-group:member:01", "uuv_00"),
            "observed_entity_count": 2,
            "missing_entity_frame_ids": {"runtime-group:member:01": (0, 1)},
        }
    )

    valid = summarize_trace(_minimal_trace(physics))
    assert valid["hard_checks"]["configured_physics_invariants"] is True

    coverage["missing_entity_frame_ids"] = {
        "runtime-group:member:01": (1,)
    }
    invalid = summarize_trace(_minimal_trace(physics))
    assert invalid["hard_checks"]["configured_physics_invariants"] is False


def test_physics_gate_allows_dynamic_entity_to_disappear_after_its_lifecycle() -> None:
    physics = _complete_physics_audit()
    physics["entity_count"] = 2
    physics["audits"].append(
        {
            "entity_id": "runtime-group:member:01",
            "limit_violation_count": 0,
            "teleport_count": 0,
            "boundary_violation_count": 0,
        }
    )
    coverage = physics["coverage"]
    assert isinstance(coverage, dict)
    coverage.update(
        {
            "expected_entity_ids": ("runtime-group:member:01", "uuv_00"),
            "expected_entity_count": 2,
            "observed_entity_ids": ("runtime-group:member:01", "uuv_00"),
            "observed_entity_count": 2,
            "missing_entity_frame_ids": {"runtime-group:member:01": (0, 2)},
        }
    )

    valid = summarize_trace(_minimal_trace(physics))

    assert valid["hard_checks"]["configured_physics_invariants"] is True


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
