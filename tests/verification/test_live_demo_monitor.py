from __future__ import annotations

import importlib.util
from copy import deepcopy
import json
from http.client import BadStatusLine
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from underwater_tracking.verification import live_demo
from underwater_tracking.verification.physics_invariants import (
    EntityMotionAudit,
    EntityMotionLimits,
    FullBattleAcceptance,
    PhysicsInvariantMonitor,
)


_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "monitor_main_battle.py"
_SPEC = importlib.util.spec_from_file_location("monitor_main_battle", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MONITOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MONITOR)


def _limits() -> EntityMotionLimits:
    return EntityMotionLimits(
        max_speed_mps=4.0,
        max_acceleration_mps2=1.0,
        max_deceleration_mps2=1.0,
        max_turn_rate_rad_s=0.5,
    )


def _entity(entity_id: str, x: float) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "entity_kind": "uuv",
        "position_xy": (x, 0.0),
        "speed_mps": 1.0,
        "heading_rad": 0.0,
    }


def _valid_uuv_execution_frame(*, frame_id: int = 10, execution_revision: int = 3) -> dict[str, object]:
    target_id = "target_00"
    region_ids = tuple(f"{target_id}:task:{index:02d}" for index in range(1, 5))
    groups = tuple(
        {
            "task_group_id": f"TG-{index:02d}",
            "target_id": target_id,
            "region_id": region_id,
            "execution_revision": execution_revision,
            "member_uuv_ids": [f"uuv_{2 * (index - 1):02d}", f"uuv_{2 * (index - 1) + 1:02d}"],
            "active_verifier_uuv_id": f"uuv_{2 * (index - 1):02d}",
            "passive_tracker_uuv_id": f"uuv_{2 * (index - 1) + 1:02d}",
            "status": "active",
            "evidence_ids": [f"evidence:group:{index}"],
        }
        for index, region_id in enumerate(region_ids, start=1)
    )
    regions = tuple(
        {
            "region_id": region_id,
            "target_id": target_id,
            "slot_index": index,
            "execution_revision": execution_revision,
            "prediction_id": "prediction:3",
            "geometry": [[index, 0], [index + 1, 0], [index + 1, 1]],
            "start_s": float((index - 1) * 450),
            "end_s": float(index * 450),
            "geometry_revision": execution_revision,
            "predecessor_region_id": region_ids[index - 2] if index > 1 else None,
            "successor_region_id": region_ids[index] if index < 4 else None,
            "status": "active",
            "task_group_id": f"TG-{index:02d}",
            "evidence_ids": [f"evidence:region:{index}"],
        }
        for index, region_id in enumerate(region_ids, start=1)
    )
    execution_members = {uuv_id for group in groups for uuv_id in group["member_uuv_ids"]}
    uuvs = [
        {
            "uuv_id": f"uuv_{index:02d}",
            "physically_exposed": f"uuv_{index:02d}" in execution_members,
            "sensor_mode": (
                "active"
                if f"uuv_{index:02d}" in execution_members and index % 2 == 0
                else "passive"
            ),
            "group_id": target_id if f"uuv_{index:02d}" in execution_members else None,
            "tracked_target_id": target_id if f"uuv_{index:02d}" in execution_members else None,
        }
        for index in range(12)
    ]
    return {
        "frame_id": frame_id,
        "sim_time_s": frame_id * 5,
        "scenario_id": "uuv-only-single-target",
        "plan_version": execution_revision,
        "run_phase": "running",
        "uuv_only": True,
        "execution": {
            "target_id": target_id,
            "execution_revision": execution_revision,
            "source_snapshot_revision": frame_id,
            "prediction_revision": 3,
            "prediction_id": "prediction:3",
            "intent_revision": 3,
            "current_region_id": region_ids[0],
            "next_region_id": region_ids[1],
            "evidence_ids": ["evidence:execution:3"],
            "regions": list(regions),
            "task_groups": list(groups),
            "reserve_uuv_ids": [f"uuv_{index:02d}" for index in range(8, 12)],
        },
        "uuvs": uuvs,
        "events": [],
    }


def _strict_live_checkpoint_frame() -> dict[str, object]:
    frame = _valid_uuv_execution_frame()
    target_id = "target_00"
    prediction_id = "prediction:3"
    prediction = {
        "prediction_id": prediction_id,
        "prediction_revision": 3,
        "origin_sim_time_s": 50,
        "health": {
            "status": "valid",
            "regime": "imm",
            "reason_codes": [],
            "source_track_age_s": 0,
            "clipped_point_fraction": 0,
            "maximum_radius_m": 3,
            "raw_prediction_id": prediction_id,
        },
        "horizon_s": 900,
        "sample_step_s": 300,
        "centerline_xy": [{"x": 2, "y": 2}, {"x": 4, "y": 4}],
        "radius_m": [2, 3],
        "point_confidence": [0.9, 0.8],
    }
    frame["map_bounds"] = {"min_x": 0, "min_y": 0, "max_x": 20, "max_y": 20}
    frame["target_estimates"] = [
        {
            "target_id": target_id,
            "mean": {"x": 2, "y": 2},
            "prediction": prediction,
            "detection_range_m": 500,
        }
    ]
    execution = frame["execution"]
    assert isinstance(execution, dict)
    execution.update(
        {
            "prediction_id": prediction_id,
            "prediction_revision": 3,
            "data_age_s": 0,
            "valid_from_s": 50,
            "valid_until_s": 950,
            "health_status": "current",
        }
    )
    frame["sim_time_s"] = 100
    return frame


def test_strict_live_checkpoint_validator_accepts_a_bounded_usable_frame() -> None:
    frame = _strict_live_checkpoint_frame()

    violations = live_demo.validate_live_checkpoint_frame(
        frame,
        prediction_radius_cap_m=5,
        execution_max_age_s=900,
    )

    assert violations == ()


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            lambda frame: frame["target_estimates"][0]["prediction"]["centerline_xy"].__setitem__(
                1, {"x": 21, "y": 4}
            ),
            "prediction_point_out_of_map",
        ),
        (
            lambda frame: frame["target_estimates"][0]["prediction"]["radius_m"].__setitem__(
                1, 6
            ),
            "prediction_radius_cap_exceeded",
        ),
        (
            lambda frame: frame["execution"].update({"health_status": "failed"}),
            "execution_health_unusable",
        ),
        (
            lambda frame: frame["execution"].update({"data_age_s": 901}),
            "execution_age_exceeded",
        ),
    ],
)
def test_strict_live_checkpoint_validator_rejects_unsafe_semantics(
    change, expected: str
) -> None:
    frame = _strict_live_checkpoint_frame()
    change(frame)

    violations = live_demo.validate_live_checkpoint_frame(
        frame,
        prediction_radius_cap_m=5,
        execution_max_age_s=900,
    )

    assert expected in violations


def test_transport_hash_validator_rejects_a_payload_mismatch() -> None:
    frame = _strict_live_checkpoint_frame()
    websocket_frame = deepcopy(frame)
    websocket_frame["frame_id"] = 11

    hashes, violations = live_demo.validate_transport_payload_hashes(
        {"http": frame, "websocket": websocket_frame, "jsonl": frame}
    )

    assert hashes["http"] != hashes["websocket"]
    assert "transport_payload_hash_mismatch:websocket" in violations


def test_execution_regions_must_link_one_to_one_to_task_groups() -> None:
    frame = _strict_live_checkpoint_frame()
    execution = frame["execution"]
    assert isinstance(execution, dict)
    regions = execution["regions"]
    assert isinstance(regions, list)
    regions[0]["task_group_id"] = regions[1]["task_group_id"]

    violations = live_demo.validate_uuv_only_frame(frame)

    assert "execution_region_task_group_mismatch" in violations


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            lambda frame: frame["execution"]["regions"][0].update(
                {"prediction_id": "prediction:other"}
            ),
            "execution_region_prediction_mismatch",
        ),
        (
            lambda frame: frame["execution"]["regions"][0].update(
                {"target_id": "target_other"}
            ),
            "execution_region_target_mismatch",
        ),
        (
            lambda frame: frame["execution"].update(
                {"prediction_id": "prediction:other"}
            ),
            "execution_region_prediction_mismatch",
        ),
    ],
)
def test_execution_region_prediction_and_target_pairing_is_strict(
    change, expected: str
) -> None:
    frame = _valid_uuv_execution_frame()
    change(frame)

    violations = live_demo.validate_uuv_only_frame(frame)

    assert expected in violations


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            lambda frame: frame["execution"]["regions"].__setitem__(0, "malformed"),
            "execution_region_shape_invalid",
        ),
        (
            lambda frame: frame["execution"]["task_groups"].__setitem__(0, "malformed"),
            "execution_task_group_shape_invalid",
        ),
        (
            lambda frame: frame["uuvs"].__setitem__(0, "malformed"),
            "uuv_inventory_shape_invalid",
        ),
        (
            lambda frame: frame["execution"]["task_groups"][0].pop("task_group_id"),
            "execution_task_group_id_invalid",
        ),
        (
            lambda frame: frame["execution"]["task_groups"][0].pop("active_verifier_uuv_id"),
            "execution_task_group_active_role_invalid",
        ),
        (
            lambda frame: frame["execution"]["task_groups"][0].pop("passive_tracker_uuv_id"),
            "execution_task_group_passive_role_invalid",
        ),
        (
            lambda frame: frame["uuvs"][0].pop("physically_exposed"),
            "execution_member_physical_exposure_invalid",
        ),
        (
            lambda frame: frame["uuvs"][0].pop("sensor_mode"),
            "execution_member_sensor_mode_invalid",
        ),
        (
            lambda frame: frame["uuvs"][0].update({"sensor_mode": "passive"}),
            "execution_member_sensor_role_mismatch",
        ),
        (
            lambda frame: frame["uuvs"][0].update({"tracked_target_id": "target_other"}),
            "execution_member_target_mismatch",
        ),
    ],
)
def test_execution_contract_rejects_malformed_or_semantically_unbound_entries(
    change, expected: str
) -> None:
    frame = _valid_uuv_execution_frame()
    change(frame)

    violations = live_demo.validate_uuv_only_frame(frame)

    assert expected in violations


def test_execution_region_ids_must_be_unique() -> None:
    frame = _valid_uuv_execution_frame()
    regions = frame["execution"]["regions"]
    regions[1]["region_id"] = regions[0]["region_id"]

    violations = live_demo.validate_uuv_only_frame(frame)

    assert "execution_region_id_duplicate" in violations


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            lambda frame: frame["target_estimates"].append(
                deepcopy(frame["target_estimates"][0])
            ),
            "target_estimate_count_mismatch",
        ),
        (
            lambda frame: frame["target_estimates"][0].update({"prediction": None}),
            "execution_prediction_missing",
        ),
        (
            lambda frame: frame["execution"].pop("prediction_id"),
            "execution_prediction_id_invalid",
        ),
        (
            lambda frame: frame["execution"].pop("prediction_revision"),
            "execution_prediction_revision_invalid",
        ),
        (
            lambda frame: frame["execution"].update({"prediction_revision": 4}),
            "prediction_execution_revision_mismatch",
        ),
    ],
)
def test_checkpoint_requires_one_target_and_execution_prediction_pair(
    change, expected: str
) -> None:
    frame = _strict_live_checkpoint_frame()
    change(frame)

    violations = live_demo.validate_live_checkpoint_frame(
        frame,
        prediction_radius_cap_m=5,
        execution_max_age_s=900,
    )

    assert expected in violations


def test_database_execution_evidence_must_match_the_published_frame(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE execution_revisions ("
            "commit_id TEXT, scenario_id TEXT, execution_revision INTEGER, "
            "candidate_execution_revision INTEGER, base_execution_revision INTEGER, "
            "status TEXT, source_snapshot_revision INTEGER, "
            "active_plan_preserved INTEGER, reason TEXT, snapshot_payload TEXT, "
            "result_payload TEXT, created_at INTEGER)"
        )
        connection.execute(
            "INSERT INTO execution_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "commit-2",
                "uuv-only-single-target",
                2,
                None,
                1,
                "committed",
                10,
                0,
                "",
                json.dumps(
                    {
                        "execution_revision": 2,
                        "prediction_revision": 3,
                        "source_snapshot_revision": 10,
                        "source_sim_time_s": 100,
                        "valid_from_s": 50,
                        "valid_until_s": 950,
                    }
                ),
                json.dumps(
                    {
                        "execution_revision": 2,
                        "prediction_revision": 3,
                        "source_snapshot_revision": 10,
                        "source_sim_time_s": 100,
                        "valid_from_s": 50,
                        "valid_until_s": 950,
                    }
                ),
                2,
            ),
        )

    evidence = live_demo.read_latest_execution_database_evidence(
        database,
        "uuv-only-single-target",
        execution_revision=2,
        source_snapshot_revision=10,
        frame_sim_time_s=100,
    )
    violations = live_demo.validate_database_execution_consistency(
        _strict_live_checkpoint_frame(), evidence
    )

    assert evidence["execution_revision"] == 2
    assert "database_execution_revision_mismatch" in violations

    bound_evidence = {
        **evidence,
        "execution_revision": 3,
        "prediction_revision": 3,
        "valid_from_s": 50,
        "valid_until_s": 950,
        "source_snapshot_revision": 11,
    }
    assert "database_source_snapshot_revision_mismatch" in live_demo.validate_database_execution_consistency(
        _strict_live_checkpoint_frame(), bound_evidence
    )


def test_database_evidence_is_bound_to_the_selected_checkpoint_not_latest_row(
    tmp_path: Path,
) -> None:
    database = tmp_path / "agent.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE execution_revisions ("
            "commit_id TEXT, scenario_id TEXT, execution_revision INTEGER, "
            "candidate_execution_revision INTEGER, base_execution_revision INTEGER, "
            "status TEXT, source_snapshot_revision INTEGER, "
            "active_plan_preserved INTEGER, reason TEXT, snapshot_payload TEXT, "
            "result_payload TEXT, created_at INTEGER)"
        )
        for revision, source_revision, source_time, created_at in (
            (3, 10, 100, 3),
            (4, 11, 200, 4),
        ):
            payload = {
                "execution_revision": revision,
                "prediction_revision": 3,
                "source_snapshot_revision": source_revision,
                "source_sim_time_s": source_time,
                "valid_from_s": source_time - 50,
                "valid_until_s": source_time + 850,
            }
            connection.execute(
                "INSERT INTO execution_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"commit-{revision}",
                    "uuv-only-single-target",
                    revision,
                    None,
                    revision - 1,
                    "committed",
                    source_revision,
                    0,
                    "",
                    json.dumps(payload),
                    json.dumps(payload),
                    created_at,
                ),
            )

    evidence = live_demo.read_latest_execution_database_evidence(
        database,
        "uuv-only-single-target",
        execution_revision=3,
        source_snapshot_revision=10,
        frame_sim_time_s=100,
    )

    assert evidence["execution_revision"] == 3
    assert evidence["source_snapshot_revision"] == 10
    assert evidence["source_sim_time_s"] == 100
    assert evidence["checkpoint_binding_valid"] is True
    assert live_demo.validate_database_execution_consistency(
        _strict_live_checkpoint_frame(), evidence
    ) == ()

    expired_evidence = live_demo.read_latest_execution_database_evidence(
        database,
        "uuv-only-single-target",
        execution_revision=3,
        source_snapshot_revision=10,
        frame_sim_time_s=1_000,
    )

    assert expired_evidence["execution_revision"] == 3
    assert expired_evidence["checkpoint_binding_valid"] is False
    assert "database_checkpoint_binding_mismatch" in live_demo.validate_database_execution_consistency(
        _strict_live_checkpoint_frame(), expired_evidence
    )


def _valid_uuv_tracking_evidence() -> dict[str, object]:
    return {
        "events": [
            {
                "event_id": "entry-1",
                "event_type": "uuv_boundary_entry_started",
                "entity_id": "uuv_00",
                "sim_time_s": 10,
                "region_id": "target_00:task:01",
                "uuv_id": "uuv_00",
            },
            {
                "event_id": "ping-1",
                "event_type": "active_ping",
                "entity_id": "target_00",
                "sim_time_s": 20,
                "uuv_ids": ("uuv_00", "uuv_01"),
            },
            {
                "event_id": "detection-1",
                "event_type": "target_detection_acquired",
                "entity_id": "target_00",
                "sim_time_s": 20,
                "platform_id": "uuv_00",
            },
            {
                "event_id": "estimate-1",
                "event_type": "target_maneuver_observed",
                "entity_id": "target_00",
                "sim_time_s": 30,
                "source_observation_ids": ("obs-1",),
            },
            {
                "event_id": "handoff-1",
                "event_type": "handoff_completed",
                "entity_id": "target_00:task:01",
                "sim_time_s": 40,
                "target_id": "target_00",
                "predecessor_region_id": "target_00:task:01",
                "successor_region_id": "target_00:task:02",
                "predecessor_uuv_ids": ("uuv_00", "uuv_01"),
                "successor_uuv_ids": ("uuv_02", "uuv_03"),
                "plan_revision": 3,
            },
            {
                "event_id": "exit-1",
                "event_type": "uuv_boundary_exit_started",
                "entity_id": "uuv_00",
                "sim_time_s": 40,
                "region_id": "target_00:task:01",
            },
            {
                "event_id": "exited-1",
                "event_type": "uuv_boundary_exited",
                "entity_id": "uuv_00",
                "sim_time_s": 45,
                "region_id": "target_00:task:01",
            },
            {
                "event_id": "replacement-1",
                "event_type": "uuv_boundary_replacement",
                "entity_id": "uuv_02",
                "sim_time_s": 45,
                "region_id": "target_00:task:01",
                "outgoing_uuv_id": "uuv_00",
                "replacement_uuv_id": "uuv_02",
            },
            {
                "event_id": "blue-response-1",
                "event_type": "state_changed",
                "entity_id": "target_00",
                "sim_time_s": 50,
                "phase": "blue_response",
                "plan_version": 3,
            },
        ]
    }


def test_physics_coverage_rejects_missing_and_duplicate_entity_frames() -> None:
    monitor = PhysicsInvariantMonitor(
        {"uuv_00": _limits(), "uuv_01": _limits()}
    )
    monitor.observe(
        {"frame_id": 0, "sim_time_s": 0, "entities": [_entity("uuv_00", 0), _entity("uuv_01", 0)]}
    )
    monitor.observe(
        {
            "frame_id": 1,
            "sim_time_s": 5,
            "entities": [_entity("uuv_00", 5), _entity("uuv_00", 5)],
        }
    )

    coverage = monitor.coverage(physics_step_s=5)

    assert coverage["observed_entity_ids"] == ("uuv_00", "uuv_01")
    assert coverage["duplicate_entity_frame_ids"] == ("uuv_00@1",)
    assert coverage["missing_entity_frame_ids"] == {"uuv_01": (1,)}


def _valid_evidence() -> dict[str, object]:
    return {
        "blue_epoch_id": "epoch:S1:1:a1",
        "blue_plan_version": 3,
        "public_observations": [
            {"observation_id": "obs-1", "target_id": "target_00", "sim_time_s": 30}
        ],
        "events": [
            {
                "event_id": "det-1",
                "event_type": "target_detection_acquired",
                "entity_id": "target_00",
                "sim_time_s": 10,
            },
            {
                "event_id": "target_mission_decision:target_00:dec-1",
                "event_type": "target_mission_decision",
                "entity_id": "target_00",
                "sim_time_s": 20,
            },
            {
                "event_id": "estimate-1",
                "event_type": "target_maneuver_observed",
                "entity_id": "target_00",
                "sim_time_s": 30,
                "source_observation_ids": ("obs-1",),
            },
            {
                "event_id": "chain-1:motion_effect",
                "event_type": "state_changed",
                "entity_id": "target_00",
                "sim_time_s": 21,
                "phase": "adversary_motion_effect",
                "chain_id": "chain-1",
                "decision_id": "dec-1",
                "speed_delta_mps": 1.0,
                "heading_delta_rad": 0.1,
                "depth_delta_m": 0.0,
            },
            {
                "event_id": "chain-1:blue_response",
                "event_type": "state_changed",
                "entity_id": "target_00",
                "sim_time_s": 40,
                "phase": "blue_response",
                "plan_version": 3,
                "decision_id": "dec-1",
            },
        ],
        "adversary_decisions": [
            {
                "target_id": "target_00",
                "decision_id": "dec-1",
                "sim_time_s": 20,
                "decision_event_id": "target_mission_decision:target_00:dec-1",
                "trigger_event_ids": ("det-1",),
                "provider_call_id": "LLM-ADV-1",
            }
        ],
        "llm_calls": [
            {
                "call_id": "LLM-ADV-1",
                "operation": "adversary_mission_decision",
                "model": "LongCat-Flash-Chat",
                "prompt_version": "adversary-mission-v1",
                "request_hash": "adversary-request-1",
                "response_hash": "adversary-response-1",
                "error_category": "",
                "sim_time_s": 20,
            }
        ],
        "blue_response_chains": [
            {
                "target_id": "target_00",
                "decision_id": "dec-1",
                "maneuver_time_s": 20,
                "response_event_id": "chain-1:blue_response",
                "motion_effect_event_id": "chain-1:motion_effect",
                "plan_version": 3,
                "blue_estimate_ids": ("estimate-1",),
                "public_observation_ids": ("obs-1",),
            }
        ],
    }


def _valid_blue_tracking_evidence() -> dict[str, object]:
    return {
        "events": [
            {
                "event_id": "dispatch-1",
                "event_type": "carrier_dispatch_completed",
                "entity_id": "carrier_01",
                "sim_time_s": 10,
                "candidate_id": "target_00:r1",
                "uuv_ids": ("uuv_00",),
            },
            {
                "event_id": "deploy-1",
                "event_type": "uuv_deployed",
                "entity_id": "uuv_00",
                "sim_time_s": 10,
                "candidate_id": "target_00:r1",
                "reason": "deploy:target_00:r1",
            },
            {
                "event_id": "ping-1",
                "event_type": "active_ping",
                "entity_id": "target_00",
                "sim_time_s": 20,
                "uuv_ids": ("uuv_00",),
            },
            {
                "event_id": "estimate-1",
                "event_type": "target_estimate_updated",
                "entity_id": "target_00",
                "sim_time_s": 30,
                "source_observation_ids": ("obs-1",),
            },
            {
                "event_id": "handoff-1",
                "event_type": "handoff_completed",
                "entity_id": "target_00:r1",
                "sim_time_s": 40,
                "target_id": "target_00",
                "predecessor_region_id": "target_00:r1",
                "predecessor_uuv_ids": ("uuv_00",),
                "plan_version": 3,
            },
            {
                "event_id": "resource-1",
                "event_type": "uuv_range_exhausted",
                "entity_id": "uuv_00",
                "sim_time_s": 50,
            },
            {
                "event_id": "recover-request-1",
                "event_type": "uuv_recovery_requested",
                "entity_id": "uuv_00",
                "sim_time_s": 60,
            },
            {
                "event_id": "recovered-1",
                "event_type": "uuv_recovered",
                "entity_id": "uuv_00",
                "sim_time_s": 70,
            },
            {
                "event_id": "return-1",
                "event_type": "carrier_returned_to_fleet",
                "entity_id": "carrier_01",
                "sim_time_s": 80,
                "sortie_uuv_ids": ("uuv_00",),
            },
        ]
    }


def test_blue_tracking_chain_binds_target_carrier_candidate_and_uuv() -> None:
    chains, violations = _MONITOR._blue_tracking_chains(
        _valid_blue_tracking_evidence()
    )

    assert len(chains) == 1
    assert violations == []
    assert chains[0].target_id == "target_00"
    assert chains[0].carrier_id == "carrier_01"
    assert chains[0].uuv_ids == ("uuv_00",)


def test_blue_tracking_chain_rejects_a_missing_handoff_link() -> None:
    evidence = deepcopy(_valid_blue_tracking_evidence())
    evidence["events"] = [
        event
        for event in evidence["events"]
        if event["event_type"] != "handoff_completed"
    ]

    chains, violations = _MONITOR._blue_tracking_chains(evidence)

    assert chains == []
    assert "missing_blue_tracking_evidence_chain" in violations


def test_uuv_only_tracking_chain_binds_boundary_handoff_and_replacement() -> None:
    chains, violations = _MONITOR._uuv_only_tracking_chains(
        _valid_uuv_tracking_evidence()
    )

    assert len(chains) == 1
    assert violations == []
    assert chains[0].target_id == "target_00"
    assert chains[0].region_id == "target_00:task:01"
    assert chains[0].uuv_ids == ("uuv_00", "uuv_01")
    assert chains[0].replacement_uuv_id == "uuv_02"


def test_uuv_only_tracking_chain_rejects_carrier_lifecycle() -> None:
    evidence = _valid_uuv_tracking_evidence()
    evidence["events"].append(
        {
            "event_id": "legacy-carrier-event",
            "event_type": "carrier_returned_to_fleet",
            "entity_id": "carrier_01",
            "sim_time_s": 60,
        }
    )

    chains, violations = _MONITOR._uuv_only_tracking_chains(evidence)

    assert chains == []
    assert "legacy_carrier_lifecycle_event" in violations


def test_uuv_only_frame_contract_and_transport_consistency_are_strict() -> None:
    frame = _valid_uuv_execution_frame()

    assert live_demo.validate_uuv_only_frame(frame) == ()
    assert live_demo.validate_transport_frame_consistency(
        {"http": frame, "websocket": frame, "replay": frame}
    ) == ()

    missing_execution = deepcopy(frame)
    missing_execution.pop("execution")
    assert "execution_snapshot_missing" in live_demo.validate_uuv_only_frame(
        missing_execution
    )

    bad_revision = deepcopy(frame)
    bad_revision["execution"]["task_groups"][0]["execution_revision"] = 4
    assert "execution_task_group_revision_mismatch" in live_demo.validate_uuv_only_frame(
        bad_revision
    )

    mismatched_channel = deepcopy(frame)
    mismatched_channel["frame_id"] = 11
    assert "transport_frame_id_mismatch:websocket" in live_demo.validate_transport_frame_consistency(
        {"http": frame, "websocket": mismatched_channel, "replay": frame}
    )


def test_frame_contract_rejects_non_monotonic_frame_id() -> None:
    frame = _valid_uuv_execution_frame(frame_id=10)

    violations = live_demo.validate_uuv_only_frame(frame, previous_frame_id=10)

    assert "frame_id_not_strictly_increasing" in violations


def test_evidence_chain_requires_matching_public_causal_records() -> None:
    valid = _MONITOR._evidence_chains(_valid_evidence())
    forged = _valid_evidence()
    forged["blue_response_chains"] = [
        {
            **forged["blue_response_chains"][0],
            "public_observation_ids": ("unrelated-observation",),
        }
    ]

    assert len(valid) == 1
    assert valid[0].adversary_provider_call_id == "LLM-ADV-1"
    assert _MONITOR._evidence_chains(forged) == []


def test_evidence_chain_requires_exact_successful_adversary_provider_call() -> None:
    missing_call = _valid_evidence()
    missing_call["llm_calls"] = []
    failed_call = _valid_evidence()
    failed_call["llm_calls"][0]["error_category"] = "provider"

    assert _MONITOR._evidence_chains(missing_call) == []
    assert _MONITOR._evidence_chains(failed_call) == []


def _valid_prediction_intent_evidence() -> dict[str, object]:
    return {
        "prediction_diffs": [
            {
                "diff_id": "D1",
                "target_id": "target_00",
                "previous_prediction_id": "P1",
                "current_prediction_id": "P2",
                "absolute_rms_m": 300.0,
                "normalized_rms": 3.0,
                "absolute_floor_m": 250.0,
                "normalized_threshold": 2.45,
                "exceeded": True,
                "overlap_start_s": 60.0,
                "overlap_end_s": 660.0,
            }
        ],
        "public_observations": [
            {
                "observation_id": "obs-intent-1",
                "target_id": "target_00",
                "observer_id": "uuv_00",
                "sim_time_s": 61,
            }
        ],
        "llm_calls": [
            {
                "call_id": "LLM-1",
                "operation": "intent",
                "model": "LongCat-Flash-Chat",
                "prompt_version": "intent-v2",
                "request_hash": "request-1",
                "response_hash": "response-1",
                "error_category": "",
                "sim_time_s": 61,
            },
            {
                "call_id": "LLM-2",
                "operation": "intent",
                "model": "LongCat-Flash-Chat",
                "prompt_version": "intent-v2",
                "request_hash": "request-2",
                "response_hash": "response-2",
                "error_category": "",
                "sim_time_s": 62,
            },
        ],
        "events": [
            {
                "event_id": "E-suspect",
                "event_type": "target_intent_change_suspected",
                "entity_id": "target_00",
                "sim_time_s": 60,
                "payload": {"diff_id": "D1"},
            },
            {
                "event_id": "E-confirmed",
                "event_type": "target_intent_changed",
                "entity_id": "target_00",
                "sim_time_s": 62,
                "payload": {
                    "diff_id": "D1",
                    "suspicion_event_id": "E-suspect",
                    "intent_llm_call_ids": ["LLM-1", "LLM-2"],
                    "intent_llm_calls": [
                        {
                            key: value
                            for key, value in call.items()
                            if key not in {"call_id", "error_category"}
                        }
                        for call in [
                            {
                                "call_id": "LLM-1",
                                "operation": "intent",
                                "model": "LongCat-Flash-Chat",
                                "prompt_version": "intent-v2",
                                "request_hash": "request-1",
                                "response_hash": "response-1",
                                "error_category": "",
                                "sim_time_s": 61,
                            },
                            {
                                "call_id": "LLM-2",
                                "operation": "intent",
                                "model": "LongCat-Flash-Chat",
                                "prompt_version": "intent-v2",
                                "request_hash": "request-2",
                                "response_hash": "response-2",
                                "error_category": "",
                                "sim_time_s": 62,
                            },
                        ]
                    ],
                    "source": "real_intent_llm",
                },
            },
            {
                "event_id": "mission-plan:3:target_00:applied",
                "event_type": "state_changed",
                "entity_id": "target_00",
                "sim_time_s": 70,
                "payload": {
                    "phase": "plan_applied",
                    "plan_revision": 3,
                    "region_id": "target_00:cell:1:1",
                    "member_ids": ["uuv_00", "uuv_01"],
                },
            },
            {
                "event_id": "E-estimate-after-suspicion",
                "event_type": "target_maneuver_observed",
                "entity_id": "target_00",
                "sim_time_s": 65,
                "payload": {},
                "source_observation_ids": ["obs-intent-1"],
            },
        ],
        "decisions": [
            {
                "decision_id": "decision-3",
                "sim_time_s": 63,
                "trigger_event_ids": ["E-confirmed"],
                "final_plan_id": "plan-3",
            }
        ],
        "committed_plans": [
            {
                "plan_id": "plan-3",
                "revision": 3,
                "status": "active",
                "target_ids": ["target_00"],
                "trigger_event_ids": ["E-confirmed"],
            }
        ],
    }


def test_prediction_intent_report_chain_requires_durable_ids() -> None:
    chains, violations = _MONITOR._prediction_intent_chains(
        _valid_prediction_intent_evidence()
    )

    assert violations == []
    assert len(chains) == 1
    chain = chains[0]
    assert chain.diff_id == "D1"
    assert chain.suspicion_event_id == "E-suspect"
    assert chain.intent_llm_call_ids == ("LLM-1", "LLM-2")
    assert chain.confirmed_event_id == "E-confirmed"
    assert chain.resulting_plan_revision == 3
    assert chain.blue_response_event_ids == (
        "mission-plan:3:target_00:applied",
    )


def test_prediction_intent_chain_rejects_unrelated_plan_application() -> None:
    evidence = deepcopy(_valid_prediction_intent_evidence())
    response = evidence["events"][2]
    response["event_id"] = "unrelated-plan-application"

    chains, violations = _MONITOR._prediction_intent_chains(evidence)

    assert chains == []
    assert "missing_blue_response" in violations


@pytest.mark.parametrize(
    ("field", "value"),
    (("absolute_rms_m", 250.0), ("normalized_rms", 2.45)),
)
def test_prediction_intent_chain_requires_both_diff_thresholds_exceeded(
    field: str,
    value: float,
) -> None:
    evidence = _valid_prediction_intent_evidence()
    evidence["prediction_diffs"][0][field] = value
    evidence["prediction_diffs"][0]["exceeded"] = False

    chains, violations = _MONITOR._prediction_intent_chains(evidence)

    assert chains == []
    assert "prediction_diff_below_threshold" in violations


def test_adversary_response_event_cannot_impersonate_plan_application() -> None:
    evidence = _valid_prediction_intent_evidence()
    evidence["events"][2]["payload"]["phase"] = "blue_response"

    chains, violations = _MONITOR._prediction_intent_chains(evidence)

    assert chains == []
    assert "missing_blue_response" in violations


def test_final_acceptance_requires_at_least_one_prediction_intent_chain() -> None:
    chains, violations = _MONITOR._prediction_intent_chains({"events": []})

    assert chains == []
    assert violations == ["missing_prediction_intent_evidence_chain"]


@pytest.mark.parametrize(
    ("break_link", "expected_violation"),
    [
        ("prediction_diff", "missing_prediction_diff"),
        ("provider", "missing_real_intent_provider"),
        ("provider_hash", "missing_real_intent_provider"),
        ("confirmation", "missing_intent_confirmation"),
        ("decision", "missing_regional_replan"),
        ("plan", "missing_committed_plan"),
        ("blue_response", "missing_blue_response"),
    ],
)
def test_prediction_intent_report_chain_rejects_each_missing_link(
    break_link: str,
    expected_violation: str,
) -> None:
    evidence = deepcopy(_valid_prediction_intent_evidence())
    if break_link == "prediction_diff":
        evidence["prediction_diffs"] = []
    elif break_link == "provider":
        evidence["llm_calls"][0]["model"] = "test-intent-model"
    elif break_link == "provider_hash":
        evidence["events"][1]["payload"]["intent_llm_calls"][0][
            "request_hash"
        ] = "forged-request"
    elif break_link == "confirmation":
        evidence["events"] = [
            event
            for event in evidence["events"]
            if event["event_type"] != "target_intent_changed"
        ]
    elif break_link == "decision":
        evidence["decisions"] = []
    elif break_link == "plan":
        evidence["committed_plans"] = []
    elif break_link == "blue_response":
        evidence["events"] = [
            event
            for event in evidence["events"]
            if event["payload"].get("phase") != "plan_applied"
        ]

    chains, violations = _MONITOR._prediction_intent_chains(evidence)

    assert chains == []
    assert expected_violation in violations


def test_reports_render_prediction_intent_chain_in_json_and_markdown(
    tmp_path: Path,
) -> None:
    chains, violations = _MONITOR._prediction_intent_chains(
        _valid_prediction_intent_evidence()
    )
    assert not violations
    report_path = tmp_path / "acceptance.json"
    _MONITOR._write_reports(
        FullBattleAcceptance(
            completed=True,
            final_sim_time_s=28_800,
            final_plan_version=3,
            prediction_intent_chains=tuple(chains),
        ),
        report_path,
    )

    json_text = report_path.read_text(encoding="utf-8")
    markdown = report_path.with_suffix(".md").read_text(encoding="utf-8")
    assert '"prediction_intent_chains"' in json_text
    assert "## Prediction Intent Chains" in markdown
    assert "`300.0/250.0`" in markdown
    assert "`3.0/2.45`" in markdown
    assert "LongCat-Flash-Chat" in markdown
    assert "`mission-plan:3:target_00:applied`" in markdown


def test_report_renders_exact_adversary_provider_and_motion_breakdown(
    tmp_path: Path,
) -> None:
    chains = _MONITOR._evidence_chains(_valid_evidence())
    report_path = tmp_path / "acceptance.json"
    _MONITOR._write_reports(
        FullBattleAcceptance(
            completed=False,
            final_sim_time_s=30,
            final_plan_version=3,
            battle_evidence_chains=tuple(chains),
            motion_audits=(
                EntityMotionAudit(
                    entity_id="uuv_00",
                    entity_kind="uuv",
                    observed_steps=1,
                    max_speed_mps=1.0,
                    max_acceleration_mps2=0.0,
                    max_deceleration_mps2=0.0,
                    max_turn_rate_rad_s=0.0,
                    route_violation_count=1,
                    limit_violation_count=1,
                ),
            ),
        ),
        report_path,
    )

    markdown = report_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "`LLM-ADV-1`" in markdown
    assert "LongCat-Flash-Chat" in markdown
    assert "Total / teleport / boundary / owner / route / formation / resource" in markdown
    assert "`1/0/0/0/1/0/0`" in markdown


def test_collect_stage_ids_uses_same_semantics_for_browser_and_acceptance() -> None:
    stages = live_demo.collect_stage_ids(
        {
            "events": [{"event_type": "uuv_boundary_entry_started"}],
            "planning": {"status": "committed"},
        }
    )

    assert stages == frozenset({"initial_plan_committed", "uuv_boundary_entry"})


def test_report_never_renders_pass_when_postcheck_violations_exist(
    tmp_path: Path,
) -> None:
    inconsistent = FullBattleAcceptance(
        completed=True,
        final_sim_time_s=28_800,
        final_plan_version=3,
    ).model_copy(update={"violations": ("shutdown_exceeded_10s",)})

    report_path = tmp_path / "acceptance.json"
    _MONITOR._write_reports(inconsistent, report_path)

    markdown = report_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "Status: **BLOCKED/FAIL**" in markdown
    assert "Status: **PASS**" not in markdown


def test_mileage_exhaustion_counts_as_a_resource_threshold_stage() -> None:
    assert "uuv_range_exhausted" in live_demo._STAGE_MARKERS["resource_threshold"]


def test_real_provider_attestation_rejects_injected_longcat_named_fake() -> None:
    evidence = {
        "provider_attestations": [
            {
                "role": role,
                "model": "LongCat-2.0",
                "client_type": "ScriptedLongCat",
                "transport": "injected",
                "configured_endpoint": "https://provider.example/v1",
                "attested": False,
            }
            for role in ("master", "slave", "adversary")
        ]
    }

    assert _MONITOR._real_provider_attestation_violations(evidence) == (
        "real_provider_not_attested:adversary,master,slave",
    )


def test_real_provider_attestation_requires_successful_call_for_every_role() -> None:
    evidence = {
        "provider_attestations": [
            {
                "role": role,
                "model": f"LongCat-{role}",
                "client_type": "underwater_tracking.agent.llm_factory.RoleHTTPStructuredLLM",
                "transport": "httpx",
                "configured_endpoint": "https://provider.example/v1",
                "attested": True,
            }
            for role in ("master", "slave", "adversary")
        ],
        "llm_calls": [
            {
                "call_id": f"LLM-{index}",
                "operation": operation,
                "model": model,
                "prompt_version": "v1",
                "request_hash": f"request-{index}",
                "response_hash": f"response-{index}",
                "error_category": "",
                "sim_time_s": index,
            }
            for index, operation, model in (
                (1, "regional_strategy", "LongCat-master"),
                (2, "slave_sonar_decision", "LongCat-slave"),
            )
        ],
    }

    assert _MONITOR._real_provider_attestation_violations(evidence) == (
        "real_provider_no_successful_call:adversary",
    )

    evidence["llm_calls"].append(
        {
            "call_id": "LLM-3",
            "operation": "adversary_mission_decision",
            "model": "LongCat-adversary",
            "prompt_version": "v1",
            "request_hash": "request-3",
            "response_hash": "response-3",
            "error_category": "",
            "sim_time_s": 3,
        }
    )
    assert _MONITOR._real_provider_attestation_violations(evidence) == ()


def test_real_provider_attestation_accepts_successful_role_probe() -> None:
    evidence = {
        "provider_attestations": [
            {
                "role": role,
                "model": "LongCat-2.0",
                "client_type": (
                    "underwater_tracking.agent.llm_factory.RoleHTTPStructuredLLM"
                ),
                "transport": "httpx",
                "configured_endpoint": "https://provider.example/v1",
                "attested": True,
                "probe_successful": True,
            }
            for role in ("master", "slave", "adversary")
        ],
        "llm_calls": [],
    }

    assert _MONITOR._real_provider_attestation_violations(evidence) == ()


def test_uuv_tracking_stages_must_follow_the_required_order() -> None:
    ordered = {
        "initial_plan_committed": 0,
        "uuv_boundary_entry": 90,
        "active_scan": 95,
        "target_detection": 100,
        "passive_track": 120,
        "target_maneuver": 300,
        "handoff": 600,
        "resource_threshold": 900,
        "uuv_boundary_exit": 1_000,
        "uuv_boundary_replacement": 1_200,
    }
    reversed_resource_and_recovery = {
        **ordered,
        "resource_threshold": 1_100,
        "uuv_boundary_exit": 1_000,
    }

    assert live_demo.required_stage_order_violations(ordered) == ()
    assert live_demo.required_stage_order_violations(
        reversed_resource_and_recovery
    ) == ("stage_order:resource_threshold_after_uuv_boundary_exit",)


def test_live_view_reader_retries_a_transient_planning_boundary(monkeypatch) -> None:
    responses = iter(
        (
            ({"planning": {"status": "committed"}}, 1.0),
            ({"planning": {"status": "running"}}, 2.0),
            ({"planning": {"status": "running"}}, 3.0),
            ({"planning": {"status": "running"}}, 4.0),
        )
    )

    monkeypatch.setattr(live_demo, "_get_json", lambda *_args, **_kwargs: next(responses))

    health, frame, latencies = live_demo._get_consistent_live_views(
        "http://127.0.0.1:1",
        attempts=2,
        retry_delay_s=0.0,
    )

    assert health["planning"]["status"] == "running"
    assert frame["planning"]["status"] == "running"
    assert latencies == (1.0, 2.0, 3.0, 4.0)


def test_retryable_degraded_planning_state_is_not_terminal() -> None:
    planning = {
        "status": "degraded",
        "queued_event_count": 1,
        "retry_not_before_utc_ms": 1_000,
        "dead_letter_event_ids": [],
    }

    assert live_demo._planning_failure_is_terminal(planning) is False


def test_degraded_planning_state_without_retry_is_terminal() -> None:
    planning = {
        "status": "degraded",
        "queued_event_count": 0,
        "retry_not_before_utc_ms": None,
        "dead_letter_event_ids": ["event-1"],
    }

    assert live_demo._planning_failure_is_terminal(planning) is True


def test_planning_failure_is_not_misreported_as_provider_outage(monkeypatch, tmp_path: Path) -> None:
    planning = {
        "status": "failed",
        "last_error": "internal planning invariant",
        "queued_event_count": 0,
        "retry_not_before_utc_ms": None,
        "dead_letter_event_ids": ["event-1"],
    }
    health = {"planning": planning}
    frame = {
        "sim_time_s": 0,
        "plan_version": 0,
        "run_phase": "bootstrap_planning",
        "events": [],
        "planning": planning,
    }

    monkeypatch.setattr(
        live_demo,
        "_get_consistent_live_views",
        lambda _base_url: (health, frame, (1.0,)),
    )

    def get_json(_base_url: str, path: str, *, query=None):
        del query
        if path == "/api/operational/snapshot":
            return frame, 1.0
        if path == "/api/verification/evidence":
            return {"adversary_decisions": []}, 1.0
        raise AssertionError(path)

    monkeypatch.setattr(live_demo, "_get_json", get_json)

    result = live_demo.verify_live_demo(
        base_url="http://127.0.0.1:1",
        output_dir=tmp_path,
        require_real_provider=True,
        wall_timeout_s=1.0,
        expected_duration_s=0,
        poll_interval_s=0.01,
    )

    assert any(item.startswith("planning_failed:") for item in result.violations)
    assert "real_provider_unavailable" not in result.violations


def test_memory_reader_retries_a_transient_http_failure(monkeypatch) -> None:
    responses = iter((live_demo.URLError("temporary"), ({"events": []}, 5.0)))

    def read(*_args, **_kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(live_demo, "_get_json", read)

    payload, latency = live_demo._get_json_with_retries(
        "http://127.0.0.1:1",
        "/api/assistant/memory/stream",
        attempts=2,
        retry_delay_s=0.0,
    )

    assert payload == {"events": []}
    assert latency == 5.0


def test_final_verification_reader_retries_a_transient_http_failure(monkeypatch) -> None:
    responses = iter((OSError("temporary"), {"evidence": "ready"}))

    def read(*_args, **_kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(_MONITOR, "_get_json", read)

    payload, failed = _MONITOR._safe_get_json(
        "http://127.0.0.1:1", "/api/verification/evidence"
    )

    assert payload == {"evidence": "ready"}
    assert failed is False


def test_api_startup_probe_retries_a_malformed_http_response(monkeypatch) -> None:
    responses = iter(
        (
            BadStatusLine("server is still starting"),
            {"status": "ok"},
            {"frame_id": 1},
        )
    )

    def read(*_args, **_kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(_MONITOR, "_get_json", read)

    assert _MONITOR._wait_for_api("http://127.0.0.1:1", timeout_s=1.0) is True


def test_browser_error_detail_includes_console_resource_location() -> None:
    message = SimpleNamespace(
        text="Failed to load resource: the server responded with a status of 500",
        location=lambda: {"url": "http://127.0.0.1:5173/api/assistant/memory/stream"},
    )

    detail = _MONITOR._browser_error_detail(message)

    assert detail.endswith(
        "(url=http://127.0.0.1:5173/api/assistant/memory/stream)"
    )


def test_ui_consistency_allows_a_bounded_live_frame_lag() -> None:
    assert _MONITOR._ui_consistency_violations(
        dom_plan_version=2,
        api_plan_version=3,
        dom_sim_time_s=120,
        api_sim_time_s=125,
    ) == ()
    assert "ui_sim_time_stale" in _MONITOR._ui_consistency_violations(
        dom_plan_version=2,
        api_plan_version=3,
        dom_sim_time_s=0,
        api_sim_time_s=125,
    )
    assert "ui_plan_version_stale" in _MONITOR._ui_consistency_violations(
        dom_plan_version=1,
        api_plan_version=3,
        dom_sim_time_s=120,
        api_sim_time_s=125,
    )


def test_persistent_cross_epoch_planning_conflict_is_reported() -> None:
    health = {"planning": {"status": "running", "epoch_id": "epoch-2"}}
    frame = {
        "sim_time_s": 90,
        "plan_version": 1,
        "planning": {"status": "committed", "epoch_id": "epoch-1"},
        "events": [],
        "ledger": [],
        "plan_timeline": [],
    }

    assert "planning_health_frame_mismatch" in live_demo._operational_consistency_violations(
        health, frame, None
    )


def test_newer_health_epoch_does_not_conflict_with_an_older_frame() -> None:
    health = {
        "planning": {
            "status": "running",
            "epoch_id": "epoch-2",
            "base_physics_revision": 10,
            "latest_physics_revision": 10,
        }
    }
    frame = {
        "sim_time_s": 90,
        "plan_version": 1,
        "planning_snapshot_revision": 3,
        "planning": {"status": "committed", "epoch_id": "epoch-1"},
        "events": [],
        "ledger": [],
        "plan_timeline": [],
    }

    assert "planning_health_frame_mismatch" not in live_demo._operational_consistency_violations(
        health, frame, None
    )


def test_terminal_planning_conflict_between_same_epoch_views_is_reported() -> None:
    health = {"planning": {"status": "failed", "epoch_id": "epoch-2"}}
    frame = {
        "sim_time_s": 90,
        "plan_version": 1,
        "planning": {"status": "committed", "epoch_id": "epoch-2"},
        "events": [],
        "ledger": [],
        "plan_timeline": [],
    }

    assert "planning_health_frame_mismatch" in live_demo._operational_consistency_violations(
        health, frame, None
    )


def test_operational_views_require_populated_causal_content() -> None:
    health = {"planning": {"status": "running", "epoch_id": "epoch-1"}}
    frame = {
        "run_phase": "running",
        "sim_time_s": 90,
        "plan_version": 1,
        "llm_thinking": "基于当前观测更新接力方案。",
        "llm_thinking_epoch_id": "epoch-1",
        "llm_thinking_source_event_ids": ["event-1"],
        "planning": {"status": "running", "epoch_id": "epoch-1"},
        "events": [{"event_id": "event-1", "sim_time_s": 90}],
        "ledger": [{"decision_id": "decision-1", "sim_time_s": 90}],
        "plan_timeline": [
            {
                "sim_time_s": 90,
                "plan": {"plan_id": "plan-1", "version": 1},
            }
        ],
    }
    memory = {
        "user_id": "operator",
        "conversation_id": "verification",
        "after_cursor": 0,
        "next_cursor": 1,
        "events": [
            {
                "cursor": 1,
                "user_id": "operator",
                "payload": {"source_event_ids": ["event-1"]},
            }
        ],
    }

    assert live_demo._operational_consistency_violations(health, frame, memory) == ()

    frame["llm_thinking"] = ""
    violations = live_demo._operational_consistency_violations(health, frame, memory)

    assert "llm_thinking_empty" in violations


def test_operational_views_reject_empty_timeline_and_unlinked_memory() -> None:
    health = {"planning": {"status": "completed", "epoch_id": "epoch-1"}}
    frame = {
        "run_phase": "completed",
        "sim_time_s": 90,
        "plan_version": 1,
        "llm_thinking": "已完成方案更新。",
        "llm_thinking_epoch_id": "epoch-1",
        "llm_thinking_source_event_ids": ["event-1"],
        "planning": {"status": "completed", "epoch_id": "epoch-1"},
        "events": [{"event_id": "event-1", "sim_time_s": 90}],
        "ledger": [],
        "plan_timeline": [],
    }
    memory = {
        "user_id": "operator",
        "conversation_id": "verification",
        "after_cursor": 0,
        "next_cursor": 1,
        "events": [
            {
                "cursor": 1,
                "user_id": "operator",
                "payload": {"source_plan_ids": ["plan-not-in-frame"]},
            }
        ],
    }

    violations = live_demo._operational_consistency_violations(health, frame, memory)

    assert "plan_timeline_empty" in violations
    assert "plan_timeline_current_plan_missing" in violations
    assert "memory_source_missing_from_operational_views" in violations


def test_persisted_replay_audit_reports_missing_and_corrupt_logs(
    tmp_path: Path,
) -> None:
    stages: set[str] = set()
    stage_times: dict[str, int] = {}
    stage_versions: dict[str, int] = {}

    assert live_demo._collect_persisted_replay_stages(
        tmp_path,
        stages,
        stage_times,
        stage_versions,
    ) == "persisted_replay_unavailable"

    (tmp_path / "operational_frames.jsonl").write_text(
        "{not-valid-json}\n",
        encoding="utf-8",
    )
    assert live_demo._collect_persisted_replay_stages(
        tmp_path,
        stages,
        stage_times,
        stage_versions,
    ) == "persisted_replay_invalid"


def test_persisted_replay_audit_rejects_empty_and_truncated_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    replay_path = tmp_path / "operational_frames.jsonl"
    replay_path.write_text("", encoding="utf-8")
    assert live_demo._collect_persisted_replay_stages(
        tmp_path,
        set(),
        {},
        {},
        expected_duration_s=28_800,
        expected_plan_version=4,
    ) == "persisted_replay_empty"

    class TruncatedReplay:
        def __init__(self, _path: Path) -> None:
            pass

        def count(self) -> int:
            return 1

        def range(self, *, offset: int, limit: int):
            del offset, limit
            return (
                type(
                    "Frame",
                    (),
                    {
                        "events": (),
                        "mission_events": (),
                        "planning": None,
                        "sim_time_s": 600,
                        "plan_version": 2,
                        "run_phase": "running",
                    },
                )(),
            )

    monkeypatch.setattr(live_demo, "ReplayService", TruncatedReplay)
    assert live_demo._collect_persisted_replay_stages(
        tmp_path,
        set(),
        {},
        {},
        expected_duration_s=28_800,
        expected_plan_version=4,
    ) == "persisted_replay_terminal_mismatch"


def test_nonzero_main_process_exit_is_a_shutdown_violation() -> None:
    assert _MONITOR._process_exit_violation(17) == "main_process_exit:17"
    assert _MONITOR._process_exit_violation(0) is None
    assert _MONITOR._process_exit_violation(130) is None


def test_required_ui_surface_audit_does_not_treat_missing_nodes_as_optional() -> None:
    violations = _MONITOR._required_ui_surface_violations(
        {
            "drawer_toggle": 0,
            "mission_panel": 1,
            "plan_version": 0,
            "sim_time": 1,
            "memory_window": 0,
            "brain_section": 1,
        }
    )

    assert violations == (
        "missing_ui_surface:drawer_toggle",
        "missing_ui_surface:plan_version",
        "missing_ui_surface:memory_window",
    )
