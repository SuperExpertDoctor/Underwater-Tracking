from __future__ import annotations

import importlib.util
from pathlib import Path

from underwater_tracking.verification.physics_invariants import (
    EntityMotionLimits,
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
            }
        ],
        "blue_response_chains": [
            {
                "target_id": "target_00",
                "decision_id": "dec-1",
                "maneuver_time_s": 20,
                "response_event_id": "chain-1:blue_response",
                "plan_version": 3,
                "blue_estimate_ids": ("estimate-1",),
                "public_observation_ids": ("obs-1",),
            }
        ],
    }


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
    assert _MONITOR._evidence_chains(forged) == []
