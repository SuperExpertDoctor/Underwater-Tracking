from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from underwater_tracking.verification import live_demo
from underwater_tracking.verification.physics_invariants import (
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
                "overlap_start_s": 60.0,
                "overlap_end_s": 660.0,
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
                "event_id": "E-blue-response",
                "event_type": "state_changed",
                "entity_id": "target_00",
                "sim_time_s": 70,
                "payload": {"phase": "blue_response", "plan_revision": 3},
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
    assert chain.blue_response_event_ids == ("E-blue-response",)


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
            if event["payload"].get("phase") != "blue_response"
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
    assert "`E-blue-response`" in markdown


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


def test_planning_status_transition_between_live_views_is_not_a_contradiction() -> None:
    health = {"planning": {"status": "running", "epoch_id": "epoch-2"}}
    frame = {
        "sim_time_s": 90,
        "plan_version": 1,
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
