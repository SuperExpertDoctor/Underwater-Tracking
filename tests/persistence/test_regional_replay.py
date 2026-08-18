from __future__ import annotations

import json

from underwater_tracking.api.frame_builder import build_operational_frame
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.domain.models import RuntimeEvent
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository

from tests.api.test_frame_builder_regional_views import _regional_plan
from tests.api.test_frame_pipeline import _plan, _report, _snapshot


def test_regional_revision_roundtrips_complete_payload_and_llm_hashes(tmp_path):
    database_path = tmp_path / "regional-replay.db"
    repository = PlanRepository(database_path)
    plan = _plan().model_copy(
        update={
            "regional_plans": {"T1": _regional_plan()},
            "trigger_event_ids": ("event-target-turn", "event-relay-drop"),
            "evidence_ids": ("evidence-contact",),
        }
    )
    repository.set_snapshot_revision(plan.scenario_id, plan.base_snapshot_revision)
    repository.commit(plan)

    revisions = repository.list_regional_revisions(plan.scenario_id, target_id="T1")

    assert len(revisions) == 1
    revision = revisions[0]
    assert revision.plan_revision == plan.revision
    assert revision.trigger_event_ids == ("event-target-turn", "event-relay-drop")
    assert revision.regional_plan.grid_spec.target_grid_cells == 64
    cell = revision.regional_plan.cells[0]
    task = revision.regional_plan.tasks[0]
    assert cell.visit_windows[0].start_s == 101
    assert task.uuv_roles == ("passive_tracker", "handoff_reserve")
    assert task.communication_links == ("UUV-1->USV-1", "USV-1->carrier-01")
    assert task.sonar_policy.active_mode == "probe"
    assert task.successor_region_id == "T1:cell:1:0"
    assert revision.regional_plan.tasks[2].degraded_reasons == ("relay_margin_low",)
    assert revision.regional_plan.evidence_ids == ("plan-evidence",)

    ledger = DecisionLedger(database_path)
    ledger.record_llm_call(
        operation="regional_strategy",
        model="test-model",
        prompt_version="regional-v1",
        request_hash="request-hash",
        response_hash="response-hash",
        scenario_id=plan.scenario_id,
        sim_time_s=100,
    )

    calls = ledger.list_llm_calls(
        scenario_id=plan.scenario_id, operation="regional_strategy"
    )
    assert [(call.request_hash, call.response_hash) for call in calls] == [
        ("request-hash", "response-hash")
    ]


def test_replay_restores_regional_frame_and_accepts_prior_optional_shape(tmp_path):
    plan = _plan().model_copy(
        update={
            "regional_plans": {"T1": _regional_plan()},
            "trigger_event_ids": ("event-replan",),
        }
    )
    event = RuntimeEvent(
        event_id="event-replan",
        scenario_id=plan.scenario_id,
        sim_time_s=100,
        event_type="regional_replan",
        level="tactical",
        entity_id="T1",
    )
    frame = build_operational_frame(
        _snapshot(
            reports=(_report("T1", "G1", (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0))),)
        ),
        plan,
        (),
        (event,),
        (),
    )
    current_path = tmp_path / "current.jsonl"
    current_path.write_text(frame.model_dump_json() + "\n", encoding="utf-8")

    restored = ReplayService(current_path).range()

    assert restored[0].regional_plans["T1"] == frame.regional_plans["T1"]
    assert restored[0].regional_plans["T1"].causal_event_ids == ("event-replan",)

    legacy_payload = frame.model_dump(mode="json")
    regional = legacy_payload["regional_plans"]["T1"]
    for field in (
        "grid_spec",
        "evidence_ids",
        "current_handoff_region_id",
        "next_handoff_region_id",
        "causal_event_ids",
    ):
        regional.pop(field)
    for region in regional["regions"]:
        for field in (
            "grid_x",
            "grid_y",
            "visit_window_index",
            "visit_window",
            "uuv_roles",
            "usv_role",
            "sonar_policy",
            "communication",
            "communication_links",
            "degraded_reasons",
            "evidence_ids",
            "revision",
        ):
            region.pop(field)
    legacy_path = tmp_path / "legacy.jsonl"
    legacy_path.write_text(json.dumps(legacy_payload) + "\n", encoding="utf-8")

    legacy = ReplayService(legacy_path).range()[0]

    assert legacy.regional_plans["T1"].grid_spec is None
    assert legacy.regional_plans["T1"].regions[0].visit_window is None
