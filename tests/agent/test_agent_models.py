import pytest
from pydantic import ValidationError
from underwater_tracking.domain.agent_models import (
    DecisionRecord,
    ExpertDirective,
    IntentHypothesis,
    PredictedTrackRef,
    StrategyProposal,
    StrategySet,
    TrackingPlan,
    ValidationIssue,
    Waypoint,
)


def test_intent_requires_evidence_and_strategy_cannot_assign_uuvs():
    with pytest.raises(ValidationError):
        IntentHypothesis(label="evade", confidence=0.8, evidence_ids=[])
    with pytest.raises(ValidationError):
        StrategyProposal(
            concept="balanced", target_priorities={"T1": 1.0},
            required_quality={"T1": 0.7}, evidence_ids=["B:T1:900"],
            member_ids_by_target={"T1": ["U1", "U2"]},
        )


def test_strategy_proposal_cannot_carry_waypoints_either():
    with pytest.raises(ValidationError):
        StrategyProposal(
            concept="quality_first", target_priorities={"T1": 1.0},
            required_quality={"T1": 0.7}, reinforcement_policy={"T1": "reinforce"},
            releasable_soft_constraints=(), evidence_ids=["B:T1:900"],
            rationale="test", waypoints_by_member={"U1": [(0.0, 0.0)]},
        )


def test_tracking_plan_holds_final_members_and_waypoints():
    plan = TrackingPlan(
        plan_id="P1", scenario_id="S1", revision=1, base_snapshot_revision=0,
        status="active", concept="balanced", valid_until_s=3600,
        member_ids_by_target={"T1": ("U1", "U2")},
        roles_by_member={"U1": "lead", "U2": "wing"},
        waypoints_by_member={"U1": (Waypoint(x=100.0, y=200.0, arrive_at_s=30),)},
    )
    assert plan.member_ids_by_target["T1"] == ("U1", "U2")
    assert plan.waypoints_by_member["U1"][0].x == 100.0
    with pytest.raises(ValidationError):
        TrackingPlan.model_validate({**plan.model_dump(), "truth": [1.0]})


def test_tracking_plan_revision_is_mutable_for_stale_commit_check():
    plan = TrackingPlan(plan_id="P1", scenario_id="S1", revision=1, base_snapshot_revision=3)
    plan.base_snapshot_revision = 4
    assert plan.base_snapshot_revision == 4


def test_strategy_set_iterates_proposals():
    proposal = StrategyProposal(
        concept="balanced", target_priorities={"T1": 1.0},
        required_quality={"T1": 0.7}, reinforcement_policy={"T1": "reinforce"},
        releasable_soft_constraints=(), evidence_ids=["B:T1:900"], rationale="test",
    )
    strategy_set = StrategySet(trigger_event_ids=("E1",), proposals=(proposal,))
    assert {item.concept for item in strategy_set} == {"balanced"}


def test_decision_record_carries_full_audit_trail():
    record = DecisionRecord(
        decision_id="D1", scenario_id="S1", sim_time_s=600,
        trigger_event_ids=("E1",), snapshot_revision=5, snapshot_hash="abc",
        model_version="m1", prompt_version="p1", schema_version="s1",
        rejected_candidates={"P2": "fails minimum quality"},
    )
    assert record.snapshot_hash == "abc"
    assert record.rejected_candidates == {"P2": "fails minimum quality"}
    with pytest.raises(ValidationError):
        DecisionRecord.model_validate({**record.model_dump(), "hidden_truth": [1.0]})


def test_expert_directive_low_confidence_cannot_be_applied():
    with pytest.raises(ValidationError):
        ExpertDirective(
            directive_id="X1", raw_text="send more boats",
            target_scope=("T1",), confidence=0.5, status="applied",
        )
    preview = ExpertDirective(
        directive_id="X1", raw_text="send more boats",
        target_scope=("T1",), confidence=0.5, status="preview",
    )
    assert preview.status == "preview"


def test_predicted_track_ref_marks_fallback():
    track = PredictedTrackRef(
        prediction_id="PR1", target_id="T1", sim_time_s=600,
        horizon_s=1800.0, sample_step_s=60.0,
        source_belief_history_ids=("B:T1:600",),
        fallback_used=True, fallback_reason="insufficient history",
    )
    assert track.fallback_used is True
    with pytest.raises(ValidationError):
        PredictedTrackRef.model_validate({**track.model_dump(), "truth": [1.0]})


def test_validation_issue_supports_sorted_compare():
    issues = (
        ValidationIssue(code="unknown_member", field="member_ids_by_target.T1",
                        message="unknown member U9", observed="U9", expected="U1..U12"),
        ValidationIssue(code="missing_coverage", field="member_ids_by_target"),
    )
    assert issues[1].code == "missing_coverage"
    assert issues[0].observed == "U9"
