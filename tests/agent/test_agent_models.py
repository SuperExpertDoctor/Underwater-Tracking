import pytest
from pydantic import ValidationError
from underwater_tracking.domain.agent_models import (
    DecisionRecord,
    ExpertDirective,
    IntentHypothesis,
    PlanAdjustmentSuggestion,
    PlanAdjustmentSuggestionSet,
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
            concept="balanced",
            target_priorities={"T1": 1.0},
            required_quality={"T1": 0.7},
            evidence_ids=["B:T1:900"],
            member_ids_by_target={"T1": ["U1", "U2"]},
        )


def test_strategy_proposal_cannot_carry_waypoints_either():
    with pytest.raises(ValidationError):
        StrategyProposal(
            concept="quality_first",
            target_priorities={"T1": 1.0},
            required_quality={"T1": 0.7},
            reinforcement_policy={"T1": "reinforce"},
            releasable_soft_constraints=(),
            evidence_ids=["B:T1:900"],
            rationale="test",
            waypoints_by_member={"U1": [(0.0, 0.0)]},
        )


def test_tracking_plan_holds_final_members_and_waypoints():
    plan = TrackingPlan(
        plan_id="P1",
        scenario_id="S1",
        revision=1,
        base_snapshot_revision=0,
        status="active",
        concept="balanced",
        valid_until_s=3600,
        member_ids_by_target={"T1": ("U1", "U2")},
        roles_by_member={"U1": "lead", "U2": "wing"},
        waypoints_by_member={"U1": (Waypoint(x=100.0, y=200.0, arrive_at_s=30),)},
    )
    assert plan.member_ids_by_target["T1"] == ("U1", "U2")
    assert plan.waypoints_by_member["U1"][0].x == 100.0
    with pytest.raises(ValidationError):
        TrackingPlan.model_validate({**plan.model_dump(), "truth": [1.0]})


def test_current_plan_contract_rejects_legacy_usv_assignments():
    with pytest.raises(ValidationError):
        TrackingPlan(
            plan_id="P-usv",
            scenario_id="S1",
            revision=1,
            base_snapshot_revision=0,
            member_ids_by_target={"T1": ("U1",)},
            usv_ids_by_target={"T1": ("USV-1",)},
        )


def test_current_command_contract_rejects_legacy_usv_actions():
    from underwater_tracking.domain.agent_models import PlanCommand

    with pytest.raises(ValidationError):
        PlanCommand(
            command_id="C-usv",
            plan_id="P1",
            plan_revision=1,
            scenario_id="S1",
            group_id="G1",
            target_id="T1",
            sim_time_s=0,
            member_ids=("U1",),
            usv_ids=("USV-1",),
            usv_actions={"USV-1": "relay"},
        )


def test_tracking_plan_revision_is_mutable_for_stale_commit_check():
    plan = TrackingPlan(plan_id="P1", scenario_id="S1", revision=1, base_snapshot_revision=3)
    plan.base_snapshot_revision = 4
    assert plan.base_snapshot_revision == 4


def test_strategy_set_iterates_proposals():
    proposal = StrategyProposal(
        concept="balanced",
        target_priorities={"T1": 1.0},
        required_quality={"T1": 0.7},
        reinforcement_policy={"T1": "reinforce"},
        releasable_soft_constraints=(),
        evidence_ids=["B:T1:900"],
        rationale="test",
    )
    strategy_set = StrategySet(trigger_event_ids=("E1",), proposals=(proposal,))
    assert {item.concept for item in strategy_set} == {"balanced"}


def test_plan_adjustment_suggestions_require_four_distinct_categories():
    categories = (
        "tracking_quality",
        "segmented_handoff",
        "resource_rotation",
        "commander_preference",
    )
    suggestions = tuple(
        PlanAdjustmentSuggestion(
            suggestion_id=f"S-{index}",
            category=category,
            title=f"Suggestion {index}",
            rationale="Current observation factors support this option.",
            proposed_feedback=f"Please consider option {index}.",
            target_ids=("T1",),
            evidence_ids=("E1",),
            confidence=0.8,
        )
        for index, category in enumerate(categories, start=1)
    )
    result = PlanAdjustmentSuggestionSet(suggestions=suggestions)
    assert len(result.suggestions) == 4
    with pytest.raises(ValidationError):
        PlanAdjustmentSuggestionSet(suggestions=(suggestions[0],) * 4)


def test_decision_record_carries_full_audit_trail():
    record = DecisionRecord(
        decision_id="D1",
        scenario_id="S1",
        sim_time_s=600,
        trigger_event_ids=("E1",),
        snapshot_revision=5,
        snapshot_hash="abc",
        model_version="m1",
        prompt_version="p1",
        schema_version="s1",
        rejected_candidates={"P2": "fails minimum quality"},
    )
    assert record.snapshot_hash == "abc"
    assert record.rejected_candidates == {"P2": "fails minimum quality"}
    with pytest.raises(ValidationError):
        DecisionRecord.model_validate({**record.model_dump(), "hidden_truth": [1.0]})


def test_expert_directive_low_confidence_cannot_be_applied():
    with pytest.raises(ValidationError):
        ExpertDirective(
            directive_id="X1",
            raw_text="send more boats",
            target_scope=("T1",),
            confidence=0.5,
            status="applied",
        )
    preview = ExpertDirective(
        directive_id="X1",
        raw_text="send more boats",
        target_scope=("T1",),
        confidence=0.5,
        status="preview",
    )
    assert preview.status == "preview"


def test_expert_directive_can_explicitly_request_uuv_return():
    directive = ExpertDirective(
        directive_id="X2",
        raw_text="UUV-1 返回母舰",
        target_scope=("T1",),
        return_uuv_ids=("UUV-1",),
        confidence=1.0,
        status="preview",
    )
    assert directive.return_uuv_ids == ("UUV-1",)


def test_predicted_track_ref_marks_fallback():
    track = PredictedTrackRef(
        prediction_id="PR1",
        target_id="T1",
        sim_time_s=600,
        horizon_s=1800.0,
        sample_step_s=60.0,
        source_belief_history_ids=("B:T1:600",),
        fallback_used=True,
        fallback_reason="insufficient history",
    )
    assert track.fallback_used is True
    with pytest.raises(ValidationError):
        PredictedTrackRef.model_validate({**track.model_dump(), "truth": [1.0]})


@pytest.mark.parametrize(
    "probabilities",
    (
        {"cv": -0.1, "left_turn": 1.1},
        {"cv": float("nan")},
        {"cv": 0.0, "left_turn": 0.0},
    ),
)
def test_predicted_track_ref_rejects_invalid_imm_probabilities(probabilities):
    with pytest.raises(ValidationError):
        PredictedTrackRef(
            prediction_id="PR-invalid",
            target_id="T1",
            sim_time_s=600,
            horizon_s=1800.0,
            sample_step_s=60.0,
            imm_model_probabilities=probabilities,
        )


def test_validation_issue_supports_sorted_compare():
    issues = (
        ValidationIssue(
            code="unknown_member",
            field="member_ids_by_target.T1",
            message="unknown member U9",
            observed="U9",
            expected="U1..U12",
        ),
        ValidationIssue(code="missing_coverage", field="member_ids_by_target"),
    )
    assert issues[1].code == "missing_coverage"
    assert issues[0].observed == "U9"


def test_reinforcement_policy_coerces_scalar_values_to_str():
    proposal = StrategyProposal(
        concept="balanced",
        target_priorities={"T1": 1.0},
        required_quality={"T1": 0.7},
        reinforcement_policy={
            "release_when_stable": "release_when_stable",
            "max_additional_groups": 1,
            "priority_boost": 1.5,
            "strict_mode": True,
        },
        releasable_soft_constraints=("energy_reserve_0.1",),
        evidence_ids=("B:T1:900",),
        rationale="schema coercion round",
    )
    assert proposal.reinforcement_policy == {
        "release_when_stable": "release_when_stable",
        "max_additional_groups": "1",
        "priority_boost": "1.5",
        "strict_mode": "True",
    }
    with pytest.raises(ValidationError):
        StrategyProposal(
            concept="balanced",
            target_priorities={"T1": 1.0},
            required_quality={"T1": 0.7},
            reinforcement_policy=[("release_when_stable", "release_when_stable")],
            releasable_soft_constraints=("energy_reserve_0.1",),
            evidence_ids=("B:T1:900",),
            rationale="non-dict input must still fail",
        )


@pytest.mark.parametrize("required_quality", [{"T1": float("nan")}, {"T1": float("inf")}])
def test_strategy_proposal_rejects_non_finite_required_quality(required_quality):
    with pytest.raises(ValidationError):
        StrategyProposal(
            concept="balanced",
            target_priorities={"T1": 1.0},
            required_quality=required_quality,
            reinforcement_policy={"release_when_stable": "release_when_stable"},
            releasable_soft_constraints=("energy_reserve_0.1",),
            evidence_ids=("B:T1:900",),
            rationale="finite quality",
        )
