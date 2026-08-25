from __future__ import annotations

from types import SimpleNamespace

from underwater_tracking.agent.nodes.adversary import (
    ADVERSARY_SYSTEM_PROMPT,
    build_adversary_payload,
)
from underwater_tracking.agent.nodes.regions import RegionGenerationNode
from underwater_tracking.domain.adversary_models import (
    AdversaryBelief,
    AdversaryEscapeInput,
    AdversaryKinematicLimits,
    AdversaryMissionState,
    AdversaryOperatingBoundary,
    CommunicationsAcousticExposure,
)
from underwater_tracking.domain.agent_models import IntentHypothesis, PredictedTrackRef
from underwater_tracking.domain.regional_models import (
    GridSpec,
    TaskRegionProposal,
    TaskRegionProposalSet,
)
from underwater_tracking.planning.regions import build_llm_task_region_plan


def test_task_region_prompt_and_payload_define_grid_ordering_and_resource_policy() -> None:
    node = RegionGenerationNode(
        snapshot_provider=lambda _: None,
        map_bounds_provider=lambda _: (-12_000.0, 12_000.0, -12_000.0, 12_000.0),
        grid_spec=GridSpec(origin_xy=(-12_000.0, -12_000.0)),
        llm=SimpleNamespace(invoke_structured=lambda *_: TaskRegionProposalSet(regions=())),
    )
    prediction = SimpleNamespace(
        target_id="target_00",
        prediction_id="prediction:target_00:0",
        points_xy=((-4_000.0, -6_000.0), (-3_000.0, -5_000.0)),
        times_s=(0, 300),
        corridor_radius_m=(500.0, 1_500.0),
        source_belief_history_ids=("belief:target_00:0",),
    )
    payload = node._payload(
        SimpleNamespace(scenario_id="scenario", sim_time_s=0),
        prediction,
        IntentHypothesis(
            label="evade",
            confidence=0.8,
            evidence_ids=("belief:target_00:0",),
            model_id="model",
            prompt_version="intent-v1",
        ),
        (-12_000.0, 12_000.0, -12_000.0, 12_000.0),
    )

    assert payload["task_region_constraints"] == {
        "max_regions": 4,
        "grid_alignment_m": 1_000.0,
        "regions_must_not_overlap": True,
        "ordered_by_first_covered_prediction_time": True,
        "uuv_demand_policy": "min(4, 1 + ceil(sqrt(cell_count)))",
    }
    assert payload["output_token_budget"] == 1024
    assert payload["thinking_mode"] == "disabled"
    prompt = str(payload["system_prompt"])
    assert "chronological" in prompt
    assert "non-overlapping" in prompt
    assert "grid line" in prompt
    assert "uuv_demand_policy" in prompt


def test_task_region_payload_exposes_prior_regions_and_uuv_change_context() -> None:
    node = RegionGenerationNode(
        snapshot_provider=lambda _: None,
        map_bounds_provider=lambda _: (-2_000.0, 4_000.0, -2_000.0, 4_000.0),
        grid_spec=GridSpec(),
        llm=SimpleNamespace(invoke_structured=lambda *_: TaskRegionProposalSet(regions=())),
    )
    prediction = SimpleNamespace(
        target_id="target_00",
        prediction_id="prediction:target_00:1",
        points_xy=((500.0, 500.0), (1_500.0, 500.0)),
        times_s=(100, 200),
        corridor_radius_m=(200.0, 600.0),
        source_belief_history_ids=("belief:target_00:1",),
    )
    snapshot = SimpleNamespace(
        scenario_id="scenario",
        sim_time_s=100,
        active_plan=SimpleNamespace(
            regional_plans={
                "target_00": SimpleNamespace(
                    task_regions=(
                        SimpleNamespace(
                            region_id="target_00:task:01",
                            lower_left_xy=(0.0, 0.0),
                            upper_right_xy=(1_000.0, 1_000.0),
                            cell_ids=("target_00:cell:0:0",),
                            required_uuv_count=1,
                            active_window=SimpleNamespace(start_s=100, end_s=200),
                        ),
                    )
                )
            },
            region_tasks={
                "target_00:cell:0:0": SimpleNamespace(assigned_uuv_ids=("uuv_00",))
            },
        ),
    )

    payload = node._payload(
        snapshot,
        prediction,
        IntentHypothesis(
            label="evade",
            confidence=0.8,
            evidence_ids=("belief:target_00:1",),
            model_id="model",
            prompt_version="intent-v1",
        ),
        (-2_000.0, 4_000.0, -2_000.0, 4_000.0),
    )

    rolling = payload["rolling_planning_context"]
    assert rolling["iou_retention_threshold"] == 0.6
    assert rolling["prior_task_regions"][0]["assigned_uuv_ids"] == ["uuv_00"]
    assert payload["expected_uuv_allocation"]["rule"] == "min(4, 1 + ceil(sqrt(cell_count)))"


def test_task_region_generation_reflects_on_iou_robustness_and_force_demand() -> None:
    intent = IntentHypothesis(
        label="evade",
        confidence=0.8,
        evidence_ids=("belief:target_00:1",),
        model_id="model",
        prompt_version="intent-v1",
    )
    prediction = PredictedTrackRef(
        prediction_id="prediction:target_00:1",
        target_id="target_00",
        sim_time_s=100,
        horizon_s=200.0,
        sample_step_s=100.0,
        times_s=(100.0, 200.0),
        points_xy=((500.0, 500.0), (1_500.0, 500.0)),
        corridor_radius_m=(200.0, 600.0),
        source_belief_history_ids=("belief:target_00:1",),
    )
    bounds = (-2_000.0, 4_000.0, -2_000.0, 4_000.0)
    old_plan = build_llm_task_region_plan(
        prediction,
        intent,
        TaskRegionProposalSet(
            regions=(
                TaskRegionProposal(
                    lower_left_xy=(0.0, 0.0),
                    upper_right_xy=(1_000.0, 1_000.0),
                    rationale="old early coverage",
                ),
            )
        ),
        bounds,
        GridSpec(),
    )

    class RevisionLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def invoke_structured(self, _operation, payload, _response_model, **_kwargs):
            self.calls.append(payload)
            return TaskRegionProposalSet(
                regions=(
                    TaskRegionProposal(
                        lower_left_xy=(0.0, 0.0),
                        upper_right_xy=(2_000.0, 1_000.0),
                        rationale="cover the uncertainty corridor",
                    ),
                )
            )

    llm = RevisionLLM()
    snapshot = SimpleNamespace(
        scenario_id="scenario",
        sim_time_s=100,
        active_plan=SimpleNamespace(
            regional_plans={"target_00": old_plan},
            region_tasks={task.region_id: task for task in old_plan.tasks},
        ),
    )
    node = RegionGenerationNode(
        snapshot_provider=lambda _: snapshot,
        map_bounds_provider=lambda _: bounds,
        grid_spec=GridSpec(),
        llm=llm,
    )

    result = node(
        {
            "snapshot_ref": "snapshot",
            "intent_hypotheses": {"target_00": intent},
            "predictions": {"target_00": prediction},
        }
    )

    assert len(llm.calls) == 2
    reflection = llm.calls[1]["rolling_reflection"]
    assert reflection["draft_stability"]["mean_best_iou"] == 0.5
    assert reflection["draft_robustness"]["corridor_capture"] == 0.916667
    assert reflection["draft_expected_uuv_allocation"]["peak_required_uuv_count"] == 3
    assert result["regional_plans"]["target_00"].task_regions[0].required_uuv_count == 3


def test_adversary_prompt_defines_counter_tracking_tradeoffs_and_feedback() -> None:
    context = AdversaryEscapeInput(
        target_id="target_00",
        sim_time_s=120,
        mission_state=AdversaryMissionState(
            target_id="target_00",
            task_region_id="mission",
            task_region_polygon_xy=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            mission_route_xy=((0.0, 0.0), (1_000.0, 0.0)),
            escape_regions={"north": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))},
            current_route_index=0,
        ),
        belief=AdversaryBelief(
            target_id="target_00",
            as_of_s=120,
            estimated_position_xy=(0.0, 0.0),
            estimated_velocity_xy=(4.0, 0.0),
            position_uncertainty_m=50.0,
            velocity_uncertainty_mps=0.5,
            estimated_heading=0.0,
            estimated_speed_mps=4.0,
            intent_hypothesis="silent_transit",
            intent_confidence=0.6,
        ),
        communications_acoustic_exposure=CommunicationsAcousticExposure(
            as_of_s=120,
            passive_signature_level=0.1,
            active_emitter_exposure=0.2,
            communication_intercept_risk=0.2,
            relay_detection_risk=0.2,
            acoustic_clutter_level=0.3,
            own_emission_mode="passive",
        ),
        kinematic_limits=AdversaryKinematicLimits(
            max_speed_mps=14.0,
            max_turn_rate_rad_s=0.1,
            decision_horizon_s=30.0,
            max_decoy_count=2,
            decoy_inventory=2,
        ),
        operating_boundary=AdversaryOperatingBoundary(
            min_x=-1_000.0, max_x=1_000.0, min_y=-1_000.0, max_y=1_000.0
        ),
    )

    payload = build_adversary_payload(context)

    assert payload["decision_policy"]["objective"] == "reduce_detectability_while_preserving_mission_feasibility"
    assert "continue_mission" in payload["decision_policy"]["intent_semantics"]
    assert "break_contact" in payload["decision_policy"]["intent_semantics"]
    assert "decision_history" in payload
    assert "mission progress" in ADVERSARY_SYSTEM_PROMPT
    assert "previous decision outcomes" in ADVERSARY_SYSTEM_PROMPT
