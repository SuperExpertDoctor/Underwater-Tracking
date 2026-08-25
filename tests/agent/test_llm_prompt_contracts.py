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
from underwater_tracking.domain.agent_models import IntentHypothesis
from underwater_tracking.domain.regional_models import GridSpec, TaskRegionProposalSet


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
