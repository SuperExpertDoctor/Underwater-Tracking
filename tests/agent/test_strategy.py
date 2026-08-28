"""Strategy payload coverage for executable operational constraints."""

from typing import Any
from typing import cast

from underwater_tracking.agent.llm import StructuredLLM
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.agent.nodes.strategy import StrategyGenerationNode, _platform_aggregate
from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    IntentHypothesis,
    PlanAdjustmentSuggestion,
    PlanAdjustmentSuggestionSet,
    StrategyProposal,
)
from underwater_tracking.domain.models import (
    GroupQuality,
    GroupReport,
    IntelligenceReport,
    OperationalScheme,
    SituationSnapshot,
    SurveillanceCapability,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    CommunicationCapability,
    CommunicationLink,
    MotionLimits,
    PlatformCapability,
    PlatformKind,
    PlatformRoster,
    PlatformSnapshot,
    SonarCapability,
    UUVPlatformState,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent


def test_strategy_generation_has_no_external_knowledge_field() -> None:
    node = StrategyGenerationNode(cast(StructuredLLM[StrategyProposal], object()))

    payload = node.build_payload({}, "balanced")

    assert "external_knowledge" not in payload


def test_strategy_payload_summarizes_valid_scheme_intelligence_and_capabilities() -> None:
    situation = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=4,
        sim_time_s=50,
        uuvs=(
            UUVState(
                uuv_id="U1",
                position_xy=(0.0, 0.0),
                heading_rad=0.0,
                speed_mps=3.0,
                energy_fraction=0.8,
                status=UUVStatus.TRACKING,
                capability=SurveillanceCapability(
                    passive_range_m=2500.0,
                    bearing_variance_rad2=0.02,
                    max_speed_mps=3.0,
                    max_turn_rate_rad_s=0.04,
                ),
            ),
        ),
        group_reports=(
            GroupReport(
                group_id="G-T1",
                target_id="T1",
                sim_time_s=50,
                member_ids=("U1",),
                belief=TargetBelief(
                    target_id="T1",
                    sim_time_s=50,
                    mean=(0.0, 0.0),
                    covariance=((1.0, 0.0), (0.0, 1.0)),
                    model_probabilities={"cv": 1.0},
                    fim_min_eigenvalue=0.01,
                    fim_condition=10.0,
                ),
                quality=GroupQuality(
                    instant=0.7,
                    window_mean=0.7,
                    ewma=0.7,
                    components={},
                ),
                plan_revision=1,
            ),
        ),
        pending_events=(),
        operational_scheme=OperationalScheme(
            scheme_id="scheme-1",
            version=2,
            target_priorities={"T1": 3.0},
            minimum_quality={"T1": 0.85},
            valid_from_s=10,
            valid_until_s=100,
            constraints=("keep-passive",),
        ),
        intelligence_reports=(
            IntelligenceReport(
                report_id="intel-current",
                source="sonar",
                target_id="T1",
                confidence=0.8,
                issued_at_s=20,
                valid_until_s=90,
                content_summary="Operator task summary: maintain passive coverage.",
                assessment={"maneuver": "evasive"},
            ),
            IntelligenceReport(
                report_id="intel-expired",
                source="sonar",
                target_id="T1",
                confidence=0.8,
                issued_at_s=20,
                valid_until_s=49,
                assessment={"maneuver": "stale"},
            ),
        ),
    )
    node = StrategyGenerationNode(
        cast(StructuredLLM[StrategyProposal], object()),
        snapshot_provider=lambda _: PlanningSnapshot(situation, None, ()),
    )
    payload = node.build_payload(
        {
            "scenario_id": "S1",
            "snapshot_ref": "snapshot:4",
            "intent_hypotheses": {
                "T1": IntentHypothesis(
                    label="evade",
                    confidence=0.8,
                    evidence_ids=("B:T1:50",),
                    model_id="model",
                    prompt_version="intent-v1",
                )
            },
        },
        "balanced",
    )

    factors = payload["decision_factors"]
    assert factors["operational_scheme"]["minimum_quality"] == {"T1": 0.85}
    assert factors["intelligence_summaries"] == [
        {
            "report_id": "intel-current",
            "source": "sonar",
            "target_id": "T1",
            "confidence": 0.8,
            "valid_until_s": 90,
            "assessment": {"maneuver": "evasive"},
        }
    ]
    assert "content_summary" not in factors["intelligence_summaries"][0]
    assert factors["capability_summary"]["passive_range_m"]["minimum"] == 2500.0
    assert factors["capability_summary"]["active_range_m"]["minimum"] == 3000.0
    assert factors["capability_summary"]["endurance_s"]["minimum"] == 28_800.0
    assert factors["capability_summary"]["availability"]["minimum"] == 1.0
    assert factors["capability_summary"]["passive_sonar_available_count"] == 1
    assert factors["capability_summary"]["active_sonar_available_count"] == 1
    assert factors["required_quality_constraints"] == {"T1": 0.85}
    assert "required decision checklist" in str(payload["system_prompt"]).lower()


def test_strategy_payload_includes_applied_expert_feedback() -> None:
    situation = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=4,
        sim_time_s=50,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )
    feedback = ExpertDirective(
        directive_id="D-FEEDBACK",
        raw_text="region_2 交接延迟",
        target_scope=("T1",),
        directive_type="feedback",
        feedback_region_ids=("region_2",),
        feedback_text="region_2 交接延迟，请增加下一窗口的接力余量",
        confidence=0.95,
        status="applied",
    )
    node = StrategyGenerationNode(
        cast(StructuredLLM[StrategyProposal], object()),
        snapshot_provider=lambda _: PlanningSnapshot(situation, None, (feedback,)),
    )

    payload = node.build_payload(
        {"scenario_id": "S1", "snapshot_ref": "snapshot:4"},
        "balanced",
    )

    assert payload["decision_factors"]["expert_feedback"] == [
        {
            "directive_id": "D-FEEDBACK",
            "target_scope": ["T1"],
            "region_ids": ["region_2"],
            "feedback": "region_2 交接延迟，请增加下一窗口的接力余量",
        }
    ]


def test_strategy_payload_includes_uuv_platform_core_capabilities() -> None:
    capability = PlatformCapability(
        kind=PlatformKind.UUV,
        motion=MotionLimits(
            max_speed_mps=4.0,
            max_acceleration_mps2=0.5,
            max_turn_rate_rad_s=0.05,
        ),
        sonar=SonarCapability(
            passive_range_m=3000.0,
            passive_bearing_variance_rad2=0.02,
            active_source_range_m=3500.0,
            active_receive_range_m=2800.0,
            active_range_sigma_m=20.0,
            active_bearing_sigma_rad=0.03,
            active_capable=False,
            ping_cooldown_s=30,
            ping_energy_cost_fraction=0.01,
            clutter_sensitivity=0.2,
            exposure_cost=0.3,
        ),
        communications=CommunicationCapability(
            surface_range_m=1000.0,
            acoustic_range_m=4000.0,
        ),
    )

    situation = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=4,
        sim_time_s=50,
        uuvs=(),
        group_reports=(),
        pending_events=(),
        platform_snapshot=PlatformSnapshot(
            scenario_id="S1",
            sim_time_s=50,
            carrier=CarrierPlatformState(
                carrier_id="carrier-01",
                position_xy=(0.0, 0.0),
                heading_rad=0.0,
                speed_mps=2.0,
                support_radius_m=7000.0,
                onboard_platform_ids=(),
                deployed_platform_ids=("uuv-01",),
                returning_platform_ids=(),
            ),
            roster=PlatformRoster(
                usvs=(),
                uuvs=(
                    UUVPlatformState(
                        platform_id="uuv-01",
                        platform_index=0,
                        position_xy=(200.0, 0.0),
                        heading_rad=0.0,
                        speed_mps=2.0,
                        energy_fraction=0.6,
                        deployment_state="deployed",
                        capability=capability,
                        group_id="G-T1",
                        sensor_mode="passive",
                        is_group_leader=False,
                        master_connected=True,
                    ),
                ),
            ),
            communication_links=(
                CommunicationLink(
                    source_id="carrier-01",
                    target_id="uuv-01",
                    medium="acoustic",
                    distance_m=100.0,
                ),
            ),
        ),
    )
    node = StrategyGenerationNode(
        cast(StructuredLLM[StrategyProposal], object()),
        snapshot_provider=lambda _: PlanningSnapshot(situation, None, ()),
    )

    payload = node.build_payload(
        {"scenario_id": "S1", "snapshot_ref": "snapshot:4"}, "balanced"
    )
    summary = payload["decision_factors"]["capability_summary"]

    assert summary["carrier"]["support_radius_m"] == 7000.0
    uuv = summary["by_kind"]["uuv"]["platforms"][0]
    assert [platform["platform_id"] for platform in summary["platforms"]] == ["uuv-01"]
    assert uuv["passive_range_m"] == 3000.0
    assert uuv["active_available"] is False
    assert uuv["master_connected"] is True
    assert uuv["sensor_mode"] == "passive"
    assert summary["by_kind"]["uuv"]["aggregate"]["energy_fraction"]["mean"] == 0.6
    assert summary["communication_links"] == [
        {"source_id": "carrier-01", "target_id": "uuv-01", "medium": "acoustic", "distance_m": 100.0},
    ]


def test_strategy_payload_keeps_legacy_snapshot_capability_shape() -> None:
    situation = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=0,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )
    node = StrategyGenerationNode(
        cast(StructuredLLM[StrategyProposal], object()),
        snapshot_provider=lambda _: PlanningSnapshot(situation, None, ()),
    )

    summary = node.build_payload(
        {"scenario_id": "S1", "snapshot_ref": "snapshot:1"}, "balanced"
    )["decision_factors"]["capability_summary"]

    assert summary["uuv_count"] == 0
    assert "platforms" not in summary


def test_strategy_prompt_requires_platform_complementarity_and_no_final_geometry() -> None:
    prompt = str(
        StrategyGenerationNode(
            cast(StructuredLLM[StrategyProposal], object())
        ).build_payload({}, "balanced")["system_prompt"]
    ).lower()

    for required in (
        "uuv passive/active sonar",
        "active sonar",
        "connectivity",
        "support radius",
        "energy",
        "deployment state",
        "never output final group members",
        "waypoints",
    ):
        assert required in prompt


class _SuggestionLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        del payload, prompt_version
        self.calls.append(operation)
        if response_model is StrategyProposal:
            return StrategyProposal(
                concept="balanced",
                target_priorities={"T1": 1.0},
                required_quality={"T1": 0.8},
                reinforcement_policy={"T1": "maintain"},
                releasable_soft_constraints=(),
                evidence_ids=("E1",),
                rationale="Maintain the current estimated track.",
            )
        if response_model is PlanAdjustmentSuggestionSet:
            categories = (
                "tracking_quality",
                "segmented_handoff",
                "resource_rotation",
                "commander_preference",
            )
            return PlanAdjustmentSuggestionSet(
                suggestions=tuple(
                    PlanAdjustmentSuggestion(
                        suggestion_id=f"suggestion-{index}",
                        category=category,
                        title=f"Suggestion {index}",
                        rationale="The current observation packet supports this option.",
                        proposed_feedback=f"Please consider suggestion {index}.",
                        target_ids=("T1",),
                        evidence_ids=("E1",),
                        confidence=0.8,
                    )
                    for index, category in enumerate(categories, start=1)
                )
            )
        raise AssertionError(f"unexpected response model {response_model!r}")


def test_strategy_generation_publishes_four_llm_suggestions_from_current_observation() -> None:
    llm = _SuggestionLLM()
    node = StrategyGenerationNode(llm)
    state = {
        "scenario_id": "S1",
        "route": EventLevel.STRATEGIC,
        "coalesced_events": (
            RuntimeEvent(
                event_id="E1",
                scenario_id="S1",
                sim_time_s=30,
                event_type="target_added",
                entity_id="T1",
                level=EventLevel.STRATEGIC,
                payload={},
            ),
        ),
        "intent_hypotheses": {
            "T1": IntentHypothesis(
                label="transit",
                confidence=0.8,
                evidence_ids=("E1",),
                model_id="model",
                prompt_version="intent-v1",
            )
        },
    }

    result = node(state)

    assert llm.calls == ["strategy", "strategy", "strategy", "plan_adjustment_suggestions"]
    assert len(result["plan_adjustment_suggestions"]) == 4
    assert "knowledge_query_ids" not in result
    assert result["llm_provenance"]["plan_adjustment_suggestions"].operation == (
        "plan_adjustment_suggestions"
    )


def test_platform_capability_aggregate_handles_zero_ping_energy_cost() -> None:
    aggregate = _platform_aggregate(
        [
            {
                "passive_range_m": 1000.0,
                "active_source_range_m": 1000.0,
                "active_receive_range_m": 1000.0,
                "bearing_quality": {
                    "passive_variance_rad2": 0.1,
                    "active_sigma_rad": 0.1,
                },
                "speed_mps": 1.0,
                "max_speed_mps": 2.0,
                "max_turn_rate_rad_s": 0.1,
                "energy_fraction": 0.5,
                "endurance_s": None,
                "surface_communication_range_m": 1000.0,
                "acoustic_communication_range_m": 1000.0,
                "distance_to_carrier_m": None,
                "passive_available": True,
                "active_available": True,
                "operational_available": True,
                "deployment_state": "deployed",
                "sensor_mode": "passive",
            }
        ]
    )

    assert aggregate["endurance_s"] == {"minimum": 0.0, "maximum": 0.0, "mean": 0.0}
