"""Strategy payload coverage for executable operational constraints."""

from typing import cast

from underwater_tracking.agent.llm import StructuredLLM
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.agent.nodes.strategy import StrategyGenerationNode
from underwater_tracking.domain.agent_models import IntentHypothesis, StrategyProposal
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
    assert factors["capability_summary"]["passive_range_m"]["minimum"] == 2500.0
    assert factors["required_quality_constraints"] == {"T1": 0.85}
    assert "required decision checklist" in str(payload["system_prompt"]).lower()
