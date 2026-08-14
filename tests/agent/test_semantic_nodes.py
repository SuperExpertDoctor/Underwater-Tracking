# tests/agent/test_semantic_nodes.py
"""Versioned-prompt and semantic-node boundary tests (spec 12.2, 15.2, 16).

Covers the brief's two verbatim tests (intent payloads carry estimated
trajectory features, never ground reality; strategic events request exactly
the three candidate concepts), plus the node contracts: curated payloads,
sorted evidence ids, deterministic payloads, per-call provenance attached
to state, multi-target intent analysis, and the periodic-review
``hold_current`` path. All LLM responses come from the deterministic
MockStructuredLLM queue keyed by operation.
"""

import pytest

from underwater_tracking.agent.llm import MockStructuredLLM
from underwater_tracking.agent.nodes.intent import IntentAnalysisNode
from underwater_tracking.agent.nodes.strategy import StrategyGenerationNode
from underwater_tracking.agent.prompts import (
    DIRECTIVE_SYSTEM_PROMPT,
    EXPLANATION_SYSTEM_PROMPT,
    INTENT_PROMPT_VERSION,
    INTENT_SYSTEM_PROMPT,
    STRATEGY_PROMPT_VERSION,
    STRATEGY_SYSTEM_PROMPT,
    canonical_digest,
)
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import IntentHypothesis
from underwater_tracking.domain.models import (
    EventLevel,
    GroupQuality,
    GroupReport,
    RuntimeEvent,
    SituationSnapshot,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from tests.fixtures.llm_responses import (
    EVADING_INTENT_HYPOTHESIS,
    VALID_INTENT_HYPOTHESIS,
    VALID_STRATEGY_PROPOSAL,
)

# Downsampled estimated trajectories (sim_time_s, x, y), strictly increasing
# in time and consistent with each snapshot's latest belief.
T1_HISTORY = (
    (600, 80.0, 150.0),
    (660, 90.0, 170.0),
    (720, 100.0, 190.0),
    (780, 110.0, 205.0),
    (840, 120.0, 215.0),
    (900, 130.0, 220.0),
)
T2_HISTORY = (
    (600, 200.0, 40.0),
    (660, 205.0, 55.0),
    (720, 210.0, 70.0),
    (780, 218.0, 82.0),
    (840, 228.0, 90.0),
    (900, 240.0, 60.0),
)

_POSITIONS = {"T1": (130.0, 220.0), "T2": (240.0, 60.0)}

QUALITY_FIRST = {**VALID_STRATEGY_PROPOSAL, "concept": "quality_first"}
RESOURCE_SAVING = {**VALID_STRATEGY_PROPOSAL, "concept": "resource_saving"}
HOLD_CURRENT = {**VALID_STRATEGY_PROPOSAL, "concept": "hold_current"}


def make_snapshot(*target_ids: str) -> SituationSnapshot:
    """SituationSnapshot with one group report per target at t=900 s."""
    reports = tuple(
        GroupReport(
            group_id=f"G-{target_id}",
            target_id=target_id,
            sim_time_s=900,
            member_ids=("U1", "U2"),
            belief=TargetBelief(
                target_id=target_id,
                sim_time_s=900,
                mean=(*_POSITIONS.get(target_id, (130.0, 220.0)), 1.0, 0.5),
                covariance=(
                    (400.0, 0.0, 0.0, 0.0),
                    (0.0, 400.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                model_probabilities={"cv": 0.7, "ct": 0.3},
                source_observation_ids=(
                    f"B:{target_id}:900",
                    f"B:{target_id}:870",
                ),
                fim_min_eigenvalue=0.005,
                fim_condition=12.0,
            ),
            quality=GroupQuality(
                instant=0.8,
                window_mean=0.75,
                ewma=0.76,
                components={"cov": 0.7},
                hard_guard_reasons=(),
            ),
            plan_revision=1,
        )
        for target_id in target_ids
    )
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=3,
        sim_time_s=900,
        uuvs=(
            UUVState(
                uuv_id="U1",
                position_xy=(0.0, 0.0),
                heading_rad=0.0,
                speed_mps=2.0,
                energy_fraction=0.8,
                status=UUVStatus.TRACKING,
                group_id="G-T1",
            ),
        ),
        group_reports=reports,
        pending_events=(),
    )


@pytest.fixture
def snapshot() -> SituationSnapshot:
    return make_snapshot("T1")


@pytest.fixture
def intent_node() -> IntentAnalysisNode:
    llm = MockStructuredLLM(
        {"intent": [VALID_INTENT_HYPOTHESIS, EVADING_INTENT_HYPOTHESIS]}
    )
    return IntentAnalysisNode(
        llm,
        model_id="mock",
        belief_history=lambda snap, target_id: T1_HISTORY,
        snapshot_provider=lambda snapshot_ref: make_snapshot("T1"),
    )


@pytest.fixture
def strategic_state() -> CarrierState:
    return {
        "scenario_id": "S1",
        "snapshot_revision": 3,
        "snapshot_ref": "snap:3",
        "route": EventLevel.STRATEGIC,
        "coalesced_events": (
            RuntimeEvent(
                event_id="E:target_added:900",
                scenario_id="S1",
                sim_time_s=900,
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
                evidence_ids=("B:T1:900",),
                model_id="mock",
                prompt_version="intent-v1",
            ),
        },
        "predictions": {},
    }


@pytest.fixture
def strategy_node() -> StrategyGenerationNode:
    llm = MockStructuredLLM(
        {"strategy": [QUALITY_FIRST, VALID_STRATEGY_PROPOSAL, RESOURCE_SAVING]}
    )
    return StrategyGenerationNode(llm, model_id="mock")


def test_intent_payload_uses_history_features_not_truth(intent_node, snapshot):
    payload = intent_node.build_payload(snapshot, target_id="T1")
    assert "truth" not in repr(payload).lower()
    assert payload["trajectory_features"]
    assert payload["sampled_belief_history"]


def test_major_event_requests_three_concepts(strategy_node, strategic_state):
    result = strategy_node(strategic_state)
    assert {item.concept for item in result["strategy_set"]} == {
        "quality_first", "balanced", "resource_saving"
    }


def test_intent_payload_is_curated_and_sorts_evidence_ids(intent_node, snapshot):
    payload = intent_node.build_payload(snapshot, target_id="T1")
    assert payload["evidence_ids"] == ["B:T1:870", "B:T1:900"]
    assert payload["sim_time_s"] == 900
    assert "uuvs" not in payload
    assert "group_reports" not in payload
    assert "covariance" not in repr(payload).lower()


def test_intent_payload_is_deterministic(intent_node, snapshot):
    assert intent_node.build_payload(snapshot, target_id="T1") == intent_node.build_payload(
        snapshot, target_id="T1"
    )


def test_intent_node_loops_over_targets_and_attaches_provenance(intent_node, snapshot):
    result = intent_node({"scenario_id": "S1", "snapshot_ref": "snap:3"})
    hypothesis = result["intent_hypotheses"]["T1"]
    assert hypothesis.label == "transit"
    assert hypothesis.model_id == "mock"
    assert hypothesis.prompt_version == "intent-v1"
    metadata = result["llm_provenance"]["intent:T1"]
    assert metadata.operation == "intent"
    assert metadata.model == "mock"
    assert metadata.prompt_version == "intent-v1"
    assert metadata.scenario_id == "S1"
    assert metadata.sim_time_s == 900
    expected_payload = intent_node.build_payload(snapshot, target_id="T1")
    assert metadata.request_hash == canonical_digest(expected_payload)
    assert metadata.response_hash


def test_intent_node_analyzes_each_snapshot_target():
    snapshot = make_snapshot("T1", "T2")
    llm = MockStructuredLLM(
        {"intent": [VALID_INTENT_HYPOTHESIS, EVADING_INTENT_HYPOTHESIS]}
    )
    node = IntentAnalysisNode(
        llm,
        model_id="mock",
        belief_history=lambda snap, target_id: (
            T1_HISTORY if target_id == "T1" else T2_HISTORY
        ),
        snapshot_provider=lambda snapshot_ref: snapshot,
    )
    result = node({"scenario_id": "S1", "snapshot_ref": "snap:3"})
    assert set(result["intent_hypotheses"]) == {"T1", "T2"}
    assert result["intent_hypotheses"]["T1"].label == "transit"
    assert result["intent_hypotheses"]["T2"].label == "evade"
    assert result["intent_hypotheses"]["T1"].evidence_ids == ("B:T1:900",)
    assert set(result["llm_provenance"]) == {"intent:T1", "intent:T2"}


def test_strategy_payload_is_curated_and_sorted(strategy_node, strategic_state):
    payload = strategy_node.build_payload(strategic_state, "quality_first")
    assert payload["requested_concept"] == "quality_first"
    assert payload["mode"] == "strategic"
    assert payload["evidence_ids"] == ["B:T1:900"]
    assert payload["targets"][0]["target_id"] == "T1"
    assert "truth" not in repr(payload).lower()


def test_strategy_attaches_provenance_per_concept(strategy_node, strategic_state):
    result = strategy_node(strategic_state)
    assert set(result["llm_provenance"]) == {
        "strategy:quality_first", "strategy:balanced", "strategy:resource_saving"
    }
    metadata = result["llm_provenance"]["strategy:quality_first"]
    assert metadata.operation == "strategy"
    assert metadata.model == "mock"
    assert metadata.prompt_version == STRATEGY_PROMPT_VERSION
    assert metadata.request_hash == canonical_digest(
        strategy_node.build_payload(strategic_state, "quality_first")
    )
    assert metadata.response_hash
    assert result["strategy_set"].trigger_event_ids == ("E:target_added:900",)


def test_periodic_review_requests_hold_current():
    llm = MockStructuredLLM({"strategy": HOLD_CURRENT})
    node = StrategyGenerationNode(llm, model_id="mock")
    state: CarrierState = {
        "scenario_id": "S1",
        "snapshot_revision": 5,
        "route": EventLevel.INFORMATIONAL,
        "coalesced_events": (
            RuntimeEvent(
                event_id="E:review:1500",
                scenario_id="S1",
                sim_time_s=1500,
                event_type="state_changed",
                entity_id="S1",
                level=EventLevel.INFORMATIONAL,
                payload={},
            ),
        ),
        "intent_hypotheses": {
            "T1": IntentHypothesis(
                label="transit",
                confidence=0.8,
                evidence_ids=("B:T1:900",),
                model_id="mock",
                prompt_version="intent-v1",
            ),
        },
    }
    result = node(state)
    assert [p.concept for p in result["strategy_set"].proposals] == ["hold_current"]
    assert result["strategy_set"].trigger_event_ids == ("E:review:1500",)
    assert result["llm_provenance"]["strategy:hold_current"].prompt_version == (
        STRATEGY_PROMPT_VERSION
    )


def test_all_system_prompts_declare_boundaries():
    for prompt in (
        INTENT_SYSTEM_PROMPT,
        STRATEGY_SYSTEM_PROMPT,
        DIRECTIVE_SYSTEM_PROMPT,
        EXPLANATION_SYSTEM_PROMPT,
    ):
        # The no-truth rule must never leak the hidden-ground-reality word
        # into the prompt text (payload reprs must stay clean of it).
        assert "truth" not in prompt.lower()
        assert "never" in prompt.lower()
        assert "evidence" in prompt.lower()
        assert "waypoint" in prompt.lower()


def test_prompt_version_constants_match_llm_fixtures(intent_node, snapshot):
    assert INTENT_PROMPT_VERSION == "intent-v1"
    assert VALID_INTENT_HYPOTHESIS["prompt_version"] == INTENT_PROMPT_VERSION
    hypothesis = intent_node.build_payload(snapshot, target_id="T1")
    assert hypothesis["system_prompt"] == INTENT_SYSTEM_PROMPT
