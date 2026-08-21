# tests/agent/test_semantic_nodes.py
"""Versioned-prompt and semantic-node boundary tests (spec 12.2, 15.2, 16).

Covers the brief's two verbatim tests (intent payloads carry estimated
trajectory features, never ground reality; strategic events request exactly
the three candidate concepts), plus the node contracts: curated payloads,
sorted evidence ids, deterministic payloads, per-call provenance attached
to state, multi-target intent analysis, and the periodic-review
``hold_current`` path.

Payload and prompt tests are pure node logic (the LLM is not part of the
unit under test: the client is constructed but never invoked). Tests whose
subject IS LLM behavior — intent/strategy semantic outputs and their
provenance — run live against the real LongCat provider and skip when the
API key is unset (per the user directive, addendum A: no mock substitutes
real LLM functionality anywhere).
"""

import pytest

from underwater_tracking.agent.llm import HTTPStructuredLLM
from underwater_tracking.agent.nodes.intent import IntentAnalysisNode
from underwater_tracking.agent.nodes.strategy import StrategyGenerationNode
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
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
from tests.conftest import (
    REAL_LLM_SKIP_REASON,
    has_live_api_key,
)

pytestmark = pytest.mark.skipif(
    not has_live_api_key(),
    reason=REAL_LLM_SKIP_REASON,
)

# The intent label set enforced by the model's schema (variance-robust
# assertion target).
INTENT_LABELS = ("transit", "patrol", "loiter", "evade", "approach", "withdraw", "unknown")
STRATEGY_CONCEPTS = ("quality_first", "balanced", "resource_saving", "hold_current")

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
def intent_node(live_llm: HTTPStructuredLLM) -> IntentAnalysisNode:
    """Intent node over a real client that these tests never invoke."""
    return IntentAnalysisNode(
        live_llm,
        model_id="LongCat-2.0",
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
                model_id="LongCat-2.0",
                prompt_version=INTENT_PROMPT_VERSION,
            ),
        },
        "predictions": {},
    }


def test_intent_payload_uses_history_features_not_truth(intent_node, snapshot):
    payload = intent_node.build_payload(snapshot, target_id="T1")
    assert "truth" not in repr(payload).lower()
    assert payload["trajectory_features"]
    assert payload["sampled_belief_history"]


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


def test_strategy_payload_is_curated_and_sorted(live_llm, strategic_state):
    node = StrategyGenerationNode(live_llm, model_id="LongCat-2.0")
    payload = node.build_payload(strategic_state, "quality_first")
    assert payload["requested_concept"] == "quality_first"
    assert payload["mode"] == "strategic"
    assert payload["evidence_ids"] == ["B:T1:900", "E:target_added:900"]
    assert payload["targets"][0]["target_id"] == "T1"
    assert "truth" not in repr(payload).lower()


def test_strategy_payload_exposes_bounded_decision_factors(live_llm, strategic_state):
    snapshot = make_snapshot("T1")
    node = StrategyGenerationNode(
        live_llm,
        model_id="LongCat-2.0",
        snapshot_provider=lambda ref: PlanningSnapshot(snapshot, None, ()),
    )
    payload = node.build_payload(strategic_state, "balanced")
    factors = payload["decision_factors"]
    assert factors["target_quality"]["T1"]["fim_min_eigenvalue"] == 0.005
    assert factors["resource_summary"]["available_count"] == 1
    assert "member_ids" not in repr(factors)
    assert "waypoint" not in repr(factors).lower()


def test_prompt_version_constants_and_payload_prompt(intent_node, snapshot):
    assert INTENT_PROMPT_VERSION == "intent-v1"
    assert STRATEGY_PROMPT_VERSION == "strategy-v2"
    payload = intent_node.build_payload(snapshot, target_id="T1")
    assert payload["system_prompt"] == INTENT_SYSTEM_PROMPT


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


# --- Live semantic tests (subject IS LLM behavior) --------------------------


@pytest.mark.real_llm
def test_intent_node_analyzes_snapshot_targets_and_attaches_provenance(live_llm):
    """Live intent analysis over two targets (2 requests).

    Assertions are variance-robust: the hypothesis must validate, cite only
    evidence from the payload, and carry the configured model identity and
    prompt version with per-call provenance hashing the exact payload.
    """
    snapshot = make_snapshot("T1", "T2")
    node = IntentAnalysisNode(
        live_llm,
        model_id="LongCat-2.0",
        belief_history=lambda snap, target_id: (
            T1_HISTORY if target_id == "T1" else T2_HISTORY
        ),
        snapshot_provider=lambda snapshot_ref: snapshot,
    )
    result = node({"scenario_id": "S1", "snapshot_ref": "snap:3"})
    assert set(result["intent_hypotheses"]) == {"T1", "T2"}
    for target_id in ("T1", "T2"):
        hypothesis = result["intent_hypotheses"][target_id]
        assert isinstance(hypothesis, IntentHypothesis)
        assert hypothesis.label in INTENT_LABELS
        assert 0.0 <= hypothesis.confidence <= 1.0
        # The schema requires model_id/prompt_version strings; the values
        # are model-written (the prompts never instruct them), so only
        # non-empty sanity is asserted here — the deterministic node-side
        # identity lives in the provenance metadata below.
        assert hypothesis.model_id
        assert hypothesis.prompt_version
        assert hypothesis.evidence_ids
        payload = node.build_payload(snapshot, target_id=target_id)
        assert set(hypothesis.evidence_ids) <= set(payload["evidence_ids"])
        metadata = result["llm_provenance"][f"intent:{target_id}"]
        assert metadata.operation == "intent"
        assert metadata.model == "LongCat-2.0"
        assert metadata.prompt_version == INTENT_PROMPT_VERSION
        assert metadata.scenario_id == "S1"
        assert metadata.sim_time_s == 900
        assert metadata.request_hash == canonical_digest(payload)
        assert metadata.response_hash
    assert set(result["llm_provenance"]) == {"intent:T1", "intent:T2"}


@pytest.mark.real_llm
def test_strategic_event_requests_three_concepts_with_provenance(
    live_llm, strategic_state
):
    """Live strategic generation requests exactly three concepts (3 requests).

    The response set must contain one validated proposal per requested
    concept, each citing only payload evidence, with the deterministic
    provenance keys and trigger event ids.
    """
    node = StrategyGenerationNode(live_llm, model_id="LongCat-2.0")
    result = node(strategic_state)
    proposals = result["strategy_set"].proposals
    assert len(proposals) == 3
    for proposal in proposals:
        assert proposal.concept in STRATEGY_CONCEPTS
        assert proposal.evidence_ids
        payload = node.build_payload(strategic_state, proposal.concept)
        assert set(proposal.evidence_ids) <= set(payload["evidence_ids"])
    assert set(result["llm_provenance"]) == {
        "strategy:quality_first",
        "strategy:balanced",
        "strategy:resource_saving",
    }
    for key, metadata in result["llm_provenance"].items():
        assert metadata.operation == "strategy"
        assert metadata.model == "LongCat-2.0"
        assert metadata.prompt_version == STRATEGY_PROMPT_VERSION
        assert metadata.request_hash == canonical_digest(
            node.build_payload(strategic_state, key.split(":", 1)[1])
        )
        assert metadata.response_hash
    assert result["strategy_set"].trigger_event_ids == ("E:target_added:900",)


@pytest.mark.real_llm
def test_periodic_review_requests_hold_current(live_llm):
    """Live periodic review requests exactly the ``hold_current`` concept."""
    node = StrategyGenerationNode(live_llm, model_id="LongCat-2.0")
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
                model_id="LongCat-2.0",
                prompt_version=INTENT_PROMPT_VERSION,
            ),
        },
    }
    result = node(state)
    assert len(result["strategy_set"].proposals) == 1
    assert result["strategy_set"].proposals[0].concept in STRATEGY_CONCEPTS
    assert result["strategy_set"].trigger_event_ids == ("E:review:1500",)
    metadata = result["llm_provenance"]["strategy:hold_current"]
    assert metadata.operation == "strategy"
    assert metadata.prompt_version == STRATEGY_PROMPT_VERSION
    assert metadata.response_hash
