"""Explicit real-provider smoke across every production decision role.

This module is intentionally opt-in through the shared ``real_llm``
collection guard.  It records only typed outputs and hashed ledger metadata;
prompts, responses, and credentials never enter test output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from underwater_tracking.agent.graphs.adversary import build_adversary_graph
from underwater_tracking.agent.graphs.slave import build_slave_graph
from underwater_tracking.agent.llm import HTTPStructuredLLM
from underwater_tracking.agent.nodes.intent import IntentAnalysisNode
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.adversary_models import AdversaryIntentDecision
from underwater_tracking.domain.agent_models import IntentHypothesis
from underwater_tracking.domain.memory_models import (
    MemoryExtractionResult,
    MemoryFilterDecision,
    ShortTermCompressionResult,
    ShortTermContext,
    ShortTermMessage,
)
from underwater_tracking.domain.regional_models import UUVRegionalStrategySet
from underwater_tracking.memory.reasoner import MemoryReasoner
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.memory import LongTermMemoryRepository
from tests.agent.test_adversary_graph import make_context as make_adversary_context
from tests.agent.test_slave_graph import _context as make_slave_context
from tests.integration.test_llm_real_api import T1_HISTORY, _single_target_snapshot
from tests.conftest import CONFIG_PATH, make_live_llm

pytestmark = pytest.mark.real_llm


def _regional_payload() -> dict[str, object]:
    return {
        "system_prompt": (
            "Return one typed UUVRegionalStrategySet policy for candidate-1. "
            "The candidate geometry is locked. Use only candidate-1, UUV-1, "
            "and evidence role-smoke:source."
        ),
        "scenario_id": "role-smoke",
        "sim_time_s": 120,
        "target_id": "target-01",
        "candidate_regions": [
            {
                "candidate_id": "candidate-1",
                "target_id": "target-01",
                "center_xy": [1000.0, 1000.0],
                "cell_size_m": 500.0,
                "first_entry_s": 120,
                "last_exit_s": 600,
                "occupancy_likelihood": 0.8,
                "evidence_ids": ["role-smoke:source"],
            }
        ],
        "platform_candidates": [
            {"platform_id": "UUV-1", "kind": "uuv"},
            {"platform_id": "UUV-2", "kind": "uuv"},
        ],
        "operational_constraints": {
            "candidate_geometry_locked": True,
            "passive_sonar_required": True,
            "allowed_tracking_modes": ["active_scan", "passive_track", "handoff_reserve"],
        },
        "evidence_ids": ["role-smoke:source"],
    }


def test_real_roles_return_typed_outputs_and_record_complete_ledger(
    tmp_path: Path,
) -> None:
    config = load_app_config(CONFIG_PATH)
    assert config.memory is not None
    ledger = DecisionLedger(tmp_path / "role-smoke.db")
    repository = LongTermMemoryRepository(tmp_path / "role-smoke.db")
    llm: HTTPStructuredLLM | None = None
    calls = ()
    try:
        llm = make_live_llm(ledger=ledger, scenario_id="role-smoke")

        # Master planning lane: intent analysis plus the typed regional strategy.
        intent = IntentAnalysisNode(llm, model_id="LongCat-2.0").build_payload(
            _single_target_snapshot(),
            "T1",
            belief_history=T1_HISTORY,
        )
        intent_result = llm.invoke_structured(
            "intent",
            intent,
            IntentHypothesis,
            prompt_version="intent-v1",
        )
        assert intent_result.evidence_ids
        regional_result = llm.invoke_structured(
            "regional_strategy",
            _regional_payload(),
            UUVRegionalStrategySet,
            prompt_version="regional-strategy-v1",
        )
        assert isinstance(regional_result, UUVRegionalStrategySet)
        assert regional_result.policies
        allowed_evidence = {"role-smoke:source"}
        for policy in regional_result.policies:
            assert policy.candidate_id == "candidate-1"
            assert set(policy.evidence_ids) <= allowed_evidence

        # Local group lane: the graph admits only a deployed-group context.
        slave_result = build_slave_graph(llm).invoke({"context": make_slave_context()})
        assert slave_result["decision"].target_id == "target-01"
        assert slave_result["decision"].receiver_ids

        # Target lane: the model returns high-level intent only; the graph
        # validates it against the supplied mission-local evidence.
        adversary_result = build_adversary_graph(llm).invoke(
            {"context": make_adversary_context()}
        )
        assert isinstance(adversary_result["decision"], AdversaryIntentDecision)
        assert adversary_result["decision"].target_id == "SUB-1"

        reasoner = MemoryReasoner(llm=llm, repository=repository, config=config.memory)
        source_text = "Operator prefers concise evidence-backed reporting."
        source_message_id = "role-smoke:message"
        short_term = ShortTermContext(
            user_id="operator",
            scenario_id="role-smoke",
            conversation_id="role-smoke-conversation",
            recent_messages=(
                ShortTermMessage(
                    message_id=source_message_id,
                    scenario_id="role-smoke",
                    role="user",
                    text=source_text,
                ),
            ),
        )
        filter_result = reasoner.filter(
            user_id="operator",
            scenario_id="role-smoke",
            source_texts=(source_text,),
            source_message_ids=(source_message_id,),
            short_term_context=short_term,
        )
        assert isinstance(filter_result, MemoryFilterDecision)
        extract_result = reasoner.extract(
            user_id="operator",
            source_texts=(source_text,),
            source_message_ids=(source_message_id,),
        )
        assert isinstance(extract_result, MemoryExtractionResult)
        assert set(extract_result.source_message_ids) <= {source_message_id}
        compression_result = reasoner.compress_short_term(short_term)
        assert isinstance(compression_result, ShortTermCompressionResult)
        assert set(compression_result.source_message_ids) <= {source_message_id}
        assert all(
            message.message_id == source_message_id
            for message in compression_result.retained_messages
        )
    finally:
        calls = ledger.list_llm_calls(scenario_id="role-smoke", limit=100)
        if llm is not None:
            llm.close()
        repository.close()
        ledger.close()

    operations = {call.operation for call in calls}
    assert {
        "intent",
        "regional_strategy",
        "slave_sonar_decision",
        "adversary_mission_decision",
        "memory_filter",
        "memory_extract",
        # ``short_term_compress`` is the canonical persisted operation name
        # used by the memory contract for the required memory-compress role.
        "short_term_compress",
    } <= operations
    assert all(call.latency_ms > 0 for call in calls)
