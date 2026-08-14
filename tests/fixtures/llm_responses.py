# tests/fixtures/llm_responses.py
"""Canonical structured LLM response dictionaries for agent tests (spec 22).

These are the deterministic responses served by the Mock LLM and the stub
HTTP transport so semantic-node tests (Task 5+) exercise the same response
shapes everywhere. Every valid payload carries its required provenance
fields: ``model_id``/``prompt_version`` for ``IntentHypothesis``, and
``releasable_soft_constraints``/``rationale`` for ``StrategyProposal``
(pre-flight ruling #3 — the Mock fixtures for later tasks must supply them).
"""

VALID_INTENT_HYPOTHESIS = {
    "label": "transit",
    "confidence": 0.8,
    "evidence_ids": ["B:T1:900"],
    "model_id": "mock",
    "prompt_version": "intent-v1",
}

EVADING_INTENT_HYPOTHESIS = {
    "label": "evade",
    "confidence": 0.75,
    "evidence_ids": ["B:T1:900"],
    "model_id": "mock",
    "prompt_version": "intent-v1",
}

INVALID_INTENT_HYPOTHESIS = {
    "label": "disco",
    "confidence": 0.8,
    "evidence_ids": ["B:T1:900"],
    "model_id": "mock",
    "prompt_version": "intent-v1",
}

VALID_STRATEGY_PROPOSAL = {
    "concept": "balanced",
    "target_priorities": {"T1": 1.0},
    "required_quality": {"T1": 0.7},
    "reinforcement_policy": {"T1": "release_when_stable"},
    "releasable_soft_constraints": ["energy_reserve_0.1"],
    "evidence_ids": ["B:T1:900"],
    "rationale": "balanced coverage keeps standby UUVs fresh",
}

INVALID_STRATEGY_PROPOSAL = {
    "concept": "balanced",
    "target_priorities": {"T1": 1.0},
    "required_quality": {"T1": 0.7},
    "reinforcement_policy": {"T1": "release_when_stable"},
    "evidence_ids": ["B:T1:900"],
}
