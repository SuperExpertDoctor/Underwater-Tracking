# tests/agent/test_verify_graph.py
"""Bounded semantic Verify subgraph tests (spec 8.3, plan Task 6).

Covers the brief's verbatim repair-limit test (two repairs, then degrade to
the last valid strategy), the valid-first-pass and repaired-once cases, the
deterministic emergency fallback when no last-valid strategy exists, the
repair payload pinning the true original candidate, and the per-rule
semantic validators.

Per the user directive (addendum A) no mock substitutes real LLM
functionality: the repair-loop and payload tests drive the deterministic
nodes with explicit candidates (the LLM is not part of the unit under
test), and the test whose subject IS LLM behavior — live repair rounds —
runs against the real LongCat provider. The transport-retry independence
tests (injected stub transports / mock exceptions) were deleted as an
accepted consequence; transport retry classification is exercised only
against the real provider in ``tests/integration/test_llm_real_api.py``.
The whole module is skipped when the API key is unset.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from underwater_tracking.agent.graphs.verify import build_verify_graph
from underwater_tracking.agent.nodes.verify import (
    FallbackNode,
    RepairNode,
    ValidateNode,
    VerifyContext,
    VerifyState,
    route_validity,
    validate_strategy,
)
from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    StrategyProposal,
    ValidationIssue,
)
from underwater_tracking.persistence.ledger import DecisionLedger
from tests.conftest import make_live_llm

pytestmark = pytest.mark.skipif(
    not os.environ.get("UNDERWATER_TRACKING_API_KEY"),
    reason="UNDERWATER_TRACKING_API_KEY is not set; the live LongCat API tests are skipped",
)

TARGETS = ("T1",)
EVIDENCE = ("B:T1:900",)
ALLOWED_SOFT_CONSTRAINTS = ("energy_reserve_0.1",)

VALID_STRATEGY_PROPOSAL = {
    "concept": "balanced",
    "target_priorities": {"T1": 1.0},
    "required_quality": {"T1": 0.7},
    "reinforcement_policy": {"T1": "release_when_stable"},
    "releasable_soft_constraints": ["energy_reserve_0.1"],
    "evidence_ids": ["B:T1:900"],
    "rationale": "balanced coverage keeps standby UUVs fresh",
}

# Schema-valid but semantically invalid: priorities/quality/policy name the
# unknown target T9 and omit T1, so every validation round fails on the
# semantic checks (never on schema), keeping the repair loop deterministic.
INVALID_CANDIDATE = {
    **VALID_STRATEGY_PROPOSAL,
    "target_priorities": {"T9": 1.0},
    "required_quality": {"T9": 0.7},
    "reinforcement_policy": {"T9": "release_when_stable"},
}


def _issue_key(issue: ValidationIssue) -> tuple[str, str, str, str]:
    """Deterministic sort key, matching the validator's issue ordering."""
    return (issue.code, issue.field, issue.message, str(issue.observed))


def run_verify_loop(
    *,
    candidate: dict[str, object],
    repairs: Sequence[dict[str, object]],
    last_valid_strategy: StrategyProposal | None,
    max_repairs: int = 2,
) -> VerifyState:
    """Drive the bounded Verify wiring with explicit candidates (no LLM).

    Mirrors the compiled subgraph's edges (validate -> repair -> validate
    -> ... -> fallback) using the deterministic nodes; the LLM is not part
    of this unit, so each repair round's candidate is fed explicitly.
    """
    context = VerifyContext(
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
    )
    validate = ValidateNode(context)
    fallback = FallbackNode(context)
    state: VerifyState = {
        "candidate": candidate,
        "attempt": 0,
        "max_repairs": max_repairs,
        "last_valid_strategy": last_valid_strategy,
    }
    for round_index in range(max_repairs + 1):
        state = {**state, **validate(state)}
        route = route_validity(state)
        if route == "end":
            return state
        if route == "fallback":
            return {**state, **fallback(state)}
        state = {
            **state,
            "candidate": repairs[round_index],
            "attempt": round_index + 1,
        }
    raise AssertionError("the bounded verify loop did not terminate")


def test_verify_repairs_twice_then_degrades():
    result = run_verify_loop(
        candidate=dict(INVALID_CANDIDATE),
        repairs=(dict(INVALID_CANDIDATE), dict(INVALID_CANDIDATE)),
        last_valid_strategy=StrategyProposal.model_validate(VALID_STRATEGY_PROPOSAL),
    )
    assert result["repair_attempts"] == 2
    assert result["verified_strategy"] == StrategyProposal.model_validate(
        VALID_STRATEGY_PROPOSAL
    )
    assert result["degraded"] is True


def test_verify_valid_candidate_passes_first_attempt():
    result = run_verify_loop(
        candidate=dict(VALID_STRATEGY_PROPOSAL),
        repairs=(),
        last_valid_strategy=None,
    )
    assert result["repair_attempts"] == 0
    assert result["verified_strategy"] == StrategyProposal.model_validate(
        VALID_STRATEGY_PROPOSAL
    )
    assert result["degraded"] is False


def test_verify_repairs_once_to_valid():
    result = run_verify_loop(
        candidate=dict(INVALID_CANDIDATE),
        repairs=(dict(VALID_STRATEGY_PROPOSAL),),
        last_valid_strategy=None,
    )
    assert result["repair_attempts"] == 1
    assert result["verified_strategy"] == StrategyProposal.model_validate(
        VALID_STRATEGY_PROPOSAL
    )
    assert result["degraded"] is False


def test_verified_strategy_never_carries_an_invalid_object():
    def run() -> VerifyState:
        return run_verify_loop(
            candidate=dict(INVALID_CANDIDATE),
            repairs=(dict(INVALID_CANDIDATE), dict(INVALID_CANDIDATE)),
            last_valid_strategy=None,
        )

    result = run()
    verified = result["verified_strategy"]
    assert isinstance(verified, StrategyProposal)
    # The exhausted budget with no last-valid strategy degrades to the
    # deterministic emergency strategy, which must itself be fully valid.
    assert verified.concept == "quality_first"
    assert verified.target_priorities == {"T1": 1.0}
    assert verified.required_quality == {"T1": 0.7}
    assert verified.releasable_soft_constraints == ()
    assert verified.evidence_ids == ("B:T1:900",)
    report = validate_strategy(
        verified,
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
    )
    assert report.valid is True
    # The emergency fallback is deterministic across repeated invocations.
    again = run()
    assert again["verified_strategy"] == verified
    assert again["degraded"] is True


def test_validate_strategy_accepts_a_complete_proposal():
    report = validate_strategy(
        StrategyProposal.model_validate(VALID_STRATEGY_PROPOSAL),
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
    )
    assert report.valid is True
    assert report.issues == ()


def test_validate_strategy_flags_schema_invalid_candidate():
    report = validate_strategy(
        {"concept": "balanced"},
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
    )
    assert report.valid is False
    assert {issue.code for issue in report.issues} == {"schema_invalid"}


def test_validate_strategy_flags_every_semantic_rule():
    directive = ExpertDirective(
        directive_id="D1",
        raw_text="keep T1 quality above 0.9",
        target_scope=("T1",),
        target_priorities={"T1": 0.5},
        minimum_quality={"T1": 0.9},
        confidence=0.9,
        status="applied",
    )
    candidate = {
        "concept": "balanced",
        "target_priorities": {"T1": 1.0, "T9": float("nan")},
        "required_quality": {"T9": 0.7},
        "reinforcement_policy": {"T1": "release_when_stable"},
        "releasable_soft_constraints": ["release_uuv3"],
        "evidence_ids": ["B:T1:999"],
        "rationale": "assign member U3 to keep T1 stable",
    }
    report = validate_strategy(
        candidate,
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
        expert_directive=directive,
    )
    assert report.valid is False
    assert {issue.code for issue in report.issues} == {
        "unknown_target",
        "non_finite",
        "missing_coverage",
        "unknown_evidence",
        "disallowed_soft_constraint",
        "member_or_waypoint",
        "expert_constraint_violation",
    }
    assert report.issues == tuple(sorted(report.issues, key=_issue_key))


def test_repair_payload_keeps_true_original_candidate_across_rounds(live_llm):
    # Round 1 returns a DIFFERENT proposal (only the concept changes), so the
    # round-2 payload can tell the pinned original apart from the candidate
    # under repair. ``build_payload`` is pure — the client is never invoked.
    context = VerifyContext(
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
    )
    repair = RepairNode(live_llm, context=context)
    round_one = {**INVALID_CANDIDATE, "concept": "quality_first"}
    issues = validate_strategy(
        INVALID_CANDIDATE,
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
    ).issues
    # Round 1: both payload fields are the round-0 original.
    first = repair.build_payload(
        {"candidate": dict(INVALID_CANDIDATE), "attempt": 0}, context, issues
    )
    assert first["original_candidate"] == INVALID_CANDIDATE
    assert first["candidate"] == INVALID_CANDIDATE
    # Round 2: original_candidate is STILL the round-0 original, while
    # candidate is round 1's output.
    second = repair.build_payload(
        {
            "original_candidate": dict(INVALID_CANDIDATE),
            "candidate": round_one,
            "attempt": 1,
        },
        context,
        issues,
    )
    assert second["original_candidate"] == INVALID_CANDIDATE
    assert second["candidate"] == round_one


def test_evidence_ids_embedding_uuv_ids_are_not_member_markers():
    # Real observation ids embed the producing UUV id; citing them must not
    # be treated as smuggling final members or waypoints.
    evidence = ("B:T1:uuv_00:900",)
    candidate = {**VALID_STRATEGY_PROPOSAL, "evidence_ids": ["B:T1:uuv_00:900"]}
    report = validate_strategy(
        candidate,
        target_ids=TARGETS,
        evidence_ids=evidence,
        allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
    )
    assert report.valid is True
    assert "member_or_waypoint" not in {issue.code for issue in report.issues}


def test_member_marker_smuggled_in_structural_field_is_rejected():
    # A member id smuggled into a structural field is still flagged.
    smuggled = {
        **VALID_STRATEGY_PROPOSAL,
        "reinforcement_policy": {"T1": "assign_uuv_03_when_unstable"},
    }
    report = validate_strategy(
        smuggled,
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
    )
    assert report.valid is False
    assert {issue.code for issue in report.issues} == {"member_or_waypoint"}


def test_verify_graph_accepts_candidate_citing_uuv_produced_evidence(live_llm):
    # Valid first pass: the graph never invokes the client.
    graph = build_verify_graph(
        live_llm,
        target_ids=TARGETS,
        evidence_ids=("B:T1:uuv_00:900",),
        allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
    )
    result = graph.invoke({
        "candidate": {**VALID_STRATEGY_PROPOSAL, "evidence_ids": ["B:T1:uuv_00:900"]},
        "attempt": 0,
        "max_repairs": 2,
        "last_valid_strategy": None,
    })
    assert result["repair_attempts"] == 0
    assert result["degraded"] is False
    assert result["verified_strategy"] == StrategyProposal.model_validate(
        {**VALID_STRATEGY_PROPOSAL, "evidence_ids": ["B:T1:uuv_00:900"]}
    )


# --- Live semantic repair rounds (subject IS LLM behavior) ------------------


@pytest.mark.real_llm
def test_verify_live_repair_rounds_keep_valid_schema(tmp_path: Path):
    """Live bounded repairs: whatever the provider returns, the outcome validates.

    The repair budget bounds the requests; every outcome — a repaired
    proposal or the deterministic emergency fallback — must validate under
    the same rules, and the ledger records the calls.
    """
    ledger = DecisionLedger(tmp_path / "verify.db")
    client = make_live_llm(ledger=ledger, scenario_id="S1", sim_time_s=900)
    try:
        graph = build_verify_graph(
            client,
            target_ids=TARGETS,
            evidence_ids=EVIDENCE,
            allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
        )
        result = graph.invoke({
            "candidate": dict(INVALID_CANDIDATE),
            "attempt": 0,
            "max_repairs": 2,
            "last_valid_strategy": None,
        })
        verified = result["verified_strategy"]
        assert isinstance(verified, StrategyProposal)
        report = validate_strategy(
            verified,
            target_ids=TARGETS,
            evidence_ids=EVIDENCE,
            allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
        )
        assert report.valid is True
        assert isinstance(result["degraded"], bool)
        assert result["repair_attempts"] in (0, 1, 2)
        assert ledger.list_llm_calls()
    finally:
        client.close()
