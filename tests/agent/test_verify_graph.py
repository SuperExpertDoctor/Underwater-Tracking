# tests/agent/test_verify_graph.py
"""Bounded semantic Verify subgraph tests (spec 8.3, plan Task 6).

Covers the brief's verbatim repair-limit test (two repairs, then degrade to
the last valid strategy), the valid-first-pass and repaired-once cases, the
deterministic emergency fallback when no last-valid strategy exists, the
transport-retry independence (retries inside the LLM port never count as
semantic repair attempts), and the per-rule semantic validators. All LLM
responses come from the deterministic MockStructuredLLM queue or the stub
HTTP transport.
"""

from types import SimpleNamespace

import httpx
import pytest

from underwater_tracking.agent.graphs.verify import build_verify_graph
from underwater_tracking.agent.llm import (
    HTTPStructuredLLM,
    MockStructuredLLM,
    TransientLLMError,
)
from underwater_tracking.agent.nodes.verify import validate_strategy
from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    StrategyProposal,
    ValidationIssue,
)
from tests.fixtures.llm_responses import VALID_STRATEGY_PROPOSAL

API_KEY_ENV = "UNDERWATER_TRACKING_TEST_KEY"
TEST_BASE_URL = "http://llm.test/v1/chat"
TEST_MODEL = "mock-model"

TARGETS = ("T1",)
EVIDENCE = ("B:T1:900",)
ALLOWED_SOFT_CONSTRAINTS = ("energy_reserve_0.1",)

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


class CountingTransport(httpx.BaseTransport):
    """httpx transport that fails transiently on demand, counting every call."""

    def __init__(self, *, success_json: object) -> None:
        self._success_json = success_json
        self.calls = 0
        self._failures_remaining = 0
        self._error_type: type[TransientLLMError] | None = None

    def fail_with(self, error_type: type[TransientLLMError], times: int) -> None:
        self._error_type = error_type
        self._failures_remaining = times

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            assert self._error_type is not None
            raise self._error_type("injected transient failure")
        return httpx.Response(200, json=self._success_json)


@pytest.fixture
def verify_graph():
    """Verify subgraph whose two repair rounds both return invalid proposals."""
    llm = MockStructuredLLM({"strategy": [INVALID_CANDIDATE, INVALID_CANDIDATE]})
    return build_verify_graph(
        llm,
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
    )


@pytest.fixture
def invalid_strategy_queue() -> SimpleNamespace:
    """The brief's queue: an invalid first candidate and a valid last strategy."""
    return SimpleNamespace(
        first=dict(INVALID_CANDIDATE),
        last_valid=StrategyProposal.model_validate(VALID_STRATEGY_PROPOSAL),
    )


def test_verify_repairs_twice_then_degrades(verify_graph, invalid_strategy_queue):
    result = verify_graph.invoke({
        "candidate": invalid_strategy_queue.first,
        "attempt": 0,
        "max_repairs": 2,
        "last_valid_strategy": invalid_strategy_queue.last_valid,
    })
    assert result["repair_attempts"] == 2
    assert result["verified_strategy"] == invalid_strategy_queue.last_valid
    assert result["degraded"] is True


def test_verify_valid_candidate_passes_first_attempt():
    llm = MockStructuredLLM({})
    graph = build_verify_graph(
        llm,
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
    )
    result = graph.invoke({
        "candidate": dict(VALID_STRATEGY_PROPOSAL),
        "attempt": 0,
        "max_repairs": 2,
        "last_valid_strategy": None,
    })
    assert result["repair_attempts"] == 0
    assert result["verified_strategy"] == StrategyProposal.model_validate(
        VALID_STRATEGY_PROPOSAL
    )
    assert result["degraded"] is False


def test_verify_repairs_once_to_valid():
    llm = MockStructuredLLM({"strategy": [VALID_STRATEGY_PROPOSAL]})
    graph = build_verify_graph(
        llm,
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
    assert result["repair_attempts"] == 1
    assert result["verified_strategy"] == StrategyProposal.model_validate(
        VALID_STRATEGY_PROPOSAL
    )
    assert result["degraded"] is False


def test_verified_strategy_never_carries_an_invalid_object(verify_graph):
    result = verify_graph.invoke({
        "candidate": dict(INVALID_CANDIDATE),
        "attempt": 0,
        "max_repairs": 2,
        "last_valid_strategy": None,
    })
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
    again = verify_graph.invoke({
        "candidate": dict(INVALID_CANDIDATE),
        "attempt": 0,
        "max_repairs": 2,
        "last_valid_strategy": None,
    })
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


def test_verify_transport_retries_do_not_increment_semantic_attempts(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(API_KEY_ENV, "test-token")
    transport = CountingTransport(success_json=VALID_STRATEGY_PROPOSAL)
    transport.fail_with(TransientLLMError, times=2)
    client = HTTPStructuredLLM(
        base_url=TEST_BASE_URL,
        model=TEST_MODEL,
        api_key_env=API_KEY_ENV,
        max_retries=3,
        backoff_base_s=0.001,
        jitter=lambda: 0.0,
        transport=transport,
    )
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
    # One call plus two transport retries inside the port; the semantic
    # attempt budget counts only the single content repair.
    assert transport.calls == 3
    assert result["repair_attempts"] == 1
    assert result["verified_strategy"] == StrategyProposal.model_validate(
        VALID_STRATEGY_PROPOSAL
    )
    assert result["degraded"] is False


def test_verify_transient_failure_propagates_without_consuming_semantic_attempt():
    llm = MockStructuredLLM({"strategy": TransientLLMError("injected transient failure")})
    graph = build_verify_graph(
        llm,
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=ALLOWED_SOFT_CONSTRAINTS,
    )
    with pytest.raises(TransientLLMError):
        graph.invoke({
            "candidate": dict(INVALID_CANDIDATE),
            "attempt": 0,
            "max_repairs": 2,
            "last_valid_strategy": None,
        })
