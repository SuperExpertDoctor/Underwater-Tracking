# tests/integration/test_llm_real_api.py
"""The real LongCat API: structured round-trip, outbound wire, content sanity.

Per the user directive (addendum A) these three tests hit the real
provider — three requests total. The wire test wraps the outbound request
in a PASS-THROUGH recorder (the call still goes out over the real network;
the wrapper only observes), never a substitute. The API key is read from
the environment at call time, sent by the client as a bearer token, and
never printed, logged, or asserted by value anywhere. The module is
skipped when the key is unset.
"""

from __future__ import annotations

import json

import httpx
import pytest

from underwater_tracking.agent.llm import HTTPStructuredLLM
from underwater_tracking.agent.nodes.intent import IntentAnalysisNode
from underwater_tracking.agent.prompts import INTENT_PROMPT_VERSION
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import IntentHypothesis
from underwater_tracking.domain.models import (
    GroupQuality,
    GroupReport,
    SituationSnapshot,
    StrictModel,
    TargetBelief,
)
from tests.conftest import (
    CONFIG_PATH,
    REAL_LLM_SKIP_REASON,
    has_live_api_key,
)

pytestmark = pytest.mark.skipif(
    not has_live_api_key(),
    reason=REAL_LLM_SKIP_REASON,
)

INTENT_LABELS = ("transit", "patrol", "loiter", "evade", "approach", "withdraw", "unknown")

# Estimated belief history for the intent payload (strictly increasing).
T1_HISTORY: tuple[tuple[int, float, float], ...] = (
    (600, 80.0, 150.0),
    (660, 90.0, 170.0),
    (720, 100.0, 190.0),
    (780, 110.0, 205.0),
    (840, 120.0, 215.0),
    (900, 130.0, 220.0),
)


class _RoundTripAnswer(StrictModel):
    """Minimal response model: the smallest schema the provider must satisfy."""

    answer: str
    confidence: float


@pytest.mark.real_llm
def test_small_pydantic_round_trip(live_llm: HTTPStructuredLLM) -> None:
    """1 request: a minimal schema round-trips with a bounded confidence."""
    result = live_llm.invoke_structured(
        "round_trip",
        {
            "system_prompt": (
                "Answer with exactly one short sentence and a confidence in [0, 1]."
            ),
            "question_text": "Is the target heading north?",
        },
        _RoundTripAnswer,
        prompt_version="round-trip-v1",
    )
    assert isinstance(result, _RoundTripAnswer)
    assert isinstance(result.answer, str) and result.answer
    assert 0.0 <= result.confidence <= 1.0


class RecordingTransport(httpx.BaseTransport):
    """Pass-through recorder: the request still reaches the provider."""

    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner
        self.request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        return self._inner.handle_request(request)


@pytest.mark.real_llm
def test_outbound_request_carries_longcat_model_and_bearer_token() -> None:
    """1 request: the wire observes the configured model, the completions
    endpoint, and a bearer token.

    The wrapper only records the outbound request — the call still goes to
    the real LongCat endpoint through an ordinary ``httpx.HTTPTransport``
    built from the shipped ``configs/llm.yaml`` (observation, not
    substitution). ``base_url`` is the API root, so the request URL must
    end with ``/chat/completions`` (POSTing to the root 404s — fix round 1
    regression).
    """
    config = load_app_config(CONFIG_PATH)
    assert config.llm is not None
    recorder = RecordingTransport(httpx.HTTPTransport())
    client = HTTPStructuredLLM(
        base_url=config.llm.base_url,
        model=config.llm.model,
        api_key_env=config.llm.api_key_env,
        request_timeout_s=config.llm.request_timeout_s,
        connect_timeout_s=config.llm.connect_timeout_s,
        temperature=config.llm.temperature,
        transport=recorder,
    )
    try:
        result = client.invoke_structured(
            "round_trip",
            {
                "system_prompt": 'Answer "ok" with confidence 1.0.',
                "question_text": "ok?",
            },
            _RoundTripAnswer,
            prompt_version="round-trip-v1",
        )
        assert isinstance(result, _RoundTripAnswer)
        assert recorder.request is not None
        # Regression (fix round 1): the POST targets the completions
        # endpoint derived from the API root, never the root itself.
        assert str(recorder.request.url).rstrip("/").endswith(
            "/chat/completions"
        )
        assert str(recorder.request.url).startswith(
            f"{config.llm.base_url.rstrip('/')}/"
        )
        body = json.loads(recorder.request.content)
        assert body["model"] == "LongCat-2.0"
        authorization = recorder.request.headers.get("authorization", "")
        # The token itself is never printed, logged, or asserted by value.
        assert authorization.startswith("Bearer ")
        assert authorization[len("Bearer "):].strip() != ""
    finally:
        client.close()


def _single_target_snapshot() -> SituationSnapshot:
    """One tracked target at t=900 s with two bearing observations."""
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=3,
        sim_time_s=900,
        uuvs=(),
        group_reports=(
            GroupReport(
                group_id="G-T1",
                target_id="T1",
                sim_time_s=900,
                member_ids=(),
                belief=TargetBelief(
                    target_id="T1",
                    sim_time_s=900,
                    mean=(130.0, 220.0, 1.0, 0.5),
                    covariance=(
                        (400.0, 0.0, 0.0, 0.0),
                        (0.0, 400.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (0.0, 0.0, 0.0, 1.0),
                    ),
                    model_probabilities={"cv": 0.7, "ct": 0.3},
                    source_observation_ids=("B:T1:900", "B:T1:870"),
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
            ),
        ),
        pending_events=(),
    )


@pytest.mark.real_llm
def test_intent_payload_semantic_content_sanity(live_llm: HTTPStructuredLLM) -> None:
    """1 request: the real curated intent payload yields a sane hypothesis.

    The payload is built by the intent node's pure ``build_payload`` (the
    exact payload the agent sends), then answered by the real provider:
    the label must be one of the schema's labels, the confidence bounded,
    and every cited evidence id a non-empty string.
    """
    node = IntentAnalysisNode(live_llm, model_id="LongCat-2.0")
    payload = node.build_payload(
        _single_target_snapshot(), target_id="T1", belief_history=T1_HISTORY
    )
    assert payload["evidence_ids"] == ["B:T1:870", "B:T1:900"]
    hypothesis = live_llm.invoke_structured(
        "intent", payload, IntentHypothesis, prompt_version=INTENT_PROMPT_VERSION
    )
    assert isinstance(hypothesis, IntentHypothesis)
    assert hypothesis.label in INTENT_LABELS
    assert 0.0 <= hypothesis.confidence <= 1.0
    assert hypothesis.evidence_ids
    assert all(isinstance(eid, str) and eid for eid in hypothesis.evidence_ids)
    # The schema requires these strings; the values are model-written, so
    # only non-empty sanity is asserted.
    assert hypothesis.model_id
    assert hypothesis.prompt_version
