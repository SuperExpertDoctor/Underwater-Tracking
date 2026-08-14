# tests/agent/test_llm_port.py
"""Provider-neutral structured LLM port tests (spec 22, 8.3).

Covers the brief's two verbatim tests (validated mock output; transient
failures retried at most ``max_retries`` times and then raised), the retry
classification (timeout/connection/429/5xx retried, other 4xx never), the
deterministic jittered exponential backoff, call-time bearer token reads,
metadata hooks, and DecisionLedger persistence of hashes only — never
authorization headers, API keys, payloads, or the environment.
"""

import hashlib
import json
import os
from collections.abc import Callable

import httpx
import pytest

import underwater_tracking.agent.llm as llm_module
from underwater_tracking.agent.llm import (
    HTTPStructuredLLM,
    LLMConfigError,
    LLMContentError,
    LLMError,
    LLMCallMetadata,
    MockStructuredLLM,
    TransientLLMError,
)
from underwater_tracking.domain.agent_models import IntentHypothesis, StrategyProposal
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.sqlite import json_dumps
from tests.fixtures.llm_responses import (
    INVALID_INTENT_HYPOTHESIS,
    INVALID_STRATEGY_PROPOSAL,
    VALID_INTENT_HYPOTHESIS,
    VALID_STRATEGY_PROPOSAL,
)

API_KEY_ENV = "UNDERWATER_TRACKING_TEST_KEY"
TEST_BASE_URL = "http://llm.test/v1/chat"
TEST_MODEL = "mock-model"


def sha256_hex(value: object) -> str:
    """Canonical-JSON SHA-256, matching the client's request/response hashes."""
    return hashlib.sha256(json_dumps(value).encode("utf-8")).hexdigest()


class StubTransport(httpx.BaseTransport):
    """Deterministic httpx transport: serves queued responses or raises
    queued exceptions in order, recording every request."""

    def __init__(self, items: list[httpx.Response | BaseException]) -> None:
        self._items = list(items)
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class CountingTransport(httpx.BaseTransport):
    """Transport wrapper whose client fails on demand, counting every call.

    ``fail_with(error_type, times)`` makes the next ``times`` transport calls
    raise ``error_type`` before the success response is served, so tests can
    inject exactly N transient failures and count the client's attempts.
    """

    def __init__(
        self,
        *,
        success_json: object,
        api_key_env: str,
        max_retries: int = 3,
        base_url: str = TEST_BASE_URL,
    ) -> None:
        self._success_json = success_json
        self.calls = 0
        self._failures_remaining = 0
        self._error_type: type[TransientLLMError] | None = None
        self.client = HTTPStructuredLLM(
            base_url=base_url,
            model=TEST_MODEL,
            api_key_env=api_key_env,
            max_retries=max_retries,
            backoff_base_s=0.001,
            jitter=lambda: 0.0,
            transport=self,
        )

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


def make_client(
    transport: httpx.BaseTransport,
    *,
    max_retries: int = 3,
    backoff_base_s: float = 0.001,
    backoff_max_s: float = 60.0,
    jitter: Callable[[], float] | None = None,
    ledger: DecisionLedger | None = None,
    scenario_id: str = "S1",
    sim_time_s: int = 120,
    before_request: Callable[[LLMCallMetadata], None] | None = None,
    after_response: Callable[[LLMCallMetadata], None] | None = None,
) -> HTTPStructuredLLM:
    """HTTP client over the given stub transport with fast deterministic retries."""
    return HTTPStructuredLLM(
        base_url=TEST_BASE_URL,
        model=TEST_MODEL,
        api_key_env=API_KEY_ENV,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
        backoff_max_s=backoff_max_s,
        jitter=jitter,
        transport=transport,
        ledger=ledger,
        scenario_id=scenario_id,
        sim_time_s=sim_time_s,
        before_request=before_request,
        after_response=after_response,
    )


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """A bearer token the client can read from the configured env var."""
    monkeypatch.setenv(API_KEY_ENV, "test-token")
    return "test-token"


@pytest.fixture
def counting_transport(monkeypatch: pytest.MonkeyPatch) -> CountingTransport:
    monkeypatch.setenv(API_KEY_ENV, "test-token")
    return CountingTransport(success_json=VALID_INTENT_HYPOTHESIS, api_key_env=API_KEY_ENV)


def test_mock_llm_returns_validated_model():
    llm = MockStructuredLLM({"intent": {"label": "transit", "confidence": 0.8,
        "evidence_ids": ["B:T1:900"], "model_id": "mock", "prompt_version": "intent-v1"}})
    result = llm.invoke_structured("intent", {}, IntentHypothesis)
    assert result.label == "transit"


def test_transient_failures_retry_exactly_three_times(counting_transport):
    counting_transport.fail_with(TransientLLMError, times=3)
    with pytest.raises(TransientLLMError):
        counting_transport.client.invoke_structured("intent", {}, IntentHypothesis)
    assert counting_transport.calls == 3


def test_mock_llm_raises_content_error_for_invalid_response():
    llm = MockStructuredLLM({"intent": INVALID_INTENT_HYPOTHESIS})
    with pytest.raises(LLMContentError):
        llm.invoke_structured("intent", {}, IntentHypothesis)


def test_mock_llm_re_raises_injected_exception():
    llm = MockStructuredLLM({"intent": RuntimeError("provider exploded")})
    with pytest.raises(RuntimeError, match="provider exploded"):
        llm.invoke_structured("intent", {}, IntentHypothesis)


def test_mock_llm_consumes_queue_in_fifo_order():
    llm = MockStructuredLLM(
        {"intent": [VALID_INTENT_HYPOTHESIS, {**VALID_INTENT_HYPOTHESIS, "label": "evade"}]}
    )
    assert llm.invoke_structured("intent", {}, IntentHypothesis).label == "transit"
    assert llm.invoke_structured("intent", {}, IntentHypothesis).label == "evade"
    with pytest.raises(LLMError, match="no mock response"):
        llm.invoke_structured("intent", {}, IntentHypothesis)


def test_valid_strategy_proposal_response_validates():
    llm = MockStructuredLLM({"strategy": VALID_STRATEGY_PROPOSAL})
    result = llm.invoke_structured("strategy", {}, StrategyProposal)
    assert result.concept == "balanced"
    assert result.releasable_soft_constraints == ("energy_reserve_0.1",)
    assert result.rationale


def test_invalid_strategy_response_raises_content_error():
    llm = MockStructuredLLM({"strategy": INVALID_STRATEGY_PROPOSAL})
    with pytest.raises(LLMContentError):
        llm.invoke_structured("strategy", {}, StrategyProposal)


def test_mock_llm_records_metadata_to_ledger(tmp_path):
    ledger = DecisionLedger(tmp_path / "run.db")
    llm = MockStructuredLLM(
        {"intent": VALID_INTENT_HYPOTHESIS},
        model="mock",
        ledger=ledger,
        scenario_id="S1",
        sim_time_s=60,
    )
    llm.invoke_structured("intent", {}, IntentHypothesis, prompt_version="intent-v1")
    rows = ledger.list_llm_calls()
    assert len(rows) == 1
    row = rows[0]
    assert row.operation == "intent"
    assert row.model == "mock"
    assert row.prompt_version == "intent-v1"
    assert row.request_hash == sha256_hex({})
    assert row.response_hash == sha256_hex(VALID_INTENT_HYPOTHESIS)
    assert row.error_category == ""
    assert row.sim_time_s == 60
    assert row.scenario_id == "S1"


def test_http_client_posts_payload_with_bearer_token(api_key):
    payload = {"trajectory_features": [1.0, 2.0], "sampled_belief_history": [[0.0, 0.0]]}
    transport = StubTransport([httpx.Response(200, json=VALID_INTENT_HYPOTHESIS)])
    client = make_client(transport)
    result = client.invoke_structured("intent", payload, IntentHypothesis)
    assert result.label == "transit"
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.method == "POST"
    assert str(request.url) == TEST_BASE_URL
    assert request.headers["Authorization"] == "Bearer test-token"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == payload


def test_http_client_reads_bearer_token_at_call_time(api_key, monkeypatch):
    transport = StubTransport([httpx.Response(200, json=VALID_INTENT_HYPOTHESIS)] * 2)
    client = make_client(transport)
    client.invoke_structured("intent", {}, IntentHypothesis)
    monkeypatch.setenv(API_KEY_ENV, "rotated-token")
    client.invoke_structured("intent", {}, IntentHypothesis)
    assert transport.requests[0].headers["Authorization"] == "Bearer test-token"
    assert transport.requests[1].headers["Authorization"] == "Bearer rotated-token"


def test_http_client_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    transport = StubTransport([httpx.Response(200, json=VALID_INTENT_HYPOTHESIS)])
    client = make_client(transport)
    with pytest.raises(LLMConfigError, match=API_KEY_ENV):
        client.invoke_structured("intent", {}, IntentHypothesis)
    assert transport.requests == []


def test_timeout_is_retried_then_succeeds(api_key):
    transport = StubTransport(
        [httpx.ReadTimeout("read timed out"), httpx.Response(200, json=VALID_INTENT_HYPOTHESIS)]
    )
    client = make_client(transport)
    assert client.invoke_structured("intent", {}, IntentHypothesis).label == "transit"
    assert len(transport.requests) == 2


def test_connection_error_is_retried_then_succeeds(api_key):
    transport = StubTransport(
        [httpx.ConnectError("connection refused"), httpx.Response(200, json=VALID_INTENT_HYPOTHESIS)]
    )
    client = make_client(transport)
    assert client.invoke_structured("intent", {}, IntentHypothesis).label == "transit"
    assert len(transport.requests) == 2


def test_rate_limit_is_retried_then_succeeds(api_key):
    transport = StubTransport([httpx.Response(429), httpx.Response(200, json=VALID_INTENT_HYPOTHESIS)])
    client = make_client(transport)
    assert client.invoke_structured("intent", {}, IntentHypothesis).label == "transit"
    assert len(transport.requests) == 2


def test_server_error_is_retried_then_succeeds(api_key):
    transport = StubTransport([httpx.Response(503), httpx.Response(200, json=VALID_INTENT_HYPOTHESIS)])
    client = make_client(transport)
    assert client.invoke_structured("intent", {}, IntentHypothesis).label == "transit"
    assert len(transport.requests) == 2


def test_transient_failures_exhausted_raise_transient(api_key):
    transport = StubTransport([httpx.Response(429)] * 3)
    client = make_client(transport)
    with pytest.raises(TransientLLMError):
        client.invoke_structured("intent", {}, IntentHypothesis)
    assert len(transport.requests) == 3


def test_config_errors_are_not_retried(api_key):
    transport = StubTransport(
        [httpx.Response(400, json={"error": "bad request"}), httpx.Response(200, json=VALID_INTENT_HYPOTHESIS)]
    )
    client = make_client(transport)
    with pytest.raises(LLMConfigError):
        client.invoke_structured("intent", {}, IntentHypothesis)
    assert len(transport.requests) == 1


def test_backoff_is_exponential_with_injected_jitter(api_key, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(llm_module, "_sleep", sleeps.append)
    transport = StubTransport(
        [httpx.Response(429), httpx.Response(429), httpx.Response(200, json=VALID_INTENT_HYPOTHESIS)]
    )
    client = make_client(transport, backoff_base_s=0.5, jitter=lambda: 0.0)
    client.invoke_structured("intent", {}, IntentHypothesis)
    assert sleeps == [0.5, 1.0]


def test_backoff_jitter_scales_delay(api_key, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(llm_module, "_sleep", sleeps.append)
    transport = StubTransport(
        [httpx.Response(429), httpx.Response(429), httpx.Response(200, json=VALID_INTENT_HYPOTHESIS)]
    )
    client = make_client(transport, backoff_base_s=0.5, jitter=lambda: 0.5)
    client.invoke_structured("intent", {}, IntentHypothesis)
    assert sleeps == [0.75, 1.5]


def test_backoff_is_capped_at_max(api_key, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(llm_module, "_sleep", sleeps.append)
    transport = StubTransport([httpx.Response(503)] * 4)
    client = make_client(
        transport, max_retries=4, backoff_base_s=0.5, backoff_max_s=1.0, jitter=lambda: 0.0
    )
    with pytest.raises(TransientLLMError):
        client.invoke_structured("intent", {}, IntentHypothesis)
    assert sleeps == [0.5, 1.0, 1.0]


def test_invalid_response_schema_raises_content_error(api_key):
    transport = StubTransport([httpx.Response(200, json=INVALID_INTENT_HYPOTHESIS)])
    client = make_client(transport)
    with pytest.raises(LLMContentError):
        client.invoke_structured("intent", {}, IntentHypothesis)
    assert len(transport.requests) == 1


def test_non_json_response_raises_content_error(api_key):
    transport = StubTransport([httpx.Response(200, content=b"<html>not json</html>")])
    client = make_client(transport)
    with pytest.raises(LLMContentError):
        client.invoke_structured("intent", {}, IntentHypothesis)


def test_usage_tokens_extracted_when_present(api_key):
    response = {**VALID_INTENT_HYPOTHESIS, "usage": {"total_tokens": 321}}
    transport = StubTransport([httpx.Response(200, json=response)])
    seen_after: list[LLMCallMetadata] = []
    client = make_client(transport, after_response=seen_after.append)
    client.invoke_structured("intent", {}, IntentHypothesis)
    assert seen_after[0].token_count == 321
    assert seen_after[0].response_hash == sha256_hex(VALID_INTENT_HYPOTHESIS)


def test_metadata_hooks_receive_hashes_not_secrets(api_key):
    seen_before: list[LLMCallMetadata] = []
    seen_after: list[LLMCallMetadata] = []
    transport = StubTransport([httpx.Response(200, json=VALID_INTENT_HYPOTHESIS)])
    client = make_client(transport, before_request=seen_before.append, after_response=seen_after.append)
    client.invoke_structured("intent", {}, IntentHypothesis, prompt_version="intent-v1")
    assert len(seen_before) == 1
    assert seen_before[0].operation == "intent"
    assert seen_before[0].prompt_version == "intent-v1"
    assert seen_before[0].request_hash == sha256_hex({})
    assert seen_before[0].response_hash == ""
    assert len(seen_after) == 1
    assert seen_after[0].response_hash == sha256_hex(VALID_INTENT_HYPOTHESIS)
    assert seen_after[0].error_category == ""
    for metadata in seen_before + seen_after:
        assert "test-token" not in repr(metadata)
        assert "Authorization" not in repr(metadata)


def test_ledger_records_success_call_metadata(tmp_path, api_key):
    payload = {"sampled_belief_history": [[10.0, 20.0], [11.0, 21.0]]}
    ledger = DecisionLedger(tmp_path / "run.db")
    transport = StubTransport([httpx.Response(200, json=VALID_INTENT_HYPOTHESIS)])
    client = make_client(transport, ledger=ledger, scenario_id="S1", sim_time_s=120)
    client.invoke_structured("intent", payload, IntentHypothesis, prompt_version="intent-v1")
    rows = ledger.list_llm_calls()
    assert len(rows) == 1
    row = rows[0]
    assert row.operation == "intent"
    assert row.model == TEST_MODEL
    assert row.prompt_version == "intent-v1"
    assert row.request_hash == sha256_hex(payload)
    assert row.response_hash == sha256_hex(VALID_INTENT_HYPOTHESIS)
    assert row.latency_ms >= 0
    assert row.token_count == 0
    assert row.error_category == ""
    assert row.sim_time_s == 120
    assert row.scenario_id == "S1"


def test_ledger_records_transient_and_content_error_categories(tmp_path, api_key):
    ledger = DecisionLedger(tmp_path / "run.db")
    transport = StubTransport([httpx.Response(503)] * 3)
    client = make_client(transport, ledger=ledger)
    with pytest.raises(TransientLLMError):
        client.invoke_structured("intent", {}, IntentHypothesis)
    rows = ledger.list_llm_calls()
    assert len(rows) == 3
    assert {row.error_category for row in rows} == {"server"}

    content_ledger = DecisionLedger(tmp_path / "content.db")
    content_transport = StubTransport([httpx.Response(200, json=INVALID_INTENT_HYPOTHESIS)])
    content_client = make_client(content_transport, ledger=content_ledger)
    with pytest.raises(LLMContentError):
        content_client.invoke_structured("intent", {}, IntentHypothesis)
    content_row = content_ledger.list_llm_calls()[0]
    assert content_row.error_category == "content"


def test_ledger_never_persists_tokens_or_payloads(tmp_path, api_key):
    payload = {"trajectory_features": [1.0, 2.0], "secret_marker": "mission-data-42"}
    ledger = DecisionLedger(tmp_path / "run.db")
    transport = StubTransport([httpx.Response(200, json=VALID_INTENT_HYPOTHESIS)])
    client = make_client(transport, ledger=ledger)
    client.invoke_structured("intent", payload, IntentHypothesis)
    rows = ledger.list_llm_calls()
    assert len(rows) == 1
    raw = (tmp_path / "run.db").read_bytes()
    assert b"test-token" not in raw
    assert b"Bearer" not in raw
    assert b"Authorization" not in raw
    assert b"mission-data-42" not in raw
    env_name = next(iter(os.environ))
    assert env_name.encode() not in raw
