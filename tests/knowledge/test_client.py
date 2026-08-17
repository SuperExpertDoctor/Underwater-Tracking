from __future__ import annotations

import httpx

from underwater_tracking.knowledge.client import OntologyKnowledgeClient
from underwater_tracking.persistence.ledger import DecisionLedger


def _client(
    tmp_path, handler, *, max_retries: int = 1
) -> tuple[OntologyKnowledgeClient, DecisionLedger]:
    ledger = DecisionLedger(tmp_path / "knowledge.db")
    client = OntologyKnowledgeClient(
        base_url="http://ontology.test",
        max_retries=max_retries,
        backoff_base_s=0.01,
        backoff_max_s=0.01,
        ledger=ledger,
    )
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, ledger


def test_query_parses_trace_references_and_persists_audit(tmp_path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "response": "保持被动连续监听，链路恶化时优先保留高价值信息。",
                "mode": "mix",
                "trace": {
                    "reference_details": [
                        {
                            "locator": "C07#C07-T04-P02",
                            "document": "协同跟踪知识手册",
                            "title": "持续跟踪与质量维护",
                            "references": [{"url": "https://example.test/source"}],
                        }
                    ]
                },
            },
            request=request,
        )

    client, ledger = _client(tmp_path, handler)
    try:
        result = client.query(
            query_text="如何在通信受限时维持跟踪？",
            sim_time_s=30,
            scenario_id="S1",
        )
        assert result.answer.startswith("保持被动")
        assert result.references[0].reference_id == "C07#C07-T04-P02"
        assert result.references[0].url == "https://example.test/source"
        assert requests[0].url.path == "/api/query"
        assert requests[0].read().decode("utf-8").find("通信受限") >= 0
        rows = ledger.list_knowledge_queries("S1")
        assert len(rows) == 1
        assert rows[0].status == "completed"
        assert rows[0].response_hash
    finally:
        client.close()
        ledger.close()


def test_query_retries_transient_service_failure(tmp_path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, json={"answer": "可用", "references": []}, request=request)

    client, ledger = _client(tmp_path, handler, max_retries=2)
    try:
        result = client.query(query_text="链路状态", sim_time_s=60, scenario_id="S1")
        assert result.answer == "可用"
        assert attempts == 2
    finally:
        client.close()
        ledger.close()
