"""Bounded client for the tracking-plan ontology knowledge service.

The knowledge service is an evidence provider, not a policy engine. A query is
made before an LLM strategy adjustment when the planning route is strategic;
the returned answer and references are injected into the strategy prompt and
persisted for replay. Service failures are surfaced as ``LLMError`` so the
carrier pauses and retries instead of silently replacing expert knowledge
with a local rule.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from underwater_tracking.agent.llm import LLMError
from underwater_tracking.persistence.ledger import DecisionLedger


class KnowledgeServiceError(LLMError):
    """The ontology service could not provide the requested evidence."""


@dataclass(frozen=True, slots=True)
class KnowledgeReference:
    """One bounded source reference returned by the ontology service."""

    reference_id: str
    title: str = ""
    url: str = ""
    excerpt: str = ""

    def to_prompt_dict(self) -> dict[str, str]:
        return {
            "reference_id": self.reference_id,
            "title": self.title,
            "url": self.url,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeQueryResult:
    """A successful, replayable ontology answer."""

    query_id: str
    query_text: str
    answer: str
    references: tuple[KnowledgeReference, ...]
    trace_summary: str = ""
    mode: str = "mix"

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "answer": self.answer,
            "references": [reference.to_prompt_dict() for reference in self.references],
            "trace_summary": self.trace_summary,
            "mode": self.mode,
        }


class KnowledgeProvider(Protocol):
    """Small injected port used by strategy generation and tests."""

    def query(
        self,
        *,
        query_text: str,
        sim_time_s: int,
        scenario_id: str,
    ) -> KnowledgeQueryResult: ...


class OntologyKnowledgeClient:
    """HTTP client with bounded retry and durable query audit rows."""

    def __init__(
        self,
        *,
        base_url: str,
        query_path: str = "/api/query",
        mode: str = "mix",
        include_trace: bool = True,
        request_timeout_s: float = 15.0,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 8.0,
        ledger: DecisionLedger | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/{query_path.lstrip('/')}"
        self._mode = mode
        self._include_trace = include_trace
        self._max_attempts = max(1, int(max_retries))
        self._backoff_base_s = max(0.01, float(backoff_base_s))
        self._backoff_max_s = max(self._backoff_base_s, float(backoff_max_s))
        self._ledger = ledger
        self._client = httpx.Client(timeout=request_timeout_s)

    def close(self) -> None:
        self._client.close()

    def query(
        self,
        *,
        query_text: str,
        sim_time_s: int,
        scenario_id: str,
    ) -> KnowledgeQueryResult:
        """Query the ontology and persist the bounded answer for replay."""
        text = query_text.strip()
        if not text:
            raise KnowledgeServiceError("ontology query must not be empty")
        query_id = (
            f"{scenario_id}:knowledge:{sim_time_s}:"
            f"{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"
        )
        body = {
            "query": text,
            "mode": self._mode,
            "include_trace": self._include_trace,
            "conversation_history": [],
        }
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = self._client.post(self._url, json=body)
                status = response.status_code
                if status == 429 or 500 <= status <= 599:
                    raise _RetryableKnowledgeError(f"ontology service returned HTTP {status}")
                if status < 200 or status >= 300:
                    raise KnowledgeServiceError(
                        f"ontology service returned non-retryable HTTP {status}"
                    )
                payload = response.json()
                result = self._parse_result(query_id, text, payload)
                self._record(query_id, scenario_id, sim_time_s, text, "completed", result)
                return result
            except _RetryableKnowledgeError as exc:
                last_error = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
            except KnowledgeServiceError as exc:
                self._record_error(query_id, scenario_id, sim_time_s, text, str(exc))
                raise
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                error = KnowledgeServiceError(f"invalid ontology response: {exc}")
                self._record_error(query_id, scenario_id, sim_time_s, text, str(error))
                raise error from exc
            if attempt + 1 < self._max_attempts:
                time.sleep(min(self._backoff_max_s, self._backoff_base_s * (2**attempt)))
        error = KnowledgeServiceError(
            f"ontology service unavailable after {self._max_attempts} attempts: {last_error}"
        )
        self._record_error(query_id, scenario_id, sim_time_s, text, str(error))
        raise error from last_error

    def _parse_result(
        self,
        query_id: str,
        query_text: str,
        payload: object,
    ) -> KnowledgeQueryResult:
        if not isinstance(payload, Mapping):
            raise KnowledgeServiceError("ontology response must be a JSON object")
        answer = _first_text(payload, ("response", "answer", "message", "result"))
        if not answer:
            raise KnowledgeServiceError("ontology response did not contain an answer")
        trace = payload.get("trace")
        reference_payload = (
            payload.get("references")
            or payload.get("reference")
            or payload.get("sources")
            or payload.get("reference_details")
        )
        if reference_payload is None and isinstance(trace, Mapping):
            reference_payload = trace.get("reference_details")
        references = _references(reference_payload)
        trace_summary = _bounded_json(trace, 800) if trace is not None else ""
        return KnowledgeQueryResult(
            query_id=query_id,
            query_text=query_text,
            answer=answer[:4000],
            references=tuple(references[:8]),
            trace_summary=trace_summary,
            mode=self._mode,
        )

    def _record(
        self,
        query_id: str,
        scenario_id: str,
        sim_time_s: int,
        query_text: str,
        status: str,
        result: KnowledgeQueryResult,
    ) -> None:
        if self._ledger is not None:
            self._ledger.save_knowledge_query(
                query_id=query_id,
                scenario_id=scenario_id,
                sim_time_s=sim_time_s,
                query_text=query_text,
                mode=result.mode,
                status=status,
                response=result.to_prompt_dict(),
            )

    def _record_error(
        self,
        query_id: str,
        scenario_id: str,
        sim_time_s: int,
        query_text: str,
        error: str,
    ) -> None:
        if self._ledger is not None:
            self._ledger.save_knowledge_query(
                query_id=query_id,
                scenario_id=scenario_id,
                sim_time_s=sim_time_s,
                query_text=query_text,
                mode=self._mode,
                status="failed",
                response={"error": error[:1000]},
            )


class _RetryableKnowledgeError(Exception):
    pass


def _first_text(payload: Mapping[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            nested = _first_text(value, ("answer", "content", "text", "message"))
            if nested:
                return nested
    return ""


def _references(raw: object) -> list[KnowledgeReference]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    result: list[KnowledgeReference] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            result.append(KnowledgeReference(reference_id=f"ref-{index + 1}", excerpt=item[:600]))
            continue
        if not isinstance(item, Mapping):
            continue
        reference_id = str(item.get("id") or item.get("reference_id") or item.get("name") or f"ref-{index + 1}")
        nested_references = item.get("references")
        first_nested = (
            nested_references[0]
            if isinstance(nested_references, Sequence)
            and not isinstance(nested_references, (str, bytes))
            and nested_references
            and isinstance(nested_references[0], Mapping)
            else {}
        )
        result.append(
            KnowledgeReference(
                reference_id=str(
                    item.get("locator")
                    or item.get("source_key")
                    or reference_id
                )[:160],
                title=str(
                    item.get("title")
                    or item.get("name")
                    or item.get("document")
                    or ""
                )[:240],
                url=str(
                    item.get("url")
                    or item.get("source")
                    or first_nested.get("url", "")
                )[:500],
                excerpt=str(
                    item.get("excerpt")
                    or item.get("snippet")
                    or item.get("content")
                    or item.get("chunk_type")
                    or ""
                )[:600],
            )
        )
    return result


def _bounded_json(value: object, limit: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text[:limit]
