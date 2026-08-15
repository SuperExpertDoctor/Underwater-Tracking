# src/underwater_tracking/agent/llm.py
"""Provider-neutral structured LLM port (spec 22, 8.3).

Business code never depends on a concrete provider: ``StructuredLLM`` is the
single call surface and ``HTTPStructuredLLM`` is the only client — it talks
to the configured chat endpoint (the LongCat OpenAI-compatible provider)
with the bearer token read at call time from the environment variable
configured via ``api_key_env``. Per the user directive (addendum A) no mock
ever substitutes real LLM functionality: tests drive the real client or do
not invoke the LLM at all.

Transport retries (spec 8.3) use an independent counter from content
repairs: only timeout, connection, rate-limit (429) and server (5xx)
failures are retried — at most ``max_retries`` attempts with exponential
backoff — while config errors (other 4xx, missing API key) and content
errors (invalid response JSON or schema) are never blindly retried. The
bearer token is read at call time from the configured environment variable
only (spec 22) and never stored, logged, or persisted: every metadata record
and hook carries request/response hashes exclusively.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Protocol, Self, TypeVar, cast

import httpx
from pydantic import BaseModel, ValidationError

from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.sqlite import json_dumps

# OpenAI chat/completions budget for the structured-output requests (the
# LongCat provider has no native response_format support, so the schema is
# carried in the system prompt instead).
_DEFAULT_MAX_TOKENS = 4096

# Error categories persisted to the DecisionLedger ``llm_calls`` table so
# transport and content failures stay distinguishable for retry bookkeeping
# (spec 8.3, 16).
_CATEGORY_TIMEOUT = "timeout"
_CATEGORY_CONNECTION = "connection"
_CATEGORY_RATE_LIMIT = "rate_limit"
_CATEGORY_SERVER = "server"
_CATEGORY_CONFIG = "config"
_CATEGORY_CONTENT = "content"

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base class for structured LLM port failures."""


class LLMConfigError(LLMError):
    """Non-retryable configuration failure: missing API key, other 4xx."""


class TransientLLMError(LLMError):
    """Retryable transport failure: timeout, connection, 429, 5xx (spec 8.3).

    ``category`` is one of the ``_CATEGORY_*`` constants and is persisted
    with the call metadata; the empty string means an injected/unknown
    transient failure.
    """

    def __init__(self, message: str, *, category: str = "") -> None:
        super().__init__(message)
        self.category = category


class LLMContentError(LLMError):
    """The provider response failed schema validation (spec 8.3 content path).

    Content failures are never retried by the transport; the caller's Verify
    subgraph repairs them with bounded error re-injection.
    """


@dataclass
class LLMCallMetadata:
    """Per-attempt LLM call metadata: hashes only, never payloads or secrets.

    The same shape feeds the request/response hooks and
    ``DecisionLedger.record_llm_call``, so neither observers nor the ledger
    can ever see authorization headers, API keys, or the environment.
    """

    operation: str
    model: str
    prompt_version: str
    request_hash: str
    response_hash: str = ""
    latency_ms: int = 0
    token_count: int = 0
    error_category: str = ""
    sim_time_s: int = 0
    scenario_id: str = ""


class StructuredLLM(Protocol[T]):
    """Provider-neutral structured-output port (spec 22).

    ``invoke_structured`` is the only call surface: it sends the caller-built
    ``payload`` for an ``operation``, validates the raw response against
    ``response_model``, and returns the validated model.
    """

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[T],
        *,
        prompt_version: str = "",
    ) -> T: ...


class HTTPStructuredLLM:
    """HTTP structured-output client with bounded exponential-backoff retries.

    The client speaks the OpenAI chat/completions shape (the LongCat
    provider's compatible surface): the request body carries ``model``, a
    ``messages`` array — the target model's JSON schema as a system
    instruction, the caller-built ``payload`` as the user content — plus
    ``temperature`` and ``max_tokens``. The response is parsed from
    ``choices[0].message.content`` as JSON and validated against the
    requested model, extracting ``usage`` token counts when the provider
    includes them. The bearer token is read at call time from the configured
    environment variable and is never stored on the instance.

    ``max_retries`` bounds the total transport attempts (the initial call
    plus retries); between attempts the client sleeps
    ``min(max, base * 2**(attempt-1)) * (1 + jitter())`` seconds, with the
    ``jitter`` callable injected so embedders can make the backoff
    deterministic (the default is time-based). ``transport`` injects the
    underlying ``httpx.BaseTransport`` for embedders; per the user
    directive (addendum A) tests never inject either — every test drives
    the real network or does not invoke the client at all. Transient
    failures (timeout, connection, 429, 5xx) are retried; config and
    content failures are not.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "",
        api_key_env: str,
        request_timeout_s: float = 60.0,
        connect_timeout_s: float = 10.0,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
        jitter: Callable[[], float] | None = None,
        transport: httpx.BaseTransport | None = None,
        ledger: DecisionLedger | None = None,
        scenario_id: str = "",
        sim_time_s: int = 0,
        temperature: float = 0.2,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        before_request: Callable[[LLMCallMetadata], None] | None = None,
        after_response: Callable[[LLMCallMetadata], None] | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._api_key_env = api_key_env
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_attempts = max(1, max_retries)
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self._jitter: Callable[[], float] = (
            jitter if jitter is not None else _time_based_jitter
        )
        self._ledger = ledger
        self._scenario_id = scenario_id
        self._sim_time_s = sim_time_s
        self._before_request = before_request
        self._after_response = after_response
        self._client = httpx.Client(
            timeout=httpx.Timeout(request_timeout_s, connect=connect_timeout_s),
            transport=transport,
        )

    def close(self) -> None:
        """Close the underlying HTTP client (no-op for stub transports)."""
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[T],
        *,
        prompt_version: str = "",
    ) -> T:
        """Send one structured-output request with bounded transport retries."""
        request_hash = _digest(payload)
        attempt = 0
        while True:
            attempt += 1
            started = _now_ms()
            metadata = LLMCallMetadata(
                operation=operation,
                model=self._model,
                prompt_version=prompt_version,
                request_hash=request_hash,
                sim_time_s=self._sim_time_s,
                scenario_id=self._scenario_id,
            )
            try:
                response_json, token_count = self._request_once(
                    metadata, payload, response_model
                )
            except TransientLLMError as exc:
                metadata.latency_ms = _now_ms() - started
                metadata.error_category = exc.category
                _record_call(self._ledger, metadata)
                self._emit_after_response(metadata)
                if attempt >= self._max_attempts:
                    raise
                _sleep(self._backoff_delay(attempt))
                continue
            except LLMConfigError:
                metadata.latency_ms = _now_ms() - started
                metadata.error_category = _CATEGORY_CONFIG
                _record_call(self._ledger, metadata)
                self._emit_after_response(metadata)
                raise
            metadata.response_hash = _digest(response_json)
            metadata.token_count = token_count
            metadata.latency_ms = _now_ms() - started
            try:
                result = response_model.model_validate(response_json)
            except ValidationError as exc:
                metadata.error_category = _CATEGORY_CONTENT
                _record_call(self._ledger, metadata)
                self._emit_after_response(metadata)
                raise LLMContentError(
                    f"response for operation {operation!r} failed schema validation"
                ) from exc
            _record_call(self._ledger, metadata)
            self._emit_after_response(metadata)
            return result

    def _request_once(
        self,
        metadata: LLMCallMetadata,
        payload: dict[str, object],
        response_model: type[T],
    ) -> tuple[object, int]:
        """One transport attempt; raises typed LLM errors, never retries here."""
        token = os.environ.get(self._api_key_env)
        if token is None:
            raise LLMConfigError(f"environment variable {self._api_key_env!r} is not set")
        if self._before_request is not None:
            # The hook observes a snapshot; the client mutates its own copy
            # as the attempt completes.
            self._before_request(replace(metadata))
        request_body = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object matching this JSON schema:\n"
                        + json_dumps(response_model.model_json_schema())
                    ),
                },
                {"role": "user", "content": json_dumps(payload)},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        # ``base_url`` is the OpenAI-compatible API root (e.g.
        # ``https://api.longcat.chat/openai/v1``); the completions endpoint is
        # ``{root}/chat/completions`` — POSTing to the root itself 404s.
        completions_url = f"{self._base_url.rstrip('/')}/chat/completions"
        try:
            response = self._client.post(
                completions_url,
                json=request_body,
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.TimeoutException as exc:
            raise TransientLLMError(
                f"timeout while calling {metadata.operation}", category=_CATEGORY_TIMEOUT
            ) from exc
        except httpx.TransportError as exc:
            raise TransientLLMError(
                f"connection error while calling {metadata.operation}",
                category=_CATEGORY_CONNECTION,
            ) from exc
        status = response.status_code
        if status == 429:
            raise TransientLLMError(
                f"rate limited (429) while calling {metadata.operation}",
                category=_CATEGORY_RATE_LIMIT,
            )
        if 500 <= status <= 599:
            raise TransientLLMError(
                f"server error ({status}) while calling {metadata.operation}",
                category=_CATEGORY_SERVER,
            )
        if 400 <= status <= 499:
            raise LLMConfigError(
                f"provider config error ({status}) while calling {metadata.operation}"
            )
        if not 200 <= status < 300:
            raise LLMConfigError(
                f"unexpected status ({status}) while calling {metadata.operation}"
            )
        try:
            response_json: object = response.json()
        except ValueError as exc:
            raise LLMContentError("provider response is not valid JSON") from exc
        if not isinstance(response_json, dict):
            raise LLMContentError("provider response is not a JSON object")
        token_count = 0
        usage = response_json.get("usage")
        if isinstance(usage, dict):
            raw_tokens = (
                usage.get("total_tokens")
                or usage.get("completion_tokens")
                or usage.get("prompt_tokens")
            )
            token_count = int(raw_tokens) if isinstance(raw_tokens, int) else 0
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMContentError("provider response has no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMContentError("provider choice is not a JSON object")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise LLMContentError("provider response has no message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMContentError("provider response message has no content")
        extracted = _extract_json_value(content)
        if extracted is None:
            raise LLMContentError("provider response content is not valid JSON")
        return extracted, token_count

    def _backoff_delay(self, attempt: int) -> float:
        """Jittered exponential backoff for the given (1-based) failed attempt."""
        exponential = min(
            self._backoff_max_s, self._backoff_base_s * (2.0 ** (attempt - 1))
        )
        return exponential * (1.0 + self._jitter())

    def _emit_after_response(self, metadata: LLMCallMetadata) -> None:
        if self._after_response is not None:
            self._after_response(metadata)


def _time_based_jitter() -> float:
    """Time-seeded jitter in [0, 1) for the default exponential backoff."""
    return random.uniform(0.0, 1.0)


def _record_call(ledger: DecisionLedger | None, metadata: LLMCallMetadata) -> None:
    """Persist one call's metadata (hashes only) when a ledger is provided."""
    if ledger is not None:
        ledger.record_llm_call(
            operation=metadata.operation,
            model=metadata.model,
            prompt_version=metadata.prompt_version,
            request_hash=metadata.request_hash,
            response_hash=metadata.response_hash,
            latency_ms=metadata.latency_ms,
            token_count=metadata.token_count,
            error_category=metadata.error_category,
            sim_time_s=metadata.sim_time_s,
            scenario_id=metadata.scenario_id,
        )


def _digest(value: object) -> str:
    """SHA-256 of the canonical JSON encoding (sorted keys, compact separators)."""
    return hashlib.sha256(json_dumps(value).encode("utf-8")).hexdigest()


def _now_ms() -> int:
    """Monotonic milliseconds for latency measurement."""
    return int(time.monotonic() * 1000)


def _sleep(seconds: float) -> None:
    """Module-level sleep so tests can intercept backoff without touching ``time``."""
    time.sleep(seconds)


def _extract_json_value(content: str) -> object | None:
    """Recover the JSON value from a provider response (spec 8.3 content path).

    Deterministic and non-retrying: a markdown code fence is stripped first,
    a bare value that parses directly wins, and otherwise the first balanced
    ``{...}``/``[...]`` block in the text (found by a bracket scan that
    respects quoted strings) is parsed. Returns None when no parseable value
    exists — the only case that raises ``LLMContentError``.
    """
    stripped = content.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        closing = stripped.find("```", 3)
        stripped = stripped[3 : (closing if closing != -1 else len(stripped))].strip()
        newline = stripped.find("\n")
        if newline != -1 and not stripped[:newline].strip().startswith(("{", "[")):
            # Drop the fence's optional language tag line (e.g. ``json``).
            stripped = stripped[newline + 1 :].strip()
    if stripped[:1] in "{[\"":
        try:
            return cast(object, json.loads(stripped))
        except ValueError:
            pass
    for block in _balanced_json_blocks(stripped):
        try:
            return cast(object, json.loads(block))
        except ValueError:
            continue
    return None


def _balanced_json_blocks(text: str) -> Iterator[str]:
    """Yield each top-level balanced ``{...}``/``[...]`` block in order.

    The scan walks the string once, tracking bracket depth while skipping
    quoted strings (including escapes), so prose before, inside, or after
    the JSON never disturbs the bracket count.
    """
    start = -1
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            if start == -1:
                start = index
            depth += 1
        elif char in "}]" and start != -1:
            depth -= 1
            if depth == 0:
                yield text[start : index + 1]
                start = -1
