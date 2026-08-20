"""Real OpenAI-compatible embeddings with typed failures and hashed audit rows."""

from __future__ import annotations

import hashlib
import math
import os
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import httpx

from underwater_tracking.agent.llm import (
    LLMConfigError,
    LLMContentError,
    TransientLLMError,
)
from underwater_tracking.config.models import MemoryConfig
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.sqlite import json_dumps

_MAX_VECTOR_DIMENSIONS = 16_384
_MAX_RETRY_DELAY_S = 60.0


@dataclass(frozen=True)
class EmbeddingResult:
    """One validated provider vector and the version to persist beside it."""

    vector: tuple[float, ...]
    model: str
    vector_version: str
    token_count: int = 0

    @property
    def dimensions(self) -> int:
        return len(self.vector)


class EmbeddingProvider(Protocol):
    """Provider-neutral boundary for real query and memory embeddings."""

    def embed(self, text: str) -> EmbeddingResult: ...


class HTTPEmbeddingProvider:
    """OpenAI-compatible ``/embeddings`` client without a local fallback."""

    def __init__(
        self,
        config: MemoryConfig,
        *,
        ledger: DecisionLedger | None = None,
        scenario_id: str = "",
        sim_time_s: int = 0,
    ) -> None:
        if not config.enabled or not config.embedding_base_url or not config.embedding_model:
            raise LLMConfigError("memory embedding provider is not configured")
        self._base_url = config.embedding_base_url
        self._model = config.embedding_model
        self._api_key_env = config.embedding_api_key_env
        self._vector_version = config.embedding_vector_version
        self._max_attempts = max(1, config.max_attempts)
        self._retry_backoff_s = config.retry_backoff_s
        self._ledger = ledger
        self._scenario_id = scenario_id
        self._sim_time_s = sim_time_s
        self._client = httpx.Client(timeout=httpx.Timeout(config.embedding_timeout_s))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HTTPEmbeddingProvider":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def embed(self, text: str) -> EmbeddingResult:
        """Embed one bounded input through the configured provider only."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("embedding text must be a non-blank string")
        request_hash = _digest({"model": self._model, "input": text})
        attempt = 0
        while True:
            attempt += 1
            started = _now_ms()
            try:
                response, token_count = self._request_once(text)
                result = parse_embedding_response(
                    response,
                    model=self._model,
                    vector_version=self._vector_version,
                    token_count=token_count,
                )
            except TransientLLMError as exc:
                self._record(
                    request_hash=request_hash,
                    latency_ms=_now_ms() - started,
                    error_category=exc.category,
                )
                if attempt >= self._max_attempts:
                    raise
                time.sleep(self._retry_delay(attempt))
                continue
            except LLMConfigError:
                self._record(
                    request_hash=request_hash,
                    latency_ms=_now_ms() - started,
                    error_category="config",
                )
                raise
            except LLMContentError:
                self._record(
                    request_hash=request_hash,
                    latency_ms=_now_ms() - started,
                    error_category="content",
                )
                raise
            self._record(
                request_hash=request_hash,
                response_hash=_digest(response),
                latency_ms=_now_ms() - started,
                token_count=result.token_count,
            )
            return result

    def _request_once(self, text: str) -> tuple[object, int]:
        token = os.environ.get(self._api_key_env)
        if not token:
            raise LLMConfigError(
                f"environment variable {self._api_key_env!r} is not set for memory embeddings"
            )
        try:
            response = self._client.post(
                f"{self._base_url.rstrip('/')}/embeddings",
                json={"model": self._model, "input": text},
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.TimeoutException as exc:
            raise TransientLLMError(
                "timeout while calling memory_embedding", category="timeout"
            ) from exc
        except httpx.TransportError as exc:
            raise TransientLLMError(
                "connection error while calling memory_embedding", category="connection"
            ) from exc
        if response.status_code == 429:
            raise TransientLLMError(
                "rate limited (429) while calling memory_embedding", category="rate_limit"
            )
        if 500 <= response.status_code <= 599:
            raise TransientLLMError(
                f"server error ({response.status_code}) while calling memory_embedding",
                category="server",
            )
        if 400 <= response.status_code <= 499:
            raise LLMConfigError(
                f"provider config error ({response.status_code}) while calling memory_embedding"
            )
        if not 200 <= response.status_code < 300:
            raise LLMConfigError(
                f"unexpected status ({response.status_code}) while calling memory_embedding"
            )
        try:
            payload: object = response.json()
        except ValueError as exc:
            raise LLMContentError("embedding provider response is not valid JSON") from exc
        return payload, _token_count(payload)

    def _retry_delay(self, attempt: int) -> float:
        exponential = min(_MAX_RETRY_DELAY_S, self._retry_backoff_s * (2.0 ** (attempt - 1)))
        return exponential * (1.0 + random.uniform(0.0, 1.0))

    def _record(
        self,
        *,
        request_hash: str,
        response_hash: str = "",
        latency_ms: int = 0,
        token_count: int = 0,
        error_category: str = "",
    ) -> None:
        if self._ledger is not None:
            self._ledger.record_llm_call(
                operation="memory_embedding",
                model=self._model,
                prompt_version=self._vector_version,
                request_hash=request_hash,
                response_hash=response_hash,
                latency_ms=latency_ms,
                token_count=token_count,
                error_category=error_category,
                sim_time_s=self._sim_time_s,
                scenario_id=self._scenario_id,
            )


def parse_embedding_response(
    response: object,
    *,
    model: str,
    vector_version: str,
    token_count: int | None = None,
) -> EmbeddingResult:
    """Validate the provider envelope before an embedding can be persisted."""
    if not isinstance(response, Mapping):
        raise LLMContentError("embedding provider response is not a JSON object")
    response_model = response.get("model")
    if not isinstance(response_model, str) or response_model != model:
        raise LLMContentError("embedding provider response model does not match request")
    data = response.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], Mapping):
        raise LLMContentError("embedding provider response must contain one data item")
    raw_vector = data[0].get("embedding")
    if not isinstance(raw_vector, list) or not raw_vector or len(raw_vector) > _MAX_VECTOR_DIMENSIONS:
        raise LLMContentError("embedding provider vector has an invalid dimension")
    vector: list[float] = []
    for value in raw_vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LLMContentError("embedding provider vector contains a non-numeric value")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise LLMContentError("embedding provider vector contains a non-finite value")
        vector.append(numeric)
    actual_token_count = _token_count(response) if token_count is None else token_count
    return EmbeddingResult(
        vector=tuple(vector),
        model=model,
        vector_version=vector_version,
        token_count=actual_token_count,
    )


def _token_count(response: object) -> int:
    if not isinstance(response, Mapping):
        return 0
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return 0
    raw = usage.get("total_tokens") or usage.get("prompt_tokens")
    return raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else 0


def _digest(value: object) -> str:
    return hashlib.sha256(json_dumps(value).encode("utf-8")).hexdigest()


def _now_ms() -> int:
    return int(time.monotonic() * 1000)
