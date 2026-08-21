"""Real semantic embedding providers with typed failures and hashed audit rows."""

from __future__ import annotations

import hashlib
import math
import os
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol

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


class SentenceTransformerEmbeddingProvider:
    """Generate semantic embeddings from a local SentenceTransformer model.

    Model loading is lazy so constructing the runtime does not block on a
    potentially large model. The provider is deliberately local-only: a
    missing package or model raises a typed configuration error and never
    downloads a model or fabricates a vector.
    """

    def __init__(
        self,
        config: MemoryConfig,
        *,
        ledger: DecisionLedger | None = None,
        scenario_id: str = "",
        sim_time_s: int = 0,
    ) -> None:
        if (
            not config.enabled
            or config.embedding_provider != "sentence_transformers"
            or not config.embedding_model
        ):
            raise LLMConfigError("local sentence-transformer embedding is not configured")
        if not config.embedding_local_files_only:
            raise LLMConfigError(
                "local sentence-transformer embedding requires local_files_only=true"
            )
        self._model_name = config.embedding_model
        self._vector_version = config.embedding_vector_version
        self._device = config.embedding_device
        self._normalize = config.embedding_normalize
        self._ledger = ledger
        self._scenario_id = scenario_id
        self._sim_time_s = sim_time_s
        self._model: Any | None = None
        self._lock = RLock()
        self._closed = False

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._closed = True

    def __enter__(self) -> "SentenceTransformerEmbeddingProvider":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def embed(self, text: str) -> EmbeddingResult:
        """Return one vector from the configured local model only."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("embedding text must be a non-blank string")
        request_hash = _digest({"model": self._model_name, "input": text})
        started = _now_ms()
        with self._lock:
            if self._closed:
                closed_error = LLMConfigError("local sentence-transformer provider is closed")
                self._record(
                    request_hash=request_hash,
                    latency_ms=_now_ms() - started,
                    error_category="config",
                )
                raise closed_error
            try:
                model = self._model
                if model is None:
                    model = self._load_model()
                    self._model = model
                raw_vector = model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                )
                vector = _validate_local_vector(raw_vector)
                result = EmbeddingResult(
                    vector=vector,
                    model=self._model_name,
                    vector_version=self._vector_version,
                    token_count=(len(text) + 3) // 4,
                )
            except LLMConfigError as exc:
                self._record(
                    request_hash=request_hash,
                    latency_ms=_now_ms() - started,
                    error_category="config",
                )
                raise exc
            except LLMContentError as exc:
                self._record(
                    request_hash=request_hash,
                    latency_ms=_now_ms() - started,
                    error_category="content",
                )
                raise exc
            except Exception as exc:  # noqa: BLE001 - provider boundary becomes typed
                content_error = LLMContentError(
                    f"sentence-transformer embedding failed for {self._model_name!r}"
                )
                self._record(
                    request_hash=request_hash,
                    latency_ms=_now_ms() - started,
                    error_category="content",
                )
                raise content_error from exc
            self._record(
                request_hash=request_hash,
                response_hash=_digest(result.vector),
                latency_ms=_now_ms() - started,
                token_count=result.token_count,
            )
            return result

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise LLMConfigError(
                "sentence-transformers is required for local memory retrieval"
            ) from exc
        try:
            return SentenceTransformer(
                self._model_name,
                device=self._device,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:  # noqa: BLE001 - expose local model availability
            raise LLMConfigError(
                f"local sentence-transformer model {self._model_name!r} is unavailable"
            ) from exc

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
                model=self._model_name,
                prompt_version=self._vector_version,
                request_hash=request_hash,
                response_hash=response_hash,
                latency_ms=latency_ms,
                token_count=token_count,
                error_category=error_category,
                sim_time_s=self._sim_time_s,
                scenario_id=self._scenario_id,
            )


def _validate_local_vector(raw_vector: object) -> tuple[float, ...]:
    """Convert a model tensor/array to the same validated vector contract."""
    to_list = getattr(raw_vector, "tolist", None)
    value: object = to_list() if callable(to_list) else raw_vector
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise LLMContentError("sentence-transformer returned a batched vector")
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value or len(value) > _MAX_VECTOR_DIMENSIONS:
        raise LLMContentError("sentence-transformer vector has an invalid dimension")
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise LLMContentError("sentence-transformer vector contains a non-numeric value")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise LLMContentError("sentence-transformer vector contains a non-finite value")
        vector.append(numeric)
    return tuple(vector)


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
