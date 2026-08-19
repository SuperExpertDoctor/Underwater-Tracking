"""Role-aware construction of the real structured LLM clients.

The carrier, group-slave and adversary brains deliberately do not share a
flat provider configuration.  ``build_role_llm`` resolves one of the three
explicit role entries and binds every role-specific client setting to a real
``HTTPStructuredLLM`` instance.  A legacy flat ``LLMConfig`` is rejected so a
missing role cannot silently change the brain that is running.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeVar

import httpx
from pydantic import BaseModel

from underwater_tracking.agent.llm import (
    HTTPStructuredLLM,
    LLMCallMetadata,
)
from underwater_tracking.config.models import LLMConfig, LLMRoleName
from underwater_tracking.persistence.ledger import DecisionLedger


RoleName = Literal["master", "slave", "adversary"]
T = TypeVar("T", bound=BaseModel)


class RoleHTTPStructuredLLM(HTTPStructuredLLM):
    """A real HTTP client carrying the selected role's prompt contract.

    The HTTP transport, retries, response validation and ledger behavior all
    remain those of ``HTTPStructuredLLM``.  The subclass only adds the role
    identity and makes the configured prompt version the default when a call
    site does not provide one explicitly.
    """

    def __init__(
        self,
        *,
        role: LLMRoleName,
        prompt_version: str,
        base_url: str,
        model: str = "",
        api_key_env: str,
        api_key: str | None = None,
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
        max_tokens: int = 4096,
        before_request: Callable[[LLMCallMetadata], None] | None = None,
        after_response: Callable[[LLMCallMetadata], None] | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            api_key=api_key,
            request_timeout_s=request_timeout_s,
            connect_timeout_s=connect_timeout_s,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_max_s=backoff_max_s,
            jitter=jitter,
            transport=transport,
            ledger=ledger,
            scenario_id=scenario_id,
            sim_time_s=sim_time_s,
            temperature=temperature,
            max_tokens=max_tokens,
            before_request=before_request,
            after_response=after_response,
        )
        self._role = role
        self._prompt_version = prompt_version

    @property
    def role(self) -> LLMRoleName:
        """The explicit role used to construct this client."""

        return self._role

    @property
    def prompt_version(self) -> str:
        """The role-specific prompt contract version."""

        return self._prompt_version

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[T],
        *,
        prompt_version: str = "",
    ) -> T:
        """Use the role prompt version unless the caller supplies one."""

        return super().invoke_structured(
            operation,
            payload,
            response_model,
            prompt_version=prompt_version or self._prompt_version,
        )


def build_role_llm(
    config: LLMConfig,
    role: RoleName | str,
    *,
    request_timeout_s: float | None = None,
    max_retries: int | None = None,
    ledger: DecisionLedger | None = None,
    scenario_id: str = "",
    sim_time_s: int = 0,
    before_request: Callable[[LLMCallMetadata], None] | None = None,
    after_response: Callable[[LLMCallMetadata], None] | None = None,
) -> RoleHTTPStructuredLLM:
    """Build the real HTTP client for one configured brain role.

    Authentication remains a shared provider concern on ``LLMConfig``.  All
    behavior that can differ between brains is read from the selected
    ``LLMRoleConfig``: model, endpoint, timeouts, retry/backoff limits, token
    budget, temperature and prompt version.

    ``LLMConfig`` still accepts legacy flat settings for compatibility with
    unrelated configuration loading.  This factory intentionally does not:
    constructing a role client without all three explicit roles is an
    operational configuration error, never a reason to use the flat values.
    """

    if not isinstance(config, LLMConfig):
        raise TypeError("build_role_llm requires an LLMConfig instance")
    if config.roles is None:
        raise ValueError(
            "role-aware LLM construction requires explicit config.roles; "
            "flat LLM settings cannot be used as a fallback"
        )
    if role not in ("master", "slave", "adversary"):
        raise ValueError(
            f"unknown LLM role {role!r}; expected master, slave or adversary"
        )

    role_name: LLMRoleName = role
    role_config = config.for_role(role_name)
    return RoleHTTPStructuredLLM(
        role=role_name,
        prompt_version=role_config.prompt_version,
        base_url=role_config.base_url,
        model=role_config.model,
        api_key_env=config.api_key_env,
        api_key=config.api_key,
        request_timeout_s=(
            role_config.request_timeout_s
            if request_timeout_s is None
            else request_timeout_s
        ),
        connect_timeout_s=role_config.connect_timeout_s,
        max_retries=role_config.max_retries if max_retries is None else max_retries,
        backoff_base_s=role_config.backoff_base_s,
        backoff_max_s=role_config.backoff_max_s,
        temperature=role_config.temperature,
        max_tokens=role_config.max_tokens,
        ledger=ledger,
        scenario_id=scenario_id,
        sim_time_s=sim_time_s,
        before_request=before_request,
        after_response=after_response,
    )


__all__ = ["RoleHTTPStructuredLLM", "RoleName", "build_role_llm"]
