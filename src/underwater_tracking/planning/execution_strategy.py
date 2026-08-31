"""Constrained LLM strategy revisions for the executable four-slot chain.

The LLM port is deliberately kept at the semantic edge of execution.  This
module validates the response against the current snapshot and returns an
auditable report; it never mutates a mission plan or queues an unconfirmed
proposal.  Deterministic planners remain the only owners of physical layout,
resource membership, and motion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from underwater_tracking.agent.llm import (
    CancelledLLMError,
    LLMConfigError,
    LLMContentError,
    LLMError,
    StructuredLLM,
    TransientLLMError,
)
from underwater_tracking.agent.prompts import (
    EXECUTION_STRATEGY_PROMPT_VERSION,
    EXECUTION_STRATEGY_SYSTEM_PROMPT,
)
from underwater_tracking.domain.regional_models import (
    ExecutionStrategyProposal,
    RegionSlotPolicy,
    StrategyValidationReport,
)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stable_region_ids(target_id: str) -> tuple[str, ...]:
    return tuple(f"{target_id}:task:{index:02d}" for index in range(1, 5))


def _error_fields(exc: ValidationError) -> tuple[str, ...]:
    fields: list[str] = []
    for item in exc.errors():
        location = item.get("loc", ())
        if location:
            field = str(location[0])
            if field == "region_slots":
                field = "region_id"
            if field not in fields:
                fields.append(field)
    return tuple(fields)


def _preserved_revision(current_execution_revision: int | None) -> int | None:
    return current_execution_revision if current_execution_revision is not None else None


def validate_execution_strategy(
    proposal: ExecutionStrategyProposal | Mapping[str, object],
    *,
    target_id: str | None = None,
    allowed_region_ids: Sequence[str] | None = None,
    allowed_evidence_ids: Sequence[str] | None = None,
    allowed_uuv_ids: Sequence[str] = (),
    current_execution_revision: int | None = None,
    current_resource_revision: int | None = None,
    current_manual_revision: int | None = None,
) -> StrategyValidationReport:
    """Validate a semantic proposal without changing the active execution.

    ``allowed_uuv_ids`` is accepted as an explicit boundary argument for
    callers that share a generic validation harness.  The proposal model has
    no resource-ID field; any attempt to add one is rejected while parsing.
    """

    del allowed_uuv_ids
    raw_target_id = (
        target_id
        or (proposal.get("target_id", "") if isinstance(proposal, Mapping) else proposal.target_id)
    )
    preserved = _preserved_revision(current_execution_revision)
    try:
        normalized = (
            proposal
            if isinstance(proposal, ExecutionStrategyProposal)
            else ExecutionStrategyProposal.model_validate(proposal)
        )
    except ValidationError as exc:
        return StrategyValidationReport(
            status="invalid_output",
            target_id=str(raw_target_id),
            base_execution_revision=(
                int(proposal["base_execution_revision"])
                if isinstance(proposal, Mapping)
                and isinstance(proposal.get("base_execution_revision"), int)
                else None
            ),
            preserved_execution_revision=preserved,
            active_plan_preserved=preserved is not None,
            errors=(str(exc),),
            rejected_fields=_error_fields(exc),
        )

    errors: list[str] = []
    rejected_fields: list[str] = []
    expected_ids = tuple(allowed_region_ids or _stable_region_ids(normalized.target_id))
    actual_ids = tuple(slot.region_id for slot in normalized.region_slots)
    if target_id is not None and normalized.target_id != target_id:
        errors.append("proposal target_id does not match the captured target")
        rejected_fields.append("target_id")
    if actual_ids != expected_ids:
        errors.append("proposal must address exactly the captured region slots")
        rejected_fields.append("region_id")
    if allowed_evidence_ids is not None:
        allowed_evidence = set(allowed_evidence_ids)
        cited_ids = set(normalized.evidence_ids)
        for slot in normalized.region_slots:
            cited_ids.update(slot.evidence_ids)
        unknown_evidence = sorted(cited_ids - allowed_evidence)
        if unknown_evidence:
            errors.append(f"proposal cites unknown evidence: {unknown_evidence}")
            rejected_fields.append("evidence_ids")
    if (
        current_execution_revision is not None
        and normalized.base_execution_revision != current_execution_revision
    ):
        errors.append(
            "proposal base_execution_revision does not match the captured execution"
        )
        rejected_fields.append("base_execution_revision")
    if (
        current_resource_revision is not None
        and normalized.resource_revision != current_resource_revision
    ):
        errors.append("proposal resource_revision is stale")
        rejected_fields.append("resource_revision")
    if (
        current_manual_revision is not None
        and normalized.manual_revision != current_manual_revision
    ):
        errors.append("proposal manual_revision is stale")
        rejected_fields.append("manual_revision")

    if errors:
        status = (
            "stale"
            if any(field in rejected_fields for field in ("base_execution_revision",))
            else "resource_conflict"
            if any(field in rejected_fields for field in ("resource_revision", "manual_revision"))
            else "invalid_output"
        )
        return StrategyValidationReport(
            status=status,
            target_id=normalized.target_id,
            proposal=normalized,
            base_execution_revision=normalized.base_execution_revision,
            preserved_execution_revision=preserved,
            active_plan_preserved=preserved is not None,
            errors=tuple(errors),
            rejected_fields=tuple(dict.fromkeys(rejected_fields)),
        )
    return StrategyValidationReport(
        status="validated",
        valid=True,
        proposal=normalized,
        target_id=normalized.target_id,
        base_execution_revision=normalized.base_execution_revision,
        preserved_execution_revision=preserved,
        active_plan_preserved=preserved is not None,
        accepted_region_ids=actual_ids,
    )


class ExecutionStrategyRevisionNode:
    """Run one bounded semantic strategy revision against an immutable input."""

    def __init__(
        self,
        llm: StructuredLLM[Any],
        *,
        model_id: str = "underwater-assistant-model",
        prompt_version: str = EXECUTION_STRATEGY_PROMPT_VERSION,
    ) -> None:
        self._llm = llm
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._pending_suggestions: tuple[ExecutionStrategyProposal, ...] = ()

    @property
    def pending_suggestions(self) -> tuple[ExecutionStrategyProposal, ...]:
        """The strategy node never places unconfirmed output in a mailbox."""

        return self._pending_suggestions

    def build_payload(
        self,
        *,
        target_id: str,
        base_execution_revision: int,
        region_ids: Sequence[str],
        evidence_ids: Sequence[str],
        current_slots: Sequence[RegionSlotPolicy | Mapping[str, object]] = (),
        sim_time_s: int = 0,
        scenario_id: str = "",
        target_position_xy: tuple[float, float] | None = None,
        target_velocity_xy: tuple[float, float] | None = None,
        resource_revision: int = 0,
        manual_revision: int = 0,
    ) -> dict[str, object]:
        expected_ids = _stable_region_ids(target_id)
        supplied_ids = tuple(region_ids)
        if supplied_ids != expected_ids:
            raise ValueError("execution strategy payload requires four stable task slots")
        slot_by_id = {
            slot.region_id: slot
            if isinstance(slot, RegionSlotPolicy)
            else RegionSlotPolicy.model_validate(slot)
            for slot in current_slots
        }
        slots = [
            {
                "region_id": region_id,
                "slot_index": index,
                **(
                    {
                        key: value
                        for key, value in slot_by_id[region_id].model_dump(mode="json").items()
                        if key not in {"region_id", "slot_index", "rationale", "evidence_ids"}
                    }
                    if region_id in slot_by_id
                    else {}
                ),
            }
            for index, region_id in enumerate(supplied_ids, start=1)
        ]
        runtime_frame: dict[str, object] = {"sim_time_s": sim_time_s}
        if target_position_xy is not None:
            runtime_frame["target_position_xy"] = list(target_position_xy)
        if target_velocity_xy is not None:
            runtime_frame["target_velocity_xy"] = list(target_velocity_xy)
        return {
            "model": self._model_id,
            "output_token_budget": 2048,
            "thinking_mode": "disabled",
            "system_prompt": EXECUTION_STRATEGY_SYSTEM_PROMPT,
            "scenario_id": scenario_id,
            "sim_time_s": sim_time_s,
            "target_id": target_id,
            "base_execution_revision": base_execution_revision,
            "resource_revision": resource_revision,
            "manual_revision": manual_revision,
            "region_slots": slots,
            "semantic_constraints": {
                "priority_range": [0.0, 1.0],
                "window_ratio_range": [0.0, 1.0],
                "width_scale_range": [0.5, 2.0],
                "overlap_ratio_range": [0.0, 0.35],
                "allowed_tracking_modes": [
                    "active_scan",
                    "passive_track",
                    "handoff_reserve",
                ],
                "allowed_sonar_modes": ["passive", "active", "passive_then_active"],
                "allowed_task_group_roles": [
                    "passive_tracker",
                    "active_verifier",
                    "handoff_reserve",
                ],
            },
            "runtime_frame": runtime_frame,
            "evidence_ids": sorted({str(item) for item in evidence_ids}),
        }

    def revise(
        self,
        *,
        target_id: str,
        base_execution_revision: int,
        region_ids: Sequence[str],
        evidence_ids: Sequence[str],
        current_execution_revision: int | None = None,
        current_resource_revision: int | None = None,
        current_manual_revision: int | None = None,
        current_slots: Sequence[RegionSlotPolicy | Mapping[str, object]] = (),
        sim_time_s: int = 0,
        scenario_id: str = "",
        target_position_xy: tuple[float, float] | None = None,
        target_velocity_xy: tuple[float, float] | None = None,
    ) -> StrategyValidationReport:
        payload = self.build_payload(
            target_id=target_id,
            base_execution_revision=base_execution_revision,
            region_ids=region_ids,
            evidence_ids=evidence_ids,
            current_slots=current_slots,
            sim_time_s=sim_time_s,
            scenario_id=scenario_id,
            target_position_xy=target_position_xy,
            target_velocity_xy=target_velocity_xy,
            resource_revision=(current_resource_revision or 0),
            manual_revision=(current_manual_revision or 0),
        )
        payload["active_plan_preserved"] = current_execution_revision is not None
        request_hash = _digest(payload)
        common = {
            "target_id": target_id,
            "request_hash": request_hash,
            "model_id": self._model_id,
            "prompt_version": self._prompt_version,
            "base_execution_revision": base_execution_revision,
            "preserved_execution_revision": current_execution_revision,
            "active_plan_preserved": current_execution_revision is not None,
        }
        try:
            result = self._llm.invoke_structured(
                "execution_strategy",
                payload,
                ExecutionStrategyProposal,
                prompt_version=self._prompt_version,
            )
            proposal = (
                result
                if isinstance(result, ExecutionStrategyProposal)
                else ExecutionStrategyProposal.model_validate(result)
            )
        except (TransientLLMError, LLMConfigError, CancelledLLMError):
            # Provider availability is a hard execution prerequisite. The
            # caller must stop the run instead of preserving a stale plan.
            raise
        except (LLMContentError, ValidationError, ValueError, TypeError) as exc:
            return StrategyValidationReport(
                status="invalid_output",
                errors=(str(exc),),
                retry_condition="reissue_structured_strategy",
                failed_fields=("response",),
                **common,
            )
        except LLMError:
            raise

        report = validate_execution_strategy(
            proposal,
            target_id=target_id,
            allowed_region_ids=region_ids,
            allowed_evidence_ids=evidence_ids,
            current_execution_revision=current_execution_revision,
            current_resource_revision=current_resource_revision,
            current_manual_revision=current_manual_revision,
        )
        response_hash = _digest(proposal.model_dump(mode="json"))
        return report.model_copy(
            update={
                **common,
                "response_hash": response_hash,
                "active_plan_preserved": (
                    report.active_plan_preserved or current_execution_revision is not None
                ),
            }
        )

    def __call__(self, state: Mapping[str, object] | None = None, **kwargs: object) -> StrategyValidationReport:
        """Small graph-friendly adapter around :meth:`revise`."""

        arguments = dict(state or {})
        arguments.update(kwargs)
        return self.revise(**arguments)  # type: ignore[arg-type]


ExecutionStrategyService = ExecutionStrategyRevisionNode


__all__ = [
    "ExecutionStrategyRevisionNode",
    "ExecutionStrategyService",
    "validate_execution_strategy",
]
