from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from underwater_tracking.agent.llm import LLMContentError, TransientLLMError
from underwater_tracking.domain.regional_models import (
    ExecutionStrategyProposal,
    RegionSlotPolicy,
)
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.planning.execution_strategy import (
    ExecutionStrategyRevisionNode,
    validate_execution_strategy,
)


REGION_IDS = tuple(f"T1:task:{index:02d}" for index in range(1, 5))


def _slot(region_id: str, slot_index: int, *, evidence: tuple[str, ...] = ("pred:T1",)) -> RegionSlotPolicy:
    return RegionSlotPolicy(
        region_id=region_id,
        slot_index=slot_index,
        priority=1.0 - (slot_index - 1) * 0.1,
        window_start_ratio=(slot_index - 1) * 0.2,
        window_end_ratio=0.2 + (slot_index - 1) * 0.2,
        width_scale=1.0,
        overlap_ratio=0.1,
        tracking_mode="passive_track",
        sonar_mode="passive",
        task_group_role="passive_tracker",
        reserve_priority=0.2,
        rationale="retain the deterministic slot",
        evidence_ids=evidence,
    )


def _proposal(
    *,
    base_revision: int = 3,
    region_ids: tuple[str, ...] = REGION_IDS,
    evidence: tuple[str, ...] = ("pred:T1",),
) -> ExecutionStrategyProposal:
    return ExecutionStrategyProposal(
        target_id="T1",
        base_execution_revision=base_revision,
        resource_revision=2,
        manual_revision=1,
        region_slots=tuple(
            _slot(region_id, index, evidence=evidence)
            for index, region_id in enumerate(region_ids, start=1)
        ),
        intent_explanation="stable transit calls for compact coverage",
        recommendation="revise",
        rationale="retain the four-slot execution topology",
        evidence_ids=evidence,
    )


class ReturningLLM:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, dict[str, object]]] = []

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        del prompt_version
        self.calls.append((operation, payload))
        if self.error is not None:
            raise self.error
        return response_model.model_validate(self.result)


def test_execution_strategy_is_exactly_four_existing_slots() -> None:
    proposal = _proposal()

    report = validate_execution_strategy(
        proposal,
        allowed_region_ids=REGION_IDS,
        allowed_evidence_ids=("pred:T1",),
        current_execution_revision=3,
        current_resource_revision=2,
        current_manual_revision=1,
    )

    assert report.valid is True
    assert report.status == "validated"
    assert report.accepted_region_ids == REGION_IDS


@pytest.mark.parametrize(
    "extra",
    (
        {"geometry": [[0, 0], [1, 0], [1, 1]]},
        {"assigned_uuv_ids": ["U99"]},
    ),
)
def test_execution_strategy_rejects_geometry_and_platform_assignment(extra: dict[str, object]) -> None:
    payload = _proposal().model_dump(mode="json")
    payload.update(extra)

    with pytest.raises(ValidationError):
        ExecutionStrategyProposal.model_validate(payload)


def test_execution_strategy_rejects_fifth_slot_and_invalid_enum() -> None:
    payload = _proposal().model_dump(mode="json")
    payload["region_slots"].append(
        {**payload["region_slots"][0], "region_id": "T1:task:05", "slot_index": 5}
    )
    with pytest.raises(ValidationError):
        ExecutionStrategyProposal.model_validate(payload)

    invalid = _proposal().model_dump(mode="json")
    invalid["region_slots"][0]["tracking_mode"] = "invented_mode"
    with pytest.raises(ValidationError):
        ExecutionStrategyProposal.model_validate(invalid)


def test_execution_strategy_rejects_unknown_region_and_evidence() -> None:
    raw = _proposal().model_dump(mode="json")
    raw["region_slots"][3]["region_id"] = "T1:task:99"
    region_report = validate_execution_strategy(
        raw,
        allowed_region_ids=REGION_IDS,
        allowed_evidence_ids=("pred:T1",),
        current_execution_revision=3,
        current_resource_revision=2,
        current_manual_revision=1,
    )
    assert region_report.status == "invalid_output"
    assert "region_id" in region_report.rejected_fields

    evidence_report = validate_execution_strategy(
        _proposal(evidence=("not-recorded",)),
        allowed_region_ids=REGION_IDS,
        allowed_evidence_ids=("pred:T1",),
        current_execution_revision=3,
        current_resource_revision=2,
        current_manual_revision=1,
    )
    assert evidence_report.status == "invalid_output"
    assert "evidence_ids" in evidence_report.rejected_fields


def test_timeout_is_propagated_and_does_not_enqueue() -> None:
    llm = ReturningLLM(error=TransientLLMError("provider timeout", category="timeout"))
    node = ExecutionStrategyRevisionNode(llm)

    with pytest.raises(TransientLLMError, match="provider timeout"):
        node.revise(
            target_id="T1",
            base_execution_revision=3,
            region_ids=REGION_IDS,
            evidence_ids=("pred:T1",),
            current_execution_revision=3,
            current_resource_revision=2,
            current_manual_revision=1,
            sim_time_s=90,
            scenario_id="S1",
        )

    assert node.pending_suggestions == ()


def test_invalid_output_and_stale_output_preserve_active_plan() -> None:
    invalid_node = ExecutionStrategyRevisionNode(
        ReturningLLM(error=LLMContentError("invalid JSON"))
    )
    invalid_report = invalid_node.revise(
        target_id="T1",
        base_execution_revision=3,
        region_ids=REGION_IDS,
        evidence_ids=("pred:T1",),
        current_execution_revision=3,
        current_resource_revision=2,
        current_manual_revision=1,
    )
    assert invalid_report.status == "invalid_output"
    assert invalid_report.active_plan_preserved is True

    stale_node = ExecutionStrategyRevisionNode(ReturningLLM(_proposal(base_revision=2).model_dump(mode="json")))
    stale_report = stale_node.revise(
        target_id="T1",
        base_execution_revision=3,
        region_ids=REGION_IDS,
        evidence_ids=("pred:T1",),
        current_execution_revision=3,
        current_resource_revision=2,
        current_manual_revision=1,
    )
    assert stale_report.status == "stale"
    assert stale_report.active_plan_preserved is True


def test_execution_strategy_payload_has_no_geometry_or_uuv_assignment_surface() -> None:
    llm = ReturningLLM(_proposal().model_dump(mode="json"))
    node = ExecutionStrategyRevisionNode(llm)

    report = node.revise(
        target_id="T1",
        base_execution_revision=3,
        region_ids=REGION_IDS,
        evidence_ids=("pred:T1",),
        current_execution_revision=3,
        current_resource_revision=2,
        current_manual_revision=1,
        target_position_xy=(100.0, 200.0),
        target_velocity_xy=(2.0, 0.0),
    )

    assert report.valid is True
    payload = llm.calls[0][1]
    assert "region_slots" in payload
    assert payload["output_token_budget"] == 2048
    assert payload["thinking_mode"] == "disabled"
    assert "geometry" not in str(payload).lower()
    assert "assigned_uuv" not in str(payload).lower()
    assert payload["sim_time_s"] == 0


def test_strategy_attempt_audit_fields_round_trip_through_ledger(tmp_path: Any) -> None:
    ledger = DecisionLedger(tmp_path / "strategy.db")
    ledger.record_llm_call(
        operation="execution_strategy",
        model="model",
        prompt_version="execution-strategy-v1",
        request_hash="request",
        response_hash="response",
        base_execution_revision=3,
        failed_fields=("response",),
        active_plan_preserved=True,
        scenario_id="S1",
    )

    row = ledger.list_llm_calls(operation="execution_strategy")[0]
    assert row.base_execution_revision == 3
    assert row.failed_fields == ("response",)
    assert row.active_plan_preserved is True
    ledger.close()
