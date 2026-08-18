import json
from types import SimpleNamespace
from typing import Any

import pytest

from underwater_tracking.agent.llm import LLMContentError
from underwater_tracking.agent.nodes.regional_strategy import (
    RegionalStrategyGenerationNode,
    validate_regional_strategy,
)
from underwater_tracking.domain.agent_models import IntentHypothesis
from underwater_tracking.domain.regional_models import (
    CommunicationRequirement,
    GridSpec,
    RegionCell,
    RegionTask,
    RegionalPolicy,
    RegionalStrategySet,
    SonarPolicy,
    TargetRegionPlan,
    TimeWindow,
)


def region_plan() -> TargetRegionPlan:
    cell = RegionCell(
        region_id="T1:cell:0:0", target_id="T1", grid_x=0, grid_y=0,
        min_x=0.0, max_x=100.0, min_y=0.0, max_y=100.0,
        center_xy=(50.0, 50.0), cell_size_m=100.0,
        first_entry_s=100, last_exit_s=180,
        visit_windows=(TimeWindow(start_s=100, end_s=180),),
        evidence_ids=("belief:T1", "intent:T1"),
    )
    task = RegionTask(
        region_id=cell.region_id, target_id="T1",
        active_window=TimeWindow(start_s=100, end_s=180),
        required_quality=0.8, required_uuv_count=1, required_usv_count=1,
        uuv_roles=("passive_tracker",), usv_role="surface_relay",
        sonar_policy=SonarPolicy(passive_required=True, active_allowed=False),
        communication=CommunicationRequirement(), evidence_ids=cell.evidence_ids,
    )
    return TargetRegionPlan(
        target_id="T1", grid_spec=GridSpec(), cell_size_m=100.0,
        cells=(cell,), tasks=(task,), prediction_id="pred:T1",
        intent_label="patrol", intent_confidence=0.8,
        evidence_ids=("belief:T1", "intent:T1"),
    )


def intent() -> IntentHypothesis:
    return IntentHypothesis(
        label="patrol", confidence=0.8, evidence_ids=("intent:T1",),
        model_id="fake", prompt_version="intent-v1",
    )


def policy(region_id: str = "T1:cell:0:0") -> RegionalPolicy:
    return RegionalPolicy(
        region_id=region_id, coverage_mode="required", priority=1.0,
        required_quality=0.8, required_uuv_count=1, required_usv_count=1,
        uuv_roles=("passive_tracker",), usv_role="surface_relay",
        sonar_policy=SonarPolicy(passive_required=True, active_allowed=False),
        communication=CommunicationRequirement(), rationale="preserve passive coverage",
        evidence_ids=("intent:T1",),
    )


class FakeLLM:
    def __init__(self, *, content_failure_once: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.content_failure_once = content_failure_once

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: Any,
        *,
        prompt_version: str = "",
    ) -> Any:
        self.calls.append((operation, payload))
        if self.content_failure_once and len(self.calls) == 1:
            raise LLMContentError("invalid regional policy")
        return response_model(policies=(policy(),))


SNAPSHOT = SimpleNamespace(scenario_id="S1", sim_time_s=100)


def test_regional_payload_contains_geometry_and_no_platform_ids() -> None:
    node = RegionalStrategyGenerationNode(FakeLLM())
    payload = node.build_payload(SNAPSHOT, region_plan(), {"T1": intent()})
    assert [item["region_id"] for item in payload["regions"]] == ["T1:cell:0:0"]
    assert "uuv_ids" not in json.dumps(payload)
    assert payload["regions"][0]["geometry"]["center_xy"] == [50.0, 50.0]


def test_regional_policy_validation_requires_exactly_one_policy_per_region() -> None:
    with pytest.raises(ValueError, match="missing regional policy"):
        validate_regional_strategy(region_plan(), RegionalStrategySet(policies=()))


def test_regional_strategy_rejects_unknown_region() -> None:
    with pytest.raises(ValueError, match="unknown region"):
        validate_regional_strategy(
            region_plan(),
            RegionalStrategySet(policies=(policy("T1:cell:9:9"),)),
        )


def test_regional_strategy_reasks_once_after_content_failure() -> None:
    llm = FakeLLM(content_failure_once=True)
    node = RegionalStrategyGenerationNode(llm)
    result = node.invoke_for_plan(SNAPSHOT, region_plan(), {"T1": intent()})
    assert result.policies[0].region_id == "T1:cell:0:0"
    assert len(llm.calls) == 2
    assert "correction_feedback" in llm.calls[1][1]
