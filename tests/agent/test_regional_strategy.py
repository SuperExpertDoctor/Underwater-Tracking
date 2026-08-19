from types import SimpleNamespace
from typing import Any

import pytest

from underwater_tracking.agent.llm import LLMContentError
from underwater_tracking.agent.nodes.regional_strategy import (
    RegionalStrategyGenerationNode,
    _platform_candidates,
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
        assigned_uuv_ids=(), assigned_usv_ids=(),
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
        return response_model(
            policies=tuple(
                policy(item["region_id"])
                for item in payload.get("regions", [])
            )
        )


SNAPSHOT = SimpleNamespace(
    scenario_id="S1",
    sim_time_s=100,
    snapshot_revision=1,
    active_plan=None,
    applied_directives=(),
)


def test_regional_payload_contains_geometry_and_platform_candidates() -> None:
    node = RegionalStrategyGenerationNode(FakeLLM())
    payload = node.build_payload(SNAPSHOT, region_plan(), {"T1": intent()})
    assert [item["region_id"] for item in payload["regions"]] == ["T1:cell:0:0"]
    assert payload["platform_candidates"] == []
    assert payload["regions"][0]["geometry"]["center_xy"] == [50.0, 50.0]


def test_platform_candidates_read_kind_from_capability_state() -> None:
    from pathlib import Path

    from underwater_tracking.config.loader import load_app_config
    from underwater_tracking.simulation.engine import SimulationEngine

    config = load_app_config(Path("configs/scenario/segmented_single_target.yaml"))
    engine = SimulationEngine(config, seed=42)
    snapshot = SimpleNamespace(
        situation=SimpleNamespace(platform_snapshot=engine.platform_snapshot())
    )

    candidates = _platform_candidates(snapshot)

    assert candidates
    assert {candidate["kind"] for candidate in candidates} == {"uuv", "usv"}


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


def test_large_regional_strategy_is_batched_for_structured_output() -> None:
    base = region_plan()
    cells = []
    tasks = []
    for index in range(17):
        region_id = f"T1:cell:{index}:0"
        cells.append(
            base.cells[0].model_copy(
                update={
                    "region_id": region_id,
                    "grid_x": index,
                    "min_x": index * 100.0,
                    "max_x": (index + 1) * 100.0,
                    "center_xy": (index * 100.0 + 50.0, 50.0),
                }
            )
        )
        tasks.append(base.tasks[0].model_copy(update={"region_id": region_id}))
    plan = base.model_copy(update={"cells": tuple(cells), "tasks": tuple(tasks)})
    llm = FakeLLM()
    node = RegionalStrategyGenerationNode(
        llm,
        snapshot_provider=lambda _: SNAPSHOT,
    )

    result = node(
        {
            "snapshot_ref": "snapshot",
            "regional_plans": {"T1": plan},
            "intent_hypotheses": {"T1": intent()},
        }
    )

    assert len(result["regional_policies"]["T1"].policies) == 17
    assert len(llm.calls) == 2
    assert [len(call[1]["regions"]) for call in llm.calls] == [16, 1]
