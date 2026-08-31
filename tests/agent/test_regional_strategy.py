import json
from threading import Lock
from time import sleep
from types import SimpleNamespace
from typing import Any

import pytest

from underwater_tracking.agent.llm import LLMContentError
from underwater_tracking.agent.graphs.central import RegionalStrategyToStrategySetNode
from underwater_tracking.agent.nodes.regional_strategy import (
    RegionalStrategyGenerationNode,
    _platform_candidates,
    _select_uuv_provider_candidates,
    validate_regional_strategy,
)
from underwater_tracking.agent.nodes.regions import regional_plan_to_mission_candidates
from underwater_tracking.domain.agent_models import IntentHypothesis, IntentMotive
from underwater_tracking.domain.regional_models import (
    CommunicationRequirement,
    GridSpec,
    RegionCell,
    RegionTask,
    TaskRegion,
    RegionalPolicy,
    RegionalStrategySet,
    RegionalMissionCandidate,
    TimeWindow,
    UUVRegionalPolicyDecision,
    UUVRegionalPolicy,
    UUVRegionalStrategySet,
    SonarPolicy,
    TargetRegionPlan,
)
from underwater_tracking.planning.regional_plan_validator import RegionalSemanticRejection


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
        required_quality=0.8, required_uuv_count=1,
        uuv_roles=("passive_tracker",),
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
        ranked_motives=(
            IntentMotive(
                label="persistent_straight_transit",
                probability=0.85,
                rationale="stable heading and speed",
            ),
        ),
        planning_effects=("keep early coverage compact",),
        model_id="fake", prompt_version="intent-v1",
    )


def policy(region_id: str = "T1:cell:0:0") -> RegionalPolicy:
    return RegionalPolicy(
        region_id=region_id, coverage_mode="required", priority=1.0,
        required_quality=0.8, required_uuv_count=1,
        uuv_roles=("passive_tracker",),
        sonar_policy=SonarPolicy(passive_required=True, active_allowed=False),
        communication=CommunicationRequirement(), rationale="preserve passive coverage",
        assigned_uuv_ids=(),
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


class UUVFakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: Any,
        *,
        prompt_version: str = "",
    ) -> Any:
        del prompt_version
        self.calls.append((operation, payload))
        candidate_regions = payload["candidate_regions"]
        assert isinstance(candidate_regions, list)
        return response_model(
            policies=tuple(
                UUVRegionalPolicyDecision(
                    candidate_id=str(item["candidate_id"]),
                    coverage_mode="required",
                    tracking_mode="active_scan",
                    priority=1.0,
                    required_quality=0.8,
                    assigned_uuv_ids=("U1",),
                    rationale="keep the candidate covered",
                    evidence_ids=("intent:T1",),
                )
                for item in candidate_regions
            )
        )


class DelayedUUVLLM:
    """Deterministic test transport whose batches finish out of order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.completion_order: list[int] = []
        self._lock = Lock()
        self._active = 0
        self.max_active = 0

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: Any,
        *,
        prompt_version: str = "",
    ) -> Any:
        del prompt_version
        batch = payload["candidate_batch"]
        assert isinstance(batch, dict)
        batch_index = int(batch["index"])
        with self._lock:
            self.calls.append((operation, payload))
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            # The first three batches start together and intentionally finish
            # in reverse order; the node must still merge by batch index.
            sleep((3 - batch_index) * 0.02 if batch_index < 3 else 0.0)
            candidate_regions = payload["candidate_regions"]
            assert isinstance(candidate_regions, list)
            result = response_model(
                policies=tuple(
                    UUVRegionalPolicyDecision(
                        candidate_id=str(item["candidate_id"]),
                        coverage_mode="required",
                        tracking_mode="active_scan",
                        priority=1.0,
                        required_quality=0.8,
                        active_scan_uuv_count=1,
                        passive_track_uuv_count=0,
                        assigned_uuv_ids=(),
                        rationale="keep the candidate covered",
                        evidence_ids=("intent:T1",),
                    )
                    for item in candidate_regions
                )
            )
            with self._lock:
                self.completion_order.append(batch_index)
            return result
        finally:
            with self._lock:
                self._active -= 1


class SemanticCorrectionLLM:
    def __init__(self, *, repairs: bool) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.repairs = repairs

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: Any,
        *,
        prompt_version: str = "",
    ) -> Any:
        del prompt_version
        self.calls.append((operation, payload))
        candidate_regions = payload["candidate_regions"]
        assert isinstance(candidate_regions, list)
        candidate_id = str(candidate_regions[0]["candidate_id"])
        if not self.repairs or len(self.calls) == 1:
            candidate_id = "T1:unknown-candidate"
        return response_model(
            policies=(
                UUVRegionalPolicyDecision(
                    candidate_id=candidate_id,
                    coverage_mode="required",
                    tracking_mode="active_scan",
                    priority=1.0,
                    required_quality=0.8,
                    assigned_uuv_ids=("U1",),
                    rationale="cover the candidate",
                    evidence_ids=("intent:T1",),
                ),
            )
        )


class PassiveFirstUUVLLM:
    """Return a semantically incomplete first window, then repair it once."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: Any,
        *,
        prompt_version: str = "",
    ) -> Any:
        del prompt_version
        self.calls.append((operation, payload))
        active = len(self.calls) > 1
        candidate_regions = payload["candidate_regions"]
        assert isinstance(candidate_regions, list)
        return response_model(
            policies=tuple(
                UUVRegionalPolicyDecision(
                    candidate_id=str(item["candidate_id"]),
                    coverage_mode="required",
                    tracking_mode="active_scan" if active else "passive_track",
                    priority=1.0,
                    required_quality=0.8,
                    active_scan_uuv_count=1 if active else 0,
                    passive_track_uuv_count=1,
                    assigned_uuv_ids=(),
                    rationale="repair the current search window",
                    evidence_ids=("intent:T1",),
                )
                for item in candidate_regions
            )
        )


def test_uuv_current_window_requires_active_scan_and_gets_one_semantic_repair() -> None:
    llm = PassiveFirstUUVLLM()
    node = RegionalStrategyGenerationNode(llm, uuv_only=True)

    result = node.invoke_for_candidates(
        SNAPSHOT,
        (uuv_candidate(),),
        {"T1": intent()},
        available_uuv_ids={"U1"},
    )

    assert result.policies[0].active_scan_uuv_count == 1
    assert len(llm.calls) == 2
    feedback = llm.calls[1][1]["correction_feedback"]
    assert "active-scan allocation" in feedback["message"]


def test_uuv_semantic_failure_gets_one_bounded_correction() -> None:
    llm = SemanticCorrectionLLM(repairs=True)
    node = RegionalStrategyGenerationNode(llm, uuv_only=True)

    result = node.invoke_for_candidates(
        SNAPSHOT,
        (uuv_candidate(),),
        {"T1": intent()},
        available_uuv_ids={"U1"},
    )

    assert result.policies[0].candidate_id == uuv_candidate().candidate_id
    assert len(llm.calls) == 2
    feedback = llm.calls[1][1]["correction_feedback"]
    assert feedback["category"] == "semantic"
    assert feedback["allowed_candidate_ids"] == [uuv_candidate().candidate_id]


def test_uuv_second_semantic_failure_is_rejected_without_retry_loop() -> None:
    llm = SemanticCorrectionLLM(repairs=False)
    node = RegionalStrategyGenerationNode(llm, uuv_only=True)

    with pytest.raises(RegionalSemanticRejection, match="bounded regional semantic"):
        node.invoke_for_candidates(
            SNAPSHOT,
            (uuv_candidate(),),
            {"T1": intent()},
            available_uuv_ids={"U1"},
        )

    assert len(llm.calls) == 2


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
    assert {candidate["kind"] for candidate in candidates} == {"uuv"}


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
    assert len(llm.calls) == 5
    assert [len(call[1]["regions"]) for call in llm.calls] == [4, 4, 4, 4, 1]


def test_uuv_batches_run_concurrently_but_merge_deterministically() -> None:
    candidates = tuple(
        uuv_candidate().model_copy(
            update={"candidate_id": f"T1:r1:square:{index}:0:1"}
        )
        for index in range(13)
    )
    llm = DelayedUUVLLM()
    node = RegionalStrategyGenerationNode(
        llm,
        snapshot_provider=lambda _: SNAPSHOT,
        uuv_only=True,
        max_concurrency=3,
    )

    batch_results = node._run_uuv_batches(
        SNAPSHOT,
        "T1",
        tuple(candidates[index : index + 4] for index in range(0, len(candidates), 4)),
        {"T1": intent()},
        {"U1"},
    )

    policies = tuple(
        policy
        for _, decisions in batch_results
        for policy in decisions.policies
    )
    assert [policy.candidate_id for policy in policies] == [
        candidate.candidate_id for candidate in candidates
    ]
    assert llm.max_active == 3
    assert llm.completion_order[0] == 2


def test_parallel_uuv_batches_receive_disjoint_resource_pools() -> None:
    candidates = tuple(
        uuv_candidate().model_copy(
            update={"candidate_id": f"T1:r1:square:{index}:0:1"}
        )
        for index in range(4)
    )
    llm = DelayedUUVLLM()
    node = RegionalStrategyGenerationNode(llm, uuv_only=True)

    node._run_uuv_batches(
        SNAPSHOT,
        "T1",
        (candidates[:2], candidates[2:]),
        {"T1": intent()},
        {f"U{index}": {"active_capable": True} for index in range(4)},
    )

    pools = [
        {
            str(platform["platform_id"])
            for platform in payload["platform_candidates"]
        }
        for _, payload in sorted(
            llm.calls,
            key=lambda item: item[1]["candidate_batch"]["index"],
        )
    ]
    assert pools == [{"U0", "U2"}, {"U1", "U3"}]
    assert pools[0].isdisjoint(pools[1])


def test_uuv_provider_input_is_bounded_and_protects_active_regions() -> None:
    candidates = tuple(
        uuv_candidate().model_copy(
            update={
                "candidate_id": f"T1:r1:square:{index}:0:1",
                "predecessor_candidate_ids": (
                    (f"T1:r1:square:{index - 1}:0:1",) if index else ()
                ),
                "successor_candidate_ids": (
                    (f"T1:r1:square:{index + 1}:0:1",)
                    if index < 7
                    else ()
                ),
            }
        )
        for index in range(8)
    )
    active_region_id = candidates[6].candidate_id
    active_snapshot = SimpleNamespace(
        scenario_id="S1",
        sim_time_s=100,
        snapshot_revision=1,
        active_plan=SimpleNamespace(
            region_tasks={
                active_region_id: SimpleNamespace(
                    region_id=active_region_id,
                    target_id="T1",
                    assignment_status="active",
                )
            }
        ),
        applied_directives=(),
    )
    llm = EmptyAssignmentUUVLLM()
    node = RegionalStrategyGenerationNode(
        llm,
        snapshot_provider=lambda _: active_snapshot,
        uuv_only=True,
    )

    result = node(
        {
            "snapshot_ref": "snapshot",
            "regional_candidates": {"T1": candidates},
            "intent_hypotheses": {"T1": intent()},
        }
    )

    payload_candidates = [
        candidate
        for _, payload in llm.calls
        for candidate in payload["candidate_regions"]
    ]
    assert all(
        len(payload["candidate_regions"]) <= 2
        for _, payload in llm.calls
    )
    selected_ids = [str(item["candidate_id"]) for item in payload_candidates]
    assert len(selected_ids) == 4
    assert active_region_id in selected_ids
    selected_set = set(selected_ids)
    assert all(
        relation in selected_set
        for item in payload_candidates
        for relation in (
            *item["predecessor_candidate_ids"],
            *item["successor_candidate_ids"],
        )
    )
    assert len(result["regional_policies"]["T1"].policies) == 4


def test_uuv_provider_input_spans_temporal_search_window_without_active_region() -> None:
    candidates = tuple(
        uuv_candidate().model_copy(
            update={
                "candidate_id": f"T1:r1:square:{index}:0:1",
                "time_window": TimeWindow(start_s=index * 100, end_s=index * 100 + 80),
            }
        )
        for index in range(8)
    )
    snapshot = SimpleNamespace(active_plan=None)

    selected = _select_uuv_provider_candidates(
        candidates,
        snapshot=snapshot,
        target_id="T1",
    )

    assert len(selected) == 4
    starts = tuple(candidate.time_window.start_s for candidate in selected)
    assert starts == (0, 100, 200, 300)
    assert starts[-1] < candidates[-1].time_window.start_s


def test_uuv_provider_input_follows_a_public_spatial_temporal_path() -> None:
    def candidate(
        candidate_id: str,
        start_s: int,
        offset_m: float,
    ) -> RegionalMissionCandidate:
        return uuv_candidate().model_copy(
            update={
                "candidate_id": candidate_id,
                "time_window": TimeWindow(start_s=start_s, end_s=start_s + 80),
                "perimeter_points": (
                    (offset_m, 0.0),
                    (offset_m, 100.0),
                    (offset_m + 100.0, 0.0),
                    (offset_m + 100.0, 100.0),
                ),
            }
        )

    path = tuple(
        candidate(f"T1:path:{index}", index * 100, index * 100.0)
        for index in range(4)
    )
    distractors = tuple(
        candidate(f"T1:far:{index}", index * 100, 10_000.0)
        for index in range(4)
    )
    snapshot = SimpleNamespace(
        active_plan=None,
        situation=SimpleNamespace(
            sim_time_s=0,
            target_search_priors=(
                SimpleNamespace(
                    target_id="T1",
                    issued_at_s=0,
                    valid_until_s=1_000,
                    center_xy=(50.0, 50.0),
                    confidence=0.9,
                ),
            ),
        ),
    )

    selected = _select_uuv_provider_candidates(
        path + distractors,
        snapshot=snapshot,
        target_id="T1",
    )

    assert [candidate.candidate_id for candidate in selected] == [
        "T1:path:0",
        "T1:path:1",
        "T1:path:2",
        "T1:path:3",
    ]
    assert all(
        candidate.successor_candidate_ids == (
            selected[index + 1].candidate_id,
        )
        if index < len(selected) - 1
        else candidate.successor_candidate_ids == ()
        for index, candidate in enumerate(selected)
    )
    assert all(
        candidate.predecessor_candidate_ids == (
            selected[index - 1].candidate_id,
        )
        if index
        else candidate.predecessor_candidate_ids == ()
        for index, candidate in enumerate(selected)
    )


def test_uuv_provider_input_preserves_nearest_public_prior_candidate() -> None:
    candidates = tuple(
        uuv_candidate().model_copy(
            update={
                "candidate_id": f"T1:r1:square:{index}:0:1",
                "time_window": TimeWindow(start_s=index * 100, end_s=index * 100 + 80),
                "perimeter_points": (
                    (index * 100.0, 0.0),
                    (index * 100.0, 100.0),
                    (index * 100.0 + 100.0, 0.0),
                    (index * 100.0 + 100.0, 100.0),
                ),
            }
        )
        for index in range(8)
    )
    snapshot = SimpleNamespace(
        active_plan=None,
        situation=SimpleNamespace(
            sim_time_s=600,
            target_search_priors=(
                SimpleNamespace(
                    prior_id="prior:T1",
                    target_id="T1",
                    issued_at_s=0,
                    valid_until_s=1_000,
                    center_xy=(650.0, 50.0),
                    confidence=0.9,
                ),
            ),
        ),
    )

    selected = _select_uuv_provider_candidates(
        candidates,
        snapshot=snapshot,
        target_id="T1",
    )

    assert len(selected) == 4
    assert "T1:r1:square:6:0:1" in {candidate.candidate_id for candidate in selected}


def uuv_candidate() -> RegionalMissionCandidate:
    return RegionalMissionCandidate(
        candidate_id="T1:r1:square:0:0:1",
        cell_ids=("T1:r1:cell:0:0",),
        time_window=TimeWindow(start_s=100, end_s=180),
        perimeter_points=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
    )


def test_uuv_only_payload_contains_no_legacy_platform_or_policy_fields() -> None:
    node = RegionalStrategyGenerationNode(UUVFakeLLM(), uuv_only=True)

    payload = node.build_payload(SNAPSHOT, region_plan(), {"T1": intent()})

    assert payload["operational_constraints"]["allowed_tracking_modes"] == [
        "active_scan",
        "passive_track",
        "handoff_reserve",
    ]
    assert payload["operational_constraints"]["provider_candidate_cap"] == 4
    assert payload["operational_constraints"]["provider_batch_cap"] == 2
    assert payload["operational_constraints"]["executable_region_cap"] == 4
    assert payload["operational_constraints"]["resource_allocation"] == (
        "deterministic_mission_optimizer"
    )
    assert "final resource allocation is deterministic" in payload["system_prompt"]
    assert "assigned_usv_ids" not in json.dumps(payload)
    assert all(item["kind"] == "uuv" for item in payload["platform_candidates"])


def test_uuv_candidate_payload_publishes_only_square_corner_coordinates() -> None:
    payload = RegionalStrategyGenerationNode(
        UUVFakeLLM(), uuv_only=True
    )._candidate_payload(uuv_candidate())

    assert payload["top_left_xy"] == [0.0, 100.0]
    assert payload["bottom_right_xy"] == [100.0, 0.0]
    assert "perimeter_points" not in payload


def test_uuv_payload_exposes_intent_motives_and_prior_plan_change_cost() -> None:
    prior = region_plan()
    prior_region = TaskRegion(
        region_id="T1:task:01",
        lower_left_xy=(0.0, 0.0),
        upper_right_xy=(100.0, 100.0),
        cell_ids=("T1:cell:0:0",),
        active_window=TimeWindow(start_s=100, end_s=180),
        required_uuv_count=2,
        rationale="previous coverage",
    )
    prior_task = prior.tasks[0].model_copy(update={"assigned_uuv_ids": ("U1", "U2")})
    prior = prior.model_copy(
        update={"task_regions": (prior_region,), "tasks": (prior_task,)}
    )
    snapshot = SimpleNamespace(
        **{
            **vars(SNAPSHOT),
            "active_plan": SimpleNamespace(
            regional_plans={"T1": prior},
            region_tasks={prior_task.region_id: prior_task},
            ),
        }
    )

    payload = RegionalStrategyGenerationNode(UUVFakeLLM(), uuv_only=True).build_uuv_payload(
        snapshot, (uuv_candidate(),), {"T1": intent()}
    )

    assert payload["output_token_budget"] == 1024
    assert payload["thinking_mode"] == "disabled"
    assert payload["intent"]["ranked_motives"][0]["probability"] == 0.85
    comparison = payload["regional_context"]["rolling_change_control"][
        "candidate_comparisons"
    ][0]
    assert comparison["iou_with_previous"] == 1.0
    assert comparison["previous_assigned_uuv_ids"] == ["U1", "U2"]


def test_uuv_strategy_uses_second_reflection_pass_when_prior_region_exists() -> None:
    prior = region_plan()
    prior_region = TaskRegion(
        region_id="T1:task:01",
        lower_left_xy=(0.0, 0.0),
        upper_right_xy=(100.0, 100.0),
        cell_ids=("T1:cell:0:0",),
        active_window=TimeWindow(start_s=100, end_s=180),
        required_uuv_count=1,
        rationale="previous coverage",
    )
    prior = prior.model_copy(update={"task_regions": (prior_region,)})
    snapshot = SimpleNamespace(
        **{
            **vars(SNAPSHOT),
            "active_plan": SimpleNamespace(
                regional_plans={"T1": prior},
                region_tasks={prior.tasks[0].region_id: prior.tasks[0]},
            ),
        }
    )
    llm = UUVFakeLLM()

    RegionalStrategyGenerationNode(llm, uuv_only=True).invoke_for_candidates(
        snapshot,
        (uuv_candidate(),),
        {"T1": intent()},
        available_uuv_ids={"U1"},
    )

    assert len(llm.calls) == 2
    assert "rolling_reflection" in llm.calls[1][1]


def test_uuv_strategy_uses_only_generated_candidates_and_is_validated() -> None:
    llm = UUVFakeLLM()
    node = RegionalStrategyGenerationNode(llm, uuv_only=True)

    result = node.invoke_for_candidates(
        SNAPSHOT,
        (uuv_candidate(),),
        {"T1": intent()},
        available_uuv_ids={"U1"},
    )

    assert isinstance(result, UUVRegionalStrategySet)
    assert result.policies[0].candidate_id == "T1:r1:square:0:0:1"
    assert llm.calls[0][1]["candidate_regions"][0]["candidate_id"] == result.policies[0].candidate_id


class EmptyAssignmentUUVLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: Any,
        *,
        prompt_version: str = "",
    ) -> Any:
        del prompt_version
        self.calls.append((operation, payload))
        return response_model(
            policies=tuple(
                UUVRegionalPolicyDecision(
                    candidate_id=str(item["candidate_id"]),
                    coverage_mode="required",
                    tracking_mode="active_scan",
                    priority=1.0,
                    required_quality=0.8,
                    active_scan_uuv_count=1,
                    passive_track_uuv_count=0,
                    assigned_uuv_ids=(),
                    rationale="leave the final resource selection to the optimizer",
                    evidence_ids=("intent:T1",),
                )
                for item in payload["candidate_regions"]
            )
        )


def test_uuv_only_strategy_bounds_provider_input_and_allows_optimizer_selection() -> None:
    candidates = tuple(
        uuv_candidate().model_copy(
            update={"candidate_id": f"T1:r1:square:{index}:0:1"}
        )
        for index in range(17)
    )
    llm = EmptyAssignmentUUVLLM()
    node = RegionalStrategyGenerationNode(
        llm,
        snapshot_provider=lambda _: SNAPSHOT,
        uuv_only=True,
    )

    result = node(
        {
            "snapshot_ref": "snapshot",
            "regional_candidates": {"T1": candidates},
            "intent_hypotheses": {"T1": intent()},
        }
    )

    policies = result["regional_policies"]["T1"].policies
    assert len(policies) == 4
    assert all(not policy.assigned_uuv_ids for policy in policies)
    assert [len(call[1]["candidate_regions"]) for call in llm.calls] == [2, 2]


def test_strategy_adapter_accepts_validated_uuv_policy_sets() -> None:
    uuv_policy = UUVRegionalPolicy(
        candidate_id="T1:cell:0:0",
        coverage_mode="required",
        tracking_mode="active_scan",
        priority=1.0,
        required_quality=0.8,
        assigned_uuv_ids=("U1",),
        rationale="candidate remains covered",
        evidence_ids=("intent:T1",),
    )

    result = RegionalStrategyToStrategySetNode()(
        {
            "regional_plans": {"T1": region_plan()},
            "regional_policies": {
                "T1": UUVRegionalStrategySet(policies=(uuv_policy,))
            },
        }
    )

    assert result["strategy_set"].proposals[0].required_quality == {"T1": 0.8}


def test_region_generation_exposes_immutable_uuv_mission_candidates() -> None:
    candidates = regional_plan_to_mission_candidates(region_plan())

    assert [candidate.candidate_id for candidate in candidates] == [
        "T1:cell:0:0"
    ]
    assert candidates[0].cell_ids == ("T1:cell:0:0",)
    assert candidates[0].time_window.start_s == 100
    assert candidates[0].perimeter_points == (
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 100.0),
        (0.0, 100.0),
    )
