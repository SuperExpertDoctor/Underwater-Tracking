from __future__ import annotations

from collections.abc import Callable

from underwater_tracking.agent.llm import LLMCallMetadata, LLMContentError, StructuredLLM
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.agent.prompts import (
    TASK_REGION_PROMPT_VERSION,
    TASK_REGION_SYSTEM_PROMPT,
    canonical_digest,
)
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.regional_models import (
    GridSpec,
    RegionalMissionCandidate,
    TaskRegionProposalSet,
    TargetRegionPlan,
    TimeWindow,
)
from underwater_tracking.planning.plan_stability import rectangle_iou
from underwater_tracking.planning.dynamic_regions import (
    DynamicRegionChain,
    build_dynamic_region_chain,
)
from underwater_tracking.planning.regions import build_llm_task_region_plan


class RegionGenerationNode:
    """Build deterministic target region plans from stored prediction references."""

    def __init__(
        self,
        *,
        snapshot_provider: Callable[[str], PlanningSnapshot],
        map_bounds_provider: Callable[[PlanningSnapshot], tuple[float, float, float, float]],
        grid_spec: GridSpec,
        llm: StructuredLLM[TaskRegionProposalSet],
        model_id: str = "underwater-assistant-model",
        required_quality: float = 0.0,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        self._map_bounds_provider = map_bounds_provider
        self._grid_spec = grid_spec
        self._llm = llm
        self._model_id = model_id
        self._required_quality = required_quality

    def __call__(self, state: CarrierState) -> CarrierState:
        snapshot_ref = state.get("snapshot_ref")
        if not snapshot_ref:
            raise ValueError("region generation requires snapshot_ref")
        snapshot = self._snapshot_provider(snapshot_ref)
        intents = state.get("intent_hypotheses", {})
        predictions = state.get("predictions", {})
        map_bounds = self._map_bounds_provider(snapshot)
        plans: dict[str, TargetRegionPlan] = {}
        dynamic_chains: dict[str, DynamicRegionChain] = {}
        prior_chains = state.get("dynamic_region_chains") or {}
        execution_revision = max(
            1, int(state.get("execution_revision", snapshot.snapshot_revision))
        )
        for target_id, prediction in sorted(predictions.items()):
            intent = intents.get(target_id)
            if intent is None:
                raise ValueError(f"region generation requires intent for target {target_id!r}")
            payload = self._payload(snapshot, prediction, intent, map_bounds)
            dynamic_chains[target_id] = build_dynamic_region_chain(
                prediction,
                execution_revision=execution_revision,
                map_bounds_xy=map_bounds,
                previous_chain=prior_chains.get(target_id),
            )
            proposal_set = self._invoke_proposals(payload)
            uuv_scan_range_m = _uuv_active_scan_range_m(snapshot)
            draft_plan = self._materialize_with_correction(
                prediction,
                intent,
                proposal_set,
                map_bounds,
                payload,
                uuv_scan_range_m,
            )
            previous_plan = _previous_target_plan(snapshot, target_id)
            if previous_plan is None:
                plans[target_id] = draft_plan
                continue
            reflection_payload = {
                **payload,
                "rolling_reflection": {
                    "draft_stability": _draft_stability(previous_plan, draft_plan),
                    "draft_robustness": _draft_robustness(draft_plan, prediction),
                    "draft_expected_uuv_allocation": _plan_uuv_demand(draft_plan),
                    "instruction": (
                        "Review the draft against its measured IoU change cost, predicted "
                        "corridor capture, and area-derived UUV demand. Return only the "
                        "least-changed region set that preserves or improves robust tracking "
                        "within the eligible force."
                    ),
                },
            }
            revised_set = self._invoke_proposals(reflection_payload)
            plans[target_id] = self._materialize_with_correction(
                prediction,
                intent,
                revised_set,
                map_bounds,
                reflection_payload,
                uuv_scan_range_m,
            )
        return {
            "regional_plans": plans,
            "dynamic_region_chains": dynamic_chains,
            "regional_candidates": {
                target_id: regional_plan_to_mission_candidates(plan)
                for target_id, plan in sorted(plans.items())
            },
            "llm_provenance": {
                **state.get("llm_provenance", {}),
                **{
                    f"task_regions:{target_id}": LLMCallMetadata(
                        operation="task_regions",
                        model=self._model_id,
                        prompt_version=TASK_REGION_PROMPT_VERSION,
                        request_hash=canonical_digest(
                            self._payload(snapshot, predictions[target_id], intents[target_id], map_bounds)
                        ),
                        response_hash=canonical_digest(
                            [
                                region.model_dump(mode="json")
                                for region in plans[target_id].task_regions
                            ]
                        ),
                        sim_time_s=snapshot.sim_time_s,
                        scenario_id=snapshot.scenario_id,
                    )
                    for target_id in plans
                },
            },
        }

    def _materialize(
        self,
        prediction,
        intent,
        proposal_set: TaskRegionProposalSet,
        map_bounds,
        uuv_scan_range_m: float,
    ) -> TargetRegionPlan:
        return build_llm_task_region_plan(
            prediction,
            intent,
            proposal_set,
            map_bounds,
            self._grid_spec,
            required_quality=self._required_quality,
            uuv_scan_range_m=uuv_scan_range_m,
        )

    def _invoke_proposals(
        self, payload: dict[str, object]
    ) -> TaskRegionProposalSet:
        try:
            return self._llm.invoke_structured(
                "task_regions",
                payload,
                TaskRegionProposalSet,
                prompt_version=TASK_REGION_PROMPT_VERSION,
            )
        except LLMContentError as exc:
            return self._llm.invoke_structured(
                "task_regions",
                {
                    **payload,
                    "correction_feedback": (
                        f"The previous structured response was invalid: {exc}. "
                        "Return exactly four task regions with lower_left_xy and "
                        "upper_right_xy coordinates."
                    ),
                },
                TaskRegionProposalSet,
                prompt_version=TASK_REGION_PROMPT_VERSION,
            )

    def _materialize_with_correction(
        self,
        prediction,
        intent,
        proposal_set: TaskRegionProposalSet,
        map_bounds,
        payload: dict[str, object],
        uuv_scan_range_m: float,
    ) -> TargetRegionPlan:
        try:
            return self._materialize(
                prediction,
                intent,
                proposal_set,
                map_bounds,
                uuv_scan_range_m,
            )
        except ValueError as exc:
            # Geometry is planner-owned. Give the model one bounded chance to
            # correct coordinates, then re-run hard grid and coverage checks.
            repaired_set = self._llm.invoke_structured(
                "task_regions",
                {
                    **payload,
                    "correction_feedback": (
                        f"The previous coordinates were rejected by deterministic geometry "
                        f"validation: {exc}. Return exactly four rectangles at least "
                        "3000 m wide and high. Every rectangle must contain a "
                        "supplied prediction centerline point. Consecutive rectangles need "
                        "a small handoff overlap; non-consecutive rectangles must not overlap."
                    ),
                },
                TaskRegionProposalSet,
                prompt_version=TASK_REGION_PROMPT_VERSION,
            )
            return self._materialize(
                prediction,
                intent,
                repaired_set,
                map_bounds,
                uuv_scan_range_m,
            )

    def _payload(self, snapshot: PlanningSnapshot, prediction, intent, map_bounds) -> dict[str, object]:
        return {
            "model": self._model_id,
            "temperature": 0.2,
            # Four coordinate rectangles need a short structured response;
            # keeping this bounded avoids exhausting a shared master budget.
            "output_token_budget": 1024,
            # Region geometry is a bounded extraction task.  Disable the
            # provider's long reasoning channel so its response budget is
            # reserved for the strict coordinate object.
            "thinking_mode": "disabled",
            "system_prompt": TASK_REGION_SYSTEM_PROMPT,
            "scenario_id": snapshot.scenario_id,
            "sim_time_s": snapshot.sim_time_s,
            "target_id": prediction.target_id,
            "coordinate_system": {
                "name": self._grid_spec.map_coordinate_convention,
                "origin_xy": list(self._grid_spec.origin_xy),
                "map_bounds_xy": list(map_bounds),
                "cell_size_m": 1000.0,
            },
            "task_region_constraints": {
                "region_count": 4,
                "grid_alignment_m": 1000.0,
                "minimum_width_m": 3000.0,
                "minimum_height_m": 3000.0,
                "must_contain_prediction_centerline_sample": True,
                "adjacent_handoff_overlap_required": True,
                "maximum_adjacent_overlap_ratio": 0.35,
                "non_adjacent_regions_must_not_overlap": True,
                "ordered_by_first_covered_prediction_time": True,
                "uuv_demand_policy": (
                    "min(4, max(2, ceil(region_area_m2 / "
                    "(pi * active_scan_range_m^2))))"
                ),
            },
            "rolling_planning_context": _rolling_planning_context(
                snapshot, prediction.target_id
            ),
            "expected_uuv_allocation": _expected_uuv_allocation(snapshot),
            "intent": intent.model_dump(mode="json"),
            "prediction": {
                "prediction_id": prediction.prediction_id,
                "points_xy": [list(point) for point in prediction.points_xy],
                "times_s": list(prediction.times_s),
                "corridor_radius_m": list(prediction.corridor_radius_m),
            },
            "evidence_ids": sorted({*prediction.source_belief_history_ids, *intent.evidence_ids, prediction.prediction_id}),
        }


def _rolling_planning_context(
    snapshot: PlanningSnapshot,
    target_id: str,
) -> dict[str, object]:
    """Expose previous region geometry and task groups before LLM revision."""
    active_plan = getattr(snapshot, "active_plan", None)
    previous_plan = _previous_target_plan(snapshot, target_id)
    assignments = {} if active_plan is None else getattr(active_plan, "region_tasks", {})
    regions: list[dict[str, object]] = []
    for region in (() if previous_plan is None else previous_plan.task_regions):
        assigned_uuv_ids = sorted(
            {
                uuv_id
                for cell_id in region.cell_ids
                for uuv_id in getattr(assignments.get(cell_id), "assigned_uuv_ids", ())
            }
        )
        regions.append(
            {
                "region_id": region.region_id,
                "lower_left_xy": list(region.lower_left_xy),
                "upper_right_xy": list(region.upper_right_xy),
                "cell_ids": list(region.cell_ids),
                "active_window": {
                    "start_s": region.active_window.start_s,
                    "end_s": region.active_window.end_s,
                },
                "required_uuv_count": region.required_uuv_count,
                "assigned_uuv_ids": assigned_uuv_ids,
            }
        )
    return {
        "objective": "minimize_task_region_and_uuv_group_change_subject_to_robust_tracking",
        "iou_retention_threshold": 0.6,
        "prior_task_regions": regions,
        "decision_rule": (
            "Retain an overlapping prior region and its UUV group when it still "
            "covers the predicted corridor. Expand, move, split, or replace it only "
            "when predicted uncertainty, anti-tracking intent, time coverage, or "
            "available force feasibility makes the robustness gain worth the change."
        ),
    }


def _previous_target_plan(
    snapshot: PlanningSnapshot,
    target_id: str,
) -> TargetRegionPlan | None:
    active_plan = getattr(snapshot, "active_plan", None)
    if active_plan is None:
        return None
    return getattr(active_plan, "regional_plans", {}).get(target_id)


def _draft_stability(
    previous_plan: TargetRegionPlan,
    draft_plan: TargetRegionPlan,
) -> dict[str, object]:
    """Summarize post-grid-alignment geometric change for LLM reflection."""
    previous = tuple(previous_plan.task_regions)
    comparisons: list[dict[str, object]] = []
    for region in draft_plan.task_regions:
        bounds = _region_bounds(region)
        old_region, iou = max(
            (
                (old, rectangle_iou(bounds, _region_bounds(old)))
                for old in previous
            ),
            key=lambda item: (item[1], item[0].region_id),
            default=(None, 0.0),
        )
        comparisons.append(
            {
                "draft_region_id": region.region_id,
                "best_prior_region_id": None if old_region is None else old_region.region_id,
                "iou": round(iou, 6),
            }
        )
    mean_iou = (
        sum(float(item["iou"]) for item in comparisons) / len(comparisons)
        if comparisons
        else 0.0
    )
    return {
        "mean_best_iou": round(mean_iou, 6),
        "region_comparisons": comparisons,
    }


def _draft_robustness(
    draft_plan: TargetRegionPlan,
    prediction,
) -> dict[str, object]:
    """Measure how much of each forecast uncertainty corridor fits in a region."""
    captures: list[float] = []
    for point, radius in zip(
        prediction.points_xy, prediction.corridor_radius_m, strict=True
    ):
        capture = max(
            (_corridor_capture(_region_bounds(region), point, float(radius)) for region in draft_plan.task_regions),
            default=0.0,
        )
        captures.append(capture)
    return {
        "corridor_capture": round(sum(captures) / len(captures), 6) if captures else 0.0,
        "covered_prediction_samples": sum(capture > 0.0 for capture in captures),
        "prediction_sample_count": len(captures),
    }


def _plan_uuv_demand(plan: TargetRegionPlan) -> dict[str, object]:
    """Expose the deterministic region-area demand before allocation runs."""
    regions = tuple(plan.task_regions)
    sample_times = sorted(
        {
            time
            for region in regions
            for time in (region.active_window.start_s, region.active_window.end_s - 1)
        }
    )
    peak = max(
        (
            sum(
                region.required_uuv_count
                for region in regions
                if region.active_window.start_s <= time < region.active_window.end_s
            )
            for time in sample_times
        ),
        default=0,
    )
    return {
        "rule": (
            "min(4, max(2, ceil(region_area_m2 / "
            "(pi * active_scan_range_m^2))))"
        ),
        "region_required_uuv_counts": [
            {"region_id": region.region_id, "required_uuv_count": region.required_uuv_count}
            for region in regions
        ],
        "peak_required_uuv_count": peak,
    }


def _region_bounds(region) -> tuple[float, float, float, float]:
    return (
        float(region.lower_left_xy[0]),
        float(region.upper_right_xy[0]),
        float(region.lower_left_xy[1]),
        float(region.upper_right_xy[1]),
    )


def _corridor_capture(
    bounds: tuple[float, float, float, float],
    point: tuple[float, float],
    radius: float,
) -> float:
    if not (bounds[0] <= point[0] <= bounds[1] and bounds[2] <= point[1] <= bounds[3]):
        return 0.0
    clearance = min(
        point[0] - bounds[0], bounds[1] - point[0], point[1] - bounds[2], bounds[3] - point[1]
    )
    return min(1.0, clearance / max(radius, 1.0))


def _expected_uuv_allocation(snapshot: PlanningSnapshot) -> dict[str, object]:
    """Summarize the deployable UUV force used with the area demand rule."""
    situation = getattr(snapshot, "situation", None)
    platform_snapshot = getattr(situation, "platform_snapshot", None)
    roster = getattr(platform_snapshot, "roster", None)
    uuvs = () if roster is None else getattr(roster, "uuvs", ())
    eligible: list[dict[str, object]] = []
    for uuv in uuvs:
        state = getattr(uuv, "deployment_state", "")
        state_value = getattr(state, "value", str(state))
        energy_fraction = float(getattr(uuv, "energy_fraction", 0.0))
        if state_value != "failed" and energy_fraction > 0.0:
            eligible.append(
                {
                    "uuv_id": uuv.platform_id,
                    "deployment_state": state_value,
                    "energy_fraction": energy_fraction,
                    "active_scan_range_m": float(
                        getattr(uuv.capability.sonar, "active_source_range_m", 0.0)
                    ),
                }
            )
    return {
        "rule": "min(4, max(2, ceil(region_area / active_scan_footprint)))",
        "maximum_uuvs_per_region": 4,
        "eligible_uuvs": sorted(eligible, key=lambda item: str(item["uuv_id"])),
        "eligible_uuv_count": len(eligible),
    }


def _uuv_active_scan_range_m(snapshot: PlanningSnapshot) -> float:
    situation = getattr(snapshot, "situation", None)
    platform_snapshot = getattr(situation, "platform_snapshot", None)
    roster = getattr(platform_snapshot, "roster", None)
    ranges = tuple(
        float(uuv.capability.sonar.active_source_range_m)
        for uuv in (() if roster is None else getattr(roster, "uuvs", ()))
        if uuv.capability.sonar.active_capable
    )
    return min(ranges, default=3_500.0)


def regional_plan_to_mission_candidates(
    plan: TargetRegionPlan,
) -> tuple[RegionalMissionCandidate, ...]:
    """Expose planner-owned region geometry as strict UUV candidates."""
    if plan.task_regions:
        return tuple(
            RegionalMissionCandidate(
                candidate_id=region.region_id,
                cell_ids=region.cell_ids,
                time_window=region.active_window,
                perimeter_points=(
                    region.lower_left_xy,
                    (region.upper_right_xy[0], region.lower_left_xy[1]),
                    region.upper_right_xy,
                    (region.lower_left_xy[0], region.upper_right_xy[1]),
                ),
                required_uuv_count=region.required_uuv_count,
                predecessor_candidate_ids=(
                    () if index == 0 else (plan.task_regions[index - 1].region_id,)
                ),
                successor_candidate_ids=(
                    ()
                    if index == len(plan.task_regions) - 1
                    else (plan.task_regions[index + 1].region_id,)
                ),
            )
            for index, region in enumerate(plan.task_regions)
        )
    return tuple(
        RegionalMissionCandidate(
            candidate_id=cell.region_id,
            cell_ids=(cell.region_id,),
            time_window=TimeWindow(
                start_s=cell.first_entry_s,
                end_s=max(cell.first_entry_s + 1, cell.last_exit_s),
            ),
            perimeter_points=(
                (cell.min_x, cell.min_y),
                (cell.max_x, cell.min_y),
                (cell.max_x, cell.max_y),
                (cell.min_x, cell.max_y),
            ),
            predecessor_candidate_ids=tuple(cell.predecessor_region_ids),
            successor_candidate_ids=tuple(cell.successor_region_ids),
        )
        for cell in plan.cells
    )
