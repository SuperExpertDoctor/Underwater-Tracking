from __future__ import annotations

from collections.abc import Callable

from underwater_tracking.agent.llm import LLMCallMetadata, StructuredLLM
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
        for target_id, prediction in sorted(predictions.items()):
            intent = intents.get(target_id)
            if intent is None:
                raise ValueError(f"region generation requires intent for target {target_id!r}")
            payload = self._payload(
                snapshot, prediction, intent, map_bounds
            )
            proposal_set = self._llm.invoke_structured(
                "task_regions",
                payload,
                TaskRegionProposalSet,
                prompt_version=TASK_REGION_PROMPT_VERSION,
            )
            try:
                plans[target_id] = self._materialize(
                    prediction, intent, proposal_set, map_bounds
                )
            except ValueError as exc:
                # Geometry is planner-owned.  Give the same LLM exactly one
                # chance to correct its coordinates, then re-run the hard
                # grid and trajectory checks.  Never synthesize a region.
                repaired_set = self._llm.invoke_structured(
                    "task_regions",
                    {
                        **payload,
                        "correction_feedback": (
                            f"The previous coordinates were rejected by deterministic "
                            f"geometry validation: {exc}. Return a replacement JSON object "
                            "with one to four rectangles that are non-overlapping after "
                            "1000 m grid alignment and each cover a supplied prediction point."
                        ),
                    },
                    TaskRegionProposalSet,
                    prompt_version=TASK_REGION_PROMPT_VERSION,
                )
                plans[target_id] = self._materialize(
                    prediction, intent, repaired_set, map_bounds
                )
        return {
            "regional_plans": plans,
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
        self, prediction, intent, proposal_set: TaskRegionProposalSet, map_bounds
    ) -> TargetRegionPlan:
        return build_llm_task_region_plan(
            prediction,
            intent,
            proposal_set,
            map_bounds,
            self._grid_spec,
            required_quality=self._required_quality,
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
                "max_regions": 4,
                "grid_alignment_m": 1000.0,
                "regions_must_not_overlap": True,
                "ordered_by_first_covered_prediction_time": True,
                "uuv_demand_policy": "min(4, 1 + ceil(sqrt(cell_count)))",
            },
            "intent": intent.model_dump(mode="json"),
            "prediction": {
                "prediction_id": prediction.prediction_id,
                "points_xy": [list(point) for point in prediction.points_xy],
                "times_s": list(prediction.times_s),
                "corridor_radius_m": list(prediction.corridor_radius_m),
            },
            "evidence_ids": sorted({*prediction.source_belief_history_ids, *intent.evidence_ids, prediction.prediction_id}),
        }


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
            )
            for region in plan.task_regions
        )
    return tuple(
        RegionalMissionCandidate(
            candidate_id=cell.region_id,
            cell_ids=(cell.region_id,),
            time_window=TimeWindow(
                start_s=cell.first_entry_s,
                end_s=max(cell.first_entry_s + 1, cell.last_exit_s),
            ),
            perimeter_points=tuple(
                (
                    (cell.min_x, cell.min_y),
                    (cell.max_x, cell.min_y),
                    (cell.max_x, cell.max_y),
                    (cell.min_x, cell.max_y),
                )
            ),
            predecessor_candidate_ids=tuple(cell.predecessor_region_ids),
            successor_candidate_ids=tuple(cell.successor_region_ids),
        )
        for cell in plan.cells
    )
