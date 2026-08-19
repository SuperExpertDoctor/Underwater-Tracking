from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from pydantic import ValidationError

from underwater_tracking.api.frame_builder import (
    operational_frame_json,
    operational_frame_payload,
    build_uuv_only_frame,
)
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    ExecutableMissionPlan,
    MissionCandidate,
    PredictionGrid,
    PredictionGridCell,
    RegionMissionState,
    UUVMissionBatch,
)
from underwater_tracking.planning.astar import AStarRoutePlanner, RoutePlan
from underwater_tracking.planning.carrier_tasks import CarrierTaskPlanner
from underwater_tracking.planning.mission_optimizer import MissionOptimizer
from underwater_tracking.runtime.mission_controller import MissionController, MissionSnapshot


@dataclass(frozen=True)
class AcceptanceTrace:
    seed: int
    llm_provider_id: str
    target_ids: tuple[str, ...]
    carrier_ids: tuple[str, ...]
    deployment_stop_ids: tuple[str, ...]
    route_points: tuple[tuple[float, float], ...]
    route_hashes: tuple[str, ...]
    route_avoids_forbidden: bool
    carrier_task_count: int
    plan_revisions: tuple[int, ...]
    plan_hashes: tuple[str, ...]
    grid_revisions: tuple[int, ...]
    lifecycle_trace: tuple[tuple[str, tuple[str, ...]], ...]
    mode_trace: tuple[tuple[str, str], ...]
    event_types: tuple[str, ...]
    frame_hashes: tuple[str, ...]
    frame_payloads: tuple[str, ...]
    legacy_replay_loaded: bool
    malformed_llm_retained_previous_plan: bool
    insufficient_resource_lifecycles: tuple[str, ...]


def run_uuv_only_acceptance(seed: int) -> AcceptanceTrace:
    """Run the complete fixed-seed UUV-only contract without target truth."""
    rng = random.Random(seed)
    route = _deployment_route()
    plan_v1 = _mission_plan(route, revision=1)
    plan_v2 = _mission_plan(route, revision=2)
    plan_v3 = _mission_plan(route, revision=3)
    grid_v1 = _prediction_grid(1, 0.78 + round(rng.random() * 0.05, 3), "transit")
    grid_v2 = _prediction_grid(2, 0.86 + round(rng.random() * 0.05, 3), "evade")

    controller = MissionController(
        scenario_id="uuv-acceptance",
        region_entry_probability_threshold=0.70,
        region_transition_confirm_cycles=2,
        max_uuv_mileage_m=1_000.0,
    )
    assert controller.apply_verified_plan(plan_v1)
    lifecycle_trace: list[tuple[str, tuple[str, ...]]] = []
    mode_trace: list[tuple[str, str]] = []

    def record(label: str, snapshot: MissionSnapshot) -> None:
        lifecycle_trace.append(
            (label, tuple(region.lifecycle.value for region in snapshot.regions))
        )
        mode_trace.extend(
            (f"{label}:{uuv_id}", mode.value)
            for uuv_id, mode in sorted(snapshot.uuv_modes.items())
        )

    record("plan-v1", controller.snapshot())
    record(
        "deployed-r1",
        controller.advance(10, {"deployed_uuv_ids": {"R1": ("U1", "U2")}}),
    )
    record(
        "entry-confirmation-1",
        controller.advance(20, {"entry_probability": {"R1": 0.82}}),
    )
    record(
        "entry-confirmation-2",
        controller.advance(30, {"entry_probability": {"R1": 0.82}}),
    )
    record(
        "deployed-r2",
        controller.advance(40, {"deployed_uuv_ids": {"R2": ("U3", "U4")}}),
    )
    handoff_snapshot = controller.advance(
        50,
        {
            "handoff_ready": {"R1": "R2"},
            "successor_passive_ready": {"R2": True},
        },
    )
    record("handoff-r1-r2", handoff_snapshot)
    mileage_snapshot = controller.advance(60, {"mileage_m": {"U1": 1_001.0}})
    record("mileage-recovery", mileage_snapshot)
    controller.advance(
        70,
        {"target_intent_changed": "T1", "imm_confidence_shifted": "T1"},
    )

    assert controller.apply_verified_plan(plan_v2)
    malformed_llm_output_rejected = False
    try:
        MissionCandidate(
            candidate_id="llm-malformed",
            target_id="T1",
            entry_s=70,
            exit_s=60,
            probability=1.5,
            perimeter_points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        )
    except ValidationError:
        malformed_llm_output_rejected = True
    stale_plan_rejected = not controller.apply_verified_plan(plan_v1)
    malformed_llm_retained = (
        malformed_llm_output_rejected
        and stale_plan_rejected
        and controller.snapshot().plan_revision == 2
    )
    assert controller.apply_verified_plan(plan_v3)
    record("replanned-v3", controller.snapshot())

    insufficient = _insufficient_resource_plan()
    insufficient_lifecycles = tuple(
        assignment.lifecycle.value for assignment in insufficient.region_assignments
    )
    carrier_tasks = CarrierTaskPlanner().build_tasks(
        plan_v1,
        tuple(plan_v1.carrier_missions.values()),
    )
    candidate_regions = {"T1": _mission_candidates()}
    frame_v1 = build_uuv_only_frame(
        snapshot=handoff_snapshot,
        mission=plan_v1,
        prediction_grids=(grid_v1,),
        candidate_regions=candidate_regions,
        events=handoff_snapshot.events,
    )
    final_snapshot = controller.snapshot()
    frame_v3 = build_uuv_only_frame(
        snapshot=final_snapshot,
        mission=plan_v3,
        prediction_grids=(grid_v2,),
        candidate_regions=candidate_regions,
        events=final_snapshot.events,
    )
    frame_payloads = (
        operational_frame_json(frame_v1),
        operational_frame_json(frame_v3),
    )
    legacy_replay_loaded = _legacy_replay_accepts_usv_fields(frame_payloads[-1])
    return AcceptanceTrace(
        seed=seed,
        llm_provider_id="deterministic-test-provider-v1",
        target_ids=("T1",),
        carrier_ids=tuple(sorted(final_snapshot.carrier_missions)),
        deployment_stop_ids=("R1", "R2", "R3"),
        route_points=route.points,
        route_hashes=(_hash_json(route.points),),
        route_avoids_forbidden=_route_avoids_forbidden(route),
        carrier_task_count=len(carrier_tasks),
        plan_revisions=(plan_v1.revision, plan_v2.revision, plan_v3.revision),
        plan_hashes=tuple(_hash_json(plan.model_dump(mode="json")) for plan in (plan_v1, plan_v2, plan_v3)),
        grid_revisions=(grid_v1.revision, grid_v2.revision),
        lifecycle_trace=tuple(lifecycle_trace),
        mode_trace=tuple(mode_trace),
        event_types=tuple(event.event_type for event in final_snapshot.events),
        frame_hashes=tuple(_hash_json(json.loads(payload)) for payload in frame_payloads),
        frame_payloads=frame_payloads,
        legacy_replay_loaded=legacy_replay_loaded,
        malformed_llm_retained_previous_plan=malformed_llm_retained,
        insufficient_resource_lifecycles=insufficient_lifecycles,
    )


def assert_uuv_only_acceptance(trace: AcceptanceTrace) -> None:
    """Single end-to-end assertion used by pytest and the audit record."""
    assert trace.target_ids == ("T1",)
    assert len(trace.carrier_ids) >= 2
    assert len(trace.deployment_stop_ids) >= 3
    assert trace.route_points[0] == trace.route_points[-1]
    assert trace.route_avoids_forbidden
    assert trace.carrier_task_count >= 6
    assert trace.plan_revisions == (1, 2, 3)
    assert trace.grid_revisions == (1, 2)
    assert "DEGRADED" in trace.insufficient_resource_lifecycles
    assert trace.malformed_llm_retained_previous_plan
    assert trace.legacy_replay_loaded
    assert "uuv_range_exhausted" in trace.event_types
    assert "target_intent_changed" in trace.event_types
    assert "imm_confidence_shifted" in trace.event_types
    assert "handoff_completed" in trace.event_types
    assert all("usv" not in payload.lower() for payload in trace.frame_payloads)

    r1_states = [states[0] for _, states in trace.lifecycle_trace]
    assert r1_states.index("ACTIVE_SCAN") < r1_states.index("PASSIVE_TRACK")
    handoff_states = dict(trace.lifecycle_trace)["handoff-r1-r2"]
    assert handoff_states[0] == "TRACKING_COMPLETED"
    assert handoff_states[1] == "PASSIVE_TRACK"
    mode_values = [mode for label, mode in trace.mode_trace if label.endswith(":U1")]
    assert mode_values.index("ACTIVE_SCAN") < mode_values.index("PASSIVE_TRACK")
    assert "RETURN_REQUIRED" in mode_values


def test_fixed_seed_uuv_only_acceptance_trace_is_complete_and_repeatable() -> None:
    first = run_uuv_only_acceptance(20260820)
    second = run_uuv_only_acceptance(20260820)

    assert_uuv_only_acceptance(first)
    assert first == second


def _deployment_route() -> RoutePlan:
    return AStarRoutePlanner(grid_size_m=10.0).plan(
        start=(0.0, 0.0),
        stops=((0.0, 100.0), (100.0, 100.0), (100.0, 0.0)),
        home=(0.0, 0.0),
        forbidden_regions=((20.0, 60.0, 20.0, 80.0), (120.0, 160.0, 40.0, 100.0)),
        map_bounds=(-100.0, 200.0, -100.0, 200.0),
    ) or _raise_route_failure()


def _raise_route_failure() -> RoutePlan:
    raise AssertionError("fixed acceptance route could not be planned")


def _mission_plan(route: RoutePlan, *, revision: int) -> ExecutableMissionPlan:
    region_specs = (
        ("R1", "U1", "U2", (0.0, 100.0), (100.0, 0.0), 10, 100, None, "R2"),
        ("R2", "U3", "U4", (100.0, 100.0), (100.0, 100.0), 110, 200, "R1", "R3"),
        ("R3", "U5", "U6", (100.0, 0.0), (100.0, 0.0), 210, 300, "R2", None),
    )
    assignments = tuple(
        RegionMissionState(
            region_id=region_id,
            target_id="T1",
            active_scan_uuv_ids=(active_id,),
            passive_track_uuv_ids=(passive_id,),
            handoff_from=handoff_from,
            handoff_to=handoff_to,
            plan_revision=revision,
        )
        for region_id, active_id, passive_id, _, _, _, _, handoff_from, handoff_to in region_specs
    )
    batches = tuple(
        UUVMissionBatch(
            carrier_id="carrier_01",
            candidate_id=region_id,
            uuv_ids=(active_id, passive_id),
            active_scan_uuv_ids=(active_id,),
            passive_track_uuv_ids=(passive_id,),
            deployment_point=deployment_point,
            recovery_point=recovery_point,
            entry_s=entry_s,
            exit_s=exit_s,
        )
        for region_id, active_id, passive_id, deployment_point, recovery_point, entry_s, exit_s, _, _ in region_specs
    )
    carrier_01 = CarrierMissionModel(
        carrier_id="carrier_01",
        home_battle_group_id="BG-01",
        route_xy=route.points,
        stop_ids=("R1", "R2", "R3"),
        ready_uuv_ids=("U1", "U2", "U3", "U4", "U5", "U6"),
        reserved_uuv_ids=("U7",),
    )
    carrier_02 = CarrierMissionModel(
        carrier_id="carrier_02",
        home_battle_group_id="BG-01",
        ready_uuv_ids=("U8", "U9"),
    )
    return ExecutableMissionPlan(
        revision=revision,
        uuv_batches_by_carrier={"carrier_01": batches},
        reserved_uuv_ids=("U7",),
        region_assignments=assignments,
        carrier_missions={"carrier_01": carrier_01, "carrier_02": carrier_02},
    )


def _prediction_grid(revision: int, probability: float, intent: str) -> PredictionGrid:
    cell = PredictionGridCell(
        target_id="T1",
        revision=revision,
        grid_x=0,
        grid_y=0,
        min_x=0.0,
        max_x=20.0,
        min_y=0.0,
        max_y=20.0,
        cell_size_m=20.0,
        probability=probability,
        first_entry_s=10,
        last_exit_s=100,
        imm_model_probabilities={"CV": 0.6, "CT": 0.4},
        covariance_summary=(12.0, 8.0, 0.1),
        intent_label=intent,
        intent_confidence=probability,
    )
    return PredictionGrid(
        target_id="T1",
        revision=revision,
        origin=(0.0, 0.0),
        cell_size_m=20.0,
        cells=(cell,),
        centerline_region_ids=(cell.region_id,),
    )


def _mission_candidates() -> tuple[MissionCandidate, ...]:
    regions = (
        ("R1", ((0.0, 100.0), (20.0, 100.0), (20.0, 120.0), (0.0, 120.0)), 10, 100),
        ("R2", ((100.0, 100.0), (120.0, 100.0), (120.0, 120.0), (100.0, 120.0)), 110, 200),
        ("R3", ((100.0, 0.0), (120.0, 0.0), (120.0, 20.0), (100.0, 20.0)), 210, 300),
    )
    return tuple(
        MissionCandidate(
            candidate_id=region_id,
            target_id="T1",
            entry_s=entry_s,
            exit_s=exit_s,
            probability=0.8,
            perimeter_points=points,
            active_scan_uuv_count=1,
            passive_track_uuv_count=1,
        )
        for region_id, points, entry_s, exit_s in regions
    )


def _insufficient_resource_plan() -> ExecutableMissionPlan:
    snapshot = SimpleNamespace(
        snapshot_revision=1,
        uuvs=(
            SimpleNamespace(uuv_id="U1", status="available"),
            SimpleNamespace(uuv_id="U2", status="available"),
        ),
    )
    candidate = MissionCandidate(
        candidate_id="R-resource-starved",
        target_id="T1",
        entry_s=10,
        exit_s=100,
        probability=0.9,
        perimeter_points=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
        active_scan_uuv_count=2,
        passive_track_uuv_count=1,
    )
    return MissionOptimizer().optimize(snapshot, (candidate,))


def _route_avoids_forbidden(route: RoutePlan) -> bool:
    forbidden = ((20.0, 60.0, 20.0, 80.0), (120.0, 160.0, 40.0, 100.0))
    return all(
        not (left < point[0] < right and bottom < point[1] < top)
        for point in route.points
        for left, right, bottom, top in forbidden
    )


def _legacy_replay_accepts_usv_fields(frame_json: str) -> bool:
    payload = json.loads(frame_json)
    payload["usvs"] = [{"usv_id": "USV-legacy", "position": {"x": 0, "y": 0}}]
    with TemporaryDirectory() as directory:
        path = f"{directory}/acceptance.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        frames = ReplayService(path).range()
    return bool(frames) and "usvs" not in operational_frame_payload(frames[0])


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return sha256(encoded).hexdigest()
