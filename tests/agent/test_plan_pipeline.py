# tests/agent/test_plan_pipeline.py
"""Versioned plan pipeline tests (spec 15.3, plan Task 7).

Covers the brief's verbatim stale-plan test, the atomic commit contract
(one PlanCommand per group, published only after transaction success), the
periodic ``hold_current`` review with no material improvement, the DEGRADED
emergency plan after a member failure, deterministic candidate selection,
independent commit validation, and the immutable planning snapshot. All
behaviour is deterministic: the same state produces the same candidate and
the same commit outcome.
"""

import json
import math
import sqlite3
from pathlib import Path

import pytest

from underwater_tracking.agent.nodes.commit import CommitNode, CommitResult
from underwater_tracking.agent.nodes.optimize import OptimizeNode, PlanningConfig
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot, SnapshotNode
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import (
    Concept,
    PlanCommand,
    StrategyProposal,
    StrategySet,
    TrackingPlan,
)
from underwater_tracking.domain.models import (
    EventLevel,
    GroupQuality,
    GroupReport,
    RuntimeEvent,
    SituationSnapshot,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository

SCENARIO_ID = "S1"
TRIGGER_EVENT = "E:plan:900"
SIM_TIME_S = 900

# Every UUV sits on its target's waypoint lattice so the planner always has
# a reachable, deterministic lattice point.
TARGET_MEAN = {"T1": (0.0, 0.0), "T2": (2000.0, 0.0)}
UUV_POSITIONS = {
    "U1": (500.0, 0.0),
    "U2": (0.0, 500.0),
    "U3": (1500.0, 0.0),
    "U4": (2500.0, 0.0),
    "U5": (0.0, -1000.0),
    "U6": (3000.0, 0.0),
}


def build_situation(
    *,
    snapshot_revision: int,
    targets: tuple[str, ...] = ("T1",),
    uuv_count: int = 6,
    failed: tuple[str, ...] = (),
    quality: float = 0.8,
    sim_time_s: int = SIM_TIME_S,
    group_members: dict[str, tuple[str, ...]] | None = None,
) -> SituationSnapshot:
    """A deterministic world: ``uuv_count`` UUVs and one group report per target."""
    if group_members is None:
        group_members = {}
    uuvs = tuple(
        UUVState(
            uuv_id=uuv_id,
            position_xy=UUV_POSITIONS[uuv_id],
            heading_rad=0.0,
            speed_mps=20.0,
            energy_fraction=0.9,
            status=UUVStatus.FAILED if uuv_id in failed else UUVStatus.TRACKING,
            group_id=None,
        )
        for uuv_id in tuple(sorted(UUV_POSITIONS))[:uuv_count]
    )
    reports = tuple(
        GroupReport(
            group_id=f"G-{target}",
            target_id=target,
            sim_time_s=sim_time_s,
            member_ids=group_members.get(target, ()),
            belief=TargetBelief(
                target_id=target,
                sim_time_s=sim_time_s,
                mean=(*TARGET_MEAN[target], 1.0, 0.5),
                covariance=(
                    (400.0, 0.0, 0.0, 0.0),
                    (0.0, 400.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                model_probabilities={"cv": 0.7, "ct": 0.3},
                source_observation_ids=(f"B:{target}:900", f"B:{target}:870"),
                fim_min_eigenvalue=0.005,
                fim_condition=12.0,
            ),
            quality=GroupQuality(
                instant=quality,
                window_mean=quality,
                ewma=quality,
                components={"cov": 0.7},
                hard_guard_reasons=(),
            ),
            plan_revision=1,
        )
        for target in targets
    )
    return SituationSnapshot(
        scenario_id=SCENARIO_ID,
        snapshot_revision=snapshot_revision,
        sim_time_s=sim_time_s,
        uuvs=uuvs,
        group_reports=reports,
        pending_events=(),
    )


def build_proposal(concept: Concept, priorities: dict[str, float]) -> StrategyProposal:
    """A verified strategy proposal covering every target with the given priorities."""
    return StrategyProposal(
        concept=concept,
        target_priorities=priorities,
        required_quality={target: 0.7 for target in priorities},
        reinforcement_policy={target: "release_when_stable" for target in priorities},
        releasable_soft_constraints=("energy_reserve_0.1",),
        evidence_ids=tuple(sorted(f"B:{target}:900" for target in priorities)),
        rationale=f"{concept} review",
    )


class _Snapshots:
    """Test-facing snapshot-revision facade over ``PlanRepository``."""

    def __init__(self, repository: PlanRepository) -> None:
        self._repository = repository

    def set_revision(self, scenario_id: str, revision: int) -> None:
        self._repository.set_snapshot_revision(scenario_id, revision)

    def get_revision(self, scenario_id: str) -> int:
        return self._repository.get_snapshot_revision(scenario_id)


class _Commands:
    """Test-facing per-scenario command listing over the shared database."""

    def __init__(self, database_path: Path) -> None:
        self._path = database_path

    def list_for_scenario(self, scenario_id: str) -> list[PlanCommand]:
        conn = sqlite3.connect(self._path)
        try:
            rows = conn.execute(
                "SELECT payload FROM plan_commands WHERE scenario_id = ?"
                " ORDER BY command_id",
                (scenario_id,),
            ).fetchall()
        finally:
            conn.close()
        return [PlanCommand.model_validate(json.loads(row[0])) for row in rows]


class _Repositories:
    """The repository group backing one test: plans, ledger, snapshots, commands."""

    def __init__(self, database_path: Path) -> None:
        self.plans = PlanRepository(database_path)
        self.ledger = DecisionLedger(database_path)
        self.snapshots = _Snapshots(self.plans)
        self.commands = _Commands(database_path)

    def close(self) -> None:
        self.plans.close()
        self.ledger.close()


class _PlanPipeline:
    """Harness composing the snapshot, optimize, and commit nodes (plan Task 7).

    ``make_state`` persists the situation and its revision, then runs the
    SnapshotNode so the immutable planning snapshot is assembled exactly as
    the graph would; ``optimize`` runs the OptimizeNode and returns the
    selected candidate plan; ``commit`` runs the CommitNode over one
    candidate.
    """

    def __init__(self, repositories: _Repositories) -> None:
        self.config = PlanningConfig()
        self.published: list[PlanCommand] = []
        self._repositories = repositories
        self._situation_store: dict[str, SituationSnapshot] = {}
        self._snapshot_store: dict[str, PlanningSnapshot] = {}
        self._candidate_store: dict[str, TrackingPlan] = {}
        self._snapshot_node = SnapshotNode(
            snapshot_provider=lambda ref: self._situation_store[ref],
            active_plan_provider=repositories.plans.get_active,
            directives_provider=lambda scenario_id: repositories.ledger.list_directives(
                scenario_id, status="applied"
            ),
            store=self._snapshot_store,
        )
        self._optimize_node = OptimizeNode(
            snapshot_provider=lambda ref: self._snapshot_store[ref],
            store=self._candidate_store,
            config=self.config,
        )
        self._commit_node = CommitNode(
            repository=repositories.plans,
            snapshot_provider=lambda ref: self._snapshot_store[ref],
            config=self.config,
            publish=self.published.append,
        )

    def make_state(
        self,
        snapshot_revision: int,
        *,
        targets: tuple[str, ...] = ("T1",),
        uuv_count: int = 6,
        failed: tuple[str, ...] = (),
        quality: float = 0.8,
        concepts: tuple[Concept, ...] = ("balanced",),
        priorities: dict[str, float] | None = None,
        route: EventLevel = EventLevel.STRATEGIC,
        group_members: dict[str, tuple[str, ...]] | None = None,
    ) -> CarrierState:
        priorities = priorities if priorities is not None else {t: 1.0 for t in targets}
        situation = build_situation(
            snapshot_revision=snapshot_revision,
            targets=targets,
            uuv_count=uuv_count,
            failed=failed,
            quality=quality,
            group_members=group_members,
        )
        self._situation_store[f"base:{snapshot_revision}"] = situation
        self._repositories.snapshots.set_revision(SCENARIO_ID, snapshot_revision)
        trigger = RuntimeEvent(
            event_id=TRIGGER_EVENT,
            scenario_id=SCENARIO_ID,
            sim_time_s=SIM_TIME_S,
            event_type="state_changed",
            entity_id=SCENARIO_ID,
            level=route,
            payload={},
        )
        state: CarrierState = {
            "scenario_id": SCENARIO_ID,
            "snapshot_revision": snapshot_revision,
            "snapshot_ref": f"base:{snapshot_revision}",
            "route": route,
            "pending_events": (trigger,),
            "coalesced_events": (trigger,),
            "strategy_set": StrategySet(
                trigger_event_ids=(TRIGGER_EVENT,),
                proposals=tuple(
                    build_proposal(concept, priorities) for concept in concepts
                ),
            ),
        }
        return {**state, **self._snapshot_node(state)}

    def optimize(self, state: CarrierState) -> TrackingPlan:
        result = self._optimize_node(state)
        return self._candidate_store[result["selected_plan_ref"]]

    def commit(self, state: CarrierState, candidate: TrackingPlan) -> CommitResult:
        return self._commit_node(state, candidate)


@pytest.fixture
def repositories(tmp_path: Path) -> _Repositories:
    repos = _Repositories(tmp_path / "run.db")
    yield repos
    repos.close()


@pytest.fixture
def plan_pipeline(repositories: _Repositories) -> _PlanPipeline:
    return _PlanPipeline(repositories)


def test_stale_plan_is_rejected_before_broadcast(plan_pipeline, repositories):
    state = plan_pipeline.make_state(snapshot_revision=4)
    candidate = plan_pipeline.optimize(state)
    repositories.snapshots.set_revision("S1", 5)
    result = plan_pipeline.commit(state, candidate)
    assert result["commit_status"] == "stale"
    assert repositories.commands.list_for_scenario("S1") == []


def test_happy_path_commits_one_command_per_group(plan_pipeline, repositories):
    state = plan_pipeline.make_state(snapshot_revision=4, targets=("T1", "T2"))
    candidate = plan_pipeline.optimize(state)
    result = plan_pipeline.commit(state, candidate)
    assert result["commit_status"] == "committed"
    commands = repositories.commands.list_for_scenario("S1")
    assert [command.target_id for command in commands] == ["T1", "T2"]
    assert len(plan_pipeline.published) == 2
    assert {command.command_id for command in plan_pipeline.published} == {
        command.command_id for command in commands
    }
    by_target = {command.target_id: command for command in commands}
    assert by_target["T1"].member_ids == ("U1", "U2")
    assert by_target["T2"].member_ids == ("U3", "U4")
    assert by_target["T1"].actions == {"U1": "track", "U2": "track"}
    assert by_target["T1"].plan_id == "S1:plan:1"
    active = repositories.plans.get_active("S1")
    assert active is not None and active.revision == 1
    # Waypoint contract: in the scenario box, one replan step away, separated.
    for member in ("U1", "U2", "U3", "U4"):
        sequence = active.waypoints_by_member[member]
        assert len(sequence) == 3
        for waypoint in sequence:
            assert -5000.0 <= waypoint.x <= 5000.0
            assert -5000.0 <= waypoint.y <= 5000.0
        first = sequence[0]
        position = UUV_POSITIONS[member]
        assert math.hypot(first.x - position[0], first.y - position[1]) <= 20.0 * 30.0
    pairs = [
        (active.waypoints_by_member["U1"], active.waypoints_by_member["U2"]),
        (active.waypoints_by_member["U3"], active.waypoints_by_member["U4"]),
    ]
    for first_member, second_member in pairs:
        for step in range(3):
            distance = math.hypot(
                first_member[step].x - second_member[step].x,
                first_member[step].y - second_member[step].y,
            )
            assert distance >= 300.0


def test_next_cycle_supersedes_previous_broadcast_plan(plan_pipeline, repositories):
    state = plan_pipeline.make_state(snapshot_revision=4, targets=("T1", "T2"))
    first = plan_pipeline.optimize(state)
    assert plan_pipeline.commit(state, first)["commit_status"] == "committed"
    state = plan_pipeline.make_state(snapshot_revision=5, targets=("T1", "T2"))
    second = plan_pipeline.optimize(state)
    assert plan_pipeline.commit(state, second)["commit_status"] == "committed"
    assert second.revision == 2
    active = repositories.plans.get_active("S1")
    assert active is not None and active.revision == 2
    assert repositories.plans.get_plan("S1:plan:1").status == "superseded"


def test_stable_concept_order_selects_quality_first(plan_pipeline):
    state = plan_pipeline.make_state(
        snapshot_revision=4,
        concepts=("quality_first", "balanced", "resource_saving"),
    )
    candidate = plan_pipeline.optimize(state)
    assert candidate.concept == "quality_first"


def test_optimizer_is_deterministic(plan_pipeline):
    state = plan_pipeline.make_state(
        snapshot_revision=4, targets=("T1", "T2"), concepts=("quality_first", "balanced")
    )
    first = plan_pipeline.optimize(state)
    second = plan_pipeline.optimize(state)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_invalid_candidate_is_rejected_without_write_or_publish(plan_pipeline, repositories):
    state = plan_pipeline.make_state(snapshot_revision=4)
    candidate = plan_pipeline.optimize(state)
    candidate.member_ids_by_target = {"T1": ("U1",)}
    result = plan_pipeline.commit(state, candidate)
    assert result["commit_status"] == "rejected"
    assert any(issue.code == "group_size" for issue in result["issues"])
    assert repositories.plans.get_active("S1") is None
    assert repositories.commands.list_for_scenario("S1") == []
    assert plan_pipeline.published == []


def test_candidate_on_wrong_snapshot_is_rejected(plan_pipeline, repositories):
    state = plan_pipeline.make_state(snapshot_revision=4)
    candidate = plan_pipeline.optimize(state)
    candidate.base_snapshot_revision = 9
    result = plan_pipeline.commit(state, candidate)
    assert result["commit_status"] == "rejected"
    assert any(issue.code == "base_revision_mismatch" for issue in result["issues"])
    assert repositories.plans.get_active("S1") is None


def test_hold_current_keeps_plan_without_new_revision(plan_pipeline, repositories):
    state = plan_pipeline.make_state(
        snapshot_revision=4, group_members={"T1": ("U1", "U2")}
    )
    first = plan_pipeline.optimize(state)
    assert plan_pipeline.commit(state, first)["commit_status"] == "committed"
    state = plan_pipeline.make_state(snapshot_revision=5, concepts=("hold_current",))
    candidate = plan_pipeline.optimize(state)
    result = plan_pipeline.commit(state, candidate)
    assert result["commit_status"] == "hold_current"
    active = repositories.plans.get_active("S1")
    assert active is not None and active.revision == 1
    assert len(repositories.commands.list_for_scenario("S1")) == 1
    assert len(plan_pipeline.published) == 1


def test_failed_member_yields_degraded_emergency_plan(plan_pipeline, repositories):
    state = plan_pipeline.make_state(
        snapshot_revision=4, targets=("T1", "T2"), uuv_count=4
    )
    first = plan_pipeline.optimize(state)
    assert plan_pipeline.commit(state, first)["commit_status"] == "committed"
    state = plan_pipeline.make_state(
        snapshot_revision=5,
        targets=("T1", "T2"),
        uuv_count=4,
        failed=("U2",),
        priorities={"T1": 2.0, "T2": 1.0},
    )
    candidate = plan_pipeline.optimize(state)
    assert candidate.status == "degraded"
    assert set(candidate.member_ids_by_target) == {"T1"}
    # T1 (highest priority) is retained with the cheapest feasible members;
    # the degraded group may hold 2-4, so the allocator keeps two.
    assert candidate.member_ids_by_target["T1"] == ("U1", "U3")
    result = plan_pipeline.commit(state, candidate)
    assert result["commit_status"] == "committed"
    active = repositories.plans.get_active("S1")
    assert active is not None and active.status == "degraded"
    assert active.revision == 2
    commands = repositories.commands.list_for_scenario("S1")
    assert [command.plan_id for command in commands] == [
        "S1:plan:1",
        "S1:plan:1",
        "S1:plan:2",
    ]
    assert [command.target_id for command in commands if command.plan_id == "S1:plan:2"] == ["T1"]


def test_quality_drop_grows_group_with_churn_diff(plan_pipeline, repositories):
    state = plan_pipeline.make_state(snapshot_revision=4)
    first = plan_pipeline.optimize(state)
    assert plan_pipeline.commit(state, first)["commit_status"] == "committed"
    state = plan_pipeline.make_state(snapshot_revision=5, quality=0.6)
    candidate = plan_pipeline.optimize(state)
    assert candidate.member_ids_by_target["T1"] == ("U1", "U2", "U5")
    assert candidate.diff is not None
    assert candidate.diff.members_added == {"T1": ("U5",)}
    assert candidate.diff.from_revision == 1
    assert candidate.diff.to_revision == 2
    assert plan_pipeline.commit(state, candidate)["commit_status"] == "committed"


def test_snapshot_is_immutable_after_assembly(plan_pipeline):
    state = plan_pipeline.make_state(snapshot_revision=4)
    live = plan_pipeline._situation_store["base:4"]
    live.group_reports[0].quality.ewma = 0.99
    candidate = plan_pipeline.optimize(state)
    assert candidate.predicted_quality["T1"] == 0.8
