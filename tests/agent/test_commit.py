"""Commit-time quality and selective rotation regression tests."""

from underwater_tracking.agent.nodes.commit import CommitNode, build_commands, validate_plan
from underwater_tracking.agent.nodes.optimize import PlanningConfig
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.domain.agent_models import TrackingPlan, Waypoint
from underwater_tracking.domain.models import (
    GroupQuality,
    GroupReport,
    OperationalScheme,
    SituationSnapshot,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.persistence.plans import PlanRepository


def _snapshot() -> PlanningSnapshot:
    report = GroupReport(
        group_id="G-T1",
        target_id="T1",
        sim_time_s=50,
        member_ids=("U1", "U2"),
        belief=TargetBelief(
            target_id="T1",
            sim_time_s=50,
            mean=(0.0, 0.0),
            covariance=((1.0, 0.0), (0.0, 1.0)),
            model_probabilities={"cv": 1.0},
        ),
        quality=GroupQuality(
            instant=0.8,
            window_mean=0.8,
            ewma=0.8,
            components={},
        ),
        plan_revision=1,
    )
    situation = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=4,
        sim_time_s=50,
        uuvs=(
            UUVState(
                uuv_id="U1",
                position_xy=(500.0, 0.0),
                heading_rad=0.0,
                speed_mps=4.0,
                energy_fraction=0.8,
                status=UUVStatus.TRACKING,
            ),
            UUVState(
                uuv_id="U2",
                position_xy=(0.0, 500.0),
                heading_rad=0.0,
                speed_mps=4.0,
                energy_fraction=0.8,
                status=UUVStatus.TRACKING,
            ),
        ),
        group_reports=(report,),
        pending_events=(),
        operational_scheme=OperationalScheme(
            scheme_id="scheme-1",
            version=1,
            minimum_quality={"T1": 0.9},
            valid_from_s=0,
            valid_until_s=100,
        ),
    )
    return PlanningSnapshot(situation, None, ())


def _snapshot_with_speed_and_quality(speed_mps: float, minimum_quality: float) -> PlanningSnapshot:
    snapshot = _snapshot()
    situation = snapshot.situation.model_copy(
        update={
            "uuvs": (
                snapshot.situation.uuvs[0].model_copy(update={"speed_mps": speed_mps}),
                snapshot.situation.uuvs[1].model_copy(update={"speed_mps": speed_mps}),
            ),
            "operational_scheme": snapshot.situation.operational_scheme.model_copy(
                update={"minimum_quality": {"T1": minimum_quality}}
            ),
        }
    )
    return PlanningSnapshot(situation, None, ())


def _plan(**changes: object) -> TrackingPlan:
    fields: dict[str, object] = {
        "plan_id": "S1:plan:1",
        "scenario_id": "S1",
        "revision": 1,
        "base_snapshot_revision": 4,
        "valid_from_s": 50,
        "valid_until_s": 650,
        "target_priorities": {"T1": 1.0},
        "required_quality": {"T1": 0.1},
        "member_ids_by_target": {"T1": ("U1", "U2")},
        "waypoints_by_member": {
            "U1": (Waypoint(x=500.0, y=0.0),),
            "U2": (Waypoint(x=0.0, y=500.0),),
        },
        "predicted_quality": {"T1": 1.0},
    }
    fields.update(changes)
    return TrackingPlan(**fields)


def test_commit_recomputes_scheme_quality_floor_without_candidate_metrics() -> None:
    issues = validate_plan(_snapshot(), _plan(), PlanningConfig())

    assert [(issue.code, issue.expected) for issue in issues if issue.code == "required_quality"] == [
        ("required_quality", ">= 0.900")
    ]


def test_plan_commands_rotate_only_marked_group_members() -> None:
    plan = _plan(
        rotation_conditions={"T1": "energy_reserve_0.3"},
        rotation_uuv_ids=("U1",),
    )

    command = build_commands(_snapshot(), plan)[0]

    assert command.actions == {"U1": "rotate", "U2": "track"}


def test_commit_rejects_rotation_for_non_member() -> None:
    issues = validate_plan(
        _snapshot(),
        _plan(rotation_conditions={"T1": "energy_reserve_0.3"}, rotation_uuv_ids=("U3",)),
        PlanningConfig(),
    )

    assert any(issue.code == "rotation_member" for issue in issues)


def test_commit_recomputes_quality_with_actual_speed_not_capability_max() -> None:
    issues = validate_plan(
        _snapshot_with_speed_and_quality(speed_mps=1.0, minimum_quality=0.7),
        _plan(),
        PlanningConfig(),
    )

    assert any(issue.code == "required_quality" for issue in issues)


def test_commit_rejects_assigned_member_without_passive_sonar() -> None:
    snapshot = _snapshot()
    snapshot.situation.uuvs[0].capability = snapshot.situation.uuvs[0].capability.model_copy(
        update={"passive_sonar_available": False}
    )

    issues = validate_plan(snapshot, _plan(), PlanningConfig())

    assert any(issue.code == "passive_sonar" for issue in issues)


def test_commit_rejects_candidate_when_live_snapshot_is_newer(tmp_path) -> None:
    repository = PlanRepository(tmp_path / "plans.db")
    repository.set_snapshot_revision("S1", 4)
    node = CommitNode(
        repository=repository,
        snapshot_provider=lambda _ref: _snapshot(),
        current_snapshot_revision=lambda: 5,
    )

    result = node({"snapshot_ref": "S1:snapshot:4"}, _plan())

    assert result["commit_status"] == "stale"
    assert repository.get_active("S1") is None
    repository.close()
