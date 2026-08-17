"""Engine-owned UUV deployment and carrier recovery lifecycle."""

from __future__ import annotations

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import PlanCommand, Waypoint
from underwater_tracking.domain.models import DeploymentState
from underwater_tracking.simulation.engine import SimulationEngine
from tests.conftest import CONFIG_PATH


def _engine(tmp_path) -> SimulationEngine:
    return SimulationEngine(load_app_config(CONFIG_PATH), seed=7, output_dir=tmp_path)


def _command(*, uuv_id: str, action: str) -> PlanCommand:
    return PlanCommand(
        command_id=f"command-{action}",
        plan_id="plan-1",
        plan_revision=1,
        scenario_id="underwater-default",
        group_id="G-target_00",
        target_id="target_00",
        sim_time_s=0,
        member_ids=(uuv_id,),
        waypoints_by_member={uuv_id: (Waypoint(x=1200.0, y=300.0),)},
        actions={uuv_id: action},
    )


def test_recovery_then_deployment_preserves_commanded_waypoint_and_emits_events(tmp_path) -> None:
    engine = _engine(tmp_path)
    uuv_id = "uuv_00"
    commanded_waypoints = [(1200.0, 300.0)]
    engine._uuvs[uuv_id].set_waypoints(commanded_waypoints)

    engine.request_uuv_recovery(uuv_id, reason="rotate")

    assert engine._uuv_state(uuv_id).deployment_state is DeploymentState.RETURNING
    assert engine._uuv_state(uuv_id).group_id is None
    engine._uuvs[uuv_id].position_xy = (-2950.0, -3000.0)
    recovery_frame = engine.step()
    recovered = {uuv["uuv_id"]: uuv for uuv in recovery_frame["uuvs"]}[uuv_id]
    assert recovered["deployment_state"] == "onboard"
    assert recovered["position_xy"] == (-2950.0, -3000.0)
    assert recovered["speed_mps"] == 0.0
    assert uuv_id in recovery_frame["carrier"]["onboard_uuv_ids"]
    assert any(event["event_type"] == "uuv_recovered" for event in recovery_frame["events"])

    engine.request_uuv_deployment(uuv_id, reason="track")
    deployment_frame = engine.step()

    deployed = {uuv["uuv_id"]: uuv for uuv in deployment_frame["uuvs"]}[uuv_id]
    assert deployed["deployment_state"] == "deployed"
    assert engine._uuvs[uuv_id].waypoints == commanded_waypoints
    assert uuv_id in deployment_frame["carrier"]["deployed_uuv_ids"]
    assert any(event["event_type"] == "uuv_deployed" for event in deployment_frame["events"])


@pytest.mark.parametrize("action", ["rotate", "return"])
def test_plan_recovery_actions_request_authoritative_lifecycle(tmp_path, action: str) -> None:
    engine = _engine(tmp_path)
    uuv_id = "uuv_00"

    engine.apply_plan_command(_command(uuv_id=uuv_id, action=action))

    assert engine._uuv_state(uuv_id).deployment_state is DeploymentState.RETURNING
    assert engine._uuv_state(uuv_id).group_id is None


def test_track_plan_action_deploys_an_onboard_uuv(tmp_path) -> None:
    engine = _engine(tmp_path)
    uuv_id = "uuv_00"
    engine.request_uuv_recovery(uuv_id)
    engine._uuvs[uuv_id].position_xy = (-2950.0, -3000.0)
    engine.step()
    assert engine._uuv_state(uuv_id).deployment_state is DeploymentState.ONBOARD

    engine.apply_plan_command(_command(uuv_id=uuv_id, action="track"))

    assert engine._uuv_state(uuv_id).deployment_state is DeploymentState.DEPLOYED
    assert engine._uuv_state(uuv_id).group_id == "target_00"
    assert engine._uuvs[uuv_id].waypoints == [(1200.0, 300.0)]


def test_refresh_situation_projects_post_command_lifecycle_state(tmp_path) -> None:
    config = load_app_config("configs/scenario/segmented_single_target.yaml")
    captured = []
    engine = SimulationEngine(
        config, seed=7, output_dir=tmp_path, carrier=captured.append
    )
    for _ in range(3):
        engine.step()

    original = captured[-1]
    uuv_id = "uuv_00"
    assert original.uuvs[0].deployment_state is DeploymentState.ONBOARD

    engine.apply_plan_command(
        PlanCommand(
            command_id="dispatch-refresh",
            plan_id="plan-1",
            plan_revision=1,
            scenario_id=config.scenario.scenario_id,
            group_id="G-target_00",
            target_id="target_00",
            sim_time_s=original.sim_time_s,
            member_ids=(uuv_id,),
            waypoints_by_member={
                uuv_id: (Waypoint(x=1200.0, y=300.0),),
            },
            actions={uuv_id: "track"},
        )
    )

    refreshed = engine.refresh_situation(original)
    original_uuv = next(state for state in original.uuvs if state.uuv_id == uuv_id)
    refreshed_uuv = next(state for state in refreshed.uuvs if state.uuv_id == uuv_id)

    assert original_uuv.deployment_state is DeploymentState.ONBOARD
    assert refreshed_uuv.deployment_state is DeploymentState.DEPLOYED
    assert refreshed_uuv.group_id == "target_00"
    assert uuv_id in refreshed.carrier.deployed_uuv_ids
    assert uuv_id not in refreshed.carrier.onboard_uuv_ids
    assert any(
        uuv_id in (link.source_id, link.target_id)
        for link in refreshed.platform_snapshot.communication_links
    )


def test_lifecycle_operations_reject_unknown_illegal_and_failed_transitions(tmp_path) -> None:
    engine = _engine(tmp_path)

    with pytest.raises(ValueError, match="unknown uuv 'missing'"):
        engine.request_uuv_recovery("missing")
    with pytest.raises(ValueError, match="cannot deploy uuv 'uuv_00' from deployed"):
        engine.request_uuv_deployment("uuv_00")

    engine.fail_uuv("uuv_00")

    assert engine._uuv_state("uuv_00").deployment_state is DeploymentState.FAILED
    with pytest.raises(ValueError, match="cannot recover uuv 'uuv_00' from failed"):
        engine.request_uuv_recovery("uuv_00")
    with pytest.raises(ValueError, match="cannot deploy uuv 'uuv_00' from failed"):
        engine.request_uuv_deployment("uuv_00")


def test_returning_uuv_is_excluded_from_group_observations(tmp_path) -> None:
    engine = _engine(tmp_path)
    engine.request_uuv_recovery("uuv_00")

    frames = [engine.step() for _ in range(3)]

    target = next(contact for contact in frames[-1]["contacts"] if contact["contact_id"] == "target_00")
    assert all(ray["uuv_id"] != "uuv_00" for ray in target["bearing_rays"])
    assert "uuv_00" not in frames[-1]["assignments"]["target_00"]


@pytest.mark.parametrize(
    "departure_state",
    [DeploymentState.RETURNING, DeploymentState.ONBOARD, DeploymentState.FAILED],
)
def test_replan_replaces_non_deployed_member_in_live_group_and_frame(
    tmp_path, departure_state: DeploymentState
) -> None:
    engine = _engine(tmp_path)
    target_id = "target_00"
    original_members = engine._latest_reports[target_id].member_ids
    returning_uuv = original_members[0]
    replacement_uuv = next(
        uuv_id for uuv_id in sorted(engine._uuvs) if uuv_id not in engine._uuv_groups
    )

    if departure_state is DeploymentState.RETURNING:
        engine.request_uuv_recovery(returning_uuv)
    elif departure_state is DeploymentState.ONBOARD:
        engine.request_uuv_recovery(returning_uuv)
        engine._uuvs[returning_uuv].position_xy = (-2950.0, -3000.0)
        engine.step()
    else:
        engine.fail_uuv(returning_uuv)

    replacement_members = tuple(
        replacement_uuv if member == returning_uuv else member for member in original_members
    )
    engine.apply_plan_command(
        PlanCommand(
            command_id="replace-returning-member",
            plan_id="plan-1",
            plan_revision=2,
            scenario_id="underwater-default",
            group_id="G-target_00",
            target_id=target_id,
            sim_time_s=30,
            member_ids=replacement_members,
        )
    )
    for _ in range(3):
        frame = engine.step()

    report = engine._latest_reports[target_id]
    assert report.member_ids == replacement_members
    assert report.plan_revision == 2
    assert frame["assignments"][target_id] == list(report.member_ids)
    assert frame["assignments"][target_id] == list(replacement_members)
    assert engine._uuv_groups[replacement_uuv] == target_id
    assert returning_uuv not in engine._uuv_groups


def _apply_roster(engine: SimulationEngine, member_ids: tuple[str, ...], revision: int) -> dict[str, object]:
    engine.apply_plan_command(
        PlanCommand(
            command_id=f"roster-{revision}",
            plan_id="plan-1",
            plan_revision=revision,
            scenario_id="underwater-default",
            group_id="G-target_00",
            target_id="target_00",
            sim_time_s=revision * 30,
            member_ids=member_ids,
        )
    )
    frame: dict[str, object] = {}
    for _ in range(3):
        frame = engine.step()
    return frame


def test_committed_plan_grows_group_manager_roster_from_two_to_three(tmp_path) -> None:
    engine = _engine(tmp_path)
    target_id = "target_00"
    initial = engine._latest_reports[target_id].member_ids
    two_members = initial[:2]
    _apply_roster(engine, two_members, revision=2)
    added = next(uuv_id for uuv_id in sorted(engine._uuvs) if uuv_id not in engine._uuv_groups)

    frame = _apply_roster(engine, (*two_members, added), revision=3)

    assert engine._latest_reports[target_id].member_ids == (*two_members, added)
    assert frame["assignments"][target_id] == [*two_members, added]
    assert engine._uuv_groups[added] == target_id


def test_committed_plan_shrinks_group_manager_roster_from_three_to_two(tmp_path) -> None:
    engine = _engine(tmp_path)
    target_id = "target_00"
    initial = engine._latest_reports[target_id].member_ids
    removed = initial[-1]

    frame = _apply_roster(engine, initial[:2], revision=2)

    assert engine._latest_reports[target_id].member_ids == initial[:2]
    assert frame["assignments"][target_id] == list(initial[:2])
    assert removed not in engine._uuv_groups


@pytest.mark.parametrize("replacement", [False, True])
def test_replan_cycle_matches_observations_from_the_precommit_roster(
    tmp_path, replacement: bool
) -> None:
    """A roster command applies after this cycle's existing observers update."""
    engine = _engine(tmp_path)
    target_id = "target_00"
    current_members = engine._latest_reports[target_id].member_ids
    departed_member = current_members[-1]
    replacement_member = next(
        uuv_id for uuv_id in sorted(engine._uuvs) if uuv_id not in engine._uuv_groups
    )
    desired_members = (
        (*current_members[:-1], replacement_member) if replacement else current_members[:-1]
    )

    engine.apply_plan_command(
        PlanCommand(
            command_id="replan-keeps-current-observations",
            plan_id="plan-1",
            plan_revision=2,
            scenario_id="underwater-default",
            group_id="G-target_00",
            target_id=target_id,
            sim_time_s=30,
            member_ids=desired_members,
        )
    )
    engine._observation_cycle(30)

    report = engine._latest_reports[target_id]
    assert report.member_ids == desired_members
    assert f"{target_id}:{departed_member}:30" in report.belief.source_observation_ids


def test_committed_plan_replaces_a_returning_member_with_same_size_roster(tmp_path) -> None:
    engine = _engine(tmp_path)
    target_id = "target_00"
    initial = engine._latest_reports[target_id].member_ids
    returning = initial[0]
    replacement = next(uuv_id for uuv_id in sorted(engine._uuvs) if uuv_id not in engine._uuv_groups)
    engine.request_uuv_recovery(returning)
    desired = (replacement, *initial[1:])

    frame = _apply_roster(engine, desired, revision=2)

    assert engine._latest_reports[target_id].member_ids == desired
    assert frame["assignments"][target_id] == list(desired)
    assert engine._uuv_groups[replacement] == target_id
    assert returning not in engine._uuv_groups
