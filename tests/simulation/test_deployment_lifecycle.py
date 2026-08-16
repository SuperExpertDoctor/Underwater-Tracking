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
