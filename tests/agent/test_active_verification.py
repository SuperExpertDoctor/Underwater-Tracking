# tests/agent/test_active_verification.py
"""Deterministic active-sonar verification protocol (spec 17.3, R5).

The node is pure graph logic; the engine's ping simulation is out of
scope (the active-sonar task). These tests drive the state machine with
explicit events and assert the emitted verification commands and the
per-contact state transitions — nearest-first pinger selection, reserved
and unavailable UUV exclusion, submarine dispatch, decoy drop, and the
geometric in-position gate.
"""

from pathlib import Path

from underwater_tracking.agent.nodes.active_verification import (
    ActiveVerificationNode,
    _STATE_CLASSIFIED_SUBMARINE,
    _STATE_IN_POSITION,
    _STATE_VERIFYING,
)
from underwater_tracking.cli import _AgentLoop
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import VerificationCommand
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
from underwater_tracking.planning.reservations import ReservationRegistry
from underwater_tracking.simulation.engine import SimulationEngine
from tests.conftest import CONFIG_PATH
from tests.integration.test_agent_loop import AgentLoop


def _uuv(
    uuv_id: str, x: float, y: float, *, status: UUVStatus = UUVStatus.AVAILABLE
) -> UUVState:
    return UUVState(
        uuv_id=uuv_id,
        position_xy=(x, y),
        heading_rad=0.0,
        speed_mps=2.0,
        energy_fraction=1.0,
        status=status,
        group_id=None,
    )


def _report(target_id: str, member_ids: tuple[str, ...]) -> GroupReport:
    return GroupReport(
        group_id=f"G-{target_id}",
        target_id=target_id,
        sim_time_s=1200,
        member_ids=member_ids,
        belief=TargetBelief(
            target_id=target_id,
            sim_time_s=1200,
            mean=(130.0, 220.0, 1.0, 0.5),
            covariance=(
                (400.0, 0.0, 0.0, 0.0),
                (0.0, 400.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            model_probabilities={"cv": 0.7, "ct": 0.3},
            source_observation_ids=(f"B:{target_id}:1200",),
            fim_min_eigenvalue=0.005,
            fim_condition=12.0,
        ),
        quality=GroupQuality(
            instant=0.8,
            window_mean=0.75,
            ewma=0.76,
            components={"cov": 0.7},
            hard_guard_reasons=(),
        ),
        plan_revision=1,
    )


def _situation(
    uuvs: tuple[UUVState, ...], reports: tuple[GroupReport, ...] = ()
) -> SituationSnapshot:
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=1200,
        uuvs=uuvs,
        group_reports=reports,
        pending_events=(),
    )


def _event(event_type: str, target_id: str, *, payload: dict) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"S1:{event_type}:{target_id}:1200",
        scenario_id="S1",
        sim_time_s=1200,
        event_type=event_type,
        entity_id=target_id,
        level=EventLevel.INFORMATIONAL,
        payload=payload,
    )


def _active_ping(target_id: str, x: float, y: float) -> RuntimeEvent:
    return _event("active_ping", target_id, payload={"position_xy": (x, y)})


def _classified(target_id: str, outcome: str) -> RuntimeEvent:
    return _event("contact_classified", target_id, payload={"outcome": outcome})


def _run(
    node: ActiveVerificationNode, *events: RuntimeEvent
) -> dict[str, object]:
    return node(
        {
            "snapshot_ref": "S1:live",
            "coalesced_events": events,
        }
    )


def test_contact_pings_the_nearest_available_uuv() -> None:
    situation = _situation(
        (
            _uuv("uuv_01", 100.0, 100.0),
            _uuv("uuv_02", 500.0, 500.0),
            _uuv("uuv_03", 1200.0, 1200.0),
        )
    )
    node = ActiveVerificationNode(None, lambda ref: situation)
    result = _run(node, _active_ping("C1", 120.0, 110.0))
    assert result["verification_states"] == {"C1": _STATE_VERIFYING}
    assert result["verification_pingers"] == {"C1": "uuv_01"}
    commands = result["verification_commands"]
    assert len(commands) == 1
    assert commands[0].sensor_mode == "ping"
    assert commands[0].target_id == "C1"
    assert commands[0].uuv_ids == ("uuv_01",)


def test_reserved_uuvs_are_never_picked_as_pingers() -> None:
    reservations = ReservationRegistry()
    reservations.reserve(("uuv_01",), "T1")
    situation = _situation(
        (_uuv("uuv_01", 100.0, 100.0), _uuv("uuv_02", 500.0, 500.0))
    )
    node = ActiveVerificationNode(reservations, lambda ref: situation)
    result = _run(node, _active_ping("C1", 110.0, 110.0))
    assert result["verification_commands"][0].uuv_ids == ("uuv_02",)


def test_failed_and_busy_uuvs_are_skipped() -> None:
    situation = _situation(
        (
            _uuv("uuv_01", 100.0, 100.0, status=UUVStatus.FAILED),
            _uuv("uuv_02", 110.0, 110.0),
        )
    )
    node = ActiveVerificationNode(None, lambda ref: situation)
    first = _run(node, _active_ping("C0", 105.0, 105.0))
    assert first["verification_pingers"] == {"C0": "uuv_02"}
    # uuv_02 is busy pinging C0, so C1 must wait (no command, no state).
    second = _run(node, _active_ping("C1", 105.0, 105.0))
    assert second["verification_commands"] == ()
    assert "C1" not in second["verification_states"]


def test_repeated_ping_events_are_idempotent() -> None:
    situation = _situation((_uuv("uuv_01", 100.0, 100.0),))
    node = ActiveVerificationNode(None, lambda ref: situation)
    first = _run(node, _active_ping("C1", 110.0, 110.0))
    assert first["verification_commands"] != ()
    second = _run(node, _active_ping("C1", 110.0, 110.0))
    assert second["verification_commands"] == ()


def test_submarine_classification_dispatches_and_closes_the_gate() -> None:
    holder: dict[str, SituationSnapshot] = {}
    holder["situation"] = _situation(
        (
            _uuv("uuv_01", 100.0, 100.0),
            _uuv("uuv_02", 300.0, 300.0),
            _uuv("uuv_03", 2000.0, 2000.0),
        )
    )
    node = ActiveVerificationNode(None, lambda ref: holder["situation"])
    first = _run(node, _active_ping("T2", 250.0, 250.0))
    assert first["verification_commands"][0].sensor_mode == "ping"
    assert first["verification_pingers"] == {"T2": "uuv_02"}
    second = _run(node, _classified("T2", "submarine"))
    assert second["verification_states"] == {"T2": _STATE_CLASSIFIED_SUBMARINE}
    dispatch = [
        command
        for command in second["verification_commands"]
        if command.sensor_mode == "dispatch"
    ]
    assert len(dispatch) == 1
    assert dispatch[0].target_id == "T2"
    # The dispatched group forms but uuv_03 is still 2582 m from the belief
    # mean at (130, 220), beyond the 1200 m gate.
    holder["situation"] = _situation(
        (
            _uuv("uuv_01", 100.0, 100.0),
            _uuv("uuv_02", 300.0, 300.0),
            _uuv("uuv_03", 2000.0, 2000.0),
        ),
        reports=(_report("T2", ("uuv_01", "uuv_03")),),
    )
    not_yet = _run(node)
    assert not_yet["verification_states"] == {"T2": _STATE_CLASSIFIED_SUBMARINE}
    assert not_yet["verification_commands"] == ()
    # uuv_03 closes the gate (604 m to the mean at (130, 220)); both members
    # and the original pinger uuv_02 return to passive.
    holder["situation"] = _situation(
        (
            _uuv("uuv_01", 100.0, 100.0),
            _uuv("uuv_02", 300.0, 300.0),
            _uuv("uuv_03", 600.0, 600.0),
        ),
        reports=(_report("T2", ("uuv_01", "uuv_03")),),
    )
    closed = _run(node)
    assert closed["verification_states"] == {"T2": _STATE_IN_POSITION}
    passive = [
        command
        for command in closed["verification_commands"]
        if command.sensor_mode == "return_to_passive"
    ]
    assert len(passive) == 1
    assert set(passive[0].uuv_ids) == {"uuv_01", "uuv_02", "uuv_03"}
    assert closed["verification_pingers"] == {}


def test_decoy_classification_drops_and_releases_the_pinger() -> None:
    situation = _situation((_uuv("uuv_01", 100.0, 100.0),))
    node = ActiveVerificationNode(None, lambda ref: situation)
    _run(node, _active_ping("C3", 110.0, 110.0))
    result = _run(node, _classified("C3", "decoy"))
    modes = {command.sensor_mode for command in result["verification_commands"]}
    assert modes == {"drop", "return_to_passive"}
    assert "C3" not in result["verification_states"]
    assert result["verification_pingers"] == {}


class _FakeEngine:
    """Records every ``set_sensor_mode`` call (no simulation internals)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def set_sensor_mode(
        self, uuv_id: str, mode: str, ping_contact_id: str | None = None
    ) -> None:
        self.calls.append((uuv_id, mode, ping_contact_id))


def _rearm_loops(engine: _FakeEngine) -> tuple[object, object]:
    """The two fixed ``_apply_verification_commands`` call sites (cli + harness)."""
    cli_loop = _AgentLoop.__new__(_AgentLoop)  # type: ignore[attr-defined]
    cli_loop._engine = engine
    harness = AgentLoop.__new__(AgentLoop)  # type: ignore[attr-defined]
    harness._engine = engine
    return cli_loop._apply_verification_commands, harness._apply_verification_commands


def test_rearm_pinger_loop_preserves_contact_ids() -> None:
    """C3 re-arm must unpack dict items, not keys (regression).

    ``verification_pingers`` maps contact id -> pinger uuv id; iterating
    the dict directly unpacks each key string (a 2-char id like "T1"
    becomes contact_id="T", pinger="1", and longer ids raise ValueError),
    so the loop iterates ``.items()``. Both call sites must land the ping
    on the real pinger with the full contact id.
    """
    engine = _FakeEngine()
    result = {
        "verification_commands": (),
        "verification_pingers": {"T1": "uuv_01"},
    }
    for rearm in _rearm_loops(engine):
        rearm(result)
    assert engine.calls == [
        ("uuv_01", "active", "T1"),
        ("uuv_01", "active", "T1"),
    ]


def test_engine_apply_verification_command_mapping(tmp_path: Path) -> None:
    """C2 mapping: ``ping`` -> active with the target, ``return_to_passive`` -> passive."""
    config = load_app_config(CONFIG_PATH)
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.apply_verification_command(
        VerificationCommand(
            command_id="S1:verify:C1:ping:1200",
            target_id="T1",
            sensor_mode="ping",
            uuv_ids=("uuv_00",),
            sim_time_s=1200,
        )
    )
    assert engine._sensor_modes["uuv_00"] == "active"
    assert engine._ping_targets["uuv_00"] == "T1"
    engine.apply_verification_command(
        VerificationCommand(
            command_id="S1:verify:T1:return_to_passive:1200",
            target_id="T1",
            sensor_mode="return_to_passive",
            uuv_ids=("uuv_00",),
            sim_time_s=1200,
        )
    )
    assert engine._sensor_modes["uuv_00"] == "passive"
    assert engine._ping_targets["uuv_00"] is None
