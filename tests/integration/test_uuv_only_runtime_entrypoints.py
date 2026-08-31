from __future__ import annotations

import json
from types import SimpleNamespace

from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.cli import _AgentLoop
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.planning.reservations import ReservationRegistry
from underwater_tracking.simulation.engine import SimulationEngine


class _RecordingEngine:
    def __init__(self) -> None:
        self.applied_executable: list[int] = []
        self.applied_legacy: list[object] = []

    def apply_verified_mission_plan(self, plan: ExecutableMissionPlan) -> bool:
        self.applied_executable.append(plan.revision)
        return True

    def apply_tracking_plan(self, plan: object) -> None:
        self.applied_legacy.append(plan)

    def set_reservations(self, reservations: object) -> None:
        self.reservations = reservations

    def set_dedicated_tracking_groups(self, groups: object) -> None:
        self.dedicated_groups = groups


def _uuv_only_loop(engine: _RecordingEngine, plan: ExecutableMissionPlan) -> _AgentLoop:
    loop = _AgentLoop.__new__(_AgentLoop)
    loop._config = SimpleNamespace(scenario=SimpleNamespace(uuv_only=True))
    loop._engine = engine
    loop._runtime = SimpleNamespace(active_mission_plan=lambda: plan)
    loop.plans = SimpleNamespace(get_active=lambda scenario_id: None)
    loop._last_plan_id = None
    loop.situation = None
    return loop


def test_uuv_only_agent_loop_applies_executable_plan_without_legacy_tracking_plan() -> None:
    engine = _RecordingEngine()
    plan = ExecutableMissionPlan(revision=7)
    _uuv_only_loop(engine, plan)._apply_new_commands()

    assert engine.applied_executable == [7]
    assert engine.applied_legacy == []


def test_carrier_runtime_exposes_checkpointed_executable_plan() -> None:
    plan = ExecutableMissionPlan(revision=3)
    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._cycle_running = True
    runtime._state_cache = {"executable_mission_plan": plan}

    assert runtime.active_mission_plan() == plan


def test_agent_loop_syncs_dedicated_groups_separately_from_reservations() -> None:
    engine = _RecordingEngine()
    reservations = ReservationRegistry()
    reservations.reserve(("U1",), "T1")
    reservations.dedicate(("U2", "U3"), "T2")
    runtime = SimpleNamespace(reservations=lambda: reservations)

    _AgentLoop._sync_reservation_projections(engine, runtime)

    assert engine.reservations is reservations
    assert engine.dedicated_groups == {"T2": ("U2", "U3")}


def test_agent_loop_releases_dedicated_reservation_after_endurance_return() -> None:
    engine = _RecordingEngine()
    engine._mission_controller = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            events=(
                SimpleNamespace(
                    event_type="dedicated_mode_released",
                    payload={"target_id": "T2"},
                    sim_time_s=360,
                ),
            )
        )
    )
    reservations = ReservationRegistry()
    reservations.dedicate(("U2", "U3"), "T2")
    regional_replans: list[dict[str, object]] = []
    runtime = SimpleNamespace(
        reservations=lambda: reservations,
        submit_regional_replan=lambda **kwargs: regional_replans.append(kwargs),
    )

    _AgentLoop._sync_reservation_projections(engine, runtime)
    _AgentLoop._sync_reservation_projections(engine, runtime)

    assert reservations.dedicated_for("T2") == frozenset()
    assert engine.dedicated_groups == {}
    assert regional_replans == [
        {
            "reason": "endurance",
            "entity_id": "T2",
            "sim_time_s": 360,
            "payload": {"source": "dedicated_mode_released"},
        }
    ]


def test_real_uuv_only_entrypoint_publishes_onboard_inventory_without_usv_fields() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=42)

    frame = engine.step()

    assert "usvs" not in frame
    assert len(frame["uuvs"]) == 12
    assert all(uuv["deployment_state"] == "onboard" for uuv in frame["uuvs"])
    assert "usv" not in json.dumps(frame, sort_keys=True).casefold()
