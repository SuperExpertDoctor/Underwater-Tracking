from __future__ import annotations

from types import SimpleNamespace

from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.cli import _AgentLoop
from underwater_tracking.domain.mission_models import ExecutableMissionPlan


class _RecordingEngine:
    def __init__(self) -> None:
        self.applied_executable: list[int] = []
        self.applied_legacy: list[object] = []

    def apply_verified_mission_plan(self, plan: ExecutableMissionPlan) -> bool:
        self.applied_executable.append(plan.revision)
        return True

    def apply_tracking_plan(self, plan: object) -> None:
        self.applied_legacy.append(plan)


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
