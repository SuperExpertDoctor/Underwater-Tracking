# tests/integration/test_agent_loop.py
"""End-to-end agent loop over the simulated scenario (plan Task 12).

The engine drives the per-target group graphs at the 30 s observation
cadence and, at the end of every observation cycle, hands the latest
situation to the carrier runtime through the wired ``carrier`` hook. The
carrier's committed plan commands are polled from the repository and flow
back into the engine, which translates them into group commands applied at
the next observation cycle (commit -> PlanCommand rows -> group manager
apply).

Per the user directive (addendum A) no mock substitutes real LLM
functionality: every strategic cycle and every question answer here runs
against the real LongCat provider through the same ``HTTPStructuredLLM``
client ``agent-run`` uses, with the real B-spline prediction port (no
straight-line substitute). Request counts are deliberately modest: the
scenarios are shortened (120/90 simulated steps instead of 540) and the
assertions are variance-robust (at least one valid commit, a committed
cycle with a successful verification, the deterministic emergency fallback
never invoked, every committed plan independently valid, group updates
every 30 s with a generous stall bound, the manifest written). The former
scripted/mock harnesses and the scripted provider-exhaustion test were
deleted as an accepted consequence. The whole module is skipped when the
API key is unset.
"""

from __future__ import annotations

import importlib
import json
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from langgraph.checkpoint.memory import InMemorySaver

from underwater_tracking.agent.graphs.central import (
    CarrierDependencies,
    build_carrier_graph,
    live_situation_ref,
)
from underwater_tracking.agent.llm import HTTPStructuredLLM, LLMCallMetadata
from underwater_tracking.agent.nodes.commit import validate_plan
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.snapshot import build_planning_snapshot
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.cli import main
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AppConfig
from underwater_tracking.domain.agent_models import TrackingPlan, VerificationCommand
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.prediction.port import make_snapshot_predictor
from underwater_tracking.simulation.clock import SimulationClock
from underwater_tracking.simulation.engine import SimulationEngine
from tests.conftest import (
    REAL_LLM_SKIP_REASON,
    has_live_api_key,
    make_live_llm,
)

pytestmark = pytest.mark.skipif(
    not has_live_api_key(),
    reason=REAL_LLM_SKIP_REASON,
)

SCENARIO_ID = "underwater-default"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/scenario/default.yaml"
PHYSICS_STEP_S = 10
OBSERVATION_STEP_S = 30
E2E_STEPS = 120  # 120 x 10 s = 20 simulated minutes
CLI_STEPS = 90  # 90 x 10 s = 15 simulated minutes
RESTART_STEPS = 90

# Real engine observation ids look like ``target_00:uuv_03:450`` and the
# scenario's trigger event ids like ``underwater-default:target_added:...``;
# every evidence id an answer may cite lives in one of the two namespaces.
_EVIDENCE_ID = re.compile(r"target_\d\d:uuv_\d\d:\d+")
_TRIGGER_ID = re.compile(r"underwater-default:(initialization|target_added):[A-Za-z0-9_-]+:\d+")


class AgentLoop:
    """Wires the engine's group reports into CarrierRuntime and back.

    ``on_situation`` is the engine hook called at the end of every
    observation cycle: it submits the scenario events, runs one carrier
    tick, records the cycle result, and applies newly committed plan
    commands back to the engine (the engine translates them into group
    commands at the next observation cycle). The LLM client is the real
    ``HTTPStructuredLLM`` over the shipped LongCat config; every call is
    recorded for the request-count and provenance assertions.
    """

    def __init__(self, config: AppConfig, database_path: Path) -> None:
        self._config = config
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self.scenario_id = SCENARIO_ID
        self.plans = PlanRepository(database_path)
        self.events = EventRepository(database_path)
        self.ledger = DecisionLedger(database_path)
        self.calls: list[LLMCallMetadata] = []
        self.llm: HTTPStructuredLLM = make_live_llm(
            before_request=self.calls.append,
            ledger=self.ledger,
            scenario_id=self.scenario_id,
        )
        self.situation: SituationSnapshot | None = None
        self.commits: list[tuple[TrackingPlan, SituationSnapshot]] = []
        self.cycle_results: list[dict[str, Any]] = []
        self.cycle_errors: list[tuple[int, str]] = []
        self.answer: Any = None
        self._engine: SimulationEngine | None = None
        self._runtime: CarrierRuntime | None = None
        self._clock = SimulationClock(step_s=OBSERVATION_STEP_S)
        self._initialization_submitted = False
        self._last_plan_id: str | None = None

    def attach(self, engine: SimulationEngine) -> None:
        """Create the carrier runtime over the same SQLite database."""
        self._engine = engine
        self._runtime = CarrierRuntime(
            self._deps(), scenario_id=self.scenario_id, database_path=self.database_path
        )

    def _deps(self) -> CarrierDependencies:
        config = self._config
        agent = config.agent
        return CarrierDependencies(
            plans=self.plans,
            events=self.events,
            ledger=self.ledger,
            llm=self.llm,
            predictor=make_snapshot_predictor(
                belief_history=self._belief_history,
                horizon_s=config.timing.prediction_horizon_s,
                sample_step_s=config.timing.observation_step_s,
                max_speed_mps=config.tracking.uuv_max_speed_mps,
                max_turn_rate_rad_s=config.tracking.uuv_max_turn_rate_rad_s,
            ),
            situation_provider=self._live_situation,
            belief_history=self._belief_history,
            clock=self._clock,
            monitor=EventMonitor(
                scenario_id=self.scenario_id,
                warning_threshold=config.tracking.quality_warning,
                warning_hold_s=agent.quality_warning_persist_s if agent else 120,
                critical_threshold=config.tracking.quality_critical,
                cooldown_s=agent.event_cooldown_s if agent else 300,
                critical_hold_s=agent.quality_critical_persist_s if agent else 30,
                group_min_size=config.tracking.group_min_size,
            ),
            semantic_repairs=agent.semantic_repairs if agent else 2,
            model_id=config.llm.model if config.llm else "http",
        )

    def _live_situation(self, ref: str) -> SituationSnapshot:
        situation = self.situation
        if situation is None:
            raise RuntimeError(f"no live situation recorded for {ref!r}")
        return situation

    def _belief_history(self, snapshot: SituationSnapshot, target_id: str):
        del snapshot
        engine = self._engine
        assert engine is not None
        return engine.belief_history(target_id)

    def _initialization_ready(self, situation: SituationSnapshot) -> bool:
        engine = self._engine
        assert engine is not None
        return all(
            len(engine.belief_history(report.target_id)) >= 3
            for report in situation.group_reports
        )

    def on_situation(self, situation: SituationSnapshot) -> None:
        """Engine hook: run one carrier cycle over the latest situation."""
        runtime = self._runtime
        assert runtime is not None
        engine = self._engine
        assert engine is not None
        self.situation = situation
        engine.set_reservations(runtime.reservations())
        if not self._initialization_submitted and self._initialization_ready(situation):
            self._initialization_submitted = True
            runtime.submit_event(
                event_type="initialization",
                entity_id=situation.scenario_id,
                sim_time_s=situation.sim_time_s,
            )
        plan_before = self._last_plan_id
        try:
            result = runtime.tick()
        except Exception as exc:  # noqa: BLE001 - the group loop must keep running
            self.cycle_errors.append((situation.sim_time_s, repr(exc)))
            return
        self.cycle_results.append(result)
        self._apply_new_commands()
        self._apply_verification_commands(result)
        # A real commit broadcasts a new plan id; the ``commit_status``
        # channel is checkpointed and persists on informational cycles, so
        # only a broadcast change counts as a commit.
        if self._last_plan_id != plan_before:
            active = self.plans.get_active(self.scenario_id)
            if active is not None:
                self.commits.append((active, situation))

    def _apply_new_commands(self) -> None:
        """Apply newly committed plan commands back to the group manager."""
        engine = self._engine
        assert engine is not None
        active = self.plans.get_active(self.scenario_id)
        if active is None or active.plan_id == self._last_plan_id:
            return
        self._last_plan_id = active.plan_id
        for command in self.plans.list_commands(active.plan_id):
            engine.apply_plan_command(command)

    def _apply_verification_commands(self, result: dict[str, Any]) -> None:
        """Apply the deterministic verification protocol commands to the engine."""
        engine = self._engine
        assert engine is not None
        for command in result.get("verification_commands") or ():
            assert isinstance(command, VerificationCommand)
            engine.apply_verification_command(command)
        # Re-arm the pingers every cycle: a plan command's sensor-mode write
        # resets ``_ping_targets`` and would otherwise kill a live ping
        # mid-protocol. Pingers are popped when the protocol closes, so this
        # stops exactly then.
        for contact_id, pinger in (result.get("verification_pingers") or {}).items():
            engine.set_sensor_mode(pinger, "active", ping_contact_id=contact_id)

    def submit_event(
        self,
        *,
        event_type: str,
        entity_id: str,
        sim_time_s: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        runtime = self._runtime
        assert runtime is not None
        runtime.submit_event(
            event_type=event_type,
            entity_id=entity_id,
            sim_time_s=sim_time_s,
            payload=payload,
        )

    def ask(self, raw_text: str) -> Any:
        runtime = self._runtime
        assert runtime is not None
        self.answer = runtime.ask(raw_text)
        return self.answer

    def get_state(self) -> dict[str, Any]:
        runtime = self._runtime
        assert runtime is not None
        return runtime.get_state()

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
        self.llm.close()
        self.plans.close()
        self.events.close()
        self.ledger.close()


def _report_times(frames: list[dict[str, object]]) -> list[int]:
    """The belief sim times of the first group at every observation frame."""
    return [
        int(track["sim_time_s"])
        for frame in frames
        if int(frame["sim_time_s"]) % OBSERVATION_STEP_S == 0
        for track in frame["tracks"][:1]  # type: ignore[union-attr]
    ]


def _no_invalid_plan_committed(commits: list[tuple[TrackingPlan, SituationSnapshot]]) -> None:
    """Every committed plan re-validates independently against its
    commit-time situation (the Step 1 no-invalid-commit invariant)."""
    for plan, situation in commits:
        issues = validate_plan(build_planning_snapshot(situation), plan)
        assert not issues, f"invalid plan {plan.plan_id} committed: {issues}"


# --- Step 1: the end-to-end agent loop over the real provider ---------------


@pytest.mark.real_llm
def test_agent_loop_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """20-minute scenario: init, target addition, one question — live.

    Roughly two strategic cycles plus one question answer against the real
    provider (about 15-30 requests): the initialization commits the first
    plan, the target addition re-plans, and the question answer cites only
    real evidence. The deterministic emergency fallback must never run, no
    invalid plan may be committed, and the group loop keeps updating every
    30 s throughout.
    """
    emergency_fallbacks: list[object] = []
    verify_module = importlib.import_module("underwater_tracking.agent.nodes.verify")
    original_emergency = verify_module._emergency_strategy

    def counting_emergency(context: object) -> object:
        emergency_fallbacks.append(context)
        return original_emergency(context)

    monkeypatch.setattr(verify_module, "_emergency_strategy", counting_emergency)
    config = load_app_config(CONFIG_PATH)
    run_dir = tmp_path / "e2e"
    loop = AgentLoop(config, run_dir / "agent.db")
    engine = SimulationEngine(
        config, seed=42, output_dir=run_dir, carrier=loop.on_situation
    )
    loop.attach(engine)

    frames: list[dict[str, object]] = []
    for step in range(E2E_STEPS):
        sim_time_s = (step + 1) * PHYSICS_STEP_S
        if sim_time_s == 600:
            loop.submit_event(
                event_type="target_added",
                entity_id="target_01",
                sim_time_s=sim_time_s,
            )
        if sim_time_s == 900:
            loop.ask("Which plan is active and what evidence supports it?")
        frames.append(engine.step())

    try:
        # The initialization strategic cycle committed a plan whose
        # proposal survived semantic verification (the deterministic
        # emergency fallback would otherwise have run and been counted).
        assert len(loop.commits) >= 1
        assert any(
            result.get("commit_status") == "committed"
            for result in loop.cycle_results
        )
        assert emergency_fallbacks == []

        # No invalid plan committed.
        _no_invalid_plan_committed(loop.commits)

        # Group updates every 30 s: one belief entry per observation cycle,
        # advancing one observation step per cycle and reaching the final
        # observation. The generous stall bound tolerates predict-only
        # cycles without ever stopping the loop.
        report_times = _report_times(frames)
        assert report_times[0] == OBSERVATION_STEP_S
        assert report_times[-1] == E2E_STEPS * PHYSICS_STEP_S
        assert len(report_times) == E2E_STEPS * PHYSICS_STEP_S // OBSERVATION_STEP_S
        stalled_cycles = sum(
            later == earlier for earlier, later in pairwise(report_times)
        )
        assert stalled_cycles <= 2

        # The committed plan flowed back: the group manager adopted a
        # committed revision by the final frame.
        applied = {
            int(report["plan_revision"])
            for report in frames[-1]["group_reports"]  # type: ignore[union-attr]
        }
        revisions = {plan.revision for plan, _ in loop.commits}
        assert max(applied) in revisions

        # Evidence ids in the answer, all within the scenario's citable
        # namespaces (real observation ids and trigger event ids).
        answer = loop.answer
        assert answer is not None
        assert answer.evidence_ids
        assert all(
            _EVIDENCE_ID.fullmatch(eid) or _TRIGGER_ID.fullmatch(eid)
            for eid in answer.evidence_ids
        )

        # The 20-minute run deferred essentially no carrier errors: the
        # client retries transient failures internally (3 attempts with
        # backoff), so a genuine provider outage of several minutes could
        # slip one or two degraded cycles through without stopping the
        # loop. A systematic failure (e.g. the round-1 wrong-endpoint 404,
        # which hit every cycle) is still caught by the commit, cadence,
        # and applied-revision invariants above.
        assert len(loop.cycle_errors) <= 2
    finally:
        loop.close()


# --- Step 2: the agent-run CLI command --------------------------------------


@pytest.mark.real_llm
def test_cli_agent_run_writes_manifest_plans_and_decisions(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """``agent-run`` drives the wired loop and writes manifest, frames, and
    the plan/decision records — against the real provider (no ``--llm``)."""
    monkeypatch.chdir(tmp_path)
    exit_code = main(
        [
            "agent-run",
            "--config",
            str(CONFIG_PATH),
            "--steps",
            str(CLI_STEPS),
            "--seed",
            "42",
        ]
    )
    assert exit_code == 0
    runs = sorted((tmp_path / "outputs").glob("run-*"))
    assert len(runs) == 1
    run_dir = runs[0]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["steps"] == CLI_STEPS
    assert manifest["seed"] == 42
    assert manifest["llm"] == "LongCat-2.0"
    frames = (run_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(frames) == CLI_STEPS
    plans = PlanRepository(run_dir / "agent.db")
    ledger = DecisionLedger(run_dir / "agent.db")
    try:
        assert plans.get_active(SCENARIO_ID) is not None
        assert ledger.list_decisions(SCENARIO_ID)
    finally:
        plans.close()
        ledger.close()


# --- Step 3: storage failure injection -------------------------------------


class FailingCheckpointer(InMemorySaver):
    """Checkpointer that fails on every write (simulated storage outage)."""

    def put(self, config, checkpoint, metadata, new_versions):
        raise RuntimeError("simulated checkpoint storage failure")

    def put_writes(self, config, writes, task_id, task_path=""):
        raise RuntimeError("simulated checkpoint storage failure")


class _FailingCarrier:
    """Minimal carrier adapter over a graph whose checkpointer fails.

    The adapter defers the injected storage failure so the engine's group
    loop keeps running: every carrier cycle records the failure and no
    cycle ever completes, so no plan command is ever broadcast to the
    groups. The cycles carry no pending events, so the informational route
    never invokes the LLM: the real client is constructed (the same
    configuration ``agent-run`` uses) but no request is ever made.
    """

    def __init__(self, config: AppConfig, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._situation: SituationSnapshot | None = None
        plans = PlanRepository(database_path)
        events = EventRepository(database_path)
        ledger = DecisionLedger(database_path)
        self.llm: HTTPStructuredLLM = make_live_llm(
            ledger=ledger, scenario_id=SCENARIO_ID
        )
        deps = CarrierDependencies(
            plans=plans,
            events=events,
            ledger=ledger,
            llm=self.llm,
            predictor=make_snapshot_predictor(
                belief_history=_empty_belief_history,
                horizon_s=config.timing.prediction_horizon_s,
                sample_step_s=OBSERVATION_STEP_S,
            ),
            situation_provider=self._live_situation,
            belief_history=_empty_belief_history,
            clock=SimulationClock(step_s=OBSERVATION_STEP_S),
            monitor=EventMonitor(scenario_id=SCENARIO_ID),
            model_id=config.llm.model if config.llm else "http",
        )
        self.deps = deps
        self._graph = build_carrier_graph(deps, FailingCheckpointer(), {})
        self._config = {"configurable": {"thread_id": f"{SCENARIO_ID}:carrier"}}
        self.errors: list[str] = []

    def _live_situation(self, ref: str) -> SituationSnapshot:
        situation = self._situation
        if situation is None:
            raise RuntimeError(f"no live situation recorded for {ref!r}")
        return situation

    def on_situation(self, situation: SituationSnapshot) -> None:
        self._situation = situation
        self.deps.clock.tick()
        try:
            self._graph.invoke(
                {
                    "scenario_id": situation.scenario_id,
                    "snapshot_ref": live_situation_ref(situation.scenario_id),
                    "pending_events": (),
                },
                config=self._config,
            )
        except Exception as exc:  # noqa: BLE001 - the group loop must keep running
            self.errors.append(f"{situation.sim_time_s}: {exc}")

    def close(self) -> None:
        self.llm.close()
        self.deps.plans.close()
        self.deps.events.close()
        self.deps.ledger.close()


def _empty_belief_history(snapshot: SituationSnapshot, target_id: str):
    del snapshot, target_id
    return ()


def test_checkpoint_failure_stops_commits_but_not_group_updates(tmp_path: Path) -> None:
    """A failing checkpointer blocks every central broadcast while the
    per-target group updates keep advancing every 30 s.

    The commit node writes the plan row before the cycle's final checkpoint
    write fails, so a partial plan row can exist in the repository as a side
    effect; the observable invariant is operational: no committed plan ever
    reaches the group manager (its plan revision stays 0) and every carrier
    cycle defers the injected storage failure.
    """
    config = load_app_config(CONFIG_PATH)
    failing = _FailingCarrier(config, tmp_path / "ckpt" / "agent.db")
    engine = SimulationEngine(
        config, seed=42, output_dir=tmp_path / "ckpt", carrier=failing.on_situation
    )
    frames: list[dict[str, object]] = []
    for _ in range(180):
        frames.append(engine.step())
    try:
        # The group loop never stopped: reports advance every 30 s.
        report_times = _report_times(frames)
        assert report_times[0] == OBSERVATION_STEP_S
        assert all(
            later - earlier == OBSERVATION_STEP_S
            for earlier, later in pairwise(report_times)
        )
        # Every carrier cycle deferred the injected storage failure and no
        # central plan revision ever reached the group manager.
        assert len(failing.errors) == len(report_times)
        assert all("checkpoint storage failure" in error for error in failing.errors)
        adopted = {
            int(report["plan_revision"])
            for frame in frames
            for report in frame["group_reports"]  # type: ignore[union-attr]
        }
        assert adopted == {0}
    finally:
        failing.close()


@pytest.mark.real_llm
def test_restart_restores_last_committed_revision(tmp_path: Path) -> None:
    """A reopened runtime over the same database resumes from the last
    committed plan revision and the checkpointed carrier state — live."""
    config = load_app_config(CONFIG_PATH)
    database_path = tmp_path / "restart" / "agent.db"
    first = AgentLoop(config, database_path)
    engine1 = SimulationEngine(
        config, seed=42, output_dir=tmp_path / "restart", carrier=first.on_situation
    )
    first.attach(engine1)
    for _ in range(RESTART_STEPS):
        engine1.step()
    active = first.plans.get_active(SCENARIO_ID)
    assert active is not None
    last_revision = active.revision
    state_before = dict(first.get_state())
    first.close()

    second = AgentLoop(config, database_path)
    engine2 = SimulationEngine(
        config, seed=42, output_dir=tmp_path / "restart", carrier=second.on_situation
    )
    second.attach(engine2)
    try:
        # The restored carrier state carries the checkpointed route and
        # strategy set; the repository still broadcasts the last revision.
        restored = second.get_state()
        assert restored.get("route") == state_before.get("route")
        assert restored.get("strategy_set") is not None
        assert second.plans.get_active(SCENARIO_ID).revision == last_revision
        # The resumed loop keeps committing from the restored revision.
        for _ in range(RESTART_STEPS):
            engine2.step()
        resumed = second.plans.get_active(SCENARIO_ID)
        assert resumed is not None
        assert resumed.revision > last_revision
    finally:
        second.close()
