# src/underwater_tracking/cli.py
"""Command-line entry points for the underwater tracking assistant.

``simulate`` runs the deterministic headless simulation and writes frames.
``agent-run`` runs the same scenario through the resilient LangGraph
carrier: it loads the config, creates the SQLite repositories and
checkpointer, builds the real LongCat HTTP provider (the API key is read
from the configured environment variable at call time; ``agent-run`` fails
with a message naming the variable when it is missing), wires the engine's
group reports into ``CarrierRuntime`` (the carrier hook is called at the
end of every observation cycle), applies the carrier's committed plan
commands back to the group manager at the next observation cycle, and
writes a run manifest (``manifest.json``) plus the frame log
(``frames.jsonl``) into ``outputs/run-<seed>-<id>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, cast

from underwater_tracking.agent.graphs.central import CarrierDependencies
from underwater_tracking.agent.llm import HTTPStructuredLLM
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AppConfig
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.persistence.sqlite import now_ms
from underwater_tracking.prediction.port import make_snapshot_predictor
from underwater_tracking.simulation.clock import SimulationClock
from underwater_tracking.simulation.engine import SimulationEngine

_SCENARIO_ID = "underwater-default"
_OBSERVATION_STEP_S = 30


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="underwater-tracking")
    sub = parser.add_subparsers(dest="command", required=True)

    simulate = sub.add_parser("simulate")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--steps", type=int, required=True)
    simulate.add_argument("--seed", type=int, required=True)
    simulate.set_defaults(handler=_simulate)

    agent_run = sub.add_parser("agent-run")
    agent_run.add_argument("--config", required=True)
    agent_run.add_argument("--steps", type=int, required=True)
    agent_run.add_argument("--seed", type=int, required=True)
    agent_run.set_defaults(handler=_agent_run)

    args = parser.parse_args(argv)
    return cast(int, args.handler(load_app_config(args.config), args))


def _simulate(config: AppConfig, args: argparse.Namespace) -> int:
    engine = SimulationEngine(config, seed=args.seed)
    for _ in range(args.steps):
        engine.step()
    return 0


def _agent_run(config: AppConfig, args: argparse.Namespace) -> int:
    """Run the agent-coupled scenario and write manifest plus JSONL."""
    run_dir = Path("outputs") / f"run-{args.seed}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    database_path = run_dir / "agent.db"
    loop = _AgentLoop(
        config,
        database_path=database_path,
        llm=_build_llm(config),
        run_id=run_dir.name,
        steps=args.steps,
        seed=args.seed,
    )
    engine = SimulationEngine(
        config, seed=args.seed, output_dir=run_dir, carrier=loop.on_situation
    )
    loop.attach(engine)
    try:
        for _ in range(args.steps):
            engine.step()
    except Exception as exc:  # noqa: BLE001 - surface as a CLI failure
        print(f"agent-run failed: {exc}", file=sys.stderr)
        loop.close()
        return 1
    loop.write_manifest(run_dir)
    loop.close()
    return 0


def _build_llm(config: AppConfig) -> HTTPStructuredLLM:
    """The real LongCat HTTP client, failing clearly when it cannot run.

    ``agent-run`` has no mock fallback: the API key is read from the
    configured environment variable at call time, so a missing variable is
    detected up front and reported by name (never by value).
    """
    llm_config = config.llm
    if llm_config is None:
        print(
            "agent-run requires an 'llm' section in the config file",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if os.environ.get(llm_config.api_key_env) is None:
        print(
            f"agent-run requires the {llm_config.api_key_env} environment variable"
            " (the LongCat API key is read from the environment at call time)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return HTTPStructuredLLM(
        base_url=llm_config.base_url,
        model=llm_config.model,
        api_key_env=llm_config.api_key_env,
        request_timeout_s=llm_config.request_timeout_s,
        connect_timeout_s=llm_config.connect_timeout_s,
        temperature=llm_config.temperature,
    )


class _AgentLoop:
    """Wires the engine's group reports into CarrierRuntime and back.

    ``on_situation`` is the engine hook called at the end of every
    observation cycle: it submits the initialization event once the belief
    history is warm, runs one carrier tick (deferring any carrier error so
    the group loop keeps running), and applies newly committed plan
    commands back to the engine (translated into group commands at the next
    observation cycle).
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        database_path: Path,
        llm: HTTPStructuredLLM,
        run_id: str,
        steps: int,
        seed: int,
    ) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._config = config
        self.database_path = database_path
        self.scenario_id = _SCENARIO_ID
        self.run_id = run_id
        self.steps = steps
        self.seed = seed
        self.plans = PlanRepository(database_path)
        self.events = EventRepository(database_path)
        self.ledger = DecisionLedger(database_path)
        self.llm = llm
        self.situation: SituationSnapshot | None = None
        self.carrier_error_count = 0
        self._runtime: CarrierRuntime | None = None
        self._engine: SimulationEngine | None = None
        self._clock = SimulationClock(step_s=_OBSERVATION_STEP_S)
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

    def _belief_history(
        self, snapshot: SituationSnapshot, target_id: str
    ) -> tuple[tuple[int, float, float], ...]:
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
        self.situation = situation
        if not self._initialization_submitted and self._initialization_ready(situation):
            self._initialization_submitted = True
            runtime.submit_event(
                event_type="initialization",
                entity_id=situation.scenario_id,
                sim_time_s=situation.sim_time_s,
            )
        try:
            result = runtime.tick()
        except Exception:  # noqa: BLE001 - the group loop must keep running
            self.carrier_error_count += 1
            return
        if result.get("commit_status") == "committed":
            self._apply_new_commands()

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

    def write_manifest(self, run_dir: Path) -> None:
        """Write the run manifest summarizing the finished agent run."""
        active = self.plans.get_active(self.scenario_id)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "steps": self.steps,
            "seed": self.seed,
            "llm": self._config.llm.model if self._config.llm else "http",
            "created_at_ms": now_ms(),
            "carrier_error_count": self.carrier_error_count,
            "decision_count": len(self.ledger.list_decisions(self.scenario_id)),
            "llm_call_count": len(self.ledger.list_llm_calls()),
            "active_plan_id": active.plan_id if active is not None else None,
            "active_plan_revision": active.revision if active is not None else None,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
        self.plans.close()
        self.events.close()
        self.ledger.close()


if __name__ == "__main__":
    sys.exit(main())
