# src/underwater_tracking/cli.py
"""Command-line entry points for the underwater tracking assistant.

``simulate`` runs the deterministic headless simulation and writes frames.
``agent-run`` runs the same scenario through the resilient LangGraph
carrier: it loads the config, creates the SQLite repositories and
checkpointer, selects ``--llm mock`` or the configured HTTP provider, wires
the engine's group reports into ``CarrierRuntime`` (the carrier hook is
called at the end of every observation cycle), applies the carrier's
committed plan commands back to the group manager at the next observation
cycle, and writes a run manifest (``manifest.json``) plus the frame log
(``frames.jsonl``) into ``outputs/run-<seed>-<id>/``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, TypeVar

from underwater_tracking.agent.graphs.central import CarrierDependencies
from underwater_tracking.agent.llm import HTTPStructuredLLM, MockStructuredLLM
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AppConfig
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.persistence.sqlite import now_ms
from underwater_tracking.simulation.clock import SimulationClock
from underwater_tracking.simulation.engine import SimulationEngine

_SCENARIO_ID = "underwater-default"
_OBSERVATION_STEP_S = 30
_RESPONSE_MODEL = TypeVar("_RESPONSE_MODEL")
# Real observation ids are ``target_00:uuv_04:270``; the question payload
# also carries trigger-event ids (e.g. ``G-target_00:group_quality_warning:720``),
# which are not observations and must never be cited as evidence.
_OBSERVATION_EVIDENCE_ID = re.compile(r"target_\d\d:uuv_\d\d:\d+")


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
    agent_run.add_argument(
        "--llm", choices=("mock", "http"), default="mock", help="LLM provider"
    )
    agent_run.set_defaults(handler=_agent_run)

    args = parser.parse_args(argv)
    return args.handler(load_app_config(args.config), args)


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
        llm=_build_llm(config, args.llm),
        run_id=run_dir.name,
        steps=args.steps,
        seed=args.seed,
        llm_name=args.llm,
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


def _build_llm(config: AppConfig, name: str) -> MockStructuredLLM | HTTPStructuredLLM:
    if name == "mock":
        return _ScriptedMockLLM()
    llm_config = config.llm
    if llm_config is None:
        print(
            "--llm http requires an 'llm' section in the config file",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return HTTPStructuredLLM(
        base_url=llm_config.base_url,
        model=llm_config.model,
        api_key_env=llm_config.api_key_env,
        request_timeout_s=llm_config.request_timeout_s,
        connect_timeout_s=llm_config.connect_timeout_s,
    )


class _ScriptedMockLLM(MockStructuredLLM):
    """Deterministic offline provider: every response derives from its payload.

    The responses are payload-aware (real evidence ids, every known target
    covered), so an offline run never exhausts the provider or hits content
    errors; ``call_count`` feeds the run manifest.
    """

    def __init__(self) -> None:
        super().__init__({})
        self.call_count = 0

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[_RESPONSE_MODEL],
        *,
        prompt_version: str = "",
    ) -> _RESPONSE_MODEL:
        self.call_count += 1
        return response_model.model_validate(_scripted_response(operation, payload))


def _scripted_response(operation: str, payload: dict[str, object]) -> dict[str, object]:
    """Derive one valid structured response from the payload itself."""
    if operation == "intent":
        evidence = sorted(set(payload.get("evidence_ids", [])))
        if not evidence:
            raise ValueError("scripted intent response needs evidence ids in the payload")
        return {
            "label": "transit",
            "confidence": 0.8,
            "evidence_ids": [evidence[0]],
            "alternatives": {},
            "planning_effects": (),
            "model_id": payload["model"],
            "prompt_version": "scripted",
        }
    if operation == "strategy":
        requested = str(payload.get("requested_concept") or "quality_first")
        target_ids = [target["target_id"] for target in payload.get("targets", [])]
        if not target_ids:
            target_ids = list(payload.get("target_ids", []))
        evidence = sorted(
            {
                evidence_id
                for target in payload.get("targets", [])
                for evidence_id in target.get("evidence_ids", [])
            }
            | set(payload.get("evidence_ids", []))
        )
        if not target_ids or not evidence:
            raise ValueError(
                "scripted strategy response needs targets and evidence in the payload"
            )
        return {
            "concept": requested,
            "target_priorities": {target: 1.0 for target in target_ids},
            "required_quality": {target: 0.7 for target in target_ids},
            "reinforcement_policy": {target: "release_when_stable" for target in target_ids},
            "releasable_soft_constraints": ["energy_reserve_0.1"],
            "evidence_ids": evidence[:4],
            "rationale": f"balanced coverage of {', '.join(target_ids)}",
        }
    if operation == "directive":
        known_targets = list(payload.get("known_target_ids", []))
        if not known_targets:
            raise ValueError("scripted directive response needs known targets in the payload")
        target = known_targets[0]
        return {
            "directive_id": payload["directive_id"],
            "raw_text": payload["raw_text"],
            "target_scope": [target],
            "locked_members": {},
            "target_priorities": {target: 1.0},
            "minimum_quality": {target: 0.6},
            "disabled_uuv_ids": [],
            "confidence": 0.9,
            "conflicts": [],
            "status": "preview",
        }
    if operation == "question":
        # Only real observation ids are citable evidence: the payload also
        # carries trigger-event ids, which are not observations and must
        # never appear in an answer.
        evidence = sorted(
            {
                evidence_id
                for evidence_id in payload.get("evidence_ids", [])
                if _OBSERVATION_EVIDENCE_ID.fullmatch(str(evidence_id))
            }
        )
        if not evidence:
            raise ValueError(
                "scripted question response needs observation evidence ids"
            )
        return {
            "answer": (
                "The active plan is the latest committed revision; every commit was"
                " independently validated and revisions increase monotonically."
            ),
            "evidence_ids": evidence[:2],
            "counterfactual_plan_id": None,
            "counterfactual_summary": None,
        }
    raise ValueError(f"no scripted response for operation {operation!r}")


def _straight_line_predictor(
    snapshot: SituationSnapshot, target_id: str
) -> PredictedTrackRef:
    return PredictedTrackRef(
        prediction_id=(
            f"{snapshot.scenario_id}:track:{target_id}:{snapshot.snapshot_revision}"
        ),
        target_id=target_id,
        sim_time_s=snapshot.sim_time_s,
        horizon_s=600.0,
        sample_step_s=30.0,
        points_xy=((0.0, 0.0),),
        corridor_radius_m=(400.0,),
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
        llm: MockStructuredLLM | HTTPStructuredLLM,
        run_id: str,
        steps: int,
        seed: int,
        llm_name: str,
    ) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._config = config
        self.database_path = database_path
        self.scenario_id = _SCENARIO_ID
        self.run_id = run_id
        self.steps = steps
        self.seed = seed
        self.llm_name = llm_name
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
            predictor=_straight_line_predictor,
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
            model_id=config.llm.model if config.llm else "mock",
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
        llm_calls = self.llm.call_count if isinstance(self.llm, _ScriptedMockLLM) else 0
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "steps": self.steps,
            "seed": self.seed,
            "llm": self.llm_name,
            "created_at_ms": now_ms(),
            "carrier_error_count": self.carrier_error_count,
            "decision_count": len(self.ledger.list_decisions(self.scenario_id)),
            "llm_call_count": llm_calls,
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
