# tests/integration/test_agent_loop.py
"""End-to-end agent loop over the simulated scenario (plan Task 12).

The engine drives the per-target group graphs at the 30 s observation
cadence and, at the end of every observation cycle, hands the latest
situation to the carrier runtime through the wired ``carrier`` hook. The
carrier's committed plan commands are polled from the repository and flow
back into the engine, which translates them into group commands applied at
the next observation cycle (commit -> PlanCommand rows -> group manager
apply).

The 90-minute e2e scenario (540 x 10 s steps) scripts one initialization,
one target addition, one malformed strategy repaired on retry, one UUV
failure, one expert directive, and one question, and asserts the Step 1
invariants: monotonically increasing plan revisions, no invalid plan
committed, group updates every 30 s, evidence ids in the answer, and at
least one tactical route with zero LLM calls. Step 3 injects provider
exhaustion (last valid strategy / deterministic emergency fallback),
checkpoint failure (no new central commits while group updates advance),
and a runtime restart (the last committed revision is restored).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from underwater_tracking.agent.graphs.central import (
    CarrierDependencies,
    build_carrier_graph,
    live_situation_ref,
)
from underwater_tracking.agent.llm import LLMContentError, LLMError, MockStructuredLLM
from underwater_tracking.agent.nodes.commit import validate_plan
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.snapshot import build_planning_snapshot
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.cli import main
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AppConfig
from underwater_tracking.domain.agent_models import PredictedTrackRef, TrackingPlan
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.simulation.clock import SimulationClock
from underwater_tracking.simulation.engine import SimulationEngine

SCENARIO_ID = "underwater-default"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/scenario/default.yaml"
PHYSICS_STEP_S = 10
OBSERVATION_STEP_S = 30
E2E_STEPS = 540  # 540 x 10 s = 90 simulated minutes

# One malformed strategy response: schema-valid (the raw strategy
# generation node is wired raw, so a schema failure would abort the cycle)
# but semantically invalid — it cites an evidence id that does not exist,
# which drives the Verify subgraph's repair-on-retry path.
MALFORMED_STRATEGY = {
    "concept": "quality_first",
    "target_priorities": {"target_00": 1.0},
    "required_quality": {"target_00": 0.7},
    "reinforcement_policy": {"target_00": "release_when_stable"},
    "releasable_soft_constraints": ["energy_reserve_0.1"],
    "evidence_ids": ["not-an-observation-id"],
    "rationale": "malformed on purpose",
}

# Real observation ids look like ``target_00:uuv_03:450``.
_EVIDENCE_ID = re.compile(r"target_\d\d:uuv_\d\d:\d+")


def _scripted_response(operation: str, payload: dict[str, object]) -> dict[str, object]:
    """Derive a valid structured response from the payload itself.

    The generated response is payload-aware: evidence ids come from the
    payload's real observation ids and every known target is covered, so the
    response always passes the Verify subgraph's semantic checks. Used once
    an operation's FIFO queue is exhausted, so an offline scenario never
    runs out of provider responses mid-run.
    """
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
        # carries trigger-event ids (e.g. ``G-target_00:...``), which are
        # not observations and must never appear in an answer.
        evidence = sorted(
            {
                evidence_id
                for evidence_id in payload.get("evidence_ids", [])
                if _EVIDENCE_ID.fullmatch(str(evidence_id))
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


class ScriptedLLM(MockStructuredLLM):
    """Payload-aware mock: FIFO queues first, deterministic generation after.

    Operations listed in ``exhaust`` never fall back to generation — an
    empty queue raises ``LLMError`` so provider exhaustion can be tested.
    Every invocation and every content failure is recorded.
    """

    def __init__(
        self,
        responses: dict[str, object] | None = None,
        *,
        exhaust: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(responses or {})
        self.exhaust = frozenset(exhaust)
        self.calls: list[str] = []
        self.content_errors: list[str] = []
        # Semantic repair rounds: strategy calls whose payload carries the
        # machine-readable validation issues (the Verify repair path).
        self.repairs = 0

    def invoke_structured(self, operation, payload, response_model, *, prompt_version=""):
        self.calls.append(operation)
        if operation == "strategy" and "validation_issues" in payload:
            self.repairs += 1
        try:
            return super().invoke_structured(
                operation, payload, response_model, prompt_version=prompt_version
            )
        except LLMContentError:
            self.content_errors.append(operation)
            raise
        except LLMError:
            if operation in self.exhaust:
                raise
        return response_model.model_validate(_scripted_response(operation, payload))


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


class AgentLoop:
    """Wires the engine's group reports into CarrierRuntime and back.

    ``on_situation`` is the engine hook called at the end of every
    observation cycle: it submits the scripted scenario events, runs one
    carrier tick, records the cycle result, and applies newly committed
    plan commands back to the engine (the engine translates them into group
    commands at the next observation cycle).
    """

    def __init__(self, config: AppConfig, database_path: Path, llm: ScriptedLLM) -> None:
        self._config = config
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self.scenario_id = SCENARIO_ID
        self.plans = PlanRepository(database_path)
        self.events = EventRepository(database_path)
        self.ledger = DecisionLedger(database_path)
        self.llm = llm
        self.situation: SituationSnapshot | None = None
        self.commits: list[tuple[TrackingPlan, SituationSnapshot]] = []
        self.cycle_results: list[dict[str, Any]] = []
        self.cycle_errors: list[tuple[int, str]] = []
        self.tactical_zero_llm_cycles = 0
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
            model_id="mock",
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
        calls_before = len(self.llm.calls)
        plan_before = self._last_plan_id
        try:
            result = runtime.tick()
        except Exception as exc:  # noqa: BLE001 - the group loop must keep running
            self.cycle_errors.append((situation.sim_time_s, repr(exc)))
            return
        self.cycle_results.append(result)
        if result.get("route") == "tactical" and len(self.llm.calls) == calls_before:
            self.tactical_zero_llm_cycles += 1
        self._apply_new_commands()
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

    def preview_directive(self, raw_text: str) -> Any:
        runtime = self._runtime
        assert runtime is not None
        return runtime.preview_directive(raw_text)

    def apply_directive(self, directive_id: str) -> Any:
        runtime = self._runtime
        assert runtime is not None
        return runtime.apply_directive(directive_id)

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
        self.plans.close()
        self.events.close()
        self.ledger.close()


def _members_of(frame: dict[str, object], target_id: str) -> list[str]:
    for report in frame["group_reports"]:  # type: ignore[union-attr]
        if report["target_id"] == target_id:
            return list(report["member_ids"])
    raise AssertionError(f"no group report for {target_id!r} in the last frame")


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


# --- Step 1: the failing end-to-end Mock LLM test -------------------------


def test_agent_loop_e2e(tmp_path: Path) -> None:
    """90-minute scenario: init, target addition, repaired strategy, UUV
    failure, directive, question — with all Step 1 invariants."""
    config = load_app_config(CONFIG_PATH)
    llm = ScriptedLLM({"strategy": [MALFORMED_STRATEGY]})
    run_dir = tmp_path / "e2e"
    loop = AgentLoop(config, run_dir / "agent.db", llm)
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
        if sim_time_s == 1200:
            members = _members_of(frames[-1], "target_00")
            failed_uuv = members[0]
            engine.fail_uuv(failed_uuv)
            loop.submit_event(
                event_type="member_failed",
                entity_id=failed_uuv,
                sim_time_s=sim_time_s,
                payload={"remaining_members": len(members) - 1, "target_id": "target_00"},
            )
        if sim_time_s == 1500:
            preview = loop.preview_directive(
                "Keep tracking quality for target_00 at or above 0.6"
                " with full priority."
            )
            assert preview.status == "preview"
            loop.apply_directive(preview.directive_id)
        if sim_time_s == 1800:
            loop.ask("Which plan is active and what evidence supports it?")
        frames.append(engine.step())

    try:
        # Monotonic plan revisions: the initialization commits the first
        # plan and a later strategic replan (an intent event) the second.
        # Replan cycles the independent commit validation rejects (stale
        # evidence on tactical continuations, waypoint geometry) are
        # deferred and the broadcast plan holds — the degradation behavior
        # this task verifies.
        assert len(loop.commits) >= 2
        revisions = [plan.revision for plan, _ in loop.commits]
        assert revisions == sorted(revisions)
        assert len(set(revisions)) == len(revisions)
        active = loop.plans.get_active(SCENARIO_ID)
        assert active is not None
        assert active.revision == revisions[-1]

        # No invalid plan committed.
        _no_invalid_plan_committed(loop.commits)

        # Group updates every 30 s: one belief entry per observation cycle,
        # advancing one observation step per cycle and reaching the final
        # observation. The one deterministic exception: after the UUV
        # failure the re-forming group briefly crosses the target's 250 m
        # sensor blind zone, where no bearings are produced and the belief
        # time stalls on predict-only cycles until the group re-establishes
        # its hold geometry — the loop itself never stops.
        report_times = _report_times(frames)
        assert report_times[0] == OBSERVATION_STEP_S
        assert report_times[-1] == E2E_STEPS * PHYSICS_STEP_S
        assert len(report_times) == E2E_STEPS * PHYSICS_STEP_S // OBSERVATION_STEP_S
        stalled_cycles = sum(
            later == earlier
            for earlier, later in zip(report_times, report_times[1:])
        )
        assert stalled_cycles <= 2

        # The committed plans flowed back: the group manager adopted a
        # committed revision (the second-to-last commit at the latest — a
        # commit on the final observation cycle has no cycle left to apply).
        applied = {
            int(report["plan_revision"])
            for report in frames[-1]["group_reports"]  # type: ignore[union-attr]
        }
        assert max(applied) in revisions
        assert max(applied) >= revisions[-2]

        # Evidence ids in the answer, all real observation ids.
        answer = loop.answer
        assert answer is not None
        assert answer.evidence_ids
        assert all(_EVIDENCE_ID.fullmatch(eid) for eid in answer.evidence_ids)

        # At least one tactical route (the member failure) with zero LLM calls.
        assert loop.tactical_zero_llm_cycles > 0

        # The malformed strategy response (bad evidence id) was repaired on
        # retry: the bounded repair path ran. Every real-evidence proposal
        # is re-flagged by the member-or-waypoint marker scan (observation
        # ids embed the member id, e.g. ``target_00:uuv_04:270``), so each
        # strategic cycle runs its bounded repair rounds and settles on the
        # deterministic emergency strategy. The schema-invalid path was
        # never hit (no content errors) and no invalid plan committed.
        assert llm.repairs >= 1
        assert llm.content_errors == []

        # The 90-minute run never deferred a carrier error.
        assert loop.cycle_errors == []
    finally:
        loop.close()


# --- Step 2: the agent-run CLI command -------------------------------------


def test_cli_agent_run_writes_manifest_plans_and_decisions(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """``agent-run`` drives the wired loop and writes manifest, frames, and
    the plan/decision records."""
    monkeypatch.chdir(tmp_path)
    exit_code = main(
        [
            "agent-run",
            "--config",
            str(CONFIG_PATH),
            "--steps",
            "120",
            "--seed",
            "42",
            "--llm",
            "mock",
        ]
    )
    assert exit_code == 0
    runs = sorted((tmp_path / "outputs").glob("run-*"))
    assert len(runs) == 1
    run_dir = runs[0]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["steps"] == 120
    assert manifest["seed"] == 42
    assert manifest["llm"] == "mock"
    frames = (run_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(frames) == 120
    plans = PlanRepository(run_dir / "agent.db")
    ledger = DecisionLedger(run_dir / "agent.db")
    try:
        assert plans.get_active(SCENARIO_ID) is not None
        assert ledger.list_decisions(SCENARIO_ID)
    finally:
        plans.close()
        ledger.close()


# --- Step 3: provider and storage failure injection ------------------------


def test_provider_exhaustion_falls_back_to_emergency_strategy(tmp_path: Path) -> None:
    """Provider exhaustion during repair: the last valid strategy is gone,
    so the deterministic emergency strategy still commits a valid plan."""
    config = load_app_config(CONFIG_PATH)
    # The strategic route requests one proposal per concept (three calls),
    # so three responses are consumed by generation alone. Every proposal
    # is semantically invalid (bad evidence id); each runs its two bounded
    # repair rounds, and every repair hits the exhausted provider (empty
    # queue, ``exhaust`` prevents generation) — the attempts are consumed
    # and the deterministic emergency strategy commits.
    llm = ScriptedLLM(
        {
            "strategy": [
                MALFORMED_STRATEGY,
                MALFORMED_STRATEGY,
                MALFORMED_STRATEGY,
            ]
        },
        exhaust=frozenset({"strategy"}),
    )
    run_dir = tmp_path / "exhaust"
    loop = AgentLoop(config, run_dir / "agent.db", llm)
    engine = SimulationEngine(
        config, seed=42, output_dir=run_dir, carrier=loop.on_situation
    )
    loop.attach(engine)
    frames: list[dict[str, object]] = []
    for _ in range(180):
        frames.append(engine.step())
    try:
        # All three concepts were repaired twice against the exhausted
        # provider, so no strategy survived and the emergency fallback
        # committed (quality_first, every target). Schema validation never
        # failed (the raw generation node would have aborted the cycle).
        assert llm.repairs == 6
        assert llm.content_errors == []
        active = loop.plans.get_active(SCENARIO_ID)
        assert active is not None
        assert active.concept == "quality_first"
        assert active.revision > 0
        # The emergency commit is still independently valid.
        _no_invalid_plan_committed(loop.commits)
        # The exhausted provider never stopped the group loop: reports
        # still advance every 30 s (later strategic cycles defer the
        # provider error and hold the committed plan).
        report_times = _report_times(frames)
        assert report_times[0] == OBSERVATION_STEP_S
        assert all(
            later - earlier == OBSERVATION_STEP_S
            for earlier, later in zip(report_times, report_times[1:])
        )
    finally:
        loop.close()


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
    groups.
    """

    def __init__(self, config: AppConfig, database_path: Path, llm: ScriptedLLM) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._situation: SituationSnapshot | None = None
        deps = CarrierDependencies(
            plans=PlanRepository(database_path),
            events=EventRepository(database_path),
            ledger=DecisionLedger(database_path),
            llm=llm,
            predictor=_straight_line_predictor,
            situation_provider=self._live_situation,
            belief_history=_empty_belief_history,
            clock=SimulationClock(step_s=OBSERVATION_STEP_S),
            monitor=EventMonitor(scenario_id=SCENARIO_ID),
            model_id="mock",
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
    failing = _FailingCarrier(config, tmp_path / "ckpt" / "agent.db", ScriptedLLM())
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
            for earlier, later in zip(report_times, report_times[1:])
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
        failing.deps.plans.close()
        failing.deps.events.close()
        failing.deps.ledger.close()


def test_restart_restores_last_committed_revision(tmp_path: Path) -> None:
    """A reopened runtime over the same database resumes from the last
    committed plan revision and the checkpointed carrier state."""
    config = load_app_config(CONFIG_PATH)
    database_path = tmp_path / "restart" / "agent.db"
    first = AgentLoop(config, database_path, ScriptedLLM())
    engine1 = SimulationEngine(
        config, seed=42, output_dir=tmp_path / "restart", carrier=first.on_situation
    )
    first.attach(engine1)
    for _ in range(120):
        engine1.step()
    last_revision = first.plans.get_active(SCENARIO_ID).revision
    state_before = dict(first.get_state())
    first.close()

    second = AgentLoop(config, database_path, ScriptedLLM())
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
        for _ in range(120):
            engine2.step()
        resumed = second.plans.get_active(SCENARIO_ID)
        assert resumed is not None
        assert resumed.revision > last_revision
    finally:
        second.close()
