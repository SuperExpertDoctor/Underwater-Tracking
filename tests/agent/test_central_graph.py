# tests/agent/test_central_graph.py
"""Persistent carrier graph and runtime tests (spec 8.1-8.4, plan Task 8).

Covers the brief's two verbatim route tests (tactical never calls the LLM;
strategic runs the full chain and commits), the controller rulings (critical
quality persistence and hard-protection triggers, target-loss gating,
deferred error handling, confirmed-intent-label tracking), and Step 4's
checkpoint-restart continuation test via ``CarrierRuntime``.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from underwater_tracking.agent.graphs.central import (
    CarrierDependencies,
    IntentWiringNode,
    build_carrier_graph,
)
from underwater_tracking.agent.llm import MockStructuredLLM
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.intent import BeliefHistoryProvider, IntentAnalysisNode
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import PredictedTrackRef
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
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository
from tests.fixtures.llm_responses import (
    EVADING_INTENT_HYPOTHESIS,
    VALID_INTENT_HYPOTHESIS,
    VALID_STRATEGY_PROPOSAL,
)

SCENARIO_ID = "S1"
LIVE_REF = f"{SCENARIO_ID}:live"
SIM_TIME_S = 900

TARGET_MEAN = {"T1": (0.0, 0.0)}
UUV_POSITIONS = {
    "U1": (500.0, 0.0),
    "U2": (0.0, 500.0),
    "U3": (1500.0, 0.0),
    "U4": (2500.0, 0.0),
    "U5": (0.0, -1000.0),
    "U6": (3000.0, 0.0),
}

# Estimated per-target belief history for the intent analysis.
T1_HISTORY: tuple[tuple[int, float, float], ...] = (
    (600, 80.0, 150.0),
    (660, 90.0, 170.0),
    (720, 100.0, 190.0),
    (780, 110.0, 205.0),
    (840, 120.0, 215.0),
    (900, 130.0, 220.0),
)

QUALITY_FIRST_PROPOSAL = {
    "concept": "quality_first",
    "target_priorities": {"T1": 1.0},
    "required_quality": {"T1": 0.7},
    "reinforcement_policy": {"T1": "release_when_stable"},
    "releasable_soft_constraints": ["energy_reserve_0.1"],
    "evidence_ids": ["B:T1:900"],
    "rationale": "quality first keeps the target locked",
}

RESOURCE_SAVING_PROPOSAL = {
    "concept": "resource_saving",
    "target_priorities": {"T1": 1.0},
    "required_quality": {"T1": 0.7},
    "reinforcement_policy": {"T1": "release_when_stable"},
    "releasable_soft_constraints": ["energy_reserve_0.1"],
    "evidence_ids": ["B:T1:900"],
    "rationale": "resource saving holds the group small",
}


class SpyLLM(MockStructuredLLM):
    """Mock LLM recording one call record per operation, in first-call order.

    The strategic chain invokes the strategy operation once per concept
    (three times), so the integration spy records distinct operations —
    the binding test asserts exactly ``["intent", "strategy"]``.
    """

    def __init__(self, responses: dict[str, object]) -> None:
        super().__init__(responses)
        self.calls: list[SimpleNamespace] = []
        self._seen: set[str] = set()

    def invoke_structured(self, operation, payload, response_model, *, prompt_version=""):
        if operation not in self._seen:
            self._seen.add(operation)
            self.calls.append(SimpleNamespace(operation=operation))
        return super().invoke_structured(
            operation, payload, response_model, prompt_version=prompt_version
        )


def build_situation(
    *,
    snapshot_revision: int,
    sim_time_s: int = SIM_TIME_S,
    quality: float = 0.8,
    evidence: bool = True,
    hard_guard_reasons: tuple[str, ...] = (),
) -> SituationSnapshot:
    """A deterministic world: six UUVs and one group report for target T1."""
    uuvs = tuple(
        UUVState(
            uuv_id=uuv_id,
            position_xy=UUV_POSITIONS[uuv_id],
            heading_rad=0.0,
            speed_mps=20.0,
            energy_fraction=0.9,
            status=UUVStatus.TRACKING,
            group_id=None,
        )
        for uuv_id in sorted(UUV_POSITIONS)
    )
    reports = (
        GroupReport(
            group_id="G-T1",
            target_id="T1",
            sim_time_s=sim_time_s,
            member_ids=(),
            belief=TargetBelief(
                target_id="T1",
                sim_time_s=sim_time_s,
                mean=(*TARGET_MEAN["T1"], 1.0, 0.5),
                covariance=(
                    (400.0, 0.0, 0.0, 0.0),
                    (0.0, 400.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                model_probabilities={"cv": 0.7, "ct": 0.3},
                source_observation_ids=(("B:T1:900", "B:T1:870") if evidence else ()),
                fim_min_eigenvalue=0.005,
                fim_condition=12.0,
            ),
            quality=GroupQuality(
                instant=quality,
                window_mean=quality,
                ewma=quality,
                components={"cov": 0.7},
                hard_guard_reasons=hard_guard_reasons,
            ),
            plan_revision=1,
        ),
    )
    return SituationSnapshot(
        scenario_id=SCENARIO_ID,
        snapshot_revision=snapshot_revision,
        sim_time_s=sim_time_s,
        uuvs=uuvs,
        group_reports=reports,
        pending_events=(),
    )


def _straight_line_predictor(snapshot: SituationSnapshot, target_id: str) -> PredictedTrackRef:
    """Deterministic straight-line prediction stub for the predictor port."""
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


class SituationHolder:
    """Mutable live-situation provider: tests swap the current situation."""

    def __init__(self, situation: SituationSnapshot) -> None:
        self.situation = situation

    def __call__(self, ref: str) -> SituationSnapshot:
        return self.situation


class CarrierRig:
    """One carrier test rig: dependencies plus the mutable live situation."""

    def __init__(self, deps: CarrierDependencies, holder: SituationHolder, database_path: Path) -> None:
        self.deps = deps
        self.holder = holder
        self.database_path = database_path

    def set_situation(self, situation: SituationSnapshot) -> None:
        self.holder.situation = situation


def make_rig(
    tmp_path: Path,
    *,
    llm: SpyLLM | None = None,
    belief_history: BeliefHistoryProvider | None = None,
) -> CarrierRig:
    """A complete injected-dependency rig over one SQLite database file."""
    database_path = tmp_path / "carrier.db"
    plans = PlanRepository(database_path)
    events = EventRepository(database_path)
    ledger = DecisionLedger(database_path)
    holder = SituationHolder(build_situation(snapshot_revision=3))
    monitor = EventMonitor(
        scenario_id=SCENARIO_ID,
        critical_hold_s=30,
        target_lost_gap_s=300,
        covariance_cap_m2=50_000.0,
    )
    deps = CarrierDependencies(
        plans=plans,
        events=events,
        ledger=ledger,
        llm=llm if llm is not None else SpyLLM(_default_responses()),
        predictor=_straight_line_predictor,
        situation_provider=holder,
        belief_history=(
            belief_history
            if belief_history is not None
            else lambda snapshot, target_id: T1_HISTORY
        ),
        monitor=monitor,
    )
    return CarrierRig(deps, holder, database_path)


def _default_responses() -> dict[str, object]:
    return {
        "intent": [VALID_INTENT_HYPOTHESIS],
        "strategy": [
            QUALITY_FIRST_PROPOSAL,
            VALID_STRATEGY_PROPOSAL,
            RESOURCE_SAVING_PROPOSAL,
        ],
    }


def _both_targets_proposal(concept: str) -> dict[str, object]:
    """A strategy proposal covering both tracked targets (T1 and T2)."""
    return {
        "concept": concept,
        "target_priorities": {"T1": 1.0, "T2": 1.0},
        "required_quality": {"T1": 0.7, "T2": 0.7},
        "reinforcement_policy": {
            "T1": "release_when_stable",
            "T2": "release_when_stable",
        },
        "releasable_soft_constraints": ["energy_reserve_0.1"],
        "evidence_ids": ["B:T1:900", "B:T2:900"],
        "rationale": f"{concept} keeps both targets locked",
    }


def build_two_target_situation(
    *, snapshot_revision: int, sim_time_s: int = SIM_TIME_S
) -> SituationSnapshot:
    """A world with six UUVs and two tracked targets (T1 and T2).

    T2 mirrors T1's belief exactly, so the shared fixture history and
    predictor ports behave identically for both targets.
    """
    single = build_situation(snapshot_revision=snapshot_revision, sim_time_s=sim_time_s)
    second = GroupReport(
        group_id="G-T2",
        target_id="T2",
        sim_time_s=sim_time_s,
        member_ids=(),
        belief=TargetBelief(
            target_id="T2",
            sim_time_s=sim_time_s,
            mean=(*TARGET_MEAN["T1"], 1.0, 0.5),
            covariance=(
                (400.0, 0.0, 0.0, 0.0),
                (0.0, 400.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            model_probabilities={"cv": 0.7, "ct": 0.3},
            source_observation_ids=("B:T2:900", "B:T2:870"),
            fim_min_eigenvalue=0.005,
            fim_condition=12.0,
        ),
        quality=GroupQuality(
            instant=0.8,
            window_mean=0.8,
            ewma=0.8,
            components={"cov": 0.7},
            hard_guard_reasons=(),
        ),
        plan_revision=1,
    )
    return SituationSnapshot(
        scenario_id=SCENARIO_ID,
        snapshot_revision=snapshot_revision,
        sim_time_s=sim_time_s,
        uuvs=single.uuvs,
        group_reports=(*single.group_reports, second),
        pending_events=(),
    )


def _event(
    event_type: str,
    entity_id: str,
    sim_time_s: int = SIM_TIME_S,
    payload: dict[str, Any] | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"{SCENARIO_ID}:{event_type}:{sim_time_s}",
        scenario_id=SCENARIO_ID,
        sim_time_s=sim_time_s,
        event_type=event_type,
        entity_id=entity_id,
        level=EventLevel.INFORMATIONAL,
        payload=payload or {},
    )


def event_state(*events: RuntimeEvent) -> CarrierState:
    return {"scenario_id": SCENARIO_ID, "snapshot_ref": LIVE_REF, "pending_events": events}


class CarrierInvoker:
    """Compiled carrier graph that pins one deterministic thread per fixture.

    The binding tests invoke ``carrier.invoke(state)`` without a config;
    langgraph requires a thread id whenever a checkpointer is compiled in,
    so the harness injects a fixed thread id per fixture instance.
    """

    def __init__(self, graph: Any, thread_id: str) -> None:
        self._graph = graph
        self._config = {"configurable": {"thread_id": thread_id}}

    def invoke(self, state: CarrierState) -> dict[str, Any]:
        return self._graph.invoke(state, config=self._config)


@pytest.fixture
def rig(tmp_path: Path) -> CarrierRig:
    return make_rig(tmp_path)


@pytest.fixture
def spy_llm(rig: CarrierRig) -> SpyLLM:
    return rig.deps.llm


@pytest.fixture
def carrier(rig: CarrierRig) -> CarrierInvoker:
    graph = build_carrier_graph(rig.deps, InMemorySaver(), {})
    return CarrierInvoker(graph, f"carrier-test:{id(graph)}")


@pytest.fixture
def quality_warning_state(rig: CarrierRig) -> CarrierState:
    rig.set_situation(build_situation(snapshot_revision=3, quality=0.6))
    return event_state(_event("group_quality_warning", "G-T1"))


@pytest.fixture
def target_added_state(rig: CarrierRig) -> CarrierState:
    rig.set_situation(build_situation(snapshot_revision=3))
    return event_state(_event("target_added", "T1"))


# --- Brief Step 1: verbatim route integration tests -------------------------


def test_tactical_route_never_calls_llm(carrier, quality_warning_state, spy_llm):
    result = carrier.invoke(quality_warning_state)
    assert result["route"] == "tactical"
    assert spy_llm.calls == []
    assert result["selected_plan"] is not None


def test_strategic_route_runs_full_chain(carrier, target_added_state, spy_llm):
    result = carrier.invoke(target_added_state)
    assert [call.operation for call in spy_llm.calls] == ["intent", "strategy"]
    assert result["commit_status"] == "committed"


# --- Additional integration tests (controller rulings) ----------------------


def test_strategic_cycle_commits_and_records_decision(carrier, target_added_state, rig):
    result = carrier.invoke(target_added_state)
    assert result["commit_status"] == "committed"
    active = rig.deps.plans.get_active(SCENARIO_ID)
    assert active is not None and active.revision == 1
    decisions = rig.deps.ledger.list_decisions(SCENARIO_ID)
    assert len(decisions) == 1
    assert decisions[0].final_plan_id == active.plan_id
    assert [proposal.concept for proposal in decisions[0].candidates] == [
        "quality_first",
        "balanced",
        "resource_saving",
    ]
    stored = rig.deps.events.list_events(scenario_id=SCENARIO_ID)
    assert [event.event_type for event in stored] == ["target_added"]


def test_critical_quality_requires_thirty_second_persistence():
    monitor = EventMonitor(critical_threshold=0.40, critical_hold_s=30)
    assert monitor.observe_quality("G-T1", 0, 0.35) == ()
    assert monitor.observe_quality("G-T1", 29, 0.30) == ()
    critical = monitor.observe_quality("G-T1", 30, 0.35)
    assert [event.event_type for event in critical] == ["group_quality_critical"]
    assert critical[0].level == EventLevel.STRATEGIC
    assert critical[0].payload["quality"] == 0.35


def test_critical_quality_recovers_before_hold_resets_the_streak():
    monitor = EventMonitor(critical_threshold=0.40, critical_hold_s=30)
    assert monitor.observe_quality("G-T1", 0, 0.35) == ()
    assert monitor.observe_quality("G-T1", 10, 0.90) == ()
    assert monitor.observe_quality("G-T1", 20, 0.35) == ()
    assert monitor.observe_quality("G-T1", 45, 0.35) == ()
    critical = monitor.observe_quality("G-T1", 50, 0.35)
    assert [event.event_type for event in critical] == ["group_quality_critical"]


def test_hard_protection_trigger_escalates_critical_immediately():
    monitor = EventMonitor(critical_threshold=0.40, critical_hold_s=30)
    critical = monitor.observe_quality(
        "G-T1", 100, 0.80, hard_guard_reasons=("covariance_out_of_bounds",)
    )
    assert len(critical) == 1
    assert critical[0].event_type == "group_quality_critical"
    assert critical[0].level == EventLevel.STRATEGIC
    assert critical[0].payload["hard_guard_reasons"] == ["covariance_out_of_bounds"]


def test_target_lost_requires_gap_and_covariance_above_cap():
    monitor = EventMonitor(target_lost_gap_s=300, covariance_cap_m2=50_000.0)
    # Short gap: never lost.
    assert (
        monitor.observe_bearing_gap(
            "G-T1", 900, last_gated_bearing_s=700, position_covariance_trace=1_000_000.0
        )
        == ()
    )
    # Long gap but covariance under the scenario cap: still tracked.
    assert (
        monitor.observe_bearing_gap(
            "G-T1", 900, last_gated_bearing_s=500, position_covariance_trace=10_000.0
        )
        == ()
    )
    # Long gap AND covariance above the cap: confirm target loss.
    lost = monitor.observe_bearing_gap(
        "G-T1", 900, last_gated_bearing_s=500, position_covariance_trace=1_000_000.0
    )
    assert [event.event_type for event in lost] == ["target_lost"]
    assert lost[0].level == EventLevel.STRATEGIC
    assert lost[0].payload["gap_s"] == 400
    assert lost[0].payload["last_gated_bearing_s"] == 500


def test_intent_wiring_tracks_confirmed_labels():
    llm = MockStructuredLLM(
        {
            "intent": [
                EVADING_INTENT_HYPOTHESIS,
                EVADING_INTENT_HYPOTHESIS,
                EVADING_INTENT_HYPOTHESIS,
            ]
        }
    )
    monitor = EventMonitor()
    situation = build_situation(snapshot_revision=3)
    wiring = IntentWiringNode(
        IntentAnalysisNode(
            llm,
            model_id="mock",
            belief_history=lambda snapshot, target_id: T1_HISTORY,
            snapshot_provider=lambda ref: situation,
        ),
        monitor,
        lambda ref: situation,
    )
    base: CarrierState = {"scenario_id": SCENARIO_ID, "snapshot_ref": LIVE_REF}
    first = wiring(base)
    assert first["confirmed_intent_labels"] == {}
    assert first["coalesced_events"] == ()
    second = wiring({**base, **first})
    assert second["confirmed_intent_labels"] == {"T1": "evade"}
    assert [event.event_type for event in second["coalesced_events"]] == [
        "intent_change_confirmed"
    ]
    # The unchanged label is never re-confirmed: no new event joins the
    # cycle's coalesced events and the confirmed tracking stays.
    third = wiring({**base, **second})
    assert third["coalesced_events"] == second["coalesced_events"]
    assert third["confirmed_intent_labels"] == {"T1": "evade"}


def test_intent_history_too_short_routes_to_handle_error(tmp_path: Path):
    rig = make_rig(
        tmp_path,
        belief_history=lambda snapshot, target_id: ((600, 0.0, 0.0), (900, 5.0, 0.0)),
    )
    graph = build_carrier_graph(rig.deps, InMemorySaver(), {})
    result = graph.invoke(
        event_state(_event("target_added", "T1")),
        config={"configurable": {"thread_id": "error-run"}},
    )
    assert result["errors"]
    assert "insufficient estimated trajectory history" in result["errors"][0]
    assert result.get("commit_status") is None
    assert result.get("selected_plan") is None


def test_verify_degraded_path_records_error_and_continues(tmp_path: Path):
    rig = make_rig(tmp_path)
    rig.set_situation(build_situation(snapshot_revision=3, evidence=False))
    graph = build_carrier_graph(rig.deps, InMemorySaver(), {})
    result = graph.invoke(
        event_state(_event("target_added", "T1")),
        config={"configurable": {"thread_id": "degraded-run"}},
    )
    assert result["errors"]
    assert "no verified strategy" in result["errors"][0]
    assert result.get("commit_status") is None
    assert [call.operation for call in rig.deps.llm.calls] == ["intent", "strategy"]


# --- Brief Step 4: checkpoint restart ---------------------------------------


def test_checkpoint_restart_continues_plan_revisions(tmp_path: Path):
    rig = make_rig(tmp_path)
    first_runtime = CarrierRuntime(
        rig.deps, scenario_id=SCENARIO_ID, database_path=rig.database_path
    )
    first_runtime.submit_event(
        event_type="target_added", entity_id="T1", sim_time_s=900, payload={}
    )
    first = first_runtime.tick()
    assert first["route"] == "strategic"
    assert first["commit_status"] == "committed"
    assert first["strategy_set"] is not None
    first_plan = rig.deps.plans.get_active(SCENARIO_ID)
    assert first_plan is not None and first_plan.revision == 1
    assert first_runtime.get_state()["route"] == "strategic"
    first_runtime.close()

    rig.set_situation(build_situation(snapshot_revision=5, sim_time_s=1200, quality=0.6))
    second_runtime = CarrierRuntime(
        rig.deps, scenario_id=SCENARIO_ID, database_path=rig.database_path
    )
    # The checkpoint survived the reopen: the strategic strategy set and the
    # intent hypotheses are present before any new cycle runs.
    assert second_runtime.get_state()["strategy_set"] is not None
    assert second_runtime.get_state()["intent_hypotheses"]["T1"].label == "transit"
    second_runtime.submit_event(
        event_type="group_quality_warning",
        entity_id="G-T1",
        sim_time_s=1200,
        payload={"quality": 0.6},
    )
    calls_before = len(rig.deps.llm.calls)
    second = second_runtime.tick()
    assert second["route"] == "tactical"
    assert rig.deps.llm.calls[calls_before:] == []
    assert second["commit_status"] == "committed"
    assert second["strategy_set"] is not None
    second_plan = rig.deps.plans.get_active(SCENARIO_ID)
    assert second_plan is not None and second_plan.revision == 2
    assert second_plan.revision == first_plan.revision + 1
    assert second_plan.base_snapshot_revision == 5
    second_runtime.close()


# --- Review fix round 1: optimizer error routing and empty cycles -----------


def test_optimizer_infeasibility_routes_to_handle_error_and_does_not_divert_next_cycle(
    tmp_path: Path,
) -> None:
    """Review fix 1: an infeasible optimization is deferred to handle_error.

    Cycle 1 commits a two-target strategic plan; cycle 2 loses T2 from the
    live situation while the checkpointed strategy set still proposes for
    it, so the optimizer fails. The failure must be appended to ``errors``
    and the recorded ``node_error`` cleared — never re-validating or
    re-committing the stale selected plan — and cycle 3 must not be
    diverted by the stale error but commit the next revision.
    """
    rig = make_rig(
        tmp_path,
        llm=SpyLLM(
            {
                "intent": [VALID_INTENT_HYPOTHESIS, VALID_INTENT_HYPOTHESIS],
                "strategy": [
                    _both_targets_proposal("quality_first"),
                    _both_targets_proposal("balanced"),
                    _both_targets_proposal("resource_saving"),
                ],
            }
        ),
    )
    graph = build_carrier_graph(rig.deps, InMemorySaver(), {})
    carrier = CarrierInvoker(graph, "optimizer-infeasible")

    # Cycle 1 (strategic): both targets tracked; plan committed at rev 1.
    rig.set_situation(build_two_target_situation(snapshot_revision=3))
    first = carrier.invoke(event_state(_event("target_added", "T1")))
    assert first["commit_status"] == "committed"
    first_active = rig.deps.plans.get_active(SCENARIO_ID)
    assert first_active is not None and first_active.revision == 1

    # Cycle 2 (tactical): T2 disappeared, the checkpointed strategy set
    # still proposes for it -> the optimizer is infeasible.
    rig.set_situation(build_situation(snapshot_revision=4, sim_time_s=1200, quality=0.6))
    second = carrier.invoke(
        event_state(_event("group_quality_warning", "G-T1", sim_time_s=1200))
    )
    assert len(second["errors"]) == 1
    assert "resource_optimizer failed" in second["errors"][0]
    assert "no group report for target 'T2'" in second["errors"][0]
    assert second["node_error"] is None
    # The stale selected plan was NOT re-validated or re-committed.
    second_active = rig.deps.plans.get_active(SCENARIO_ID)
    assert second_active is not None and second_active.revision == 1

    # Cycle 3 (tactical): targets are back; the stale node_error must not
    # divert the cycle away from the commit path.
    rig.set_situation(build_two_target_situation(snapshot_revision=6, sim_time_s=1500))
    third = carrier.invoke(
        event_state(_event("group_quality_warning", "G-T1", sim_time_s=1500))
    )
    assert third["node_error"] is None
    assert third["commit_status"] == "committed"
    third_active = rig.deps.plans.get_active(SCENARIO_ID)
    assert third_active is not None and third_active.revision == 2
    assert len(third["errors"]) == 1


def test_runtime_tick_with_no_pending_events_routes_informational(tmp_path: Path):
    """Review fix 2: an empty cycle routes informational instead of crashing.

    ``CarrierRuntime.tick()`` with an empty pending queue (a reopen
    continuation) must complete an informational cycle: no plan work, no
    recorded errors, and no crash on the empty coalesced-event sequence.
    """
    rig = make_rig(tmp_path)
    runtime = CarrierRuntime(
        rig.deps, scenario_id=SCENARIO_ID, database_path=rig.database_path
    )
    result = runtime.tick()
    assert result["route"] == "informational"
    assert result["coalesced_events"] == ()
    assert not result.get("errors")
    assert result.get("commit_status") is None
    assert runtime.get_state()["route"] == "informational"
    runtime.close()
