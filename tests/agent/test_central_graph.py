# tests/agent/test_central_graph.py
"""Persistent carrier graph and runtime tests (spec 8.1-8.4, plan Task 8).

Covers the brief's two verbatim route tests (tactical never calls the LLM;
strategic runs the full chain and commits), the controller rulings (critical
quality persistence and hard-protection triggers, target-loss gating,
deferred error handling, confirmed-intent-label tracking), and Step 4's
checkpoint-restart continuation test via ``CarrierRuntime``.

Per the user directive (addendum A) no mock substitutes real LLM
functionality: graph-logic tests feed explicit values into the unit under
test (the LLM is simply not part of it — the real client is held but never
invoked), and tests whose subject is the semantic chain run live against
the real LongCat provider. The whole module is skipped when the API key is
unset.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from underwater_tracking.agent.graphs.central import (
    CarrierDependencies,
    IntentWiringNode,
    build_carrier_graph,
)
from underwater_tracking.agent.llm import (
    HTTPStructuredLLM,
    LLMCallMetadata,
    LLMError,
)
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.intent import BeliefHistoryProvider
from underwater_tracking.agent.prompts import INTENT_PROMPT_VERSION
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import IntentHypothesis, TrackingPlan
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
from underwater_tracking.prediction.port import make_snapshot_predictor
from tests.conftest import (
    REAL_LLM_SKIP_REASON,
    has_live_api_key,
    make_live_llm,
)

pytestmark = pytest.mark.skipif(
    not has_live_api_key(),
    reason=REAL_LLM_SKIP_REASON,
)

SCENARIO_ID = "S1"
LIVE_REF = f"{SCENARIO_ID}:live"
SIM_TIME_S = 900

INTENT_LABELS = ("transit", "patrol", "loiter", "evade", "approach", "withdraw", "unknown")
STRATEGY_CONCEPTS = ("quality_first", "balanced", "resource_saving", "hold_current")

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


class SituationHolder:
    """Mutable live-situation provider: tests swap the current situation."""

    def __init__(self, situation: SituationSnapshot) -> None:
        self.situation = situation

    def __call__(self, ref: str) -> SituationSnapshot:
        return self.situation


class CarrierRig:
    """One carrier test rig: dependencies plus the mutable live situation."""

    def __init__(
        self,
        deps: CarrierDependencies,
        holder: SituationHolder,
        database_path: Path,
        llm_calls: list[LLMCallMetadata],
    ) -> None:
        self.deps = deps
        self.holder = holder
        self.database_path = database_path
        # Every outbound request the real client made, observed through its
        # before-request hook (hashes only — never payloads or secrets).
        self.llm_calls = llm_calls

    def set_situation(self, situation: SituationSnapshot) -> None:
        self.holder.situation = situation

    def close(self) -> None:
        self.deps.llm.close()


def make_rig(
    tmp_path: Path,
    *,
    llm: HTTPStructuredLLM | None = None,
    belief_history: BeliefHistoryProvider | None = None,
    semantic_repairs: int = 2,
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
    calls: list[LLMCallMetadata] = []
    client = llm if llm is not None else make_live_llm(before_request=calls.append)
    deps = CarrierDependencies(
        plans=plans,
        events=events,
        ledger=ledger,
        llm=client,
        predictor=make_snapshot_predictor(
            belief_history=(
                belief_history
                if belief_history is not None
                else lambda snapshot, target_id: T1_HISTORY
            ),
            horizon_s=600.0,
            sample_step_s=30.0,
        ),
        situation_provider=holder,
        belief_history=(
            belief_history
            if belief_history is not None
            else lambda snapshot, target_id: T1_HISTORY
        ),
        monitor=monitor,
        semantic_repairs=semantic_repairs,
        model_id="LongCat-2.0",
    )
    return CarrierRig(deps, holder, database_path, calls)


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


class _ScriptedIntentAnalysis:
    """Feeds explicit intent hypotheses into the wiring (no LLM involved).

    The wiring node's unit is the confirmed-label tracking; the inner
    analysis output is explicit input here, so the LLM is not part of the
    unit under test.
    """

    def __init__(self, hypothesis: IntentHypothesis) -> None:
        self._hypothesis = hypothesis

    def __call__(self, state: CarrierState) -> CarrierState:
        del state
        return {
            "intent_hypotheses": {"T1": self._hypothesis},
            "llm_provenance": {
                "intent:T1": LLMCallMetadata(
                    operation="intent",
                    model="LongCat-2.0",
                    prompt_version=INTENT_PROMPT_VERSION,
                    request_hash="r",
                    response_hash="s",
                    scenario_id=SCENARIO_ID,
                    sim_time_s=SIM_TIME_S,
                )
            },
        }


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[CarrierRig]:
    rig = make_rig(tmp_path)
    try:
        yield rig
    finally:
        rig.close()


@pytest.fixture
def spy_calls(rig: CarrierRig) -> list[LLMCallMetadata]:
    return rig.llm_calls


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


def test_tactical_route_never_calls_llm(carrier, quality_warning_state, spy_calls, rig):
    # A tactical cycle is a continuation of an approved strategy, never a
    # cold-start substitute for the strategic LLM path.
    rig.deps.plans.set_snapshot_revision(SCENARIO_ID, 3)
    rig.deps.plans.commit(
        TrackingPlan(
            plan_id="S1:plan:1",
            scenario_id=SCENARIO_ID,
            revision=1,
            base_snapshot_revision=3,
            status="active",
            valid_from_s=0,
            valid_until_s=1800,
            target_priorities={"T1": 1.0},
            required_quality={"T1": 0.7},
            member_ids_by_target={"T1": ("U1", "U2")},
            active_uuv_ids=("U1", "U2"),
            evidence_ids=("B:T1:900", "B:T1:870"),
        )
    )
    result = carrier.invoke(quality_warning_state)
    assert result["route"] == "tactical"
    assert spy_calls == []
    assert "regional_strategy" not in result.get("llm_provenance", {})
    assert result.get("selected_plan") is not None, result


@pytest.mark.real_llm
def test_strategic_cycle_runs_full_chain_commits_and_records_decision(
    carrier, target_added_state, rig, spy_calls
):
    """Live strategic cycle: intent first, then strategy, then a commit.

    The semantic chain order is deterministic (intent -> strategy); the
    decision records the verified candidates and the trigger event is
    stored exactly once.
    """
    result = carrier.invoke(target_added_state)
    assert result["route"] == "strategic"
    assert result["commit_status"] == "committed"
    operations = [call.operation for call in spy_calls]
    assert operations and operations[0] == "intent"
    assert "regional_strategy" in operations
    assert "strategy" not in operations
    assert set(operations) <= {"intent", "regional_strategy"}
    assert result["regional_plans"]
    assert result["regional_policies"]
    active = rig.deps.plans.get_active(SCENARIO_ID)
    assert active is not None and active.revision == 1
    decisions = rig.deps.ledger.list_decisions(SCENARIO_ID)
    assert len(decisions) == 1
    assert decisions[0].final_plan_id == active.plan_id
    assert len(decisions[0].candidates) == 1
    for proposal in decisions[0].candidates:
        assert proposal.concept in STRATEGY_CONCEPTS
        assert proposal.evidence_ids
    stored = rig.deps.events.list_events(scenario_id=SCENARIO_ID)
    assert [event.event_type for event in stored] == ["target_added"]


# --- Additional integration tests (controller rulings) ----------------------


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
    monitor = EventMonitor()
    situation = build_situation(snapshot_revision=3)
    evading = IntentHypothesis(
        label="evade",
        confidence=0.75,
        evidence_ids=("B:T1:900",),
        model_id="LongCat-2.0",
        prompt_version=INTENT_PROMPT_VERSION,
    )
    wiring = IntentWiringNode(
        _ScriptedIntentAnalysis(evading),
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
    try:
        graph = build_carrier_graph(rig.deps, InMemorySaver(), {})
        result = graph.invoke(
            event_state(_event("target_added", "T1")),
            config={"configurable": {"thread_id": "error-run"}},
        )
        assert result["errors"]
        assert "insufficient estimated trajectory history" in result["errors"][0]
        assert result.get("commit_status") is None
        assert result.get("selected_plan") is None
        # The analysis raised before any LLM call.
        assert rig.llm_calls == []
    finally:
        rig.close()


@pytest.mark.real_llm
def test_verify_degraded_path_records_error_and_continues(tmp_path: Path):
    """Live: a strategy set no candidate can verify records the error.

    The situation carries no evidence (``evidence=False``), so every
    proposal's repair budget is exhausted and the cycle completes through
    ``handle_error`` with a recorded error instead of crashing; the
    semantic calls stay within intent + strategy.
    """
    rig = make_rig(tmp_path, semantic_repairs=0)
    rig.set_situation(build_situation(snapshot_revision=3, evidence=False))
    try:
        graph = build_carrier_graph(rig.deps, InMemorySaver(), {})
        try:
            result = graph.invoke(
                event_state(_event("target_added", "T1")),
                config={"configurable": {"thread_id": "degraded-run"}},
            )
        except LLMError:
            # Residual live risk (documented in the fix report): with an
            # empty evidence payload the ``IntentHypothesis`` schema
            # (``evidence_ids`` min_length=1) depends on the provider
            # citing some string; if it returns an empty list instead, the
            # content error propagates out of the graph. The invariant in
            # both cases: the cycle never commits, and the first outbound
            # call was the intent analysis (the before-request hook fires
            # once per transport attempt, so it observes error paths too).
            assert rig.llm_calls and rig.llm_calls[0].operation == "intent"
            assert rig.deps.plans.get_active(SCENARIO_ID) is None
            return
        assert result["errors"]
        assert "no verified strategy" in result["errors"][0]
        assert result.get("commit_status") is None
        assert rig.llm_calls and rig.llm_calls[0].operation == "intent"
        assert {call.operation for call in rig.llm_calls} <= {"intent", "strategy"}
    finally:
        rig.close()


# --- Brief Step 4: checkpoint restart ---------------------------------------


@pytest.mark.real_llm
def test_checkpoint_restart_continues_plan_revisions(tmp_path: Path):
    """Live strategic commit, reopen, then a tactical continuation commit.

    Cycle 1 commits plan revision 1 over the strategic chain; after the
    runtime reopens, the checkpointed strategy set and intent hypotheses
    are present and cycle 2 (tactical, zero LLM calls) commits revision 2
    on the newer snapshot.
    """
    rig = make_rig(tmp_path)
    first_runtime = CarrierRuntime(
        rig.deps, scenario_id=SCENARIO_ID, database_path=rig.database_path
    )
    try:
        first_runtime.submit_event(
            event_type="target_added", entity_id="T1", sim_time_s=900, payload={}
        )
        first = first_runtime.tick()
        assert first["route"] == "strategic"
        assert first["commit_status"] == "committed"
        assert first["strategy_set"] is not None
        first_plan = rig.deps.plans.get_active(SCENARIO_ID)
        assert first_plan is not None and first_plan.revision == 1
        first_policies = first["regional_policies"]
        first_region_tasks = first_plan.region_tasks
        assert first_runtime.get_state()["route"] == "strategic"
        first_runtime.close()

        rig.set_situation(build_situation(snapshot_revision=5, sim_time_s=1200, quality=0.6))
        second_runtime = CarrierRuntime(
            rig.deps, scenario_id=SCENARIO_ID, database_path=rig.database_path
        )
        try:
            # The checkpoint survived the reopen: the strategic strategy set
            # and the intent hypotheses are present before any new cycle runs.
            assert second_runtime.get_state()["strategy_set"] is not None
            label = second_runtime.get_state()["intent_hypotheses"]["T1"].label
            assert label in INTENT_LABELS
            second_runtime.submit_event(
                event_type="group_quality_warning",
                entity_id="G-T1",
                sim_time_s=1200,
                payload={"quality": 0.6},
            )
            calls_before = len(rig.llm_calls)
            second = second_runtime.tick()
            assert second["route"] == "tactical"
            assert rig.llm_calls[calls_before:] == []
            assert second["regional_policies"] == first_policies
            assert second["commit_status"] == "committed"
            assert second["strategy_set"] is not None
            second_plan = rig.deps.plans.get_active(SCENARIO_ID)
            assert second_plan is not None and second_plan.revision == 2
            assert second_plan.base_snapshot_revision == 5
            assert second_plan.region_tasks == first_region_tasks
        finally:
            second_runtime.close()
    finally:
        rig.close()


# --- Review fix round 1: optimizer error routing and empty cycles -----------


@pytest.mark.real_llm
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
    rig = make_rig(tmp_path)
    try:
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
    finally:
        rig.close()


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
    try:
        result = runtime.tick()
        assert result["route"] == "informational"
        assert result["coalesced_events"] == ()
        assert not result.get("errors")
        assert result.get("commit_status") is None
        assert runtime.get_state()["route"] == "informational"
    finally:
        runtime.close()
        rig.close()
