# tests/agent/test_questions.py
"""Evidence-backed question and isolated counterfactual tests (spec 10.2, plan Task 11).

Covers the brief's verbatim read-only binding test (an expert question with
a counterfactual dry-run never changes the online plan), evidence retrieval
with deterministic entity matching, rejection of answers citing absent
evidence ids, the isolated dry-run (``dry-run:<uuid>`` plan id, no
repository involvement), question-run persistence with deterministic
dedupe, and the question branch surfacing the latest question run on the
checkpointed state.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from underwater_tracking.agent.counterfactual import run_counterfactual_dry_run
from underwater_tracking.agent.graphs.central import CarrierDependencies
from underwater_tracking.agent.llm import MockStructuredLLM
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.questions import (
    QUESTION_OPERATION,
    QuestionEvidenceError,
    question_run_id,
)
from underwater_tracking.agent.nodes.snapshot import build_planning_snapshot
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.domain.agent_models import PredictedTrackRef, TrackingPlan
from underwater_tracking.domain.models import (
    GroupQuality,
    GroupReport,
    SituationSnapshot,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository
from tests.fixtures.llm_responses import VALID_INTENT_HYPOTHESIS

SCENARIO_ID = "S1"
SIM_TIME_S = 900

# T2 sits at (2000, 0), inside the planner's 4000 m max range of every UUV.
TARGET_MEAN = {"T1": (0.0, 0.0), "T2": (2000.0, 0.0)}
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


def _default_responses() -> dict[str, object]:
    return {
        "intent": [VALID_INTENT_HYPOTHESIS, VALID_INTENT_HYPOTHESIS],
        "strategy": [
            _both_targets_proposal("quality_first"),
            _both_targets_proposal("balanced"),
            _both_targets_proposal("resource_saving"),
        ],
    }


class QuestionLLM(MockStructuredLLM):
    """Mock LLM answering questions from the curated payload's evidence.

    The ``question`` operation is served from the payload: the answer
    cites exactly the payload's citable evidence ids (or the injected
    override, for the rejection test), so the mock answer is deterministic
    for a given payload. All other operations are served from the FIFO
    queues. First-call operation order is recorded so tests can assert a
    question run never invokes the strategic chain.
    """

    def __init__(
        self,
        responses: dict[str, object],
        *,
        cited_evidence: list[str] | None = None,
    ) -> None:
        super().__init__(responses)
        self.operations: list[str] = []
        self._seen: set[str] = set()
        self.question_payloads: list[dict[str, Any]] = []
        self._cited = cited_evidence

    def invoke_structured(self, operation, payload, response_model, *, prompt_version=""):
        if operation not in self._seen:
            self._seen.add(operation)
            self.operations.append(operation)
        if operation != QUESTION_OPERATION:
            return super().invoke_structured(
                operation, payload, response_model, prompt_version=prompt_version
            )
        self.question_payloads.append(dict(payload))
        cited = list(payload["evidence_ids"]) if self._cited is None else list(self._cited)
        counterfactual = payload.get("counterfactual")
        answer = (
            "T2 保持双机编队：其估计跟踪质量 0.80 高于预警阈值，按资源经济原则未增派。"
        )
        if counterfactual:
            answer += f" 反事实 {counterfactual['plan_id']} 将 T2 增至三机编队。"
        return response_model.model_validate({"answer": answer, "evidence_ids": cited})


def _group_report(target_id: str) -> GroupReport:
    return GroupReport(
        group_id=f"G-{target_id}",
        target_id=target_id,
        sim_time_s=SIM_TIME_S,
        member_ids=(),
        belief=TargetBelief(
            target_id=target_id,
            sim_time_s=SIM_TIME_S,
            mean=(*TARGET_MEAN[target_id], 1.0, 0.5),
            covariance=(
                (400.0, 0.0, 0.0, 0.0),
                (0.0, 400.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            model_probabilities={"cv": 0.7, "ct": 0.3},
            source_observation_ids=(f"B:{target_id}:900", f"B:{target_id}:870"),
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


def build_two_target_situation(
    *, snapshot_revision: int, sim_time_s: int = SIM_TIME_S
) -> SituationSnapshot:
    """A deterministic world: six UUVs and two tracked targets at 0.80 quality."""
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
    return SituationSnapshot(
        scenario_id=SCENARIO_ID,
        snapshot_revision=snapshot_revision,
        sim_time_s=sim_time_s,
        uuvs=uuvs,
        group_reports=tuple(_group_report(target_id) for target_id in ("T1", "T2")),
        pending_events=(),
    )


class SituationHolder:
    """Mutable live-situation provider: tests swap the current situation."""

    def __init__(self, situation: SituationSnapshot) -> None:
        self.situation = situation

    def __call__(self, ref: str) -> SituationSnapshot:
        return self.situation


class QuestionHarness:
    """CarrierRuntime wrapper exposing the question-test helpers."""

    def __init__(
        self,
        runtime: CarrierRuntime,
        deps: CarrierDependencies,
        holder: SituationHolder,
    ) -> None:
        self._runtime = runtime
        self._deps = deps
        self._holder = holder

    def active_plan(self) -> TrackingPlan | None:
        return self._deps.plans.get_active(SCENARIO_ID)

    def ask(self, raw_text: str, counterfactual: dict[str, object] | None = None) -> Any:
        return self._runtime.ask(raw_text, counterfactual=counterfactual)

    def submit_event(self, event_type: str, entity_id: str, sim_time_s: int) -> None:
        self._runtime.submit_event(
            event_type=event_type, entity_id=entity_id, sim_time_s=sim_time_s
        )

    def tick(self) -> dict[str, Any]:
        return self._runtime.tick()

    def get_state(self) -> dict[str, Any]:
        return self._runtime.get_state()

    def questions(self) -> list[Any]:
        return self._deps.ledger.list_questions(SCENARIO_ID)

    def events(self, *, event_type: str) -> list[Any]:
        return self._deps.events.list_events(
            scenario_id=SCENARIO_ID, event_type=event_type
        )

    def operations(self) -> list[str]:
        return self._deps.llm.operations

    def question_payloads(self) -> list[dict[str, Any]]:
        return self._deps.llm.question_payloads

    def situation(self) -> SituationSnapshot:
        return self._holder.situation

    def close(self) -> None:
        self._runtime.close()


def make_harness(
    tmp_path: Path, *, cited_evidence: list[str] | None = None
) -> QuestionHarness:
    """One question rig: injected dependencies over one SQLite database.

    Raw bearing observations are seeded into the EventRepository so the
    evidence-id lookup path is genuinely exercised; the live situation
    tracks T1 and T2 at 0.80 quality with no members assigned yet.
    """
    database_path = tmp_path / "questions.db"
    plans = PlanRepository(database_path)
    events = EventRepository(database_path)
    ledger = DecisionLedger(database_path)
    for target_id in ("T1", "T2"):
        for stamp in (900, 870):
            events.append(
                event_id=f"B:{target_id}:{stamp}",
                event_type="bearing_observation",
                scenario_id=SCENARIO_ID,
                sim_time_s=stamp,
                target_id=target_id,
                payload={"azimuth_rad": 0.0},
            )
    holder = SituationHolder(build_two_target_situation(snapshot_revision=3))
    deps = CarrierDependencies(
        plans=plans,
        events=events,
        ledger=ledger,
        llm=QuestionLLM(_default_responses(), cited_evidence=cited_evidence),
        predictor=_straight_line_predictor,
        situation_provider=holder,
        belief_history=lambda snapshot, target_id: T1_HISTORY,
        monitor=EventMonitor(scenario_id=SCENARIO_ID),
    )
    runtime = CarrierRuntime(deps, scenario_id=SCENARIO_ID, database_path=database_path)
    return QuestionHarness(runtime, deps, holder)


def prime_plan(harness: QuestionHarness) -> TrackingPlan:
    """Commit plan revision 1 over both targets via one strategic cycle."""
    harness.submit_event(
        event_type="target_added", entity_id="T1", sim_time_s=SIM_TIME_S
    )
    result = harness.tick()
    assert result["commit_status"] == "committed"
    active = harness.active_plan()
    assert active is not None and active.revision == 1
    return active


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


@pytest.fixture
def runtime(tmp_path: Path) -> Iterator[QuestionHarness]:
    harness = make_harness(tmp_path)
    prime_plan(harness)
    yield harness
    harness.close()


# --- Brief Step 1: verbatim read-only question test -------------------------


def test_question_and_counterfactual_do_not_change_online_plan(runtime):
    before = runtime.active_plan()
    answer = runtime.ask(
        "为什么没有给 T2 增派 UUV？", counterfactual={"T2.min_quality": 0.85}
    )
    assert answer.evidence_ids
    assert answer.counterfactual_plan_id.startswith("dry-run:")
    assert runtime.active_plan() == before


# --- Evidence retrieval (brief Step 2) --------------------------------------


def test_question_without_counterfactual_has_no_dry_run(runtime):
    answer = runtime.ask("为什么 T2 是双机编队？")
    assert answer.evidence_ids
    assert answer.counterfactual_plan_id is None
    assert answer.counterfactual_summary is None


def test_question_evidence_retrieval_builds_bounded_payload(runtime):
    answer = runtime.ask(
        "为什么没有给 T2 增派 UUV？", counterfactual={"T2.min_quality": 0.85}
    )
    assert answer.evidence_ids
    payloads = runtime.question_payloads()
    assert len(payloads) == 1
    payload = payloads[0]
    # Deterministic entity matching: only T2 was named in the question.
    assert payload["matched_entities"]["target_ids"] == ["T2"]
    assert payload["matched_entities"]["uuv_ids"] == []
    # The citable namespace: the decision's evidence ids, the trigger
    # event id, and the plan's evidence ids.
    assert set(payload["evidence_ids"]) == {
        "B:T1:870",
        "B:T1:900",
        "B:T2:870",
        "B:T2:900",
        "S1:target_added:T1:900",
    }
    # The ledger decision and the plan diff channel are present.
    assert len(payload["decisions"]) == 1
    assert payload["decisions"][0]["decision_id"] == "S1:decision:3"
    assert payload["active_plan"] is not None
    # Observations are resolved by evidence id from the event repository.
    observation_ids = {observation["event_id"] for observation in payload["observations"]}
    assert "B:T2:900" in observation_ids
    assert "S1:target_added:T1:900" in observation_ids
    # The counterfactual dry-run summary is part of the payload.
    counterfactual = payload["counterfactual"]
    assert counterfactual["run_id"].startswith("dry-run:")
    assert counterfactual["plan_id"].startswith("dry-run:")


def test_question_rejects_answers_citing_absent_evidence(tmp_path: Path):
    harness = make_harness(tmp_path, cited_evidence=["B:GHOST:1"])
    prime_plan(harness)
    try:
        with pytest.raises(QuestionEvidenceError, match="B:GHOST:1"):
            harness.ask("为什么没有给 T2 增派 UUV？")
    finally:
        harness.close()


def test_question_never_invokes_the_carrier_graph(runtime):
    state_before = dict(runtime.get_state())
    plan_before = runtime.active_plan()
    runtime.ask("为什么没有给 T2 增派 UUV？", counterfactual={"T2.min_quality": 0.85})
    # Only the question operation ran: no intent/strategy/verify calls.
    assert runtime.operations() == ["intent", "strategy", "question"]
    assert runtime.active_plan() == plan_before
    assert dict(runtime.get_state()) == state_before


# --- Question-run persistence and the branch surface -------------------------


def test_question_run_persisted_once_and_reask_dedupes(runtime):
    first = runtime.ask("为什么没有给 T2 增派 UUV？")
    second = runtime.ask("为什么没有给 T2 增派 UUV？")
    assert first.evidence_ids == second.evidence_ids
    runs = runtime.questions()
    assert len(runs) == 1
    assert runs[0].status == "completed"
    assert runs[0].question_text == "为什么没有给 T2 增派 UUV？"


def test_question_event_surfaces_latest_question_on_next_cycle(runtime):
    raw_text = "为什么没有给 T2 增派 UUV？"
    overrides = {"T2.min_quality": 0.85}
    runtime.ask(raw_text, counterfactual=overrides)
    run_id = question_run_id(SCENARIO_ID, raw_text, overrides)
    result = runtime.tick()
    assert result["route"] == "informational"
    assert runtime.get_state().get("latest_question") == run_id
    emitted = runtime.events(event_type="question")
    assert len(emitted) == 1
    assert emitted[0].payload["run_id"] == run_id


# --- Isolated counterfactual dry-run (brief Step 3) -------------------------


def test_counterfactual_dry_run_grows_unmet_quality_target(runtime):
    answer = runtime.ask(
        "为什么没有给 T2 增派 UUV？", counterfactual={"T2.min_quality": 0.85}
    )
    # The online plan still keeps T2 at two members.
    assert runtime.active_plan() is not None
    assert len(runtime.active_plan().member_ids_by_target["T2"]) == 2
    # The dry-run summary documents the reinforcement the requirement would
    # force (measured 0.80 < required 0.85 -> quality risk -> group grows).
    assert answer.counterfactual_summary is not None
    assert "T2" in answer.counterfactual_summary
    assert "added" in answer.counterfactual_summary


def test_counterfactual_dry_run_is_deterministic_with_fixed_run_id(runtime):
    snapshot = build_planning_snapshot(
        runtime.situation(), active_plan=runtime.active_plan()
    )
    first = run_counterfactual_dry_run(
        snapshot, {"T2.min_quality": 0.85}, run_id="dry-run:fixed"
    )
    second = run_counterfactual_dry_run(
        snapshot, {"T2.min_quality": 0.85}, run_id="dry-run:fixed"
    )
    assert first.plan_id == second.plan_id == "dry-run:fixed:plan:S1:2"
    assert first.diff is not None and "T2" in first.diff.members_added
    assert first.plan.member_ids_by_target["T2"] == second.plan.member_ids_by_target["T2"]
    assert first.objective.active_count_after > first.objective.active_count_before


def test_counterfactual_disabled_uuv_override_is_validated(runtime):
    snapshot = build_planning_snapshot(
        runtime.situation(), active_plan=runtime.active_plan()
    )
    result = run_counterfactual_dry_run(
        snapshot, {"U6.disabled": True}, run_id="dry-run:fixed"
    )
    assert result.plan_id.startswith("dry-run:")


def test_counterfactual_rejects_unknown_override_keys(runtime):
    with pytest.raises(ValueError, match="unknown counterfactual override"):
        runtime.ask(
            "为什么没有给 T2 增派 UUV？", counterfactual={"T2.warp_factor": 9}
        )


def test_counterfactual_rejects_unknown_entities_and_bad_values(runtime):
    with pytest.raises(ValueError, match="no group report"):
        runtime.ask(
            "为什么没有给 T2 增派 UUV？", counterfactual={"T-NOPE.min_quality": 0.85}
        )
    with pytest.raises(ValueError, match="outside"):
        runtime.ask(
            "为什么没有给 T2 增派 UUV？", counterfactual={"T2.min_quality": 1.5}
        )
