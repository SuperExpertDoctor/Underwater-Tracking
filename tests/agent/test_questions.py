# tests/agent/test_questions.py
"""Evidence-backed question and isolated counterfactual tests (spec 10.2, plan Task 11).

Covers the brief's verbatim read-only binding test (an expert question with
a counterfactual dry-run never changes the online plan), evidence retrieval
with deterministic entity matching, rejection of answers citing absent
evidence ids, the isolated dry-run (``dry-run:<uuid>`` plan id, no
repository involvement), question-run persistence with deterministic
dedupe, and the question branch surfacing the latest question run on the
checkpointed state.

Per the user directive (addendum A) no mock substitutes real LLM
functionality: the only LLM behavior here — writing the natural-language
answer — runs live against the real LongCat provider. Everything else is
deterministic: entity matching, evidence retrieval, payload building,
answer validation, counterfactual dry-runs, and run-id dedupe are driven
explicitly, and the online plan/decision state is seeded directly through
the repositories (no live strategic cycle per test). The former mock
answer queue (``QuestionLLM``) was deleted as an accepted consequence. The
whole module is skipped when the API key is unset.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from underwater_tracking.agent.counterfactual import run_counterfactual_dry_run
from underwater_tracking.agent.graphs.central import CarrierDependencies
from underwater_tracking.agent.llm import HTTPStructuredLLM, LLMCallMetadata
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.questions import (
    QuestionAnswer,
    QuestionEvidence,
    QuestionEvidenceError,
    build_question_payload,
    match_question_entities,
    question_run_id,
    retrieve_question_evidence,
    validate_question_answer,
)
from underwater_tracking.agent.nodes.snapshot import (
    PlanningSnapshot,
    build_planning_snapshot,
)
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.domain.agent_models import DecisionRecord, TrackingPlan
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
from underwater_tracking.prediction.port import make_snapshot_predictor
from tests.conftest import make_live_llm

pytestmark = pytest.mark.skipif(
    not os.environ.get("UNDERWATER_TRACKING_API_KEY"),
    reason="UNDERWATER_TRACKING_API_KEY is not set; the live LongCat API tests are skipped",
)

SCENARIO_ID = "S1"
SIM_TIME_S = 900
RAW_QUESTION = "为什么没有给 T2 增派 UUV？"
COUNTERFACTUAL = {"T2.min_quality": 0.85}

# The four bearing ids shared by the seeded decision and plan, plus the
# trigger event id: the complete citable evidence namespace.
EVIDENCE_IDS = ("B:T1:900", "B:T1:870", "B:T2:900", "B:T2:870")
TRIGGER_EVENT_ID = "S1:target_added:T1:900"

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
        client: HTTPStructuredLLM,
        calls: list[LLMCallMetadata],
    ) -> None:
        self._runtime = runtime
        self._deps = deps
        self._holder = holder
        self._client = client
        self.calls = calls

    def active_plan(self) -> TrackingPlan | None:
        return self._deps.plans.get_active(SCENARIO_ID)

    def ask(
        self, raw_text: str, counterfactual: dict[str, object] | None = None
    ) -> Any:
        return self._runtime.ask(raw_text, counterfactual=counterfactual)

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
        return [call.operation for call in self.calls]

    def situation(self) -> SituationSnapshot:
        return self._holder.situation

    def planning_snapshot(self) -> PlanningSnapshot:
        return build_planning_snapshot(
            self._holder.situation,
            active_plan=self._deps.plans.get_active(SCENARIO_ID),
        )

    def question_evidence(self) -> QuestionEvidence:
        return retrieve_question_evidence(
            self.planning_snapshot(), self._deps.ledger, self._deps.events
        )

    def close(self) -> None:
        self._client.close()
        self._runtime.close()


def make_harness(tmp_path: Path) -> QuestionHarness:
    """One question rig: real LLM client over one SQLite database.

    The online state is seeded deterministically — bearing observations,
    the trigger event, plan revision 1 over both targets, and decision
    ``S1:decision:3`` — so the evidence-id lookup path is genuinely
    exercised without a live strategic cycle. The live situation tracks T1
    and T2 at 0.80 quality with two members each.
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
    events.append(
        event_id=TRIGGER_EVENT_ID,
        event_type="target_added",
        scenario_id=SCENARIO_ID,
        sim_time_s=SIM_TIME_S,
        target_id="T1",
        payload={},
    )
    plans.set_snapshot_revision(SCENARIO_ID, 3)
    plans.commit(
        TrackingPlan(
            plan_id="S1:plan:1",
            scenario_id=SCENARIO_ID,
            revision=1,
            base_snapshot_revision=3,
            status="active",
            member_ids_by_target={"T1": ("U1", "U2"), "T2": ("U3", "U4")},
            evidence_ids=EVIDENCE_IDS,
            trigger_event_ids=(TRIGGER_EVENT_ID,),
        )
    )
    ledger.record(
        DecisionRecord(
            decision_id="S1:decision:3",
            scenario_id=SCENARIO_ID,
            sim_time_s=SIM_TIME_S,
            snapshot_revision=3,
            trigger_event_ids=(TRIGGER_EVENT_ID,),
            input_evidence_ids=EVIDENCE_IDS,
            final_plan_id="S1:plan:1",
        )
    )
    holder = SituationHolder(build_two_target_situation(snapshot_revision=3))
    calls: list[LLMCallMetadata] = []
    client = make_live_llm(
        before_request=calls.append,
        ledger=ledger,
        scenario_id=SCENARIO_ID,
        sim_time_s=SIM_TIME_S,
    )
    deps = CarrierDependencies(
        plans=plans,
        events=events,
        ledger=ledger,
        llm=client,
        predictor=make_snapshot_predictor(
            belief_history=lambda snapshot, target_id: T1_HISTORY,
            horizon_s=600.0,
            sample_step_s=30.0,
        ),
        situation_provider=holder,
        belief_history=lambda snapshot, target_id: T1_HISTORY,
        monitor=EventMonitor(scenario_id=SCENARIO_ID),
    )
    runtime = CarrierRuntime(deps, scenario_id=SCENARIO_ID, database_path=database_path)
    return QuestionHarness(runtime, deps, holder, client, calls)


@pytest.fixture
def runtime(tmp_path: Path) -> Iterator[QuestionHarness]:
    harness = make_harness(tmp_path)
    yield harness
    harness.close()


# --- Deterministic evidence retrieval and validation (no LLM) ---------------


def test_question_evidence_retrieval_builds_bounded_payload(runtime):
    snapshot = runtime.planning_snapshot()
    dry_run = run_counterfactual_dry_run(snapshot, COUNTERFACTUAL)
    entities = match_question_entities(RAW_QUESTION, snapshot.situation)
    evidence = runtime.question_evidence()
    payload = build_question_payload(RAW_QUESTION, entities, snapshot, evidence, dry_run)
    # Deterministic entity matching: only T2 was named in the question.
    assert payload["matched_entities"]["target_ids"] == ["T2"]
    assert payload["matched_entities"]["uuv_ids"] == []
    # The citable namespace: the decision's evidence ids, the trigger
    # event id, and the plan's evidence ids.
    assert set(payload["evidence_ids"]) == {
        *EVIDENCE_IDS,
        TRIGGER_EVENT_ID,
    }
    # The ledger decision and the plan diff channel are present.
    assert len(payload["decisions"]) == 1
    assert payload["decisions"][0]["decision_id"] == "S1:decision:3"
    assert payload["active_plan"] is not None
    # Observations are resolved by evidence id from the event repository.
    observation_ids = {
        observation["event_id"] for observation in payload["observations"]
    }
    assert "B:T2:900" in observation_ids
    assert TRIGGER_EVENT_ID in observation_ids
    # The counterfactual dry-run summary is part of the payload.
    counterfactual = payload["counterfactual"]
    assert counterfactual["run_id"].startswith("dry-run:")
    assert counterfactual["plan_id"].startswith("dry-run:")


def test_question_rejects_answers_citing_absent_evidence(runtime):
    known = runtime.question_evidence().known_evidence_ids
    with pytest.raises(QuestionEvidenceError, match="B:GHOST:1"):
        validate_question_answer(
            QuestionAnswer(answer="x", evidence_ids=("B:GHOST:1",)), known
        )
    with pytest.raises(QuestionEvidenceError, match="no evidence"):
        validate_question_answer(QuestionAnswer(answer="x", evidence_ids=()), known)


def test_question_run_id_is_deterministic(runtime):
    first = question_run_id(SCENARIO_ID, RAW_QUESTION, COUNTERFACTUAL)
    second = question_run_id(SCENARIO_ID, RAW_QUESTION, COUNTERFACTUAL)
    assert first == second
    assert first.startswith(f"{SCENARIO_ID}:question:")
    assert question_run_id(SCENARIO_ID, RAW_QUESTION) != first


def test_counterfactual_dry_run_grows_unmet_quality_target(runtime):
    # The online plan still keeps T2 at two members.
    assert runtime.active_plan() is not None
    assert len(runtime.active_plan().member_ids_by_target["T2"]) == 2
    # The dry-run optimizer grows the group: measured 0.80 < required 0.85
    # -> quality risk -> the isolated plan adds members to T2.
    snapshot = runtime.planning_snapshot()
    result = run_counterfactual_dry_run(snapshot, COUNTERFACTUAL)
    assert result.diff is not None and "T2" in result.diff.members_added
    assert result.objective.active_count_after > result.objective.active_count_before
    assert result.plan_id.startswith("dry-run:")


def test_counterfactual_dry_run_is_deterministic_with_fixed_run_id(runtime):
    snapshot = runtime.planning_snapshot()
    first = run_counterfactual_dry_run(snapshot, COUNTERFACTUAL, run_id="dry-run:fixed")
    second = run_counterfactual_dry_run(snapshot, COUNTERFACTUAL, run_id="dry-run:fixed")
    assert first.plan_id == second.plan_id == "dry-run:fixed:plan:S1:2"
    assert first.plan.member_ids_by_target["T2"] == second.plan.member_ids_by_target["T2"]
    assert first.objective.active_count_after > first.objective.active_count_before


def test_counterfactual_disabled_uuv_override_is_validated(runtime):
    snapshot = runtime.planning_snapshot()
    result = run_counterfactual_dry_run(
        snapshot, {"U6.disabled": True}, run_id="dry-run:fixed"
    )
    assert result.plan_id.startswith("dry-run:")


def test_counterfactual_rejects_unknown_override_keys(runtime):
    # The override validation fails before any LLM call.
    with pytest.raises(ValueError, match="unknown counterfactual override"):
        runtime.ask(RAW_QUESTION, counterfactual={"T2.warp_factor": 9})


def test_counterfactual_rejects_unknown_entities_and_bad_values(runtime):
    with pytest.raises(ValueError, match="no group report"):
        runtime.ask(RAW_QUESTION, counterfactual={"T-NOPE.min_quality": 0.85})
    with pytest.raises(ValueError, match="outside"):
        runtime.ask(RAW_QUESTION, counterfactual={"T2.min_quality": 1.5})


# --- Live question answers (subject IS LLM behavior) ------------------------


@pytest.mark.real_llm
def test_question_and_counterfactual_do_not_change_online_plan(runtime):
    """Verbatim read-only binding test (1 request): ask + dry-run, plan untouched."""
    before = runtime.active_plan()
    assert before is not None and len(before.member_ids_by_target["T2"]) == 2
    state_before = dict(runtime.get_state())
    answer = runtime.ask(RAW_QUESTION, counterfactual=COUNTERFACTUAL)
    assert answer.evidence_ids
    assert answer.counterfactual_plan_id.startswith("dry-run:")
    assert answer.counterfactual_summary is not None
    # The online plan and the checkpointed state never changed, and only the
    # question operation ran: the carrier graph was never invoked.
    assert runtime.active_plan() == before
    assert runtime.operations() == ["question"]
    assert dict(runtime.get_state()) == state_before


@pytest.mark.real_llm
def test_question_without_counterfactual_has_no_dry_run(runtime):
    answer = runtime.ask(RAW_QUESTION)
    assert answer.evidence_ids
    assert answer.counterfactual_plan_id is None
    assert answer.counterfactual_summary is None


@pytest.mark.real_llm
def test_question_run_persisted_once_and_reask_dedupes(runtime):
    first = runtime.ask(RAW_QUESTION)
    second = runtime.ask(RAW_QUESTION)
    assert first.evidence_ids
    assert second.evidence_ids
    runs = runtime.questions()
    assert len(runs) == 1
    assert runs[0].status == "completed"
    assert runs[0].question_text == RAW_QUESTION
    # Re-asking the same question with the same overrides reuses the stored
    # run and does not queue a duplicate event.
    assert len(runtime.events(event_type="question")) == 1


@pytest.mark.real_llm
def test_question_event_surfaces_latest_question_on_next_cycle(runtime):
    runtime.ask(RAW_QUESTION, counterfactual=COUNTERFACTUAL)
    run_id = question_run_id(SCENARIO_ID, RAW_QUESTION, COUNTERFACTUAL)
    result = runtime.tick()
    assert result["route"] == "informational"
    assert runtime.get_state().get("latest_question") == run_id
    emitted = runtime.events(event_type="question")
    assert len(emitted) == 1
    assert emitted[0].payload["run_id"] == run_id
