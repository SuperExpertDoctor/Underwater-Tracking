# tests/agent/test_directives.py
"""Non-blocking expert directive tests (spec 10.1, plan Task 10).

Covers the brief's binding non-blocking ambiguity test (an annotation
requests clarification without stopping the running plan), the
preview/apply lifecycle (parse -> validate -> preview -> apply -> strategic
event), rejection of low-confidence/conflicting previews, the structured
shortcut helpers, and the directive branch surfacing the latest applied
directive on the checkpointed state.

Per the user directive (addendum A) no mock substitutes real LLM
functionality: the only LLM behavior here — parsing raw text into an
``ExpertDirective`` and the strategic re-planning cycle after an apply —
runs live against the real LongCat provider. Everything else is
deterministic node logic driven explicitly: the shortcut helpers and
``validate_directive`` resolve ambiguity, conflicts, and low confidence
without any LLM, and ``apply_directive`` re-validates and queues the
strategic event purely. The former mock parse-queue (raw-text -> template
mapping) was deleted as an accepted consequence. The whole module is
skipped when the API key is unset.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from underwater_tracking.agent.graphs.central import CarrierDependencies
from underwater_tracking.agent.llm import HTTPStructuredLLM, LLMCallMetadata
from underwater_tracking.agent.nodes.directives import (
    DirectiveNotApplicableError,
    disable_uuv,
    lock_group_members,
    submit_expert_feedback,
    set_minimum_quality,
    set_target_priority,
    validate_directive,
)
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.domain.agent_models import ExpertDirective, TrackingPlan
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


def build_situation(
    *, snapshot_revision: int, sim_time_s: int = SIM_TIME_S
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
                source_observation_ids=("B:T1:900", "B:T1:870"),
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


class DirectiveHarness:
    """CarrierRuntime wrapper exposing the binding-test helpers."""

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

    def preview_directive(self, raw_text: str) -> ExpertDirective:
        return self._runtime.preview_directive(raw_text)

    def apply_directive(self, directive_id: str) -> ExpertDirective:
        return self._runtime.apply_directive(directive_id)

    def save_directive(self, directive: ExpertDirective) -> None:
        self._deps.ledger.save_directive(directive, SCENARIO_ID)

    def tick(self) -> dict[str, Any]:
        return self._runtime.tick()

    def get_state(self) -> dict[str, Any]:
        return self._runtime.get_state()

    def directives(self, *, status: str | None = None) -> list[ExpertDirective]:
        return self._deps.ledger.list_directives(SCENARIO_ID, status=status)

    def events(self, *, event_type: str) -> list[Any]:
        return self._deps.events.list_events(
            scenario_id=SCENARIO_ID, event_type=event_type
        )

    def operations(self) -> list[str]:
        return [call.operation for call in self.calls]

    def situation(self) -> SituationSnapshot:
        return self._holder.situation

    def close(self) -> None:
        self._client.close()
        self._runtime.close()


def make_harness(tmp_path: Path) -> DirectiveHarness:
    """One directive rig: real LLM client over one SQLite database."""
    database_path = tmp_path / "directives.db"
    plans = PlanRepository(database_path)
    events = EventRepository(database_path)
    ledger = DecisionLedger(database_path)
    holder = SituationHolder(build_situation(snapshot_revision=3))
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
    runtime = CarrierRuntime(
        deps, scenario_id=SCENARIO_ID, database_path=database_path
    )
    return DirectiveHarness(runtime, deps, holder, client, calls)


@pytest.fixture
def runtime(tmp_path: Path) -> Iterator[DirectiveHarness]:
    harness = make_harness(tmp_path)
    yield harness
    harness.close()


# --- Structured shortcuts and deterministic validation (no LLM) -------------


def test_structured_shortcuts_create_validated_directives(runtime):
    situation = runtime.situation()
    locked = lock_group_members(
        directive_id="D-LOCK",
        raw_text="lock T1's members",
        target_scope=("T1",),
        target_id="T1",
        member_ids=("U1", "U2"),
        confidence=0.9,
        situation=situation,
    )
    assert isinstance(locked, ExpertDirective)
    assert locked.status == "preview"
    assert locked.locked_members == {"T1": ("U1", "U2")}

    priority = set_target_priority(
        directive_id="D-PRIO",
        raw_text="prioritize T1",
        target_scope=("T1",),
        target_id="T1",
        priority=1.0,
        confidence=0.9,
        situation=situation,
    )
    assert priority.target_priorities == {"T1": 1.0}
    assert priority.status == "preview"

    quality = set_minimum_quality(
        directive_id="D-QUAL",
        raw_text="keep T1's quality at 0.8",
        target_scope=("T1",),
        target_id="T1",
        quality=0.8,
        confidence=0.9,
        situation=situation,
    )
    assert quality.minimum_quality == {"T1": 0.8}
    assert quality.status == "preview"

    disabled = disable_uuv(
        directive_id="D-DISABLE",
        raw_text="disable U3",
        target_scope=("T1",),
        uuv_id="U3",
        confidence=0.9,
        situation=situation,
    )
    assert disabled.disabled_uuv_ids == ("U3",)
    assert disabled.status == "preview"

    # The same validator flags unknown ids instead of auto-applying.
    unknown = lock_group_members(
        directive_id="D-BAD",
        raw_text="lock an unknown group",
        target_scope=("T1",),
        target_id="T-NOPE",
        member_ids=("U1",),
        confidence=0.9,
        situation=situation,
    )
    assert unknown.status == "needs_clarification"
    assert unknown.conflicts


def test_ambiguous_directive_requests_clarification_without_any_llm():
    # The deterministic side of the binding non-blocking test: a directive
    # naming no target or resource constraint resolves to clarification.
    ambiguous = ExpertDirective(
        directive_id="D-AMBIG",
        raw_text="多派一些艇过去",
        target_scope=(),
        confidence=0.9,
        status="preview",
    )
    resolved = validate_directive(
        ambiguous, situation=build_situation(snapshot_revision=3)
    )
    assert resolved.status == "needs_clarification"
    assert any("ambiguous_scope" in conflict for conflict in resolved.conflicts)


def test_expert_feedback_is_scoped_to_regions_without_assigning_members():
    feedback = submit_expert_feedback(
        directive_id="D-FEEDBACK",
        raw_text="region_2 交接延迟，请增加下一窗口的接力余量",
        target_id="T1",
        region_ids=("region_2",),
        feedback="region_2 交接延迟，请增加下一窗口的接力余量",
        confidence=0.95,
        situation=build_situation(snapshot_revision=3),
    )

    assert feedback.directive_type == "feedback"
    assert feedback.feedback_region_ids == ("region_2",)
    assert feedback.feedback_text == "region_2 交接延迟，请增加下一窗口的接力余量"
    assert feedback.assignment_uuv_ids == ()


def test_conflicting_directive_requests_clarification():
    situation = build_situation(snapshot_revision=3)
    first = lock_group_members(
        directive_id="D-LOCK",
        raw_text="lock T1's members",
        target_scope=("T1",),
        target_id="T1",
        member_ids=("U1", "U2"),
        confidence=0.92,
        situation=situation,
    )
    assert first.status == "preview"
    second = lock_group_members(
        directive_id="D-ALT",
        raw_text="replace T1's members",
        target_scope=("T1",),
        target_id="T1",
        member_ids=("U3", "U4"),
        confidence=0.92,
        situation=situation,
        applied_directives=(first,),
    )
    assert second.status == "needs_clarification"
    assert any("locked members" in conflict for conflict in second.conflicts)


def test_low_confidence_preview_is_rejected_by_apply(tmp_path: Path):
    harness = make_harness(tmp_path)
    try:
        low = lock_group_members(
            directive_id="D-LOW",
            raw_text="low-confidence lock",
            target_scope=("T1",),
            target_id="T1",
            member_ids=("U1", "U2"),
            confidence=0.5,
            situation=harness.situation(),
        )
        assert low.status == "needs_clarification"
        harness.save_directive(low)
        with pytest.raises(DirectiveNotApplicableError, match="confidence"):
            harness.apply_directive(low.directive_id)
        # The preview stays in the ledger unchanged; nothing was applied.
        assert harness.directives()[0].status == "needs_clarification"
        assert harness.active_plan() is None
        assert harness.events(event_type="directive_applied") == []
    finally:
        harness.close()


def test_apply_unknown_directive_raises(tmp_path: Path):
    harness = make_harness(tmp_path)
    try:
        with pytest.raises(ValueError, match="unknown directive"):
            harness.apply_directive("S1:directive:does-not-exist")
    finally:
        harness.close()


# --- Live preview / apply lifecycle (subject IS LLM behavior) ---------------


@pytest.mark.real_llm
def test_preview_directive_parses_live_and_never_touches_graph_state(runtime):
    """Live parse of one annotation (1 request), fully non-blocking.

    Whatever the parse resolves to, the preview is validated and persisted
    with its resolved status, and the graph is never invoked — the running
    plan keeps executing while the expert reviews the preview.
    """
    state_before = dict(runtime.get_state())
    preview = runtime.preview_directive("多派一些艇过去")
    assert preview.status in ("preview", "needs_clarification")
    assert isinstance(preview.conflicts, tuple)
    assert preview.directive_id.startswith(f"{SCENARIO_ID}:directive:")
    # The graph was never invoked: no plan and no state channels changed.
    assert runtime.active_plan() is None
    assert dict(runtime.get_state()) == state_before
    stored = runtime.directives()
    assert len(stored) == 1
    assert stored[0].directive_id == preview.directive_id
    assert stored[0].status == preview.status
    assert runtime.operations() == ["directive"]


@pytest.mark.real_llm
def test_apply_clean_preview_queues_strategic_event_and_surfaces_branch(runtime):
    """Applying a clean (shortcut-built, deterministic) preview re-plans live.

    The preview itself is built by the typed shortcut — the deterministic
    apply path re-validates it, queues the strategic ``directive_applied``
    event, and the next cycle re-plans with the real client (intent +
    strategy + verification requests); the directive branch surfaces the
    latest applied directive on the checkpointed state.
    """
    clean = lock_group_members(
        directive_id="D-CLEAN",
        raw_text="lock T1's members",
        target_scope=("T1",),
        target_id="T1",
        member_ids=("U1", "U2"),
        confidence=0.92,
        situation=runtime.situation(),
    )
    assert clean.status == "preview"
    runtime.save_directive(clean)
    applied = runtime.apply_directive(clean.directive_id)
    assert applied.status == "applied"
    assert applied.directive_id == clean.directive_id
    assert [directive.status for directive in runtime.directives()] == ["applied"]
    # The event is queued for the next cycle: the existing plan (none yet)
    # stays active until that cycle commits.
    assert runtime.active_plan() is None
    result = runtime.tick()
    assert result["route"] == "strategic"
    assert result["commit_status"] == "committed"
    active = runtime.active_plan()
    assert active is not None and active.revision == 1
    emitted = runtime.events(event_type="directive_applied")
    assert len(emitted) == 1
    assert emitted[0].payload["directive_id"] == clean.directive_id
    latest = runtime.get_state().get("latest_directive")
    assert latest is not None
    assert latest.directive_id == applied.directive_id
    assert latest.status == "applied"
    operations = runtime.operations()
    assert operations[0] == "intent"
    assert set(operations) <= {"intent", "strategy"}


def test_directive_state_survives_runtime_reopen(tmp_path: Path):
    harness = make_harness(tmp_path)
    clean = lock_group_members(
        directive_id="D-CLEAN",
        raw_text="lock T1's members",
        target_scope=("T1",),
        target_id="T1",
        member_ids=("U1", "U2"),
        confidence=0.92,
        situation=harness.situation(),
    )
    harness.save_directive(clean)
    harness.apply_directive(clean.directive_id)
    harness.close()

    reopened = make_harness(tmp_path)
    try:
        stored = reopened.directives()
        assert [directive.status for directive in stored] == ["applied"]
        assert stored[0].directive_id == clean.directive_id
        assert reopened.active_plan() is None
    finally:
        reopened.close()
