# tests/agent/test_directives.py
"""Non-blocking expert directive tests (spec 10.1, plan Task 10).

Covers the brief's binding non-blocking ambiguity test (an ambiguous
annotation requests clarification without stopping the running plan), the
preview/apply lifecycle (parse -> validate -> preview -> apply -> strategic
event), rejection of low-confidence/conflicting previews, the structured
shortcut helpers, and the directive branch surfacing the latest applied
directive on the checkpointed state.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from underwater_tracking.agent.graphs.central import CarrierDependencies
from underwater_tracking.agent.llm import MockStructuredLLM
from underwater_tracking.agent.nodes.directives import (
    DirectiveNotApplicableError,
    disable_uuv,
    lock_group_members,
    set_minimum_quality,
    set_target_priority,
)
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    PredictedTrackRef,
    TrackingPlan,
)
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
from tests.fixtures.llm_responses import (
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

# Deterministic directive parse fixtures keyed by exact raw text.
AMBIGUOUS_TEXT = "多派一些艇过去"
CLEAN_TEXT = "锁定 T1 的成员为 U1 和 U2"
ALTERNATE_TEXT = "把 T1 的成员换成 U3 和 U4"

AMBIGUOUS_DIRECTIVE_TEMPLATE = {
    "directive_id": "",
    "raw_text": "",
    "target_scope": [],
    "confidence": 0.45,
    "conflicts": ["ambiguous instruction: no target or resource constraint named"],
    "status": "preview",
}

CLEAN_DIRECTIVE_TEMPLATE = {
    "directive_id": "",
    "raw_text": "",
    "target_scope": ["T1"],
    "locked_members": {"T1": ["U1", "U2"]},
    "confidence": 0.92,
    "status": "preview",
}

ALTERNATE_LOCK_TEMPLATE = {
    "directive_id": "",
    "raw_text": "",
    "target_scope": ["T1"],
    "locked_members": {"T1": ["U3", "U4"]},
    "confidence": 0.92,
    "status": "preview",
}

_DIRECTIVE_TEMPLATES: dict[str, dict[str, object]] = {
    AMBIGUOUS_TEXT: AMBIGUOUS_DIRECTIVE_TEMPLATE,
    CLEAN_TEXT: CLEAN_DIRECTIVE_TEMPLATE,
    ALTERNATE_TEXT: ALTERNATE_LOCK_TEMPLATE,
}


class DirectiveLLM(MockStructuredLLM):
    """Mock LLM with a deterministic raw-text -> directive parse mapping.

    The ``directive`` operation is served from the fixed templates keyed by
    the payload's raw text (the ambiguous template for the binding text,
    clean lock-member templates otherwise); all other operations are served
    from the FIFO queues. First-call operation order is recorded so tests
    can assert the directive parse ran before the strategic chain.
    """

    def __init__(self, responses: dict[str, object]) -> None:
        super().__init__(responses)
        self.operations: list[str] = []
        self._seen: set[str] = set()

    def invoke_structured(self, operation, payload, response_model, *, prompt_version=""):
        if operation not in self._seen:
            self._seen.add(operation)
            self.operations.append(operation)
        if operation != "directive":
            return super().invoke_structured(
                operation, payload, response_model, prompt_version=prompt_version
            )
        raw_text = str(payload["raw_text"])
        template = _DIRECTIVE_TEMPLATES.get(raw_text, CLEAN_DIRECTIVE_TEMPLATE)
        response = {
            **template,
            "directive_id": payload["directive_id"],
            "raw_text": raw_text,
        }
        return response_model.model_validate(response)


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

    def __init__(self, runtime: CarrierRuntime, deps: CarrierDependencies, holder: SituationHolder) -> None:
        self._runtime = runtime
        self._deps = deps
        self._holder = holder

    def active_plan(self) -> TrackingPlan | None:
        return self._deps.plans.get_active(SCENARIO_ID)

    def preview_directive(self, raw_text: str) -> ExpertDirective:
        return self._runtime.preview_directive(raw_text)

    def apply_directive(self, directive_id: str) -> ExpertDirective:
        return self._runtime.apply_directive(directive_id)

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
        return self._deps.llm.operations

    def situation(self) -> SituationSnapshot:
        return self._holder.situation

    def group_updates_advanced(self) -> bool:
        """True when a completed carrier cycle reported the group updates."""
        messages = self.get_state().get("output_messages") or ()
        return any(
            str(message).startswith(f"{SCENARIO_ID} cycle:") for message in messages
        )

    def close(self) -> None:
        self._runtime.close()


def make_harness(tmp_path: Path) -> DirectiveHarness:
    """One directive rig: injected dependencies over one SQLite database."""
    database_path = tmp_path / "directives.db"
    plans = PlanRepository(database_path)
    events = EventRepository(database_path)
    ledger = DecisionLedger(database_path)
    holder = SituationHolder(build_situation(snapshot_revision=3))
    deps = CarrierDependencies(
        plans=plans,
        events=events,
        ledger=ledger,
        llm=DirectiveLLM(_default_responses()),
        predictor=_straight_line_predictor,
        situation_provider=holder,
        belief_history=lambda snapshot, target_id: T1_HISTORY,
        monitor=EventMonitor(scenario_id=SCENARIO_ID),
    )
    runtime = CarrierRuntime(
        deps, scenario_id=SCENARIO_ID, database_path=database_path
    )
    return DirectiveHarness(runtime, deps, holder)


def _default_responses() -> dict[str, object]:
    return {
        "intent": [VALID_INTENT_HYPOTHESIS],
        "strategy": [
            QUALITY_FIRST_PROPOSAL,
            VALID_STRATEGY_PROPOSAL,
            RESOURCE_SAVING_PROPOSAL,
        ],
    }


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
def runtime(tmp_path: Path) -> Iterator[DirectiveHarness]:
    harness = make_harness(tmp_path)
    yield harness
    harness.close()


# --- Brief Step 1: verbatim non-blocking ambiguity test --------------------


def test_ambiguous_directive_requests_clarification_without_stopping_plan(runtime):
    before = runtime.active_plan()
    preview = runtime.preview_directive("多派一些艇过去")
    assert preview.status == "needs_clarification"
    assert runtime.active_plan() == before
    runtime.tick()
    assert runtime.group_updates_advanced()


# --- Preview / apply lifecycle (brief Step 2) ------------------------------


def test_preview_persists_validated_directive_and_never_touches_graph_state(runtime):
    state_before = dict(runtime.get_state())
    preview = runtime.preview_directive(CLEAN_TEXT)
    assert preview.status == "preview"
    assert preview.conflicts == ()
    # The graph was never invoked: no plan and no state channels changed.
    assert runtime.active_plan() is None
    assert dict(runtime.get_state()) == state_before
    stored = runtime.directives()
    assert len(stored) == 1
    assert stored[0].directive_id == preview.directive_id
    assert stored[0].status == "preview"
    assert runtime.operations() == ["directive"]


def test_apply_marks_clean_preview_applied_and_emits_strategic_event(runtime):
    preview = runtime.preview_directive(CLEAN_TEXT)
    applied = runtime.apply_directive(preview.directive_id)
    assert applied.status == "applied"
    assert applied.directive_id == preview.directive_id
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
    assert emitted[0].payload["directive_id"] == preview.directive_id
    assert runtime.operations() == ["directive", "intent", "strategy"]


def test_apply_rejects_low_confidence_or_conflicting_previews(runtime):
    preview = runtime.preview_directive(AMBIGUOUS_TEXT)
    assert preview.status == "needs_clarification"
    with pytest.raises(DirectiveNotApplicableError, match="confidence"):
        runtime.apply_directive(preview.directive_id)
    # The preview stays in the ledger unchanged; nothing was applied.
    assert runtime.directives()[0].status == "needs_clarification"
    assert runtime.active_plan() is None
    assert runtime.events(event_type="directive_applied") == []


def test_apply_unknown_directive_raises(runtime):
    with pytest.raises(ValueError, match="unknown directive"):
        runtime.apply_directive("S1:directive:does-not-exist")


def test_conflicting_directive_requests_clarification(runtime):
    first = runtime.preview_directive(CLEAN_TEXT)
    assert first.status == "preview"
    runtime.apply_directive(first.directive_id)
    second = runtime.preview_directive(ALTERNATE_TEXT)
    assert second.status == "needs_clarification"
    assert any("locked members" in conflict for conflict in second.conflicts)
    assert runtime.active_plan() is None


# --- Structured shortcuts (brief Step 3) -----------------------------------


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


# --- Directive branch on the checkpointed state ----------------------------


def test_directive_branch_surfaces_latest_applied_directive(runtime):
    preview = runtime.preview_directive(CLEAN_TEXT)
    applied = runtime.apply_directive(preview.directive_id)
    result = runtime.tick()
    assert result["route"] == "strategic"
    assert result["commit_status"] == "committed"
    latest = runtime.get_state().get("latest_directive")
    assert latest is not None
    assert latest.directive_id == applied.directive_id
    assert latest.status == "applied"


def test_directive_state_survives_runtime_reopen(tmp_path: Path):
    harness = make_harness(tmp_path)
    preview = harness.preview_directive(CLEAN_TEXT)
    harness.apply_directive(preview.directive_id)
    harness.close()

    reopened = make_harness(tmp_path)
    try:
        stored = reopened.directives()
        assert [directive.status for directive in stored] == ["applied"]
        assert stored[0].directive_id == preview.directive_id
        assert reopened.active_plan() is None
    finally:
        reopened.close()
