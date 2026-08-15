# tests/agent/test_assignment_directives.py
"""Human assignment directives (spec 17.2, R4).

``assign_target_uuvs`` builds an assignment directive that reserves UUVs
for one target; validation rejects unknown ids and empty assignments and
conflicts against other applied assignments; applying the preview through
the runtime reserves the UUVs immediately, so the allocator and the
verification protocol both exclude them. No LLM is involved anywhere in
this module (the typed shortcut never parses).
"""

from pathlib import Path

import pytest  # noqa: F401 - the brief keeps the import; the module never calls it

from underwater_tracking.agent.graphs.central import CarrierDependencies
from underwater_tracking.agent.nodes.directives import (
    assign_target_uuvs,
    validate_directive,
)
from underwater_tracking.agent.runtime import CarrierRuntime
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
from underwater_tracking.simulation.clock import SimulationClock


def _uuv(uuv_id: str, x: float, y: float) -> UUVState:
    return UUVState(
        uuv_id=uuv_id,
        position_xy=(x, y),
        heading_rad=0.0,
        speed_mps=2.0,
        energy_fraction=1.0,
        status=UUVStatus.AVAILABLE,
        group_id=None,
    )


def _report(target_id: str, members: tuple[str, ...]) -> GroupReport:
    return GroupReport(
        group_id=f"G-{target_id}",
        target_id=target_id,
        sim_time_s=900,
        member_ids=members,
        belief=TargetBelief(
            target_id=target_id,
            sim_time_s=900,
            mean=(130.0, 220.0, 1.0, 0.5),
            covariance=(
                (400.0, 0.0, 0.0, 0.0),
                (0.0, 400.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            model_probabilities={"cv": 0.7, "ct": 0.3},
            source_observation_ids=(f"B:{target_id}:900",),
            fim_min_eigenvalue=0.005,
            fim_condition=12.0,
        ),
        quality=GroupQuality(
            instant=0.8,
            window_mean=0.75,
            ewma=0.76,
            components={"cov": 0.7},
            hard_guard_reasons=(),
        ),
        plan_revision=1,
    )


def _situation() -> SituationSnapshot:
    """Two tracked targets plus four healthy UUVs."""
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=900,
        uuvs=(
            _uuv("uuv_01", 100.0, 100.0),
            _uuv("uuv_02", 300.0, 300.0),
            _uuv("uuv_03", 500.0, 500.0),
            _uuv("uuv_04", 700.0, 700.0),
        ),
        group_reports=(
            _report("T1", ("uuv_01", "uuv_02")),
            _report("T2", ("uuv_03",)),
        ),
        pending_events=(),
    )


def test_assignment_shortcut_resolves_to_preview() -> None:
    directive = assign_target_uuvs(
        directive_id="S1:assign:T1:uuv_03,uuv_04",
        uuv_ids=("uuv_03", "uuv_04"),
        target_id="T1",
        situation=_situation(),
    )
    assert directive.directive_type == "assignment"
    assert directive.assignment_target_id == "T1"
    assert directive.assignment_uuv_ids == ("uuv_03", "uuv_04")
    assert directive.status == "preview"
    assert directive.conflicts == ()


def test_assignment_rejects_unknown_ids_and_empty_assignments() -> None:
    situation = _situation()
    unknown_target = assign_target_uuvs(
        directive_id="S1:assign:T9:uuv_03",
        uuv_ids=("uuv_03",),
        target_id="T9",
        situation=situation,
    )
    assert unknown_target.status == "needs_clarification"
    assert any("unknown_target" in issue for issue in unknown_target.conflicts)
    unknown_uuv = assign_target_uuvs(
        directive_id="S1:assign:T1:uuv_99",
        uuv_ids=("uuv_99",),
        target_id="T1",
        situation=situation,
    )
    assert unknown_uuv.status == "needs_clarification"
    assert any("unknown_uuv" in issue for issue in unknown_uuv.conflicts)
    empty = assign_target_uuvs(
        directive_id="S1:assign:T1:",
        uuv_ids=(),
        target_id="T1",
        situation=situation,
    )
    assert empty.status == "needs_clarification"
    assert any("empty_assignment" in issue for issue in empty.conflicts)


def test_assignment_conflicts_with_an_applied_assignment() -> None:
    situation = _situation()
    applied = validate_directive(
        assign_target_uuvs(
            directive_id="S1:assign:T1:uuv_03",
            uuv_ids=("uuv_03",),
            target_id="T1",
            situation=situation,
        ),
        situation=situation,
    )
    assert applied.status == "preview"
    conflicting = assign_target_uuvs(
        directive_id="S1:assign:T2:uuv_03",
        uuv_ids=("uuv_03",),
        target_id="T2",
        situation=situation,
        applied_directives=(applied,),
    )
    assert conflicting.status == "needs_clarification"
    assert any("uuv_03" in issue for issue in conflicting.conflicts)
    # Re-assigning the same target is idempotent, not a conflict.
    same_target = assign_target_uuvs(
        directive_id="S1:assign:T1:uuv_03,uuv_04",
        uuv_ids=("uuv_03", "uuv_04"),
        target_id="T1",
        situation=situation,
        applied_directives=(applied,),
    )
    assert same_target.status == "preview"


class _NeverLLM:
    """Stands in for the structured LLM port: the assignment flow never calls it."""

    def invoke_structured(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("the assignment flow must not call the LLM")


def _make_runtime(tmp_path: Path, situation: SituationSnapshot) -> CarrierRuntime:
    database_path = tmp_path / "assign.db"
    dependencies = CarrierDependencies(
        plans=PlanRepository(database_path),
        events=EventRepository(database_path),
        ledger=DecisionLedger(database_path),
        llm=_NeverLLM(),  # type: ignore[arg-type]
        predictor=lambda situation, target_id: None,  # type: ignore[arg-type]
        situation_provider=lambda ref: situation,
        clock=SimulationClock(step_s=30),
    )
    return CarrierRuntime(
        dependencies, scenario_id="S1", database_path=database_path
    )


def test_runtime_apply_assignment_reserves_the_uuvs(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, _situation())
    try:
        preview = runtime.preview_assignment(
            uuv_ids=("uuv_03", "uuv_04"), target_id="T1"
        )
        assert preview.status == "preview"
        applied = runtime.apply_directive(preview.directive_id)
        assert applied.status == "applied"
        assert runtime.reservations().reserved_uuvs() == frozenset(
            {"uuv_03", "uuv_04"}
        )
        assert runtime.reservations().reserved_for("T1") == frozenset(
            {"uuv_03", "uuv_04"}
        )
    finally:
        runtime.close()
