from __future__ import annotations

from threading import RLock
from types import SimpleNamespace

import pytest

from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.domain.models import IntelligenceReport, OperationalScheme


def _runtime(*, sim_time_s: int) -> CarrierRuntime:
    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._dependencies = SimpleNamespace(
        clock=SimpleNamespace(sim_time_s=sim_time_s)
    )
    runtime._pending_scheme = None
    runtime._pending_intelligence = {}
    runtime._lock = RLock()
    return runtime


def _scheme(*, valid_from_s: int = 30, valid_until_s: int = 900) -> OperationalScheme:
    return OperationalScheme(
        scheme_id="scheme-2",
        version=2,
        minimum_quality={"T1": 0.8},
        valid_from_s=valid_from_s,
        valid_until_s=valid_until_s,
    )


def _report(*, issued_at_s: int = 30, valid_until_s: int = 300) -> IntelligenceReport:
    return IntelligenceReport(
        report_id="intel-2",
        source="technical_reconnaissance",
        target_id="T1",
        confidence=0.9,
        issued_at_s=issued_at_s,
        valid_until_s=valid_until_s,
    )


def test_runtime_holds_operational_inputs_until_the_engine_commit_succeeds() -> None:
    runtime = _runtime(sim_time_s=30)
    scheme = _scheme()
    report = _report()

    runtime.set_operational_scheme(scheme)
    runtime.submit_intelligence(report)

    applied_schemes: list[OperationalScheme] = []
    applied_reports: list[IntelligenceReport] = []

    runtime.commit_operational_inputs(
        current_sim_time_s=30,
        apply_scheme=applied_schemes.append,
        apply_intelligence=applied_reports.append,
    )

    assert runtime.drain_operational_inputs() == (None, ())
    assert applied_schemes == [scheme]
    assert applied_reports == [report]


def test_runtime_keeps_operational_inputs_when_engine_submission_fails() -> None:
    runtime = _runtime(sim_time_s=30)
    report = _report()
    runtime.submit_intelligence(report)
    attempts = 0

    def submit(report_to_apply: IntelligenceReport) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("engine rejected the boundary input")
        del report_to_apply

    with pytest.raises(ValueError, match="boundary input"):
        runtime.commit_operational_inputs(
            current_sim_time_s=30,
            apply_scheme=lambda _: None,
            apply_intelligence=submit,
        )

    assert runtime.drain_operational_inputs() == (None, (report,))
    runtime.commit_operational_inputs(
        current_sim_time_s=30,
        apply_scheme=lambda _: None,
        apply_intelligence=submit,
    )
    assert runtime.drain_operational_inputs() == (None, ())


def test_runtime_rechecks_expiry_at_the_engine_boundary_time() -> None:
    runtime = _runtime(sim_time_s=0)
    report = _report(issued_at_s=0, valid_until_s=30)
    runtime.submit_intelligence(report)

    with pytest.raises(ValueError, match="expired"):
        runtime.commit_operational_inputs(
            current_sim_time_s=30,
            apply_scheme=lambda _: None,
            apply_intelligence=lambda _: None,
        )

    assert runtime.drain_operational_inputs() == (None, (report,))


def test_runtime_input_validation_uses_the_bound_engine_clock() -> None:
    runtime = _runtime(sim_time_s=0)
    runtime.bind_simulation_time(lambda: 30)

    with pytest.raises(ValueError, match="expired"):
        runtime.set_operational_scheme(_scheme(valid_from_s=0, valid_until_s=30))
    with pytest.raises(ValueError, match="expired"):
        runtime.submit_intelligence(_report(issued_at_s=0, valid_until_s=30))
