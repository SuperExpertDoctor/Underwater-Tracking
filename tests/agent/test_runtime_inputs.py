from __future__ import annotations

from threading import RLock
from types import SimpleNamespace

from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.domain.models import IntelligenceReport, OperationalScheme


def test_runtime_holds_operational_inputs_until_the_simulation_boundary() -> None:
    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._dependencies = SimpleNamespace(
        clock=SimpleNamespace(sim_time_s=30)
    )
    runtime._pending_scheme = None
    runtime._pending_intelligence = {}
    runtime._lock = RLock()

    scheme = OperationalScheme(
        scheme_id="scheme-2",
        version=2,
        minimum_quality={"T1": 0.8},
        valid_from_s=30,
        valid_until_s=900,
    )
    report = IntelligenceReport(
        report_id="intel-2",
        source="technical_reconnaissance",
        target_id="T1",
        confidence=0.9,
        issued_at_s=30,
        valid_until_s=300,
    )

    runtime.set_operational_scheme(scheme)
    runtime.submit_intelligence(report)

    assert runtime.drain_operational_inputs() == (scheme, (report,))
    assert runtime.drain_operational_inputs() == (None, ())
