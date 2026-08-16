"""Adaptive tracking inputs exposed by the deterministic engine."""

from __future__ import annotations

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.models import (
    IntelligenceReport,
    IntelligenceSource,
    OperationalScheme,
    SurveillanceCapability,
)
from underwater_tracking.simulation.engine import SimulationEngine
from tests.conftest import CONFIG_PATH


def test_engine_publishes_active_inputs_and_per_uuv_capability(tmp_path) -> None:
    """Queued operational inputs appear only in the next carrier snapshot."""
    base = load_app_config(CONFIG_PATH)
    capability = SurveillanceCapability(
        passive_range_m=5_000.0,
        active_range_m=2_000.0,
        bearing_variance_rad2=0.02,
        active_sonar_available=True,
        max_speed_mps=3.0,
        max_turn_rate_rad_s=0.04,
    )
    config = base.model_copy(
        update={
            "tracking": base.tracking.model_copy(
                update={"uuv_capabilities": {"uuv_00": capability}}
            )
        }
    )
    snapshots = []
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path, carrier=snapshots.append)
    scheme = OperationalScheme(
        scheme_id="night-watch",
        version=2,
        target_priorities={"target_00": 1.0},
        minimum_quality={"target_00": 0.8},
        valid_from_s=0,
        valid_until_s=120,
    )
    current = IntelligenceReport(
        report_id="intel-current",
        source=IntelligenceSource.SONAR,
        target_id="target_00",
        confidence=0.8,
        issued_at_s=0,
        valid_until_s=60,
        assessment={"intent": "evade"},
    )
    future = IntelligenceReport(
        report_id="intel-future",
        source=IntelligenceSource.SIGINT,
        target_id="target_00",
        confidence=0.7,
        issued_at_s=60,
        valid_until_s=120,
        assessment={"activity": "intermittent"},
    )

    engine.set_operational_scheme(scheme)
    engine.submit_intelligence(current)
    engine.submit_intelligence(future)
    for _ in range(3):
        engine.step()

    snapshot = snapshots[-1]
    assert snapshot.operational_scheme == scheme
    assert snapshot.intelligence_reports == (current,)
    uuv = next(state for state in snapshot.uuvs if state.uuv_id == "uuv_00")
    assert uuv.capability == capability
    target = next(contact for contact in snapshot.contacts if contact.contact_id == "target_00")
    observation = next(ray for ray in target.bearing_rays if ray.uuv_id == "uuv_00")
    assert observation.variance_rad2 == capability.bearing_variance_rad2
