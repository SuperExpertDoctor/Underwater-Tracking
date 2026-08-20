# tests/api/test_frame_pipeline.py
"""Runtime frame adapter, JSONL logger, and indexed replay (UI plan Task 3).

The brief's Step-1 append-and-replay test is kept verbatim. The remaining
tests pin the behavioral requirements of Steps 3-4: deterministic entity
ordering, covariance-to-ellipse conversion, map-bounds clipping, committed
plan-version attachment, per-line validated JSONL writes with immediate
flush, and indexed time-range replay with typed corrupt-line rejection.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from underwater_tracking.cli import _AgentLoop
from underwater_tracking.api.frame_builder import build_operational_frame
from underwater_tracking.api.frame_logger import FrameLogger
from underwater_tracking.api.replay import ReplayIndexError, ReplayService
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain import (
    MetricView,
    OperationalFrame,
)
from underwater_tracking.domain.agent_models import (
    DecisionRecord,
    PlanDiff,
    PlanStatus,
    TrackingPlan,
    ValidationReport,
    Waypoint,
)
from underwater_tracking.domain.models import (
    BearingObservation,
    CarrierState,
    Contact,
    DeploymentState,
    GroupQuality,
    GroupReport,
    IntelligenceReport,
    OperationalScheme,
    RuntimeEvent,
    SituationSnapshot,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    CommunicationCapability,
    CommunicationLink,
    MotionLimits,
    PlatformCapability,
    PlatformKind,
    PlatformRoster,
    PlatformSnapshot,
    SonarCapability,
    USVPlatformState,
    UUVPlatformState,
)
from underwater_tracking.simulation.engine import SimulationEngine

from tests.api.test_frame_contracts import _full_frame
from tests.conftest import CONFIG_PATH, make_live_llm


@pytest.fixture
def frame_factory() -> Callable[..., OperationalFrame]:
    """A factory of valid, rich operational frames (adapted from the contract
    helpers in tests/api/test_frame_contracts.py; revalidated on override so
    the float sim time coerces to the contract's int field)."""

    def factory(*, frame_id: int, sim_time_s: float) -> OperationalFrame:
        payload = _full_frame().model_dump()
        payload["frame_id"] = frame_id
        payload["sim_time_s"] = sim_time_s
        return OperationalFrame.model_validate(payload)

    return factory


# --- domain builders --------------------------------------------------------


def _uuv(uuv_id: str, x: float, y: float) -> UUVState:
    return UUVState(
        uuv_id=uuv_id,
        position_xy=(x, y),
        heading_rad=0.0,
        speed_mps=2.0,
        energy_fraction=0.8,
        status=UUVStatus.TRACKING,
        group_id="G1",
    )


def _belief(
    target_id: str,
    mean: tuple[float, float],
    covariance: tuple[tuple[float, ...], ...],
    *,
    fim_condition: float = float("inf"),
) -> TargetBelief:
    return TargetBelief(
        target_id=target_id,
        sim_time_s=100,
        mean=mean,
        covariance=covariance,
        model_probabilities={"cv": 1.0},
        fim_condition=fim_condition,
    )


def _report(
    target_id: str,
    group_id: str,
    mean: tuple[float, float],
    covariance: tuple[tuple[float, ...], ...],
    *,
    fim_condition: float = float("inf"),
) -> GroupReport:
    return GroupReport(
        group_id=group_id,
        target_id=target_id,
        sim_time_s=100,
        member_ids=("UUV-2", "UUV-1"),
        belief=_belief(target_id, mean, covariance, fim_condition=fim_condition),
        quality=GroupQuality(
            instant=0.9, window_mean=0.88, ewma=0.89, components={"fim": 0.95}
        ),
        plan_revision=4,
    )


def _observation(observation_id: str, uuv_id: str, target_id: str) -> BearingObservation:
    return BearingObservation(
        observation_id=observation_id,
        scenario_id="scenario-20260814",
        sim_time_s=100,
        uuv_id=uuv_id,
        target_id=target_id,
        azimuth_rad=0.5,
        variance_rad2=0.02,
        detection_confidence=0.9,
    )


def _contact(contact_id: str, observations: tuple[BearingObservation, ...]) -> Contact:
    return Contact(contact_id=contact_id, sim_time_s=100, bearing_rays=observations)


def _event(event_id: str) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=event_id,
        scenario_id="scenario-20260814",
        sim_time_s=100,
        event_type="plan_commit",
        level="tactical",
        entity_id="plan-7",
    )


def _decision(
    decision_id: str,
    *,
    final_plan_id: str | None = "plan-7",
    diff: PlanDiff | None = None,
    verification: tuple[ValidationReport, ...] = (),
) -> DecisionRecord:
    return DecisionRecord(
        decision_id=decision_id,
        scenario_id="scenario-20260814",
        sim_time_s=100,
        trigger_event_ids=("evt-1",),
        input_evidence_ids=("obs-1",),
        final_plan_id=final_plan_id,
        final_plan_diff=diff,
        verification_records=verification,
    )


def _plan(
    *,
    revision: int = 4,
    status: PlanStatus = "active",
    valid_until_s: int = 240,
    waypoints: dict[str, tuple[Waypoint, ...]] | None = None,
    intent_refs: dict[str, str] | None = None,
    diff: PlanDiff | None = None,
) -> TrackingPlan:
    return TrackingPlan(
        plan_id="plan-7",
        scenario_id="scenario-20260814",
        revision=revision,
        base_snapshot_revision=3,
        status=status,
        valid_from_s=60,
        valid_until_s=valid_until_s,
        concept="balanced",
        target_priorities={"T1": 0.9},
        member_ids_by_target={"T1": ("UUV-1", "UUV-2")},
        waypoints_by_member=waypoints or {},
        intent_refs=intent_refs or {},
        diff=diff,
    )


def _snapshot(
    *,
    sim_time_s: int = 100,
    revision: int = 5,
    uuvs: Sequence[UUVState] = (),
    carrier: CarrierState | None = None,
    reports: Sequence[GroupReport] = (),
    contacts: Sequence[Contact] = (),
    events: Sequence[RuntimeEvent] = (),
    operational_scheme: OperationalScheme | None = None,
    intelligence_reports: Sequence[IntelligenceReport] = (),
) -> SituationSnapshot:
    return SituationSnapshot(
        scenario_id="scenario-20260814",
        snapshot_revision=revision,
        sim_time_s=sim_time_s,
        uuvs=tuple(uuvs),
        carrier=carrier,
        group_reports=tuple(reports),
        pending_events=tuple(events),
        contacts=tuple(contacts),
        operational_scheme=operational_scheme,
        intelligence_reports=tuple(intelligence_reports),
    )


# --- Step 1: append-and-replay (verbatim from the brief) ---------------------


def test_logged_operational_frames_round_trip_in_order(tmp_path, frame_factory):
    from underwater_tracking.api.frame_logger import FrameLogger
    from underwater_tracking.api.replay import ReplayService

    path = tmp_path / "frames.jsonl"
    logger = FrameLogger(path)
    logger.append(frame_factory(frame_id=2, sim_time_s=20.0))
    logger.append(frame_factory(frame_id=3, sim_time_s=30.0))
    frames = ReplayService(path).range(start_s=0.0, end_s=30.0)
    assert [frame.frame_id for frame in frames] == [2, 3]


# --- factory and logger ------------------------------------------------------


def test_builder_maps_carrier_and_uuv_deployment_state():
    snapshot = _snapshot(
        uuvs=(
            UUVState(
                uuv_id="uuv_03",
                position_xy=(120.0, -80.0),
                heading_rad=0.0,
                speed_mps=1.5,
                energy_fraction=0.7,
                status=UUVStatus.RETURNING,
                deployment_state=DeploymentState.RETURNING,
            ),
        ),
        carrier=CarrierState(
            carrier_id="carrier-01",
            position_xy=(-3000.0, -2995.0),
            heading_rad=1.57,
            speed_mps=1.0,
            status="recovering",
            returning_uuv_ids=("uuv_03",),
        ),
    )

    frame = build_operational_frame(snapshot, plan=None, ledger_tail=(), events=(), metrics=())

    assert frame.carrier is not None
    assert frame.carrier.status == "recovering"
    assert frame.carrier.returning_uuv_ids == ("uuv_03",)
    assert frame.uuvs[0].deployment_state == "returning"


def test_builder_omits_rays_without_a_public_uuv_origin():
    snapshot = _snapshot(
        uuvs=(_uuv("uuv_00", 10.0, 20.0),),
        contacts=(
            _contact(
                "contact-1",
                (
                    _observation("obs-uuv", "uuv_00", "T1"),
                    _observation("obs-usv", "usv_00", "T1"),
                    _observation("obs-unknown", "sensor-99", "T1"),
                ),
            ),
        ),
    )

    frame = build_operational_frame(snapshot, plan=None, ledger_tail=(), events=(), metrics=())

    assert [ray.observation_id for ray in frame.bearing_rays] == ["obs-uuv"]
    assert [ray.uuv_id for ray in frame.bearing_rays] == ["uuv_00"]
    assert "usv_00" not in frame.model_dump_json()
    assert "sensor-99" not in frame.model_dump_json()


def test_builder_maps_bounded_scheme_and_current_intelligence_views():
    scheme = OperationalScheme(
        scheme_id="scheme-1",
        version=3,
        target_priorities={"T1": 1.0},
        minimum_quality={"T1": 0.8},
        valid_from_s=0,
        valid_until_s=1000,
        constraints=tuple(f"constraint-{index}" for index in range(20)),
    )
    intelligence = tuple(
        IntelligenceReport(
            report_id=f"intel-{index:02d}",
            source="technical_reconnaissance" if index % 2 == 0 else "sonar",
            target_id="T1",
            confidence=0.7,
            issued_at_s=10,
            valid_until_s=200,
            content_summary="A" * 300,
        )
        for index in range(20)
    )
    frame = build_operational_frame(
        _snapshot(operational_scheme=scheme, intelligence_reports=intelligence),
        plan=None,
        ledger_tail=(),
        events=(),
        metrics=(),
    )

    assert frame.scheme is not None
    assert frame.scheme.scheme_id == "scheme-1"
    assert frame.scheme.minimum_quality == {"T1": 0.8}
    assert len(frame.scheme.constraints) == 16
    assert len(frame.intelligence) == 16
    assert len(frame.intelligence[0].content_summary or "") == 160
    assert frame.model_validate_json(frame.model_dump_json()) == frame


def test_engine_boundary_publishes_queued_inputs_on_the_next_observation(
    tmp_path: Path,
) -> None:
    snapshots = []
    base_config = load_app_config(CONFIG_PATH)
    config = base_config.model_copy(
        update={
            "scenario": base_config.scenario.model_copy(
                update={"operational_scheme": None}
            )
        }
    )
    scheme = OperationalScheme(
        scheme_id="boundary-scheme",
        version=1,
        valid_from_s=0,
        valid_until_s=120,
    )
    report = IntelligenceReport(
        report_id="boundary-intelligence",
        source="sonar",
        target_id="target_00",
        confidence=0.8,
        issued_at_s=0,
        valid_until_s=120,
    )

    engine: SimulationEngine

    def on_situation(snapshot) -> None:
        snapshots.append(snapshot)
        if snapshot.sim_time_s == 30:
            engine.set_operational_scheme(scheme)
            engine.submit_intelligence(report)

    engine = SimulationEngine(
        config, seed=7, output_dir=tmp_path, carrier=on_situation
    )
    frames = [engine.step() for _ in range(60 // config.timing.physics_step_s)]

    assert [snapshot.sim_time_s for snapshot in snapshots] == [30, 60]
    assert snapshots[0].operational_scheme is None
    assert snapshots[0].intelligence_reports == ()
    assert snapshots[1].operational_scheme == scheme
    assert snapshots[1].intelligence_reports == (report,)
    assert frames[-1]["sim_time_s"] == 60


def test_expired_boundary_input_does_not_stop_the_real_engine_loop(
    tmp_path: Path,
) -> None:
    config = load_app_config(CONFIG_PATH)
    loop = _AgentLoop(
        config,
        database_path=tmp_path / "agent.db",
        llm=make_live_llm(),
        run_id="boundary-input-test",
        steps=60 // config.timing.physics_step_s,
        seed=7,
    )
    engine = SimulationEngine(
        config, seed=7, output_dir=tmp_path / "frames", carrier=loop.on_situation
    )
    loop.attach(engine)
    report = IntelligenceReport(
        report_id="expired-at-boundary",
        source="sonar",
        target_id="target_00",
        confidence=0.8,
        issued_at_s=0,
        valid_until_s=30,
    )
    loop.runtime.submit_intelligence(report)

    try:
        frames = [
            engine.step() for _ in range(60 // config.timing.physics_step_s)
        ]
        assert [frame["sim_time_s"] for frame in frames] == list(
            range(config.timing.physics_step_s, 61, config.timing.physics_step_s)
        )
        assert loop.carrier_error_count >= 2
        assert loop.runtime.drain_operational_inputs() == (None, (report,))
    finally:
        loop.close()


def test_frame_factory_builds_valid_frames(frame_factory):
    frame = frame_factory(frame_id=2, sim_time_s=20.0)
    assert frame.frame_id == 2
    assert frame.sim_time_s == 20
    assert frame.plan_version == 4
    assert frame.model_validate_json(frame.model_dump_json()) == frame
    assert any(plan.status == "active" and plan.version == 4 for plan in frame.plans)


def test_builder_frames_round_trip_through_logger_and_replay(tmp_path):
    snapshot = _snapshot(
        uuvs=(_uuv("UUV-1", 10.0, 20.0), _uuv("UUV-2", -30.0, 5.0)),
        reports=(_report("T1", "G1", (40.0, 40.0), ((9.0, 0.0), (0.0, 4.0))),),
        contacts=(_contact("contact-1", (_observation("obs-1", "UUV-1", "T1"),)),),
        events=(_event("evt-1"),),
    )
    ledger = (_decision("dec-1", diff=PlanDiff(to_plan_id="plan-7", to_revision=4)),)
    metrics = (MetricView(metric_id="m-1", value=1.0),)
    frame = build_operational_frame(
        snapshot, _plan(), ledger, snapshot.pending_events, metrics
    )
    path = tmp_path / "frames.jsonl"
    with FrameLogger(path) as logger:
        logger.append(frame)
    restored = ReplayService(path).range()
    assert restored == [frame]


def test_logger_appends_validated_lines_and_flushes_immediately(tmp_path, frame_factory):
    path = tmp_path / "frames.jsonl"
    logger = FrameLogger(path)
    logger.append(frame_factory(frame_id=1, sim_time_s=10.0))
    # The line is visible to a fresh reader before the logger closes.
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    logger.append(frame_factory(frame_id=2, sim_time_s=20.0))
    logger.close()
    assert logger.count == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    restored = [OperationalFrame.model_validate_json(line) for line in lines]
    assert [frame.frame_id for frame in restored] == [1, 2]
    assert restored[0] == frame_factory(frame_id=1, sim_time_s=10.0)


# --- replay ------------------------------------------------------------------


def test_replay_reloads_appended_frames(tmp_path, frame_factory):
    path = tmp_path / "frames.jsonl"
    with FrameLogger(path) as logger:
        logger.append(frame_factory(frame_id=2, sim_time_s=20.0))
        replay = ReplayService(path)
        assert [frame.frame_id for frame in replay.range()] == [2]
        logger.append(frame_factory(frame_id=3, sim_time_s=30.0))
        assert [frame.frame_id for frame in replay.range()] == [2, 3]


def test_replay_accepts_legacy_jsonl_frame_without_carrier(tmp_path, frame_factory):
    path = tmp_path / "legacy-frames.jsonl"
    path.write_text(
        frame_factory(frame_id=1, sim_time_s=10.0).model_dump_json(exclude={"carrier"}) + "\n",
        encoding="utf-8",
    )

    frame = ReplayService(path).range()[0]

    assert frame.carrier is None


def test_replay_normalizes_legacy_jsonl_without_carrier_or_deployment_states():
    path = Path(__file__).parents[1] / "fixtures" / "legacy-carrierless-deploymentless.jsonl"

    frame = ReplayService(path).range()[0]

    assert frame.carrier is None
    assert {uuv.uuv_id: uuv.deployment_state.value for uuv in frame.uuvs} == {
        "UUV-legacy-deployed": "deployed",
        "UUV-legacy-returning": "returning",
    }


def test_replay_time_range_is_inclusive_and_unbounded(tmp_path, frame_factory):
    path = tmp_path / "frames.jsonl"
    with FrameLogger(path) as logger:
        logger.append(frame_factory(frame_id=1, sim_time_s=10.0))
        logger.append(frame_factory(frame_id=2, sim_time_s=20.0))
        logger.append(frame_factory(frame_id=3, sim_time_s=30.0))
    replay = ReplayService(path)
    assert [f.frame_id for f in replay.range(start_s=0.0, end_s=30.0)] == [1, 2, 3]
    assert [f.frame_id for f in replay.range(start_s=15.0, end_s=30.0)] == [2, 3]
    assert [f.frame_id for f in replay.range(start_s=20.0, end_s=20.0)] == [2]
    assert [f.frame_id for f in replay.range(start_s=25.0, end_s=29.0)] == []
    assert [f.frame_id for f in replay.range(start_s=10.0)] == [1, 2, 3]
    assert [f.frame_id for f in replay.range(end_s=20.0)] == [1, 2]
    assert [f.frame_id for f in replay.range(offset=1, limit=1)] == [2]
    assert replay.count() == 3
    assert replay.last().frame_id == 3


def test_replay_default_page_is_bounded(tmp_path):
    path = tmp_path / "large-frames.jsonl"
    with FrameLogger(path) as logger:
        base = _full_frame()
        for frame_id in range(1001):
            logger.append(
                base.model_copy(update={"frame_id": frame_id, "sim_time_s": frame_id})
            )

    replay = ReplayService(path)

    assert len(replay.range()) == 1000
    with pytest.raises(ValueError, match="limit"):
        replay.range(limit=10_001)


def test_replay_pages_cover_the_complete_indexed_range(tmp_path):
    path = tmp_path / "paged-frames.jsonl"
    with FrameLogger(path) as logger:
        base = _full_frame()
        for frame_id in range(2_501):
            logger.append(
                base.model_copy(update={"frame_id": frame_id, "sim_time_s": frame_id * 5})
            )

    replay = ReplayService(path)

    first_page = replay.range(offset=0, limit=1_000)
    second_page = replay.range(offset=1_000, limit=1_000)
    final_page = replay.range(offset=2_000, limit=1_000)

    assert [len(first_page), len(second_page), len(final_page)] == [1_000, 1_000, 501]
    assert len(first_page) + len(second_page) + len(final_page) == replay.count() == 2_501
    assert final_page[-1].sim_time_s == 12_500


def test_replay_rejects_garbage_lines_with_line_numbers(tmp_path, frame_factory):
    path = tmp_path / "frames.jsonl"
    with FrameLogger(path) as logger:
        logger.append(frame_factory(frame_id=1, sim_time_s=10.0))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    with pytest.raises(ReplayIndexError) as excinfo:
        ReplayService(path)
    assert excinfo.value.line_number == 2
    assert "line 2" in str(excinfo.value)


def test_replay_rejects_schema_invalid_lines_with_line_numbers(tmp_path, frame_factory):
    path = tmp_path / "frames.jsonl"
    with FrameLogger(path) as logger:
        logger.append(frame_factory(frame_id=1, sim_time_s=10.0))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version": "1.0", "frame_id": 2, "sim_time_s": 20}\n')
    with pytest.raises(ReplayIndexError) as excinfo:
        ReplayService(path)
    assert excinfo.value.line_number == 2


def test_replay_empty_and_missing_logs_serve_no_frames(tmp_path):
    assert ReplayService(tmp_path / "missing.jsonl").range() == []
    path = tmp_path / "empty.jsonl"
    path.touch()
    assert ReplayService(path).range() == []


# --- builder: deterministic ordering -----------------------------------------


def test_builder_sorts_every_entity_list_by_stable_id():
    snapshot = _snapshot(
        uuvs=(
            _uuv("UUV-3", 1.0, 1.0),
            _uuv("UUV-1", 2.0, 2.0),
            _uuv("UUV-2", 3.0, 3.0),
        ),
        reports=(
            _report("T2", "G2", (50.0, 50.0), ((1.0, 0.0), (0.0, 1.0))),
            _report("T1", "G1", (40.0, 40.0), ((1.0, 0.0), (0.0, 1.0))),
        ),
        contacts=(
            _contact("contact-2", (_observation("obs-3", "UUV-3", "T2"),)),
            _contact(
                "contact-1",
                (
                    _observation("obs-2", "UUV-2", "T2"),
                    _observation("obs-1", "UUV-1", "T1"),
                ),
            ),
        ),
        events=(_event("evt-2"), _event("evt-1")),
    )
    ledger = (_decision("dec-2"), _decision("dec-1"))
    metrics = (
        MetricView(metric_id="metric-b", value=2.0),
        MetricView(metric_id="metric-a", value=1.0),
    )
    frame = build_operational_frame(snapshot, _plan(), ledger, snapshot.pending_events, metrics)
    assert [u.uuv_id for u in frame.uuvs] == ["UUV-1", "UUV-2", "UUV-3"]
    assert [e.target_id for e in frame.target_estimates] == ["T1", "T2"]
    assert [r.observation_id for r in frame.bearing_rays] == ["obs-1", "obs-2", "obs-3"]
    assert [g.group_id for g in frame.groups] == ["G1", "G2"]
    assert frame.groups[0].member_ids == ("UUV-1", "UUV-2")
    assert [e.event_id for e in frame.events] == ["evt-1", "evt-2"]
    assert [ledger_entry.decision_id for ledger_entry in frame.ledger] == ["dec-1", "dec-2"]
    assert [m.metric_id for m in frame.metrics] == ["metric-a", "metric-b"]
    assert [p.plan_id for p in frame.plans] == ["plan-7"]


def test_builder_projects_truth_safe_platform_and_brain_status() -> None:
    def capability(kind: PlatformKind) -> PlatformCapability:
        return PlatformCapability(
            kind=kind,
            motion=MotionLimits(
                max_speed_mps=8.0,
                max_acceleration_mps2=0.5,
                max_turn_rate_rad_s=0.1,
            ),
            sonar=SonarCapability(
                passive_range_m=5000.0,
                passive_bearing_variance_rad2=0.02,
                active_source_range_m=3500.0,
                active_receive_range_m=3000.0,
                active_range_sigma_m=20.0,
                active_bearing_sigma_rad=0.03,
                active_capable=True,
                ping_cooldown_s=30,
                ping_energy_cost_fraction=0.01,
                clutter_sensitivity=0.2,
                exposure_cost=0.3,
            ),
            communications=CommunicationCapability(
                surface_range_m=250.0,
                acoustic_range_m=250.0,
            ),
        )

    usv_capability = capability(PlatformKind.USV)
    uuv_capability = capability(PlatformKind.UUV)
    platform_snapshot = PlatformSnapshot(
        scenario_id="scenario-20260814",
        sim_time_s=100,
        carrier=CarrierPlatformState(
            carrier_id="carrier-01",
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=1.0,
            support_radius_m=600.0,
            onboard_platform_ids=(),
            deployed_platform_ids=("usv-01", "uuv-01", "uuv-02"),
            returning_platform_ids=(),
        ),
        roster=PlatformRoster(
            usvs=(
                USVPlatformState(
                    platform_id="usv-01",
                    platform_index=0,
                    position_xy=(100.0, 0.0),
                    heading_rad=0.0,
                    speed_mps=4.0,
                    energy_fraction=0.9,
                    deployment_state="deployed",
                    capability=usv_capability,
                    distance_to_carrier_m=100.0,
                ),
            ),
            uuvs=(
                UUVPlatformState(
                    platform_id="uuv-01",
                    platform_index=0,
                    position_xy=(200.0, 0.0),
                    heading_rad=0.0,
                    speed_mps=2.0,
                    energy_fraction=0.8,
                    deployment_state="deployed",
                    capability=uuv_capability,
                    is_group_leader=True,
                    master_connected=True,
                ),
                UUVPlatformState(
                    platform_id="uuv-02",
                    platform_index=1,
                    position_xy=(1000.0, 0.0),
                    heading_rad=0.0,
                    speed_mps=2.0,
                    energy_fraction=0.7,
                    deployment_state="deployed",
                    capability=uuv_capability,
                ),
            ),
        ),
        communication_links=(
            CommunicationLink(
                source_id="carrier-01",
                target_id="usv-01",
                medium="surface",
                distance_m=100.0,
            ),
            CommunicationLink(
                source_id="usv-01",
                target_id="uuv-01",
                medium="acoustic",
                distance_m=100.0,
            ),
        ),
    )
    snapshot = _snapshot(
        uuvs=(
            _uuv("uuv-01", 200.0, 0.0),
            _uuv("uuv-02", 1000.0, 0.0),
        ),
        carrier=CarrierState(
            carrier_id="carrier-01",
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=1.0,
            status="transit",
            deployed_uuv_ids=("uuv-01", "uuv-02"),
        ),
    ).model_copy(update={"platform_snapshot": platform_snapshot})

    frame = build_operational_frame(snapshot, _plan(), (), (), ())

    assert frame.carrier is not None
    assert frame.carrier.support_radius_m == 600.0
    assert frame.usvs[0].relay_active is True
    assert frame.usvs[0].sensor_mode == "passive"
    assert frame.uuvs[0].is_group_leader is True
    assert frame.uuvs[0].master_connected is True
    link_status = {(link.source_id, link.target_id): link.status for link in frame.communication_links}
    assert link_status[("carrier-01", "usv-01")] == "connected"
    assert link_status[("usv-01", "uuv-02")] == "disconnected"
    assert {brain.role for brain in frame.brains} == {"master", "slave", "adversary"}
    payload = frame.model_dump_json()
    assert "true_position" not in payload
    assert "target_truth" not in payload
    assert OperationalFrame.model_validate_json(payload) == frame

    wide_snapshot = snapshot.model_copy(
        update={
            "carrier": snapshot.carrier.model_copy(
                update={"position_xy": (-8000.0, -8000.0)}
            )
        }
    )
    default_frame = build_operational_frame(wide_snapshot, _plan(), (), (), ())
    explicit_frame = build_operational_frame(
        wide_snapshot,
        _plan(),
        (),
        (),
        (),
        map_bounds_xy=(-6000.0, 6000.0, -6000.0, 6000.0),
    )
    assert default_frame.carrier is not None and explicit_frame.carrier is not None
    assert (default_frame.map_bounds.min_x, default_frame.map_bounds.max_x) == (-12000.0, 12000.0)
    assert (default_frame.map_bounds.min_y, default_frame.map_bounds.max_y) == (-12000.0, 12000.0)
    assert default_frame.carrier.position.x == -8000.0
    assert explicit_frame.map_bounds.min_x == -6000.0
    assert explicit_frame.map_bounds.max_x == 6000.0
    assert explicit_frame.carrier.position.x == -6000.0


def test_builder_prefers_live_snapshot_map_bounds_over_fallback():
    snapshot = _snapshot(
        carrier=CarrierState(
            carrier_id="carrier-01",
            position_xy=(-8000.0, -8000.0),
            heading_rad=0.0,
            speed_mps=1.0,
            status="transit",
        ),
    ).model_copy(
        update={"map_bounds_xy": (-9000.0, 9000.0, -9000.0, 9000.0)}
    )

    frame = build_operational_frame(snapshot, _plan(), (), (), ())

    assert (
        frame.map_bounds.min_x,
        frame.map_bounds.max_x,
        frame.map_bounds.min_y,
        frame.map_bounds.max_y,
    ) == (-9000.0, 9000.0, -9000.0, 9000.0)
    assert frame.carrier is not None
    assert frame.carrier.position.x == -8000.0


# --- builder: covariance conversion ------------------------------------------


def test_builder_converts_covariance_into_ellipse_axes_and_rotation():
    snapshot = _snapshot(
        reports=(_report("T1", "G1", (40.0, 40.0), ((9.0, 0.0), (0.0, 4.0))),)
    )
    frame = build_operational_frame(snapshot, _plan(), (), (), ())
    ellipse = frame.target_estimates[0].covariance_ellipse
    assert ellipse.semimajor_m == pytest.approx(3.0)
    assert ellipse.semiminor_m == pytest.approx(2.0)
    assert ellipse.rotation_rad == pytest.approx(0.0)
    # The RMS position error proxy is the covariance trace (E[||x-mu||^2]).
    assert frame.target_estimates[0].quality.estimated_rmse_m == pytest.approx(
        math.sqrt(13.0)
    )


def test_builder_converts_rotated_covariance():
    snapshot = _snapshot(
        reports=(_report("T1", "G1", (40.0, 40.0), ((5.0, 3.0), (3.0, 5.0))),)
    )
    ellipse = build_operational_frame(snapshot, _plan(), (), (), ()).target_estimates[0].covariance_ellipse
    assert ellipse.semimajor_m == pytest.approx(math.sqrt(8.0))
    assert ellipse.semiminor_m == pytest.approx(math.sqrt(2.0))
    assert ellipse.rotation_rad == pytest.approx(math.pi / 4)


def test_builder_floors_degenerate_covariance_axes():
    snapshot = _snapshot(
        reports=(_report("T1", "G1", (40.0, 40.0), ((1.0, 0.0), (0.0, 0.0))),)
    )
    ellipse = build_operational_frame(snapshot, _plan(), (), (), ()).target_estimates[0].covariance_ellipse
    assert ellipse.semiminor_m > 0.0
    assert ellipse.semiminor_m <= ellipse.semimajor_m


def test_builder_raises_on_malformed_covariance():
    snapshot = _snapshot(
        reports=(_report("T1", "G1", (40.0, 40.0), ((1.0,),)),)
    )
    with pytest.raises(ValueError, match="covariance"):
        build_operational_frame(snapshot, _plan(), (), (), ())


# --- builder: geometry clipping ----------------------------------------------


def test_builder_clips_geometry_to_map_bounds():
    snapshot = _snapshot(
        uuvs=(_uuv("UUV-1", 6000.0, -6000.0),),
        reports=(_report("T1", "G1", (9000.0, 5.0), ((1.0, 0.0), (0.0, 1.0))),),
        contacts=(_contact("contact-1", (_observation("obs-1", "UUV-1", "T1"),)),),
    )
    plan = _plan(waypoints={"UUV-1": (Waypoint(x=7000.0, y=0.0),)})
    frame = build_operational_frame(
        snapshot,
        plan,
        (),
        (),
        (),
        map_bounds_xy=(-4000.0, 4000.0, -4000.0, 4000.0),
    )
    bounds = frame.map_bounds
    uuv = frame.uuvs[0]
    assert uuv.position.x == pytest.approx(bounds.max_x)
    assert uuv.position.y == pytest.approx(bounds.min_y)
    assert frame.target_estimates[0].mean.x == pytest.approx(bounds.max_x)
    assert frame.bearing_rays[0].origin.x == pytest.approx(bounds.max_x)
    assert uuv.current_waypoint is not None
    assert uuv.current_waypoint.x == pytest.approx(bounds.max_x)


def test_builder_omits_ray_when_observer_has_no_public_uuv_origin():
    snapshot = _snapshot(
        uuvs=(_uuv("UUV-1", 0.0, 0.0),),
        contacts=(_contact("contact-1", (_observation("obs-1", "UUV-9", "T1"),)),),
    )

    frame = build_operational_frame(snapshot, _plan(), (), (), ())

    assert frame.bearing_rays == ()


# --- builder: plan version and views -----------------------------------------


def test_builder_attaches_committed_plan_version():
    snapshot = _snapshot(
        reports=(_report("T1", "G1", (40.0, 40.0), ((1.0, 0.0), (0.0, 1.0))),)
    )
    frame = build_operational_frame(snapshot, _plan(revision=7), (), (), ())
    assert frame.plan_version == 7
    assert frame.plans[0].version == 7
    assert frame.plans[0].status == "active"
    assert frame.plans[0].concept == "balanced"
    assert frame.plans[0].valid_from_s == 60
    assert frame.plans[0].valid_until_s == 240
    assert frame.plans[0].affected_targets == ("T1",)
    # Before the first plan commit the frame carries no plan and version 0.
    frame0 = build_operational_frame(snapshot, None, (), (), ())
    assert frame0.plan_version == 0
    assert frame0.plans == ()


def test_builder_maps_plan_time_boundaries_and_group_changes():
    snapshot = _snapshot()
    diff = PlanDiff(
        from_plan_id="plan-6",
        from_revision=3,
        to_plan_id="plan-7",
        to_revision=4,
        members_added={"G2": ("UUV-3", "UUV-4")},
        members_removed={"G1": ("UUV-2",)},
    )
    frame = build_operational_frame(snapshot, _plan(diff=diff), (), (), ())
    assert frame.plans[0].group_changes == (
        "G1 removes UUV-2",
        "G2 adds UUV-3, UUV-4",
    )
    open_ended = build_operational_frame(snapshot, _plan(valid_until_s=0), (), (), ())
    assert open_ended.plans[0].valid_until_s is None


def test_builder_renders_intent_from_plan_references():
    snapshot = _snapshot(
        reports=(_report("T1", "G1", (40.0, 40.0), ((1.0, 0.0), (0.0, 1.0))),)
    )
    frame = build_operational_frame(snapshot, _plan(intent_refs={"T1": "transit"}), (), (), ())
    assert frame.target_estimates[0].intent.label == "transit"
    unknown_ref = build_operational_frame(snapshot, _plan(intent_refs={"T1": "flee"}), (), (), ())
    assert unknown_ref.target_estimates[0].intent.label == "unknown"
    no_plan = build_operational_frame(snapshot, None, (), (), ())
    assert no_plan.target_estimates[0].intent.label == "unknown"


# --- builder: ledger, events, metrics, quality --------------------------------


def test_builder_maps_ledger_events_and_metrics():
    snapshot = _snapshot()
    ledger = (
        _decision("dec-2", final_plan_id=None),
        _decision("dec-1", diff=PlanDiff(to_plan_id="plan-7", to_revision=4)),
    )
    events = (_event("evt-1"),)
    metrics = (
        MetricView(metric_id="m-2", value=2.0),
        MetricView(metric_id="m-1", value=1.0),
    )
    frame = build_operational_frame(snapshot, _plan(), ledger, events, metrics)
    assert frame.ledger[0].outcome == "committed"
    assert frame.ledger[0].final_plan_id == "plan-7"
    assert frame.ledger[0].final_plan_version == 4
    assert frame.ledger[0].trigger_event_ids == ("evt-1",)
    assert frame.ledger[0].evidence_ids == ("obs-1",)
    assert frame.ledger[1].outcome == "rejected"
    assert frame.ledger[1].final_plan_version is None
    assert [event.event_id for event in frame.events] == ["evt-1"]
    assert frame.events[0].level == "tactical"
    assert [metric.metric_id for metric in frame.metrics] == ["m-1", "m-2"]


def test_builder_marks_degraded_decisions():
    snapshot = _snapshot()
    degraded = _decision(
        "dec-1", verification=(ValidationReport(valid=False, degraded=True),)
    )
    frame = build_operational_frame(snapshot, _plan(), (degraded,), (), ())
    assert frame.ledger[0].outcome == "degraded"
    assert frame.ledger[0].final_plan_id == "plan-7"


def test_builder_caps_nonfinite_fim_condition():
    snapshot = _snapshot(
        reports=(_report("T1", "G1", (40.0, 40.0), ((1.0, 0.0), (0.0, 1.0))),)
    )
    frame = build_operational_frame(snapshot, _plan(), (), (), ())
    assert frame.target_estimates[0].quality.fim_condition == 1.0e6
    finite = _snapshot(
        reports=(
            _report(
                "T1",
                "G1",
                (40.0, 40.0),
                ((1.0, 0.0), (0.0, 1.0)),
                fim_condition=25.0,
            ),
        )
    )
    frame_finite = build_operational_frame(finite, _plan(), (), (), ())
    assert frame_finite.target_estimates[0].quality.fim_condition == 25.0
