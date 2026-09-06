from __future__ import annotations

from underwater_tracking.api.frame_builder import build_operational_frame
from underwater_tracking.domain.models import SituationSnapshot
from tests.domain.test_execution_models import _snapshot as execution_snapshot


def _handoff_payload(*observer_ids: str) -> dict[str, object]:
    required = ("uuv_03", "uuv_04", "uuv_05")
    return {
        "predecessor_region_id": "target_00:task:01",
        "successor_region_id": "target_00:task:02",
        "plan_revision": 9,
        "observation_cycle_s": 120,
        "required_uuv_ids": required,
        "deployed_uuv_ids": required,
        "healthy_uuv_ids": required,
        "passive_mode_uuv_ids": required,
        "accepted_observations": tuple(
            {
                "observation_id": f"obs-{observer_id}",
                "observer_uuv_id": observer_id,
                "observed_at_s": 120,
            }
            for observer_id in observer_ids
        ),
        "hard_guard_reasons": (),
        "blocked_reason": None,
    }


def _situation(*, handoff_observers: tuple[str, ...] = ("uuv_03", "uuv_04")) -> SituationSnapshot:
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=12,
        sim_time_s=120,
        uuvs=(),
        group_reports=(),
        pending_events=(),
        region_probability_evidence={
            "target_00:task:01": {
                "probability": 0.72,
                "status": "current",
                "source_observation_ids": ("obs-entry",),
                "evidence_ids": ("entry-evidence",),
            }
        },
        mission_scan_telemetry={
            "target_00:task:01": {
                "route_progress": 1.0,
                "scan_round": 2,
                "active_coverage_ratio": 0.0,
                "source_backed_ping_count": 0,
                "active_ping_count": 0,
                "scan_completed": False,
                "scan_completion_threshold": 0.8,
                "evidence_ids": (),
            }
        },
        mission_runtime_group_evidence={
            "target_00:task:01:deploy:000009": {
                "deployed_uuv_ids": ("uuv_00", "uuv_01", "uuv_02"),
                "passive_observer_ids": (),
            },
            "target_00:task:02:deploy:000009": {
                "deployed_uuv_ids": ("uuv_03", "uuv_04", "uuv_05"),
                "passive_observer_ids": ("uuv_03", "uuv_04", "uuv_05"),
            },
        },
        mission_handoff_evidence={
            "target_00:task:01": _handoff_payload(*handoff_observers),
        },
        mission_batch_ids_by_region={"target_00:task:01": "carrier-01:candidate-01:r9"},
    )


def _frame(situation: SituationSnapshot | None = None):
    return build_operational_frame(
        situation or _situation(),
        plan=None,
        ledger_tail=(),
        events=(),
        metrics=(),
        uuv_only=True,
        execution_snapshot=execution_snapshot(),
    )


def test_real_runtime_evidence_is_projected_into_execution_contract() -> None:
    frame = _frame()

    assert frame.execution is not None
    region = frame.execution.regions[0]
    assert region.scan_telemetry is not None
    assert region.scan_telemetry.route_progress == 1.0
    assert region.scan_telemetry.active_coverage_ratio == 0.0
    assert region.scan_telemetry.source_backed_ping_count == 0
    assert region.scan_telemetry.scan_completed is False

    entry = frame.execution.region_entry_evidence["target_00:task:01"]
    assert entry.probability == 0.72
    assert entry.confirmation_count == 0
    assert entry.required_cycles == 2
    assert entry.evidence_ids == ("entry-evidence",)

    group = next(
        group
        for group in frame.execution.task_groups
        if group.region_id == "target_00:task:02"
    )
    assert group.deployed_member_uuv_ids == ("uuv_03", "uuv_04", "uuv_05")
    assert group.passive_observation_uuv_ids == ("uuv_03", "uuv_04", "uuv_05")

    assert frame.execution.handoff_evidence is not None
    assert frame.execution.handoff_evidence.valid_observation_uuv_ids == (
        "uuv_03",
        "uuv_04",
    )
    assert frame.execution.handoff_evidence.status == "ready"

    assert frame.execution.replacements == ()
    assert frame.target_estimates[0].estimate_freshness is not None
    assert frame.target_estimates[0].estimate_freshness.status == "current"
    assert frame.target_estimates[0].estimate_freshness.track_revision == 7
    assert frame.target_estimates[0].source_observation_ids == ("target-step-7",)


def test_handoff_and_scan_projections_are_source_backed_and_not_route_inference() -> None:
    complete = _frame(_situation(handoff_observers=("uuv_03", "uuv_04", "uuv_05")))

    assert complete.execution is not None
    assert complete.execution.handoff_evidence is not None
    assert complete.execution.handoff_evidence.status == "completed"
    assert complete.execution.handoff_evidence.valid_observation_uuv_ids == (
        "uuv_03",
        "uuv_04",
        "uuv_05",
    )
    assert complete.execution.regions[0].scan_telemetry is not None
    assert complete.execution.regions[0].scan_telemetry.scan_completed is False


def test_expired_execution_track_is_published_as_expired() -> None:
    execution = execution_snapshot().model_copy(
        update={
            "source_sim_time_s": 2_000,
            "generated_at_s": 2_000.0,
            "valid_from_s": 2_000.0,
            "valid_until_s": 2_100.0,
            "target_track": execution_snapshot().target_track.model_copy(
                update={"sim_time_s": 120.0, "valid_until_s": 180.0}
            ),
        }
    )
    situation = _situation().model_copy(update={"sim_time_s": 200})
    frame = build_operational_frame(
        situation,
        plan=None,
        ledger_tail=(),
        events=(),
        metrics=(),
        uuv_only=True,
        execution_snapshot=execution,
    )

    assert frame.target_estimates[0].estimate_freshness is not None
    assert frame.target_estimates[0].estimate_freshness.status == "expired"
    assert frame.target_estimates[0].estimate_health_status == "expired"
