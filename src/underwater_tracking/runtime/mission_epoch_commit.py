"""Runtime commit port for semantically revalidated UUV mission epochs."""

from __future__ import annotations

from collections.abc import Callable

from underwater_tracking.agent.nodes.commit import EpochCommitPort
from underwater_tracking.domain.agent_models import TrackingPlan
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.domain.planning_epoch_models import (
    EpochCommitResult,
    PlanningEpoch,
)
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.persistence.planning_epochs import PlanningEpochRepository
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.persistence.uuv_plan_commits import UUVPlanCommitRepository
from underwater_tracking.planning.mission_revalidation import revalidate_executable_mission_plan
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.runtime.scenario_transition import ScenarioTransitionCoordinator


class MissionEpochCommitPort(EpochCommitPort):
    """Commit a revalidated plan and its controller state as one transition.

    The situation and controller snapshots are acquired before entering the
    transition lock.  No provider or model call is made while the lock or the
    SQLite transaction is held; the critical section only stages, applies, and
    commits copy-on-write state.
    """

    def __init__(
        self,
        *,
        plans: PlanRepository,
        epochs: PlanningEpochRepository,
        mission_controller: MissionController,
        transition_coordinator: ScenarioTransitionCoordinator,
        situation_provider: Callable[[], SituationSnapshot],
        expert_request_version_provider: Callable[[], int | None] | None = None,
        recovered_event_ids_provider: Callable[[], frozenset[str]] | None = None,
        commit_repository: UUVPlanCommitRepository | None = None,
        uuv_only: bool = False,
    ) -> None:
        if transition_coordinator.scenario_id != mission_controller.scenario_id:
            raise ValueError("transition and mission controller scenarios must match")
        self._plans = plans
        self._epochs = epochs
        self._mission_controller = mission_controller
        self._transitions = transition_coordinator
        self._situation_provider = situation_provider
        self._expert_request_version_provider = expert_request_version_provider
        self._recovered_event_ids_provider = recovered_event_ids_provider
        self._commit_repository = commit_repository
        self._uuv_only = uuv_only

    def commit(
        self,
        *,
        epoch: PlanningEpoch,
        audit_projection: TrackingPlan,
        executable_plan: ExecutableMissionPlan,
    ) -> EpochCommitResult:
        """Revalidate against live public state and atomically commit the plan."""
        situation = self._situation_provider()
        mission = self._mission_controller.snapshot()
        expert_version = (
            self._expert_request_version_provider()
            if self._expert_request_version_provider is not None
            else None
        )
        recovered_event_ids = (
            self._recovered_event_ids_provider()
            if self._recovered_event_ids_provider is not None
            else frozenset()
        )
        report = revalidate_executable_mission_plan(
            epoch=epoch,
            candidate=executable_plan,
            current_situation=situation,
            current_mission=mission,
            current_expert_request_version=expert_version,
            recovered_event_ids=recovered_event_ids,
        )
        if not report.valid:
            result = EpochCommitResult(
                epoch_id=epoch.epoch_id,
                status="invalidated",
                validation_report_id=report.report_id,
                invalidated_reason="; ".join(
                    issue.code for issue in report.issues
                ),
            )
            with self._transitions.transition("plan"):
                self._epochs.finish_with_revalidation(report, result)
            return result

        rebased_plan = report.rebased_plan
        if rebased_plan is None:
            return self._finish_failure(
                epoch,
                "semantic revalidation returned no rebased executable plan",
            )

        # The audit projection follows the executable revision after harmless
        # physics drift; it remains an audit view, never the UUV authority.
        audit = audit_projection.model_copy(
            update={
                "revision": rebased_plan.revision,
                "base_snapshot_revision": situation.snapshot_revision,
            }
        )
        commit_repository = self._commit_repository or UUVPlanCommitRepository(
            self._plans
        )
        with self._transitions.transition("plan"):
            locked_mission = self._mission_controller.snapshot()
            if locked_mission.plan_revision != report.current_plan_version:
                return self._finish_failure(
                    epoch,
                    "mission controller advanced while the epoch was being committed",
                )
            checkpoint = self._mission_controller.checkpoint()
            prepared = None
            try:
                prepared = commit_repository.prepare(
                    epoch=epoch,
                    report=report,
                    audit_projection=audit,
                    executable_plan=rebased_plan,
                    expected_active_plan_revision=locked_mission.plan_revision,
                )
                if not self._uuv_only:
                    applied = self._mission_controller.apply_revalidated_plan(
                        rebased_plan,
                        expected_current_revision=locked_mission.plan_revision,
                    )
                    if not applied:
                        raise RuntimeError(
                            "mission controller rejected the revalidated plan"
                        )
                result = EpochCommitResult(
                    epoch_id=epoch.epoch_id,
                    status="committed",
                    plan_id=audit.plan_id,
                    plan_version=rebased_plan.revision,
                    validation_report_id=report.report_id,
                    executable_plan=rebased_plan,
                    consumed_event_ids=epoch.critical_event_ids,
                )
                prepared.finish(result)
                return result
            except Exception as exc:  # noqa: BLE001 - restore the whole boundary
                self._mission_controller.restore(checkpoint)
                if prepared is not None:
                    try:
                        prepared.rollback()
                    except Exception:
                        pass
                return self._finish_failure(
                    epoch,
                    f"{type(exc).__name__}: {exc}",
                )

    def _finish_failure(self, epoch: PlanningEpoch, message: str) -> EpochCommitResult:
        """Persist a bounded failure after any staged transaction is rolled back."""
        result = EpochCommitResult(
            epoch_id=epoch.epoch_id,
            status="failed",
            failure_category="internal",
            failure_message=message[:2000],
        )
        self._epochs.finish(result)
        return result
