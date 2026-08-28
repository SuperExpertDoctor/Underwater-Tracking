"""Atomic coordination of the authoritative UUV execution snapshot.

The legacy graph still produces a ``TrackingPlan`` for audit and replay.  It
is deliberately not the online execution source.  This module owns the
immutable execution snapshot, performs compare-and-set checks, and only then
asks the physical mission controller to apply a revision.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from threading import RLock, Thread
from typing import Any, Literal

from pydantic import ConfigDict, Field

from underwater_tracking.domain.agent_models import TrackingPlan
from underwater_tracking.domain.execution_models import OperationalExecutionSnapshot
from underwater_tracking.domain.models import StrictModel
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.runtime.execution_health import (
    ExecutionHealth,
    classify_execution_health,
)


ExecutionCommitStatus = Literal[
    "committed",
    "stale",
    "rejected",
    "preserved",
    "failed",
]


class ExecutionCommitResult(StrictModel):
    """Result of one execution proposal attempt.

    A non-committed result carries the still-active snapshot.  Callers can
    therefore update health and audit views without ever replacing the
    physical plan with a half-built candidate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    commit_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    status: ExecutionCommitStatus
    accepted: bool = False
    execution_revision: int | None = Field(default=None, ge=1)
    candidate_execution_revision: int | None = Field(default=None, ge=1)
    base_execution_revision: int | None = Field(default=None, ge=0)
    preserved_execution_revision: int | None = Field(default=None, ge=1)
    snapshot: OperationalExecutionSnapshot | None = None
    reason: str = Field(default="", max_length=2000)
    active_plan_preserved: bool = False
    was_rebased: bool = False
    audit_plan_id: str | None = None

    @property
    def committed(self) -> bool:
        """Whether this result installed a new physical revision."""

        return self.status == "committed" and self.accepted

    @property
    def preserved(self) -> bool:
        """Whether the previous physical revision remains authoritative."""

        return self.status in {"stale", "preserved", "failed"}

    @property
    def execution_snapshot(self) -> OperationalExecutionSnapshot | None:
        """Compatibility name for consumers that call the result a snapshot."""

        return self.snapshot


SnapshotApplier = Callable[[OperationalExecutionSnapshot], object]
EvidenceResolver = Callable[[str], object]
SemanticOptimizer = Callable[
    [OperationalExecutionSnapshot], OperationalExecutionSnapshot
]
SnapshotPublisher = Callable[[OperationalExecutionSnapshot], object]
SnapshotCandidate = (
    OperationalExecutionSnapshot
    | Mapping[str, Any]
    | Callable[[OperationalExecutionSnapshot | None], OperationalExecutionSnapshot]
)


class ExecutionCoordinator:
    """Scenario-scoped owner of one validated execution snapshot."""

    def __init__(
        self,
        scenario_id: str | OperationalExecutionSnapshot | None = None,
        initial_snapshot: OperationalExecutionSnapshot | None = None,
        *,
        snapshot: OperationalExecutionSnapshot | None = None,
        plans: PlanRepository | None = None,
        mission_controller: object | None = None,
        evidence_resolver: EvidenceResolver | None = None,
        rolling_interval_s: int = 450,
        audit_projection_factory: Callable[
            [OperationalExecutionSnapshot], TrackingPlan | None
        ] | None = None,
    ) -> None:
        if isinstance(scenario_id, OperationalExecutionSnapshot):
            if initial_snapshot is not None or snapshot is not None:
                raise ValueError("initial execution snapshot was provided twice")
            initial_snapshot = scenario_id
            scenario_id = initial_snapshot.scenario_id
        if snapshot is not None:
            if initial_snapshot is not None:
                raise ValueError("initial execution snapshot was provided twice")
            initial_snapshot = snapshot
        if initial_snapshot is not None:
            scenario_id = scenario_id or initial_snapshot.scenario_id
        if not scenario_id:
            raise ValueError("scenario_id is required")
        if rolling_interval_s < 1:
            raise ValueError("rolling_interval_s must be positive")
        if initial_snapshot is not None and initial_snapshot.scenario_id != scenario_id:
            raise ValueError("execution snapshot scenario does not match coordinator")

        self._scenario_id = str(scenario_id)
        self._plans = plans
        self._mission_controller = mission_controller
        self._evidence_resolver = evidence_resolver
        self._rolling_interval_s = rolling_interval_s
        self._audit_projection_factory = (
            audit_projection_factory or tracking_plan_audit_projection
        )
        self._lock = RLock()
        self._current = _copy_snapshot(initial_snapshot)
        if self._current is None and plans is not None:
            loader = getattr(plans, "get_latest_execution_snapshot", None)
            if callable(loader):
                loaded = loader(self._scenario_id)
                if loaded is not None:
                    self._current = _copy_snapshot(loaded)
        self._last_rolling_check_s: int | None = None
        self._commit_counter = 0
        self._terminal_status: Literal["expired", "failed"] | None = None
        self._terminal_reason: str | None = None

    @property
    def scenario_id(self) -> str:
        return self._scenario_id

    @property
    def current(self) -> OperationalExecutionSnapshot | None:
        """Return a defensive copy of the currently validated snapshot."""

        with self._lock:
            return _copy_snapshot(self._current)

    @property
    def execution_revision(self) -> int:
        """Return zero before the first executable snapshot is installed."""

        current = self.current
        return current.execution_revision if current is not None else 0

    @property
    def is_executable(self) -> bool:
        """Whether a complete snapshot is available for physical execution."""

        with self._lock:
            return self._current is not None and self._terminal_status is None

    def active_mission_plan(
        self,
        *,
        sim_time_s: float | None = None,
        hard_stale_s: float | None = None,
    ) -> OperationalExecutionSnapshot | None:
        """Read the highest validated execution revision.

        The name is retained because the HTTP/runtime boundary historically
        exposed an ``active_mission_plan`` reader.  New callers receive the
        authoritative execution snapshot rather than a legacy audit plan.
        """

        with self._lock:
            persisted = None
            if self._plans is not None:
                loader = getattr(self._plans, "get_latest_execution_snapshot", None)
                if callable(loader):
                    persisted = loader(self._scenario_id)
            if persisted is not None and (
                self._current is None
                or persisted.execution_revision > self._current.execution_revision
            ):
                self._current = _copy_snapshot(persisted)
            snapshot = _copy_snapshot(self._current)
            if snapshot is None:
                return None
            if sim_time_s is not None or hard_stale_s is not None:
                if sim_time_s is None or hard_stale_s is None:
                    raise ValueError(
                        "sim_time_s and hard_stale_s are both required for health-gated reads"
                    )
                health = classify_execution_health(
                    snapshot,
                    sim_time_s=sim_time_s,
                    hard_stale_s=hard_stale_s,
                )
                if not health.executable:
                    return None
            return snapshot

    def executable_mission_plan(
        self,
        *,
        sim_time_s: float | None = None,
        hard_stale_s: float | None = None,
    ):
        """Return a controller-compatible UUV-only projection of ``current``."""

        snapshot = self.active_mission_plan(
            sim_time_s=sim_time_s,
            hard_stale_s=hard_stale_s,
        )
        if snapshot is None or not self.is_executable:
            return None
        from underwater_tracking.runtime.mission_controller import (
            execution_snapshot_to_mission_plan,
        )

        return execution_snapshot_to_mission_plan(snapshot)

    def propose(
        self,
        candidate: SnapshotCandidate,
        *,
        base_execution_revision: int | None = None,
        execution_revision: int | None = None,
    ) -> OperationalExecutionSnapshot:
        """Normalize a detached, immutable candidate without changing state."""

        current = self.current
        if callable(candidate):
            candidate = candidate(current)
        proposed = (
            candidate
            if isinstance(candidate, OperationalExecutionSnapshot)
            else OperationalExecutionSnapshot.model_validate(candidate)
        )
        if proposed.scenario_id != self._scenario_id:
            raise ValueError("execution candidate scenario does not match coordinator")

        if current is None:
            default_base = None
            default_revision = 1
        else:
            default_base = current.execution_revision
            default_revision = current.execution_revision + 1
        selected_base = (
            base_execution_revision
            if base_execution_revision is not None
            else proposed.base_execution_revision
        )
        if (
            base_execution_revision is None
            and proposed.base_execution_revision is None
        ):
            selected_base = default_base
        selected_revision = execution_revision or proposed.execution_revision
        if execution_revision is None and proposed.execution_revision <= (
            current.execution_revision if current is not None else 0
        ):
            selected_revision = default_revision
        if current is None and selected_revision != 1:
            raise ValueError("the first execution revision must be one")
        if selected_base is not None and selected_revision <= selected_base:
            raise ValueError("execution revision must follow its base revision")
        return proposed.model_copy(
            deep=True,
            update={
                "base_execution_revision": selected_base,
                "execution_revision": selected_revision,
            },
        )

    def commit(
        self,
        candidate: OperationalExecutionSnapshot | Mapping[str, Any],
        *,
        allow_rebase: bool = False,
        evidence_valid: bool | None = None,
        apply: SnapshotApplier | None = None,
        audit_projection: TrackingPlan | None = None,
    ) -> ExecutionCommitResult:
        """Compare-and-set one candidate and preserve the active plan on error."""

        try:
            proposed = (
                candidate
                if isinstance(candidate, OperationalExecutionSnapshot)
                else OperationalExecutionSnapshot.model_validate(candidate)
            )
        except (TypeError, ValueError) as exc:
            return self.preserve(f"candidate_invalid:{type(exc).__name__}")

        if proposed.scenario_id != self._scenario_id:
            return self.preserve("candidate_scenario_mismatch")

        with self._lock:
            current = self._current
            candidate_base = proposed.base_execution_revision
            was_rebased = False
            if current is None:
                if candidate_base not in (None, 0) or proposed.execution_revision != 1:
                    return self._stale_result(
                        proposed,
                        "initial_execution_revision_mismatch",
                    )
                staged = proposed.model_copy(
                    deep=True,
                    update={"base_execution_revision": None},
                )
            elif candidate_base != current.execution_revision:
                if not allow_rebase:
                    return self._stale_result(
                        proposed,
                        "base_execution_revision_mismatch",
                    )
                reasons = self._rebase_reasons(current, proposed, evidence_valid)
                if reasons:
                    return self._stale_result(proposed, ";".join(reasons))
                staged = _controlled_rebase(current, proposed)
                was_rebased = True
            else:
                if proposed.execution_revision <= current.execution_revision:
                    return self._stale_result(proposed, "execution_revision_not_forward")
                staged = proposed.model_copy(deep=True)

            checkpoint = _controller_checkpoint(self._mission_controller)
            applier = apply or _controller_applier(self._mission_controller)
            if applier is not None:
                try:
                    applied = applier(staged)
                except Exception as exc:  # noqa: BLE001 - preserve active execution
                    _restore_controller(self._mission_controller, checkpoint)
                    return self._preserved_result(
                        staged,
                        f"apply_failed:{type(exc).__name__}:{str(exc)[:240]}",
                    )
                if applied is False:
                    _restore_controller(self._mission_controller, checkpoint)
                    return self._preserved_result(staged, "apply_rejected")

            self._commit_counter += 1
            audit = audit_projection or self._audit_projection_factory(staged)
            result = ExecutionCommitResult(
                commit_id=self._commit_id(staged, "committed"),
                scenario_id=self._scenario_id,
                status="committed",
                accepted=True,
                execution_revision=staged.execution_revision,
                candidate_execution_revision=proposed.execution_revision,
                base_execution_revision=staged.base_execution_revision,
                preserved_execution_revision=(
                    current.execution_revision if current is not None else None
                ),
                snapshot=staged,
                was_rebased=was_rebased,
                audit_plan_id=audit.plan_id if audit is not None else None,
            )
            self._current = _copy_snapshot(staged)
            try:
                self._persist(result, audit)
            except Exception as exc:  # noqa: BLE001 - restore physical and memory state
                self._current = _copy_snapshot(current)
                _restore_controller(self._mission_controller, checkpoint)
                return self._preserved_result(
                    staged,
                    f"persistence_failed:{type(exc).__name__}:{str(exc)[:240]}",
                )
            self._terminal_status = None
            self._terminal_reason = None
            return result

    def mark_failed(self, reason: str) -> ExecutionCommitResult:
        """Retain the audit snapshot while making execution non-dispatchable."""

        normalized = str(reason).strip() or "execution_planning_failed"
        with self._lock:
            self._terminal_status = "failed"
            self._terminal_reason = normalized
        return self.preserve(normalized)

    def mark_expired(self, reason: str) -> ExecutionCommitResult:
        """Retain the audit snapshot while making execution non-dispatchable."""

        normalized = str(reason).strip() or "execution_snapshot_expired"
        with self._lock:
            self._terminal_status = "expired"
            self._terminal_reason = normalized
        return self.preserve(normalized)

    def execution_health(
        self,
        *,
        sim_time_s: float,
        hard_stale_s: float,
    ) -> ExecutionHealth:
        """Return authoritative runtime health including planning failures."""

        with self._lock:
            terminal_status = self._terminal_status
            terminal_reason = self._terminal_reason
            current = _copy_snapshot(self._current)
        if terminal_status in {"expired", "failed"}:
            age_s = (
                max(0.0, sim_time_s - float(current.valid_from_s))
                if current is not None
                else 0.0
            )
            return ExecutionHealth(
                status=terminal_status,
                age_s=age_s,
                reason_codes=((terminal_reason,) if terminal_reason is not None else ()),
            )
        if current is None:
            return ExecutionHealth(
                status="failed",
                age_s=0.0,
                reason_codes=("execution_snapshot_missing",),
            )
        return classify_execution_health(
            current,
            sim_time_s=sim_time_s,
            hard_stale_s=hard_stale_s,
        )

    def commit_baseline_then_optimize(
        self,
        baseline: OperationalExecutionSnapshot,
        *,
        optimizer: SemanticOptimizer | None = None,
        publish: SnapshotPublisher | None = None,
        apply: SnapshotApplier | None = None,
        audit_projection: TrackingPlan | None = None,
    ) -> ExecutionCommitResult:
        """Commit and publish a deterministic baseline before optional LLM work."""

        if baseline.plan_source != "deterministic":
            return self._rejected_result(
                baseline,
                "baseline_plan_source_must_be_deterministic",
            )
        result = self.commit(
            baseline,
            apply=apply,
            audit_projection=audit_projection,
        )
        if not result.committed or result.snapshot is None:
            return result
        committed = result.snapshot
        if publish is not None:
            publish(committed)
        if optimizer is not None:
            Thread(
                target=self._run_semantic_optimization,
                args=(optimizer, committed, publish),
                name=f"execution-semantic-{committed.execution_revision}",
                daemon=True,
            ).start()
        return result

    def commit_semantic_optimization(
        self,
        candidate: OperationalExecutionSnapshot,
        *,
        base_execution_revision: int,
        publish: SnapshotPublisher | None = None,
        apply: SnapshotApplier | None = None,
    ) -> ExecutionCommitResult:
        """CAS one semantic-only revision against its deterministic baseline."""

        with self._lock:
            current = _copy_snapshot(self._current)
            if current is None or current.execution_revision != base_execution_revision:
                return self._rejected_result(candidate, "stale_execution_base")
            if candidate.plan_source != "llm_optimized":
                return self._rejected_result(
                    candidate,
                    "semantic_optimization_plan_source_invalid",
                )
            if _physical_execution_fingerprint(candidate) != _physical_execution_fingerprint(
                current
            ):
                return self._rejected_result(
                    candidate,
                    "semantic_optimization_changed_physical_fields",
                )

        staged = candidate.model_copy(
            deep=True,
            update={"base_execution_revision": base_execution_revision},
        )
        result = self.commit(staged, apply=apply)
        if result.committed and result.snapshot is not None and publish is not None:
            publish(result.snapshot)
        return result

    def _run_semantic_optimization(
        self,
        optimizer: SemanticOptimizer,
        baseline: OperationalExecutionSnapshot,
        publish: SnapshotPublisher | None,
    ) -> None:
        try:
            candidate = optimizer(baseline.model_copy(deep=True))
        except Exception:  # noqa: BLE001 - deterministic baseline remains active
            return
        self.commit_semantic_optimization(
            candidate,
            base_execution_revision=baseline.execution_revision,
            publish=publish,
        )

    def preserve(self, reason: str) -> ExecutionCommitResult:
        """Record a planning failure while retaining the current snapshot."""

        with self._lock:
            self._commit_counter += 1
            current = _copy_snapshot(self._current)
            result = ExecutionCommitResult(
                commit_id=self._commit_id(current, "preserved"),
                scenario_id=self._scenario_id,
                status="preserved",
                accepted=False,
                execution_revision=(current.execution_revision if current else None),
                preserved_execution_revision=(
                    current.execution_revision if current else None
                ),
                snapshot=current,
                reason=str(reason)[:2000],
                active_plan_preserved=current is not None,
            )
            self._persist(result, None)
            return result

    def rolling_check_due(self, sim_time_s: int) -> bool:
        """Return whether the deterministic rolling review is due."""

        if sim_time_s < 0:
            raise ValueError("sim_time_s must be non-negative")
        with self._lock:
            return self._last_rolling_check_s is None or (
                sim_time_s - self._last_rolling_check_s >= self._rolling_interval_s
            )

    def mark_rolling_check(self, sim_time_s: int) -> None:
        """Mark one completed rolling review without moving execution state."""

        if sim_time_s < 0:
            raise ValueError("sim_time_s must be non-negative")
        with self._lock:
            if (
                self._last_rolling_check_s is not None
                and sim_time_s < self._last_rolling_check_s
            ):
                raise ValueError("rolling check time cannot move backwards")
            self._last_rolling_check_s = sim_time_s

    def prediction_leaves_chain(self, predicted_region_ids: Sequence[object]) -> bool:
        """Return whether the forecast no longer covers the current task slot."""

        current = self.current
        if current is None:
            return False
        ids = {
            str(item.region_id if hasattr(item, "region_id") else item)
            for item in predicted_region_ids
        }
        return current.current_region_id not in ids

    def replan_required_for_prediction(self, predicted_region_ids: Sequence[object]) -> bool:
        """Descriptive alias used by the background planning loop."""

        return self.prediction_leaves_chain(predicted_region_ids)

    def _rebase_reasons(
        self,
        current: OperationalExecutionSnapshot,
        candidate: OperationalExecutionSnapshot,
        evidence_valid: bool | None,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if candidate.target_id != current.target_id:
            reasons.append("target_revision_changed")
        if (
            candidate.target_track.track_revision != current.target_track.track_revision
            or candidate.prediction.source_track_revision
            != current.prediction.source_track_revision
        ):
            reasons.append("target_revision_changed")
        if _resource_fingerprint(candidate) != _resource_fingerprint(current):
            reasons.append("resource_revision_changed")
        if candidate.expert_request_version != current.expert_request_version:
            reasons.append("manual_revision_changed")
        if candidate.source_snapshot_revision <= current.source_snapshot_revision:
            reasons.append("physical_revision_not_advanced")
        if evidence_valid is False or (
            evidence_valid is None and not self._evidence_is_valid(candidate.evidence_ids)
        ):
            reasons.append("evidence_invalid")
        return tuple(dict.fromkeys(reasons))

    def _evidence_is_valid(self, evidence_ids: Sequence[str]) -> bool:
        if self._evidence_resolver is None:
            return True
        try:
            return all(bool(self._evidence_resolver(evidence_id)) for evidence_id in evidence_ids)
        except Exception:  # noqa: BLE001 - evidence failure must block rebase
            return False

    def _stale_result(
        self,
        candidate: OperationalExecutionSnapshot,
        reason: str,
    ) -> ExecutionCommitResult:
        current = _copy_snapshot(self._current)
        self._commit_counter += 1
        result = ExecutionCommitResult(
            commit_id=self._commit_id(candidate, "stale"),
            scenario_id=self._scenario_id,
            status="stale",
            accepted=False,
            candidate_execution_revision=candidate.execution_revision,
            base_execution_revision=candidate.base_execution_revision,
            execution_revision=(current.execution_revision if current else None),
            preserved_execution_revision=(current.execution_revision if current else None),
            snapshot=current,
            reason=reason[:2000],
            active_plan_preserved=current is not None,
        )
        self._persist(result, None)
        return result

    def _preserved_result(
        self,
        candidate: OperationalExecutionSnapshot,
        reason: str,
    ) -> ExecutionCommitResult:
        current = _copy_snapshot(self._current)
        self._commit_counter += 1
        result = ExecutionCommitResult(
            commit_id=self._commit_id(candidate, "preserved"),
            scenario_id=self._scenario_id,
            status="preserved",
            accepted=False,
            candidate_execution_revision=candidate.execution_revision,
            base_execution_revision=candidate.base_execution_revision,
            execution_revision=(current.execution_revision if current else None),
            preserved_execution_revision=(current.execution_revision if current else None),
            snapshot=current,
            reason=reason[:2000],
            active_plan_preserved=current is not None,
        )
        self._persist(result, None)
        return result

    def _rejected_result(
        self,
        candidate: OperationalExecutionSnapshot,
        reason: str,
    ) -> ExecutionCommitResult:
        with self._lock:
            current = _copy_snapshot(self._current)
            self._commit_counter += 1
            result = ExecutionCommitResult(
                commit_id=self._commit_id(candidate, "rejected"),
                scenario_id=self._scenario_id,
                status="rejected",
                accepted=False,
                candidate_execution_revision=candidate.execution_revision,
                base_execution_revision=candidate.base_execution_revision,
                execution_revision=(
                    current.execution_revision if current is not None else None
                ),
                preserved_execution_revision=(
                    current.execution_revision if current is not None else None
                ),
                snapshot=current,
                reason=reason,
                active_plan_preserved=current is not None,
            )
            self._persist(result, None)
            return result

    def _commit_id(
        self,
        snapshot: OperationalExecutionSnapshot | None,
        status: str,
    ) -> str:
        payload = (
            snapshot.model_dump(mode="json") if snapshot is not None else {"none": True}
        )
        digest = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return f"{self._scenario_id}:execution:{status}:{self._commit_counter}:{digest}"

    def _persist(
        self,
        result: ExecutionCommitResult,
        audit_projection: TrackingPlan | None,
    ) -> None:
        if self._plans is None:
            return
        saver = getattr(self._plans, "save_execution_commit", None)
        if not callable(saver):
            return
        saver(
            result=result,
            audit_projection=audit_projection if result.status == "committed" else None,
        )


def _copy_snapshot(
    snapshot: OperationalExecutionSnapshot | None,
) -> OperationalExecutionSnapshot | None:
    return snapshot.model_copy(deep=True) if snapshot is not None else None


def _resource_fingerprint(snapshot: OperationalExecutionSnapshot) -> tuple[object, ...]:
    groups = tuple(
        (
            group.task_group_id,
            tuple(sorted(group.member_uuv_ids)),
        )
        for group in snapshot.task_groups
    )
    reserves = tuple(
        (reserve.uuv_id, reserve.status, reserve.resource_episode)
        for reserve in snapshot.reserve_uuvs
    )
    return (*groups, *reserves)


def _physical_execution_fingerprint(
    snapshot: OperationalExecutionSnapshot,
) -> tuple[object, ...]:
    regions = tuple(
        (
            region.region_id,
            region.target_id,
            region.slot_index,
            region.prediction_id,
            region.geometry,
            region.centerline_indices,
            region.start_s,
            region.end_s,
            region.geometry_revision,
            region.predecessor_region_id,
            region.successor_region_id,
            region.handoff_start_s,
            region.handoff_end_s,
        )
        for region in snapshot.regions
    )
    groups = tuple(
        (
            group.task_group_id,
            group.target_id,
            group.region_id,
            group.member_uuv_ids,
            group.active_verifier_uuv_id,
            group.passive_tracker_uuv_id,
        )
        for group in snapshot.task_groups
    )
    reserves = tuple(
        (reserve.uuv_id, reserve.status, reserve.priority, reserve.resource_episode)
        for reserve in snapshot.reserve_uuvs
    )
    return (
        snapshot.target_id,
        snapshot.source_snapshot_revision,
        snapshot.source_sim_time_s,
        snapshot.prediction_revision,
        snapshot.prediction_id,
        snapshot.valid_from_s,
        snapshot.valid_until_s,
        snapshot.target_track,
        snapshot.prediction,
        regions,
        groups,
        reserves,
        snapshot.current_region_id,
        snapshot.next_region_id,
    )


def _controlled_rebase(
    current: OperationalExecutionSnapshot,
    candidate: OperationalExecutionSnapshot,
) -> OperationalExecutionSnapshot:
    revision = max(current.execution_revision + 1, candidate.execution_revision)
    regions = tuple(
        region.model_copy(update={"execution_revision": revision})
        for region in candidate.regions
    )
    groups = tuple(
        group.model_copy(update={"execution_revision": revision})
        for group in candidate.task_groups
    )
    return candidate.model_copy(
        deep=True,
        update={
            "execution_revision": revision,
            "base_execution_revision": current.execution_revision,
            "regions": regions,
            "task_groups": groups,
        },
    )


def _controller_applier(controller: object | None) -> SnapshotApplier | None:
    if controller is None:
        return None
    method = getattr(controller, "apply_execution_snapshot", None)
    return method if callable(method) else None


def _controller_checkpoint(controller: object | None) -> object | None:
    if controller is None:
        return None
    method = getattr(controller, "checkpoint", None)
    return method() if callable(method) else None


def _restore_controller(controller: object | None, checkpoint: object | None) -> None:
    if controller is None or checkpoint is None:
        return
    method = getattr(controller, "restore", None)
    if callable(method):
        method(checkpoint)


def tracking_plan_audit_projection(
    snapshot: OperationalExecutionSnapshot,
) -> TrackingPlan:
    """Project an execution snapshot into the legacy audit-only plan shape."""

    members_by_target: dict[str, tuple[str, ...]] = {
        snapshot.target_id: tuple(
            sorted(
                {
                    member
                    for group in snapshot.task_groups
                    for member in group.member_uuv_ids
                }
            )
        )
    }
    roles: dict[str, str] = {}
    for group in snapshot.task_groups:
        roles[group.active_verifier_uuv_id] = "active_verifier"
        roles[group.passive_tracker_uuv_id] = "passive_tracker"
    active_ids = tuple(sorted(roles))
    reserve_ids = tuple(sorted(reserve.uuv_id for reserve in snapshot.reserve_uuvs))
    return TrackingPlan(
        plan_id=f"{snapshot.scenario_id}:execution:{snapshot.execution_revision}",
        scenario_id=snapshot.scenario_id,
        revision=snapshot.execution_revision,
        base_snapshot_revision=snapshot.source_snapshot_revision,
        status="active",
        valid_from_s=int(snapshot.valid_from_s),
        valid_until_s=int(snapshot.valid_until_s),
        concept="hold_current",
        target_priorities={snapshot.target_id: 1.0},
        required_quality={snapshot.target_id: 0.0},
        member_ids_by_target=members_by_target,
        roles_by_member=roles,
        intent_refs={snapshot.target_id: f"intent:{snapshot.intent.intent_revision}"},
        prediction_refs={snapshot.target_id: snapshot.prediction_id},
        active_uuv_ids=active_ids,
        standby_uuv_ids=reserve_ids,
        predicted_quality={snapshot.target_id: 0.0},
        predicted_active_count=len(active_ids),
        evidence_ids=snapshot.evidence_ids,
    )


__all__ = [
    "ExecutionCommitResult",
    "ExecutionCoordinator",
    "tracking_plan_audit_projection",
]
