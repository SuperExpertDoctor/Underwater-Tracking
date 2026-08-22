"""Single-worker planning epoch coordination and event retry mailbox."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Callable

from pydantic import ConfigDict, Field

from underwater_tracking.domain.models import SituationSnapshot, StrictModel
from underwater_tracking.domain.planning_epoch_models import (
    EpochCommitResult,
    PlanningEpoch,
    PlanningEpochCapture,
)
from underwater_tracking.persistence.planning_epochs import PlanningEpochRepository
from underwater_tracking.persistence.sqlite import json_dumps, now_ms
from underwater_tracking.runtime.mission_controller import MissionSnapshot


@dataclass(frozen=True, slots=True)
class EpochTrigger:
    event_id: str
    event_type: str
    sim_time_s: int
    priority: int
    entity_id: str | None = None
    resource_episode: int | None = None


class PlanningEpochHealth(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str = "idle"
    epoch_id: str | None = None
    queued_event_count: int = Field(default=0, ge=0)
    started_at_ms: int | None = Field(default=None, ge=0)
    last_result_status: str | None = None
    last_error: str | None = None
    retry_attempt: int = Field(default=0, ge=0)
    retry_not_before_utc_ms: int | None = Field(default=None, ge=0)
    dead_letter_event_ids: tuple[str, ...] = ()
    dead_letter_reasons: dict[str, str] = Field(default_factory=dict)
    base_physics_revision: int | None = Field(default=None, ge=0)
    latest_physics_revision: int | None = Field(default=None, ge=0)
    base_sim_time_s: int | None = Field(default=None, ge=0)
    latest_sim_time_s: int | None = Field(default=None, ge=0)
    data_age_s: int | None = Field(default=None, ge=0)


_RETRY_DELAYS_MS = (5_000, 15_000, 45_000)
_TRANSIENT_FAILURES = frozenset({"timeout", "provider"})


class PlanningEpochCoordinator:
    """Own scheduling state for one scenario; the caller owns the worker."""

    def __init__(
        self,
        scenario_id: str,
        repository: PlanningEpochRepository | None = None,
        *,
        database_path: str | None = None,
        utc_now_ms: Callable[[], int] = now_ms,
    ) -> None:
        if repository is None:
            if database_path is None:
                raise ValueError("repository or database_path is required")
            repository = PlanningEpochRepository(database_path)
            self._owns_repository = True
        else:
            self._owns_repository = False
        self._scenario_id = scenario_id
        self._repository = repository
        self._utc_now_ms = utc_now_ms
        self._lock = RLock()
        self._latest: SituationSnapshot | None = None
        self._events: dict[str, EpochTrigger] = {}
        self._running_epoch_id: str | None = None
        self._reserved_epoch_id: str | None = None
        self._closed = False
        self._started_at_ms: int | None = None
        self._last_result_status: str | None = None
        self._last_error: str | None = None
        retries = repository.event_retries(scenario_id)
        self._retries = retries
        self._dead_letter = {
            event_id for event_id, item in retries.items() if item["status"] == "dead_letter"
        }

    def observe(self, situation: SituationSnapshot) -> None:
        with self._lock:
            self._ensure_open()
            if situation.scenario_id != self._scenario_id:
                raise ValueError("situation scenario does not match coordinator")
            self._latest = situation

    def request(self, triggers: tuple[EpochTrigger, ...]) -> None:
        with self._lock:
            self._ensure_open()
            for trigger in triggers:
                if not trigger.event_id or trigger.event_id in self._dead_letter:
                    continue
                existing = self._events.get(trigger.event_id)
                if existing is None or (trigger.priority, trigger.sim_time_s) > (
                    existing.priority,
                    existing.sim_time_s,
                ):
                    self._events[trigger.event_id] = trigger

    def retry_event(self, event_id: str) -> None:
        """Requeue one dead-letter event after an explicit expert decision."""
        with self._lock:
            self._ensure_open()
            if event_id not in self._dead_letter:
                raise ValueError(f"event {event_id!r} is not dead-lettered")
            item = self._retries.get(event_id)
            payload = item.get("payload") if item is not None else None
            if not isinstance(payload, dict):
                raise ValueError(f"dead-letter event {event_id!r} has no trigger payload")
            try:
                trigger = EpochTrigger(
                    event_id=event_id,
                    event_type=str(payload["event_type"]),
                    sim_time_s=int(payload["sim_time_s"]),
                    priority=int(payload["priority"]),
                    entity_id=(
                        str(payload["entity_id"])
                        if payload.get("entity_id") is not None
                        else None
                    ),
                    resource_episode=(
                        int(payload["resource_episode"])
                        if payload.get("resource_episode") is not None
                        else None
                    ),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"dead-letter event {event_id!r} has an invalid trigger payload"
                ) from exc
            self._dead_letter.remove(event_id)
            self._events[event_id] = trigger
            attempt = _int_value(item.get("attempt", 0)) if item is not None else 0
            retry_payload = dict(payload)
            retry_payload["expert_retry"] = True
            self._retries[event_id] = {
                "attempt": attempt,
                "retry_not_before_utc_ms": None,
                "status": "retry_wait",
                "payload": retry_payload,
            }
            self._repository.save_event_retry(
                scenario_id=self._scenario_id,
                event_id=event_id,
                attempt=attempt,
                retry_not_before_utc_ms=None,
                status="retry_wait",
                payload=retry_payload,
            )

    def retry_dead_letter_event(self, event_id: str) -> None:
        """Named API for the operator/expert dead-letter retry action."""
        self.retry_event(event_id)

    def next_epoch(self, mission: MissionSnapshot) -> PlanningEpochCapture | None:
        with self._lock:
            self._ensure_open()
            if self._running_epoch_id is not None or self._reserved_epoch_id is not None:
                return None
            situation = self._latest
            if situation is None:
                return None
            now = self._utc_now_ms()
            eligible = tuple(
                sorted(
                    (
                        trigger
                        for event_id, trigger in self._events.items()
                        if event_id not in self._dead_letter
                        and self._retry_due(event_id, now)
                    ),
                    key=lambda trigger: (-trigger.priority, trigger.sim_time_s, trigger.event_id),
                )
            )
            if not eligible:
                return None
            event_ids = tuple(trigger.event_id for trigger in eligible)
            attempt = max(
                (
                    _int_value(self._retries[event_id]["attempt"])
                    for event_id in event_ids
                    if event_id in self._retries
                ),
                default=0,
            ) + 1
            prior_ids = tuple(
                sorted(str(getattr(prior, "prior_id")) for prior in getattr(situation, "target_search_priors", ()))
            )
            estimate_ids = tuple(
                sorted(str(getattr(estimate, "estimate_id", "")) for estimate in getattr(situation, "target_estimates", ()))
            )
            resource_payload = [
                (uuv_id, resource.model_dump(mode="json"))
                for uuv_id, resource in sorted(mission.uuv_resources.items())
            ]
            manifest_hash = hashlib.sha256(json_dumps(resource_payload).encode("utf-8")).hexdigest()
            event_hash = hashlib.sha256(json_dumps(event_ids).encode("utf-8")).hexdigest()[:12]
            epoch_id = f"epoch:{self._scenario_id}:{situation.snapshot_revision}:{event_hash}:a{attempt}"
            capture = PlanningEpochCapture(
                epoch=PlanningEpoch(
                    epoch_id=epoch_id,
                    scenario_id=self._scenario_id,
                    base_physics_revision=situation.snapshot_revision,
                    base_sim_time_s=situation.sim_time_s,
                    observation_batch_id=f"observation:{self._scenario_id}:{situation.snapshot_revision}",
                    critical_event_ids=event_ids,
                    public_target_prior_ids=prior_ids,
                    public_target_estimate_ids=estimate_ids,
                    resource_manifest_hash=manifest_hash,
                    active_plan_version=mission.plan_revision,
                ),
                situation=situation,
                mission=mission,
            )
            self._repository.create(capture)
            self._reserved_epoch_id = epoch_id
            self._started_at_ms = self._utc_now_ms()
            return capture

    def mark_running(self, epoch_id: str) -> None:
        with self._lock:
            self._ensure_open()
            if self._reserved_epoch_id != epoch_id:
                raise ValueError(f"epoch {epoch_id!r} is not reserved by this coordinator")
            self._repository.mark_running(epoch_id)
            self._running_epoch_id = epoch_id

    def finish(self, result: EpochCommitResult) -> None:
        with self._lock:
            self._ensure_open()
            if result.epoch_id not in {self._reserved_epoch_id, self._running_epoch_id}:
                raise ValueError(f"epoch {result.epoch_id!r} is not active")
            capture = self._repository.get_capture(result.epoch_id)
            self._repository.finish(result)
            event_ids = result.consumed_event_ids or capture.epoch.critical_event_ids
            if result.status == "failed":
                self._record_failure(capture.epoch, result, event_ids)
            else:
                for event_id in event_ids:
                    self._events.pop(event_id, None)
                    self._retries.pop(event_id, None)
                    self._repository.clear_event_retry(self._scenario_id, event_id)
            self._last_result_status = result.status
            self._last_error = result.failure_message or result.invalidated_reason
            self._running_epoch_id = None
            self._reserved_epoch_id = None

    def latest_situation(self) -> SituationSnapshot | None:
        with self._lock:
            return self._latest

    def health(self) -> PlanningEpochHealth:
        with self._lock:
            now = self._utc_now_ms()
            latest = self._latest
            active_epoch_id = self._running_epoch_id or self._reserved_epoch_id
            base_physics_revision: int | None = None
            base_sim_time_s: int | None = None
            if active_epoch_id is not None:
                capture = self._repository.get_capture(active_epoch_id)
                base_physics_revision = capture.epoch.base_physics_revision
                base_sim_time_s = capture.epoch.base_sim_time_s
            retry_items = [
                item for item in self._retries.values() if item["status"] != "dead_letter"
            ]
            dead_letter_reasons = {
                event_id: _dead_letter_reason(item)
                for event_id, item in self._retries.items()
                if item["status"] == "dead_letter"
            }
            retry_item = min(
                retry_items,
                key=lambda item: _int_value(item["retry_not_before_utc_ms"] or 0),
                default=None,
            )
            if self._running_epoch_id is not None:
                status = "running"
            elif any(self._retry_due(event_id, now) for event_id in self._events):
                status = "queued"
            elif self._last_result_status in {"failed", "invalidated", "rejected"}:
                status = "degraded"
            elif self._last_result_status is not None:
                status = self._last_result_status
            else:
                status = "idle"
            retry_attempt = _int_value(retry_item["attempt"]) if retry_item else 0
            retry_not_before = (
                _int_value(retry_item["retry_not_before_utc_ms"])
                if retry_item and retry_item["retry_not_before_utc_ms"] is not None
                else None
            )
            return PlanningEpochHealth(
                status=status,
                epoch_id=self._running_epoch_id or self._reserved_epoch_id,
                queued_event_count=sum(event_id not in self._dead_letter for event_id in self._events),
                started_at_ms=self._started_at_ms,
                last_result_status=self._last_result_status,
                last_error=self._last_error,
                retry_attempt=retry_attempt,
                retry_not_before_utc_ms=retry_not_before,
                dead_letter_event_ids=tuple(sorted(self._dead_letter)),
                dead_letter_reasons=dead_letter_reasons,
                base_physics_revision=base_physics_revision,
                latest_physics_revision=(latest.snapshot_revision if latest is not None else None),
                base_sim_time_s=base_sim_time_s,
                latest_sim_time_s=(latest.sim_time_s if latest is not None else None),
                data_age_s=(
                    max(0, latest.sim_time_s - base_sim_time_s)
                    if latest is not None and base_sim_time_s is not None
                    else None
                ),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_repository:
                self._repository.close()

    def _retry_due(self, event_id: str, now_ms_value: int) -> bool:
        item = self._retries.get(event_id)
        if item is None:
            return True
        if item["status"] == "dead_letter":
            return False
        retry_at = item["retry_not_before_utc_ms"]
        return retry_at is None or _int_value(retry_at) <= now_ms_value

    def _record_failure(
        self,
        epoch: PlanningEpoch,
        result: EpochCommitResult,
        event_ids: tuple[str, ...],
    ) -> None:
        for event_id in event_ids:
            previous = self._retries.get(event_id)
            attempt = _int_value(previous["attempt"]) + 1 if previous is not None else 1
            transient = result.failure_category in _TRANSIENT_FAILURES
            trigger = self._events.get(event_id)
            trigger_payload: dict[str, object] = (
                asdict(trigger) if trigger is not None else {"event_id": event_id}
            )
            trigger_payload["failure_category"] = result.failure_category or "internal"
            trigger_payload["failure_message"] = result.failure_message or "unknown failure"
            if not transient or attempt >= len(_RETRY_DELAYS_MS):
                status = "dead_letter"
                retry_at = None
                self._dead_letter.add(event_id)
                self._events.pop(event_id, None)
            else:
                status = "retry_wait"
                retry_at = self._utc_now_ms() + _RETRY_DELAYS_MS[attempt - 1]
            item: dict[str, object] = {
                "attempt": attempt,
                "retry_not_before_utc_ms": retry_at,
                "status": status,
                "payload": trigger_payload,
            }
            self._retries[event_id] = item
            self._repository.save_event_retry(
                scenario_id=self._scenario_id,
                event_id=event_id,
                attempt=attempt,
                retry_not_before_utc_ms=retry_at,
                status=status,
                payload=trigger_payload,
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("planning epoch coordinator is closed")


def _int_value(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected integer retry value, got {type(value).__name__}")
    return value


def _dead_letter_reason(item: dict[str, object]) -> str:
    payload = item.get("payload")
    if not isinstance(payload, dict):
        return "unknown"
    reason = payload.get("failure_message") or payload.get("failure_category")
    return str(reason or "unknown")
