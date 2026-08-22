"""Atomic physical-observation transitions shared by live runtime components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.runtime.mission_controller import MissionController, MissionSnapshot
from underwater_tracking.runtime.scenario_transition import ScenarioTransitionCoordinator


@dataclass(frozen=True, slots=True)
class PhysicalObservationBatch:
    """One immutable physics-to-mission observation boundary."""

    physics_revision: int
    sim_time_s: int
    observations: Mapping[str, object] = field(default_factory=dict)
    deployed_uuv_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.physics_revision < 0:
            raise ValueError("physics_revision must be non-negative")
        if self.sim_time_s < 0:
            raise ValueError("sim_time_s must be non-negative")
        if len(self.deployed_uuv_ids) != len(set(self.deployed_uuv_ids)):
            raise ValueError("deployed_uuv_ids must be unique")
        object.__setattr__(self, "observations", MappingProxyType(dict(self.observations)))

    def as_observations(self) -> dict[str, object]:
        """Return a mutable adapter payload for MissionController.advance."""
        payload = dict(self.observations)
        if self.deployed_uuv_ids and "deployed_uuv_ids" not in payload:
            payload["deployed_uuv_ids"] = self.deployed_uuv_ids
        return payload


@dataclass(frozen=True, slots=True)
class CommittedStateBundle:
    """The only state bundle a publisher may consume after a boundary."""

    physics_revision: int
    mission_revision: int
    situation: SituationSnapshot
    mission: MissionSnapshot | None = None


class ObservationBoundaryCommitter:
    """Apply one physical observation atomically under a scenario transition lock.

    The class accepts explicit callables for small deterministic tests, while
    the live composition path can provide an engine and mission controller and
    let the committer discover their checkpoint/restore methods.
    """

    def __init__(
        self,
        transitions: ScenarioTransitionCoordinator,
        *,
        engine: Any | None = None,
        mission_controller: MissionController | None = None,
        apply_delta: Callable[[PhysicalObservationBatch], object] | None = None,
        reconcile: Callable[[], object] | None = None,
        situation_provider: Callable[[], SituationSnapshot] | None = None,
        mission_snapshot_provider: Callable[[], MissionSnapshot | None] | None = None,
    ) -> None:
        self._transitions = transitions
        self._engine = engine
        self._mission_controller = mission_controller
        self._apply_delta = apply_delta
        self._reconcile = reconcile
        self._situation_provider = situation_provider or self._engine_situation_provider()
        self._mission_snapshot_provider = (
            mission_snapshot_provider
            or self._engine_mission_snapshot_provider()
        )
        self._last_physics_revision = -1

    def commit(self, delta: PhysicalObservationBatch) -> CommittedStateBundle:
        """Commit one observation and return one consistent public bundle."""
        if delta.physics_revision <= self._last_physics_revision:
            raise ValueError("physical observation revision must increase")
        with self._transitions.transition("observation"):
            engine_checkpoint = self._checkpoint_engine()
            mission_checkpoint = (
                self._mission_controller.checkpoint()
                if self._mission_controller is not None
                else None
            )
            try:
                self._apply(delta)
                if self._reconcile is not None:
                    self._reconcile()
                situation = self._situation_provider()
                mission = (
                    self._mission_snapshot_provider()
                    if self._mission_snapshot_provider is not None
                    else None
                )
                self._last_physics_revision = delta.physics_revision
                return CommittedStateBundle(
                    physics_revision=delta.physics_revision,
                    mission_revision=mission.plan_revision if mission is not None else 0,
                    situation=situation,
                    mission=mission,
                )
            except BaseException:
                self._restore_engine(engine_checkpoint)
                if (
                    self._mission_controller is not None
                    and mission_checkpoint is not None
                ):
                    self._mission_controller.restore(mission_checkpoint)
                raise

    def _apply(self, delta: PhysicalObservationBatch) -> None:
        if self._apply_delta is not None:
            self._apply_delta(delta)
            return
        if self._engine is not None:
            apply_batch = getattr(self._engine, "apply_observation_batch", None)
            if callable(apply_batch):
                apply_batch(delta)
                return
        if self._mission_controller is None:
            raise RuntimeError("observation boundary has no apply operation")
        self._mission_controller.advance(delta.sim_time_s, delta.as_observations())

    def _checkpoint_engine(self) -> object | None:
        if self._engine is None:
            return None
        checkpoint = getattr(self._engine, "checkpoint", None)
        if callable(checkpoint):
            return cast(object | None, checkpoint())
        checkpoint = getattr(self._engine, "_checkpoint_explicit_platform_core", None)
        return cast(object | None, checkpoint()) if callable(checkpoint) else None

    def _restore_engine(self, checkpoint: object | None) -> None:
        if checkpoint is None or self._engine is None:
            return
        restore = getattr(self._engine, "restore", None)
        if callable(restore):
            restore(checkpoint)
            return
        restore = getattr(self._engine, "_restore_explicit_platform_core", None)
        if callable(restore):
            restore(checkpoint)

    def _engine_situation_provider(self) -> Callable[[], SituationSnapshot]:
        if self._engine is not None:
            provider = getattr(self._engine, "publication_situation", None)
            if callable(provider):
                return cast(Callable[[], SituationSnapshot], provider)
        raise ValueError("situation_provider is required without an engine")

    def _engine_mission_snapshot_provider(
        self,
    ) -> Callable[[], MissionSnapshot | None] | None:
        if self._engine is not None:
            provider = getattr(self._engine, "mission_snapshot", None)
            if callable(provider):
                return cast(Callable[[], MissionSnapshot | None], provider)
        if self._mission_controller is not None:
            return self._mission_controller.snapshot
        return None


__all__ = [
    "CommittedStateBundle",
    "ObservationBoundaryCommitter",
    "PhysicalObservationBatch",
]
