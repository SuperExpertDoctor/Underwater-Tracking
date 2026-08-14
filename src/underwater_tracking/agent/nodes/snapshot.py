# src/underwater_tracking/agent/nodes/snapshot.py
"""Immutable planning snapshot node (plan Task 7, spec 15.1).

``SnapshotNode`` assembles one self-contained :class:`PlanningSnapshot` from
the live situation — group reports and resources from the stored
``SituationSnapshot`` (spec 8.4: raw high-frequency state lives outside the
checkpoint), the pending trigger events, the currently broadcast plan, and
the applied expert directives — and stores it under a deterministic
``scenario_id:snapshot:revision`` reference. The stored snapshot is deeply
copied, so later mutation of the provider's live object cannot change it;
every downstream node (optimize, commit) reads exactly the same immutable
input. ``build_planning_snapshot`` and ``snapshot_hash`` are the pure
building blocks; the node only wires providers and the store.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass

from underwater_tracking.agent.prompts import canonical_digest
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import ExpertDirective, TrackingPlan
from underwater_tracking.domain.models import SituationSnapshot


@dataclass(frozen=True)
class PlanningSnapshot:
    """Immutable planning input: situation, broadcast plan, applied directives.

    ``situation`` is the deep-copied live world state (revision, sim time,
    UUV resources, group reports, pending events); ``active_plan`` is the
    currently broadcast (ACTIVE/DEGRADED) plan the new plan would succeed,
    or None on the first planning cycle; ``applied_directives`` are the
    expert directives with status ``applied`` (spec 10). The convenience
    properties mirror the situation's identity fields.
    """

    situation: SituationSnapshot
    active_plan: TrackingPlan | None
    applied_directives: tuple[ExpertDirective, ...]

    @property
    def scenario_id(self) -> str:
        return self.situation.scenario_id

    @property
    def snapshot_revision(self) -> int:
        return self.situation.snapshot_revision

    @property
    def sim_time_s(self) -> int:
        return self.situation.sim_time_s

    @property
    def digest(self) -> str:
        """Canonical content hash of the immutable situation (spec 16)."""
        return snapshot_hash(self.situation)


def build_snapshot_ref(scenario_id: str, revision: int) -> str:
    """Deterministic storage reference of one immutable planning snapshot."""
    return f"{scenario_id}:snapshot:{revision}"


def snapshot_hash(situation: SituationSnapshot) -> str:
    """Canonical-JSON SHA-256 of the situation (sorted keys, stable digest)."""
    return canonical_digest(situation.model_dump(mode="json"))


def build_planning_snapshot(
    situation: SituationSnapshot,
    *,
    active_plan: TrackingPlan | None = None,
    applied_directives: tuple[ExpertDirective, ...] = (),
) -> PlanningSnapshot:
    """Assemble an immutable planning snapshot from the live inputs.

    The situation and the broadcast plan are deeply copied so the snapshot
    is immune to later mutation of the provider's objects.
    """
    return PlanningSnapshot(
        situation=situation.model_copy(deep=True),
        active_plan=active_plan.model_copy(deep=True) if active_plan is not None else None,
        applied_directives=tuple(applied_directives),
    )


def _no_active_plan(scenario_id: str) -> TrackingPlan | None:
    """Default active-plan provider: no broadcast plan exists."""
    del scenario_id
    return None


def _no_directives(scenario_id: str) -> tuple[ExpertDirective, ...]:
    """Default directives provider: no applied directives."""
    del scenario_id
    return ()


class SnapshotNode:
    """LangGraph node assembling and storing the immutable planning snapshot.

    Reads the live ``SituationSnapshot`` through ``snapshot_provider``,
    the broadcast plan through ``active_plan_provider``, and the applied
    expert directives through ``directives_provider``; stores the built
    :class:`PlanningSnapshot` in ``store`` under
    ``build_snapshot_ref`` and returns the state fragment
    ``{"snapshot_ref", "snapshot_revision"}``. Pure in the state input:
    the same state and providers produce the same snapshot.
    """

    def __init__(
        self,
        *,
        snapshot_provider: Callable[[str], SituationSnapshot],
        store: MutableMapping[str, PlanningSnapshot],
        active_plan_provider: Callable[[str], TrackingPlan | None] = _no_active_plan,
        directives_provider: Callable[[str], tuple[ExpertDirective, ...]] = _no_directives,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        self._active_plan_provider = active_plan_provider
        self._directives_provider = directives_provider
        self._store = store

    def __call__(self, state: CarrierState) -> CarrierState:
        scenario_id = state.get("scenario_id")
        if not scenario_id:
            raise ValueError("SnapshotNode requires scenario_id in state")
        ref = state.get("snapshot_ref")
        if ref is None:
            raise ValueError("SnapshotNode requires snapshot_ref in state")
        snapshot = build_planning_snapshot(
            self._snapshot_provider(ref),
            active_plan=self._active_plan_provider(scenario_id),
            applied_directives=self._directives_provider(scenario_id),
        )
        snapshot_ref = build_snapshot_ref(scenario_id, snapshot.snapshot_revision)
        self._store[snapshot_ref] = snapshot
        return {
            "snapshot_ref": snapshot_ref,
            "snapshot_revision": snapshot.snapshot_revision,
        }
