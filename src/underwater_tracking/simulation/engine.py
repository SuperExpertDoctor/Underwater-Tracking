# src/underwater_tracking/simulation/engine.py
"""Deterministic headless simulation engine with explicit multirate scheduling.

The engine runs the multirate schedule from the foundation plan:

* every ``physics_step_s`` (10 s) the world advances: UUVs steer toward
  their committed waypoints and burn energy, targets sample their hidden
  intent chain and move;
* every ``observation_step_s`` (30 s) the engine generates noisy bearing
  observations from each group's members and invokes the per-target group
  graph exactly once per target (stateful threads, checkpointed);
* every ``group_report_s`` (300 s) the engine publishes the latest group
  reports. Frames always carry the *latest known* reports (never only the
  freshly published ones), so consumers see them between publish instants
  too; the publish branch itself is observable through ``RuntimeEvent``
  entries in the frame.

Each ``step()`` advances the clock once and returns one operational frame:
a plain JSON-serializable dict with public UUV states, the latest group
reports, estimated tracks, quality, current assignments, runtime events,
and waypoint commands. Frames never contain truth fields. Truth is
delivered exclusively through the ``evaluation_sink`` callback, which
defaults to a no-op. Every frame is also appended to the run's JSONL log
(``FrameLogger``); when ``output_dir`` is omitted the engine picks a
run-scoped directory under ``outputs/``.

Agent integration is additive: when a ``carrier`` hook is injected, the
engine hands the latest ``SituationSnapshot`` to it at the end of every
observation cycle (group reports -> carrier runtime), and committed plan
commands enter through ``apply_plan_command`` and are applied to the group
manager at the next observation cycle (PlanCommand rows -> group manager
apply). ``fail_uuv`` marks a fleet UUV failed — it stops observing and
steering but stays in the fleet — which the situation reports to the
carrier as ``UUVStatus.FAILED``. Without the hook the headless path is
unchanged.

Determinism: every random draw descends from the constructor ``seed``
through fixed per-entity derivations (``random.Random(seed ^ stable_hash(id))``),
so no two entities share a draw stream and no entity's stream depends on
the order in which others are advanced. Identical seeds therefore produce
byte-identical normalized logs.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
from math import atan2, cos, hypot, pi, sin
from pathlib import Path
import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np

try:
    ExceptionGroup
except NameError:  # pragma: no cover - Python 3.10 compatibility
    from exceptiongroup import ExceptionGroup

from underwater_tracking.config.models import AppConfig
from underwater_tracking.config.platform_core import EnvironmentConfig, InitialPlatformConfig
from underwater_tracking.domain.agent_models import (
    PlanCommand,
    Segment,
    TrackingPlan,
    VerificationCommand,
)
from underwater_tracking.domain.adversary_models import (
    AdversaryDecisionRecord,
    AdversaryEscapeDecision,
    AdversaryEscapeInput,
    AdversaryKinematicLimits,
    AdversaryObservation,
    AdversaryOperationalSummary,
    AdversaryOperatingBoundary,
    CommunicationsAcousticExposure,
    PlatformThreatSummary,
    AdversaryTrigger,
)
from underwater_tracking.agent.nodes.adversary import AdversaryDecisionGate
from underwater_tracking.domain.models import (
    BearingObservation,
    CarrierState,
    Contact,
    ContactClassification,
    DeploymentState,
    EventLevel,
    GroupReport,
    IntelligenceReport,
    OperationalScheme,
    RuntimeEvent,
    SituationSnapshot,
    SurveillanceCapability,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.domain.mission_models import (
    ExecutableMissionPlan,
    RegionLifecycle,
    UUVMissionMode,
)
from underwater_tracking.domain.observations import PassiveSonarObservation
from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    CommunicationCapability,
    MotionLimits,
    PlatformCapability,
    PlatformRoster,
    PlatformSnapshot,
    SonarCapability,
    USVPlatformState,
    UUVPlatformState,
)
from underwater_tracking.domain.slave_models import (
    SlaveBeliefSummary,
    SlaveCommunicationLink,
    SlaveHandoffSegment,
    SlavePlatformCapability,
    SlaveSonarContext,
    SlaveSonarDecision,
)
from underwater_tracking.groups.manager import GroupManager
from underwater_tracking.groups.state import PlanCommand as GroupPlanCommand
from underwater_tracking.persistence.frame_log import FrameLogCheckpoint, FrameLogger
from underwater_tracking.planning.allocation import AllocationInput, allocate_groups
from underwater_tracking.planning.astar import AStarRoutePlanner, RoutePlan
from underwater_tracking.planning.carrier_tasks import CarrierTaskPlanner
from underwater_tracking.planning.reservations import ReservationRegistry
from underwater_tracking.planning.waypoints import plan_group_waypoints
from underwater_tracking.simulation.clock import SimulationClock
from underwater_tracking.simulation.connectivity import (
    ConnectivityNode,
    ConnectivitySnapshot,
    build_connectivity,
    has_path,
)
from underwater_tracking.simulation.carrier import CarrierEntity
from underwater_tracking.simulation.decoy import DecoyEntity
from underwater_tracking.simulation.formation_control import apply_formation_correction
from underwater_tracking.simulation.kinematics import MotionCommand, MotionState, wrap_angle
from underwater_tracking.simulation.observability import (
    InputFrame,
    ObservabilitySupervisor,
    load_observability_config,
)
from underwater_tracking.simulation.sonar import (
    SonarNode,
    default_pd_curve,
    make_passive_observation,
    make_passive_observations,
)
from underwater_tracking.simulation.target import HiddenIntent, TargetEntity
from underwater_tracking.runtime.mission_controller import MissionController, MissionSnapshot
from underwater_tracking.simulation.uuv import UUVEntity, wrap
from underwater_tracking.simulation.usv import USVEntity
from underwater_tracking.tracking.imm import DEFAULT_PROCESS_NOISE
from underwater_tracking.tracking.uif import UnscentedInformationFilter

_SCENARIO_ID = "underwater-default"

# Coarse position prior shared by every group runtime; triangulation from
# the first real observations dominates it from t=30 onward.
_COARSE_PRIOR: tuple[float, float] = (0.0, 0.0)

# Pre-report quality assumed at t=0 for the first allocation. Below the
# warning threshold, so the first allocation grows every group to three
# members (two targets x three observers, well within the 12-UUV fleet).
_INITIAL_QUALITY = 0.5

# Bearing sensor model of the simulated fleet. The variance is the
# engine's choice (5.7 deg std is a realistic acoustic bearing error at the
# operating standoffs); the quality normalization in config/models.py is
# calibrated to 1e-3 rad^2 at 1 km standoff, and the waypoint planner's
# FIM scoring scales multiplicatively with the variance, so the planned
# standoff geometry is unchanged by this value.
_BEARING_VARIANCE_RAD2 = 1e-2

# Near-field blind zone of the simulated bearing sensor (metres). An
# observer transiting across the target - a receding-horizon artifact, not
# a commanded standoff - sits inside the filter's sigma-point cloud, so
# the predicted bearing spread wraps past +/-pi and the accepted update
# corrupts the covariance into non-positive-definiteness. A real
# near-field sonar is equally unusable. Normal standoff geometry never
# brings an observer below ~300 m of the truth, so this fires only during
# anomalous crossings.
_SENSOR_MIN_RANGE_M = 250.0
_UUV_DEPLOY_RADIUS_M = 2000.0
_TARGET_SPAWN_SPAN_M = 800.0
_RECOVERY_RADIUS_M = 50.0

# Fleet kinematics (spec 5.1 amendment, R2): the configured UUV maximum
# speed and turn rate replace the old module constants. The submarine now
# cruises faster than the UUV (8 vs 4 m/s), so closing a drifting standoff
# ring is no longer an option: intent understanding and trajectory
# prediction — not raw pursuit speed — are what keep a track alive.

# Waypoint planning knobs matching the waypoint planner's own defaults.
# max_step_m is the receding-horizon *maneuver authority* used to pick the
# committed lattice waypoint, while UUVEntity.step caps actual motion at
# the configured uuv_max_speed_mps. The authority is bounded (900 m) so no
# observer is
# ever commanded across the scenario: the bearing-only filter needs
# well-spread observers every cycle, and long transits through a collinear
# wedge are exactly what lets a triangulated track lock a mirrored
# velocity and diverge.
_WAYPOINT_MAX_STEP_M = 900.0
_WAYPOINT_MIN_SEPARATION_M = 300.0
_WAYPOINT_BEAM_WIDTH = 16

# Fleet maneuver policy: while a track's position uncertainty exceeds this
# standard deviation (metres), its observers stop closing on the standoff
# ring and re-disperse around their own centroid instead of transiting. A
# diffuse belief makes each bearing update move the estimate by hundreds
# of metres, which can drag an observer inside the filter's sigma-point
# cloud mid-cycle; the wrapped bearings then drain the covariance
# non-positive-definite. With a converged track the gains are small, so
# transit through the standoff ring is safe. The re-dispersion radius
# restores the triangulation baseline of a bunched group during track-loss
# events, instead of freezing it in the geometry that lost the track.
_TRACK_CONVERGENCE_STD_M = 100.0
_HOLD_SPREAD_RADIUS_M = 900.0


def _stable_int(text: str) -> int:
    """Stable 64-bit integer fingerprint of ``text`` (deterministic, cross-process)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _noop_sink(truth: dict[str, object]) -> None:
    """Default evaluation sink: truth goes nowhere."""
    del truth


def _adversary_event_summary(event_type: str) -> str:
    """Create a bounded, non-truth description for target-side event input."""
    labels = {
        "active_ping": "active sonar emission observed",
        "target_detection_acquired": "hostile platform entered detection range",
        "target_detection_lost": "hostile platform left detection range",
        "observability_alert": "tracking observability alert",
        "observability_feedback": "periodic tracking observability feedback",
    }
    return labels.get(event_type, event_type.replace("_", " "))[:240]


_EXPLICIT_RUNTIME_ATTRIBUTES: tuple[str, ...] = (
    "_carrier_entity",
    "_carrier_entities",
    "_carrier_home_positions",
    "_usvs",
    "_usv_deployment_states",
    "_usv_capabilities",
    "_uuvs",
    "_uuv_platform_capabilities",
    "_uuv_motion_limits",
    "_targets",
    "_uuv_groups",
    "_uuv_speeds",
    "_mission_distance_m",
    "_uuv_statuses",
    "_deployment_states",
    "_recovery_waypoints",
    "_uuv_carrier_ids",
    "_mission_plan",
    "_mission_stop_ids",
    "_mission_stop_indices",
    "_mission_stop_windows",
    "_mission_batch_by_candidate",
    "_mission_recovered_uuv_ids",
    "_mission_recovery_requested_uuv_ids",
    "_mission_dispatched_uuv_ids",
    "_events",
    "_event_ledger",
    "_event_ledger_ids",
    "_pending_runtime_events",
    "_carrier_events",
    "_intelligence_reports",
    "_belief_histories",
    "_pending_group_commands",
    "_decoys",
    "_decoy_observations",
    "_contact_state",
    "_sensor_modes",
    "_ping_targets",
    "_ping_receivers",
    "_last_ping_times",
    "_reserved_by_target",
    "_reserved_uuvs",
    "_target_rays",
    "_assignments",
    "_latest_reports",
    "_last_guard_reasons",
    "_event_counters",
    "_previous_waypoints",
    "_waypoint_commands",
    "_connectivity",
    "_platform_observations",
    "_adversary_decision_history",
    "_adversary_gate",
    "_adversary_contexts",
    "_maneuver_response_chains",
    "_usv_waypoints",
    "_usv_execution_records",
    "_usv_hold_ids",
    "_applied_plan_revisions",
    "_regional_quality_streaks",
    "_regional_quality_latches",
    "_relay_failure_streaks",
    "_relay_failure_latches",
    "_target_detected_platform_ids",
    "_target_intents",
    "_observability",
    "_segment_plans_by_target",
    "_slave_covariance_trace_by_target",
    "_manager_threads",
    "_manager_storage",
    "_manager_writes",
    "_manager_blobs",
)

_ROLLBACK_MISSING = object()
_SAFE_DETACHED_SNAPSHOT_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    range,
    type(Ellipsis),
)


@dataclass(slots=True)
class _ExplicitRuntimeGraphCheckpoint:
    """One alias-preserving snapshot of every explicit runtime root."""

    originals: dict[str, Any]
    snapshot: dict[str, Any]
    originals_by_snapshot_id: dict[int, Any]


@dataclass(slots=True)
class _ExplicitRngMapCheckpoint:
    """Original RNG map identity, entries, and states from before a tick."""

    original: dict[str, random.Random]
    entries: dict[str, random.Random]
    states: dict[str, tuple[Any, ...]]


@dataclass(frozen=True, slots=True)
class _ExplicitArrayMetadataCheckpoint:
    """Array metadata that deepcopy does not retain on the original object."""

    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    native_state: tuple[Any, ...]
    writeable: bool
    aligned: bool
    c_contiguous: bool
    f_contiguous: bool
    owndata: bool
    writebackifcopy: bool


@dataclass(slots=True)
class _ExplicitPlatformCoreCheckpoint:
    """All engine-owned mutable state changed by one explicit platform tick."""

    step_index: int
    clock: SimulationClock
    clock_state: dict[str, Any]
    runtime: _ExplicitRuntimeGraphCheckpoint
    array_metadata_by_snapshot_id: dict[int, _ExplicitArrayMetadataCheckpoint]
    master_rng: random.Random
    master_rng_state: tuple[Any, ...]
    entity_rngs: _ExplicitRngMapCheckpoint
    observer_rngs: _ExplicitRngMapCheckpoint
    quality_rngs: _ExplicitRngMapCheckpoint
    logger: FrameLogger
    logger_checkpoint: FrameLogCheckpoint


def _remember_explicit_runtime_graph(value: Any, originals_by_id: dict[int, Any]) -> None:
    """Record restorable nodes and reject runtime state without an in-place restore path."""
    value_id = id(value)
    if value_id in originals_by_id:
        return
    originals_by_id[value_id] = value

    if _is_safe_detached_snapshot_value(value) or isinstance(value, (Enum, type)):
        return
    if isinstance(value, random.Random):
        raise RuntimeError(
            "explicit runtime rollback checkpoint cannot restore nested random.Random state; "
            "use an engine-owned RNG map"
        )
    if isinstance(value, bytearray):
        _remember_explicit_runtime_attributes(value, originals_by_id)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _remember_explicit_runtime_graph(key, originals_by_id)
            _remember_explicit_runtime_graph(item, originals_by_id)
        _remember_explicit_runtime_attributes(value, originals_by_id)
        return
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        for item in value:
            _remember_explicit_runtime_graph(item, originals_by_id)
        _remember_explicit_runtime_attributes(value, originals_by_id)
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            for item in value.flat:
                _remember_explicit_runtime_graph(item, originals_by_id)
        _remember_explicit_runtime_attributes(value, originals_by_id)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            if hasattr(value, field.name):
                _remember_explicit_runtime_graph(getattr(value, field.name), originals_by_id)
        return
    if _has_explicit_object_state(value):
        for attribute in _state_attribute_names(value):
            _remember_explicit_runtime_graph(getattr(value, attribute), originals_by_id)
        return
    raise RuntimeError(
        "explicit runtime rollback checkpoint cannot restore unsupported mutable runtime node "
        f"type {type(value).__module__}.{type(value).__qualname__}"
    )


def _remember_explicit_runtime_attributes(value: Any, originals_by_id: dict[int, Any]) -> None:
    """Traverse direct attributes carried by a mutable container subclass."""
    for attribute in _state_attribute_names(value):
        _remember_explicit_runtime_graph(getattr(value, attribute), originals_by_id)


def _slot_attribute_names(value: Any) -> set[str]:
    """Return slot declarations without treating an unset slot as state."""
    names: set[str] = set()
    for class_ in type(value).__mro__:
        slots = class_.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.update(slot for slot in slots if slot not in {"__dict__", "__weakref__"})
    return names


def _state_attribute_names(value: Any) -> set[str]:
    """Return direct instance attributes, including slots on frozen models."""
    names = set(vars(value)) if hasattr(value, "__dict__") else set()
    names.update(slot for slot in _slot_attribute_names(value) if hasattr(value, slot))
    return names


def _has_explicit_object_state(value: Any) -> bool:
    """Whether regular or slot attributes provide a safe in-place restore surface."""
    return hasattr(value, "__dict__") or bool(_slot_attribute_names(value))


def _record_snapshot_identity(
    original: Any, snapshot: Any, originals_by_snapshot_id: dict[int, Any]
) -> None:
    """Associate a snapshot node with exactly one original identity."""
    snapshot_id = id(snapshot)
    existing = originals_by_snapshot_id.get(snapshot_id, _ROLLBACK_MISSING)
    if existing is _ROLLBACK_MISSING:
        originals_by_snapshot_id[snapshot_id] = original
        return
    if existing is not original:
        raise RuntimeError(
            "explicit runtime rollback checkpoint has conflicting aliases for "
            f"snapshot node {type(snapshot).__qualname__}"
        )


def _unordered_snapshot_item_index(
    original: Any,
    snapshots: list[Any],
    originals_by_snapshot_id: Mapping[int, Any],
    *,
    item_kind: str,
) -> int:
    """Find one identity-safe correspondence for an unordered graph member."""
    associated = [
        index
        for index, snapshot in enumerate(snapshots)
        if originals_by_snapshot_id.get(id(snapshot), _ROLLBACK_MISSING) is original
    ]
    if len(associated) == 1:
        return associated[0]
    if len(associated) > 1:
        raise RuntimeError(
            f"explicit runtime rollback checkpoint has duplicate {item_kind} aliases"
        )

    identical = [index for index, snapshot in enumerate(snapshots) if snapshot is original]
    if len(identical) == 1:
        return identical[0]
    if len(identical) > 1:
        raise RuntimeError(
            f"explicit runtime rollback checkpoint has duplicate {item_kind} identities"
        )

    if _is_safe_detached_snapshot_value(original):
        equal = [
            index
            for index, snapshot in enumerate(snapshots)
            if type(snapshot) is type(original) and original == snapshot
        ]
        if len(equal) == 1:
            return equal[0]
        if len(equal) > 1:
            raise RuntimeError(
                f"explicit runtime rollback checkpoint has ambiguous {item_kind} values"
            )
    raise RuntimeError(
        f"explicit runtime rollback checkpoint cannot safely associate unordered {item_kind} "
        f"{type(original).__module__}.{type(original).__qualname__}"
    )


def _associate_explicit_runtime_graph(
    original: Any,
    snapshot: Any,
    originals_by_snapshot_id: dict[int, Any],
) -> None:
    """Augment deepcopy's memo with all structurally corresponding graph nodes."""
    visited: set[tuple[int, int]] = set()

    def associate(original_value: Any, snapshot_value: Any) -> None:
        _record_snapshot_identity(original_value, snapshot_value, originals_by_snapshot_id)
        pair = (id(original_value), id(snapshot_value))
        if pair in visited or original_value is snapshot_value:
            return
        visited.add(pair)

        if type(original_value) is not type(snapshot_value):
            raise RuntimeError(
                "explicit runtime rollback checkpoint changed node type from "
                f"{type(original_value).__qualname__} to {type(snapshot_value).__qualname__}"
            )
        if isinstance(original_value, dict):
            if len(original_value) != len(snapshot_value):
                raise RuntimeError("explicit runtime rollback checkpoint changed dictionary size")
            unmatched_snapshot_items = list(snapshot_value.items())
            for original_key, original_item in original_value.items():
                index = _unordered_snapshot_item_index(
                    original_key,
                    [snapshot_key for snapshot_key, _ in unmatched_snapshot_items],
                    originals_by_snapshot_id,
                    item_kind="dictionary key",
                )
                snapshot_key, snapshot_item = unmatched_snapshot_items.pop(index)
                associate(original_key, snapshot_key)
                associate(original_item, snapshot_item)
            associate_attributes(original_value, snapshot_value)
            return
        if isinstance(original_value, (list, tuple, deque)):
            if len(original_value) != len(snapshot_value):
                raise RuntimeError("explicit runtime rollback checkpoint changed sequence size")
            for original_item, snapshot_item in zip(original_value, snapshot_value, strict=True):
                associate(original_item, snapshot_item)
            associate_attributes(original_value, snapshot_value)
            return
        if isinstance(original_value, (set, frozenset)):
            if len(original_value) != len(snapshot_value):
                raise RuntimeError("explicit runtime rollback checkpoint changed set size")
            unmatched_snapshots = list(snapshot_value)
            for original_item in original_value:
                index = _unordered_snapshot_item_index(
                    original_item,
                    unmatched_snapshots,
                    originals_by_snapshot_id,
                    item_kind="set member",
                )
                associate(original_item, unmatched_snapshots.pop(index))
            associate_attributes(original_value, snapshot_value)
            return
        if isinstance(original_value, np.ndarray):
            if original_value.shape != snapshot_value.shape:
                raise RuntimeError("explicit runtime rollback checkpoint changed array shape")
            if original_value.dtype != snapshot_value.dtype:
                raise RuntimeError("explicit runtime rollback checkpoint changed array dtype")
            if original_value.dtype.hasobject:
                for array_index in np.ndindex(original_value.shape):
                    associate(original_value[array_index], snapshot_value[array_index])
            associate_attributes(original_value, snapshot_value)
            return
        if isinstance(original_value, bytearray):
            associate_attributes(original_value, snapshot_value)
            return
        if _has_explicit_object_state(original_value):
            associate_attributes(original_value, snapshot_value)

    def associate_attributes(original_value: Any, snapshot_value: Any) -> None:
        if not _has_explicit_object_state(original_value):
            return
        original_names = _state_attribute_names(original_value)
        snapshot_names = _state_attribute_names(snapshot_value)
        if original_names != snapshot_names:
            raise RuntimeError("explicit runtime rollback checkpoint changed object attributes")
        for attribute in original_names:
            associate(getattr(original_value, attribute), getattr(snapshot_value, attribute))

    associate(original, snapshot)


def _is_safe_detached_snapshot_value(value: Any) -> bool:
    """Whether a checkpoint node can be returned without attaching mutable snapshot state."""
    return type(value) in _SAFE_DETACHED_SNAPSHOT_TYPES


def _restore_explicit_object_attributes(
    original: Any,
    snapshot: Any,
    originals_by_snapshot_id: Mapping[int, Any],
    restored: dict[int, Any],
    array_metadata_by_snapshot_id: Mapping[int, _ExplicitArrayMetadataCheckpoint],
) -> None:
    """Restore regular and frozen object attributes without replacing the object."""
    original_names = _state_attribute_names(original)
    snapshot_names = _state_attribute_names(snapshot)
    for attribute in original_names - snapshot_names:
        object.__delattr__(original, attribute)
    for attribute in snapshot_names:
        if hasattr(snapshot, attribute):
            object.__setattr__(
                original,
                attribute,
                _restore_explicit_value(
                    getattr(snapshot, attribute),
                    originals_by_snapshot_id,
                    restored,
                    array_metadata_by_snapshot_id,
                ),
            )


def _array_metadata(value: np.ndarray[Any, Any]) -> _ExplicitArrayMetadataCheckpoint:
    """Capture ndarray metadata that must remain attached to the original object."""
    native_state = value.__reduce__()[2]
    if not isinstance(native_state, tuple):
        raise RuntimeError("explicit runtime rollback checkpoint cannot capture ndarray state")
    return _ExplicitArrayMetadataCheckpoint(
        dtype=value.dtype,
        shape=value.shape,
        strides=value.strides,
        native_state=native_state,
        writeable=value.flags.writeable,
        aligned=value.flags.aligned,
        c_contiguous=value.flags.c_contiguous,
        f_contiguous=value.flags.f_contiguous,
        owndata=value.flags.owndata,
        writebackifcopy=value.flags.writebackifcopy,
    )


def _native_restorable_array_strides(native_state: tuple[Any, ...]) -> tuple[int, ...]:
    """Return the canonical strides NumPy applies when restoring ``native_state``."""
    restored = np.empty(0, dtype=np.uint8)
    restored.__setstate__(native_state)
    return restored.strides


def _validate_checkpoint_array(value: np.ndarray[Any, Any]) -> _ExplicitArrayMetadataCheckpoint:
    """Reject ndarray state that cannot be restored with its original identity."""
    if type(value) is not np.ndarray:
        raise RuntimeError(
            "explicit runtime rollback checkpoint cannot restore ndarray subclasses"
        )
    metadata = _array_metadata(value)
    if metadata.writebackifcopy:
        raise RuntimeError(
            "explicit runtime rollback checkpoint cannot restore writeback-if-copy ndarray"
        )
    if not metadata.owndata:
        raise RuntimeError("explicit runtime rollback checkpoint cannot restore non-owning ndarray")
    if not (metadata.c_contiguous or metadata.f_contiguous):
        raise RuntimeError(
            "explicit runtime rollback checkpoint cannot restore non-C/non-F owning ndarray"
        )
    native_restorable_strides = _native_restorable_array_strides(metadata.native_state)
    if metadata.strides != native_restorable_strides:
        raise RuntimeError(
            "explicit runtime rollback checkpoint cannot restore owning ndarray strides "
            f"{metadata.strides}; native state restores {native_restorable_strides}"
        )
    if value.dtype.hasobject and any(item is value for item in value.flat):
        raise RuntimeError(
            "explicit runtime rollback checkpoint cannot restore self-referential object ndarray"
        )
    if metadata.writeable:
        return metadata
    try:
        value.setflags(write=True)
    except ValueError as error:
        raise RuntimeError(
            "explicit runtime rollback checkpoint cannot restore read-only ndarray values"
        ) from error
    value.setflags(write=False)
    return metadata


def _restore_explicit_array(
    original: np.ndarray[Any, Any],
    snapshot: np.ndarray[Any, Any],
    metadata: _ExplicitArrayMetadataCheckpoint,
    originals_by_snapshot_id: Mapping[int, Any],
    restored: dict[int, Any],
    array_metadata_by_snapshot_id: Mapping[int, _ExplicitArrayMetadataCheckpoint],
) -> None:
    """Restore ndarray values and mutable metadata while retaining its identity."""
    failures: list[Exception] = []
    try:
        original.setflags(write=True)
        original.__setstate__(metadata.native_state)
        if (
            original.dtype != metadata.dtype
            or original.shape != metadata.shape
            or original.strides != metadata.strides
            or original.flags.c_contiguous != metadata.c_contiguous
            or original.flags.f_contiguous != metadata.f_contiguous
            or original.flags.owndata != metadata.owndata
            or original.flags.writebackifcopy != metadata.writebackifcopy
        ):
            raise RuntimeError(
                "explicit runtime rollback cannot restore ndarray dtype, shape, strides, C/F layout, "
                "or ownership"
            )
        if (
            snapshot.dtype != metadata.dtype
            or snapshot.shape != metadata.shape
            or snapshot.strides != metadata.strides
            or snapshot.flags.c_contiguous != metadata.c_contiguous
            or snapshot.flags.f_contiguous != metadata.f_contiguous
        ):
            raise RuntimeError(
                "explicit runtime rollback checkpoint has inconsistent ndarray dtype, shape, strides, "
                "or C/F layout"
            )
        if snapshot.dtype.hasobject:
            for index in np.ndindex(snapshot.shape):
                original[index] = _restore_explicit_value(
                    snapshot[index],
                    originals_by_snapshot_id,
                    restored,
                    array_metadata_by_snapshot_id,
                )
    except Exception as error:
        failures.append(error)
    try:
        original.setflags(write=metadata.writeable, align=metadata.aligned)
        if (
            original.dtype != metadata.dtype
            or original.shape != metadata.shape
            or original.strides != metadata.strides
            or original.flags.writeable != metadata.writeable
            or original.flags.aligned != metadata.aligned
            or original.flags.c_contiguous != metadata.c_contiguous
            or original.flags.f_contiguous != metadata.f_contiguous
            or original.flags.owndata != metadata.owndata
            or original.flags.writebackifcopy != metadata.writebackifcopy
        ):
            raise RuntimeError("explicit runtime rollback cannot restore ndarray metadata or flags")
    except Exception as error:
        failures.append(error)
    if failures:
        raise ExceptionGroup("explicit runtime rollback could not restore ndarray", failures)


def _restore_explicit_value(
    snapshot: Any,
    originals_by_snapshot_id: Mapping[int, Any],
    restored: dict[int, Any],
    array_metadata_by_snapshot_id: Mapping[int, _ExplicitArrayMetadataCheckpoint],
) -> Any:
    """Restore a graph snapshot into its pre-tick objects while preserving aliases."""
    snapshot_id = id(snapshot)
    if snapshot_id in restored:
        return restored[snapshot_id]

    if snapshot_id not in originals_by_snapshot_id:
        if _is_safe_detached_snapshot_value(snapshot):
            return snapshot
        raise RuntimeError(
            "explicit runtime rollback cannot restore unsupported mutable checkpoint node "
            f"{type(snapshot).__qualname__} without an original identity"
        )
    original = originals_by_snapshot_id[snapshot_id]
    restored[snapshot_id] = original
    if original is snapshot:
        return original
    if type(snapshot) is not type(original):
        raise RuntimeError(
            "explicit runtime rollback cannot restore checkpoint node with changed type from "
            f"{type(original).__qualname__} to {type(snapshot).__qualname__}"
        )

    if isinstance(snapshot, dict):
        restored_items = [
            (
                _restore_explicit_value(
                    key, originals_by_snapshot_id, restored, array_metadata_by_snapshot_id
                ),
                _restore_explicit_value(
                    value, originals_by_snapshot_id, restored, array_metadata_by_snapshot_id
                ),
            )
            for key, value in snapshot.items()
        ]
        original.clear()
        original.update(restored_items)
        _restore_explicit_object_attributes(
            original,
            snapshot,
            originals_by_snapshot_id,
            restored,
            array_metadata_by_snapshot_id,
        )
        return original
    if isinstance(snapshot, list):
        restored_items = [
            _restore_explicit_value(
                value, originals_by_snapshot_id, restored, array_metadata_by_snapshot_id
            )
            for value in snapshot
        ]
        original.clear()
        original.extend(restored_items)
        _restore_explicit_object_attributes(
            original,
            snapshot,
            originals_by_snapshot_id,
            restored,
            array_metadata_by_snapshot_id,
        )
        return original
    if isinstance(snapshot, deque):
        if original.maxlen != snapshot.maxlen:
            raise RuntimeError("explicit runtime rollback cannot restore deque with changed maxlen")
        restored_items = [
            _restore_explicit_value(
                value, originals_by_snapshot_id, restored, array_metadata_by_snapshot_id
            )
            for value in snapshot
        ]
        original.clear()
        original.extend(restored_items)
        _restore_explicit_object_attributes(
            original,
            snapshot,
            originals_by_snapshot_id,
            restored,
            array_metadata_by_snapshot_id,
        )
        return original
    if isinstance(snapshot, bytearray):
        original[:] = snapshot
        _restore_explicit_object_attributes(
            original,
            snapshot,
            originals_by_snapshot_id,
            restored,
            array_metadata_by_snapshot_id,
        )
        return original
    if isinstance(snapshot, set):
        restored_set_items = {
            _restore_explicit_value(
                value, originals_by_snapshot_id, restored, array_metadata_by_snapshot_id
            )
            for value in snapshot
        }
        original.clear()
        original.update(restored_set_items)
        _restore_explicit_object_attributes(
            original,
            snapshot,
            originals_by_snapshot_id,
            restored,
            array_metadata_by_snapshot_id,
        )
        return original
    if isinstance(snapshot, np.ndarray):
        metadata = array_metadata_by_snapshot_id.get(snapshot_id)
        if metadata is None:
            raise RuntimeError("explicit runtime rollback is missing ndarray metadata")
        _restore_explicit_array(
            original,
            snapshot,
            metadata,
            originals_by_snapshot_id,
            restored,
            array_metadata_by_snapshot_id,
        )
        _restore_explicit_object_attributes(
            original,
            snapshot,
            originals_by_snapshot_id,
            restored,
            array_metadata_by_snapshot_id,
        )
        return original
    if isinstance(snapshot, (tuple, frozenset)):
        for value in snapshot:
            _restore_explicit_value(
                value, originals_by_snapshot_id, restored, array_metadata_by_snapshot_id
            )
        _restore_explicit_object_attributes(
            original,
            snapshot,
            originals_by_snapshot_id,
            restored,
            array_metadata_by_snapshot_id,
        )
        return original

    if _has_explicit_object_state(original):
        _restore_explicit_object_attributes(
            original,
            snapshot,
            originals_by_snapshot_id,
            restored,
            array_metadata_by_snapshot_id,
        )
        return original
    raise RuntimeError(
        "explicit runtime rollback cannot restore unsupported mutable checkpoint node "
        f"{type(snapshot).__qualname__} in place"
    )


class SimulationEngine:
    """Deterministic headless simulation of the multi-UUV bearing-only scenario."""

    def __init__(
        self,
        config: AppConfig,
        seed: int = 42,
        output_dir: str | Path | None = None,
        evaluation_sink: Callable[[dict[str, object]], None] | None = None,
        carrier: Callable[[SituationSnapshot], None] | None = None,
        mission_controller: MissionController | None = None,
    ) -> None:
        self._config = config
        self._seed = seed
        self._scenario_id = config.scenario.scenario_id
        self._platform_core_enabled = config.environment is not None
        self._uuv_only_runtime = bool(
            config.scenario.uuv_only
            or (config.environment is not None and config.environment.uuv_only)
        )
        self._run_id = f"run-{uuid.uuid4().hex}"
        self._sink = evaluation_sink if evaluation_sink is not None else _noop_sink
        self._carrier = carrier
        self._mission_controller = mission_controller
        self._mission_controller_event_ids: set[str] = set()
        self._step_index = 0
        self._events: list[RuntimeEvent] = []
        self._event_ledger: list[RuntimeEvent] = []
        self._event_ledger_ids: set[str] = set()
        self._master_rng = random.Random(seed)
        self._entity_rngs: dict[str, random.Random] = {}
        self._observer_rngs: dict[str, random.Random] = {}
        self._quality_rngs: dict[str, random.Random] = {}
        self._clock = SimulationClock(step_s=config.timing.physics_step_s)
        self._usvs: dict[str, USVEntity] = {}
        self._usv_deployment_states: dict[str, DeploymentState] = {}
        self._usv_capabilities: dict[str, PlatformCapability] = {}
        self._uuv_platform_capabilities: dict[str, PlatformCapability] = {}
        self._uuv_motion_limits: dict[str, MotionLimits] = {}
        self._connectivity = ConnectivitySnapshot(links=())
        self._platform_observations: tuple[PassiveSonarObservation, ...] = ()
        self._adversary_decision_history: dict[
            str, tuple[AdversaryDecisionRecord, ...]
        ] = {}
        self._adversary_gate = AdversaryDecisionGate()
        self._adversary_contexts: dict[str, AdversaryEscapeInput] = {}
        self._maneuver_response_chains: dict[str, dict[str, object]] = {}
        self._usv_waypoints: dict[str, list[tuple[float, float]]] = {}
        self._usv_execution_records: dict[str, dict[str, object]] = {}
        self._usv_hold_ids: set[str] = set()
        self._applied_plan_revisions: dict[tuple[str, str], int] = {}
        self._regional_quality_streaks: dict[str, int] = {}
        self._regional_quality_latches: set[str] = set()
        self._relay_failure_streaks: dict[str, int] = {}
        self._relay_failure_latches: set[str] = set()
        self._target_detected_platform_ids: dict[str, tuple[str, ...]] = {}
        self._target_intents: dict[str, HiddenIntent] = {}
        self._belief_intent_state: dict[str, tuple[str, float]] = {}
        self._observability = self._load_observability_supervisor()
        self._segment_plans_by_target: dict[str, tuple[Segment, ...]] = {}
        self._slave_covariance_trace_by_target: dict[str, float] = {}
        environment = config.environment
        self._carrier_entities: dict[str, CarrierEntity] = {}
        self._carrier_home_positions: dict[str, tuple[float, float]] = {}
        if environment is None:
            self._carrier_entity = CarrierEntity()
            self._carrier_entities[self._carrier_entity.carrier_id] = self._carrier_entity
            self._carrier_home_positions[self._carrier_entity.carrier_id] = (
                self._carrier_entity.position_xy
            )
        else:
            carrier_configs = (
                (environment.carrier, *environment.carriers)
                if self._uuv_only_runtime
                else (environment.carrier,)
            )
            for carrier_config in carrier_configs:
                if carrier_config.platform_id in self._carrier_entities:
                    continue
                entity = CarrierEntity(
                    carrier_id=carrier_config.platform_id,
                    position_xy=carrier_config.position_xy,
                    speed_mps=carrier_config.speed_mps,
                    patrol_route_xy=carrier_config.patrol_route_xy,
                    support_radius_m=carrier_config.support_radius_m,
                    heading_rad=carrier_config.heading_rad,
                )
                self._carrier_entities[entity.carrier_id] = entity
                self._carrier_home_positions[entity.carrier_id] = entity.position_xy
            self._carrier_entity = self._carrier_entities[environment.carrier.platform_id]
        self._uuvs: dict[str, UUVEntity] = {}
        self._targets: dict[str, TargetEntity] = {}
        self._uuv_groups: dict[str, str] = {}
        self._uuv_speeds: dict[str, float] = {}
        self._uuv_statuses: dict[str, UUVStatus] = {}
        self._deployment_states: dict[str, DeploymentState] = {}
        self._recovery_waypoints: dict[str, list[tuple[float, float]]] = {}
        self._uuv_carrier_ids: dict[str, str] = {}
        self._mission_plan: ExecutableMissionPlan | None = None
        self._mission_stop_ids: dict[str, tuple[str, ...]] = {}
        self._mission_stop_indices: dict[str, tuple[int, ...]] = {}
        self._mission_stop_windows: dict[str, dict[int, tuple[int, int]]] = {}
        self._mission_batch_by_candidate: dict[tuple[str, str], tuple[str, ...]] = {}
        self._mission_recovered_uuv_ids: set[str] = set()
        self._mission_recovery_requested_uuv_ids: set[str] = set()
        self._mission_dispatched_uuv_ids: set[str] = set()
        self._pending_runtime_events: list[RuntimeEvent] = []
        self._carrier_events: list[RuntimeEvent] = []
        self._operational_scheme = config.scenario.operational_scheme
        self._intelligence_reports: dict[str, IntelligenceReport] = {}
        self._belief_histories: dict[str, list[tuple[int, float, float]]] = {}
        self._pending_group_commands: dict[str, GroupPlanCommand] = {}
        self._decoys: dict[str, DecoyEntity] = {}
        self._decoy_observations: dict[str, tuple[BearingObservation, ...]] = {}
        self._contact_state: dict[str, dict[str, Any]] = {}
        self._sensor_modes: dict[str, Literal["passive", "active"]] = {}
        self._ping_targets: dict[str, str | None] = {}
        self._ping_receivers: dict[str, tuple[str, ...]] = {}
        self._last_ping_times: dict[tuple[str, str], int] = {}
        self._reserved_by_target: dict[str, tuple[str, ...]] = {}
        self._reserved_uuvs: frozenset[str] = frozenset()
        self._target_rays: dict[str, tuple[BearingObservation, ...]] = {}
        self._spawn_world()
        self._mission_distance_m = {uuv_id: 0.0 for uuv_id in self._uuvs}
        carrier_ids = tuple(sorted(self._carrier_entities))
        for index, uuv_id in enumerate(sorted(self._uuvs)):
            carrier_id = carrier_ids[index % len(carrier_ids)]
            self._uuv_carrier_ids[uuv_id] = carrier_id
            if self._uuv_only_runtime and self._deployment_states[uuv_id] is DeploymentState.ONBOARD:
                carrier = self._carrier_entities[carrier_id]
                self._uuvs[uuv_id].position_xy = carrier.position_xy
                self._uuvs[uuv_id].heading_rad = carrier.heading_rad
        self._target_intents = {
            target_id: target.intent for target_id, target in self._targets.items()
        }
        self._assignments: dict[str, tuple[str, ...]] = {}
        self._manager = GroupManager()
        # Keep the mutable LangGraph checkpoint containers in the explicit
        # rollback graph without snapshotting the compiled graph object.
        self._manager_threads: dict[str, str] = self._manager._threads
        checkpointer: Any = self._manager._checkpointer
        self._manager_storage: Any = checkpointer.storage
        self._manager_writes: Any = checkpointer.writes
        self._manager_blobs: Any = checkpointer.blobs
        self._latest_reports: dict[str, GroupReport] = {}
        self._last_guard_reasons: dict[str, tuple[str, ...]] = {}
        self._event_counters: dict[str, int] = {}
        if self._platform_core_enabled:
            self._initialize_explicit_groups()
        else:
            self._allocate_and_create_groups()
        self._previous_waypoints: dict[str, np.ndarray[Any, Any]] = {}
        self._waypoint_commands: dict[str, dict[str, tuple[float, float]]] = {}
        self._plan_waypoints()
        directory = Path(output_dir) if output_dir is not None else Path("outputs") / self._run_id
        self.logger = FrameLogger(directory)

    def step(self) -> dict[str, object]:
        """Advance the clock once and return the operational frame."""
        explicit_checkpoint = (
            self._checkpoint_explicit_platform_core() if self._platform_core_enabled else None
        )
        previous_step_index = self._step_index
        previous_sim_time_s = self._clock.sim_time_s
        previous_events = list(self._events)
        previous_pending_events = list(self._pending_runtime_events)
        try:
            self._step_index += 1
            self._events = self._pending_runtime_events
            self._pending_runtime_events = []
            sim_time_s = self._clock.tick()
            timing = self._config.timing
            self._advance_world(sim_time_s)
            if sim_time_s % timing.observation_step_s == 0:
                self._observation_cycle(sim_time_s)
            elif self._carrier is not None:
                self._carrier_events.extend(self._events)
            if sim_time_s % timing.group_report_s == 0:
                self._publish_reports(sim_time_s)
            frame = self._build_frame(sim_time_s)
            self.logger.write(frame)
            for event in self._events:
                if event.event_id not in self._event_ledger_ids:
                    self._event_ledger.append(event)
                    self._event_ledger_ids.add(event.event_id)
            self._sink(self._truth(sim_time_s))
            return frame
        except Exception as step_error:
            try:
                if explicit_checkpoint is not None:
                    self._restore_explicit_platform_core(explicit_checkpoint)
                else:
                    self._step_index = previous_step_index
                    self._clock.sim_time_s = previous_sim_time_s
                    self._events = previous_events
                    self._pending_runtime_events = previous_pending_events
            except Exception as rollback_error:
                step_error.__context__ = rollback_error
            raise

    def _checkpoint_explicit_platform_core(self) -> _ExplicitPlatformCoreCheckpoint:
        """Snapshot explicit-world runtime before a tick that may fail late."""
        runtime_originals = {
            attribute: getattr(self, attribute) for attribute in _EXPLICIT_RUNTIME_ATTRIBUTES
        }
        originals_by_id: dict[int, Any] = {}
        for value in runtime_originals.values():
            _remember_explicit_runtime_graph(value, originals_by_id)
        array_metadata_by_original_id = {
            original_id: _validate_checkpoint_array(value)
            for original_id, value in originals_by_id.items()
            if isinstance(value, np.ndarray)
        }
        memo: dict[int, Any] = {}
        runtime_snapshot = deepcopy(runtime_originals, memo)
        originals_by_snapshot_id: dict[int, Any] = {}
        for original_id, snapshot_value in memo.items():
            if original_id in originals_by_id:
                _record_snapshot_identity(
                    originals_by_id[original_id], snapshot_value, originals_by_snapshot_id
                )
        for attribute, original_value in runtime_originals.items():
            _associate_explicit_runtime_graph(
                original_value, runtime_snapshot[attribute], originals_by_snapshot_id
            )
        array_metadata_by_snapshot_id = {
            snapshot_id: array_metadata_by_original_id[id(original)]
            for snapshot_id, original in originals_by_snapshot_id.items()
            if isinstance(original, np.ndarray)
        }
        return _ExplicitPlatformCoreCheckpoint(
            step_index=self._step_index,
            clock=self._clock,
            clock_state={
                field.name: getattr(self._clock, field.name) for field in fields(self._clock)
            },
            runtime=_ExplicitRuntimeGraphCheckpoint(
                originals=runtime_originals,
                snapshot=runtime_snapshot,
                originals_by_snapshot_id=originals_by_snapshot_id,
            ),
            array_metadata_by_snapshot_id=array_metadata_by_snapshot_id,
            master_rng=self._master_rng,
            master_rng_state=self._master_rng.getstate(),
            entity_rngs=self._checkpoint_rng_map(self._entity_rngs),
            observer_rngs=self._checkpoint_rng_map(self._observer_rngs),
            quality_rngs=self._checkpoint_rng_map(self._quality_rngs),
            logger=self.logger,
            logger_checkpoint=self.logger.checkpoint(),
        )

    def _restore_explicit_platform_core(
        self, checkpoint: _ExplicitPlatformCoreCheckpoint
    ) -> None:
        """Restore every failed-tick section before reporting restoration failures."""
        failures: list[Exception] = []

        def attempt(section: str, operation: Callable[[], None]) -> None:
            try:
                operation()
            except Exception as error:
                failure = RuntimeError(f"explicit runtime rollback failed to restore {section}")
                failure.__cause__ = error
                failures.append(failure)

        restored: dict[int, Any] = {}

        def restore_runtime_snapshot(snapshot: Any) -> None:
            _restore_explicit_value(
                snapshot,
                checkpoint.runtime.originals_by_snapshot_id,
                restored,
                checkpoint.array_metadata_by_snapshot_id,
            )

        def restore_runtime_identity(attribute: str) -> None:
            setattr(self, attribute, checkpoint.runtime.originals[attribute])

        for attribute, snapshot in checkpoint.runtime.snapshot.items():
            attempt(
                f"runtime graph {attribute}",
                lambda: restore_runtime_snapshot(snapshot),
            )
            attempt(
                f"runtime identity {attribute}",
                lambda: restore_runtime_identity(attribute),
            )
        attempt("step index", lambda: setattr(self, "_step_index", checkpoint.step_index))
        attempt("clock identity", lambda: setattr(self, "_clock", checkpoint.clock))
        for field_name, field_value in checkpoint.clock_state.items():

            def restore_clock_field(
                field_name: str = field_name, field_value: Any = field_value
            ) -> None:
                setattr(checkpoint.clock, field_name, field_value)

            attempt(
                f"clock {field_name}",
                restore_clock_field,
            )
        attempt("master RNG identity", lambda: setattr(self, "_master_rng", checkpoint.master_rng))
        attempt("master RNG state", lambda: checkpoint.master_rng.setstate(checkpoint.master_rng_state))
        attempt(
            "entity RNG map identity",
            lambda: setattr(self, "_entity_rngs", checkpoint.entity_rngs.original),
        )
        attempt("entity RNG map", lambda: self._restore_rng_map(checkpoint.entity_rngs))
        attempt(
            "observer RNG map identity",
            lambda: setattr(self, "_observer_rngs", checkpoint.observer_rngs.original),
        )
        attempt("observer RNG map", lambda: self._restore_rng_map(checkpoint.observer_rngs))
        attempt(
            "quality RNG map identity",
            lambda: setattr(self, "_quality_rngs", checkpoint.quality_rngs.original),
        )
        attempt("quality RNG map", lambda: self._restore_rng_map(checkpoint.quality_rngs))
        attempt("logger identity", lambda: setattr(self, "logger", checkpoint.logger))
        attempt("logger position", lambda: checkpoint.logger.restore(checkpoint.logger_checkpoint))
        if failures:
            raise ExceptionGroup("explicit runtime rollback failed", failures)

    def _load_observability_supervisor(self) -> ObservabilitySupervisor:
        """Load the explicit estimator-feedback configuration for this scenario."""
        configured = Path(self._config.scenario.observability_feedback_config)
        candidates = (
            configured,
            Path(__file__).resolve().parents[3] / configured,
        )
        config_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if config_path is None:
            raise FileNotFoundError(
                "observability feedback config not found: "
                f"{self._config.scenario.observability_feedback_config}"
            )
        return ObservabilitySupervisor(load_observability_config(config_path))

    @staticmethod
    def _checkpoint_rng_map(rngs: dict[str, random.Random]) -> _ExplicitRngMapCheckpoint:
        """Capture an RNG container and its entry identities before a potentially failed tick."""
        entries = dict(rngs)
        return _ExplicitRngMapCheckpoint(
            original=rngs,
            entries=entries,
            states={rng_id: rng.getstate() for rng_id, rng in entries.items()},
        )

    @staticmethod
    def _restore_rng_map(checkpoint: _ExplicitRngMapCheckpoint) -> None:
        """Restore the original RNG map and every object state, including aliases."""
        failures: list[Exception] = []
        try:
            checkpoint.original.clear()
            checkpoint.original.update(checkpoint.entries)
        except Exception as error:
            failures.append(error)
        for rng_id, rng in checkpoint.entries.items():
            try:
                rng.setstate(checkpoint.states[rng_id])
            except Exception as error:
                failures.append(error)
        if failures:
            raise ExceptionGroup("explicit runtime rollback could not restore RNG map", failures)

    def _spawn_world(self) -> None:
        environment = self._config.environment
        if environment is None:
            self._spawn_legacy_world()
            return
        self._spawn_explicit_world(environment)

    def _spawn_legacy_world(self) -> None:
        scenario = self._config.scenario
        tracking = self._config.tracking
        for index in range(scenario.uuv_count):
            uuv_id = f"uuv_{index:02d}"
            angle = 2.0 * pi * index / scenario.uuv_count
            position = (float(_UUV_DEPLOY_RADIUS_M * cos(angle)), float(_UUV_DEPLOY_RADIUS_M * sin(angle)))
            heading = atan2(-position[1], -position[0])
            self._uuvs[uuv_id] = UUVEntity(
                uuv_id,
                position,
                heading,
                1.0,
                capability=self._capability_for(uuv_id),
            )
            self._uuv_speeds[uuv_id] = 0.0
            self._deployment_states[uuv_id] = DeploymentState.DEPLOYED
        for index in range(scenario.initial_decoy_count):
            decoy_id = f"decoy_{index:02d}"
            x = self._master_rng.uniform(-_TARGET_SPAWN_SPAN_M, _TARGET_SPAWN_SPAN_M)
            y = self._master_rng.uniform(-_TARGET_SPAWN_SPAN_M, _TARGET_SPAWN_SPAN_M)
            heading = self._master_rng.uniform(-pi, pi)
            self._decoys[decoy_id] = DecoyEntity(
                decoy_id,
                (float(x), float(y)),
                heading,
                tracking.decoy_drift_speed_mps,
                tracking.decoy_heading_noise_rad_per_s,
            )
            self._contact_state[decoy_id] = {
                "classification": ContactClassification.UNVERIFIED,
                "evidence": (),
                "position_xy": None,
            }
        for index in range(scenario.initial_target_count):
            target_id = f"target_{index:02d}"
            x = self._master_rng.uniform(-_TARGET_SPAWN_SPAN_M, _TARGET_SPAWN_SPAN_M)
            y = self._master_rng.uniform(-_TARGET_SPAWN_SPAN_M, _TARGET_SPAWN_SPAN_M)
            self._targets[target_id] = TargetEntity(
                target_id,
                (float(x), float(y)),
                (tracking.submarine_cruise_speed_mps, 0.0),
                HiddenIntent.TRANSIT,
                intent_speed_mps=self._intent_speed_mps(),
                max_speed_mps=tracking.submarine_sprint_speed_mps,
                max_turn_rate_rad_s=tracking.submarine_turn_rate_rad_s,
            )
        for target_id in self._targets:
            self._contact_state[target_id] = {
                "classification": ContactClassification.SUBMARINE,
                "evidence": (),
                "position_xy": None,
            }

    def _platform_capability(self, platform: InitialPlatformConfig) -> PlatformCapability:
        catalog = self._config.platforms
        sensors = self._config.sensors
        communications = self._config.communications
        assert catalog is not None and sensors is not None and communications is not None
        motion = catalog.motion_profiles[platform.motion_profile]
        sonar = sensors.profiles[platform.sensor_profile]
        communication = communications.profiles[platform.communication_profile]
        return PlatformCapability(
            kind=platform.kind,
            motion=MotionLimits(
                max_speed_mps=motion.max_speed_mps,
                max_acceleration_mps2=motion.max_acceleration_mps2,
                max_turn_rate_rad_s=motion.max_turn_rate_rad_s,
            ),
            sonar=SonarCapability(**sonar.model_dump()),
            communications=CommunicationCapability(**communication.model_dump()),
        )

    def _spawn_explicit_world(self, environment: EnvironmentConfig) -> None:
        if environment.decoys or len(environment.submarines) != 1:
            raise ValueError("platform-core world requires one submarine and no decoys")
        catalog = self._config.platforms
        assert catalog is not None
        for initial in (() if self._uuv_only_runtime else environment.usvs):
            capability = self._platform_capability(initial)
            motion_profile = catalog.motion_profiles[initial.motion_profile]
            self._usv_capabilities[initial.platform_id] = capability
            self._usv_deployment_states[initial.platform_id] = DeploymentState(
                initial.deployment_state
            )
            self._usvs[initial.platform_id] = USVEntity(
                usv_id=initial.platform_id,
                platform_index=initial.platform_index,
                motion=MotionState(initial.position_xy, initial.heading_rad, 0.0),
                energy_fraction=initial.energy_fraction,
                limits=capability.motion,
                transit_energy_per_m=motion_profile.transit_energy_per_m,
                hotel_energy_per_s=motion_profile.hotel_energy_per_s,
            )
        for initial in environment.uuvs:
            capability = self._platform_capability(initial)
            motion_profile = catalog.motion_profiles[initial.motion_profile]
            self._uuv_platform_capabilities[initial.platform_id] = capability
            self._uuv_motion_limits[initial.platform_id] = capability.motion
            self._uuvs[initial.platform_id] = UUVEntity(
                uuv_id=initial.platform_id,
                position_xy=initial.position_xy,
                heading_rad=initial.heading_rad,
                energy_fraction=initial.energy_fraction,
                capability=SurveillanceCapability(
                    passive_range_m=capability.sonar.passive_range_m,
                    active_range_m=capability.sonar.active_source_range_m,
                    bearing_variance_rad2=capability.sonar.passive_bearing_variance_rad2,
                    active_sonar_available=capability.sonar.active_capable,
                    max_speed_mps=capability.motion.max_speed_mps,
                    max_turn_rate_rad_s=capability.motion.max_turn_rate_rad_s,
                ),
                platform_index=initial.platform_index,
                transit_energy_per_m=motion_profile.transit_energy_per_m,
                hotel_energy_per_s=motion_profile.hotel_energy_per_s,
            )
            if initial.deployment_state == "onboard":
                uuv = self._uuvs[initial.platform_id]
                uuv.position_xy = self._carrier_entity.position_xy
                uuv.heading_rad = self._carrier_entity.heading_rad
                uuv.speed_mps = 0.0
            self._uuv_speeds[initial.platform_id] = 0.0
            self._deployment_states[initial.platform_id] = DeploymentState(
                initial.deployment_state
            )
        submarine = environment.submarines[0]
        submarine_motion = catalog.motion_profiles[submarine.motion_profile]
        self._targets[submarine.target_id] = TargetEntity(
            target_id=submarine.target_id,
            position_xy=submarine.position_xy,
            velocity_xy=(
                submarine.speed_mps * cos(submarine.heading_rad),
                submarine.speed_mps * sin(submarine.heading_rad),
            ),
            intent=HiddenIntent.TRANSIT,
            bounds_xy=environment.map_bounds_xy,
            intent_speed_mps={
                intent: (
                    submarine_motion.max_speed_mps
                    if intent is HiddenIntent.EVADE
                    else submarine.speed_mps
                )
                for intent in HiddenIntent
            },
            max_speed_mps=submarine_motion.max_speed_mps,
            max_acceleration_mps2=submarine_motion.max_acceleration_mps2,
            max_turn_rate_rad_s=submarine_motion.max_turn_rate_rad_s,
            detection_range_m=submarine.detection_range_m,
        )
        self._contact_state[submarine.target_id] = {
            "classification": ContactClassification.SUBMARINE,
            "evidence": (),
            "position_xy": None,
        }
        self._target_detected_platform_ids[submarine.target_id] = ()
        self._rebuild_connectivity()

    def _capability_for(self, uuv_id: str) -> SurveillanceCapability:
        """Return one configured capability with legacy motion defaults."""
        configured = self._config.tracking.uuv_capabilities or {}
        capability = configured.get(uuv_id)
        if capability is not None:
            return capability
        return SurveillanceCapability(
            active_range_m=self._config.tracking.sensor_active_range_m,
            max_speed_mps=self._config.tracking.uuv_max_speed_mps,
            max_turn_rate_rad_s=self._config.tracking.uuv_max_turn_rate_rad_s,
        )

    def _allocate_and_create_groups(self) -> None:
        uuv_ids = tuple(sorted(self._uuvs))
        target_ids = tuple(sorted(self._targets))
        solution = allocate_groups(
            AllocationInput(
                uuv_ids=uuv_ids,
                target_ids=target_ids,
                quality_by_target={target_id: _INITIAL_QUALITY for target_id in target_ids},
                uuv_available={uuv_id: True for uuv_id in uuv_ids},
                uuv_energy_fraction={uuv_id: self._uuvs[uuv_id].energy_fraction for uuv_id in uuv_ids},
                quality_warning=self._config.tracking.quality_warning,
                quality_release=self._config.tracking.quality_release,
                release_hold_s=self._config.tracking.release_hold_s,
            )
        )
        self._assignments = dict(solution.members_by_target)
        for target_id, members in self._assignments.items():
            for uuv_id in members:
                self._uuv_groups[uuv_id] = target_id
        for target_id in target_ids:
            members = self._assignments[target_id]
            report = self._manager.create(
                target_id,
                scenario_id=self._scenario_id,
                group_id=f"G-{target_id}",
                member_ids=tuple(members),
                coarse_prior=_COARSE_PRIOR,
                member_positions={member: self._uuvs[member].position_xy for member in members},
            )
            self._latest_reports[target_id] = report
            self._last_guard_reasons[target_id] = report.quality.hard_guard_reasons
            self._event_counters[target_id] = 0

    def _initialize_explicit_groups(self) -> None:
        """Create one deterministic, truth-free bootstrap group per target."""
        environment = self._config.environment
        assert environment is not None
        tracking = self._config.tracking
        bootstrap_members = tuple(
            platform.platform_id
            for platform in sorted(environment.uuvs, key=lambda item: item.platform_index)[
                : tracking.group_min_size
            ]
        )
        map_min_x, map_max_x, map_min_y, map_max_y = environment.map_bounds_xy
        coarse_prior = (
            (map_min_x + map_max_x) / 2.0,
            (map_min_y + map_max_y) / 2.0,
        )
        for target_id in sorted(self._targets):
            report = self._manager.create(
                target_id,
                scenario_id=self._scenario_id,
                group_id=f"G-{target_id}",
                member_ids=bootstrap_members,
                coarse_prior=coarse_prior,
                member_positions={
                    member: self._uuvs[member].position_xy for member in bootstrap_members
                },
            )
            self._assignments[target_id] = report.member_ids
            self._latest_reports[target_id] = report
            self._last_guard_reasons[target_id] = report.quality.hard_guard_reasons
            self._event_counters[target_id] = 0

    def _process_mission_carrier_stops(self, sim_time_s: int) -> None:
        """Execute each reached deployment or recovery stop exactly once."""
        for carrier_id, carrier in sorted(self._carrier_entities.items()):
            stop_ids = self._mission_stop_ids.get(carrier_id, ())
            stop_indices = self._mission_stop_indices.get(carrier_id, ())
            stop_index_by_route_index = {
                route_index: stop_number
                for stop_number, route_index in enumerate(stop_indices)
            }
            for route_index in carrier.consume_arrived_mission_stop_indices():
                stop_index = stop_index_by_route_index.get(route_index, route_index - 1)
                if stop_index < 0 or stop_index >= len(stop_ids):
                    continue
                stop_id = stop_ids[stop_index]
                window = self._mission_stop_windows.get(carrier_id, {}).get(route_index)
                if window is not None and not window[0] <= sim_time_s <= window[1]:
                    self._carrier_events.append(
                        RuntimeEvent(
                            event_id=(
                                f"carrier_task_window_missed:{carrier_id}:"
                                f"{stop_id}:{sim_time_s}"
                            ),
                            scenario_id=self._scenario_id,
                            sim_time_s=sim_time_s,
                            event_type="carrier_task_window_missed",
                            entity_id=carrier_id,
                            level=EventLevel.STRATEGIC,
                            payload={
                                "task_id": stop_id,
                                "entry_s": window[0],
                                "exit_s": window[1],
                                "observed_s": sim_time_s,
                            },
                        )
                    )
                    continue
                task_type, separator, candidate_id = stop_id.partition(":")
                if not separator:
                    continue
                uuv_ids = self._mission_batch_by_candidate.get(
                    (carrier_id, candidate_id), ()
                )
                if task_type == "deploy":
                    dispatched: list[str] = []
                    for uuv_id in uuv_ids:
                        if self._deployment_states.get(uuv_id) is not DeploymentState.ONBOARD:
                            continue
                        self._uuvs[uuv_id].position_xy = carrier.position_xy
                        self._uuvs[uuv_id].heading_rad = carrier.heading_rad
                        self.request_uuv_deployment(uuv_id, reason=stop_id)
                        dispatched.append(uuv_id)
                        self._mission_dispatched_uuv_ids.add(uuv_id)
                    self._carrier_events.append(
                        RuntimeEvent(
                            event_id=f"carrier_dispatch_completed:{carrier_id}:{candidate_id}:{sim_time_s}",
                            scenario_id=self._scenario_id,
                            sim_time_s=sim_time_s,
                            event_type="carrier_dispatch_completed",
                            entity_id=carrier_id,
                            level=EventLevel.STRATEGIC,
                            payload={"candidate_id": candidate_id, "uuv_ids": tuple(dispatched)},
                        )
                    )
                elif task_type == "recover":
                    for uuv_id in uuv_ids:
                        if self._deployment_states.get(uuv_id) is DeploymentState.DEPLOYED:
                            self.request_uuv_recovery(uuv_id, reason=stop_id)

    def _advance_world(self, sim_time_s: int) -> None:
        dt_s = float(self._clock.step_s)
        tracking = self._config.tracking
        if self._uuv_only_runtime:
            for carrier in self._carrier_entities.values():
                carrier.step(dt_s, sim_time_s=sim_time_s)
            self._process_mission_carrier_stops(sim_time_s)
        elif self._platform_core_enabled:
            self._advance_usvs(dt_s)
        else:
            self._carrier_entity.step(dt_s)
        for uuv_id in sorted(self._uuvs):
            if self._uuv_statuses.get(uuv_id, UUVStatus.AVAILABLE) is UUVStatus.FAILED:
                continue
            uuv = self._uuvs[uuv_id]
            deployment_state = self._deployment_states[uuv_id]
            carrier = self._carrier_entities.get(
                self._uuv_carrier_ids.get(uuv_id, self._carrier_entity.carrier_id),
                self._carrier_entity,
            )
            if deployment_state is DeploymentState.ONBOARD:
                uuv.position_xy = carrier.position_xy
                uuv.heading_rad = carrier.heading_rad
                uuv.speed_mps = 0.0
                uuv.set_waypoints([])
                self._uuv_speeds[uuv_id] = 0.0
                continue
            if deployment_state is DeploymentState.RETURNING:
                uuv.set_waypoints([carrier.position_xy])
            if uuv_id in self._reserved_uuvs:
                # Bearing pursuit (R4): steer straight at the reserved
                # target's belief mean every physics step; the UUV is
                # exempt from plan commands.
                reserved_target = next(
                    (
                        target_id
                        for target_id, uuv_ids in self._reserved_by_target.items()
                        if uuv_id in uuv_ids
                    ),
                    None,
                )
                if reserved_target is not None:
                    report = self._latest_reports.get(reserved_target)
                    if report is not None:
                        uuv.set_waypoints(
                            [
                                (
                                    float(report.belief.mean[0]),
                                    float(report.belief.mean[1]),
                                )
                            ]
                        )
            before = uuv.position_xy
            limits = self._uuv_motion_limits.get(uuv_id)
            uuv.step(
                dt_s,
                limits.max_speed_mps if limits else tracking.uuv_max_speed_mps,
                limits.max_turn_rate_rad_s if limits else tracking.uuv_max_turn_rate_rad_s,
                limits.max_acceleration_mps2 if limits else None,
            )
            after = uuv.position_xy
            self._mission_distance_m[uuv_id] += hypot(
                after[0] - before[0], after[1] - before[1]
            )
            self._uuv_speeds[uuv_id] = (
                hypot(after[0] - before[0], after[1] - before[1]) / dt_s
            )
            if (
                deployment_state is DeploymentState.RETURNING
                and hypot(
                    uuv.position_xy[0] - carrier.position_xy[0],
                    uuv.position_xy[1] - carrier.position_xy[1],
                ) <= _RECOVERY_RADIUS_M
            ):
                self._complete_uuv_recovery(uuv_id, sim_time_s)
        for target_id in sorted(self._targets):
            target = self._targets[target_id]
            previous_intent = self._target_intents.get(target_id)
            target.step(dt_s, self._target_rng(target_id))
            if previous_intent is not None and target.intent is not previous_intent:
                self._events.append(
                    RuntimeEvent(
                        event_id=f"target_maneuver:{target_id}:{sim_time_s}",
                        scenario_id=self._scenario_id,
                        sim_time_s=sim_time_s,
                        event_type="target_maneuver",
                        entity_id=target_id,
                        level=EventLevel.TACTICAL,
                        payload={
                            "intent": target.intent.value,
                            "source": "target_public_belief",
                        },
                    )
                )
            self._target_intents[target_id] = target.intent
        for decoy_id in sorted(self._decoys):
            self._decoys[decoy_id].step(dt_s, self._decoy_rng(decoy_id))
        # Decoy bearing observations are collected every physics step (and
        # refreshed in the observation cycle), so decoy contacts with their
        # rays are visible from the very first frame.
        self._decoy_observations = self._observe_decoys(sim_time_s)
        self._process_pings(sim_time_s)
        if self._platform_core_enabled:
            self._rebuild_connectivity()
        self._update_target_detection_events(sim_time_s)

    def _update_target_detection_events(self, sim_time_s: int) -> None:
        """Emit target-owned detection transitions without exposing target truth."""
        deployed = tuple(
            state
            for state in (*self._usv_platform_states(), *self._uuv_platform_states())
            if state.deployment_state == "deployed"
        )
        for target_id, target in sorted(self._targets.items()):
            belief = target.adversary_belief(sim_time_s)
            detected = tuple(
                sorted(
                    state.platform_id
                    for state in deployed
                    if hypot(
                        state.position_xy[0] - belief.estimated_position_xy[0],
                        state.position_xy[1] - belief.estimated_position_xy[1],
                    ) <= target.detection_range_m
                )
            )
            previous = self._target_detected_platform_ids.get(target_id, ())
            if detected == previous:
                continue
            added = tuple(sorted(set(detected) - set(previous)))
            removed = tuple(sorted(set(previous) - set(detected)))
            if added:
                self._events.append(
                    RuntimeEvent(
                        event_id=f"target_detection_acquired:{target_id}:{sim_time_s}",
                        scenario_id=self._scenario_id,
                        sim_time_s=sim_time_s,
                        event_type="target_detection_acquired",
                        entity_id=target_id,
                        level=EventLevel.TACTICAL,
                        payload={"platform_ids": added},
                    )
                )
            if removed:
                self._events.append(
                    RuntimeEvent(
                        event_id=f"target_detection_lost:{target_id}:{sim_time_s}",
                        scenario_id=self._scenario_id,
                        sim_time_s=sim_time_s,
                        event_type="target_detection_lost",
                        entity_id=target_id,
                        level=EventLevel.TACTICAL,
                        payload={"platform_ids": removed},
                    )
                )
            self._target_detected_platform_ids[target_id] = detected

    def _advance_usvs(self, dt_s: float) -> None:
        previous_carrier_position = self._carrier_entity.position_xy
        previous_carrier_heading = self._carrier_entity.heading_rad
        previous_patrol_route_index = self._carrier_entity._next_corner_index
        previous_usv_states = {
            usv_id: (usv.motion, usv.energy_fraction, usv.command)
            for usv_id, usv in self._usvs.items()
        }
        try:
            self._carrier_entity.step(dt_s)
            carrier_xy = self._carrier_entity.position_xy
            for usv_id in sorted(self._usvs):
                deployment_state = self._usv_deployment_states[usv_id]
                usv = self._usvs[usv_id]
                if deployment_state is DeploymentState.ONBOARD:
                    usv.motion = MotionState(
                        position_xy=carrier_xy,
                        heading_rad=self._carrier_entity.heading_rad,
                        speed_mps=0.0,
                    )
                    usv.energy_fraction = max(
                        0.0, usv.energy_fraction - dt_s * usv.hotel_energy_per_s
                    )
                    continue
                if deployment_state is not DeploymentState.DEPLOYED:
                    continue
                if usv_id in self._usv_hold_ids:
                    usv.command = MotionCommand(
                        desired_heading_rad=usv.motion.heading_rad,
                        desired_speed_mps=0.0,
                    )
                    usv.motion = MotionState(
                        position_xy=usv.motion.position_xy,
                        heading_rad=usv.motion.heading_rad,
                        speed_mps=0.0,
                    )
                    usv.energy_fraction = max(
                        0.0, usv.energy_fraction - dt_s * usv.hotel_energy_per_s
                    )
                    continue
                waypoints = self._usv_waypoints.get(usv_id, [])
                while waypoints and hypot(
                    waypoints[0][0] - usv.motion.position_xy[0],
                    waypoints[0][1] - usv.motion.position_xy[1],
                ) <= 1.0:
                    waypoints.pop(0)
                if waypoints:
                    destination = waypoints[0]
                    dx = destination[0] - usv.motion.position_xy[0]
                    dy = destination[1] - usv.motion.position_xy[1]
                    desired_speed = usv.limits.max_speed_mps
                else:
                    dx = carrier_xy[0] - usv.motion.position_xy[0]
                    dy = carrier_xy[1] - usv.motion.position_xy[1]
                    distance = hypot(dx, dy)
                    desired_speed = min(self._carrier_entity.speed_mps, usv.limits.max_speed_mps)
                    if distance > 0.9 * self._carrier_entity.support_radius_m:
                        desired_speed = usv.limits.max_speed_mps
                usv.set_motion_command(
                    MotionCommand(
                        desired_heading_rad=atan2(dy, dx),
                        desired_speed_mps=desired_speed,
                    )
                )
                previous_motion = usv.motion
                previous_energy_fraction = usv.energy_fraction
                usv.step(dt_s)
                self._constrain_usv_to_carrier_support(
                    usv,
                    previous_motion=previous_motion,
                    previous_energy_fraction=previous_energy_fraction,
                    dt_s=dt_s,
                )
        except Exception:
            self._carrier_entity.position_xy = previous_carrier_position
            self._carrier_entity.heading_rad = previous_carrier_heading
            self._carrier_entity._next_corner_index = previous_patrol_route_index
            for usv_id, (motion, energy_fraction, command) in previous_usv_states.items():
                usv = self._usvs[usv_id]
                usv.motion = motion
                usv.energy_fraction = energy_fraction
                usv.command = command
            raise

    def _constrain_usv_to_carrier_support(
        self,
        usv: USVEntity,
        *,
        previous_motion: MotionState,
        previous_energy_fraction: float,
        dt_s: float,
    ) -> None:
        carrier_xy = self._carrier_entity.position_xy
        dx = usv.motion.position_xy[0] - carrier_xy[0]
        dy = usv.motion.position_xy[1] - carrier_xy[1]
        distance = hypot(dx, dy)
        support_radius = self._carrier_entity.support_radius_m
        if distance <= support_radius:
            return
        scale = support_radius / distance
        constrained_position = (
            carrier_xy[0] + dx * scale,
            carrier_xy[1] + dy * scale,
        )
        actual_distance = hypot(
            constrained_position[0] - previous_motion.position_xy[0],
            constrained_position[1] - previous_motion.position_xy[1],
        )
        actual_speed = actual_distance / dt_s
        actual_heading = (
            previous_motion.heading_rad
            if actual_distance <= 1e-9
            else wrap_angle(
                atan2(
                    constrained_position[1] - previous_motion.position_xy[1],
                    constrained_position[0] - previous_motion.position_xy[0],
                )
            )
        )
        turn_angle = abs(wrap_angle(actual_heading - previous_motion.heading_rad))
        if actual_speed > usv.limits.max_speed_mps + 1e-9 or (
            abs(actual_speed - previous_motion.speed_mps)
            > usv.limits.max_acceleration_mps2 * dt_s + 1e-9
        ):
            usv.motion = previous_motion
            usv.energy_fraction = previous_energy_fraction
            raise RuntimeError(
                f"carrier support constraint is infeasible for USV {usv.usv_id!r} "
                "within its motion limits"
            )
        if turn_angle > usv.limits.max_turn_rate_rad_s * dt_s + 1e-9:
            usv.motion = previous_motion
            usv.energy_fraction = previous_energy_fraction
            raise RuntimeError(
                f"carrier support constraint violates turn-rate limit for USV {usv.usv_id!r}"
            )
        usv.motion = MotionState(
            position_xy=constrained_position,
            heading_rad=actual_heading,
            speed_mps=actual_speed,
        )
        usv.energy_fraction = max(
            0.0,
            previous_energy_fraction
            - actual_distance * usv.transit_energy_per_m
            - dt_s * usv.hotel_energy_per_s,
        )

    def _rebuild_connectivity(self) -> None:
        nodes = tuple(
            ConnectivityNode(
                platform_id=state.platform_id,
                kind=state.capability.kind,
                position_xy=state.position_xy,
                surface_range_m=state.capability.communications.surface_range_m,
                acoustic_range_m=state.capability.communications.acoustic_range_m,
            )
            for state in (*self._usv_platform_states(), *self._uuv_platform_states())
            if state.deployment_state == "deployed"
        )
        self._connectivity = build_connectivity(
            carrier_id=self._carrier_entity.carrier_id,
            carrier_xy=self._carrier_entity.position_xy,
            nodes=nodes,
        )

    def _observation_cycle(self, sim_time_s: int) -> None:
        if self._platform_core_enabled:
            self._platform_core_observation_cycle(sim_time_s)
            return
        self._legacy_observation_cycle(sim_time_s)

    def _legacy_observation_cycle(self, sim_time_s: int) -> None:
        for target_id in sorted(self._targets):
            report = self._latest_reports[target_id]
            members = tuple(
                member
                for member in report.member_ids
                if member not in self._reserved_uuvs
                and self._deployment_states[member] is DeploymentState.DEPLOYED
            )
            target = self._targets[target_id]
            observations = tuple(
                observation
                for uuv_id in members
                if (
                    observation := self._sensor_observation(
                        target_id, uuv_id, sim_time_s, target.position_xy
                    )
                )
                is not None
            )
            pending_command = self._pending_group_commands.pop(target_id, None)
            positions = {member: self._uuvs[member].position_xy for member in members}
            fresh = self._manager.invoke(
                target_id,
                observations=observations,
                member_positions=positions,
                command=pending_command,
            )
            self._latest_reports[target_id] = fresh
            self._assignments[target_id] = fresh.member_ids
            self._synchronize_group_membership(target_id, fresh.member_ids)
            self._target_rays[target_id] = observations
            self._events.extend(self._guard_events(fresh))
        self._decoy_observations = self._observe_decoys(sim_time_s)
        # Ping-request trigger (A2, ruling 9): announce every contact that
        # is still unverified so the active-verification protocol can start.
        # Repeats across cycles are intended — consumers are idempotent.
        for contact_id, state in sorted(self._contact_state.items()):
            if state.get("classification") is not ContactClassification.UNVERIFIED:
                continue
            self._events.append(
                RuntimeEvent(
                    event_id=f"active_ping:{contact_id}:{sim_time_s}",
                    scenario_id=self._scenario_id,
                    sim_time_s=sim_time_s,
                    event_type="active_ping",
                    entity_id=contact_id,
                    level=EventLevel.INFORMATIONAL,
                    # The verification protocol receives only the contact id.
                    # Its nearest-node choice uses the latest operational
                    # estimate from SituationSnapshot.contacts; simulator
                    # truth never crosses this event boundary.
                    payload={"uuv_ids": ()},
                )
            )
        self._emit_belief_change_events(sim_time_s)
        self._record_belief_history(sim_time_s)
        self._plan_waypoints()
        self._update_fast_regional_replan_events(sim_time_s)
        self._advance_mission_controller(sim_time_s)
        if self._carrier is not None:
            self._carrier(self._build_situation(sim_time_s))

    def _platform_core_observation_cycle(self, sim_time_s: int) -> None:
        states = (*self._usv_platform_states(), *self._uuv_platform_states())
        nodes = tuple(
            SonarNode(state.platform_id, state.position_xy, state.capability.sonar)
            for state in states
            if state.deployment_state == "deployed"
        )
        observations: list[PassiveSonarObservation] = []
        for target_id, target in sorted(self._targets.items()):
            for node in nodes:
                rng_key = f"platform:{target_id}:{node.platform_id}"
                rng = self._observer_rngs.setdefault(
                    rng_key,
                    random.Random(self._seed ^ _stable_int(rng_key)),
                )
                quality_rng_key = f"quality:{rng_key}"
                quality_rng = self._quality_rngs.setdefault(
                    quality_rng_key,
                    random.Random(self._seed ^ _stable_int(quality_rng_key)),
                )
                detection_rng_key = f"detection:{rng_key}"
                detection_rng = self._entity_rngs.setdefault(
                    detection_rng_key,
                    random.Random(self._seed ^ _stable_int(detection_rng_key)),
                )
                clutter_rng_key = f"clutter:{rng_key}"
                clutter_rng = self._entity_rngs.setdefault(
                    clutter_rng_key,
                    random.Random(self._seed ^ _stable_int(clutter_rng_key)),
                )
                observations.extend(
                    make_passive_observations(
                        scenario_id=self._scenario_id,
                        sim_time_s=sim_time_s,
                        observer=node,
                        target_id=target_id,
                        target_xy=target.position_xy,
                        rng=rng,
                        quality_rng=quality_rng,
                        detection_rng=detection_rng,
                        clutter_rng=clutter_rng,
                        clutter_sensitivity=node.capability.clutter_sensitivity,
                        pd_curve=default_pd_curve,
                    )
                )
        self._platform_observations = tuple(observations)
        for target_id in sorted(self._latest_reports):
            report = self._latest_reports[target_id]
            member_positions = {
                member: self._uuvs[member].position_xy for member in report.member_ids
            }
            member_positions.update(
                {
                    state.platform_id: state.position_xy
                    for state in self._usv_platform_states()
                    if state.deployment_state == "deployed"
                }
            )
            bearings = tuple(
                self._bearing_from_passive_observation(observation)
                for observation in observations
                if observation.target_id == target_id
                and observation.observer_id in member_positions
                and not observation.is_false_alarm
            )
            fresh = self._manager.invoke(
                target_id,
                observations=bearings,
                member_positions=member_positions,
                command=self._pending_group_commands.pop(target_id, None),
            )
            self._latest_reports[target_id] = fresh
            self._assignments[target_id] = fresh.member_ids
            self._synchronize_group_membership(target_id, fresh.member_ids)
            # Keep every converted bearing, including USV bearings, in the
            # internal contact state. The public frame adapter can only
            # render observers whose UUV origin is in SituationSnapshot.uuvs.
            self._target_rays[target_id] = bearings
            self._events.extend(self._guard_events(fresh))
        self._emit_belief_change_events(sim_time_s)
        self._record_belief_history(sim_time_s)
        self._plan_waypoints()
        self._rebuild_connectivity()
        self._update_fast_regional_replan_events(sim_time_s)
        self._advance_mission_controller(sim_time_s)
        if self._carrier is not None:
            self._carrier(self._build_situation(sim_time_s))

    def apply_verified_mission_plan(self, plan: ExecutableMissionPlan) -> bool:
        """Validate and atomically install an executable UUV-only plan."""
        if self._mission_controller is None:
            return False
        if not self._uuv_only_runtime:
            return False
        physical_uuv_ids = set(self._uuvs)
        physical_carrier_ids = set(self._carrier_entities)
        planned_uuv_ids = set(plan.all_uuv_ids)
        if not planned_uuv_ids.issubset(physical_uuv_ids):
            return False
        if not set(plan.carrier_missions).issubset(physical_carrier_ids):
            return False
        if not self._validate_runtime_mission_resources(plan):
            return False
        batch_carrier_by_uuv: dict[str, str] = {}
        for batch in plan.batches:
            for uuv_id in batch.uuv_ids:
                previous = batch_carrier_by_uuv.setdefault(uuv_id, batch.carrier_id)
                if previous != batch.carrier_id:
                    return False

        route_missions = {
            carrier_id: mission.model_copy(deep=True)
            for carrier_id, mission in plan.carrier_missions.items()
        }
        task_planner = CarrierTaskPlanner(
            # The default one-metre grid is appropriate for small unit tests
            # but cannot expand an operational 16 km carrier sortie within a
            # bounded node budget.  Fifty metres preserves the configured
            # stop geometry while keeping production route generation bounded.
            route_planner=AStarRoutePlanner(grid_size_m=50.0),
        )
        try:
            tasks = task_planner.build_tasks(
                plan,
                tuple(route_missions.values()),
            )
        except ValueError:
            return False
        tasks_by_carrier: dict[str, tuple[str, ...]] = {}
        stop_indices_by_carrier: dict[str, tuple[int, ...]] = {}
        stop_windows_by_carrier: dict[str, dict[int, tuple[int, int]]] = {}
        for carrier_id in route_missions:
            tasks_by_carrier[carrier_id] = tuple(
                task.task_id
                for task in tasks
                if task.carrier_id == carrier_id
                and task.candidate_id
                in {
                    batch.candidate_id
                    for batch in plan.uuv_batches_by_carrier.get(carrier_id, ())
                }
            )
        for carrier_id, mission in route_missions.items():
            entity = self._carrier_entities[carrier_id]
            home_position = self._carrier_home_positions[carrier_id]
            route = mission.route_xy
            if route:
                if route[0] != entity.position_xy:
                    return False
                if route[-1] != home_position:
                    return False
                if mission.stop_indices:
                    stop_indices = mission.stop_indices
                elif len(route) == len(mission.stop_ids) + 2:
                    stop_indices = tuple(range(1, len(route) - 1))
                elif mission.stop_ids:
                    return False
                else:
                    stop_indices = ()
                if len(stop_indices) != len(mission.stop_ids):
                    return False
                if any(
                    index <= 0 or index >= len(route) - 1
                    for index in stop_indices
                ):
                    return False
                stop_windows = mission.stop_windows
                task_by_id = {
                    task.task_id: task
                    for task in tasks
                    if task.carrier_id == carrier_id
                }
                if not stop_windows and stop_indices:
                    try:
                        stop_windows = tuple(
                            (
                                task_by_id[task_id].entry_s,
                                task_by_id[task_id].exit_s,
                            )
                            for task_id in mission.stop_ids
                        )
                    except KeyError:
                        return False
                if stop_windows:
                    try:
                        route_tasks = tuple(
                            task_by_id[task_id].model_copy(
                                update={
                                    "entry_s": window[0],
                                    "exit_s": window[1],
                                }
                            )
                            for task_id, window in zip(
                                mission.stop_ids,
                                stop_windows,
                                strict=True,
                            )
                        )
                        task_planner.validate_route_windows(
                            RoutePlan(
                                points=route,
                                stop_points=tuple(route[index] for index in stop_indices),
                                stop_indices=stop_indices,
                                distance_m=0.0,
                            ),
                            route_tasks,
                            self._clock.sim_time_s,
                            entity.speed_mps,
                        )
                    except (KeyError, ValueError):
                        return False
                route_missions[carrier_id] = mission.model_copy(
                    update={
                        "stop_indices": stop_indices,
                        "stop_windows": stop_windows,
                    }
                )
                stop_indices_by_carrier[carrier_id] = stop_indices
                stop_windows_by_carrier[carrier_id] = {
                    route_index: window
                    for route_index, window in zip(
                        stop_indices,
                        stop_windows,
                        strict=True,
                    )
                }
            elif plan.uuv_batches_by_carrier.get(carrier_id):
                try:
                    carrier_plan = plan.model_copy(
                        update={
                            "uuv_batches_by_carrier": {
                                carrier_id: plan.uuv_batches_by_carrier.get(
                                    carrier_id, ()
                                )
                            }
                        }
                    )
                    generated = task_planner.build_routes(
                        carrier_plan,
                        (mission,),
                        current_positions={carrier_id: entity.position_xy},
                        home_positions={carrier_id: home_position},
                        map_bounds=self._config.environment.map_bounds_xy
                        if self._config.environment is not None
                        else (-10_000.0, 10_000.0, -10_000.0, 10_000.0),
                        current_time_s=self._clock.sim_time_s,
                        speed_mps_by_carrier={carrier_id: entity.speed_mps},
                    )[carrier_id]
                except ValueError:
                    return False
                route_missions[carrier_id] = generated
                tasks_by_carrier[carrier_id] = generated.stop_ids
                stop_indices_by_carrier[carrier_id] = generated.stop_indices
                stop_windows_by_carrier[carrier_id] = {
                    route_index: window
                    for route_index, window in zip(
                        generated.stop_indices,
                        generated.stop_windows,
                        strict=True,
                    )
                }

        inventory_carrier_by_uuv: dict[str, str] = {}
        for carrier_id, mission in route_missions.items():
            for uuv_id in (
                *mission.onboard_uuv_ids,
                *mission.ready_uuv_ids,
                *mission.reserved_uuv_ids,
                *mission.recoverable_uuv_ids,
            ):
                previous = inventory_carrier_by_uuv.get(uuv_id)
                if previous is not None and previous != carrier_id:
                    return False
                if uuv_id not in self._uuvs:
                    return False
                inventory_carrier_by_uuv[uuv_id] = carrier_id
        effective_plan = plan.model_copy(update={"carrier_missions": route_missions})
        if not self._mission_controller.apply_verified_plan(effective_plan):
            return False
        for uuv_id, carrier_id in {
            **inventory_carrier_by_uuv,
            **batch_carrier_by_uuv,
        }.items():
            self._uuv_carrier_ids[uuv_id] = carrier_id
            if self._deployment_states[uuv_id] is DeploymentState.ONBOARD:
                carrier = self._carrier_entities[carrier_id]
                self._uuvs[uuv_id].position_xy = carrier.position_xy
                self._uuvs[uuv_id].heading_rad = carrier.heading_rad
        for carrier_id, mission in route_missions.items():
            if mission.route_xy:
                self._carrier_entities[carrier_id].set_mission_route(
                    mission.route_xy,
                    stop_windows=stop_windows_by_carrier.get(carrier_id, {}),
                    home_xy=self._carrier_home_positions[carrier_id],
                )
        self._mission_plan = effective_plan
        self._mission_stop_ids = tasks_by_carrier
        self._mission_stop_indices = stop_indices_by_carrier
        self._mission_stop_windows = stop_windows_by_carrier
        batch_uuvs_by_candidate: dict[tuple[str, str], list[str]] = {}
        for batch in plan.batches:
            batch_uuvs_by_candidate.setdefault(
                (batch.carrier_id, batch.candidate_id), []
            ).extend(batch.uuv_ids)
        self._mission_batch_by_candidate = {
            batch_key: tuple(sorted(set(uuv_ids)))
            for batch_key, uuv_ids in sorted(batch_uuvs_by_candidate.items())
        }
        self._reconcile_uuv_mission_state()
        return True

    def _validate_runtime_mission_resources(
        self, plan: ExecutableMissionPlan
    ) -> bool:
        """Revalidate live UUV health, generations, and sortie endurance.

        Planning snapshots are immutable estimates.  A carrier command can
        arrive after an observation cycle, so the physical execution boundary
        must reject a stale plan before changing controller or carrier state.
        """
        if self._mission_controller is None:
            return False
        if not set(plan.resource_episode_by_uuv).issubset(self._uuvs):
            return False
        controller_snapshot = self._mission_controller.snapshot()
        live_resources = controller_snapshot.uuv_resources
        live_episodes = controller_snapshot.resource_episode_by_uuv
        expected_episodes = plan.resource_episode_by_uuv
        if any(
            expected_episodes.get(uuv_id) != live_episode
            for uuv_id, live_episode in live_episodes.items()
            if uuv_id in expected_episodes
        ):
            return False

        active_ids = {
            uuv_id
            for batch in plan.batches
            for uuv_id in batch.active_scan_uuv_ids
        }
        active_ids.update(
            uuv_id
            for assignment in plan.region_assignments
            for uuv_id in assignment.active_scan_uuv_ids
        )
        estimated_distance_by_uuv: dict[str, float] = {}
        for batch in plan.batches:
            deployment = batch.deployment_point
            recovery = batch.recovery_point
            for uuv_id in batch.uuv_ids:
                required_distance = 0.0
                current_position = self._uuvs[uuv_id].position_xy
                if deployment is not None:
                    required_distance += hypot(
                        deployment[0] - current_position[0],
                        deployment[1] - current_position[1],
                    )
                if deployment is not None and recovery is not None:
                    required_distance += hypot(
                        recovery[0] - deployment[0],
                        recovery[1] - deployment[1],
                    )
                estimated_distance_by_uuv[uuv_id] = max(
                    estimated_distance_by_uuv.get(uuv_id, 0.0),
                    required_distance,
                )

        for uuv_id in sorted(plan.all_uuv_ids):
            uuv = self._uuvs.get(uuv_id)
            if uuv is None:
                return False
            if self._uuv_statuses.get(uuv_id, UUVStatus.AVAILABLE) is UUVStatus.FAILED:
                return False
            if self._deployment_states.get(uuv_id) in {
                DeploymentState.FAILED,
                DeploymentState.RETURNING,
            }:
                return False
            if uuv.energy_fraction <= self._mission_controller.min_energy_fraction:
                return False
            mileage = self._mission_distance_m.get(uuv_id, 0.0)
            if mileage >= self._mission_controller.max_uuv_mileage_m:
                return False
            if uuv_id in active_ids and not uuv.capability.active_sonar_available:
                return False
            resource = live_resources.get(uuv_id)
            if resource is not None and (
                not resource.healthy
                or (uuv_id in active_ids and not resource.capability_active)
                or resource.energy_fraction <= self._mission_controller.min_energy_fraction
                or resource.mileage_m >= self._mission_controller.max_uuv_mileage_m
                or resource.deployment_state in {"failed", "returning"}
            ):
                return False
            expected_episode = expected_episodes.get(uuv_id)
            if (
                expected_episode is not None
                and uuv_id in live_episodes
                and live_episodes[uuv_id] != expected_episode
            ):
                return False

            required_distance = estimated_distance_by_uuv.get(uuv_id, 0.0)
            if required_distance <= 0.0:
                continue
            remaining_mileage = self._mission_controller.max_uuv_mileage_m - mileage
            if required_distance > remaining_mileage:
                return False
            motion_limit = self._uuv_motion_limits.get(uuv_id)
            max_speed = (
                motion_limit.max_speed_mps
                if motion_limit is not None
                else self._config.tracking.uuv_max_speed_mps
            )
            energy_cost_per_m = uuv.transit_energy_per_m + (
                uuv.hotel_energy_per_s / max(max_speed, 0.1)
            )
            required_energy = required_distance * energy_cost_per_m
            available_energy = uuv.energy_fraction - self._mission_controller.min_energy_fraction
            if required_energy > available_energy:
                return False
        return True

    def mission_snapshot(self) -> MissionSnapshot | None:
        """Return the controller snapshot when this is a UUV-only run."""
        return self._mission_controller.snapshot() if self._mission_controller else None

    def _mission_resource_episodes(self) -> dict[str, int]:
        """Expose controller resource generations in the planning snapshot."""
        if self._mission_controller is None:
            return {}
        return dict(self._mission_controller.snapshot().resource_episode_by_uuv)

    def carrier_states(self) -> dict[str, CarrierState]:
        """Return the current physical state for every carrier."""
        uuvs = tuple(self._situation_uuv_state(uuv_id) for uuv_id in sorted(self._uuvs))
        return {
            carrier_id: carrier.state_for(
                uuvs,
                tuple(
                    uuv_id
                    for uuv_id, assigned_carrier_id in self._uuv_carrier_ids.items()
                    if assigned_carrier_id == carrier_id
                ),
            )
            for carrier_id, carrier in sorted(self._carrier_entities.items())
        }

    def events(self) -> tuple[RuntimeEvent, ...]:
        """Return all public runtime events emitted so far."""
        return tuple(self._event_ledger)

    def mission_distance(self, uuv_id: str) -> float:
        if uuv_id not in self._mission_distance_m:
            raise ValueError(f"unknown uuv {uuv_id!r}")
        return self._mission_distance_m[uuv_id]

    def _reconcile_uuv_mission_state(self) -> None:
        controller = self._mission_controller
        if controller is None:
            return
        snapshot = controller.snapshot()
        for uuv_id, mode in snapshot.uuv_modes.items():
            if mode is UUVMissionMode.FAILED:
                if self._deployment_states.get(uuv_id) is not DeploymentState.FAILED:
                    self.fail_uuv(uuv_id)
            elif mode is UUVMissionMode.ACTIVE_SCAN:
                self.set_sensor_mode(uuv_id, "active")
            elif mode is UUVMissionMode.PASSIVE_TRACK:
                self.set_sensor_mode(uuv_id, "passive")
            elif mode is UUVMissionMode.RETURN_REQUIRED:
                if self._deployment_states.get(uuv_id) is DeploymentState.DEPLOYED:
                    self.request_uuv_recovery(uuv_id, reason="mission_controller")

    def _advance_mission_controller(self, sim_time_s: int) -> None:
        controller = self._mission_controller
        if controller is None:
            return
        failed = tuple(
            sorted(
                uuv_id
                for uuv_id, status in self._uuv_statuses.items()
                if status is UUVStatus.FAILED
            )
        )
        snapshot = controller.snapshot()
        deployed = tuple(
            sorted(
                uuv_id
                for uuv_id, state in self._deployment_states.items()
                if state is DeploymentState.DEPLOYED
            )
        )
        deployed_by_region = {
            region.region_id: tuple(
                uuv_id
                for uuv_id in (
                    *region.active_scan_uuv_ids,
                    *region.passive_track_uuv_ids,
                )
                if uuv_id in deployed
            )
            for region in snapshot.regions
        }
        deployed_ids = set(deployed)
        entry_probability = self._mission_entry_probabilities(sim_time_s, snapshot)
        handoff_ready, successor_passive_ready = self._mission_handoff_observations(
            snapshot,
            deployed_ids,
        )
        target_intent_changed = self._latest_mission_event_value(
            "target_intent_changed", sim_time_s
        )
        imm_confidence_shifted = self._latest_mission_event_value(
            "imm_confidence_shifted", sim_time_s
        )
        target_exit_predicted = self._mission_exit_prediction(sim_time_s, snapshot)
        carrier_dispatch_completed = self._latest_mission_event_value(
            "carrier_dispatch_completed", sim_time_s
        )
        updated = controller.advance(
            sim_time_s,
            {
                "deployed_uuv_ids": deployed_by_region,
                "failed_uuv_ids": failed,
                "mileage_m": dict(self._mission_distance_m),
                "energy_fraction": {
                    uuv_id: self._uuvs[uuv_id].energy_fraction
                    for uuv_id in sorted(self._uuvs)
                },
                "deployment_state": {
                    uuv_id: self._deployment_states[uuv_id].value
                    for uuv_id in sorted(self._uuvs)
                },
                "uuv_capability_active": {
                    uuv_id: self._uuvs[uuv_id].capability.active_sonar_available
                    for uuv_id in sorted(self._uuvs)
                },
                "uuv_health": {
                    uuv_id: self._uuv_statuses.get(uuv_id, UUVStatus.AVAILABLE)
                    is not UUVStatus.FAILED
                    for uuv_id in sorted(self._uuvs)
                },
                "recovered_uuv_ids": tuple(sorted(self._mission_recovered_uuv_ids)),
                "health_check_passed": {
                    uuv_id: True for uuv_id in sorted(self._mission_recovered_uuv_ids)
                },
                "recovery_requested_uuv_ids": tuple(
                    sorted(self._mission_recovery_requested_uuv_ids)
                ),
                "recovering_uuv_ids": tuple(
                    sorted(
                        uuv_id
                        for uuv_id, state in self._deployment_states.items()
                        if state is DeploymentState.RETURNING
                    )
                ),
                "entry_probability": entry_probability,
                "handoff_ready": handoff_ready,
                "successor_passive_ready": successor_passive_ready,
                **(
                    {"target_intent_changed": target_intent_changed}
                    if target_intent_changed is not None
                    else {}
                ),
                **(
                    {"imm_confidence_shifted": imm_confidence_shifted}
                    if imm_confidence_shifted is not None
                    else {}
                ),
                **(
                    {"target_exit_predicted": target_exit_predicted}
                    if target_exit_predicted is not None
                    else {}
                ),
                **(
                    {"carrier_dispatch_completed": carrier_dispatch_completed}
                    if carrier_dispatch_completed is not None
                    else {}
                ),
            },
        )
        self._mission_recovered_uuv_ids.clear()
        self._mission_recovery_requested_uuv_ids.clear()
        new_events = [
            event
            for event in updated.events
            if event.event_id not in self._mission_controller_event_ids
        ]
        self._mission_controller_event_ids.update(event.event_id for event in new_events)
        self._events.extend(new_events)
        self._reconcile_uuv_mission_state()

    def _mission_time_windows(self) -> dict[str, tuple[int, int]]:
        """Return the merged estimated service window for each active candidate."""
        if self._mission_plan is None:
            return {}
        windows: dict[str, tuple[int, int]] = {}
        for batch in self._mission_plan.batches:
            previous = windows.get(batch.candidate_id)
            if previous is None:
                windows[batch.candidate_id] = (batch.entry_s, batch.exit_s)
            else:
                windows[batch.candidate_id] = (
                    min(previous[0], batch.entry_s),
                    max(previous[1], batch.exit_s),
                )
        return windows

    def _mission_entry_probabilities(
        self,
        sim_time_s: int,
        snapshot: MissionSnapshot,
    ) -> dict[str, float]:
        """Project planner-owned candidate windows into a truth-free entry estimate."""
        windows = self._mission_time_windows()
        return {
            region.region_id: (
                1.0
                if (
                    (window := windows.get(region.region_id)) is not None
                    and window[0] <= sim_time_s <= window[1]
                )
                else 0.0
            )
            for region in snapshot.regions
        }

    def _mission_handoff_observations(
        self,
        snapshot: MissionSnapshot,
        deployed_ids: set[str],
    ) -> tuple[dict[str, str], dict[str, bool]]:
        """Return handoff readiness derived from deployed successor assignments."""
        successor_ready: dict[str, bool] = {}
        for region in snapshot.regions:
            required = {
                *region.active_scan_uuv_ids,
                *region.passive_track_uuv_ids,
            }
            successor_ready[region.region_id] = bool(required) and required.issubset(
                deployed_ids
            )
        handoffs = {
            region.region_id: region.handoff_to
            for region in snapshot.regions
            if region.lifecycle is RegionLifecycle.PASSIVE_TRACK
            and region.handoff_to is not None
            and successor_ready.get(region.handoff_to, False)
        }
        return handoffs, successor_ready

    def _mission_exit_prediction(
        self,
        sim_time_s: int,
        snapshot: MissionSnapshot,
    ) -> str | None:
        """Return the first region whose estimated service window has elapsed."""
        windows = self._mission_time_windows()
        for region in snapshot.regions:
            window = windows.get(region.region_id)
            if (
                window is not None
                and sim_time_s > window[1]
                and region.lifecycle
                in {
                    RegionLifecycle.ACTIVE_SCAN,
                    RegionLifecycle.PASSIVE_TRACK,
                }
            ):
                return region.region_id
        return None

    def _latest_mission_event_value(
        self,
        event_type: str,
        sim_time_s: int,
    ) -> Mapping[str, object] | None:
        for event in reversed((*self._carrier_events, *self._events)):
            if event.event_type == event_type and event.sim_time_s <= sim_time_s:
                value: dict[str, object] = {
                    "event_id": event.event_id,
                    **event.payload,
                }
                if event.entity_id is not None:
                    value["entity_id"] = event.entity_id
                elif "target_id" in event.payload:
                    value["entity_id"] = str(event.payload["target_id"])
                return value
        return None

    @staticmethod
    def _bearing_from_passive_observation(
        observation: PassiveSonarObservation,
    ) -> BearingObservation:
        """Adapt the public passive observation to the group graph contract."""
        return BearingObservation(
            observation_id=observation.observation_id,
            scenario_id=observation.scenario_id,
            sim_time_s=observation.sim_time_s,
            uuv_id=observation.observer_id,
            target_id=observation.target_id,
            azimuth_rad=observation.azimuth_rad,
            variance_rad2=observation.variance_rad2,
            detection_confidence=observation.detection_confidence,
            is_false_alarm=observation.is_false_alarm,
        )

    def _observe_decoys(
        self, sim_time_s: int
    ) -> dict[str, tuple[BearingObservation, ...]]:
        """Passive bearing observations of every decoy by the fleet."""
        rays_by_decoy: dict[str, tuple[BearingObservation, ...]] = {}
        for decoy_id in sorted(self._decoys):
            rays: list[BearingObservation] = []
            for uuv_id in sorted(self._uuvs):
                if uuv_id in self._reserved_uuvs:
                    continue
                observation = self._sensor_observation(
                    decoy_id, uuv_id, sim_time_s, self._decoys[decoy_id].position_xy
                )
                if observation is not None:
                    rays.append(observation)
            rays_by_decoy[decoy_id] = tuple(rays)
        return rays_by_decoy

    def _publish_reports(self, sim_time_s: int) -> None:
        """Publish the latest group reports at the 300 s cadence.

        The reports themselves are already retained by the engine and
        carried by every frame (latest known); publishing records one
        informational event per group so the cadence is observable.
        """
        for target_id in sorted(self._latest_reports):
            report = self._latest_reports[target_id]
            self._events.append(
                RuntimeEvent(
                    event_id=f"{report.group_id}:report_published:{sim_time_s}",
                    scenario_id=self._scenario_id,
                    sim_time_s=sim_time_s,
                    event_type="group_report_published",
                    entity_id=report.group_id,
                    level=EventLevel.INFORMATIONAL,
                    payload={},
                )
            )

    def _guard_events(self, report: GroupReport) -> tuple[RuntimeEvent, ...]:
        """Turn newly appeared quality hard guards into runtime events."""
        previous = self._last_guard_reasons[report.target_id]
        new_reasons = [
            reason for reason in report.quality.hard_guard_reasons if reason not in previous
        ]
        events: list[RuntimeEvent] = []
        for reason in new_reasons:
            events.append(
                RuntimeEvent(
                    event_id=(
                        f"{report.group_id}:quality_guard:{reason}:"
                        f"{self._event_counters[report.target_id]}"
                    ),
                    scenario_id=self._scenario_id,
                    sim_time_s=report.sim_time_s,
                    event_type=f"quality_guard:{reason}",
                    entity_id=report.group_id,
                    level=EventLevel.TACTICAL,
                    payload={},
                )
            )
            self._event_counters[report.target_id] += 1
        self._last_guard_reasons[report.target_id] = report.quality.hard_guard_reasons
        return tuple(events)

    def _sensor_observation(
        self,
        target_id: str,
        uuv_id: str,
        sim_time_s: int,
        target_xy: tuple[float, float],
    ) -> BearingObservation | None:
        """One noisy bearing observation, or None inside the sensor blind zone.

        A failed UUV no longer senses: it returns ``None`` regardless of
        geometry, so the group update proceeds without its bearing.
        """
        if self._uuv_statuses.get(uuv_id, UUVStatus.AVAILABLE) is UUVStatus.FAILED:
            return None
        if self._deployment_states[uuv_id] is not DeploymentState.DEPLOYED:
            return None
        uuv = self._uuvs[uuv_id]
        standoff = hypot(target_xy[0] - uuv.position_xy[0], target_xy[1] - uuv.position_xy[1])
        if standoff < _SENSOR_MIN_RANGE_M or standoff > uuv.capability.passive_range_m:
            return None
        sonar = SonarCapability(
            passive_range_m=uuv.capability.passive_range_m,
            passive_bearing_variance_rad2=uuv.capability.bearing_variance_rad2,
            active_source_range_m=uuv.capability.active_range_m,
            active_receive_range_m=uuv.capability.active_range_m,
            active_range_sigma_m=self._config.tracking.sensor_active_range_sigma_m,
            active_bearing_sigma_rad=0.01,
            active_capable=uuv.capability.active_sonar_available,
            ping_cooldown_s=self._config.tracking.sensor_ping_interval_s,
            ping_energy_cost_fraction=self._config.tracking.sensor_ping_energy_cost,
            clutter_sensitivity=0.0,
            exposure_cost=0.0,
        )
        node = SonarNode(uuv_id, uuv.position_xy, sonar)
        rng_key = f"legacy:{target_id}:{uuv_id}"
        rng = self._observer_rngs.setdefault(
            rng_key, random.Random(self._seed ^ _stable_int(rng_key))
        )
        quality_rng_key = f"quality:{rng_key}"
        quality_rng = self._quality_rngs.setdefault(
            quality_rng_key, random.Random(self._seed ^ _stable_int(quality_rng_key))
        )
        detection_rng_key = f"detection:{rng_key}"
        detection_rng = self._entity_rngs.setdefault(
            detection_rng_key,
            random.Random(self._seed ^ _stable_int(detection_rng_key)),
        )
        observation = make_passive_observation(
            scenario_id=self._scenario_id,
            sim_time_s=sim_time_s,
            observer=node,
            target_id=target_id,
            target_xy=target_xy,
            rng=rng,
            quality_rng=quality_rng,
            detection_rng=detection_rng,
            pd_curve=default_pd_curve,
        )
        if observation is None:
            return None
        return BearingObservation(
            observation_id=observation.observation_id,
            scenario_id=observation.scenario_id,
            sim_time_s=observation.sim_time_s,
            uuv_id=observation.observer_id,
            target_id=observation.target_id,
            azimuth_rad=observation.azimuth_rad,
            variance_rad2=observation.variance_rad2,
            detection_confidence=observation.detection_confidence,
            is_false_alarm=observation.is_false_alarm,
        )

    def _process_pings(self, sim_time_s: int) -> None:
        """Execute bounded active sonar for UUVs and USV relay nodes."""
        tracking = self._config.tracking
        for platform_id, contact_id in sorted(self._ping_targets.items()):
            if self._sensor_modes.get(platform_id) != "active" or contact_id is None:
                continue
            deployment_state = (
                self._deployment_states.get(platform_id)
                if platform_id in self._deployment_states
                else self._usv_deployment_states.get(platform_id)
            )
            if deployment_state is not DeploymentState.DEPLOYED:
                continue
            uuv = self._uuvs.get(platform_id)
            usv = self._usvs.get(platform_id)
            if uuv is not None:
                source_xy = uuv.position_xy
                active_available = uuv.capability.active_sonar_available
                active_range_m = uuv.capability.active_range_m
                ping_cooldown_s = tracking.sensor_ping_interval_s
                ping_energy = tracking.sensor_ping_energy_cost
                current_energy = uuv.energy_fraction
            elif usv is not None:
                capability = self._usv_capabilities[platform_id]
                source_xy = usv.motion.position_xy
                active_available = capability.sonar.active_capable
                active_range_m = capability.sonar.active_source_range_m
                ping_cooldown_s = capability.sonar.ping_cooldown_s
                ping_energy = capability.sonar.ping_energy_cost_fraction
                current_energy = usv.energy_fraction
            else:
                continue
            last = self._last_ping_times.get((platform_id, contact_id))
            if last is not None and sim_time_s - last < ping_cooldown_s:
                continue
            contact = self._contact_state.get(contact_id)
            if contact is None or not active_available or current_energy <= 0.0:
                continue
            contact_xy = contact.get("position_xy")
            if contact_xy is None:
                report = self._latest_reports.get(contact_id)
                contact_xy = (
                    (float(report.belief.mean[0]), float(report.belief.mean[1]))
                    if report is not None
                    else None
                )
            if contact_xy is None and contact_id in self._decoys:
                contact_xy = self._decoys[contact_id].position_xy
            if contact_xy is None:
                continue
            range_m = hypot(contact_xy[0] - source_xy[0], contact_xy[1] - source_xy[1])
            self._last_ping_times[(platform_id, contact_id)] = sim_time_s
            if range_m > active_range_m or range_m < _SENSOR_MIN_RANGE_M:
                continue
            azimuth_rad = atan2(contact_xy[1] - source_xy[1], contact_xy[0] - source_xy[0])
            rng = self._entity_rngs.setdefault(
                f"active:{platform_id}:{contact_id}",
                random.Random(self._seed ^ _stable_int(f"active:{platform_id}:{contact_id}")),
            )
            is_decoy = contact_id in self._decoys
            self._events.append(
                RuntimeEvent(
                    event_id=f"active_ping:{platform_id}:{contact_id}:{sim_time_s}",
                    scenario_id=self._scenario_id,
                    sim_time_s=sim_time_s,
                    event_type="active_ping",
                    entity_id=contact_id,
                    level=EventLevel.INFORMATIONAL,
                    payload={
                        "emitter_id": platform_id,
                        "contact_id": contact_id,
                        "range_m": round(range_m, 1),
                        "azimuth_rad": round(azimuth_rad, 6),
                        "receiver_ids": self._ping_receivers.get(
                            platform_id, (platform_id,)
                        ),
                        **(
                            {"uuv_id": platform_id, "uuv_ids": (platform_id,)}
                            if uuv is not None
                            else {}
                        ),
                    },
                )
            )
            if uuv is not None:
                uuv.energy_fraction = max(0.0, current_energy - ping_energy)
            else:
                assert usv is not None
                usv.energy_fraction = max(0.0, current_energy - ping_energy)
            if rng.random() > tracking.sensor_ping_heard_probability:
                continue
            state = self._contact_state[contact_id]
            classification = state.get("classification", ContactClassification.UNVERIFIED)
            if classification is ContactClassification.UNVERIFIED:
                classify_prob = (
                    tracking.sensor_active_classify_decoy_prob
                    if is_decoy
                    else tracking.sensor_active_classify_submarine_prob
                )
                classification = (
                    ContactClassification.DECOY
                    if is_decoy and rng.random() < classify_prob
                    else ContactClassification.SUBMARINE
                    if not is_decoy and rng.random() < classify_prob
                    else ContactClassification.SUBMARINE
                    if is_decoy
                    else ContactClassification.DECOY
                )
                state["classification"] = classification
                state["evidence"] = (
                    *state.get("evidence", ()),
                    f"ping:{platform_id}:{contact_id}:{sim_time_s}",
                )
            true_xy = (
                self._decoys[contact_id].position_xy
                if is_decoy
                else self._targets[contact_id].position_xy
            )
            self._contact_state[contact_id]["position_xy"] = (
                float(true_xy[0] + rng.gauss(0.0, tracking.sensor_active_range_sigma_m)),
                float(true_xy[1] + rng.gauss(0.0, tracking.sensor_active_range_sigma_m)),
            )
            if classification is ContactClassification.SUBMARINE and not is_decoy:
                self._targets[contact_id].apply_evasive_maneuver(
                    tracking.submarine_turn_rate_rad_s * tracking.sensor_ping_interval_s
                )
            self._events.append(
                RuntimeEvent(
                    event_id=f"contact_classified:{contact_id}:{platform_id}:{sim_time_s}",
                    scenario_id=self._scenario_id,
                    sim_time_s=sim_time_s,
                    event_type="contact_classified",
                    entity_id=contact_id,
                    level=EventLevel.INFORMATIONAL,
                    payload={
                        "contact_id": contact_id,
                        "classification": classification.value,
                        "platform_id": platform_id,
                        "outcome": classification.value,
                        **(
                            {"uuv_id": platform_id}
                            if uuv is not None
                            else {}
                        ),
                    },
                )
            )

    def _decoy_rng(self, decoy_id: str) -> random.Random:
        rng = self._entity_rngs.get(decoy_id)
        if rng is None:
            rng = random.Random(self._seed ^ _stable_int(decoy_id))
            self._entity_rngs[decoy_id] = rng
        return rng

    def _contacts(self) -> tuple[Contact, ...]:
        """Operational contacts: dispatched tracked targets plus decoys."""
        contacts: list[Contact] = []
        for target_id in sorted(self._targets):
            report = self._latest_reports.get(target_id)
            sim_time_s = report.sim_time_s if report is not None else 0
            state = self._contact_state.get(target_id, {})
            contacts.append(
                Contact(
                    contact_id=target_id,
                    sim_time_s=sim_time_s,
                    bearing_rays=self._target_rays.get(target_id, ()),
                    classification=state.get(
                        "classification", ContactClassification.SUBMARINE
                    ),
                    classification_evidence=tuple(state.get("evidence", ())),
                    estimated_position_xy=state.get("position_xy"),
                )
            )
        for decoy_id, rays in sorted(self._decoy_observations.items()):
            state = self._contact_state.get(decoy_id, {})
            contacts.append(
                Contact(
                    contact_id=decoy_id,
                    sim_time_s=0,
                    bearing_rays=rays,
                    classification=state.get(
                        "classification", ContactClassification.UNVERIFIED
                    ),
                    classification_evidence=tuple(state.get("evidence", ())),
                    estimated_position_xy=state.get("position_xy"),
                )
            )
        return tuple(contacts)

    def _plan_waypoints(self) -> None:
        for target_id in sorted(self._latest_reports):
            report = self._latest_reports[target_id]
            members = tuple(
                member
                for member in report.member_ids
                if self._deployment_states[member] is DeploymentState.DEPLOYED
            )
            if not members:
                self._waypoint_commands[target_id] = {}
                continue
            positions = np.asarray(
                [[self._uuvs[member].position_xy[0], self._uuvs[member].position_xy[1]] for member in members],
                dtype=float,
            )
            if not self._track_converged(report.belief):
                hold_commands = self._hold_spread_commands(members, positions)
                for member, point in hold_commands.items():
                    self._uuvs[member].set_waypoints([point])
                self._previous_waypoints[target_id] = positions
                self._waypoint_commands[target_id] = hold_commands
                continue
            previous_waypoints = self._previous_waypoints.get(target_id)
            if previous_waypoints is not None and previous_waypoints.shape != positions.shape:
                previous_waypoints = None
            plan = plan_group_waypoints(
                positions,
                self._belief_sigma_points_xy(report.belief),
                previous_waypoints=previous_waypoints,
                max_step_m=_WAYPOINT_MAX_STEP_M,
                min_separation_m=_WAYPOINT_MIN_SEPARATION_M,
                bearing_variance=_BEARING_VARIANCE_RAD2,
                beam_width=_WAYPOINT_BEAM_WIDTH,
                uuv_ids=members,
                min_range_m=_SENSOR_MIN_RANGE_M,
            )
            raw_commands = {
                member: (tuple((float(point[0]), float(point[1])) for point in plan.waypoints_xy[index : index + 1]))
                for index, member in enumerate(members)
            }
            tracking = self._config.tracking
            if tracking.formation_enabled and len(members) >= 2:
                mean = report.belief.mean
                velocity = (
                    (float(mean[2]), float(mean[3]))
                    if len(mean) >= 4
                    else (0.0, 0.0)
                )
                formation = apply_formation_correction(
                    member_ids=members,
                    waypoints_by_member=raw_commands,
                    target_position=(float(mean[0]), float(mean[1])),
                    target_velocity=velocity,
                    target_heading_rad=None,
                    radius_m=tracking.formation_radius_m,
                    horizon_s=tracking.formation_horizon_s,
                    maximum_endpoint_correction_m=tracking.formation_max_endpoint_correction_m,
                    bounds_xy=(
                        self._config.environment.map_bounds_xy
                        if self._config.environment is not None
                        else None
                    ),
                )
                raw_commands = formation.waypoints_by_member
            self._previous_waypoints[target_id] = np.asarray(
                [raw_commands[member][-1] for member in members], dtype=float
            )
            commands: dict[str, tuple[float, float]] = {}
            for member in members:
                point = raw_commands[member][-1]
                commands[member] = (float(point[0]), float(point[1]))
                self._uuvs[member].set_waypoints([commands[member]])
            self._waypoint_commands[target_id] = commands

    def _track_converged(self, belief: TargetBelief) -> bool:
        """True once the position estimate is tight enough to maneuver safely."""
        position_std = float(np.sqrt((belief.covariance[0][0] + belief.covariance[1][1]) / 2.0))
        return position_std <= _TRACK_CONVERGENCE_STD_M

    def _hold_spread_commands(
        self, members: tuple[str, ...], positions: np.ndarray[Any, Any]
    ) -> dict[str, tuple[float, float]]:
        """Re-disperse a group whose track is not converged.

        Each member is commanded onto a ring of ``_HOLD_SPREAD_RADIUS_M``
        around the group's current centroid, at evenly spaced bearings in
        the members' angular order, so a bunched group re-establishes its
        triangulation baseline without chasing the (unreliable) belief.
        """
        centroid = positions.mean(axis=0)
        bearings = [atan2(float(position[1] - centroid[1]), float(position[0] - centroid[0]))
                    for position in positions]
        order = sorted(range(len(members)), key=lambda index: bearings[index])
        commands: dict[str, tuple[float, float]] = {}
        for slot, index in enumerate(order):
            angle = 2.0 * pi * slot / len(members)
            point = (
                float(centroid[0] + _HOLD_SPREAD_RADIUS_M * cos(angle)),
                float(centroid[1] + _HOLD_SPREAD_RADIUS_M * sin(angle)),
            )
            commands[members[index]] = point
        return commands

    def _belief_sigma_points_xy(self, belief: TargetBelief) -> np.ndarray[Any, Any]:
        """2-D projections of the belief's scaled-unscented sigma points."""
        filter_ = UnscentedInformationFilter(
            mean=np.asarray(belief.mean, dtype=float),
            covariance=np.asarray(belief.covariance, dtype=float),
            process_noise=DEFAULT_PROCESS_NOISE,
        )
        return filter_.sigma_points()[:, :2]

    def _intent_speed_mps(self) -> dict[HiddenIntent, float]:
        """Configured per-intent target speeds (cruise/sprint, R2)."""
        tracking = self._config.tracking
        return {
            HiddenIntent.TRANSIT: tracking.submarine_cruise_speed_mps,
            HiddenIntent.PATROL: tracking.submarine_cruise_speed_mps,
            HiddenIntent.LOITER: tracking.submarine_cruise_speed_mps,
            HiddenIntent.EVADE: tracking.submarine_sprint_speed_mps,
            HiddenIntent.APPROACH: tracking.submarine_cruise_speed_mps,
            HiddenIntent.WITHDRAW: tracking.submarine_cruise_speed_mps,
        }

    def _target_rng(self, target_id: str) -> random.Random:
        rng = self._entity_rngs.get(target_id)
        if rng is None:
            rng = random.Random(self._seed ^ _stable_int(target_id))
            self._entity_rngs[target_id] = rng
        return rng

    def _build_frame(self, sim_time_s: int) -> dict[str, object]:
        reports = self._sorted_reports()
        uuvs = tuple(self._uuv_state(uuv_id) for uuv_id in sorted(self._uuvs))
        explicit_uuvs = self._uuv_platform_states()
        frame_uuvs = (
            [state.model_dump() for state in explicit_uuvs]
            if self._platform_core_enabled
            else [uuv.model_dump() for uuv in uuvs]
        )
        carrier = (
            self._carrier_platform_state().model_dump()
            if self._platform_core_enabled
            else self._carrier_entity.state_for(uuvs).model_dump()
        )
        frame: dict[str, object] = {
            "run_id": self._run_id,
            "scenario_id": self._scenario_id,
            "sim_time_s": sim_time_s,
            "step_index": self._step_index,
            "uuvs": frame_uuvs,
            "carrier": carrier,
            "platform_core": self._platform_core_enabled,
            "communication_links": [link.model_dump() for link in self._connectivity.links],
            "sonar_observations": [
                observation.model_dump() for observation in self._platform_observations
            ],
            "group_reports": [report.model_dump() for report in reports],
            "tracks": [report.belief.model_dump() for report in reports],
            "quality": [
                {"target_id": report.target_id, **report.quality.model_dump()}
                for report in reports
            ],
            "assignments": {
                target_id: [
                    uuv_id
                    for uuv_id in members
                    if self._deployment_states[uuv_id] is DeploymentState.DEPLOYED
                ]
                for target_id, members in sorted(self._assignments.items())
            },
            "contacts": [contact.model_dump() for contact in self._contacts()],
            "reservations": {
                target_id: list(uuv_ids)
                for target_id, uuv_ids in sorted(self._reserved_by_target.items())
            },
            "events": [event.model_dump() for event in self._events],
            "waypoint_commands": {
                target_id: {
                    uuv_id: [x, y] for uuv_id, (x, y) in sorted(commands.items())
                }
                for target_id, commands in sorted(self._waypoint_commands.items())
            },
        }
        if not self._uuv_only_runtime:
            frame["usvs"] = [
                state.model_dump() for state in self._usv_platform_states()
            ]
        return frame

    def _usv_platform_states(self) -> tuple[USVPlatformState, ...]:
        return tuple(
            USVPlatformState(
                platform_id=usv_id,
                platform_index=usv.platform_index,
                position_xy=usv.motion.position_xy,
                heading_rad=usv.motion.heading_rad,
                speed_mps=usv.motion.speed_mps,
                energy_fraction=usv.energy_fraction,
                deployment_state=self._usv_deployment_states[usv_id].value,
                capability=self._usv_capabilities[usv_id],
                sensor_mode=self._sensor_modes.get(usv_id, "passive"),
                distance_to_carrier_m=hypot(
                    usv.motion.position_xy[0] - self._carrier_entity.position_xy[0],
                    usv.motion.position_xy[1] - self._carrier_entity.position_xy[1],
                ),
            )
            for usv_id, usv in sorted(self._usvs.items())
        )

    def _uuv_platform_states(self) -> tuple[UUVPlatformState, ...]:
        if not self._platform_core_enabled:
            return ()
        return tuple(
            UUVPlatformState(
                platform_id=uuv_id,
                platform_index=uuv.platform_index,
                position_xy=uuv.position_xy,
                heading_rad=uuv.heading_rad,
                speed_mps=uuv.speed_mps,
                energy_fraction=uuv.energy_fraction,
                deployment_state=self._deployment_states[uuv_id].value,
                capability=self._uuv_platform_capabilities[uuv_id],
                group_id=self._uuv_groups.get(uuv_id),
                sensor_mode=self._sensor_modes.get(uuv_id, "passive"),
                is_group_leader=(
                    self._uuv_groups.get(uuv_id) is not None
                    and uuv_id
                    == max(
                        member
                        for member, target_id in self._uuv_groups.items()
                        if target_id == self._uuv_groups[uuv_id]
                    )
                ),
                master_connected=has_path(
                    self._connectivity,
                    self._uuv_carrier_ids.get(uuv_id, self._carrier_entity.carrier_id),
                    uuv_id,
                ),
            )
            for uuv_id, uuv in sorted(self._uuvs.items())
        )

    def _carrier_platform_state_for(self, carrier_id: str) -> CarrierPlatformState:
        states = (*self._usv_platform_states(), *self._uuv_platform_states())
        if self._uuv_only_runtime:
            states = tuple(
                state
                for state in states
                if state.platform_id in self._usvs
                or self._uuv_carrier_ids.get(state.platform_id) == carrier_id
            )
        by_state = {
            deployment: tuple(
                sorted(
                    state.platform_id
                    for state in states
                    if state.deployment_state == deployment
                )
            )
            for deployment in ("onboard", "deployed", "returning")
        }
        carrier = self._carrier_entities[carrier_id]
        return CarrierPlatformState(
            carrier_id=carrier.carrier_id,
            position_xy=carrier.position_xy,
            heading_rad=carrier.heading_rad,
            speed_mps=carrier.speed_mps,
            support_radius_m=carrier.support_radius_m,
            onboard_platform_ids=by_state["onboard"],
            deployed_platform_ids=by_state["deployed"],
            returning_platform_ids=by_state["returning"],
        )

    def _carrier_platform_state(self) -> CarrierPlatformState:
        return self._carrier_platform_state_for(self._carrier_entity.carrier_id)

    def platform_snapshot(self) -> PlatformSnapshot:
        if not self._platform_core_enabled:
            raise RuntimeError("platform_snapshot requires an explicit platform-core scenario")
        return PlatformSnapshot(
            scenario_id=self._scenario_id,
            sim_time_s=self._clock.sim_time_s,
            carrier=self._carrier_platform_state(),
            carriers=(
                tuple(
                    self._carrier_platform_state_for(carrier_id)
                    for carrier_id in sorted(self._carrier_entities)
                )
                if self._uuv_only_runtime
                else ()
            ),
            roster=PlatformRoster(
                usvs=self._usv_platform_states(),
                uuvs=self._uuv_platform_states(),
            ),
            communication_links=self._connectivity.links,
        )

    def _public_uuv_states(self) -> list[dict[str, object]]:
        return [self._uuv_state(uuv_id).model_dump() for uuv_id in sorted(self._uuvs)]

    def _uuv_state(self, uuv_id: str) -> UUVState:
        """One public UUV state, with the fleet status (default available)."""
        uuv = self._uuvs[uuv_id]
        deployment_state = self._deployment_states[uuv_id]
        limits = self._uuv_motion_limits.get(uuv_id)
        max_speed_mps = (
            limits.max_speed_mps
            if limits is not None
            else self._config.tracking.uuv_max_speed_mps
        )
        energy_cost_per_m = uuv.transit_energy_per_m + (
            uuv.hotel_energy_per_s / max(max_speed_mps, 1e-9)
        )
        return UUVState(
            uuv_id=uuv_id,
            position_xy=(float(uuv.position_xy[0]), float(uuv.position_xy[1])),
            heading_rad=float(uuv.heading_rad),
            speed_mps=self._uuv_speeds[uuv_id],
            energy_fraction=float(uuv.energy_fraction),
            remaining_range_m=(
                float(uuv.energy_fraction / max(energy_cost_per_m, 1e-12))
                if uuv.energy_fraction > 0.0
                else 0.0
            ),
            status=(
                UUVStatus.FAILED
                if deployment_state is DeploymentState.FAILED
                else UUVStatus.RETURNING
                if deployment_state is DeploymentState.RETURNING
                else UUVStatus.AVAILABLE
            ),
            deployment_state=deployment_state,
            group_id=(
                self._uuv_groups.get(uuv_id)
                if deployment_state is DeploymentState.DEPLOYED
                else None
            ),
            sensor_mode=self._sensor_modes.get(uuv_id, "passive"),
            capability=uuv.capability,
            reserved=uuv_id in self._reserved_uuvs,
        )

    def build_slave_contexts(
        self, situation: SituationSnapshot
    ) -> tuple[SlaveSonarContext, ...]:
        """Build truth-safe local contexts for the explicit group brains."""
        snapshot = situation.platform_snapshot
        if snapshot is None:
            return ()
        states = (*snapshot.roster.usvs, *snapshot.roster.uuvs)
        by_id = {state.platform_id: state for state in states}
        links: list[SlaveCommunicationLink] = []
        for link in snapshot.communication_links:
            source = by_id.get(link.source_id)
            target = by_id.get(link.target_id)
            if source is None and link.source_id == snapshot.carrier.carrier_id:
                target = target
                range_m = (
                    target.capability.communications.surface_range_m
                    if target is not None
                    else 1.0
                )
            elif target is None and link.target_id == snapshot.carrier.carrier_id:
                range_m = (
                    source.capability.communications.surface_range_m
                    if source is not None
                    else 1.0
                )
            elif source is not None and target is not None:
                range_m = min(
                    getattr(source.capability.communications, f"{link.medium}_range_m"),
                    getattr(target.capability.communications, f"{link.medium}_range_m"),
                )
            else:
                continue
            links.append(
                SlaveCommunicationLink(
                    source_id=link.source_id,
                    target_id=link.target_id,
                    medium=link.medium,
                    distance_m=link.distance_m,
                    range_m=range_m,
                )
            )

        doctrine = self._config.doctrine
        lead_s = doctrine.handoff_lead_time_s if doctrine is not None else 600
        contexts: list[SlaveSonarContext] = []
        for target_id, report in sorted(self._latest_reports.items()):
            observations = tuple(
                observation
                for observation in situation.platform_observations
                if observation.target_id == target_id
            )
            latest_s = max((item.sim_time_s for item in observations), default=0)
            snr_db = (
                sum(item.snr_db for item in observations) / len(observations)
                if observations
                else -30.0
            )
            covariance = np.asarray(report.belief.covariance, dtype=float)
            covariance_trace = float(np.trace(covariance))
            covariance_max = float(np.max(np.linalg.eigvalsh(covariance)))
            quality = float(report.quality.ewma)
            candidate_ids = tuple(
                sorted(
                    {
                        target_id,
                        *(
                            contact.contact_id
                            for contact in situation.contacts
                            if contact.contact_id != target_id
                        ),
                    }
                )
            )
            previous_covariance = self._slave_covariance_trace_by_target.get(target_id)
            covariance_growth_factor = (
                covariance_trace / previous_covariance
                if previous_covariance is not None and previous_covariance > 1e-9
                else 1.0
            )
            self._slave_covariance_trace_by_target[target_id] = covariance_trace
            handoff_segments, current_segment = self._handoff_segments(
                target_id,
                report,
                situation.sim_time_s,
                lead_s,
                covariance_trace,
                quality,
            )
            valid_intents = {
                "transit", "patrol", "loiter", "evade", "approach", "withdraw"
            }
            predicted_intent = max(
                report.belief.model_probabilities.items(),
                key=lambda item: item[1],
                default=("unknown", 0.0),
            )[0]
            if predicted_intent not in valid_intents:
                predicted_intent = "unknown"
            predicted_intent = cast(
                Literal[
                    "transit",
                    "patrol",
                    "loiter",
                    "evade",
                    "approach",
                    "withdraw",
                    "unknown",
                ],
                predicted_intent,
            )
            group_members = tuple(
                sorted(
                    member
                    for member, assigned_target in self._uuv_groups.items()
                    if assigned_target == target_id
                )
            )
            leader_id = max(group_members, default=None)
            master_connected = leader_id is not None and has_path(
                self._connectivity, snapshot.carrier.carrier_id, leader_id
            )
            platform_capabilities = tuple(
                SlavePlatformCapability(
                    platform_id=state.platform_id,
                    platform_kind=state.capability.kind.value,
                    passive_capable=True,
                    active_capable=state.capability.sonar.active_capable,
                    active_receive_capable=state.capability.sonar.active_receive_range_m > 0,
                    passive_range_m=state.capability.sonar.passive_range_m,
                    active_range_m=state.capability.sonar.active_source_range_m,
                    energy_fraction=state.energy_fraction,
                    ping_energy_cost_fraction=state.capability.sonar.ping_energy_cost_fraction,
                    exposure_cost=state.capability.sonar.exposure_cost,
                    ping_cooldown_s=state.capability.sonar.ping_cooldown_s,
                    cooldown_remaining_s=self._cooldown_remaining_s(
                        state.platform_id, situation.sim_time_s
                    ),
                    available=(
                        state.deployment_state == "deployed" and state.energy_fraction > 0.0
                    ),
                    sensor_mode=state.sensor_mode,
                    max_speed_mps=state.capability.motion.max_speed_mps,
                    max_turn_rate_rad_s=state.capability.motion.max_turn_rate_rad_s,
                    endurance_s=max(1.0, state.energy_fraction * 28_800.0),
                    deployment_state=state.deployment_state,
                    group_id=state.group_id,
                    is_group_leader=bool(
                        state.platform_id == leader_id and state.capability.kind.value == "uuv"
                    ),
                    master_connected=has_path(
                        self._connectivity,
                        snapshot.carrier.carrier_id,
                        state.platform_id,
                    ),
                    carrier_connected=has_path(
                        self._connectivity,
                        snapshot.carrier.carrier_id,
                        state.platform_id,
                    ),
                    passive_bearing_variance_rad2=state.capability.sonar.passive_bearing_variance_rad2,
                    active_bearing_sigma_rad=state.capability.sonar.active_bearing_sigma_rad,
                    active_range_sigma_m=state.capability.sonar.active_range_sigma_m,
                    clutter_sensitivity=state.capability.sonar.clutter_sensitivity,
                    distance_to_carrier_m=(
                        state.distance_to_carrier_m
                        if isinstance(state, USVPlatformState)
                        else None
                    ),
                    carrier_support_radius_m=(
                        snapshot.carrier.support_radius_m
                        if isinstance(state, USVPlatformState)
                        else None
                    ),
                )
                for state in sorted(states, key=lambda item: item.platform_id)
            )
            contexts.append(
                SlaveSonarContext(
                    scenario_id=situation.scenario_id,
                    sim_time_s=situation.sim_time_s,
                    group_id=report.group_id,
                    target_id=target_id,
                    master_id=snapshot.carrier.carrier_id,
                    master_connected=master_connected,
                    platforms=platform_capabilities,
                    communication_links=tuple(links),
                    belief=SlaveBeliefSummary(
                        target_id=target_id,
                        quality=max(0.0, min(1.0, quality)),
                        covariance_trace_m2=max(0.0, covariance_trace),
                        covariance_max_eigenvalue_m2=max(0.0, covariance_max),
                        covariance_growth_factor=max(1.0, covariance_growth_factor),
                        last_observation_age_s=max(0.0, situation.sim_time_s - latest_s),
                        passive_snr_db=snr_db,
                        background_noise_db=max(
                            0.0,
                            (doctrine.active_background_noise_db if doctrine else 6.0)
                            - snr_db,
                        ),
                        active_clutter_level=0.25,
                        target_lost=(
                            not observations
                            or situation.sim_time_s - latest_s
                            > (doctrine.target_lost_after_s if doctrine else 300)
                        ),
                        candidate_count=len(candidate_ids),
                        candidate_ids=candidate_ids,
                        association_confidence=max(0.0, min(1.0, quality)),
                    ),
                    handoff_segments=handoff_segments,
                    current_segment_id=current_segment,
                    predicted_intent=predicted_intent,
                    intent_confidence=max(
                        0.0,
                        min(1.0, max(report.belief.model_probabilities.values(), default=0.0)),
                    ),
                    passive_continuous=doctrine.passive_continuous if doctrine else True,
                    active_only_on_exception=doctrine.active_only_on_exception if doctrine else True,
                    active_quality_floor=doctrine.active_quality_floor if doctrine else 0.40,
                    active_covariance_growth_factor=(
                        doctrine.active_covariance_growth_factor if doctrine else 1.25
                    ),
                    active_background_noise_db=(
                        doctrine.active_background_noise_db if doctrine else 6.0
                    ),
                    max_active_exposure_cost=(
                        doctrine.max_active_exposure_cost if doctrine else 0.60
                    ),
                    require_connected_emitter_receiver=(
                        doctrine.require_connected_emitter_receiver if doctrine else True
                    ),
                    usv_support_radius_is_hard_limit=(
                        doctrine.usv_support_radius_is_hard_limit if doctrine else True
                    ),
                    local_autonomy_when_disconnected=(
                        doctrine.local_autonomy_when_disconnected if doctrine else True
                    ),
                )
            )
        return tuple(contexts)

    def apply_tracking_plan(self, plan: TrackingPlan) -> None:
        """Make the latest committed relay segments visible to local brains.

        The plan is an operational estimate, not simulator truth.  Keeping
        only its segment metadata here lets the next observation cycle expose
        the approved temporal/spatial handoff to the slave without coupling
        the local graph to the carrier repository.
        """
        if self._uuv_only_runtime:
            raise ValueError(
                "legacy TrackingPlan execution is disabled in UUV-only mode"
            )
        self._segment_plans_by_target.clear()
        segment_plan = plan.segment_plan
        if segment_plan is None:
            return
        targets = tuple(sorted(plan.member_ids_by_target))
        if not targets:
            targets = tuple(sorted(self._targets))
        for target_id in targets:
            matching = tuple(
                segment
                for segment in sorted(segment_plan.segments, key=lambda item: item.index)
                if segment.group_id == f"G-{target_id}"
                or segment.group_id.startswith(f"G-{target_id}:")
                or len(targets) == 1
            )
            if matching:
                self._segment_plans_by_target[target_id] = matching

    def _handoff_segments(
        self,
        target_id: str,
        report: GroupReport,
        sim_time_s: int,
        lead_s: int,
        covariance_trace: float,
        quality: float,
    ) -> tuple[tuple[SlaveHandoffSegment, ...], str]:
        """Project approved plan segments, or a bounded pre-plan estimate."""
        planned = self._segment_plans_by_target.get(target_id, ())
        if planned:
            current = next(
                (
                    segment
                    for segment in planned
                    if segment.start_s <= sim_time_s < segment.end_s
                ),
                next(
                    (segment for segment in planned if segment.start_s > sim_time_s),
                    planned[-1],
                ),
            )
            current_position = planned.index(current)
            selected = planned[current_position : current_position + 2] or (current,)
            projected: list[SlaveHandoffSegment] = []
            for offset, segment in enumerate(selected):
                projected.append(
                    SlaveHandoffSegment(
                        segment_id=(
                            f"plan:{segment.index}:{segment.start_s}-{segment.end_s}"
                        ),
                        start_s=segment.start_s,
                        end_s=segment.end_s,
                        predicted_quality=max(
                            0.0, min(1.0, quality * (0.95**offset))
                        ),
                        predicted_covariance_trace_m2=max(
                            0.0, covariance_trace * (1.10**offset)
                        ),
                        owner_group_id=segment.group_id,
                        intercept_xy=segment.intercept_xy,
                    )
                )
            return tuple(projected), f"plan:{current.index}:{current.start_s}-{current.end_s}"

        current_id = f"{report.group_id}:segment:{sim_time_s}"
        future_id = f"{report.group_id}:handoff:{sim_time_s + lead_s}"
        quality_value = max(0.0, min(1.0, quality))
        covariance_value = max(0.0, covariance_trace)
        return (
            (
                SlaveHandoffSegment(
                    segment_id=current_id,
                    start_s=sim_time_s,
                    end_s=sim_time_s + lead_s,
                    predicted_quality=quality_value,
                    predicted_covariance_trace_m2=covariance_value,
                    owner_group_id=report.group_id,
                ),
                SlaveHandoffSegment(
                    segment_id=future_id,
                    start_s=sim_time_s + lead_s,
                    end_s=sim_time_s + 2 * lead_s,
                    predicted_quality=max(0.0, min(1.0, quality_value * 0.9)),
                    predicted_covariance_trace_m2=max(0.0, covariance_value * 1.15),
                    owner_group_id=report.group_id,
                ),
            ),
            current_id,
        )

    def _cooldown_remaining_s(self, platform_id: str, sim_time_s: int) -> int:
        contact_times = tuple(
            timestamp
            for (emitter_id, _), timestamp in self._last_ping_times.items()
            if emitter_id == platform_id
        )
        if not contact_times:
            return 0
        last = max(contact_times)
        if platform_id in self._uuvs:
            cooldown = self._config.tracking.sensor_ping_interval_s
        else:
            cooldown = self._usv_capabilities[platform_id].sonar.ping_cooldown_s
        return max(0, cooldown - (sim_time_s - last))

    def build_adversary_inputs(
        self, situation: SituationSnapshot
    ) -> tuple[AdversaryEscapeInput, ...]:
        """Build target-owned evidence packets without private truth fields."""
        snapshot = situation.platform_snapshot
        environment = self._config.environment
        if snapshot is None or environment is None:
            return ()
        boundary = AdversaryOperatingBoundary(
            min_x=environment.map_bounds_xy[0],
            max_x=environment.map_bounds_xy[1],
            min_y=environment.map_bounds_xy[2],
            max_y=environment.map_bounds_xy[3],
        )
        inputs: list[AdversaryEscapeInput] = []
        for target_id, target in sorted(self._targets.items()):
            belief = target.adversary_belief(situation.sim_time_s)
            observations: list[AdversaryObservation] = []
            trigger_events: list[AdversaryTrigger] = []
            for item in situation.platform_observations:
                if item.target_id != target_id:
                    continue
                observations.append(
                    AdversaryObservation(
                        observation_id=item.observation_id,
                        observed_at_s=item.sim_time_s,
                        kind="passive_sonar",
                        bearing_rad=wrap(item.azimuth_rad + pi),
                        range_m=None,
                        confidence=item.detection_confidence,
                        assessment="platform",
                    )
                )
            for event in situation.pending_events:
                if event.entity_id != target_id and not event.event_type.startswith(
                    "observability_"
                ):
                    continue
                # Causal-chain rows are emitted for blue-side audit and
                # replan routing. They are not new target-side observations.
                if event.payload.get("chain_id") is not None:
                    continue
                if event.event_type == "active_ping":
                    observations.append(
                        AdversaryObservation(
                            observation_id=event.event_id,
                            observed_at_s=event.sim_time_s,
                            kind="active_sonar",
                            bearing_rad=event.payload.get("azimuth_rad"),
                            range_m=event.payload.get("range_m"),
                            confidence=0.75,
                            assessment="emission",
                        )
                    )
                trigger_events.append(
                    AdversaryTrigger(
                        trigger_id=event.event_id,
                        event_type=event.event_type,
                        sim_time_s=event.sim_time_s,
                        severity=event.level.value,
                        summary=_adversary_event_summary(event.event_type),
                    )
                )
            threats: list[PlatformThreatSummary] = []
            for state in (*snapshot.roster.usvs, *snapshot.roster.uuvs):
                distance_m = hypot(
                    state.position_xy[0] - belief.estimated_position_xy[0],
                    state.position_xy[1] - belief.estimated_position_xy[1],
                )
                passive_risk = min(1.0, state.capability.sonar.passive_range_m / max(distance_m, 1.0))
                active_risk = min(1.0, state.capability.sonar.active_source_range_m / max(distance_m, 1.0))
                relay = has_path(self._connectivity, snapshot.carrier.carrier_id, state.platform_id)
                relay_risk = 0.8 if relay else 0.15
                level: Literal["low", "medium", "high", "critical"] = (
                    "critical" if relay and active_risk > 0.7 else
                    "high" if active_risk > 0.5 or passive_risk > 0.7 else
                    "medium" if passive_risk > 0.25 else "low"
                )
                threats.append(
                    PlatformThreatSummary(
                        platform_id=state.platform_id,
                        platform_kind=state.capability.kind.value,
                        observed_at_s=situation.sim_time_s,
                        threat_level=level,
                        estimated_range_m=distance_m,
                        relative_bearing_rad=wrap(
                            atan2(
                                state.position_xy[1] - belief.estimated_position_xy[1],
                                state.position_xy[0] - belief.estimated_position_xy[0],
                            ) - belief.estimated_heading
                        ),
                        passive_detection_risk=passive_risk,
                        active_ping_risk=active_risk,
                        relay_detection_risk=relay_risk,
                        surface_relay_available=relay,
                    )
                )
            active_emitters = tuple(
                state.platform_id
                for state in (*snapshot.roster.usvs, *snapshot.roster.uuvs)
                if state.sensor_mode == "active"
            )
            latest_active = max(
                (
                    event.sim_time_s
                    for event in situation.pending_events
                    if event.event_type == "active_ping" and event.entity_id == target_id
                ),
                default=None,
            )
            context = AdversaryEscapeInput(
                    target_id=target_id,
                    sim_time_s=situation.sim_time_s,
                    belief=belief,
                    observations=tuple(observations),
                    platform_threats=tuple(threats),
                    trigger_events=tuple(
                        sorted(
                            trigger_events,
                            key=lambda item: (item.sim_time_s, item.trigger_id),
                        )[-16:]
                    ),
                    communications_acoustic_exposure=CommunicationsAcousticExposure(
                        as_of_s=situation.sim_time_s,
                        passive_signature_level=0.2,
                        active_emitter_exposure=1.0 if active_emitters else 0.0,
                        communication_intercept_risk=0.6 if active_emitters else 0.2,
                        relay_detection_risk=max(
                            (threat.relay_detection_risk for threat in threats), default=0.0
                        ),
                        acoustic_clutter_level=0.25,
                        last_burst_age_s=(
                            float(situation.sim_time_s - latest_active)
                            if latest_active is not None
                            else None
                        ),
                        own_emission_mode="active" if active_emitters else "passive",
                    ),
                    decision_history=self._adversary_decision_history.get(target_id, ()),
                    kinematic_limits=AdversaryKinematicLimits(
                        max_speed_mps=target.max_speed_mps,
                        max_turn_rate_rad_s=target.max_turn_rate_rad_s,
                        decision_horizon_s=float(
                            self._config.timing.observation_step_s
                        ),
                        max_decoy_count=2,
                        decoy_inventory=target.decoy_inventory,
                    ),
                    operating_boundary=boundary,
            )
            if self._adversary_gate.should_request(context):
                self._adversary_contexts[target_id] = context
                inputs.append(context)
        return tuple(inputs)

    def apply_slave_sonar_decision(self, decision: SlaveSonarDecision) -> None:
        """Apply one validated slave decision through the public sensor API."""
        known_ids = set(self._uuvs) | set(self._usvs)
        if any(platform_id not in known_ids for platform_id in decision.receiver_ids):
            raise ValueError("slave decision references an unknown platform")
        if decision.mode == "passive":
            for platform_id in decision.receiver_ids:
                self.set_sensor_mode(platform_id, "passive")
            return
        emitter_id = decision.emitter
        if emitter_id is None or emitter_id not in known_ids:
            raise ValueError("active slave decision requires a known emitter")
        self.set_sensor_mode(
            emitter_id,
            "active",
            ping_contact_id=decision.target_id,
            receiver_ids=decision.receiver_ids,
        )

    def apply_adversary_decision(self, decision: AdversaryEscapeDecision) -> None:
        """Apply a validated target decision and deploy requested decoys."""
        target = self._targets.get(decision.target_id)
        if target is None:
            raise ValueError(f"unknown adversary target {decision.target_id!r}")
        hold_steps = max(
            1,
            self._config.timing.observation_step_s
            // max(1, self._config.timing.physics_step_s),
        )
        target.apply_adversary_decision(decision, hold_steps=hold_steps)
        context = self._adversary_contexts.pop(decision.target_id, None)
        if context is not None:
            self._adversary_gate.record_decision(context)
        history = self._adversary_decision_history.setdefault(decision.target_id, ())
        self._adversary_decision_history[decision.target_id] = (
            *history,
            AdversaryDecisionRecord(
                decision_id=f"{decision.target_id}:adversary:{self._clock.sim_time_s}:{len(history)}",
                sim_time_s=self._clock.sim_time_s,
                maneuver=decision.maneuver,
                intent=decision.intent,
                segment=decision.segment,
                speed=decision.speed,
                heading=decision.heading,
                decoy_action=decision.decoy_action,
                decoy_count=decision.decoy_count,
                confidence=decision.confidence,
                rationale=decision.rationale,
                communications_discipline=decision.communications_discipline,
                trigger_event_ids=decision.trigger_event_ids,
                outcome="unknown",
            ),
        )[-8:]
        chain_id = f"{decision.target_id}:maneuver:{self._clock.sim_time_s}:{len(history)}"
        prediction_revision = len(history) + 1
        decision_id = self._adversary_decision_history[decision.target_id][-1].decision_id
        applied_plan_revision = self._applied_plan_revisions.get(
            (self._scenario_id, decision.target_id), 0
        )
        self._maneuver_response_chains[decision.target_id] = {
            "chain_id": chain_id,
            "maneuver_time_s": self._clock.sim_time_s,
            "prediction_revision": prediction_revision,
            "decision_id": decision_id,
            "applied_plan_revision": applied_plan_revision,
        }
        self._pending_runtime_events.extend(
            (
                RuntimeEvent(
                    event_id=f"{chain_id}:target_maneuver",
                    scenario_id=self._scenario_id,
                    sim_time_s=self._clock.sim_time_s,
                    event_type="intent_change_confirmed",
                    entity_id=decision.target_id,
                    level=EventLevel.STRATEGIC,
                    payload={
                        "phase": "target_maneuver",
                        "chain_id": chain_id,
                        "maneuver": decision.maneuver,
                        "decision_id": decision_id,
                        "prediction_revision": prediction_revision,
                        "plan_revision": applied_plan_revision,
                        "latency_s": 0,
                    },
                ),
                RuntimeEvent(
                    event_id=f"{chain_id}:prediction_revision",
                    scenario_id=self._scenario_id,
                    sim_time_s=self._clock.sim_time_s,
                    event_type="state_changed",
                    entity_id=decision.target_id,
                    level=EventLevel.INFORMATIONAL,
                    payload={
                        "phase": "prediction_revision",
                        "chain_id": chain_id,
                        "decision_id": decision_id,
                        "prediction_revision": prediction_revision,
                        "plan_revision": applied_plan_revision,
                        "latency_s": 0,
                    },
                ),
            )
        )
        for index in range(target.consume_decoy_request()):
            decoy_id = f"{decision.target_id}:decoy:{self._clock.sim_time_s}:{index}"
            angle = 2.0 * pi * (index + 1) / (decision.decoy_count + 1)
            distance = 250.0 + 100.0 * index
            x = target.position_xy[0] + distance * cos(angle)
            y = target.position_xy[1] + distance * sin(angle)
            min_x, max_x, min_y, max_y = target.bounds_xy
            self._decoys[decoy_id] = DecoyEntity(
                decoy_id,
                (min(max(x, min_x), max_x), min(max(y, min_y), max_y)),
                angle,
                self._config.tracking.decoy_drift_speed_mps,
                self._config.tracking.decoy_heading_noise_rad_per_s,
            )
            self._contact_state[decoy_id] = {
                "classification": ContactClassification.UNVERIFIED,
                "evidence": ("target_decoy_deployed",),
                "position_xy": None,
            }

    def fail_uuv(self, uuv_id: str) -> None:
        """Mark one fleet UUV failed: it stops observing and steering.

        The UUV stays in the fleet and its situation status becomes
        ``UUVStatus.FAILED``, so the carrier can plan its replacement.
        """
        if uuv_id not in self._uuvs:
            raise ValueError(f"unknown uuv {uuv_id!r}")
        self._uuv_statuses[uuv_id] = UUVStatus.FAILED
        self._deployment_states[uuv_id] = DeploymentState.FAILED
        self._recovery_waypoints.pop(uuv_id, None)
        self._uuv_groups.pop(uuv_id, None)

    def request_uuv_recovery(self, uuv_id: str, reason: str = "requested") -> None:
        """Begin the deterministic deployed-to-onboard recovery lifecycle."""
        state = self._deployment_state_for(uuv_id)
        if state is not DeploymentState.DEPLOYED:
            raise ValueError(f"cannot recover uuv {uuv_id!r} from {state.value}")
        self._recovery_waypoints[uuv_id] = list(self._uuvs[uuv_id].waypoints)
        self._deployment_states[uuv_id] = DeploymentState.RETURNING
        if self._uuv_only_runtime:
            self._mission_recovery_requested_uuv_ids.add(uuv_id)
        self._uuv_groups.pop(uuv_id, None)
        self._reserved_uuvs = frozenset(member for member in self._reserved_uuvs if member != uuv_id)
        self._queue_lifecycle_event("uuv_recovery_requested", uuv_id, reason)

    def request_uuv_deployment(self, uuv_id: str, reason: str = "requested") -> None:
        """Launch an onboard UUV and restore its last commanded waypoint."""
        state = self._deployment_state_for(uuv_id)
        if state is not DeploymentState.ONBOARD:
            raise ValueError(f"cannot deploy uuv {uuv_id!r} from {state.value}")
        self._deployment_states[uuv_id] = DeploymentState.DEPLOYED
        self._uuvs[uuv_id].set_waypoints(self._recovery_waypoints.pop(uuv_id, []))
        self._queue_lifecycle_event("uuv_deployed", uuv_id, reason)

    def _deployment_state_for(self, uuv_id: str) -> DeploymentState:
        if uuv_id not in self._uuvs:
            raise ValueError(f"unknown uuv {uuv_id!r}")
        return self._deployment_states[uuv_id]

    def _complete_uuv_recovery(self, uuv_id: str, sim_time_s: int) -> None:
        uuv = self._uuvs[uuv_id]
        self._deployment_states[uuv_id] = DeploymentState.ONBOARD
        carrier = self._carrier_entities.get(
            self._uuv_carrier_ids.get(uuv_id, self._carrier_entity.carrier_id),
            self._carrier_entity,
        )
        uuv.position_xy = carrier.position_xy
        uuv.heading_rad = carrier.heading_rad
        uuv.speed_mps = 0.0
        if self._uuv_only_runtime:
            uuv.energy_fraction = 1.0
        uuv.set_waypoints([])
        self._uuv_speeds[uuv_id] = 0.0
        self._mission_distance_m[uuv_id] = 0.0
        self._uuv_groups.pop(uuv_id, None)
        self._mission_recovered_uuv_ids.add(uuv_id)
        self._events.append(
            RuntimeEvent(
                event_id=f"uuv_recovered:{uuv_id}:{sim_time_s}",
                scenario_id=self._scenario_id,
                sim_time_s=sim_time_s,
                event_type="uuv_recovered",
                entity_id=uuv_id,
                level=EventLevel.INFORMATIONAL,
                payload={},
            )
        )

    def _queue_lifecycle_event(self, event_type: str, uuv_id: str, reason: str) -> None:
        self._pending_runtime_events.append(
            RuntimeEvent(
                event_id=f"{event_type}:{uuv_id}:{self._clock.sim_time_s}",
                scenario_id=self._scenario_id,
                sim_time_s=self._clock.sim_time_s,
                event_type=event_type,
                entity_id=uuv_id,
                level=EventLevel.INFORMATIONAL,
                payload={"reason": reason},
            )
        )

    def set_sensor_mode(
        self,
        uuv_id: str,
        mode: Literal["passive", "active"],
        ping_contact_id: str | None = None,
        receiver_ids: Sequence[str] = (),
    ) -> None:
        """Set one UUV or USV sensor mode through the public runtime API."""
        if uuv_id not in self._uuvs and uuv_id not in self._usvs:
            raise ValueError(f"unknown platform {uuv_id!r}")
        self._sensor_modes[uuv_id] = mode
        self._ping_targets[uuv_id] = ping_contact_id
        self._ping_receivers[uuv_id] = tuple(receiver_ids) or (uuv_id,)
        if mode == "passive":
            self._ping_targets[uuv_id] = None

    def set_operational_scheme(self, scheme: OperationalScheme) -> None:
        """Queue a validated operational scheme for subsequent snapshots."""
        self._operational_scheme = scheme
        self._pending_runtime_events.append(
            RuntimeEvent(
                event_id=(
                    "operational_scheme_updated:"
                    f"{scheme.scheme_id}:{scheme.version}:{self._clock.sim_time_s}"
                ),
                scenario_id=self._scenario_id,
                sim_time_s=self._clock.sim_time_s,
                event_type="operational_scheme_updated",
                entity_id=scheme.scheme_id,
                level=EventLevel.STRATEGIC,
                payload={"version": scheme.version},
            )
        )

    def submit_intelligence(self, report: IntelligenceReport) -> None:
        """Queue one operational intelligence report for future snapshots."""
        if report.valid_until_s <= self._clock.sim_time_s:
            raise ValueError("intelligence report is already expired")
        self._intelligence_reports[report.report_id] = report
        self._pending_runtime_events.append(
            RuntimeEvent(
                event_id=(
                    "intelligence_report_received:"
                    f"{report.report_id}:{self._clock.sim_time_s}"
                ),
                scenario_id=self._scenario_id,
                sim_time_s=self._clock.sim_time_s,
                event_type="intelligence_report_received",
                entity_id=report.target_id,
                level=EventLevel.STRATEGIC,
                payload={"report_id": report.report_id, "source": report.source.value},
            )
        )

    def drop_contact(self, contact_id: str) -> None:
        """Drop a decoy-classified contact from the operational picture (R5)."""
        if contact_id not in self._decoys:
            return
        self._decoys.pop(contact_id, None)
        self._decoy_observations.pop(contact_id, None)
        self._contact_state.pop(contact_id, None)
        for uuv_id in sorted(self._uuvs):
            if self._ping_targets.get(uuv_id) == contact_id:
                self._ping_targets[uuv_id] = None

    def set_reservations(
        self,
        reservations: Mapping[str, Sequence[str]] | ReservationRegistry,
    ) -> None:
        """Replace the human reservation set (R4): target -> reserved uuv ids.

        The runtime's ``ReservationRegistry`` is accepted as a duck-typed
        ``items()`` view (``items`` yields ``(target_id, sorted uuv ids)``);
        plain mappings with the same shape work too.
        """
        self._reserved_by_target = {
            target_id: tuple(sorted(uuv_ids))
            for target_id, uuv_ids in reservations.items()
        }
        self._reserved_uuvs = frozenset(
            uuv for uuv_ids in self._reserved_by_target.values() for uuv in uuv_ids
        )

    def promote_contact(self, contact_id: str) -> None:
        """Promote a submarine-classified contact into a tracked target (R5).

        Creates the target and, when at least two free UUVs exist (not in
        a group, not reserved, not failed), its group through the existing
        allocation. The group takes the coarse prior from the active-sonar
        measurement; with fewer than two free UUVs the promotion leaves a
        target without a group and the committed plan's command creates it
        at the next observation cycle.
        """
        if contact_id in self._targets:
            return
        state = self._contact_state.get(contact_id)
        if state is None or state.get("position_xy") is None:
            return
        tracking = self._config.tracking
        measured = state["position_xy"]
        self._targets[contact_id] = TargetEntity(
            contact_id,
            measured,
            (tracking.submarine_cruise_speed_mps, 0.0),
            HiddenIntent.TRANSIT,
            intent_speed_mps=self._intent_speed_mps(),
            max_speed_mps=tracking.submarine_sprint_speed_mps,
            max_turn_rate_rad_s=tracking.submarine_turn_rate_rad_s,
        )
        state["classification"] = ContactClassification.SUBMARINE
        self._decoys.pop(contact_id, None)
        self._decoy_observations.pop(contact_id, None)
        free = tuple(
            sorted(
                uuv_id
                for uuv_id in self._uuvs
                if uuv_id not in self._uuv_groups
                and uuv_id not in self._reserved_uuvs
                and self._deployment_states[uuv_id] is DeploymentState.DEPLOYED
                and self._uuv_statuses.get(uuv_id, UUVStatus.AVAILABLE)
                is not UUVStatus.FAILED
            )
        )
        if len(free) < 2:
            return
        solution = allocate_groups(
            AllocationInput(
                uuv_ids=free,
                target_ids=(contact_id,),
                quality_by_target={contact_id: _INITIAL_QUALITY},
                uuv_available={uuv_id: True for uuv_id in free},
                uuv_energy_fraction={
                    uuv_id: self._uuvs[uuv_id].energy_fraction for uuv_id in free
                },
                quality_warning=tracking.quality_warning,
                quality_release=tracking.quality_release,
                release_hold_s=tracking.release_hold_s,
            )
        )
        members = solution.members_by_target.get(contact_id, ())
        if len(members) < 2:
            return
        report = self._manager.create(
            contact_id,
            scenario_id=self._scenario_id,
            group_id=f"G-{contact_id}",
            member_ids=tuple(members),
            coarse_prior=measured,
            member_positions={member: self._uuvs[member].position_xy for member in members},
        )
        self._latest_reports[contact_id] = report
        self._last_guard_reasons[contact_id] = report.quality.hard_guard_reasons
        self._event_counters[contact_id] = 0
        self._assignments[contact_id] = tuple(members)
        for member in members:
            self._uuv_groups[member] = contact_id

    def apply_plan_command(self, command: PlanCommand) -> None:
        """Queue one committed plan's complete roster for the group graph."""
        if self._uuv_only_runtime:
            raise ValueError(
                "legacy PlanCommand execution is disabled in UUV-only mode"
            )
        revision_key = (command.scenario_id, command.target_id)
        applied_revision = self._applied_plan_revisions.get(revision_key, 0)
        if command.plan_revision <= applied_revision:
            raise ValueError(
                "stale plan revision for "
                f"{command.scenario_id!r}/{command.target_id!r}: "
                f"{command.plan_revision} <= {applied_revision}"
            )
        self._apply_deployment_actions(command)
        if not command.member_ids:
            self._applied_plan_revisions[revision_key] = command.plan_revision
            self._record_blue_response(command)
            return
        if self._platform_core_enabled:
            self._rebuild_connectivity()
        report = self._latest_reports.get(command.target_id)
        if report is None:
            report = self._create_missing_group(command)
            if report is None:
                return
        self._pending_group_commands[command.target_id] = GroupPlanCommand(
            command_id=command.command_id,
            scenario_id=command.scenario_id,
            target_id=command.target_id,
            sim_time_s=command.sim_time_s,
            plan_revision=command.plan_revision,
            desired_member_ids=tuple(command.member_ids),
            member_positions={
                member: self._uuvs[member].position_xy for member in command.member_ids
            },
        )
        for member in command.member_ids:
            self.set_sensor_mode(member, command.sensor_mode)
        self._applied_plan_revisions[revision_key] = command.plan_revision
        self._record_blue_response(command)

    def _synchronize_group_membership(self, target_id: str, members: tuple[str, ...]) -> None:
        """Make engine membership agree with the group graph's committed roster."""
        for uuv_id, assigned_target in tuple(self._uuv_groups.items()):
            if assigned_target == target_id:
                self._uuv_groups.pop(uuv_id)
        for uuv_id in members:
            if self._deployment_states[uuv_id] is DeploymentState.DEPLOYED:
                self._uuv_groups[uuv_id] = target_id

    def _apply_deployment_actions(self, command: PlanCommand) -> None:
        """Apply carrier lifecycle actions without changing plan version flow."""
        for uuv_id, action in sorted(command.actions.items()):
            if action in {"rotate", "return"}:
                self.request_uuv_recovery(
                    uuv_id, reason=f"plan:{command.command_id}:{action}"
                )
                continue
            if (
                action == "track"
                and self._deployment_state_for(uuv_id) is DeploymentState.ONBOARD
            ):
                self.request_uuv_deployment(
                    uuv_id, reason=f"plan:{command.command_id}:track"
                )
                self._uuv_groups[uuv_id] = command.target_id
            if action == "track" and uuv_id in command.waypoints_by_member:
                self._uuvs[uuv_id].set_waypoints(
                    [
                        (float(waypoint.x), float(waypoint.y))
                        for waypoint in command.waypoints_by_member[uuv_id]
                    ]
                )
        for usv_id in command.usv_ids:
            if usv_id not in self._usvs:
                raise ValueError(f"unknown usv {usv_id!r}")
            action = command.usv_actions.get(usv_id, "relay")
            if action not in {"track", "relay", "hold", "return"}:
                raise ValueError(f"unsupported usv action {action!r}")
            if action == "hold":
                self._usv_hold_ids.add(usv_id)
                waypoints = []
            else:
                self._usv_hold_ids.discard(usv_id)
            if action == "return":
                waypoints = [self._carrier_entity.position_xy]
            elif action != "hold":
                waypoints = [
                    (float(waypoint.x), float(waypoint.y))
                    for waypoint in command.waypoints_by_member.get(usv_id, ())
                ]
            self._usv_waypoints[usv_id] = waypoints
            usv = self._usvs[usv_id]
            if action == "hold":
                usv.motion = MotionState(
                    position_xy=usv.motion.position_xy,
                    heading_rad=usv.motion.heading_rad,
                    speed_mps=0.0,
                )
                usv.set_motion_command(
                    MotionCommand(
                        desired_heading_rad=usv.motion.heading_rad,
                        desired_speed_mps=0.0,
                    )
                )
            elif waypoints:
                destination = waypoints[0]
                usv.set_motion_command(
                    MotionCommand(
                        desired_heading_rad=atan2(
                            destination[1] - usv.motion.position_xy[1],
                            destination[0] - usv.motion.position_xy[0],
                        ),
                        desired_speed_mps=usv.limits.max_speed_mps,
                    )
                )
            self._usv_execution_records[usv_id] = {
                "command_id": command.command_id,
                "target_id": command.target_id,
                "region_id": command.region_id,
                "action": action,
                "waypoint_count": len(waypoints),
                "sim_time_s": self._clock.sim_time_s,
            }

    def _record_blue_response(self, command: PlanCommand) -> None:
        """Close one target-maneuver audit chain for a regional tracking response."""
        chain = self._maneuver_response_chains.get(command.target_id)
        if chain is None:
            return
        if command.plan_revision <= int(chain["applied_plan_revision"]):
            return
        response_actions = (
            *command.actions.values(),
            *(command.usv_actions.get(usv_id, "relay") for usv_id in command.usv_ids),
        )
        if not command.region_id or not any(
            action in {"track", "relay"} for action in response_actions
        ):
            return
        self._maneuver_response_chains.pop(command.target_id)
        chain_id = str(chain["chain_id"])
        maneuver_time_s = int(chain["maneuver_time_s"])
        decision_id = str(chain["decision_id"])
        prediction_revision = int(chain["prediction_revision"])
        latency_s = max(0, self._clock.sim_time_s - maneuver_time_s)
        response_members = (*command.member_ids, *command.usv_ids)
        self._pending_runtime_events.extend(
            (
                RuntimeEvent(
                    event_id=f"{chain_id}:regional_task_revision",
                    scenario_id=self._scenario_id,
                    sim_time_s=self._clock.sim_time_s,
                    event_type="state_changed",
                    entity_id=command.target_id,
                    level=EventLevel.INFORMATIONAL,
                    payload={
                        "phase": "regional_task_revision",
                        "chain_id": chain_id,
                        "decision_id": decision_id,
                        "prediction_revision": prediction_revision,
                        "plan_revision": command.plan_revision,
                        "region_id": command.region_id,
                        "latency_s": latency_s,
                    },
                ),
                RuntimeEvent(
                    event_id=f"{chain_id}:effect_change",
                    scenario_id=self._scenario_id,
                    sim_time_s=self._clock.sim_time_s,
                    event_type="state_changed",
                    entity_id=command.target_id,
                    level=EventLevel.INFORMATIONAL,
                    payload={
                        "phase": "effect_change",
                        "chain_id": chain_id,
                        "decision_id": decision_id,
                        "prediction_revision": prediction_revision,
                        "plan_revision": command.plan_revision,
                        "member_ids": response_members,
                        "usv_actions": dict(command.usv_actions),
                        "latency_s": latency_s,
                    },
                ),
                RuntimeEvent(
                    event_id=f"{chain_id}:blue_response",
                    scenario_id=self._scenario_id,
                    sim_time_s=self._clock.sim_time_s,
                    event_type="state_changed",
                    entity_id=command.target_id,
                    level=EventLevel.INFORMATIONAL,
                    payload={
                        "phase": "blue_response",
                        "chain_id": chain_id,
                        "decision_id": decision_id,
                        "prediction_revision": prediction_revision,
                        "plan_revision": command.plan_revision,
                        "latency_s": latency_s,
                        "response_command_id": command.command_id,
                    },
                ),
            )
        )

    def _update_fast_regional_replan_events(self, sim_time_s: int) -> None:
        """Emit one strategic replan event per sustained quality or relay failure."""
        warning = self._config.tracking.quality_warning
        for target_id, report in sorted(self._latest_reports.items()):
            degraded = report.quality.ewma < warning
            if not degraded:
                self._regional_quality_streaks.pop(target_id, None)
                self._regional_quality_latches.discard(target_id)
                continue
            streak = self._regional_quality_streaks.get(target_id, 0) + 1
            self._regional_quality_streaks[target_id] = streak
            if streak < 2 or target_id in self._regional_quality_latches:
                continue
            self._regional_quality_latches.add(target_id)
            self._pending_runtime_events.append(
                RuntimeEvent(
                    event_id=f"regional_feedback:{target_id}:{sim_time_s}",
                    scenario_id=self._scenario_id,
                    sim_time_s=sim_time_s,
                    event_type="regional_feedback_received",
                    entity_id=target_id,
                    level=EventLevel.INFORMATIONAL,
                    payload={
                        "target_id": target_id,
                        "quality": report.quality.ewma,
                        "threshold": warning,
                        "streak": streak,
                    },
                )
            )
        for usv_id, record in sorted(self._usv_execution_records.items()):
            if record.get("action") != "relay":
                self._relay_failure_streaks.pop(usv_id, None)
                self._relay_failure_latches.discard(usv_id)
                continue
            failed = not has_path(
                self._connectivity, self._carrier_entity.carrier_id, usv_id
            )
            if not failed:
                self._relay_failure_streaks.pop(usv_id, None)
                self._relay_failure_latches.discard(usv_id)
                continue
            streak = self._relay_failure_streaks.get(usv_id, 0) + 1
            self._relay_failure_streaks[usv_id] = streak
            if streak < 2 or usv_id in self._relay_failure_latches:
                continue
            self._relay_failure_latches.add(usv_id)
            self._pending_runtime_events.append(
                RuntimeEvent(
                    event_id=f"communication_link_lost:{usv_id}:{sim_time_s}",
                    scenario_id=self._scenario_id,
                    sim_time_s=sim_time_s,
                    event_type="communication_link_lost",
                    entity_id=usv_id,
                    level=EventLevel.INFORMATIONAL,
                    payload={
                        "target_id": record.get("target_id"),
                        "region_id": record.get("region_id"),
                        "relay_id": usv_id,
                        "streak": streak,
                    },
                )
            )

    def apply_verification_command(self, command: VerificationCommand) -> None:
        """Apply one active-sonar verification protocol command (R5)."""
        if command.sensor_mode == "ping":
            for uuv_id in command.uuv_ids:
                self.set_sensor_mode(uuv_id, "active", ping_contact_id=command.target_id)
        elif command.sensor_mode == "return_to_passive":
            for uuv_id in command.uuv_ids:
                self.set_sensor_mode(uuv_id, "passive")
        elif command.sensor_mode == "dispatch":
            self.promote_contact(command.target_id)
        elif command.sensor_mode == "drop":
            self.drop_contact(command.target_id)

    def _create_missing_group(self, command: PlanCommand) -> GroupReport | None:
        """Create the group of a promoted contact not yet allocated.

        The R5 protocol promotes a submarine-classified contact through
        ``promote_contact``; when fewer than two free UUVs existed at
        promotion time, the committed plan command is the first chance to
        materialize the group. The group takes the coarse prior from the
        active-sonar measurement.
        """
        if command.target_id not in self._targets:
            return None
        contact = self._contact_state.get(command.target_id, {})
        prior = contact.get("position_xy") or self._targets[command.target_id].position_xy
        report = self._manager.create(
            command.target_id,
            scenario_id=self._scenario_id,
            group_id=command.group_id,
            member_ids=tuple(command.member_ids),
            coarse_prior=prior,
            member_positions={
                member: self._uuvs[member].position_xy for member in command.member_ids
            },
        )
        self._latest_reports[command.target_id] = report
        self._last_guard_reasons[command.target_id] = report.quality.hard_guard_reasons
        self._event_counters[command.target_id] = 0
        self._assignments[command.target_id] = tuple(command.member_ids)
        for member in command.member_ids:
            self._uuv_groups[member] = command.target_id
        return report

    def _record_belief_history(self, sim_time_s: int) -> None:
        """Record each group's belief mean after an observation cycle.

        Only recorded when a carrier hook is present (the headless path is
        unchanged); the history is exposed to the carrier through
        ``belief_history`` for initialization/telemetry purposes. A group
        report can retain its last measurement time while platforms are
        temporarily out of range, so the engine clock is the authoritative
        timestamp for this sampled history. Replacing an equal timestamp
        keeps the provider strictly increasing for motion-feature extraction.
        """
        if self._carrier is None:
            return
        for target_id, report in sorted(self._latest_reports.items()):
            mean = report.belief.mean
            history = self._belief_histories.setdefault(target_id, [])
            sample = (sim_time_s, float(mean[0]), float(mean[1]))
            if history and sim_time_s < history[-1][0]:
                continue
            if history and sim_time_s == history[-1][0]:
                history[-1] = sample
            else:
                history.append(sample)

    def _emit_belief_change_events(self, sim_time_s: int) -> None:
        """Emit planner triggers from public IMM belief changes.

        This source is deliberately limited to estimator-visible group reports.
        Hidden target intent is never consulted, so the event stream is a safe
        input for an LLM or any other external mission planner.
        """
        for target_id, report in sorted(self._latest_reports.items()):
            probabilities = {
                str(label): float(probability)
                for label, probability in report.belief.model_probabilities.items()
            }
            label, confidence = max(
                sorted(probabilities.items()),
                key=lambda item: (item[1], item[0]),
                default=("unknown", 0.0),
            )
            previous = self._belief_intent_state.get(target_id)
            if previous is not None:
                previous_label, previous_confidence = previous
                payload = {
                    "intent": label,
                    "confidence": confidence,
                    "probabilities": dict(sorted(probabilities.items())),
                    "source": "public_imm_belief",
                }
                if label != previous_label:
                    self._events.append(
                        RuntimeEvent(
                            event_id=f"belief_intent:{target_id}:{sim_time_s}",
                            scenario_id=self._scenario_id,
                            sim_time_s=sim_time_s,
                            event_type="target_intent_changed",
                            entity_id=target_id,
                            level=EventLevel.STRATEGIC,
                            payload=payload,
                        )
                    )
                if abs(confidence - previous_confidence) >= 0.15:
                    self._events.append(
                        RuntimeEvent(
                            event_id=f"belief_confidence:{target_id}:{sim_time_s}",
                            scenario_id=self._scenario_id,
                            sim_time_s=sim_time_s,
                            event_type="imm_confidence_shifted",
                            entity_id=target_id,
                            level=EventLevel.STRATEGIC,
                            payload={
                                **payload,
                                "previous_confidence": previous_confidence,
                            },
                        )
                    )
            self._belief_intent_state[target_id] = (label, confidence)

    def belief_history(self, target_id: str) -> tuple[tuple[int, float, float], ...]:
        """The recorded belief means for one target (sim time, x, y)."""
        return tuple(self._belief_histories.get(target_id, ()))

    def _build_situation(self, sim_time_s: int) -> SituationSnapshot:
        """The latest operational situation for the carrier hook.

        ``snapshot_revision`` is the observation cycle index (sim time
        divided by the 30 s cadence), matching the carrier's plan snapshots.
        """
        observation_step_s = self._config.timing.observation_step_s
        uuvs = tuple(self._situation_uuv_state(uuv_id) for uuv_id in sorted(self._uuvs))
        self._append_observability_feedback(sim_time_s)
        pending_events = (
            *self._carrier_events,
            *self._events,
            *self._pending_runtime_events,
        )
        self._carrier_events.clear()
        return SituationSnapshot(
            scenario_id=self._scenario_id,
            snapshot_revision=sim_time_s // observation_step_s,
            sim_time_s=sim_time_s,
            uuvs=uuvs,
            carrier=self._carrier_entity.state_for(uuvs),
            carriers=(
                tuple(self.carrier_states().values()) if self._uuv_only_runtime else ()
            ),
            group_reports=tuple(self._sorted_reports()),
            pending_events=pending_events,
            contacts=tuple(self._contacts()),
            operational_scheme=self._active_operational_scheme(sim_time_s),
            intelligence_reports=self._valid_intelligence_reports(sim_time_s),
            platform_snapshot=self.platform_snapshot() if self._platform_core_enabled else None,
            platform_observations=self._platform_observations,
            adversary_summaries=self._adversary_summaries(sim_time_s, pending_events),
            map_bounds_xy=(
                self._config.environment.map_bounds_xy
                if self._config.environment is not None
                else None
            ),
            uuv_resource_episodes=self._mission_resource_episodes(),
        )

    def refresh_situation(self, situation: SituationSnapshot) -> SituationSnapshot:
        """Project immediate post-command state without advancing time.

        The carrier hook runs after an observation snapshot is built. A
        committed plan can therefore launch a UUV, change its group, or apply
        a sonar decision after that snapshot was created. Rebuilding the
        observation would consume feedback queues twice, so this method only
        refreshes mutable platform-facing projections and preserves the
        estimator snapshot that drove the decision.
        """
        uuvs = tuple(
            self._situation_uuv_state(uuv_id) for uuv_id in sorted(self._uuvs)
        )
        pending_by_id = {
            event.event_id: event for event in situation.pending_events
        }
        for event in (
            *self._carrier_events,
            *self._events,
            *self._pending_runtime_events,
        ):
            pending_by_id[event.event_id] = event
        pending_events = tuple(
            sorted(
                pending_by_id.values(),
                key=lambda event: (event.sim_time_s, event.event_id),
            )
        )
        return situation.model_copy(
            update={
                "uuvs": uuvs,
                "carrier": self._carrier_entity.state_for(uuvs),
                "carriers": (
                    tuple(self.carrier_states().values()) if self._uuv_only_runtime else ()
                ),
                "pending_events": pending_events,
                "platform_snapshot": (
                    self.platform_snapshot() if self._platform_core_enabled else None
                ),
                "platform_observations": self._platform_observations,
                "adversary_summaries": self._adversary_summaries(
                    situation.sim_time_s, pending_events
                ),
                "uuv_resource_episodes": self._mission_resource_episodes(),
            }
        )

    def publication_situation(self) -> SituationSnapshot:
        """Return the current public state without consuming carrier inputs.

        The carrier-owned observation snapshot is intentionally built only on
        the observation cadence. The live publisher also needs physical
        updates between those cycles, particularly while an LLM is
        reconnecting. This projection reuses the latest estimator output but
        reflects the current platform and simulation-clock state.
        """
        sim_time_s = self._clock.sim_time_s
        pending_events = tuple(
            sorted(
                (
                    *self._carrier_events,
                    *self._events,
                    *self._pending_runtime_events,
                ),
                key=lambda event: (event.sim_time_s, event.event_id),
            )
        )
        uuvs = tuple(
            self._situation_uuv_state(uuv_id) for uuv_id in sorted(self._uuvs)
        )
        return SituationSnapshot(
            scenario_id=self._scenario_id,
            snapshot_revision=self._step_index,
            sim_time_s=sim_time_s,
            uuvs=uuvs,
            carrier=self._carrier_entity.state_for(uuvs),
            carriers=(
                tuple(self.carrier_states().values()) if self._uuv_only_runtime else ()
            ),
            group_reports=tuple(self._sorted_reports()),
            pending_events=pending_events,
            contacts=tuple(self._contacts()),
            operational_scheme=self._active_operational_scheme(sim_time_s),
            intelligence_reports=self._valid_intelligence_reports(sim_time_s),
            platform_snapshot=self.platform_snapshot() if self._platform_core_enabled else None,
            platform_observations=self._platform_observations,
            adversary_summaries=self._adversary_summaries(sim_time_s, pending_events),
            map_bounds_xy=(
                self._config.environment.map_bounds_xy
                if self._config.environment is not None
                else None
            ),
            uuv_resource_episodes=self._mission_resource_episodes(),
        )

    def _append_observability_feedback(self, sim_time_s: int) -> None:
        """Run the external MVP-derived supervisor on estimator-visible data."""
        input_frame = self._observability_input_frame(sim_time_s)
        reports = self._observability.process_frame(input_frame)
        for report in reports:
            report_payload = report.to_public_dict()
            level = (
                EventLevel.TACTICAL
                if report.report_type.value == "URGENT"
                else EventLevel.INFORMATIONAL
            )
            entity_id = (
                report.tracks[0].track_id
                if len(report.tracks) == 1
                else self._scenario_id
            )
            self._events.append(
                RuntimeEvent(
                    event_id=f"observability:{report.report_id}",
                    scenario_id=self._scenario_id,
                    sim_time_s=sim_time_s,
                    event_type=f"observability_{report.report_type.value.lower()}",
                    entity_id=entity_id,
                    level=level,
                    payload=report_payload,
                )
            )

    def _observability_input_frame(self, sim_time_s: int) -> InputFrame:
        """Adapt current operational estimates to the MVP feedback contract."""
        uuv_states = tuple(self._uuv_state(uuv_id) for uuv_id in sorted(self._uuvs))
        known_uuv_ids = {state.uuv_id for state in uuv_states}
        input_uuvs = [
            {
                "uuv_id": state.uuv_id,
                "position_xy_m": state.position_xy,
                "heading_rad": state.heading_rad,
                "state_age_sec": 0.0,
                "communication_age_sec": (
                    0.0
                    if has_path(self._connectivity, self._carrier_entity.carrier_id, state.uuv_id)
                    else 31.0
                ),
                "valid": (
                    state.deployment_state is DeploymentState.DEPLOYED
                    and state.status is not UUVStatus.FAILED
                ),
            }
            for state in uuv_states
        ]
        tracks: list[dict[str, object]] = []
        observations: list[dict[str, object]] = []
        innovations: list[dict[str, object]] = []
        sequence_id = sim_time_s // max(1, self._config.timing.observation_step_s)
        for report in sorted(self._latest_reports.values(), key=lambda item: item.target_id):
            mean = report.belief.mean
            velocity = (
                (float(mean[2]), float(mean[3]))
                if len(mean) >= 4
                else (0.0, 0.0)
            )
            covariance = report.belief.covariance
            covariance_2x2 = (
                float(covariance[0][0]),
                float(covariance[0][1]),
                float(covariance[1][0]),
                float(covariance[1][1]),
            )
            tracks.append(
                {
                    "track_id": report.target_id,
                    "estimated_position_xy_m": (float(mean[0]), float(mean[1])),
                    "estimated_velocity_xy_mps": velocity,
                    "position_covariance_2x2": covariance_2x2,
                    "association_confidence": report.quality.ewma,
                    "association_entropy": None,
                    "lifecycle_state": "confirmed",
                }
            )
            for index, observation in enumerate(
                item
                for item in self._platform_observations
                if item.target_id == report.target_id
                and item.observer_id in known_uuv_ids
            ):
                observer = next(
                    state for state in uuv_states if state.uuv_id == observation.observer_id
                )
                predicted = atan2(
                    float(mean[1]) - observer.position_xy[1],
                    float(mean[0]) - observer.position_xy[0],
                )
                residual = wrap_angle(observation.azimuth_rad - predicted)
                observations.append(
                    {
                        "uuv_id": observation.observer_id,
                        "candidate_track_id": report.target_id,
                        "sequence_id": sequence_id * 1000 + index,
                        "bearing_rad": observation.azimuth_rad,
                        "bearing_variance_rad2": observation.variance_rad2,
                        "measurement_age_sec": 0.0,
                        "valid": True,
                    }
                )
                innovations.append(
                    {
                        "track_id": report.target_id,
                        "uuv_id": observation.observer_id,
                        "innovation_rad": residual,
                        "innovation_variance_rad2": observation.variance_rad2,
                    }
                )
        return InputFrame.from_mapping(
            {
                "timestamp_sec": float(sim_time_s),
                "frame_sequence_id": sequence_id,
                "frame_id": "map",
                "tracks": tracks,
                "uuvs": input_uuvs,
                "bearing_observations": observations,
                "innovations": innovations,
            }
        )

    def _adversary_summaries(
        self,
        sim_time_s: int,
        pending_events: Sequence[RuntimeEvent],
    ) -> tuple[AdversaryOperationalSummary, ...]:
        """Project target-owned detection and LLM decision state for operators."""
        summaries: list[AdversaryOperationalSummary] = []
        for target_id, target in sorted(self._targets.items()):
            history = self._adversary_decision_history.get(target_id, ())
            latest = history[-1] if history else None
            belief = target.adversary_belief(sim_time_s)
            trigger_ids = tuple(
                event.event_id
                for event in pending_events
                if event.entity_id == target_id
                and (
                    event.event_type.startswith("target_detection_")
                    or event.event_type.startswith("observability_")
                    or event.event_type == "active_ping"
                )
            )[-16:]
            if latest is not None:
                trigger_ids = latest.trigger_event_ids or trigger_ids
            inferred_intent = belief.intent_hypothesis
            inferred_maneuver = {
                "break_contact": "speed_change",
                "reposition": "course_change",
                "deception": "decoy_evasion",
                "silent_transit": "silent_running",
                "evade": "decoy_evasion",
                "hold_course": "hold_course",
            }.get(inferred_intent)
            has_llm_decision = latest is not None
            summaries.append(
                AdversaryOperationalSummary(
                    target_id=target_id,
                    sim_time_s=sim_time_s,
                    detection_range_m=target.detection_range_m,
                    detected_platform_ids=self._target_detected_platform_ids.get(
                        target_id, ()
                    ),
                    trigger_event_ids=tuple(sorted(set(trigger_ids))),
                    decision_id=(
                        latest.decision_id
                        if has_llm_decision
                        else f"{target_id}:belief:{sim_time_s}"
                        if inferred_maneuver is not None
                        else None
                    ),
                    maneuver=(latest.maneuver if has_llm_decision else inferred_maneuver),
                    intent=(latest.intent if has_llm_decision else inferred_intent if inferred_maneuver else None),
                    segment=(latest.segment if has_llm_decision else "target-belief" if inferred_maneuver else None),
                    speed=(latest.speed if has_llm_decision else belief.estimated_speed_mps if inferred_maneuver else None),
                    heading=(latest.heading if has_llm_decision else belief.estimated_heading if inferred_maneuver else None),
                    decoy_count=latest.decoy_count if has_llm_decision else 0,
                    confidence=(latest.confidence if has_llm_decision else belief.intent_confidence if inferred_maneuver else None),
                    rationale=(
                        latest.rationale
                        if has_llm_decision
                        else (
                            "目标侧公开状态估计显示当前意图；等待对手脑复核。"
                            if inferred_maneuver
                            else None
                        )
                    ),
                    communications_discipline=(
                        latest.communications_discipline if has_llm_decision else "silent"
                        if inferred_maneuver
                        else None
                    ),
                    decision_status=latest.outcome if has_llm_decision else "inconclusive" if inferred_maneuver else "unknown",
                )
            )
        return tuple(summaries)

    def _active_operational_scheme(self, sim_time_s: int) -> OperationalScheme | None:
        scheme = self._operational_scheme
        if scheme is None or not scheme.valid_from_s <= sim_time_s < scheme.valid_until_s:
            return None
        return scheme

    def _valid_intelligence_reports(self, sim_time_s: int) -> tuple[IntelligenceReport, ...]:
        return tuple(
            report
            for _, report in sorted(self._intelligence_reports.items())
            if report.issued_at_s <= sim_time_s < report.valid_until_s
        )

    def _situation_uuv_state(self, uuv_id: str) -> UUVState:
        """One UUV state for the carrier, with the planning speed floored.

        The observed speed of a parked observer is zero — it simply has no
        current command — but it can still be commanded onto a new waypoint
        at the fleet's maximum speed. The carrier's waypoint planner uses
        ``speed_mps`` as its positive kinematic bound, so a zero observed
        speed would make any allocation touching a parked UUV unplannable.
        The floor is applied only to the situation handed to the carrier;
        the logged frames keep the observed telemetry.
        """
        state = self._uuv_state(uuv_id)
        if state.speed_mps <= 0.0:
            return state.model_copy(
                update={"speed_mps": float(self._config.tracking.uuv_max_speed_mps)}
            )
        return state

    def _sorted_reports(self) -> list[GroupReport]:
        return [self._latest_reports[target_id] for target_id in sorted(self._latest_reports)]

    def _truth(self, sim_time_s: int) -> dict[str, object]:
        targets: list[dict[str, object]] = []
        for target_id in sorted(self._targets):
            target = self._targets[target_id]
            targets.append(
                {
                    "target_id": target_id,
                    "position_xy": [float(target.position_xy[0]), float(target.position_xy[1])],
                    "velocity_xy": [float(target.velocity_xy[0]), float(target.velocity_xy[1])],
                    "intent_label": target.intent.value,
                }
            )
        decoys = [
            {
                "decoy_id": decoy_id,
                "position_xy": [
                    float(decoy.position_xy[0]),
                    float(decoy.position_xy[1]),
                ],
                "heading_rad": float(decoy.heading_rad),
            }
            for decoy_id, decoy in sorted(self._decoys.items())
        ]
        return {"sim_time_s": sim_time_s, "targets": targets, "decoys": decoys}
