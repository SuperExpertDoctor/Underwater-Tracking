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

import hashlib
from math import atan2, cos, hypot, pi, sin
from pathlib import Path
import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from underwater_tracking.config.models import AppConfig
from underwater_tracking.domain.agent_models import PlanCommand, VerificationCommand
from underwater_tracking.domain.models import (
    BearingObservation,
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
from underwater_tracking.groups.manager import GroupManager
from underwater_tracking.groups.state import PlanCommand as GroupPlanCommand
from underwater_tracking.persistence.frame_log import FrameLogger
from underwater_tracking.planning.allocation import AllocationInput, allocate_groups
from underwater_tracking.planning.reservations import ReservationRegistry
from underwater_tracking.planning.waypoints import plan_group_waypoints
from underwater_tracking.simulation.clock import SimulationClock
from underwater_tracking.simulation.carrier import CarrierEntity
from underwater_tracking.simulation.decoy import DecoyEntity
from underwater_tracking.simulation.target import HiddenIntent, TargetEntity
from underwater_tracking.simulation.uuv import UUVEntity, wrap
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


class SimulationEngine:
    """Deterministic headless simulation of the multi-UUV bearing-only scenario."""

    def __init__(
        self,
        config: AppConfig,
        seed: int = 42,
        output_dir: str | Path | None = None,
        evaluation_sink: Callable[[dict[str, object]], None] | None = None,
        carrier: Callable[[SituationSnapshot], None] | None = None,
    ) -> None:
        self._config = config
        self._seed = seed
        self._scenario_id = _SCENARIO_ID
        self._run_id = f"run-{seed}-{uuid.uuid4().hex[:8]}"
        self._sink = evaluation_sink if evaluation_sink is not None else _noop_sink
        self._carrier = carrier
        self._step_index = 0
        self._events: list[RuntimeEvent] = []
        self._master_rng = random.Random(seed)
        self._entity_rngs: dict[str, random.Random] = {}
        self._observer_rngs: dict[str, random.Random] = {}
        self._clock = SimulationClock(step_s=config.timing.physics_step_s)
        self._carrier_entity = CarrierEntity()
        self._uuvs: dict[str, UUVEntity] = {}
        self._targets: dict[str, TargetEntity] = {}
        self._uuv_groups: dict[str, str] = {}
        self._uuv_speeds: dict[str, float] = {}
        self._uuv_statuses: dict[str, UUVStatus] = {}
        self._deployment_states: dict[str, DeploymentState] = {}
        self._recovery_waypoints: dict[str, list[tuple[float, float]]] = {}
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
        self._last_ping_times: dict[tuple[str, str], int] = {}
        self._reserved_by_target: dict[str, tuple[str, ...]] = {}
        self._reserved_uuvs: frozenset[str] = frozenset()
        self._target_rays: dict[str, tuple[BearingObservation, ...]] = {}
        self._spawn_world()
        self._assignments: dict[str, tuple[str, ...]] = {}
        self._manager = GroupManager()
        self._latest_reports: dict[str, GroupReport] = {}
        self._last_guard_reasons: dict[str, tuple[str, ...]] = {}
        self._event_counters: dict[str, int] = {}
        self._allocate_and_create_groups()
        self._previous_waypoints: dict[str, np.ndarray] = {}
        self._waypoint_commands: dict[str, dict[str, tuple[float, float]]] = {}
        self._plan_waypoints()
        directory = Path(output_dir) if output_dir is not None else Path("outputs") / self._run_id
        self.logger = FrameLogger(directory)

    def step(self) -> dict[str, object]:
        """Advance the clock once and return the operational frame."""
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
        self._sink(self._truth(sim_time_s))
        return frame

    def _spawn_world(self) -> None:
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
            )
        for target_id in self._targets:
            self._contact_state[target_id] = {
                "classification": ContactClassification.SUBMARINE,
                "evidence": (),
                "position_xy": None,
            }

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

    def _advance_world(self, sim_time_s: int) -> None:
        dt_s = float(self._clock.step_s)
        tracking = self._config.tracking
        self._carrier_entity.step(dt_s)
        for uuv_id in sorted(self._uuvs):
            if self._uuv_statuses.get(uuv_id, UUVStatus.AVAILABLE) is UUVStatus.FAILED:
                continue
            uuv = self._uuvs[uuv_id]
            deployment_state = self._deployment_states[uuv_id]
            if deployment_state is DeploymentState.ONBOARD:
                uuv.position_xy = self._carrier_entity.position_xy
                uuv.set_waypoints([])
                self._uuv_speeds[uuv_id] = 0.0
                continue
            if deployment_state is DeploymentState.RETURNING:
                uuv.set_waypoints([self._carrier_entity.position_xy])
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
            uuv.step(
                dt_s,
                tracking.uuv_max_speed_mps,
                tracking.uuv_max_turn_rate_rad_s,
            )
            after = uuv.position_xy
            self._uuv_speeds[uuv_id] = (
                hypot(after[0] - before[0], after[1] - before[1]) / dt_s
            )
            if (
                deployment_state is DeploymentState.RETURNING
                and hypot(
                    uuv.position_xy[0] - self._carrier_entity.position_xy[0],
                    uuv.position_xy[1] - self._carrier_entity.position_xy[1],
                ) <= _RECOVERY_RADIUS_M
            ):
                self._complete_uuv_recovery(uuv_id, sim_time_s)
        for target_id in sorted(self._targets):
            self._targets[target_id].step(dt_s, self._target_rng(target_id))
        for decoy_id in sorted(self._decoys):
            self._decoys[decoy_id].step(dt_s, self._decoy_rng(decoy_id))
        # Decoy bearing observations are collected every physics step (and
        # refreshed in the observation cycle), so decoy contacts with their
        # rays are visible from the very first frame.
        self._decoy_observations = self._observe_decoys(sim_time_s)
        self._process_pings(sim_time_s)

    def _observation_cycle(self, sim_time_s: int) -> None:
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
            if contact_id in self._decoys:
                request_xy = self._decoys[contact_id].position_xy
            elif contact_id in self._targets:
                request_xy = self._targets[contact_id].position_xy
            else:
                request_report = self._latest_reports.get(contact_id)
                if request_report is None:
                    continue
                request_xy = (
                    float(request_report.belief.mean[0]),
                    float(request_report.belief.mean[1]),
                )
            self._events.append(
                RuntimeEvent(
                    event_id=f"active_ping:{contact_id}:{sim_time_s}",
                    scenario_id=self._scenario_id,
                    sim_time_s=sim_time_s,
                    event_type="active_ping",
                    entity_id=contact_id,
                    level=EventLevel.INFORMATIONAL,
                    payload={"position_xy": request_xy, "uuv_ids": ()},
                )
            )
        self._record_belief_history()
        self._plan_waypoints()
        if self._carrier is not None:
            self._carrier(self._build_situation(sim_time_s))

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
        return self._make_observation(target_id, uuv_id, sim_time_s, target_xy)

    def _make_observation(
        self,
        target_id: str,
        uuv_id: str,
        sim_time_s: int,
        target_xy: tuple[float, float],
    ) -> BearingObservation:
        uuv = self._uuvs[uuv_id]
        truth = atan2(target_xy[1] - uuv.position_xy[1], target_xy[0] - uuv.position_xy[0])
        rng = self._observer_rngs.setdefault(
            f"{target_id}:{uuv_id}", random.Random(self._seed ^ _stable_int(f"{target_id}:{uuv_id}"))
        )
        variance_rad2 = uuv.capability.bearing_variance_rad2
        measured = wrap(truth + rng.gauss(0.0, variance_rad2 ** 0.5))
        return BearingObservation(
            observation_id=f"{target_id}:{uuv_id}:{sim_time_s}",
            scenario_id=self._scenario_id,
            sim_time_s=sim_time_s,
            uuv_id=uuv_id,
            target_id=target_id,
            azimuth_rad=measured,
            variance_rad2=variance_rad2,
            detection_confidence=1.0,
        )

    def _process_pings(self, sim_time_s: int) -> None:
        """Execute every active-sonar ping due this physics step (R5).

        A UUV in ``active`` mode with a ping contact pings every
        ``sensor_ping_interval_s`` seconds; the ping-heard and
        classification draws use a seeded per-(uuv, contact) rng, so the
        whole protocol is deterministic per seed. Contacts keep their
        first classification (sticky); a heard ping drains the pinger's
        energy and, for a contacted submarine, triggers an evasive sprint.
        """
        tracking = self._config.tracking
        for uuv_id, contact_id in sorted(self._ping_targets.items()):
            if self._sensor_modes.get(uuv_id) != "active" or contact_id is None:
                continue
            if self._deployment_states.get(uuv_id) is not DeploymentState.DEPLOYED:
                continue
            last = self._last_ping_times.get((uuv_id, contact_id))
            if last is not None and sim_time_s - last < tracking.sensor_ping_interval_s:
                continue
            uuv = self._uuvs.get(uuv_id)
            contact = self._contact_state.get(contact_id)
            if uuv is None or contact is None:
                continue
            if not uuv.capability.active_sonar_available:
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
                # Engine-known position of an unverified decoy: without a
                # prior measurement the ping needs somewhere to aim, and the
                # request trigger (A2) already exposes this position to the
                # verification node — bounded simplification (ruling 9).
                contact_xy = self._decoys[contact_id].position_xy
            if contact_xy is None:
                continue
            self._last_ping_times[(uuv_id, contact_id)] = sim_time_s
            range_m = hypot(
                contact_xy[0] - uuv.position_xy[0],
                contact_xy[1] - uuv.position_xy[1],
            )
            if range_m > uuv.capability.active_range_m or range_m < _SENSOR_MIN_RANGE_M:
                continue
            azimuth_rad = atan2(
                contact_xy[1] - uuv.position_xy[1],
                contact_xy[0] - uuv.position_xy[0],
            )
            rng = self._entity_rngs.setdefault(
                f"active:{uuv_id}:{contact_id}",
                random.Random(
                    self._seed ^ _stable_int(f"active:{uuv_id}:{contact_id}")
                ),
            )
            is_decoy = contact_id in self._decoys
            # A5: the executed-ping report is emitted for EVERY executed
            # ping (heard or not), before the heard check.
            self._events.append(
                RuntimeEvent(
                    event_id=f"active_ping:{uuv_id}:{contact_id}:{sim_time_s}",
                    scenario_id=self._scenario_id,
                    sim_time_s=sim_time_s,
                    event_type="active_ping",
                    entity_id=contact_id,
                    level=EventLevel.INFORMATIONAL,
                    payload={
                        "uuv_id": uuv_id,
                        "contact_id": contact_id,
                        "range_m": round(range_m, 1),
                        "azimuth_rad": round(azimuth_rad, 6),
                        "uuv_ids": (uuv_id,),
                        "position_xy": contact_xy,
                    },
                )
            )
            if rng.random() > tracking.sensor_ping_heard_probability:
                continue
            self._uuvs[uuv_id].energy_fraction = max(
                0.0, uuv.energy_fraction - tracking.sensor_ping_energy_cost
            )
            state = self._contact_state[contact_id]
            classification = state.get(
                "classification", ContactClassification.UNVERIFIED
            )
            if classification is ContactClassification.UNVERIFIED:
                classify_prob = (
                    tracking.sensor_active_classify_decoy_prob
                    if is_decoy
                    else tracking.sensor_active_classify_submarine_prob
                )
                # A3: classify_prob is the probability of a CORRECT
                # classification — a decoy that fails the gate masquerades
                # as a submarine, a submarine that fails looks like a decoy.
                if is_decoy:
                    classification = (
                        ContactClassification.DECOY
                        if rng.random() < classify_prob
                        else ContactClassification.SUBMARINE
                    )
                else:
                    classification = (
                        ContactClassification.SUBMARINE
                        if rng.random() < classify_prob
                        else ContactClassification.DECOY
                    )
                state["classification"] = classification
                evidence = state.get("evidence", ())
                state["evidence"] = (
                    *evidence,
                    f"ping:{uuv_id}:{contact_id}:{sim_time_s}",
                )
            # A4: a heard ping measures the contact — the true position plus
            # range-sigma Gaussian noise, for decoys and targets alike.
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
                target = self._targets[contact_id]
                target.apply_evasive_maneuver(
                    tracking.submarine_turn_rate_rad_s
                    * tracking.sensor_ping_interval_s
                )
            self._events.append(
                RuntimeEvent(
                    event_id=f"contact_classified:{contact_id}:{uuv_id}:{sim_time_s}",
                    scenario_id=self._scenario_id,
                    sim_time_s=sim_time_s,
                    event_type="contact_classified",
                    entity_id=contact_id,
                    level=EventLevel.INFORMATIONAL,
                    payload={
                        "contact_id": contact_id,
                        "classification": classification.value,
                        "uuv_id": uuv_id,
                        "outcome": classification.value,
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
            )
            self._previous_waypoints[target_id] = plan.waypoints_xy
            commands: dict[str, tuple[float, float]] = {}
            for member, point in zip(members, plan.waypoints_xy, strict=True):
                commands[member] = (float(point[0]), float(point[1]))
                self._uuvs[member].set_waypoints([commands[member]])
            self._waypoint_commands[target_id] = commands

    def _track_converged(self, belief: TargetBelief) -> bool:
        """True once the position estimate is tight enough to maneuver safely."""
        position_std = float(np.sqrt((belief.covariance[0][0] + belief.covariance[1][1]) / 2.0))
        return position_std <= _TRACK_CONVERGENCE_STD_M

    def _hold_spread_commands(
        self, members: tuple[str, ...], positions: np.ndarray
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

    def _belief_sigma_points_xy(self, belief: TargetBelief) -> np.ndarray:
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
        return {
            "run_id": self._run_id,
            "scenario_id": self._scenario_id,
            "sim_time_s": sim_time_s,
            "step_index": self._step_index,
            "uuvs": [uuv.model_dump() for uuv in uuvs],
            "carrier": self._carrier_entity.state_for(uuvs).model_dump(),
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

    def _public_uuv_states(self) -> list[dict[str, object]]:
        return [self._uuv_state(uuv_id).model_dump() for uuv_id in sorted(self._uuvs)]

    def _uuv_state(self, uuv_id: str) -> UUVState:
        """One public UUV state, with the fleet status (default available)."""
        uuv = self._uuvs[uuv_id]
        deployment_state = self._deployment_states[uuv_id]
        return UUVState(
            uuv_id=uuv_id,
            position_xy=(float(uuv.position_xy[0]), float(uuv.position_xy[1])),
            heading_rad=float(uuv.heading_rad),
            speed_mps=self._uuv_speeds[uuv_id],
            energy_fraction=float(uuv.energy_fraction),
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
        uuv.position_xy = self._carrier_entity.position_xy
        uuv.set_waypoints([])
        self._uuv_speeds[uuv_id] = 0.0
        self._uuv_groups.pop(uuv_id, None)
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
    ) -> None:
        """Set one UUV's sensor mode; ``active`` targets ``ping_contact_id`` (R5)."""
        if uuv_id not in self._uuvs:
            raise ValueError(f"unknown uuv {uuv_id!r}")
        self._sensor_modes[uuv_id] = mode
        self._ping_targets[uuv_id] = ping_contact_id

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
        self._apply_deployment_actions(command)
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

    def _record_belief_history(self) -> None:
        """Record each group's belief mean after an observation cycle.

        Only recorded when a carrier hook is present (the headless path is
        unchanged); the history is exposed to the carrier through
        ``belief_history`` for initialization/telemetry purposes.
        """
        if self._carrier is None:
            return
        for target_id, report in sorted(self._latest_reports.items()):
            mean = report.belief.mean
            self._belief_histories.setdefault(target_id, []).append(
                (report.sim_time_s, float(mean[0]), float(mean[1]))
            )

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
        pending_events = (*self._carrier_events, *self._events)
        self._carrier_events.clear()
        return SituationSnapshot(
            scenario_id=self._scenario_id,
            snapshot_revision=sim_time_s // observation_step_s,
            sim_time_s=sim_time_s,
            uuvs=uuvs,
            carrier=self._carrier_entity.state_for(uuvs),
            group_reports=tuple(self._sorted_reports()),
            pending_events=pending_events,
            contacts=tuple(self._contacts()),
            operational_scheme=self._active_operational_scheme(sim_time_s),
            intelligence_reports=self._valid_intelligence_reports(sim_time_s),
        )

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
