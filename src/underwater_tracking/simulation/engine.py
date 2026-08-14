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
from collections.abc import Callable

import numpy as np

from underwater_tracking.config.models import AppConfig
from underwater_tracking.domain.models import (
    BearingObservation,
    EventLevel,
    GroupReport,
    RuntimeEvent,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.groups.manager import GroupManager
from underwater_tracking.persistence.frame_log import FrameLogger
from underwater_tracking.planning.allocation import AllocationInput, allocate_groups
from underwater_tracking.planning.waypoints import plan_group_waypoints
from underwater_tracking.simulation.clock import SimulationClock
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
_UUV_MAX_SPEED_MPS = 6.0
_UUV_MAX_TURN_RATE_RAD_S = pi / 60.0
_UUV_DEPLOY_RADIUS_M = 2000.0
_TARGET_SPAWN_SPAN_M = 800.0

# Fleet kinematics: the observers must outrun the fastest target intent
# (WITHDRAW at 3.35 m/s), or the standoff ring drifts away faster than they
# can close, the group bunches, and the track is lost. 6 m/s gives a
# comfortable margin for closing and re-acquisition.
_UUV_MAX_SPEED_MPS = 6.0

# Waypoint planning knobs matching the waypoint planner's own defaults.
# max_step_m is the receding-horizon *maneuver authority* used to pick the
# committed lattice waypoint, while UUVEntity.step caps actual motion at
# _UUV_MAX_SPEED_MPS. The authority is bounded (900 m) so no observer is
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
    ) -> None:
        self._config = config
        self._seed = seed
        self._scenario_id = _SCENARIO_ID
        self._run_id = f"run-{seed}-{uuid.uuid4().hex[:8]}"
        self._sink = evaluation_sink if evaluation_sink is not None else _noop_sink
        self._step_index = 0
        self._events: list[RuntimeEvent] = []
        self._master_rng = random.Random(seed)
        self._entity_rngs: dict[str, random.Random] = {}
        self._observer_rngs: dict[str, random.Random] = {}
        self._clock = SimulationClock(step_s=config.timing.physics_step_s)
        self._uuvs: dict[str, UUVEntity] = {}
        self._targets: dict[str, TargetEntity] = {}
        self._uuv_groups: dict[str, str] = {}
        self._uuv_speeds: dict[str, float] = {}
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
        self._events = []
        sim_time_s = self._clock.tick()
        timing = self._config.timing
        self._advance_world(sim_time_s)
        if sim_time_s % timing.observation_step_s == 0:
            self._observation_cycle(sim_time_s)
        if sim_time_s % timing.group_report_s == 0:
            self._publish_reports(sim_time_s)
        frame = self._build_frame(sim_time_s)
        self.logger.write(frame)
        self._sink(self._truth(sim_time_s))
        return frame

    def _spawn_world(self) -> None:
        scenario = self._config.scenario
        for index in range(scenario.uuv_count):
            uuv_id = f"uuv_{index:02d}"
            angle = 2.0 * pi * index / scenario.uuv_count
            position = (float(_UUV_DEPLOY_RADIUS_M * cos(angle)), float(_UUV_DEPLOY_RADIUS_M * sin(angle)))
            heading = atan2(-position[1], -position[0])
            self._uuvs[uuv_id] = UUVEntity(uuv_id, position, heading, 1.0)
            self._uuv_speeds[uuv_id] = 0.0
        for index in range(scenario.initial_target_count):
            target_id = f"target_{index:02d}"
            x = self._master_rng.uniform(-_TARGET_SPAWN_SPAN_M, _TARGET_SPAWN_SPAN_M)
            y = self._master_rng.uniform(-_TARGET_SPAWN_SPAN_M, _TARGET_SPAWN_SPAN_M)
            self._targets[target_id] = TargetEntity(
                target_id, (float(x), float(y)), (2.0, 0.0), HiddenIntent.TRANSIT
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
        del sim_time_s
        dt_s = float(self._clock.step_s)
        for uuv_id in sorted(self._uuvs):
            uuv = self._uuvs[uuv_id]
            before = uuv.position_xy
            uuv.step(dt_s, _UUV_MAX_SPEED_MPS, _UUV_MAX_TURN_RATE_RAD_S)
            after = uuv.position_xy
            self._uuv_speeds[uuv_id] = (
                hypot(after[0] - before[0], after[1] - before[1]) / dt_s
            )
        for target_id in sorted(self._targets):
            self._targets[target_id].step(dt_s, self._target_rng(target_id))

    def _observation_cycle(self, sim_time_s: int) -> None:
        for target_id in sorted(self._targets):
            report = self._latest_reports[target_id]
            members = report.member_ids
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
            fresh = self._manager.invoke(
                target_id,
                observations=observations,
                member_positions={member: self._uuvs[member].position_xy for member in members},
            )
            self._latest_reports[target_id] = fresh
            self._events.extend(self._guard_events(fresh))
        self._plan_waypoints()

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
        """One noisy bearing observation, or None inside the sensor blind zone."""
        uuv = self._uuvs[uuv_id]
        standoff = hypot(target_xy[0] - uuv.position_xy[0], target_xy[1] - uuv.position_xy[1])
        if standoff < _SENSOR_MIN_RANGE_M:
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
        measured = wrap(truth + rng.gauss(0.0, _BEARING_VARIANCE_RAD2 ** 0.5))
        return BearingObservation(
            observation_id=f"{target_id}:{uuv_id}:{sim_time_s}",
            scenario_id=self._scenario_id,
            sim_time_s=sim_time_s,
            uuv_id=uuv_id,
            target_id=target_id,
            azimuth_rad=measured,
            variance_rad2=_BEARING_VARIANCE_RAD2,
            detection_confidence=1.0,
        )

    def _plan_waypoints(self) -> None:
        for target_id in sorted(self._latest_reports):
            report = self._latest_reports[target_id]
            members = report.member_ids
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
            plan = plan_group_waypoints(
                positions,
                self._belief_sigma_points_xy(report.belief),
                previous_waypoints=self._previous_waypoints.get(target_id),
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

    def _target_rng(self, target_id: str) -> random.Random:
        rng = self._entity_rngs.get(target_id)
        if rng is None:
            rng = random.Random(self._seed ^ _stable_int(target_id))
            self._entity_rngs[target_id] = rng
        return rng

    def _build_frame(self, sim_time_s: int) -> dict[str, object]:
        reports = self._sorted_reports()
        return {
            "run_id": self._run_id,
            "scenario_id": self._scenario_id,
            "sim_time_s": sim_time_s,
            "step_index": self._step_index,
            "uuvs": self._public_uuv_states(),
            "group_reports": [report.model_dump() for report in reports],
            "tracks": [report.belief.model_dump() for report in reports],
            "quality": [
                {"target_id": report.target_id, **report.quality.model_dump()}
                for report in reports
            ],
            "assignments": {
                target_id: list(members)
                for target_id, members in sorted(self._assignments.items())
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
        states: list[dict[str, object]] = []
        for uuv_id in sorted(self._uuvs):
            uuv = self._uuvs[uuv_id]
            states.append(
                UUVState(
                    uuv_id=uuv_id,
                    position_xy=(float(uuv.position_xy[0]), float(uuv.position_xy[1])),
                    heading_rad=float(uuv.heading_rad),
                    speed_mps=self._uuv_speeds[uuv_id],
                    energy_fraction=float(uuv.energy_fraction),
                    status=UUVStatus.AVAILABLE,
                    group_id=self._uuv_groups.get(uuv_id),
                ).model_dump()
            )
        return states

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
        return {"sim_time_s": sim_time_s, "targets": targets}
