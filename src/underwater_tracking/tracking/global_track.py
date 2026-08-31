"""Executed, globally observable target tracks for UUV-only execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite, pi
from typing import Any

from underwater_tracking.domain.execution_models import (
    GlobalTargetTrackView,
    GlobalTrackSample,
)


def _wrap_angle(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


def _point(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    point = (float(value[0]), float(value[1]))
    if not all(isfinite(item) for item in point):
        raise ValueError(f"{name} must contain finite coordinates")
    return point


@dataclass(frozen=True, slots=True)
class GlobalTrackCheckpoint:
    """Serializable in-memory checkpoint for a global track store."""

    history_limit: int
    tracks: tuple[tuple[str, GlobalTargetTrackView], ...]


GlobalTargetTrack = GlobalTargetTrackView


class GlobalTrackStore:
    """Keep only target motion that has already been physically executed.

    The simulation engine serializes access under its state lock. The store
    itself intentionally has no lock so engine rollback can deepcopy and
    restore it together with the rest of the explicit runtime graph.
    """

    def __init__(
        self,
        history_limit: int = 512,
        *,
        max_history: int | None = None,
    ) -> None:
        selected_limit = max_history if max_history is not None else history_limit
        if selected_limit < 2:
            raise ValueError("global track history_limit must be at least two")
        self.history_limit = int(selected_limit)
        self._tracks: dict[str, GlobalTargetTrackView] = {}

    def observe(
        self,
        target_id: str,
        sim_time_s: float,
        position_xy: tuple[float, float],
        velocity_xy: tuple[float, float] | None = None,
        *,
        heading_rad: float | None = None,
        acceleration_xy: tuple[float, float] | None = None,
        turn_rate_rad_s: float | None = None,
        source_event_ids: tuple[str, ...] = (),
        freshness_status: str = "fresh",
    ) -> GlobalTargetTrackView:
        """Record one executed sample and return the current immutable track."""
        if not target_id or not target_id.strip():
            raise ValueError("target_id must not be empty")
        sim_time = float(sim_time_s)
        if sim_time < 0.0 or not isfinite(sim_time):
            raise ValueError("sim_time_s must be finite and non-negative")
        position = _point(position_xy, "position_xy")
        current = self._tracks.get(target_id)
        history = list(current.bounded_history) if current is not None else []
        if history and sim_time < history[-1].sim_time_s:
            raise ValueError("global track cannot accept an older sample")

        replacing = bool(history and sim_time == history[-1].sim_time_s)
        previous = history[-2] if replacing and len(history) >= 2 else history[-1] if history else None
        delta_s = sim_time - previous.sim_time_s if previous is not None else 0.0

        if velocity_xy is None:
            if previous is not None and delta_s > 0.0:
                velocity = (
                    (position[0] - previous.position_xy[0]) / delta_s,
                    (position[1] - previous.position_xy[1]) / delta_s,
                )
            elif current is not None:
                velocity = current.velocity_xy
            else:
                velocity = (0.0, 0.0)
        else:
            velocity = _point(velocity_xy, "velocity_xy")

        if acceleration_xy is None:
            if previous is not None and delta_s > 0.0:
                acceleration = (
                    (velocity[0] - previous.velocity_xy[0]) / delta_s,
                    (velocity[1] - previous.velocity_xy[1]) / delta_s,
                )
            else:
                acceleration = (0.0, 0.0)
        else:
            acceleration = _point(acceleration_xy, "acceleration_xy")

        if heading_rad is None:
            speed = (velocity[0] * velocity[0] + velocity[1] * velocity[1]) ** 0.5
            if speed > 1e-12:
                heading = atan2(velocity[1], velocity[0])
            elif current is not None:
                heading = current.heading_rad
            else:
                heading = 0.0
        else:
            heading = float(heading_rad)
        if not isfinite(heading):
            raise ValueError("heading_rad must be finite")

        if turn_rate_rad_s is None:
            previous_heading = (
                atan2(previous.velocity_xy[1], previous.velocity_xy[0])
                if previous is not None
                and (previous.velocity_xy[0] != 0.0 or previous.velocity_xy[1] != 0.0)
                else heading
            )
            turn_rate = _wrap_angle(heading - previous_heading) / delta_s if delta_s > 0.0 else 0.0
        else:
            turn_rate = float(turn_rate_rad_s)
        if not isfinite(turn_rate):
            raise ValueError("turn_rate_rad_s must be finite")

        next_revision = current.track_revision + 1 if current is not None else 1
        sample = GlobalTrackSample(
            sim_time_s=sim_time,
            position_xy=position,
            velocity_xy=velocity,
        )
        if replacing:
            history[-1] = sample
        else:
            history.append(sample)
        history = history[-self.history_limit :]
        event_ids = tuple(event_id for event_id in source_event_ids if event_id.strip())
        if not event_ids:
            event_ids = (f"{target_id}:physical-track:{int(sim_time)}:{next_revision}",)
        track = GlobalTargetTrackView(
            target_id=target_id,
            track_revision=next_revision,
            sim_time_s=sim_time,
            position_xy=position,
            velocity_xy=velocity,
            heading_rad=heading,
            acceleration_xy=acceleration,
            turn_rate_rad_s=turn_rate,
            bounded_history=tuple(history),
            source_event_ids=event_ids,
            freshness_status=freshness_status,
        )
        self._tracks[target_id] = track
        return track

    def snapshot(self, target_id: str) -> GlobalTargetTrackView | None:
        """Return the latest track, or ``None`` when no sample exists."""

        return self._tracks.get(target_id)

    def history(self, target_id: str) -> tuple[GlobalTrackSample, ...]:
        """Return the bounded immutable sample history for one target."""

        track = self._tracks.get(target_id)
        return track.bounded_history if track is not None else ()

    def checkpoint(self) -> GlobalTrackCheckpoint:
        """Capture all tracks without exposing the mutable store mapping."""

        return GlobalTrackCheckpoint(
            history_limit=self.history_limit,
            tracks=tuple(
                (target_id, track.model_copy(deep=True))
                for target_id, track in sorted(self._tracks.items())
            ),
        )

    def restore(self, checkpoint: GlobalTrackCheckpoint) -> None:
        """Restore a checkpoint captured from this or an equivalent store."""

        if checkpoint.history_limit < 2:
            raise ValueError("global track checkpoint history_limit must be at least two")
        self.history_limit = checkpoint.history_limit
        self._tracks = {
            target_id: track.model_copy(deep=True)
            for target_id, track in checkpoint.tracks
        }

    def legacy_history(self, target_id: str) -> tuple[tuple[int, float, float], ...]:
        """Return the old tuple projection used by legacy predictor callers."""

        return tuple(
            (int(sample.sim_time_s), sample.position_xy[0], sample.position_xy[1])
            for sample in self.history(target_id)
        )


__all__ = ["GlobalTargetTrack", "GlobalTrackCheckpoint", "GlobalTrackStore"]
