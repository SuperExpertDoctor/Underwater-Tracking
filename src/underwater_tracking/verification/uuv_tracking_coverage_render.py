"""Headless media rendering for a persisted multi-UUV audit trace."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Callable, Mapping, Sequence
import json
from math import atan2, degrees, hypot, isfinite, sqrt
import os
from pathlib import Path
import re
from typing import Any, cast

import numpy as np

Point = tuple[float, float]
WriterFactory = Callable[..., Any]

_FIGURE_DPI = 100
_FIGURE_SIZE_IN = (12.0, 7.2)
_EXPECTED_FRAME_SHAPE = (
    int(_FIGURE_SIZE_IN[1] * _FIGURE_DPI),
    int(_FIGURE_SIZE_IN[0] * _FIGURE_DPI),
    3,
)
_AXES_RECT = (0.07, 0.11, 0.66, 0.80)
_TRUTH_NOTICE = "Evaluation-only ground truth; unavailable to planner/controller"
_COVARIANCE_95_SCALE = 2.447746830680816
_SUFFIX_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_TASK_REGION_PATTERN = re.compile(r"(?:^|:)task:(?P<number>\d+)(?::|$)", re.IGNORECASE)
_UUV_ID_PATTERN = re.compile(r"^uuv[_-]?(?P<number>\d+)$", re.IGNORECASE)
_UUV_LABEL_OFFSETS = ((6, 6), (6, -7), (-6, 6), (-6, -7))
_MODE_DISPLAY_LABELS = {
    "ACTIVE_SCAN": "SCAN",
    "PASSIVE_TRACK": "TRACK",
    "TRANSIT": "TRANSIT",
    "TRANSIT_TO_REGION": "TRANSIT",
}
_REGION_COLOUR = "#5B6573"
_ROUTE_COLOUR = "#8C8C8C"
_TRUTH_COLOUR = "#111111"
_ESTIMATE_COLOUR = "#D55E00"
_PING_COLOUR = "#009E73"
_COMMAND_COLOUR = "#7F3C8D"
_SEMANTIC_COLOURS = frozenset(
    {
        _REGION_COLOUR,
        _ROUTE_COLOUR,
        _TRUTH_COLOUR,
        _ESTIMATE_COLOUR,
        _PING_COLOUR,
        _COMMAND_COLOUR,
    }
)
_PALETTE = (
    "#096CB2",
    "#DB0000",
    "#009E00",
    "#0000DB",
    "#DB0B96",
    "#614918",
    "#000061",
    "#009E9E",
    "#610F38",
    "#DB6D63",
    "#9F63DB",
    "#C78500",
)
_MPL_COMPONENTS: tuple[Any, ...] | None = None
_MPL_CONFIG_DIR: Path | None = None


class _AnimationEncodingError(RuntimeError):
    """A writer backend failed while opening, appending, or closing media."""


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return cast(Mapping[str, object], value)


def _as_items(value: object) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(value)


def _finite_number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    result = float(value)
    return result if isfinite(result) else None


def _point(value: object) -> Point | None:
    values = _as_items(value)
    if len(values) < 2:
        return None
    x = _finite_number(values[0])
    y = _finite_number(values[1])
    if x is None or y is None:
        return None
    return x, y


def _all_numbers_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_numbers_finite(child) for child in value)
    return False


def _load_trace(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("trajectory root must be a JSON object")
    trace = cast(dict[str, object], payload)
    schema_version = trace.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ValueError("trajectory must declare schema_version=1")
    if not _all_numbers_finite(trace):
        raise ValueError("trajectory numeric values must be finite")
    frames = _as_items(trace.get("frames"))
    if not frames or any(not isinstance(frame, Mapping) for frame in frames):
        raise ValueError("trajectory must contain mapping frames")
    return trace


def _frames(trace: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        cast(Mapping[str, object], frame)
        for frame in _as_items(trace.get("frames"))
        if isinstance(frame, Mapping)
    )


def _validated_suffix(suffix: str) -> str:
    if not suffix:
        return ""
    if _SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise ValueError("suffix must contain only letters, digits, dot, dash, or underscore")
    return f"-{suffix}"


def _media_path(output_dir: Path, stem: str, extension: str, suffix: str) -> Path:
    return output_dir / f"{stem}{_validated_suffix(suffix)}.{extension}"


def _keyframe_paths(output_dir: Path, suffix: str) -> dict[str, Path]:
    return {
        "tracking": _media_path(output_dir, "tracking-keyframe", "png", suffix),
        "coverage": _media_path(output_dir, "coverage-keyframe", "png", suffix),
    }


def _video_paths(output_dir: Path, extension: str, suffix: str) -> dict[str, Path]:
    return {
        "tracking": _media_path(output_dir, "tracking-control", extension, suffix),
        "coverage": _media_path(output_dir, "coverage-search", extension, suffix),
    }


def _preflight_paths(paths: Sequence[Path]) -> None:
    existing = tuple(path for path in paths if path.exists())
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing media: {rendered}")


def _default_mpl_config_dir() -> Path:
    repository_root = Path(__file__).resolve().parents[3]
    return repository_root / "outputs" / "audit-mplconfig"


def _configure_matplotlib(mpl_config_dir: Path | None) -> Path:
    config_dir = (mpl_config_dir or _default_mpl_config_dir()).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    # This changes only the renderer process. It never edits the user's global config.
    os.environ["MPLCONFIGDIR"] = str(config_dir)
    return config_dir


def _matplotlib_components(mpl_config_dir: Path | None = None) -> tuple[Any, ...]:
    global _MPL_COMPONENTS, _MPL_CONFIG_DIR
    if _MPL_COMPONENTS is not None:
        if mpl_config_dir is not None:
            requested_dir = mpl_config_dir.resolve()
            if requested_dir != _MPL_CONFIG_DIR:
                raise ValueError(
                    "Matplotlib is already initialized with "
                    f"MPLCONFIGDIR={_MPL_CONFIG_DIR}; cannot switch to {requested_dir}"
                )
        return _MPL_COMPONENTS
    config_dir = _configure_matplotlib(mpl_config_dir)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Ellipse, Patch

    _MPL_CONFIG_DIR = config_dir
    _MPL_COMPONENTS = FigureCanvasAgg, Figure, Line2D, Circle, Ellipse, Patch
    return _MPL_COMPONENTS


def _uuv_ids(trace: Mapping[str, object]) -> tuple[str, ...]:
    identifiers: set[str] = set()
    for frame in _frames(trace):
        for raw in _as_items(frame.get("uuvs")):
            uuv_id = _as_mapping(raw).get("platform_id")
            if isinstance(uuv_id, str):
                identifiers.add(uuv_id)
    for raw_region_routes in _as_mapping(trace.get("routes")).values():
        identifiers.update(
            uuv_id
            for uuv_id in _as_mapping(raw_region_routes)
            if isinstance(uuv_id, str)
        )
    return tuple(sorted(identifiers))


def _colour_by_uuv(trace: Mapping[str, object]) -> dict[str, str]:
    return {
        uuv_id: _PALETTE[index % len(_PALETTE)]
        for index, uuv_id in enumerate(_uuv_ids(trace))
    }


def _positions_by_uuv(
    frame: Mapping[str, object],
    *,
    deployed_only: bool = True,
) -> dict[str, Point]:
    positions: dict[str, Point] = {}
    for raw in _as_items(frame.get("uuvs")):
        item = _as_mapping(raw)
        if deployed_only and item.get("deployment_state") != "deployed":
            continue
        uuv_id = item.get("platform_id")
        point = _point(item.get("position_xy"))
        if isinstance(uuv_id, str) and point is not None:
            positions[uuv_id] = point
    return positions


def _uuv_trail(
    frames: Sequence[Mapping[str, object]],
    frame_index: int,
    uuv_id: str,
) -> tuple[Point, ...]:
    return tuple(
        point
        for frame in frames[: frame_index + 1]
        if (point := _positions_by_uuv(frame).get(uuv_id)) is not None
    )


def _target_truth_trail(
    frames: Sequence[Mapping[str, object]],
    frame_index: int,
    target_id: str,
) -> tuple[Point, ...]:
    points: list[Point] = []
    for frame in frames[: frame_index + 1]:
        for raw in _as_items(frame.get("target_truth")):
            item = _as_mapping(raw)
            if item.get("target_id") != target_id:
                continue
            point = _point(item.get("position_xy"))
            if point is not None:
                points.append(point)
            break
    return tuple(points)


def _estimate_trail(
    frames: Sequence[Mapping[str, object]],
    frame_index: int,
    target_id: str,
) -> tuple[Point, ...]:
    by_track_time: dict[float, Point] = {}
    for frame in frames[: frame_index + 1]:
        for raw in _as_items(frame.get("tracks")):
            item = _as_mapping(raw)
            if item.get("target_id") != target_id:
                continue
            track_time = _finite_number(item.get("sim_time_s"))
            point = _point(item.get("mean"))
            if track_time is not None and point is not None:
                by_track_time[track_time] = point
    return tuple(point for _, point in sorted(by_track_time.items()))


def _target_ids(trace: Mapping[str, object]) -> tuple[str, ...]:
    identifiers: set[str] = set()
    for frame in _frames(trace):
        for collection in ("target_truth", "tracks"):
            for raw in _as_items(frame.get(collection)):
                target_id = _as_mapping(raw).get("target_id")
                if isinstance(target_id, str):
                    identifiers.add(target_id)
    return tuple(sorted(identifiers))


def _commands(frame: Mapping[str, object]) -> tuple[tuple[str, Point, Point], ...]:
    positions = _positions_by_uuv(frame)
    commands: list[tuple[str, Point, Point]] = []
    for raw_by_target in _as_mapping(frame.get("waypoint_commands")).values():
        for uuv_id, raw_destination in sorted(_as_mapping(raw_by_target).items()):
            start = positions.get(uuv_id)
            destination = _point(raw_destination)
            if start is not None and destination is not None:
                commands.append((uuv_id, start, destination))
    return tuple(commands)


def _source_backed_pings(
    trace: Mapping[str, object],
    frame: Mapping[str, object],
) -> tuple[tuple[str, Point, float, int], ...]:
    positions = _positions_by_uuv(frame)
    ranges = _as_mapping(trace.get("active_ranges_m"))
    source_counts: dict[str, int] = {}
    pings: list[tuple[str, Point, float, int]] = []
    for raw_event in _as_items(frame.get("events")):
        event = _as_mapping(raw_event)
        if event.get("event_type") != "active_ping":
            continue
        emitter_id = _as_mapping(event.get("payload")).get("emitter_id")
        if not isinstance(emitter_id, str):
            continue
        center = positions.get(emitter_id)
        radius = _finite_number(ranges.get(emitter_id))
        if center is None or radius is None or radius <= 0.0:
            continue
        source_index = source_counts.get(emitter_id, 0)
        source_counts[emitter_id] = source_index + 1
        pings.append((emitter_id, center, radius, source_index))
    return tuple(pings)


def _covariance_ellipse(
    track: Mapping[str, object],
) -> tuple[Point, float, float, float] | None:
    center = _point(track.get("mean"))
    raw_rows = _as_items(track.get("covariance"))
    if center is None or len(raw_rows) < 2:
        return None
    row_0 = _as_items(raw_rows[0])
    row_1 = _as_items(raw_rows[1])
    if len(row_0) < 2 or len(row_1) < 2:
        return None
    values = tuple(
        _finite_number(value)
        for value in (row_0[0], row_0[1], row_1[0], row_1[1])
    )
    if any(value is None for value in values):
        return None
    covariance: np.ndarray[Any, Any] = np.asarray(
        cast(tuple[float, ...], values), dtype=np.float64
    ).reshape(2, 2)
    if not np.allclose(covariance, covariance.T, rtol=1e-9, atol=1e-9):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -tolerance:
        return None
    order = np.argsort(eigenvalues)[::-1]
    major_value = max(0.0, float(eigenvalues[order[0]]))
    minor_value = max(0.0, float(eigenvalues[order[1]]))
    major_vector = eigenvectors[:, order[0]]
    angle_deg = degrees(atan2(float(major_vector[1]), float(major_vector[0])))
    width = 2.0 * _COVARIANCE_95_SCALE * sqrt(major_value)
    height = 2.0 * _COVARIANCE_95_SCALE * sqrt(minor_value)
    return center, width, height, angle_deg


def _current_tracks(frame: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _as_mapping(raw)
        for raw in _as_items(frame.get("tracks"))
        if isinstance(_as_mapping(raw).get("target_id"), str)
    )


def _fresh_tracks(frame: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    frame_time = _finite_number(frame.get("sim_time_s"))
    if frame_time is None:
        return ()
    return tuple(
        track
        for track in _current_tracks(frame)
        if _finite_number(track.get("sim_time_s")) == frame_time
    )


def _estimate_timestamp_and_age(
    frame: Mapping[str, object],
) -> tuple[float | None, float | None]:
    timestamps = tuple(
        timestamp
        for track in _current_tracks(frame)
        if (timestamp := _finite_number(track.get("sim_time_s"))) is not None
    )
    frame_time = _finite_number(frame.get("sim_time_s"))
    if not timestamps or frame_time is None:
        return None, None
    latest_timestamp = max(timestamps)
    return latest_timestamp, frame_time - latest_timestamp


def _all_plot_points(trace: Mapping[str, object]) -> tuple[Point, ...]:
    points: list[Point] = []
    for raw_region in _as_mapping(trace.get("regions")).values():
        points.extend(
            point
            for raw_point in _as_items(_as_mapping(raw_region).get("polygon"))
            if (point := _point(raw_point)) is not None
        )
    for raw_region_routes in _as_mapping(trace.get("routes")).values():
        for raw_route in _as_mapping(raw_region_routes).values():
            points.extend(
                point
                for raw_point in _as_items(raw_route)
                if (point := _point(raw_point)) is not None
            )
    for frame in _frames(trace):
        points.extend(_positions_by_uuv(frame).values())
        for raw_truth in _as_items(frame.get("target_truth")):
            point = _point(_as_mapping(raw_truth).get("position_xy"))
            if point is not None:
                points.append(point)
        for track in _current_tracks(frame):
            point = _point(track.get("mean"))
            if point is not None:
                points.append(point)
            ellipse = _covariance_ellipse(track)
            if ellipse is not None:
                center, width, height, _ = ellipse
                radius = max(width, height) / 2.0
                points.extend(
                    (
                        (center[0] - radius, center[1] - radius),
                        (center[0] + radius, center[1] + radius),
                    )
                )
        for _, start, destination in _commands(frame):
            points.extend((start, destination))
        for _, center, radius, _ in _source_backed_pings(trace, frame):
            points.extend(
                (
                    (center[0] - radius, center[1] - radius),
                    (center[0] + radius, center[1] + radius),
                )
            )
    return tuple(points)


def _axis_bounds(trace: Mapping[str, object]) -> tuple[float, float, float, float]:
    points = _all_plot_points(trace)
    if not points:
        return -1.0, 1.0, -1.0, 1.0
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    span = max(max_x - min_x, max_y - min_y, 1.0)
    margin = 0.05 * span
    return min_x - margin, max_x + margin, min_y - margin, max_y + margin


def _compact_display_text(value: object, *, fallback: str, max_length: int) -> str:
    if not isinstance(value, str):
        return fallback
    single_line = " ".join(value.split())
    safe = "".join(
        character if character.isalnum() or character in "._- " else " "
        for character in single_line
    )
    compact = " ".join(safe.split()).strip(" ._-")
    if not compact:
        return fallback
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 3].rstrip()}..."


def _region_display_label(region_id: object) -> str:
    if isinstance(region_id, str):
        task_match = _TASK_REGION_PATTERN.search(region_id)
        if task_match is not None:
            return f"Task {task_match.group('number')}"
    return _compact_display_text(region_id, fallback="Region", max_length=18)


def _uuv_mode_display_label(uuv_id: str, mode: object) -> str:
    uuv_match = _UUV_ID_PATTERN.fullmatch(uuv_id)
    identifier = (
        f"U{uuv_match.group('number')}"
        if uuv_match is not None
        else _compact_display_text(uuv_id, fallback="UUV", max_length=8)
    )
    mode_label = _MODE_DISPLAY_LABELS.get(
        mode,
        _compact_display_text(mode, fallback="UNKNOWN", max_length=8).upper(),
    )
    return f"{identifier} {mode_label}"


def _draw_regions_and_routes(
    axes: Any,
    trace: Mapping[str, object],
) -> None:
    routes = _as_mapping(trace.get("routes"))
    for region_id, raw_region in sorted(_as_mapping(trace.get("regions")).items()):
        polygon = tuple(
            point
            for raw_point in _as_items(_as_mapping(raw_region).get("polygon"))
            if (point := _point(raw_point)) is not None
        )
        if polygon:
            closed = (*polygon, polygon[0])
            axes.plot(
                [point[0] for point in closed],
                [point[1] for point in closed],
                color=_REGION_COLOUR,
                linewidth=1.4,
                gid=f"region:{region_id}",
            )
            centroid = (
                sum(point[0] for point in polygon) / len(polygon),
                sum(point[1] for point in polygon) / len(polygon),
            )
            axes.annotate(
                _region_display_label(region_id),
                centroid,
                color="#444444",
                fontsize=7,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "square,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": _REGION_COLOUR,
                    "linewidth": 0.5,
                    "alpha": 0.78,
                },
                zorder=4,
                gid=f"region-label:{region_id}",
            )
        for uuv_id, raw_route in sorted(_as_mapping(routes.get(region_id)).items()):
            route = tuple(
                point
                for raw_point in _as_items(raw_route)
                if (point := _point(raw_point)) is not None
            )
            if route:
                axes.plot(
                    [point[0] for point in route],
                    [point[1] for point in route],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.72,
                    color=_ROUTE_COLOUR,
                    gid=f"initial-route:{region_id}:{uuv_id}",
                )
                axes.annotate(
                    uuv_id,
                    route[0],
                    xytext=(3, -8),
                    textcoords="offset points",
                    fontsize=6,
                    color=_ROUTE_COLOUR,
                )


def _draw_uuvs(
    axes: Any,
    frames: Sequence[Mapping[str, object]],
    frame_index: int,
    colours: Mapping[str, str],
    *,
    show_modes: bool,
) -> None:
    current = _positions_by_uuv(frames[frame_index])
    modes = _as_mapping(frames[frame_index].get("mission_modes"))
    for label_index, (uuv_id, colour) in enumerate(colours.items()):
        trail = _uuv_trail(frames, frame_index, uuv_id)
        if trail:
            axes.plot(
                [point[0] for point in trail],
                [point[1] for point in trail],
                color=colour,
                linewidth=1.8,
                gid=f"deployed-trail:{uuv_id}",
            )
        point = current.get(uuv_id)
        if point is None:
            continue
        axes.scatter(
            [point[0]],
            [point[1]],
            color=colour,
            edgecolors="white",
            linewidths=0.5,
            marker="o",
            s=42,
            zorder=6,
            gid=f"deployed-uuv:{uuv_id}",
        )
        if show_modes:
            offset_x, offset_y = _UUV_LABEL_OFFSETS[
                label_index % len(_UUV_LABEL_OFFSETS)
            ]
            axes.annotate(
                _uuv_mode_display_label(uuv_id, modes.get(uuv_id, "UNKNOWN")),
                point,
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                fontsize=6.5,
                color=colour,
                ha="left" if offset_x > 0 else "right",
                va="bottom" if offset_y > 0 else "top",
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
                zorder=8,
                gid=f"uuv-mode-label:{uuv_id}",
            )


def _draw_tracking(
    axes: Any,
    trace: Mapping[str, object],
    frames: Sequence[Mapping[str, object]],
    frame_index: int,
) -> None:
    for target_id in _target_ids(trace):
        truth = _target_truth_trail(frames, frame_index, target_id)
        estimate = _estimate_trail(frames, frame_index, target_id)
        if truth:
            axes.plot(
                [point[0] for point in truth],
                [point[1] for point in truth],
                color=_TRUTH_COLOUR,
                linewidth=2.2,
                gid=f"evaluation-truth:{target_id}",
            )
            axes.scatter(
                [truth[-1][0]],
                [truth[-1][1]],
                color=_TRUTH_COLOUR,
                marker="*",
                s=95,
                zorder=7,
                gid=f"evaluation-truth-current:{target_id}",
            )
        if estimate:
            axes.plot(
                [point[0] for point in estimate],
                [point[1] for point in estimate],
                color=_ESTIMATE_COLOUR,
                linewidth=2.0,
                gid=f"fused-estimate:{target_id}",
            )
            axes.scatter(
                [estimate[-1][0]],
                [estimate[-1][1]],
                color=_ESTIMATE_COLOUR,
                marker="x",
                s=60,
                zorder=7,
                gid=f"fused-estimate-current:{target_id}",
            )


def _draw_covariance_ellipses(axes: Any, frame: Mapping[str, object]) -> None:
    _, _, _, _, Ellipse, _ = _matplotlib_components()
    for track in _current_tracks(frame):
        target_id = cast(str, track["target_id"])
        ellipse = _covariance_ellipse(track)
        if ellipse is None:
            continue
        center, width, height, angle_deg = ellipse
        patch = Ellipse(
            center,
            width=width,
            height=height,
            angle=angle_deg,
            fill=False,
            edgecolor=_ESTIMATE_COLOUR,
            linewidth=1.2,
            alpha=0.8,
        )
        patch.set_gid(f"covariance-ellipse:{target_id}")
        axes.add_patch(patch)


def _draw_commands(axes: Any, frame: Mapping[str, object]) -> None:
    for uuv_id, start, destination in _commands(frame):
        axes.plot(
            [start[0], destination[0]],
            [start[1], destination[1]],
            color=_COMMAND_COLOUR,
            linewidth=1.2,
            alpha=0.75,
            marker=">",
            markevery=[-1],
            markersize=5,
            gid=f"waypoint-command:{uuv_id}",
        )


def _draw_pings(
    axes: Any,
    trace: Mapping[str, object],
    frame: Mapping[str, object],
) -> int:
    _, _, _, Circle, _, _ = _matplotlib_components()
    pings = _source_backed_pings(trace, frame)
    for emitter_id, center, radius, source_index in pings:
        circle = Circle(
            center,
            radius=radius,
            fill=False,
            edgecolor=_PING_COLOUR,
            linewidth=1.0,
            linestyle=":",
            alpha=0.68,
        )
        circle.set_gid(f"active-ping:{emitter_id}:{source_index}")
        axes.add_patch(circle)
    return len(pings)


def _current_target_error(frame: Mapping[str, object]) -> float | None:
    truths = {
        target_id: point
        for raw in _as_items(frame.get("target_truth"))
        if isinstance((target_id := _as_mapping(raw).get("target_id")), str)
        and (point := _point(_as_mapping(raw).get("position_xy"))) is not None
    }
    estimates = {
        target_id: point
        for track in _fresh_tracks(frame)
        if isinstance((target_id := track.get("target_id")), str)
        and (point := _point(track.get("mean"))) is not None
    }
    errors = [
        hypot(estimates[target_id][0] - truth[0], estimates[target_id][1] - truth[1])
        for target_id, truth in sorted(truths.items())
        if target_id in estimates
    ]
    return errors[0] if errors else None


def _draw_descriptive_values(
    axes: Any,
    frame: Mapping[str, object],
    *,
    active_ping_count: int,
) -> None:
    deployed_count = len(_positions_by_uuv(frame))
    mode_values = _as_mapping(frame.get("mission_modes")).values()
    active_scan_count = sum(mode == "ACTIVE_SCAN" for mode in mode_values)
    error = _current_target_error(frame)
    error_text = "unavailable" if error is None else f"{error:.3f} m"
    estimate_timestamp, estimate_age = _estimate_timestamp_and_age(frame)
    timestamp_text = (
        "unavailable" if estimate_timestamp is None else f"{estimate_timestamp:g} s"
    )
    age_text = "unavailable" if estimate_age is None else f"{estimate_age:g} s"
    axes.text(
        0.012,
        0.985,
        "\n".join(
            (
                f"Deployed UUVs: {deployed_count}",
                f"ACTIVE_SCAN UUVs: {active_scan_count}",
                f"Source-backed active pings: {active_ping_count}",
                f"Current target error: {error_text}",
                f"Estimate timestamp: {timestamp_text}",
                f"Estimate age: {age_text}",
            )
        ),
        transform=axes.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#20252B",
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "alpha": 0.9},
        zorder=10,
    )


def _static_legend(
    figure: Any,
    trace: Mapping[str, object],
    colours: Mapping[str, str],
) -> None:
    _, _, Line2D, Circle, _, Patch = _matplotlib_components()
    handles: list[Any] = [
        Line2D([], [], color=_REGION_COLOUR, linewidth=1.4, label="Task region"),
        Line2D(
            [],
            [],
            color=_ROUTE_COLOUR,
            linestyle="--",
            linewidth=1.2,
            label="Initial/baseline assigned route",
        ),
    ]
    handles.extend(
        Line2D(
            [],
            [],
            color=colours[uuv_id],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=f"{uuv_id} deployed trail",
        )
        for uuv_id in _uuv_ids(trace)
    )
    handles.extend(
        (
            Line2D(
                [],
                [],
                color=_COMMAND_COLOUR,
                marker=">",
                linewidth=1.2,
                label="Current waypoint command",
            ),
            Circle(
                (0.0, 0.0),
                radius=0.5,
                fill=False,
                edgecolor=_PING_COLOUR,
                linestyle=":",
                label="Source-backed active sonar range",
            ),
            Line2D(
                [],
                [],
                color=_ESTIMATE_COLOUR,
                linewidth=2.0,
                marker="x",
                label="Fused target estimate",
            ),
            Patch(
                facecolor="none",
                edgecolor=_ESTIMATE_COLOUR,
                label="Finite PSD 95% covariance ellipse",
            ),
            Line2D(
                [],
                [],
                color=_TRUTH_COLOUR,
                linewidth=2.2,
                marker="*",
                label="Evaluation-only target truth",
            ),
        )
    )
    figure.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.76, 0.50),
        bbox_transform=figure.transFigure,
        fontsize=7,
        framealpha=0.95,
        borderaxespad=0.0,
    )


def _draw_frame(
    trace: Mapping[str, object],
    frame_index: int,
    *,
    view: str,
    mpl_config_dir: Path | None = None,
) -> Any:
    frames = _frames(trace)
    if not 0 <= frame_index < len(frames):
        raise IndexError("frame index is outside the trace")
    if view not in {"tracking", "coverage"}:
        raise ValueError(f"unsupported render view: {view}")
    FigureCanvasAgg, Figure, _, _, _, _ = _matplotlib_components(mpl_config_dir)
    figure = Figure(figsize=_FIGURE_SIZE_IN, dpi=_FIGURE_DPI, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_axes(_AXES_RECT)
    colours = _colour_by_uuv(trace)
    frame = frames[frame_index]

    _draw_regions_and_routes(axes, trace)
    _draw_uuvs(
        axes,
        frames,
        frame_index,
        colours,
        show_modes=view == "coverage",
    )
    if view == "tracking":
        _draw_tracking(axes, trace, frames, frame_index)
    figure.text(
        0.07,
        0.025,
        _TRUTH_NOTICE,
        fontsize=9,
        weight="bold",
        color="#8B1E1E",
    )
    _draw_covariance_ellipses(axes, frame)
    _draw_commands(axes, frame)
    ping_count = _draw_pings(axes, trace, frame)
    _draw_descriptive_values(axes, frame, active_ping_count=ping_count)

    min_x, max_x, min_y, max_y = _axis_bounds(trace)
    axes.set_xlim(min_x, max_x)
    axes.set_ylim(min_y, max_y)
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, alpha=0.25)
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    sim_time_s = frame.get("sim_time_s")
    title = (
        "Multi-UUV tracking and control"
        if view == "tracking"
        else "Multi-UUV serpentine coverage search"
    )
    axes.set_title(f"{title} - t={sim_time_s} s")
    _static_legend(figure, trace, colours)
    return figure


def _figure_rgb(figure: Any) -> np.ndarray[Any, Any]:
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    rgb = np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)
    return cast(np.ndarray[Any, Any], rgb)


def _coverage_frame_index(trace: Mapping[str, object]) -> int:
    frames = _frames(trace)
    candidates = [
        index
        for index, frame in enumerate(frames)
        if "ACTIVE_SCAN" in _as_mapping(frame.get("mission_modes")).values()
    ]
    return candidates[-1] if candidates else len(frames) - 1


def _tracking_frame_index(trace: Mapping[str, object]) -> int:
    frames = _frames(trace)
    candidates = [
        index
        for index, frame in enumerate(frames)
        if _fresh_tracks(frame)
    ]
    return candidates[-1] if candidates else len(frames) - 1


def _write_keyframe_pair(
    trace: Mapping[str, object],
    outputs: Mapping[str, Path],
    *,
    mpl_config_dir: Path | None,
) -> None:
    figures = {
        "tracking": _draw_frame(
            trace,
            _tracking_frame_index(trace),
            view="tracking",
            mpl_config_dir=mpl_config_dir,
        ),
        "coverage": _draw_frame(
            trace,
            _coverage_frame_index(trace),
            view="coverage",
            mpl_config_dir=mpl_config_dir,
        ),
    }
    try:
        for view, figure in figures.items():
            with outputs[view].open("xb") as stream:
                figure.savefig(
                    stream,
                    format="png",
                    dpi=_FIGURE_DPI,
                    facecolor="white",
                )
    finally:
        for figure in figures.values():
            figure.clear()


def render_keyframes(
    trace_path: Path,
    output_dir: Path,
    *,
    suffix: str = "",
    mpl_config_dir: Path | None = None,
) -> dict[str, Path]:
    """Render a non-overwriting tracking/coverage keyframe pair."""
    trace = _load_trace(trace_path)
    outputs = _keyframe_paths(output_dir, suffix)
    _preflight_paths(tuple(outputs.values()))
    _matplotlib_components(mpl_config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_keyframe_pair(trace, outputs, mpl_config_dir=mpl_config_dir)
    return outputs


def _animation_indices(frame_count: int, frame_stride: int) -> tuple[int, ...]:
    indices = list(range(0, frame_count, frame_stride))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return tuple(indices)


def _open_writer(path: Path, **writer_kwargs: object) -> Any:
    import imageio.v2 as imageio

    get_writer = cast(Any, imageio.get_writer)
    return get_writer(str(path), **writer_kwargs)


def _open_reader(path: Path) -> Any:
    import imageio.v2 as imageio

    get_reader = cast(Any, imageio.get_reader)
    return get_reader(str(path))


def _reader_frame_count(reader: Any) -> int:
    count_frames = getattr(reader, "count_frames", None)
    if callable(count_frames):
        count = count_frames()
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    get_length = getattr(reader, "get_length", None)
    if callable(get_length):
        length = get_length()
        numeric_length = _finite_number(length)
        if numeric_length is not None and numeric_length.is_integer():
            return int(numeric_length)
    raise ValueError("decoder did not expose a finite frame count")


def _verify_animation_output(
    path: Path,
    *,
    expected_frame_count: int,
    expected_shape: tuple[int, int, int],
) -> None:
    if not path.is_file():
        raise _AnimationEncodingError(f"animation output is missing: {path}")
    try:
        size_bytes = path.stat().st_size
    except OSError as error:
        raise _AnimationEncodingError(
            f"animation output size is unreadable for {path}: {error}"
        ) from error
    if size_bytes <= 0:
        raise _AnimationEncodingError(f"animation output is empty: {path}")

    reader: Any | None = None
    active_error: BaseException | None = None
    try:
        try:
            reader = _open_reader(path)
        except Exception as error:
            raise _AnimationEncodingError(
                f"animation decode open failed for {path}: {error}"
            ) from error
        try:
            frame_count = _reader_frame_count(reader)
        except Exception as error:
            raise _AnimationEncodingError(
                f"animation decode frame count failed for {path}: {error}"
            ) from error
        if frame_count != expected_frame_count:
            raise _AnimationEncodingError(
                "animation frame count mismatch for "
                f"{path}: expected {expected_frame_count}, got {frame_count}"
            )

        # Lossy codecs cannot support exact pixel equality. Exact frame count plus
        # ordinal reads prove that the first and final expected samples are decodable.
        for label, frame_index in (
            ("first", 0),
            ("last", expected_frame_count - 1),
        ):
            try:
                frame = np.asarray(reader.get_data(frame_index))
            except Exception as error:
                raise _AnimationEncodingError(
                    f"animation {label} frame decode failed for {path}: {error}"
                ) from error
            if tuple(frame.shape) != expected_shape:
                raise _AnimationEncodingError(
                    f"animation {label} frame dimensions mismatch for {path}: "
                    f"expected {expected_shape}, got {tuple(frame.shape)}"
                )
    except BaseException as error:
        active_error = error
        raise
    finally:
        if reader is not None:
            try:
                reader.close()
            except BaseException as error:
                if active_error is None:
                    if isinstance(error, Exception):
                        raise _AnimationEncodingError(
                            f"animation decoder close failed for {path}: {error}"
                        ) from error
                    raise


def _write_animation_pair(
    trace: Mapping[str, object],
    outputs: Mapping[str, Path],
    *,
    fps: int,
    frame_stride: int,
    writer_kwargs: Mapping[str, object],
    writer_factory: WriterFactory = _open_writer,
) -> None:
    _preflight_paths(tuple(outputs.values()))
    frame_indices = _animation_indices(len(_frames(trace)), frame_stride)
    writers: dict[str, Any] = {}
    active_error: BaseException | None = None
    try:
        for view, path in outputs.items():
            try:
                writers[view] = writer_factory(path, **writer_kwargs)
            except FileExistsError:
                raise
            except Exception as error:
                raise _AnimationEncodingError(
                    f"animation writer open failed for {path}: {error}"
                ) from error
        for frame_index in frame_indices:
            for view, writer in writers.items():
                figure = _draw_frame(trace, frame_index, view=view)
                try:
                    frame_rgb = _figure_rgb(figure)
                    try:
                        writer.append_data(frame_rgb)
                    except FileExistsError:
                        raise
                    except Exception as error:
                        raise _AnimationEncodingError(
                            "animation writer append failed for "
                            f"{outputs[view]}: {error}"
                        ) from error
                finally:
                    figure.clear()
    except BaseException as error:
        active_error = error
        raise
    finally:
        close_errors: list[tuple[Path, BaseException]] = []
        for view, writer in writers.items():
            try:
                writer.close()
            except BaseException as error:  # noqa: BLE001 - close every registered writer
                close_errors.append((outputs[view], error))
        if close_errors and active_error is None:
            failed_path, close_error = close_errors[0]
            if isinstance(close_error, FileExistsError):
                raise close_error
            if isinstance(close_error, Exception):
                raise _AnimationEncodingError(
                    f"animation writer close failed for {failed_path}: {close_error}"
                ) from close_error
            raise close_error
    for path in outputs.values():
        _verify_animation_output(
            path,
            expected_frame_count=len(frame_indices),
            expected_shape=_EXPECTED_FRAME_SHAPE,
        )


def _encoder_error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def render_videos(
    trace_path: Path,
    output_dir: Path,
    *,
    fps: int = 10,
    frame_stride: int = 3,
    suffix: str = "",
    mpl_config_dir: Path | None = None,
) -> dict[str, object]:
    """Render MP4 media, returning successful GIF paths on encoder fallback."""
    if fps < 1 or frame_stride < 1:
        raise ValueError("fps and frame_stride must be positive")
    trace = _load_trace(trace_path)
    mp4_outputs = _video_paths(output_dir, "mp4", suffix)
    gif_outputs = _video_paths(output_dir, "gif", suffix)
    _preflight_paths((*mp4_outputs.values(), *gif_outputs.values()))
    _matplotlib_components(mpl_config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _write_animation_pair(
            trace,
            mp4_outputs,
            fps=fps,
            frame_stride=frame_stride,
            writer_kwargs={"codec": "libx264", "fps": fps, "quality": 8},
            writer_factory=_open_writer,
        )
    except _AnimationEncodingError as encoder_error:
        try:
            _write_animation_pair(
                trace,
                gif_outputs,
                fps=fps,
                frame_stride=frame_stride,
                writer_kwargs={
                    "mode": "I",
                    "duration": 1000.0 / fps,
                    "loop": 0,
                },
                writer_factory=_open_writer,
            )
        except _AnimationEncodingError as fallback_error:
            raise RuntimeError(
                "MP4 encoding and GIF fallback both failed; partial files were preserved"
            ) from fallback_error
        return {
            **gif_outputs,
            "format": "gif",
            "status": "GIF_FALLBACK",
            "fallback_used": True,
            "mp4_error": _encoder_error_text(encoder_error),
        }
    return {
        **mp4_outputs,
        "format": "mp4",
        "status": "MP4",
        "fallback_used": False,
        "mp4_error": None,
    }


def _jsonable_media_result(result: Mapping[str, object]) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Render media from a persisted multi-UUV audit trace."
    )
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--mpl-config-dir", type=Path)
    args = parser.parse_args(argv)

    keyframe_paths = _keyframe_paths(args.output_dir, args.suffix)
    mp4_paths = _video_paths(args.output_dir, "mp4", args.suffix)
    gif_paths = _video_paths(args.output_dir, "gif", args.suffix)
    _preflight_paths(
        (*keyframe_paths.values(), *mp4_paths.values(), *gif_paths.values())
    )
    keyframes = render_keyframes(
        args.trace,
        args.output_dir,
        suffix=args.suffix,
        mpl_config_dir=args.mpl_config_dir,
    )
    videos = render_videos(
        args.trace,
        args.output_dir,
        fps=args.fps,
        frame_stride=args.frame_stride,
        suffix=args.suffix,
        mpl_config_dir=args.mpl_config_dir,
    )
    print(
        json.dumps(
            {
                "keyframes": {key: str(path) for key, path in keyframes.items()},
                "videos": _jsonable_media_result(videos),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0
