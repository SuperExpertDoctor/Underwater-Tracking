from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pytest

from underwater_tracking.verification import uuv_tracking_coverage_render as render


TRUTH_NOTICE = "Evaluation-only ground truth; unavailable to planner/controller"


@pytest.fixture(scope="session")
def mpl_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("renderer-mplconfig")


def _relative_luminance(colour: str) -> float:
    channels = [int(colour[index : index + 2], 16) / 255.0 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _white_contrast_ratio(colour: str) -> float:
    return 1.05 / (_relative_luminance(colour) + 0.05)


def _lab_colour(colour: str) -> tuple[float, float, float]:
    channels = [int(colour[index : index + 2], 16) / 255.0 for index in (1, 3, 5)]
    red, green, blue = (
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    )
    x = (0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047
    y = (0.2126729 * red + 0.7151522 * green + 0.0721750 * blue) / 1.0
    z = (0.0193339 * red + 0.1191920 * green + 0.9503041 * blue) / 1.08883

    def pivot(value: float) -> float:
        return value ** (1.0 / 3.0) if value > 0.008856 else 7.787 * value + 16.0 / 116.0

    x_value, y_value, z_value = pivot(x), pivot(y), pivot(z)
    return (
        116.0 * y_value - 16.0,
        500.0 * (x_value - y_value),
        200.0 * (y_value - z_value),
    )


def _lab_distance(left: str, right: str) -> float:
    return sum(
        (left_channel - right_channel) ** 2
        for left_channel, right_channel in zip(_lab_colour(left), _lab_colour(right))
    ) ** 0.5


def _synthetic_trace() -> dict[str, object]:
    return {
        "schema_version": 1,
        "scenario": "synthetic",
        "seed": 42,
        "steps": 3,
        "active_ranges_m": {"uuv_00": 20.0, "uuv_01": 6.0},
        "regions": {
            "R1": {
                "polygon": [
                    [-10.0, -10.0],
                    [10.0, -10.0],
                    [10.0, 10.0],
                    [-10.0, 10.0],
                ],
                "target_id": "target_00",
                "active_scan_uuv_ids": ["uuv_00"],
                "passive_track_uuv_ids": ["uuv_01"],
            }
        },
        "routes": {
            "R1": {
                "uuv_00": [[-10.0, -5.0], [35.0, -5.0]],
                "uuv_01": [[10.0, 5.0], [-10.0, 5.0]],
            }
        },
        "frames": [
            {
                "sim_time_s": 5,
                "uuvs": [
                    {
                        "platform_id": "uuv_00",
                        "position_xy": [-1000.0, -1000.0],
                        "deployment_state": "onboard",
                    },
                    {
                        "platform_id": "uuv_01",
                        "position_xy": [10.0, 5.0],
                        "deployment_state": "deployed",
                    },
                ],
                "tracks": [
                    {
                        "target_id": "target_00",
                        "sim_time_s": 5,
                        "mean": [0.0, 1.0, 0.0, 0.0],
                        "covariance": [
                            [4.0, 1.0, 0.0, 0.0],
                            [1.0, 2.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                ],
                "target_truth": [
                    {"target_id": "target_00", "position_xy": [0.0, 0.0]}
                ],
                "waypoint_commands": {
                    "target_00": {
                        "uuv_00": [1000.0, 1000.0],
                        "uuv_01": [16.0, 5.0],
                    }
                },
                "events": [
                    {
                        "event_type": "active_ping",
                        "entity_id": "target_00",
                        "payload": {"emitter_id": "uuv_01"},
                    },
                    {
                        "event_type": "active_ping",
                        "entity_id": "target_00",
                        "payload": {"uuv_id": "uuv_00"},
                    },
                ],
                "mission_modes": {
                    "uuv_00": "TRANSIT",
                    "uuv_01": "PASSIVE_TRACK",
                },
            },
            {
                "sim_time_s": 10,
                "uuvs": [
                    {
                        "platform_id": "uuv_00",
                        "position_xy": [-10.0, -5.0],
                        "deployment_state": "deployed",
                    },
                    {
                        "platform_id": "uuv_01",
                        "position_xy": [9.0, 5.0],
                        "deployment_state": "deployed",
                    },
                ],
                "tracks": [
                    {
                        "target_id": "target_00",
                        "sim_time_s": 5,
                        "mean": [0.5, 1.0, 0.0, 0.0],
                        "covariance": [
                            [4.0, 1.0, 0.0, 0.0],
                            [1.0, 2.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                ],
                "target_truth": [
                    {"target_id": "target_00", "position_xy": [1.0, 0.0]}
                ],
                "waypoint_commands": {
                    "target_00": {
                        "uuv_00": [12.0, -5.0],
                        "uuv_01": [15.0, 5.0],
                    }
                },
                "events": [
                    {
                        "event_type": "active_ping",
                        "entity_id": "target_00",
                        "payload": {"emitter_id": "uuv_00"},
                    }
                ],
                "mission_modes": {
                    "uuv_00": "ACTIVE_SCAN",
                    "uuv_01": "PASSIVE_TRACK",
                },
            },
            {
                "sim_time_s": 15,
                "uuvs": [
                    {
                        "platform_id": "uuv_00",
                        "position_xy": [-5.0, -5.0],
                        "deployment_state": "deployed",
                    },
                    {
                        "platform_id": "uuv_01",
                        "position_xy": [8.0, 5.0],
                        "deployment_state": "deployed",
                    },
                ],
                "tracks": [
                    {
                        "target_id": "target_00",
                        "sim_time_s": 10,
                        "mean": [2.0, 1.0, 0.0, 0.0],
                        "covariance": [
                            [2.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                ],
                "target_truth": [
                    {"target_id": "target_00", "position_xy": [2.0, 0.0]}
                ],
                "waypoint_commands": {
                    "target_00": {
                        "uuv_00": [55.0, -5.0],
                        "uuv_01": [15.0, 5.0],
                    }
                },
                "events": [],
                "mission_modes": {
                    "uuv_00": "PASSIVE_TRACK",
                    "uuv_01": "PASSIVE_TRACK",
                },
            },
        ],
    }


def _write_trace(tmp_path: Path, trace: dict[str, object] | None = None) -> Path:
    path = tmp_path / "trajectory.json"
    path.write_text(
        json.dumps(_synthetic_trace() if trace is None else trace),
        encoding="utf-8",
    )
    return path


def _artist_gids(figure: Any) -> set[str]:
    axes = figure.axes[0]
    artists = [*axes.lines, *axes.patches, *axes.collections]
    return {gid for artist in artists if (gid := artist.get_gid()) is not None}


def test_region_display_label_compacts_task_ids_and_sanitizes_unknown_ids() -> None:
    assert render._region_display_label("target_00:task:01") == "Task 01"

    fallback = render._region_display_label("$north\nsector:" + "x" * 40)

    assert len(fallback) <= 18
    assert "\n" not in fallback
    assert "$" not in fallback


def test_region_and_coverage_uuv_annotations_are_compact_and_offset(
    mpl_config_dir: Path,
) -> None:
    trace = _synthetic_trace()
    regions = trace["regions"]
    routes = trace["routes"]
    regions["target_00:task:01"] = regions.pop("R1")  # type: ignore[union-attr]
    routes["target_00:task:01"] = routes.pop("R1")  # type: ignore[union-attr]

    figure = render._draw_frame(
        trace,
        1,
        view="coverage",
        mpl_config_dir=mpl_config_dir,
    )
    try:
        labels = {
            text.get_gid(): text
            for text in figure.axes[0].texts
            if text.get_gid() is not None
        }
        region_label = labels["region-label:target_00:task:01"]
        uuv_00_label = labels["uuv-mode-label:uuv_00"]
        uuv_01_label = labels["uuv-mode-label:uuv_01"]

        assert region_label.get_text() == "Task 01"
        assert region_label.get_ha() == "center"
        assert region_label.get_va() == "center"
        assert region_label.get_bbox_patch() is not None

        assert uuv_00_label.get_text() == "U00 SCAN"
        assert uuv_01_label.get_text() == "U01 TRACK"
        assert "\n" not in uuv_00_label.get_text()
        assert "\n" not in uuv_01_label.get_text()
        assert uuv_00_label.get_position() == (6, 6)
        assert uuv_01_label.get_position() == (6, -7)
        assert uuv_00_label.get_bbox_patch() is not None
        assert uuv_01_label.get_bbox_patch() is not None
    finally:
        figure.clear()


def test_matplotlib_config_directory_contract(
    tmp_path: Path,
    mpl_config_dir: Path,
) -> None:
    components = render._matplotlib_components(mpl_config_dir)

    import matplotlib

    assert Path(matplotlib.get_configdir()).resolve() == mpl_config_dir.resolve()
    assert render._matplotlib_components(mpl_config_dir) is components
    different_dir = tmp_path / "different-mplconfig"
    with pytest.raises(ValueError, match="already initialized"):
        render._matplotlib_components(different_dir)
    assert not different_dir.exists()


def test_render_keyframes_uses_saved_trace_and_explicit_suffix(
    tmp_path: Path,
    mpl_config_dir: Path,
) -> None:
    trace_path = _write_trace(tmp_path)

    outputs = render.render_keyframes(
        trace_path,
        tmp_path / "media",
        suffix="rerun-01",
        mpl_config_dir=mpl_config_dir,
    )

    assert outputs["tracking"].name == "tracking-keyframe-rerun-01.png"
    assert outputs["coverage"].name == "coverage-keyframe-rerun-01.png"
    for path in outputs.values():
        assert path.is_file()
        assert path.stat().st_size > 0
        pixels = imageio.imread(path)
        assert pixels.shape[:2] == (720, 1200)


@pytest.mark.parametrize("schema_version", [None, 2, True])
def test_render_rejects_wrong_schema_before_creating_output(
    tmp_path: Path,
    schema_version: object,
) -> None:
    trace = _synthetic_trace()
    if schema_version is None:
        trace.pop("schema_version")
    else:
        trace["schema_version"] = schema_version
    trace_path = _write_trace(tmp_path, trace)
    output_dir = tmp_path / "media"

    with pytest.raises(ValueError, match="schema_version=1"):
        render.render_keyframes(trace_path, output_dir)

    assert not output_dir.exists()


def test_render_rejects_non_finite_trace_before_creating_output(tmp_path: Path) -> None:
    trace = _synthetic_trace()
    trace["frames"][0]["uuvs"][0]["position_xy"][0] = float("nan")  # type: ignore[index]
    trace_path = _write_trace(tmp_path, trace)
    output_dir = tmp_path / "media"

    with pytest.raises(ValueError, match="finite"):
        render.render_keyframes(trace_path, output_dir)

    assert not output_dir.exists()


def test_deployed_filter_truth_notice_and_estimate_time_deduplication() -> None:
    trace = _synthetic_trace()
    frames = render._frames(trace)

    assert render._uuv_trail(frames, 2, "uuv_00") == (
        (-10.0, -5.0),
        (-5.0, -5.0),
    )
    assert render._estimate_trail(frames, 2, "target_00") == (
        (0.5, 1.0),
        (2.0, 1.0),
    )
    figure = render._draw_frame(trace, 1, view="tracking")
    try:
        assert TRUTH_NOTICE in {text.get_text() for text in figure.texts}
        assert "waypoint-command:uuv_00" in _artist_gids(figure)
        assert "waypoint-command:uuv_01" in _artist_gids(figure)
        labels = {text.get_text() for text in figure.axes[0].texts}
        assert any("Current target error: unavailable" in text for text in labels)
        assert any("Estimate timestamp: 5 s" in text for text in labels)
        assert any("Estimate age: 5 s" in text for text in labels)
        assert any("Deployed UUVs: 2" in text for text in labels)
        assert any("Source-backed active pings: 1" in text for text in labels)
    finally:
        figure.clear()


def test_both_views_show_commands_source_backed_ping_and_covariance_ellipse() -> None:
    trace = _synthetic_trace()

    for view in ("tracking", "coverage"):
        figure = render._draw_frame(trace, 1, view=view)
        try:
            assert TRUTH_NOTICE in {text.get_text() for text in figure.texts}
            gids = _artist_gids(figure)
            assert "waypoint-command:uuv_00" in gids
            assert "waypoint-command:uuv_01" in gids
            assert "active-ping:uuv_00:0" in gids
            assert not any("legacy" in gid for gid in gids)
            assert "covariance-ellipse:target_00" in gids
            legend_labels = {
                text.get_text() for legend in figure.legends for text in legend.texts
            }
            assert "Initial/baseline assigned route" in legend_labels
            assert figure.axes[0].get_legend() is None
        finally:
            figure.clear()


def test_invalid_covariance_is_not_drawn() -> None:
    trace = _synthetic_trace()
    track = trace["frames"][1]["tracks"][0]  # type: ignore[index]
    track["covariance"] = [[1.0, 2.0], [2.0, 1.0]]

    figure = render._draw_frame(trace, 1, view="tracking")
    try:
        assert "covariance-ellipse:target_00" not in _artist_gids(figure)
    finally:
        figure.clear()


def test_current_error_uses_only_track_fresh_at_frame_time() -> None:
    trace = _synthetic_trace()

    stale_figure = render._draw_frame(trace, 1, view="coverage")
    fresh_figure = render._draw_frame(trace, 0, view="tracking")
    try:
        stale_labels = {text.get_text() for text in stale_figure.axes[0].texts}
        fresh_labels = {text.get_text() for text in fresh_figure.axes[0].texts}
        assert any("Current target error: unavailable" in text for text in stale_labels)
        assert any("Estimate timestamp: 5 s" in text for text in stale_labels)
        assert any("Estimate age: 5 s" in text for text in stale_labels)
        assert any("Current target error: 1.000 m" in text for text in fresh_labels)
        assert any("Estimate timestamp: 5 s" in text for text in fresh_labels)
        assert any("Estimate age: 0 s" in text for text in fresh_labels)
    finally:
        stale_figure.clear()
        fresh_figure.clear()


def test_uuv_palette_has_twelve_unique_nonsemantic_colours() -> None:
    assert len(render._PALETTE) == 12
    assert len(set(render._PALETTE)) == 12
    assert set(render._PALETTE).isdisjoint(render._SEMANTIC_COLOURS)
    assert min(_white_contrast_ratio(colour) for colour in render._PALETTE) >= 3.0
    assert min(
        _lab_distance(left, right)
        for index, left in enumerate(render._PALETTE)
        for right in render._PALETTE[index + 1 :]
    ) >= 18.0
    assert min(
        _lab_distance(uuv_colour, semantic_colour)
        for uuv_colour in render._PALETTE
        for semantic_colour in render._SEMANTIC_COLOURS
    ) >= 15.0

    figure = render._draw_frame(_synthetic_trace(), 1, view="coverage")
    try:
        route_colours = {
            line.get_color()
            for line in figure.axes[0].lines
            if (line.get_gid() or "").startswith("initial-route:")
        }
        assert route_colours == {render._ROUTE_COLOUR}
    finally:
        figure.clear()


def test_axes_canvas_and_legend_geometry_are_fixed_across_frames() -> None:
    trace = _synthetic_trace()
    figures = [
        render._draw_frame(trace, index, view="coverage")
        for index in range(len(trace["frames"]))  # type: ignore[arg-type]
    ]
    try:
        axes = [figure.axes[0] for figure in figures]
        assert all(axis.get_xlim() == axes[0].get_xlim() for axis in axes)
        assert all(axis.get_ylim() == axes[0].get_ylim() for axis in axes)
        assert all(axis.get_position().bounds == axes[0].get_position().bounds for axis in axes)
        assert axes[0].get_xlim()[0] <= -30.0
        assert axes[0].get_xlim()[1] >= 55.0
        arrays = [render._figure_rgb(figure) for figure in figures]
        assert all(array.shape == (720, 1200, 3) for array in arrays)
        assert all(array.dtype == np.uint8 for array in arrays)
        assert all(array.flags.c_contiguous for array in arrays)
    finally:
        for figure in figures:
            figure.clear()


def test_explicit_mpl_config_is_not_replaced_by_nested_helpers(
    mpl_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_default() -> Path:
        raise AssertionError("nested helper replaced explicit MPLCONFIGDIR")

    monkeypatch.setattr(render, "_default_mpl_config_dir", forbidden_default)

    figure = render._draw_frame(
        _synthetic_trace(),
        1,
        view="tracking",
        mpl_config_dir=mpl_config_dir,
    )
    figure.clear()


def test_frame_selection_uses_last_active_scan_or_last_frame() -> None:
    trace = _synthetic_trace()

    assert render._coverage_frame_index(trace) == 1
    for frame in trace["frames"]:  # type: ignore[union-attr]
        frame["mission_modes"] = {"uuv_00": "PASSIVE_TRACK"}
    assert render._coverage_frame_index(trace) == 2
    assert render._tracking_frame_index(trace) == 0

    fresh_then_stale = deepcopy(trace)
    fresh_then_stale["frames"][2]["tracks"][0]["sim_time_s"] = 15  # type: ignore[index]
    stale_tail = deepcopy(fresh_then_stale["frames"][2])  # type: ignore[index]
    stale_tail["sim_time_s"] = 20
    fresh_then_stale["frames"].append(stale_tail)  # type: ignore[union-attr]
    assert render._tracking_frame_index(fresh_then_stale) == 2


def test_keyframe_preflight_refuses_partial_pair_without_new_file(tmp_path: Path) -> None:
    trace_path = _write_trace(tmp_path)
    output_dir = tmp_path / "media"
    output_dir.mkdir()
    existing = output_dir / "coverage-keyframe.png"
    existing.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render.render_keyframes(trace_path, output_dir)

    assert sorted(path.name for path in output_dir.iterdir()) == [existing.name]
    assert existing.read_bytes() == b"existing"


class _FakeWriter:
    def __init__(self, *, fail_append: bool = False) -> None:
        self.fail_append = fail_append
        self.append_count = 0
        self.close_count = 0

    def append_data(self, _frame: np.ndarray[Any, Any]) -> None:
        self.append_count += 1
        if self.fail_append:
            raise RuntimeError("append failed")

    def close(self) -> None:
        self.close_count += 1


class _DelegatingWriter(_FakeWriter):
    def __init__(self, delegate: Any) -> None:
        super().__init__()
        self.delegate = delegate

    def append_data(self, frame: np.ndarray[Any, Any]) -> None:
        super().append_data(frame)
        self.delegate.append_data(frame)

    def close(self) -> None:
        super().close()
        self.delegate.close()


def test_writer_open_failure_closes_every_registered_writer(tmp_path: Path) -> None:
    trace = _synthetic_trace()
    opened = _FakeWriter()
    calls = 0

    def factory(_path: Path, **_kwargs: object) -> _FakeWriter:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("open failed")
        return opened

    with pytest.raises(render._AnimationEncodingError, match="open failed"):
        render._write_animation_pair(
            trace,
            {
                "tracking": tmp_path / "tracking.mp4",
                "coverage": tmp_path / "coverage.mp4",
            },
            fps=2,
            frame_stride=2,
            writer_kwargs={},
            writer_factory=factory,
        )

    assert opened.close_count == 1


def test_writer_append_failure_closes_all_writers(tmp_path: Path) -> None:
    trace = _synthetic_trace()
    writers = [_FakeWriter(fail_append=True), _FakeWriter()]

    def factory(_path: Path, **_kwargs: object) -> _FakeWriter:
        return writers.pop(0)

    registered = list(writers)
    with pytest.raises(render._AnimationEncodingError, match="append failed"):
        render._write_animation_pair(
            trace,
            {
                "tracking": tmp_path / "tracking.mp4",
                "coverage": tmp_path / "coverage.mp4",
            },
            fps=2,
            frame_stride=2,
            writer_kwargs={},
            writer_factory=factory,
        )

    assert [writer.close_count for writer in registered] == [1, 1]


def test_writer_close_failure_is_wrapped_as_encoding_error(tmp_path: Path) -> None:
    trace = _synthetic_trace()

    class CloseFailWriter(_FakeWriter):
        def close(self) -> None:
            super().close()
            raise OSError("close failed")

    writers = [CloseFailWriter(), _FakeWriter()]
    registered = list(writers)

    def factory(_path: Path, **_kwargs: object) -> _FakeWriter:
        return writers.pop(0)

    with pytest.raises(render._AnimationEncodingError, match="close failed"):
        render._write_animation_pair(
            trace,
            {
                "tracking": tmp_path / "tracking.mp4",
                "coverage": tmp_path / "coverage.mp4",
            },
            fps=2,
            frame_stride=2,
            writer_kwargs={},
            writer_factory=factory,
        )

    assert [writer.close_count for writer in registered] == [1, 1]


def test_writer_success_without_output_is_encoding_failure(tmp_path: Path) -> None:
    trace = _synthetic_trace()
    writers: list[_FakeWriter] = []

    def factory(_path: Path, **_kwargs: object) -> _FakeWriter:
        writer = _FakeWriter()
        writers.append(writer)
        return writer

    with pytest.raises(render._AnimationEncodingError, match="missing"):
        render._write_animation_pair(
            trace,
            {
                "tracking": tmp_path / "tracking.mp4",
                "coverage": tmp_path / "coverage.mp4",
            },
            fps=2,
            frame_stride=2,
            writer_kwargs={},
            writer_factory=factory,
        )

    assert [writer.close_count for writer in writers] == [1, 1]


@pytest.mark.parametrize(
    ("content", "message"),
    [(None, "missing"), (b"", "empty"), (b"not-media", "decode")],
)
def test_animation_validation_rejects_missing_empty_or_undecodable_output(
    tmp_path: Path,
    content: bytes | None,
    message: str,
) -> None:
    path = tmp_path / "media.mp4"
    if content is not None:
        path.write_bytes(content)

    with pytest.raises(render._AnimationEncodingError, match=message):
        render._verify_animation_output(
            path,
            expected_frame_count=2,
            expected_shape=(720, 1200, 3),
        )


def test_animation_validation_rejects_wrong_dimensions_and_frame_count(
    tmp_path: Path,
) -> None:
    wrong_size = tmp_path / "wrong-size.gif"
    one_frame = tmp_path / "one-frame.gif"
    imageio.mimsave(
        wrong_size,
        [
            np.zeros((12, 18, 3), dtype=np.uint8),
            np.full((12, 18, 3), 255, dtype=np.uint8),
        ],
        duration=500,
    )
    imageio.mimsave(
        one_frame,
        [np.zeros((720, 1200, 3), dtype=np.uint8)],
        duration=500,
    )

    with pytest.raises(render._AnimationEncodingError, match="dimensions"):
        render._verify_animation_output(
            wrong_size,
            expected_frame_count=2,
            expected_shape=(720, 1200, 3),
        )
    with pytest.raises(render._AnimationEncodingError, match="frame count"):
        render._verify_animation_output(
            one_frame,
            expected_frame_count=2,
            expected_shape=(720, 1200, 3),
        )


def test_no_output_mp4_and_gif_writers_hard_fail(
    tmp_path: Path,
    mpl_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = _write_trace(tmp_path)
    opened_paths: list[Path] = []

    def factory(path: Path, **_kwargs: object) -> _FakeWriter:
        opened_paths.append(path)
        return _FakeWriter()

    monkeypatch.setattr(render, "_open_writer", factory)

    with pytest.raises(RuntimeError, match="both failed"):
        render.render_videos(
            trace_path,
            tmp_path / "media",
            fps=2,
            frame_stride=2,
            mpl_config_dir=mpl_config_dir,
        )

    assert any(path.suffix == ".mp4" for path in opened_paths)
    assert any(path.suffix == ".gif" for path in opened_paths)


def test_existing_mp4_does_not_trigger_gif_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = _write_trace(tmp_path)
    output_dir = tmp_path / "media"
    output_dir.mkdir()
    existing = output_dir / "tracking-control.mp4"
    existing.write_bytes(b"existing")
    calls = 0

    def fail_if_called(_path: Path, **_kwargs: object) -> _FakeWriter:
        nonlocal calls
        calls += 1
        raise AssertionError("writer should not be opened")

    monkeypatch.setattr(render, "_open_writer", fail_if_called)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render.render_videos(trace_path, output_dir)

    assert calls == 0
    assert existing.read_bytes() == b"existing"
    assert not tuple(output_dir.glob("*.gif"))


def test_mp4_failure_returns_real_gif_paths_and_fallback_status(
    tmp_path: Path,
    mpl_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = _write_trace(tmp_path)
    output_dir = tmp_path / "media"
    opened: list[_FakeWriter] = []
    gif_writer_kwargs: list[dict[str, object]] = []
    real_open_writer = render._open_writer

    def factory(path: Path, **_kwargs: object) -> _FakeWriter:
        if path.suffix == ".mp4":
            raise RuntimeError("simulated encoder failure")
        gif_writer_kwargs.append(dict(_kwargs))
        writer = _DelegatingWriter(real_open_writer(path, **_kwargs))
        opened.append(writer)
        return writer

    monkeypatch.setattr(render, "_open_writer", factory)

    result = render.render_videos(
        trace_path,
        output_dir,
        fps=2,
        frame_stride=2,
        suffix="fallback-test",
        mpl_config_dir=mpl_config_dir,
    )

    assert result["format"] == "gif"
    assert result["fallback_used"] is True
    assert result["tracking"].name == "tracking-control-fallback-test.gif"
    assert result["coverage"].name == "coverage-search-fallback-test.gif"
    assert result["tracking"].is_file()
    assert result["coverage"].is_file()
    assert "simulated encoder failure" in result["mp4_error"]
    assert all(writer.close_count == 1 for writer in opened)
    assert gif_writer_kwargs
    assert all(kwargs["duration"] == 500.0 for kwargs in gif_writer_kwargs)


@pytest.mark.parametrize("failure_point", ["draw", "rgb"])
def test_rendering_error_does_not_open_gif_fallback(
    tmp_path: Path,
    mpl_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    trace_path = _write_trace(tmp_path)
    output_dir = tmp_path / "media"
    opened_paths: list[Path] = []
    writers: list[_FakeWriter] = []

    def factory(path: Path, **_kwargs: object) -> _FakeWriter:
        opened_paths.append(path)
        writer = _FakeWriter()
        writers.append(writer)
        return writer

    def fail_render(*_args: object, **_kwargs: object) -> Any:
        raise ValueError(f"{failure_point} failed")

    monkeypatch.setattr(render, "_open_writer", factory)
    monkeypatch.setattr(
        render,
        "_draw_frame" if failure_point == "draw" else "_figure_rgb",
        fail_render,
    )

    with pytest.raises(ValueError, match=f"{failure_point} failed"):
        render.render_videos(
            trace_path,
            output_dir,
            fps=2,
            frame_stride=2,
            mpl_config_dir=mpl_config_dir,
        )

    assert opened_paths
    assert all(path.suffix == ".mp4" for path in opened_paths)
    assert all(writer.close_count == 1 for writer in writers)
    assert not tuple(output_dir.glob("*.gif"))


def test_real_gif_fallback_records_millisecond_frame_duration(
    tmp_path: Path,
    mpl_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image, ImageSequence

    trace = _synthetic_trace()
    trace["steps"] = 2
    trace["frames"] = trace["frames"][:2]  # type: ignore[index]
    trace_path = _write_trace(tmp_path, trace)
    real_open_writer = render._open_writer

    def fail_mp4_only(path: Path, **writer_kwargs: object) -> Any:
        if path.suffix == ".mp4":
            raise OSError("simulated MP4 failure")
        return real_open_writer(path, **writer_kwargs)

    monkeypatch.setattr(render, "_open_writer", fail_mp4_only)

    result = render.render_videos(
        trace_path,
        tmp_path / "media",
        fps=2,
        frame_stride=1,
        mpl_config_dir=mpl_config_dir,
    )

    assert result["format"] == "gif"
    for key in ("tracking", "coverage"):
        with Image.open(result[key]) as animation:
            durations = [
                frame.info.get("duration") for frame in ImageSequence.Iterator(animation)
            ]
        assert durations == [500, 500]


def test_main_reports_success_when_gif_fallback_succeeds(
    tmp_path: Path,
    mpl_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = _write_trace(tmp_path)
    output_dir = tmp_path / "media"
    real_open_writer = render._open_writer

    def factory(path: Path, **_kwargs: object) -> _FakeWriter:
        if path.suffix == ".mp4":
            raise RuntimeError("simulated encoder failure")
        return _DelegatingWriter(real_open_writer(path, **_kwargs))

    monkeypatch.setattr(render, "_open_writer", factory)

    status = render.main(
        [
            "--trace",
            str(trace_path),
            "--output-dir",
            str(output_dir),
            "--fps",
            "2",
            "--frame-stride",
            "2",
            "--mpl-config-dir",
            str(mpl_config_dir),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["videos"]["format"] == "gif"
    assert payload["videos"]["fallback_used"] is True
    assert payload["videos"]["tracking"].endswith("tracking-control.gif")


def test_real_two_frame_mp4_is_readable_from_non_ascii_path(
    tmp_path: Path,
    mpl_config_dir: Path,
) -> None:
    trace = _synthetic_trace()
    trace["steps"] = 2
    trace["frames"] = trace["frames"][:2]  # type: ignore[index]
    non_ascii_dir = tmp_path / "媒体验证"
    non_ascii_dir.mkdir()
    trace_path = _write_trace(non_ascii_dir, trace)

    result = render.render_videos(
        trace_path,
        non_ascii_dir / "输出",
        fps=2,
        frame_stride=1,
        mpl_config_dir=mpl_config_dir,
    )

    assert result["format"] == "mp4"
    assert result["fallback_used"] is False
    for key in ("tracking", "coverage"):
        path = result[key]
        assert path.is_file()
        assert path.stat().st_size > 0
        reader = imageio.get_reader(path)
        try:
            first = reader.get_data(0)
            second = reader.get_data(1)
        finally:
            reader.close()
        assert first.shape == (720, 1200, 3)
        assert second.shape == (720, 1200, 3)
