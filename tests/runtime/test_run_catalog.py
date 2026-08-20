from __future__ import annotations

import json
from pathlib import Path

import pytest

from underwater_tracking.runtime.run_catalog import RunCatalog, RunNotFoundError

from tests.api.test_frame_contracts import _full_frame


def _write_run(root: Path, run_id: str, frame_id: int, sim_time_s: int) -> Path:
    path = root / run_id
    path.mkdir(parents=True)
    frame = _full_frame().model_copy(update={"frame_id": frame_id, "sim_time_s": sim_time_s})
    (path / "operational_frames.jsonl").write_text(
        frame.model_dump_json() + "\n", encoding="utf-8"
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "scenario_id": "scenario-a",
                "target_count": 1,
                "seed": frame_id,
                "sim_time_s": sim_time_s,
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_catalog_orders_runs_and_replays_only_selected_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_run(output_root, "serve-b", 2, 20)
    _write_run(output_root, "serve-a", 1, 10)
    (output_root / "run-not-serve").mkdir()

    catalog = RunCatalog(output_root)

    assert [item.run_id for item in catalog.list_runs()] == ["serve-a", "serve-b"]
    assert [frame.frame_id for frame in catalog.replay("serve-a").range()] == [1]
    assert catalog.get("serve-b").sim_time_s == 20


def test_catalog_summary_reads_count_and_last_without_full_range(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "outputs"
    _write_run(output_root, "serve-a", 1, 10)
    original_range = __import__("underwater_tracking.api.replay", fromlist=["ReplayService"]).ReplayService.range

    def fail_full_range(self, *args, **kwargs):
        if kwargs.get("limit") is None and not args:
            raise AssertionError("catalog loaded the full replay into memory")
        return original_range(self, *args, **kwargs)

    monkeypatch.setattr(
        "underwater_tracking.api.replay.ReplayService.range",
        fail_full_range,
    )

    summary = RunCatalog(output_root).get("serve-a")

    assert summary.frame_count == 1


def test_catalog_rejects_unknown_and_traversal_ids(tmp_path: Path) -> None:
    catalog = RunCatalog(tmp_path / "outputs")

    with pytest.raises(RunNotFoundError):
        catalog.get("serve-missing")
    with pytest.raises(RunNotFoundError):
        catalog.get("../serve-missing")
