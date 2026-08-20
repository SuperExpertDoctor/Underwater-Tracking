"""Discover immutable live-run artifacts and create isolated replay readers."""

from __future__ import annotations

import json
import re
from pathlib import Path

from underwater_tracking.api.replay import ReplayIndexError, ReplayService
from underwater_tracking.runtime.models import RunSummary

_RUN_ID = re.compile(r"serve-[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class RunNotFoundError(LookupError):
    """The requested run is not a direct, catalogued output directory."""


class RunCatalog:
    """Index ``serve-*`` output directories without exposing their internals."""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)

    def list_runs(self) -> tuple[RunSummary, ...]:
        """Return valid run summaries in deterministic run-id order."""
        if not self.output_root.is_dir():
            return ()
        summaries: list[RunSummary] = []
        for path in sorted(self.output_root.iterdir(), key=lambda item: item.name):
            if not path.is_dir() or _RUN_ID.fullmatch(path.name) is None:
                continue
            try:
                summaries.append(self._summary(path))
            except (OSError, ReplayIndexError, ValueError, json.JSONDecodeError):
                # A partial or corrupt output is not a selectable replay.
                continue
        return tuple(summaries)

    def get(self, run_id: str) -> RunSummary:
        """Resolve one run id without allowing path traversal."""
        path = self._path_for(run_id)
        if not path.is_dir():
            raise RunNotFoundError(run_id)
        try:
            return self._summary(path)
        except (OSError, ReplayIndexError, ValueError, json.JSONDecodeError) as exc:
            raise RunNotFoundError(run_id) from exc

    def replay(self, run_id: str) -> ReplayService:
        """Create the validator/indexer for exactly one catalogued run."""
        self.get(run_id)
        return ReplayService(self._path_for(run_id) / "operational_frames.jsonl")

    def _path_for(self, run_id: str) -> Path:
        if _RUN_ID.fullmatch(run_id) is None:
            raise RunNotFoundError(run_id)
        root = self.output_root.resolve()
        path = (root / run_id).resolve()
        if path.parent != root or path.name != run_id:
            raise RunNotFoundError(run_id)
        return path

    def _summary(self, path: Path) -> RunSummary:
        manifest: dict[str, object] = {}
        manifest_path = path / "manifest.json"
        if manifest_path.is_file():
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("run manifest must be an object")
            manifest = raw

        replay = ReplayService(path / "operational_frames.jsonl")
        last = replay.last()
        target_count = manifest.get("target_count")
        if not isinstance(target_count, int):
            target_count = len(last.target_estimates) if last is not None else 0
        seed = manifest.get("seed", 0)
        if not isinstance(seed, int):
            seed = 0
        scenario_id = manifest.get("scenario_id")
        if not isinstance(scenario_id, str):
            scenario_id = ""
        status = manifest.get("status")
        if not isinstance(status, str):
            status = "completed" if manifest else "running"
        sim_time_s = manifest.get("sim_time_s", last.sim_time_s if last else 0)
        if not isinstance(sim_time_s, int):
            sim_time_s = int(sim_time_s)
        return RunSummary(
            run_id=path.name,
            scenario_id=scenario_id,
            target_count=target_count,
            seed=seed,
            sim_time_s=sim_time_s,
            frame_count=replay.count(),
            status=status,
            path=path,
        )
