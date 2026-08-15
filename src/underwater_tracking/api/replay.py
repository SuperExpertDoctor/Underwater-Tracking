# src/underwater_tracking/api/replay.py
"""Indexed read-only replay over a JSONL operational frame log.

At startup ``ReplayService`` scans the log once and builds an in-memory
``(sim_time_s, byte_offset)`` index while validating every line as an
``OperationalFrame``; a corrupt line (invalid JSON or a frame that fails
validation) raises ``ReplayIndexError`` carrying the line number — corrupt
lines are never silently skipped. ``range`` serves the frames whose
simulation time falls inside ``[start_s, end_s]`` (inclusive ends; ``None``
end means unbounded) in chronological order, re-validating each frame at
read time so a concurrently rewritten log cannot yield stale or partial
lines.
"""
from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from underwater_tracking.domain import OperationalFrame


class ReplayIndexError(ValueError):
    """A log line could not be parsed or validated; carries its line number."""

    def __init__(self, line_number: int, message: str) -> None:
        self.line_number = line_number
        super().__init__(f"line {line_number}: {message}")


@dataclass(frozen=True)
class _IndexEntry:
    sim_time_s: int
    byte_offset: int


class ReplayService:
    """Read-only replay over one JSONL operational frame log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: list[_IndexEntry] = []
        # Permutation of entry indices ordered by (sim_time_s, byte_offset).
        self._by_time: list[int] = []
        self._build_index()

    def _build_index(self) -> None:
        """Scan the log once, validating every line and recording offsets."""
        try:
            handle = open(self.path, "rb")  # noqa: SIM115 (error-handled open)
        except FileNotFoundError:
            return
        with handle:
            entries: list[_IndexEntry] = []
            byte_offset = 0
            for line_number, raw in enumerate(handle, start=1):
                try:
                    frame = OperationalFrame.model_validate_json(raw)
                except (ValidationError, json.JSONDecodeError) as exc:
                    raise ReplayIndexError(
                        line_number, f"corrupt frame line: {exc}"
                    ) from exc
                entries.append(_IndexEntry(frame.sim_time_s, byte_offset))
                byte_offset += len(raw)
        self._entries = entries
        self._by_time = sorted(
            range(len(entries)),
            key=lambda index: (entries[index].sim_time_s, entries[index].byte_offset),
        )

    def range(
        self, start_s: float = 0.0, end_s: float | None = None
    ) -> list[OperationalFrame]:
        """Frames with ``start_s <= sim_time_s <= end_s``, in chronological order.

        Ends are inclusive so a range that ends exactly on a frame's time
        includes it; ``end_s=None`` is unbounded.
        """
        times = [float(self._entries[index].sim_time_s) for index in self._by_time]
        left = bisect_left(times, start_s)
        right = bisect_right(times, end_s) if end_s is not None else len(times)
        if left >= right:
            return []
        frames: list[OperationalFrame] = []
        with open(self.path, "rb") as handle:
            for entry_index in self._by_time[left:right]:
                entry = self._entries[entry_index]
                handle.seek(entry.byte_offset)
                raw = handle.readline()
                try:
                    frame = OperationalFrame.model_validate_json(raw)
                except (ValidationError, json.JSONDecodeError) as exc:
                    raise ReplayIndexError(
                        entry_index + 1, f"corrupt frame line: {exc}"
                    ) from exc
                frames.append(frame)
        return frames
