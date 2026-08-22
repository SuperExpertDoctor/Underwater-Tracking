# src/underwater_tracking/api/replay.py
"""Indexed read-only replay over a JSONL operational frame log.

At startup ``ReplayService`` scans the log once and builds an in-memory
``(sim_time_s, byte_offset)`` index while validating every line as an
``OperationalFrame``; a corrupt line (invalid JSON or a frame that fails
validation) raises ``ReplayIndexError`` carrying the line number — corrupt
lines are never silently skipped. ``range`` refreshes that index when a live
simulation appends frames, then serves the frames whose simulation time falls
inside ``[start_s, end_s]`` (inclusive ends; ``None`` end means unbounded) in
chronological order, re-validating each frame at read time so a concurrently
rewritten log cannot yield stale or partial lines.
"""
from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pydantic import ValidationError

from underwater_tracking.api.legacy_frame_adapter import read_legacy_frame
from underwater_tracking.domain import OperationalFrame

_DEFAULT_PAGE_SIZE = 1_000
_MAX_PAGE_SIZE = 10_000


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
        self._times: list[int] = []
        self._file_signature: tuple[int, int] = (-1, -1)
        self._build_index()

    def _build_index(self) -> None:
        """Scan the log once, validating every line and recording offsets."""
        try:
            handle = open(self.path, "rb")  # noqa: SIM115 (error-handled open)
        except FileNotFoundError:
            self._entries = []
            self._by_time = []
            self._times = []
            self._file_signature = (0, 0)
            return
        with handle:
            entries: list[_IndexEntry] = []
            byte_offset = 0
            for line_number, raw in enumerate(handle, start=1):
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict) and payload.get("record_type") == "frame_log_limit_reached":
                        byte_offset += len(raw)
                        continue
                    frame = _read_frame(raw)
                except (ValidationError, TypeError, ValueError, json.JSONDecodeError) as exc:
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
        self._times = [entries[index].sim_time_s for index in self._by_time]
        stat = self.path.stat()
        self._file_signature = (stat.st_size, stat.st_mtime_ns)

    def _refresh_if_changed(self) -> None:
        """Refresh the index when a live simulation appends new frames."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            if self._file_signature != (0, 0):
                self._build_index()
            return
        signature = (stat.st_size, stat.st_mtime_ns)
        if signature != self._file_signature:
            self._build_index()

    def range(
        self,
        start_s: float = 0.0,
        end_s: float | None = None,
        *,
        offset: int = 0,
        limit: int | None = _DEFAULT_PAGE_SIZE,
    ) -> list[OperationalFrame]:
        """Frames with ``start_s <= sim_time_s <= end_s``, in chronological order.

        Ends are inclusive so a range that ends exactly on a frame's time
        includes it; ``end_s=None`` is unbounded.
        """
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive when provided")
        if limit is not None and limit > _MAX_PAGE_SIZE:
            raise ValueError(f"limit must not exceed {_MAX_PAGE_SIZE}")
        left, right = self._matching_bounds(start_s, end_s)
        start = min(left + offset, right)
        stop = right if limit is None else min(start + limit, right)
        selected = self._by_time[start:stop]
        if not selected:
            return []
        frames: list[OperationalFrame] = []
        with open(self.path, "rb") as handle:
            for entry_index in selected:
                frames.append(self._read_entry(handle, entry_index))
        return frames

    def count(self, start_s: float = 0.0, end_s: float | None = None) -> int:
        """Return the number of indexed frames in a time range."""
        left, right = self._matching_bounds(start_s, end_s)
        return right - left

    def last(self) -> OperationalFrame | None:
        """Read the chronologically latest frame without loading the full log."""
        self._refresh_if_changed()
        if not self._by_time:
            return None
        with open(self.path, "rb") as handle:
            return self._read_entry(handle, self._by_time[-1])

    def _matching_bounds(self, start_s: float, end_s: float | None) -> tuple[int, int]:
        self._refresh_if_changed()
        left = bisect_left(self._times, start_s)
        right = bisect_right(self._times, end_s) if end_s is not None else len(self._times)
        return left, right

    def _read_entry(self, handle: BinaryIO, entry_index: int) -> OperationalFrame:
        entry = self._entries[entry_index]
        handle.seek(entry.byte_offset)
        raw = handle.readline()
        try:
            return _read_frame(raw)
        except (ValidationError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReplayIndexError(
                entry_index + 1, f"corrupt frame line: {exc}"
            ) from exc


def _read_frame(raw: bytes) -> OperationalFrame:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("operational frame JSON must be an object")
    return read_legacy_frame(payload)
