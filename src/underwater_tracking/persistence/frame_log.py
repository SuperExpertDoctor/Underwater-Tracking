# src/underwater_tracking/persistence/frame_log.py
"""Append-only JSONL operational frame log.

``FrameLogger`` serializes one operational frame per line as UTF-8 JSON,
flushes after every write, and retries transient ``PermissionError``
failures up to 20 times with a 50 ms delay before giving up. ``path`` and
``count`` expose the output file and the number of frames written.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

_FILENAME = "frames.jsonl"
_MAX_WRITE_ATTEMPTS = 20
_WRITE_RETRY_DELAY_S = 0.05


@dataclass(frozen=True, slots=True)
class FrameLogCheckpoint:
    """A recoverable position in a frame log."""

    offset: int
    count: int


class FrameLogger:
    """Append-only JSONL writer for operational frames."""

    def __init__(self, output_dir: str | Path) -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / _FILENAME
        self.count = 0
        # The handle lives for the logger's lifetime (flush-per-write, retry
        # on PermissionError), so it is not a context-managed local open.
        self._handle = open(self.path, "a", encoding="utf-8")  # noqa: SIM115

    def write(self, frame: dict[str, object]) -> None:
        """Append one frame as a UTF-8 JSON line and flush it.

        The line is written to the buffered handle exactly once. A
        transient ``PermissionError`` surfacing at ``flush`` (the usual
        shape on a shared-volume writer) is retried up to 20 times with a
        50 ms delay; retrying only the flush keeps a recovered write from
        appending a second copy of the already-buffered line. Persistent
        failures propagate to the caller and ``count`` is not incremented.
        """
        line = json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n"
        self._handle.write(line)
        for attempt in range(_MAX_WRITE_ATTEMPTS):
            try:
                self._handle.flush()
                break
            except PermissionError:
                if attempt == _MAX_WRITE_ATTEMPTS - 1:
                    raise
                time.sleep(_WRITE_RETRY_DELAY_S)
        self.count += 1

    def checkpoint(self) -> FrameLogCheckpoint:
        """Record the current durable log position for an engine tick."""
        self._handle.flush()
        self._handle.seek(0, 2)
        return FrameLogCheckpoint(offset=self._handle.tell(), count=self.count)

    def restore(self, checkpoint: FrameLogCheckpoint) -> None:
        """Discard frames appended after ``checkpoint`` and restore its count."""
        self._handle.seek(checkpoint.offset)
        self._handle.truncate()
        self._handle.flush()
        self._handle.seek(0, 2)
        self.count = checkpoint.count

    def close(self) -> None:
        self._handle.close()


class MemoryFrameLogger(FrameLogger):
    """Maintain rollback-compatible frame counts without writing files."""

    def __init__(self) -> None:
        self.count = 0
        self._closed = False

    def write(self, frame: dict[str, object]) -> None:
        del frame
        if self._closed:
            raise ValueError("I/O operation on closed frame logger")
        self.count += 1

    def checkpoint(self) -> FrameLogCheckpoint:
        if self._closed:
            raise ValueError("I/O operation on closed frame logger")
        return FrameLogCheckpoint(offset=self.count, count=self.count)

    def restore(self, checkpoint: FrameLogCheckpoint) -> None:
        if self._closed:
            raise ValueError("I/O operation on closed frame logger")
        self.count = checkpoint.count

    def close(self) -> None:
        self._closed = True
