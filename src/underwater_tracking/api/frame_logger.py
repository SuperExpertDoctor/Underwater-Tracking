# src/underwater_tracking/api/frame_logger.py
"""Append-only JSONL logger for validated operational frames.

One validated frame per line as canonical model JSON, flushed immediately
after every append so a crash never loses the last frame and a concurrent
replay reader sees each frame as soon as it is written. ``append`` accepts
an already-validated ``OperationalFrame`` and serializes it in one write;
the writer takes a file path directly (the engine's
``persistence.frame_log`` writes raw dicts into a directory-named file and
stays separate by design).
"""
from __future__ import annotations

from pathlib import Path
import json
try:
    from typing import Self
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    from typing_extensions import Self

from underwater_tracking.api.frame_builder import operational_frame_json
from underwater_tracking.domain import OperationalFrame


class FrameLogger:
    """Append-only JSONL writer of validated operational frames."""

    def __init__(self, path: str | Path, *, max_run_bytes: int | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.max_run_bytes = max_run_bytes
        self.log_truncated = False
        self._limit_record_written = False
        # The handle lives for the logger's lifetime (append + flush per
        # write), so it is not a context-managed local open.
        self._handle = open(self.path, "a", encoding="utf-8")  # noqa: SIM115

    def append(self, frame: OperationalFrame) -> None:
        """Append one validated frame as a canonical JSON line and flush it."""
        line = operational_frame_json(frame) + "\n"
        if self.max_run_bytes is not None:
            self._handle.flush()
            current_size = self.path.stat().st_size
            if current_size + len(line.encode("utf-8")) > self.max_run_bytes:
                self.log_truncated = True
                if not self._limit_record_written:
                    marker = json.dumps(
                        {
                            "record_type": "frame_log_limit_reached",
                            "max_run_bytes": self.max_run_bytes,
                            "persisted_bytes": current_size,
                        },
                        separators=(",", ":"),
                    ) + "\n"
                    self._handle.write(marker)
                    self._handle.flush()
                    self._limit_record_written = True
                return
        self._handle.write(line)
        self._handle.flush()
        self.count += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
