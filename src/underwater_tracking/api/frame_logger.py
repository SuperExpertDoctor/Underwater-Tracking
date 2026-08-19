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
try:
    from typing import Self
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    from typing_extensions import Self

from underwater_tracking.api.frame_builder import operational_frame_json
from underwater_tracking.domain import OperationalFrame


class FrameLogger:
    """Append-only JSONL writer of validated operational frames."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        # The handle lives for the logger's lifetime (append + flush per
        # write), so it is not a context-managed local open.
        self._handle = open(self.path, "a", encoding="utf-8")  # noqa: SIM115

    def append(self, frame: OperationalFrame) -> None:
        """Append one validated frame as a canonical JSON line and flush it."""
        self._handle.write(operational_frame_json(frame) + "\n")
        self._handle.flush()
        self.count += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
