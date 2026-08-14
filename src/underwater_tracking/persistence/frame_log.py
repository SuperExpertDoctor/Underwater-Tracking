# src/underwater_tracking/persistence/frame_log.py
"""Append-only JSONL operational frame log.

``FrameLogger`` serializes one operational frame per line as UTF-8 JSON,
flushes after every write, and retries transient ``PermissionError``
failures up to 20 times with a 50 ms delay before giving up. ``path`` and
``count`` expose the output file and the number of frames written.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

_FILENAME = "frames.jsonl"
_MAX_WRITE_ATTEMPTS = 20
_WRITE_RETRY_DELAY_S = 0.05


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

        A transient ``PermissionError`` (for example a concurrent reader on
        a shared volume) is retried up to 20 times with a 50 ms delay;
        persistent failures propagate to the caller.
        """
        line = json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n"
        for attempt in range(_MAX_WRITE_ATTEMPTS):
            try:
                self._handle.write(line)
                self._handle.flush()
                break
            except PermissionError:
                if attempt == _MAX_WRITE_ATTEMPTS - 1:
                    raise
                time.sleep(_WRITE_RETRY_DELAY_S)
        self.count += 1

    def close(self) -> None:
        self._handle.close()
