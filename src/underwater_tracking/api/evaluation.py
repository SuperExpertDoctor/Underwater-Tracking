"""Explicitly separated evaluation-frame transport."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from underwater_tracking.domain.ui_models import EvaluationFrame


class EvaluationPort(Protocol):
    def range(
        self, start_s: float = 0.0, end_s: float | None = None
    ) -> Sequence[EvaluationFrame]: ...


class EvaluationReplayService:
    """Read truth-side JSONL only when explicitly mounted by the app."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def range(self, start_s: float = 0.0, end_s: float | None = None) -> list[EvaluationFrame]:
        if not self.path.is_file():
            return []
        frames: list[EvaluationFrame] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    frame = EvaluationFrame.model_validate_json(line)
                except (ValueError, ValidationError) as exc:
                    raise ValueError(f"line {line_number}: invalid evaluation frame") from exc
                if frame.sim_time_s < start_s or (end_s is not None and frame.sim_time_s > end_s):
                    continue
                frames.append(frame)
        return frames
