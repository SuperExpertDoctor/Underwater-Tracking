"""Public request and summary models shared by runtime services and API."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Parameters that create a new live simulation run."""

    target_count: int = Field(ge=1)
    seed: int | None = Field(default=None, ge=0)


class RunSummary(BaseModel):
    """Public, truth-safe description of one live or completed run."""

    run_id: str
    scenario_id: str
    target_count: int
    seed: int
    sim_time_s: int
    frame_count: int
    status: str
    path: Path
    effective_demo_speed: float | None = None
