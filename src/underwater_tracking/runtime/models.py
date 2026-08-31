"""Public request and summary models shared by runtime services and API."""

from __future__ import annotations

from pathlib import Path
from enum import StrEnum

from pydantic import BaseModel, Field

from underwater_tracking.domain.models import StrictModel


class RunRequest(BaseModel):
    """Parameters that create a new live simulation run."""

    target_count: int = Field(ge=1)
    seed: int | None = Field(default=None, ge=0)


class RunPhase(StrEnum):
    CREATED = "created"
    BOOTSTRAP_PLANNING = "bootstrap_planning"
    AWAITING_RETRY = "awaiting_retry"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


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
    phase: RunPhase = RunPhase.RUNNING


class ShutdownReport(StrictModel):
    """Bounded shutdown outcome for the resources owned by one live run."""

    completed: bool
    remaining_resources: tuple[str, ...] = ()
