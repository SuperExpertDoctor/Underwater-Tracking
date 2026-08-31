"""One scenario-scoped serialization boundary for state transitions."""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Iterator, Literal

TransitionKind = Literal["plan", "observation"]


class ScenarioTransitionCoordinator:
    """Serialize physical observation and plan commits with one lock.

    Providers must be called before entering this context.  The lock is held
    only while applying copy-on-write state and the corresponding SQLite
    transaction, so a slow model cannot block API reads or physics.
    """

    def __init__(self, scenario_id: str) -> None:
        if not scenario_id:
            raise ValueError("scenario_id is required")
        self.scenario_id = scenario_id
        self._lock = RLock()
        self._active_kind: TransitionKind | None = None

    @property
    def lock(self) -> RLock:
        return self._lock

    @property
    def active_kind(self) -> TransitionKind | None:
        return self._active_kind

    @contextmanager
    def transition(self, kind: TransitionKind) -> Iterator[None]:
        with self._lock:
            if self._active_kind is not None:
                raise RuntimeError("scenario transition coordinator is already active")
            self._active_kind = kind
            try:
                yield
            finally:
                self._active_kind = None
