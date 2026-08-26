"""Runtime lifecycle services for the command center.

The public names are loaded lazily.  ``SimulationEngine`` imports the
mission-controller submodule, and eagerly importing ``run_controller`` here
would make that otherwise independent import cycle back through
``SimulationEngine``.
"""

from typing import Any

__all__ = ["ProcessSupervisor", "RunController", "RunRequest", "RunSummary"]


def __getattr__(name: str) -> Any:
    if name == "RunController":
        from underwater_tracking.runtime.run_controller import RunController

        return RunController
    if name == "ProcessSupervisor":
        from underwater_tracking.runtime.process_supervisor import ProcessSupervisor

        return ProcessSupervisor
    if name in {"RunRequest", "RunSummary"}:
        from underwater_tracking.runtime.models import RunRequest, RunSummary

        return {"RunRequest": RunRequest, "RunSummary": RunSummary}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
