# src/underwater_tracking/api/__init__.py
"""Runtime frame adapter, transport hub, and indexed replay service."""

from underwater_tracking.api.hub import OperationalHub, RuntimeDirectiveQueue
from underwater_tracking.api.frame_builder import (
    DEFAULT_MAP_BOUNDS,
    build_operational_frame,
    build_uuv_only_frame,
    operational_frame_json,
    operational_frame_payload,
)
from underwater_tracking.api.frame_logger import FrameLogger
from underwater_tracking.api.evaluation import EvaluationReplayService
from underwater_tracking.api.live import OperationalFramePublisher
from underwater_tracking.api.replay import ReplayIndexError, ReplayService
from underwater_tracking.api.legacy_frame_adapter import read_legacy_frame


def create_app(*args: object, **kwargs: object) -> object:
    """Lazily import the FastAPI app to keep replay/catalog imports acyclic."""
    from underwater_tracking.api.app import create_app as _create_app

    return _create_app(*args, **kwargs)

__all__ = [
    "DEFAULT_MAP_BOUNDS",
    "EvaluationReplayService",
    "FrameLogger",
    "OperationalFramePublisher",
    "OperationalHub",
    "ReplayIndexError",
    "ReplayService",
    "RuntimeDirectiveQueue",
    "build_operational_frame",
    "build_uuv_only_frame",
    "create_app",
    "operational_frame_json",
    "operational_frame_payload",
    "read_legacy_frame",
]
