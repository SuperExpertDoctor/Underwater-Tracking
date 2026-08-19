from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.optimize import PlanningConfig
from underwater_tracking.config.models import (
    DEFAULT_QUALITY_CRITICAL,
    DEFAULT_QUALITY_RELEASE,
    DEFAULT_QUALITY_WARNING,
    DEFAULT_RELEASE_HOLD_S,
    TrackingConfig,
)
from underwater_tracking.planning.allocation import AllocationInput


def test_quality_defaults_are_defined_by_tracking_config_contract() -> None:
    config = TrackingConfig()
    assert config.quality_warning == DEFAULT_QUALITY_WARNING
    assert config.quality_critical == DEFAULT_QUALITY_CRITICAL
    assert config.quality_release == DEFAULT_QUALITY_RELEASE
    assert config.release_hold_s == DEFAULT_RELEASE_HOLD_S


def test_offline_planning_defaults_follow_config_defaults() -> None:
    planning = PlanningConfig()
    allocation = AllocationInput(
        uuv_ids=("U1",),
        target_ids=("T1",),
        quality_by_target={"T1": 0.8},
    )
    assert planning.quality_warning == DEFAULT_QUALITY_WARNING
    assert planning.quality_release == DEFAULT_QUALITY_RELEASE
    assert planning.release_hold_s == DEFAULT_RELEASE_HOLD_S
    assert allocation.quality_warning == DEFAULT_QUALITY_WARNING
    assert allocation.quality_release == DEFAULT_QUALITY_RELEASE
    assert allocation.release_hold_s == DEFAULT_RELEASE_HOLD_S


def test_event_monitor_defaults_follow_quality_config() -> None:
    monitor = EventMonitor()
    assert monitor._warning_threshold == DEFAULT_QUALITY_WARNING
    assert monitor._critical_threshold == DEFAULT_QUALITY_CRITICAL
