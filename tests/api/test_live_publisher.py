from __future__ import annotations

from pathlib import Path

from underwater_tracking.api.frame_logger import FrameLogger
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.api.live import OperationalFramePublisher
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.domain.models import CarrierState, SituationSnapshot, UUVState, UUVStatus


class Runtime:
    def active_plan(self):
        return None

    def get_state(self):
        return {"intent_hypotheses": {}, "predictions": {}}


class Ledger:
    def list_decisions(self, scenario_id: str, limit: int = 100):
        return []

    def list_directives(self, scenario_id: str, status: str | None = None):
        return []


class Events:
    def list_events(self, **kwargs):
        return []


def test_publisher_bridges_runtime_state_to_hub_and_operational_replay(tmp_path: Path) -> None:
    hub = OperationalHub()
    log_path = tmp_path / "operational-frames.jsonl"
    logger = FrameLogger(log_path)
    publisher = OperationalFramePublisher(
        runtime=Runtime(), ledger=Ledger(), events=Events(), hub=hub, logger=logger
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=2,
        sim_time_s=60,
        uuvs=(UUVState(
            uuv_id="U1", position_xy=(1.0, 2.0), heading_rad=0.0,
            speed_mps=2.0, energy_fraction=0.9, status=UUVStatus.RETURNING,
            deployment_state="returning",
        ),),
        carrier=CarrierState(
            carrier_id="carrier-01",
            position_xy=(-3000.0, -2995.0),
            heading_rad=1.57,
            speed_mps=1.0,
            status="recovering",
            returning_uuv_ids=("U1",),
        ),
        group_reports=(), pending_events=(),
    )

    frame = publisher.publish(snapshot)

    assert hub.snapshot() == frame
    assert frame.frame_id == 2
    assert frame.uuvs[0].uuv_id == "U1"
    assert frame.carrier is not None
    assert (frame.carrier.position.x, frame.carrier.position.y) == snapshot.carrier.position_xy
    logged_frame = ReplayService(log_path).range()[0]
    assert logged_frame.carrier == frame.carrier
    assert ReplayService(log_path).range() == [frame]
    publisher.close()
