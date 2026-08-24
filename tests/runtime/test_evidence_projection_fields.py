from types import SimpleNamespace

from underwater_tracking.runtime.run_controller import _stored_verification_event_projection


def test_stored_projection_promotes_tracking_chain_payload_fields() -> None:
    projection = _stored_verification_event_projection(
        SimpleNamespace(
            event_id="carrier_dispatch_completed:carrier_02:90",
            event_type="carrier_dispatch_completed",
            target_id="carrier_02",
            sim_time_s=90,
            payload={
                "candidate_id": "target_00:cell:-3:-4",
                "carrier_id": "carrier_02",
                "uuv_ids": ["uuv_00", "uuv_02"],
                "plan_version": 1,
                "sortie_uuv_ids": ["uuv_00", "uuv_02"],
            },
        )
    )

    assert projection["candidate_id"] == "target_00:cell:-3:-4"
    assert projection["carrier_id"] == "carrier_02"
    assert projection["uuv_ids"] == ["uuv_00", "uuv_02"]
    assert projection["plan_version"] == 1
    assert projection["sortie_uuv_ids"] == ["uuv_00", "uuv_02"]
