# tests/planning/test_reservations.py
"""Reservation registry for human-assigned UUVs (spec 17.2, R4).

UUVs the operator has explicitly assigned to a target are reserved: the
allocator must never assign them elsewhere and the verification protocol
must never pick them as pingers. One registry is owned by the
CarrierRuntime and shared with the engine through
``SimulationEngine.set_reservations``; every consumer reads only the
immutable ``reserved_uuvs()`` projection.
"""

import pytest

from underwater_tracking.planning.reservations import ReservationRegistry


def test_reserve_release_round_trip() -> None:
    registry = ReservationRegistry()
    registry.reserve(("uuv_01", "uuv_02"), "T2")
    assert registry.reserved_uuvs() == frozenset({"uuv_01", "uuv_02"})
    assert registry.reserved_for("T2") == frozenset({"uuv_01", "uuv_02"})
    assert registry.is_reserved("uuv_01") is True
    assert registry.is_reserved("uuv_03") is False
    registry.release(("uuv_01",))
    assert registry.reserved_uuvs() == frozenset({"uuv_02"})
    assert registry.is_reserved("uuv_01") is False
    registry.release(("uuv_02",))
    assert registry.reserved_uuvs() == frozenset()
    assert registry.reserved_for("T2") == frozenset()


def test_reserving_for_another_target_is_rejected() -> None:
    registry = ReservationRegistry()
    registry.reserve(("uuv_01",), "T2")
    with pytest.raises(ValueError):
        registry.reserve(("uuv_01",), "T3")


def test_reserving_the_same_target_again_is_idempotent() -> None:
    registry = ReservationRegistry()
    registry.reserve(("uuv_01",), "T2")
    registry.reserve(("uuv_01",), "T2")
    assert registry.reserved_for("T2") == frozenset({"uuv_01"})
