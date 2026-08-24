from __future__ import annotations

from underwater_tracking.planning.cooperative_auction import (
    AuctionTask,
    AuctionUUV,
    market_allocate,
)


def _uuv(
    uuv_id: str,
    position: tuple[float, float],
    *,
    active: bool = True,
    energy: float = 1.0,
) -> AuctionUUV:
    return AuctionUUV(
        uuv_id=uuv_id,
        position_xy=position,
        speed_mps=4.0,
        energy_fraction=energy,
        active_capable=active,
        carrier_id="carrier-01",
    )


def test_market_auction_assigns_unique_roles_and_protects_active_capability() -> None:
    tasks = (
        AuctionTask(
            task_id="near",
            center_xy=(100.0, 0.0),
            entry_s=0,
            exit_s=300,
            probability=0.8,
            priority=0.5,
            active_slots=1,
            passive_slots=1,
        ),
    )
    allocation = market_allocate(
        tasks,
        (_uuv("active-near", (0.0, 0.0)), _uuv("passive-near", (90.0, 0.0), active=False)),
    )

    award = allocation.awards[0]
    assert award.active_uuv_ids == ("active-near",)
    assert award.passive_uuv_ids == ("passive-near",)
    assert allocation.unfilled_slots == ()
    assert len({*award.active_uuv_ids, *award.passive_uuv_ids}) == 2


def test_market_auction_prefers_continuity_then_distance_and_energy() -> None:
    task = AuctionTask(
        task_id="handoff",
        center_xy=(0.0, 0.0),
        entry_s=100,
        exit_s=500,
        probability=0.7,
        priority=0.2,
        passive_slots=1,
        continuity_uuv_ids=("far-continuous",),
    )
    allocation = market_allocate(
        (task,),
        (
            _uuv("far-continuous", (2_000.0, 0.0), energy=0.95),
            _uuv("near-new", (100.0, 0.0), energy=0.95),
            _uuv("near-low-energy", (90.0, 0.0), energy=0.11),
        ),
    )

    assert allocation.awards[0].passive_uuv_ids == ("far-continuous",)


def test_market_auction_does_not_assign_same_uuv_to_two_tasks() -> None:
    tasks = (
        AuctionTask(
            task_id="current",
            center_xy=(0.0, 0.0),
            entry_s=0,
            exit_s=100,
            probability=0.9,
            priority=1.0,
            active_slots=1,
        ),
        AuctionTask(
            task_id="future",
            center_xy=(200.0, 0.0),
            entry_s=200,
            exit_s=300,
            probability=0.9,
            priority=1.0,
            passive_slots=1,
        ),
    )
    allocation = market_allocate(tasks, (_uuv("U01", (0.0, 0.0)),))

    assigned = [
        uuv_id
        for award in allocation.awards
        for uuv_id in (*award.active_uuv_ids, *award.passive_uuv_ids, *award.reserve_uuv_ids)
    ]
    assert len(assigned) == len(set(assigned)) == 1
    assert len(allocation.unfilled_slots) == 1


def test_market_auction_honors_locked_member() -> None:
    task = AuctionTask(
        task_id="locked",
        center_xy=(0.0, 0.0),
        entry_s=0,
        exit_s=100,
        probability=0.5,
        priority=0.0,
        passive_slots=1,
    )
    allocation = market_allocate(
        (task,),
        (_uuv("U01", (10_000.0, 0.0)), _uuv("U02", (0.0, 0.0))),
        locked_uuv_ids_by_task={"locked": ("U01",)},
    )

    assert allocation.awards[0].passive_uuv_ids == ("U01",)
