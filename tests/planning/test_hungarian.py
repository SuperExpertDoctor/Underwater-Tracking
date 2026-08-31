from underwater_tracking.planning.carrier_tasks import CarrierServiceTask
from underwater_tracking.planning.hungarian import (
    HungarianMatcher,
    VirtualServiceSlot,
)


def task(
    task_id: str,
    *,
    point: tuple[float, float],
    required_uuv_count: int,
    carrier_id: str = "carrier_01",
) -> CarrierServiceTask:
    return CarrierServiceTask(
        carrier_id=carrier_id,
        task_id=task_id,
        candidate_id=task_id,
        task_type="deploy",
        point=point,
        required_uuv_count=required_uuv_count,
        entry_s=0,
        exit_s=100,
    )


def slot(
    slot_id: str,
    *,
    carrier_id: str,
    point: tuple[float, float],
    ready_uuv_count: int,
) -> VirtualServiceSlot:
    return VirtualServiceSlot(
        slot_id=slot_id,
        carrier_id=carrier_id,
        current_xy=point,
        home_xy=(0.0, 0.0),
        ready_uuv_count=ready_uuv_count,
    )


def test_hungarian_prefers_low_incremental_cost_and_is_deterministic() -> None:
    tasks = (
        task("A", point=(1.0, 0.0), required_uuv_count=1),
        task("B", carrier_id="carrier_02", point=(9.0, 0.0), required_uuv_count=1),
    )
    slots = (
        slot("carrier_01.slot_1", carrier_id="carrier_01", point=(0.0, 0.0), ready_uuv_count=1),
        slot("carrier_02.slot_1", carrier_id="carrier_02", point=(10.0, 0.0), ready_uuv_count=1),
    )

    matcher = HungarianMatcher()
    first = matcher.match(tasks, slots)
    second = matcher.match(tasks, slots)

    assert first == second
    assert tuple((assignment.task_id, assignment.slot_id) for assignment in first.assignments) == (
        ("A", "carrier_01.slot_1"),
        ("B", "carrier_02.slot_1"),
    )
    assert first.assignments[0].cost > 2.0
    assert first.assignments[1].cost > 10.0


def test_hungarian_rejects_impossible_capacity_and_falls_back_to_other_carrier() -> None:
    tasks = (
        task("A", carrier_id="carrier_02", point=(1.0, 0.0), required_uuv_count=2),
    )
    slots = (
        slot("carrier_01.slot_1", carrier_id="carrier_01", point=(1.0, 0.0), ready_uuv_count=1),
        slot("carrier_02.slot_1", carrier_id="carrier_02", point=(5.0, 0.0), ready_uuv_count=2),
    )

    result = HungarianMatcher().match(tasks, slots)

    assert result.assignments[0].slot_id == "carrier_02.slot_1"
    assert result.unassigned_task_ids == ()


def test_hungarian_reports_unmatched_task_when_no_slot_can_return_home() -> None:
    task_to_block = task("blocked", point=(3.0, 3.0), required_uuv_count=1)
    slot_to_block = slot(
        "carrier_01.slot_1",
        carrier_id="carrier_01",
        point=(0.0, 0.0),
        ready_uuv_count=1,
    )

    result = HungarianMatcher(
        forbidden_regions=((0.0, 6.0, 2.0, 4.0),),
        map_bounds=(0.0, 6.0, 0.0, 6.0),
    ).match((task_to_block,), (slot_to_block,))

    assert result.assignments == ()
    assert result.unassigned_task_ids == ("blocked",)


def test_hungarian_rejects_a_slot_owned_by_another_carrier() -> None:
    result = HungarianMatcher().match(
        (task("A", point=(1.0, 0.0), required_uuv_count=1).model_copy(
            update={"carrier_id": "carrier_02"}
        ),),
        (slot(
            "carrier_01.slot_1",
            carrier_id="carrier_01",
            point=(0.0, 0.0),
            ready_uuv_count=1,
        ),),
    )

    assert result.assignments == ()
    assert result.unassigned_task_ids == ("A",)


def test_hungarian_rejects_a_time_window_that_cannot_be_reached() -> None:
    result = HungarianMatcher().match(
        (task("A", point=(10.0, 0.0), required_uuv_count=1).model_copy(
            update={"exit_s": 5}
        ),),
        (slot(
            "carrier_01.slot_1",
            carrier_id="carrier_01",
            point=(0.0, 0.0),
            ready_uuv_count=1,
        ).model_copy(update={"speed_mps": 1.0}),),
    )

    assert result.assignments == ()
    assert result.unassigned_task_ids == ("A",)


def test_hungarian_preserves_future_ready_reserve() -> None:
    result = HungarianMatcher().match(
        (task("A", point=(1.0, 0.0), required_uuv_count=2),),
        (slot(
            "carrier_01.slot_1",
            carrier_id="carrier_01",
            point=(0.0, 0.0),
            ready_uuv_count=2,
        ).model_copy(update={"minimum_ready_uuv_count": 1}),),
    )

    assert result.assignments == ()
    assert result.unassigned_task_ids == ("A",)
