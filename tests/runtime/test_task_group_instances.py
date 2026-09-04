from underwater_tracking.domain.execution_models import TaskGroupLifecycle
from underwater_tracking.planning.region_baseline import build_four_region_baseline
from underwater_tracking.runtime.task_group_instances import (
    AlwaysAvailableTaskGroupFactory,
    RegionReplacementState,
    RegionTransitionQueue,
)

from tests.runtime.test_execution_snapshot_factory import _inputs


def test_factory_creates_unique_reproducible_three_member_deployments() -> None:
    factory = AlwaysAvailableTaskGroupFactory(scenario_id="S1")

    first = factory.create(
        region_id="T1:task:01",
        deployment_revision=1,
        reason="initial_deployment",
        sensor_mode="active",
    )
    replacement = factory.create(
        region_id="T1:task:01",
        deployment_revision=2,
        reason="region_replacement",
        sensor_mode="active",
    )

    assert first.group_instance_id == "S1:T1:task:01:deploy:000001"
    assert first.member_uuv_ids == (
        "S1:T1:task:01:deploy:000001:member:01",
        "S1:T1:task:01:deploy:000001:member:02",
        "S1:T1:task:01:deploy:000001:member:03",
    )
    assert first == factory.create(
        region_id="T1:task:01",
        deployment_revision=1,
        reason="initial_deployment",
        sensor_mode="active",
    )
    assert set(first.member_uuv_ids).isdisjoint(replacement.member_uuv_ids)
    assert first.group_instance_id != replacement.group_instance_id
    assert first.lifecycle is TaskGroupLifecycle.ENTERING


def test_factory_accepts_the_authoritative_region_object() -> None:
    _, target_track, _, baseline, _, _ = _inputs()
    region = baseline.regions[0]
    factory = AlwaysAvailableTaskGroupFactory(scenario_id="S1")

    group = factory.create(
        region,
        deployment_revision=3,
        reason="initial_deployment",
        sensor_mode="active",
    )

    assert group.target_id == target_track.target_id
    assert group.region_id == region.region_id
    assert group.deployment_revision == 3


def test_region_transition_queue_keeps_only_latest_pending_revision_per_slot() -> None:
    situation, target_track, accepted, _, _, _ = _inputs()
    queue = RegionTransitionQueue()

    for revision in (2, 3, 4):
        baseline = build_four_region_baseline(
            accepted,
            target_id=target_track.target_id,
            execution_revision=revision,
            origin_sim_time_s=float(situation.sim_time_s),
            map_bounds_xy=situation.map_bounds_xy,
        )
        queue.offer(baseline.regions[0])

    latest = queue.pop_latest(0)

    assert latest is not None
    assert latest.execution_revision == 4
    assert queue.pop_latest(0) is None


def test_region_transition_queue_does_not_mutate_in_flight_transition() -> None:
    situation, target_track, accepted, _, _, _ = _inputs()
    queue = RegionTransitionQueue()
    baselines = tuple(
        build_four_region_baseline(
            accepted,
            target_id=target_track.target_id,
            execution_revision=revision,
            origin_sim_time_s=float(situation.sim_time_s),
            map_bounds_xy=situation.map_bounds_xy,
        )
        for revision in (2, 3, 4)
    )
    queue.offer(baselines[0].regions[0])
    active_transition = queue.pop_latest(0)

    queue.offer(baselines[1].regions[0])
    queue.offer(baselines[2].regions[0])

    assert active_transition is not None
    assert active_transition.execution_revision == 2
    assert queue.pop_latest(0).execution_revision == 4


def test_region_transition_queue_discards_stale_revision_offers() -> None:
    situation, target_track, accepted, _, _, _ = _inputs()
    queue = RegionTransitionQueue()
    latest = build_four_region_baseline(
        accepted,
        target_id=target_track.target_id,
        execution_revision=4,
        origin_sim_time_s=float(situation.sim_time_s),
        map_bounds_xy=situation.map_bounds_xy,
    ).regions[0]
    stale = build_four_region_baseline(
        accepted,
        target_id=target_track.target_id,
        execution_revision=3,
        origin_sim_time_s=float(situation.sim_time_s),
        map_bounds_xy=situation.map_bounds_xy,
    ).regions[0]

    queue.offer(latest)
    queue.offer(stale)

    assert queue.pop_latest(0) == latest


def test_region_replacement_state_records_one_bounded_slot_transition() -> None:
    _, _, _, baseline, _, _ = _inputs()
    region = baseline.regions[0]
    state = RegionReplacementState(
        region_id=region.region_id,
        source_geometry_revision=1,
        target_geometry_revision=2,
        outgoing_group_id="group-old",
        incoming_group_id="group-new",
        latest_pending_region=region,
    )

    assert state.target_geometry_revision == 2
    assert state.latest_pending_region == region
