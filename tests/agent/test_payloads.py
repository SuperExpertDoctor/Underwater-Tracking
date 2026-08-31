from tests.agent.test_repositories import build_plan

from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.persistence.payloads import RuntimePayloadStore


def test_runtime_payload_store_survives_reopen_and_bounds_database(tmp_path) -> None:
    path = tmp_path / "runtime.db"
    store = RuntimePayloadStore(
        path,
        owner="S1",
        cache_limit=1,
        database_limit=2,
    )
    store["candidate:1"] = build_plan(plan_id="P1", revision=1)
    store["candidate:2"] = build_plan(plan_id="P2", revision=2)
    store["candidate:3"] = build_plan(plan_id="P3", revision=3)

    assert len(store) == 2
    assert len(store._cache) <= 1
    store.close()

    reopened = RuntimePayloadStore(
        path,
        owner="S1",
        cache_limit=1,
        database_limit=2,
    )
    assert reopened["candidate:2"].plan_id == "P2"
    assert reopened["candidate:3"].plan_id == "P3"
    try:
        reopened["candidate:1"]
    except KeyError:
        pass
    else:
        raise AssertionError("old payload was not pruned")
    reopened.close()


def test_runtime_payload_store_isolated_by_owner(tmp_path) -> None:
    path = tmp_path / "runtime.db"
    first = RuntimePayloadStore(path, owner="S1", database_limit=4)
    second = RuntimePayloadStore(path, owner="S2", database_limit=4)
    first["same-ref"] = build_plan(plan_id="P1", revision=1)
    second["same-ref"] = build_plan(plan_id="P2", revision=2)

    assert first["same-ref"].plan_id == "P1"
    assert second["same-ref"].plan_id == "P2"
    first.close()
    second.close()


def test_runtime_payload_store_restores_planning_snapshot_type(tmp_path) -> None:
    from tests.agent.test_commit import _snapshot

    path = tmp_path / "runtime.db"
    store = RuntimePayloadStore(path, owner="S1")
    store["S1:snapshot:4"] = PlanningSnapshot(_snapshot().situation, None, ())
    store.close()

    reopened = RuntimePayloadStore(path, owner="S1")
    restored = reopened["S1:snapshot:4"]

    assert isinstance(restored, PlanningSnapshot)
    assert restored.situation.snapshot_revision == 4
    reopened.close()
