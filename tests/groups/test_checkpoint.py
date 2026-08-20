from langgraph.checkpoint.memory import InMemorySaver

from underwater_tracking.groups.checkpoint import BoundedInMemorySaver
from underwater_tracking.groups.nodes import apply_plan_command
from underwater_tracking.groups.state import GroupState, PlanCommand


def test_bounded_saver_prunes_checkpoints_writes_and_blobs() -> None:
    saver = BoundedInMemorySaver(max_checkpoints=2)
    graph = saver

    # The graph saver receives this shape from LangGraph. Keeping the test at
    # the saver boundary makes the retention contract independent of graph
    # node details.
    assert isinstance(graph, InMemorySaver)
    thread_id = "scenario:target"
    namespace = ""
    checkpoint_ids = []

    for index in range(5):
        checkpoint_id = f"checkpoint-{index}"
        checkpoint_ids.append(checkpoint_id)
        saver.storage[thread_id][namespace][checkpoint_id] = (
            saver.serde.dumps_typed(
                {
                    "v": 1,
                    "id": checkpoint_id,
                    "ts": f"2026-08-20T00:00:0{index}Z",
                    "channel_values": {},
                    "channel_versions": {"state": f"version-{index}"},
                    "versions_seen": {},
                    "updated_channels": None,
                }
            ),
            saver.serde.dumps_typed({"source": "test", "step": index}),
            checkpoint_ids[index - 1] if index else None,
        )
        saver.writes[(thread_id, namespace, checkpoint_id)][("task", 0)] = (
            "task",
            "state",
            saver.serde.dumps_typed(index),
            "",
        )
        saver.blobs[(thread_id, namespace, "state", f"version-{index}")] = (
            "json",
            saver.serde.dumps_typed(index)[1],
        )

    saver._prune_thread(thread_id)

    retained = saver.storage[thread_id][namespace]
    assert tuple(retained) == ("checkpoint-3", "checkpoint-4")
    assert set(saver.writes) == {
        (thread_id, namespace, "checkpoint-3"),
        (thread_id, namespace, "checkpoint-4"),
    }
    assert set(saver.blobs) == {
        (thread_id, namespace, "state", "version-3"),
        (thread_id, namespace, "state", "version-4"),
    }


def test_bounded_saver_rejects_invalid_limit() -> None:
    try:
        BoundedInMemorySaver(max_checkpoints=0)
    except ValueError as exc:
        assert "max_checkpoints" in str(exc)
    else:
        raise AssertionError("invalid checkpoint limit was accepted")


def test_group_emitted_events_are_bounded_with_stable_ids() -> None:
    state = GroupState.initial(
        "S1",
        "G1",
        "T1",
        ("U1", "U2"),
        coarse_prior=(0.0, 0.0),
        member_positions={"U1": (0.0, 0.0), "U2": (1.0, 0.0), "U3": (2.0, 0.0)},
        event_history_limit=2,
    )

    for index in range(4):
        desired = ("U1", "U2", "U3") if index % 2 == 0 else ("U1", "U2")
        result = apply_plan_command(
            state.model_copy(
                update={
                    "pending_command": PlanCommand(
                        command_id=f"command-{index}",
                        scenario_id="S1",
                        target_id="T1",
                        sim_time_s=index,
                        plan_revision=index + 1,
                        desired_member_ids=desired,
                        member_positions=state.member_positions,
                    )
                }
            )
        )
        state = state.model_copy(update=result)

    assert len(state.emitted_events) == 2
    assert len({event.event_id for event in state.emitted_events}) == 2
