from __future__ import annotations

from typing import Any

from underwater_tracking.agent.runtime import CarrierRuntime


class _Closable:
    def __init__(self, calls: list[str], name: str) -> None:
        self.calls = calls
        self.name = name

    def close(self) -> None:
        self.calls.append(self.name)


class _Checkpointer:
    def __init__(self, calls: list[str]) -> None:
        self.conn = _Closable(calls, "checkpointer")


def test_runtime_close_is_idempotent_and_closes_runtime_resources_once() -> None:
    calls: list[str] = []
    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._closed = False
    runtime._payload_store = _Closable(calls, "payload")
    runtime._checkpointer = _Checkpointer(calls)
    runtime._dependencies = Any  # type: ignore[assignment]
    runtime._pre_close_hooks = [lambda: calls.append("worker")]

    runtime.close()
    runtime.close()

    assert calls == ["worker", "payload", "checkpointer"]
