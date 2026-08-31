from __future__ import annotations

import asyncio
import sys
from threading import Thread
from types import SimpleNamespace

from underwater_tracking.cli import _run_api_server


def test_api_server_started_from_main_wrapper_thread_does_not_register_signals(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class FakeConfig:
        def __init__(self, *_: object, **__: object) -> None:
            pass

    class FakeServer:
        def __init__(self, _: FakeConfig) -> None:
            self.should_exit = False

        def run(self) -> None:
            asyncio.run(self.serve())

        async def _serve(self, sockets: object = None) -> None:
            del sockets
            calls.append("served")

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(Config=FakeConfig, Server=FakeServer))
    errors: list[BaseException] = []

    def run() -> None:
        try:
            _run_api_server(app=object(), host="127.0.0.1", port=0)
        except BaseException as exc:  # the assertion runs on the parent thread
            errors.append(exc)

    thread = Thread(target=run)
    thread.start()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert not errors
    assert calls == ["served"]
