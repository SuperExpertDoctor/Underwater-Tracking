from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "monitor_main_battle.py"
_SPEC = importlib.util.spec_from_file_location("monitor_main_battle_ui_probe", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MONITOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MONITOR)


class _Response:
    ok = True
    status = 200

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class _Request:
    def __init__(self, responses: list[_Response]) -> None:
        self._responses = iter(responses)
        self.calls = 0

    def get(self, *_args: object, **_kwargs: object) -> _Response:
        self.calls += 1
        return next(self._responses)


class _Locator:
    def __init__(self, *, plan_version: int = 2, text: str = "120s") -> None:
        self.first = self
        self._plan_version = plan_version
        self._text = text

    def count(self) -> int:
        return 1

    def get_attribute(self, name: str) -> str | None:
        return str(self._plan_version) if name == "data-plan-version" else None

    def text_content(self) -> str:
        return self._text


class _Page:
    def __init__(self, request: _Request) -> None:
        self.request = request
        self._locator = _Locator()

    def evaluate(self, _script: str) -> dict[str, int]:
        return {"width": 1280, "documentWidth": 1280, "bodyWidth": 1280}

    def locator(self, _selector: str) -> _Locator:
        return self._locator

    def wait_for_timeout(self, _milliseconds: int) -> None:
        return None


def test_ui_consistency_refetches_snapshot_when_live_frame_is_stale() -> None:
    request = _Request(
        [
            _Response({"run_phase": "running", "sim_time_s": 0, "plan_version": 1}),
            _Response({"run_phase": "running", "sim_time_s": 120, "plan_version": 2}),
        ]
    )
    page = _Page(request)
    page_errors: list[object] = []
    request_errors: list[object] = []

    _MONITOR._probe_ui_consistency(
        page,
        "http://127.0.0.1:8000",
        page_errors,
        request_errors,
        probe_memory=False,
    )

    assert request.calls == 2
    assert page_errors == []
    assert request_errors == []
