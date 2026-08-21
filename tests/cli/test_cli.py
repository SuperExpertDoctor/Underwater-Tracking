from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import underwater_tracking.cli as cli
from underwater_tracking.config.loader import load_app_config


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/scenario/default.yaml"


def test_serve_leaves_configured_demo_speed_in_control_when_speed_is_omitted(
    monkeypatch,
) -> None:
    config = load_app_config(CONFIG_PATH)
    captured: dict[str, object] = {}

    class Controller:
        def __init__(self, _config, *, steps, speed) -> None:
            captured["steps"] = steps
            captured["speed"] = speed

        def start_run(self, _target_count, *, seed) -> None:
            captured["seed"] = seed

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "RunController", Controller)
    monkeypatch.setattr(cli, "RunCatalog", lambda _root: object())
    monkeypatch.setattr(cli, "create_app", lambda **_kwargs: object())
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *_args, **_kwargs: None))

    result = cli._serve(
        config,
        Namespace(
            steps=0,
            speed=None,
            seed=42,
            host="127.0.0.1",
            port=8000,
        ),
    )

    assert result == 0
    assert captured == {"steps": 0, "speed": None, "seed": 42, "closed": True}


def test_serve_passes_explicit_speed_as_an_override(monkeypatch) -> None:
    config = load_app_config(CONFIG_PATH)
    captured: dict[str, object] = {}

    class Controller:
        def __init__(self, _config, *, steps, speed) -> None:
            captured["speed"] = speed

        def start_run(self, _target_count, *, seed) -> None:
            del seed

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "RunController", Controller)
    monkeypatch.setattr(cli, "RunCatalog", lambda _root: object())
    monkeypatch.setattr(cli, "create_app", lambda **_kwargs: object())
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *_args, **_kwargs: None))

    assert cli._serve(
        config,
        Namespace(
            steps=0,
            speed=4.0,
            seed=42,
            host="127.0.0.1",
            port=8000,
        ),
    ) == 0
    assert captured["speed"] == 4.0
