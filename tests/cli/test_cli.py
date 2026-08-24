from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import underwater_tracking.cli as cli
from underwater_tracking.config.loader import load_app_config


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/scenario/uuv_only_single_target.yaml"


def test_parse_args_uses_shared_api_port_environment_for_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNDERWATER_TRACKING_API_PORT", "8001")
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_app_config", lambda _path: object())

    def capture_serve(_config: object, args: object) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "_serve", capture_serve)

    assert cli.main(
        [
            "serve",
            "--config",
            str(CONFIG_PATH),
            "--seed",
            "42",
        ]
    ) == 0

    assert captured["args"].port == 8001


def test_local_memory_embedding_provider_verifies_readiness_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config(CONFIG_PATH).memory
    assert config is not None
    calls: list[str] = []

    class Provider:
        def __init__(self, received_config, **kwargs: object) -> None:
            assert received_config is config
            del kwargs

        def verify_ready(self) -> None:
            calls.append("verify_ready")

    monkeypatch.setattr(cli, "SentenceTransformerEmbeddingProvider", Provider)

    provider = cli._build_memory_embedding_provider(config)

    assert isinstance(provider, Provider)
    assert calls == ["verify_ready"]


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

        def abort(self) -> None:
            captured["aborted"] = True

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "RunController", Controller)
    monkeypatch.setattr(cli, "RunCatalog", lambda _root: object())
    monkeypatch.setattr(cli, "create_app", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "_run_api_server", lambda *_args, **_kwargs: None)

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

        def abort(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "RunController", Controller)
    monkeypatch.setattr(cli, "RunCatalog", lambda _root: object())
    monkeypatch.setattr(cli, "create_app", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "_run_api_server", lambda *_args, **_kwargs: None)

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


def test_serve_aborts_controller_on_first_keyboard_interrupt(monkeypatch) -> None:
    config = load_app_config(CONFIG_PATH)
    captured: list[str] = []

    class Controller:
        def __init__(self, _config, *, steps, speed) -> None:
            del steps, speed

        def start_run(self, _target_count, *, seed) -> None:
            del seed

        def abort(self) -> None:
            captured.append("abort")

        def close(self) -> None:
            captured.append("close")

    monkeypatch.setattr(cli, "RunController", Controller)
    monkeypatch.setattr(cli, "RunCatalog", lambda _root: object())
    monkeypatch.setattr(cli, "create_app", lambda **_kwargs: object())

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run_api_server", interrupt)

    with pytest.raises(KeyboardInterrupt):
        cli._serve(
            config,
            Namespace(
                steps=0,
                speed=None,
                seed=42,
                host="127.0.0.1",
                port=8000,
            ),
        )

    assert captured == ["abort", "close"]
