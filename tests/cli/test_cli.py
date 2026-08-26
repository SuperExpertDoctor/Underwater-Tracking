from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

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


def test_serve_passes_one_static_ui_directory_to_fastapi(monkeypatch, tmp_path: Path) -> None:
    config = load_app_config(CONFIG_PATH)
    captured: dict[str, object] = {}

    class Controller:
        def __init__(self, _config, *, steps, speed, output_root=None) -> None:
            captured["steps"] = steps
            captured["speed"] = speed
            captured["output_root"] = output_root

        def start_run(self, _target_count, *, seed) -> None:
            captured["seed"] = seed

        def abort(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "RunController", Controller)
    monkeypatch.setattr(cli, "RunCatalog", lambda _root: object())
    monkeypatch.setattr(
        cli,
        "create_app",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(cli, "_run_api_server", lambda *_args, **_kwargs: None)

    static_dir = tmp_path / "dist"
    assert cli._serve(
        config,
        Namespace(
            steps=1,
            speed=0.0,
            seed=42,
            host="127.0.0.1",
            port=8000,
            static_ui_dir=static_dir,
            output_root=tmp_path / "outputs",
        ),
    ) == 0

    assert captured["static_ui_dir"] == static_dir
    assert captured["output_root"] == tmp_path / "outputs"


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


def test_agent_dependencies_use_configured_prediction_history_and_target_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config(CONFIG_PATH)
    predictor_args: dict[str, object] = {}

    def capture_predictor(**kwargs: object) -> object:
        predictor_args.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "make_snapshot_predictor", capture_predictor)
    monkeypatch.setattr(
        cli,
        "CarrierDependencies",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    loop = object.__new__(cli._AgentLoop)
    loop._config = config
    loop.plans = object()
    loop.events = object()
    loop.ledger = object()
    loop.llm = object()
    loop._clock = object()
    loop.scenario_id = config.scenario.scenario_id
    loop._knowledge_client = None
    loop._memory_service = object()
    loop._memory_short_term = object()
    loop._memory_port = object()
    loop._active_epoch = None
    loop._epoch_commit_port = None
    loop._role_model = lambda _role: "model"  # type: ignore[method-assign]

    dependencies = loop._deps()

    assert predictor_args["belief_history"].__self__ is loop  # type: ignore[union-attr]
    assert predictor_args["belief_history"].__name__ == "_belief_history"  # type: ignore[union-attr]
    assert predictor_args["max_speed_mps"] == config.tracking.submarine_sprint_speed_mps
    assert (
        predictor_args["max_turn_rate_rad_s"]
        == config.tracking.submarine_turn_rate_rad_s
    )
    assert dependencies.belief_history.__self__ is loop
    assert dependencies.belief_history.__name__ == "_belief_history"
    assert dependencies.world_model_config is config.world_model


def test_uuv_only_prediction_history_uses_globally_known_target_trajectory() -> None:
    config = load_app_config(CONFIG_PATH)
    calls: list[str] = []

    class Engine:
        def global_target_history(self, target_id: str):
            calls.append(f"global:{target_id}")
            return ((0, 1.0, 2.0), (30, 4.0, 6.0))

        def belief_history(self, target_id: str):
            calls.append(f"belief:{target_id}")
            return ((0, -1.0, -2.0),)

    loop = object.__new__(cli._AgentLoop)
    loop._config = config
    loop._engine = Engine()

    history = loop._belief_history(SimpleNamespace(), "target_00")

    assert history == ((0, 1.0, 2.0), (30, 4.0, 6.0))
    assert calls == ["global:target_00"]


def test_observation_checks_deterministic_region_rollover_after_prediction_refresh() -> None:
    prediction_state = {"predictions": {"target_00": object()}}
    calls: list[tuple[object, object]] = []
    situation = SimpleNamespace(sim_time_s=30)
    loop = object.__new__(cli._AgentLoop)
    loop._runtime = SimpleNamespace(
        refresh_predictions=lambda current: prediction_state,
    )
    loop._refresh_deterministic_mission = lambda current, state: calls.append(  # type: ignore[method-assign]
        (current, state)
    )
    loop._epoch_coordinator = None
    loop._background_carrier = True
    loop._submit_due_periodic_summary = lambda _current: None  # type: ignore[method-assign]
    loop._start_background_cycle = lambda _current: None  # type: ignore[method-assign]
    loop._record_carrier_error = lambda *_args: None  # type: ignore[method-assign]

    loop.on_situation(situation)

    assert calls == [(situation, prediction_state)]
