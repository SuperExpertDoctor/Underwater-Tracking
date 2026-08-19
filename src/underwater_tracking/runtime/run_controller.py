"""Own the replaceable live-simulation bundle used by ``serve``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any

from underwater_tracking.agent.llm import StructuredLLM
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.config.models import AppConfig
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.runtime.models import RunRequest, RunSummary
from underwater_tracking.runtime.mission_controller import MissionController


@dataclass(slots=True)
class _RunBundle:
    config: AppConfig
    run_dir: Path
    loop: Any
    engine: SimulationEngine
    replay: ReplayService
    hub: OperationalHub
    stop: Event
    worker_errors: list[BaseException]
    mission_controller: MissionController | None = None
    worker: Thread | None = None


class RunController:
    """Create, replace, and close a single running simulation bundle.

    A request is completely validated and constructed before the installed
    bundle is touched. This makes an invalid target count a no-op for the
    active run, which is important once the command center exposes this as
    an operator control.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        output_root: Path = Path("outputs"),
        llm: Mapping[str, StructuredLLM[Any]] | None = None,
        steps: int = 0,
        speed: float = 1.0,
        synthetic_max_target_count: int | None = None,
    ) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if speed < 0:
            raise ValueError("speed must be non-negative")
        if synthetic_max_target_count is not None and synthetic_max_target_count < 1:
            raise ValueError("synthetic_max_target_count must be positive")
        self._config = config
        self._output_root = output_root
        self._llm = llm
        self._steps = steps
        self._speed = speed
        self._synthetic_max_target_count = synthetic_max_target_count
        self._lock = RLock()
        self._bundle: _RunBundle | None = None

    def start_run(self, target_count: int, seed: int | None = None) -> RunSummary:
        """Build and atomically install a new bundle for ``target_count``."""
        request = RunRequest(target_count=target_count, seed=seed)
        config = self._config_for(request.target_count)
        selected_seed = config.scenario.seed if request.seed is None else request.seed
        candidate: _RunBundle | None = None
        try:
            candidate = self._build_bundle(config, selected_seed)
            candidate.worker = self._start_worker(candidate)
            with self._lock:
                previous = self._bundle
                if previous is not None:
                    self._close_bundle(previous)
                self._bundle = candidate
                return self._summary(candidate)
        except BaseException:
            if candidate is not None:
                self._close_bundle(candidate)
            raise

    def current(self) -> RunSummary:
        """Return the installed run's latest public state."""
        with self._lock:
            if self._bundle is None:
                raise RuntimeError("no live run has been started")
            return self._summary(self._bundle)

    @property
    def runtime(self) -> Any:
        with self._lock:
            if self._bundle is None:
                raise RuntimeError("no live run has been started")
            return self._bundle.loop.runtime

    @property
    def replay(self) -> ReplayService:
        with self._lock:
            if self._bundle is None:
                raise RuntimeError("no live run has been started")
            return self._bundle.replay

    @property
    def hub(self) -> OperationalHub:
        with self._lock:
            if self._bundle is None:
                raise RuntimeError("no live run has been started")
            return self._bundle.hub

    @property
    def mission_controller(self) -> MissionController | None:
        with self._lock:
            if self._bundle is None:
                raise RuntimeError("no live run has been started")
            return self._bundle.mission_controller

    def close(self) -> None:
        """Stop and release the currently installed bundle, if any."""
        with self._lock:
            bundle = self._bundle
            self._bundle = None
        if bundle is not None:
            self._close_bundle(bundle)

    def _config_for(self, target_count: int) -> AppConfig:
        scenario = self._config.scenario
        if self._config.environment is not None:
            roster_size = len(self._config.environment.submarines)
            if target_count > roster_size:
                raise ValueError(
                    "target_count exceeds the loaded platform-core target roster"
                )
            return self._config

        capacity = (
            self._synthetic_max_target_count
            if self._synthetic_max_target_count is not None
            else scenario.max_target_count
        )
        if target_count > capacity:
            raise ValueError(
                f"target_count must be between 1 and {capacity} for this scenario"
            )
        return self._config.model_copy(
            update={
                "scenario": scenario.model_copy(
                    update={
                        "initial_target_count": target_count,
                        "max_target_count": target_count,
                    }
                )
            }
        )

    def _build_bundle(self, config: AppConfig, seed: int) -> _RunBundle:
        """Construct all resources before replacing the active bundle."""
        # Kept lazy to avoid a module cycle while ``cli`` owns _AgentLoop.
        from underwater_tracking.cli import _AgentLoop, _create_public_run_dir

        run_dir = _create_public_run_dir("serve", output_root=self._output_root)
        loop: Any | None = None
        try:
            mission_controller = (
                MissionController(
                    scenario_id=config.scenario.scenario_id,
                    region_entry_probability_threshold=(
                        config.scenario.region_entry_probability_threshold
                    ),
                    region_transition_confirm_cycles=(
                        config.scenario.region_transition_confirm_cycles
                    ),
                )
                if config.scenario.uuv_only
                else None
            )
            loop = _AgentLoop(
                config,
                database_path=run_dir / "agent.db",
                llm=self._llm,
                run_id=run_dir.name,
                steps=self._steps,
                seed=seed,
                background_carrier=True,
            )
            engine = SimulationEngine(
                config,
                seed=seed,
                output_dir=run_dir,
                carrier=loop.on_situation,
                mission_controller=mission_controller,
            )
            loop.attach(engine)
            return _RunBundle(
                config=config,
                run_dir=run_dir,
                loop=loop,
                engine=engine,
                replay=ReplayService(run_dir / "operational_frames.jsonl"),
                hub=loop.hub,
                stop=Event(),
                worker_errors=[],
                mission_controller=mission_controller,
            )
        except BaseException:
            if loop is not None:
                loop.close()
            raise

    def _start_worker(self, bundle: _RunBundle) -> Thread:
        """Start a completed bundle's simulation worker."""
        from underwater_tracking.cli import _step_with_llm_retries

        def drive() -> None:
            completed = 0
            try:
                while not bundle.stop.is_set() and (
                    self._steps == 0 or completed < self._steps
                ):
                    if not _step_with_llm_retries(
                        bundle.engine, bundle.loop, bundle.config, stop=bundle.stop
                    ):
                        if bundle.stop.wait(1.0):
                            break
                        continue
                    completed += 1
                    if self._speed > 0:
                        bundle.stop.wait(bundle.config.timing.physics_step_s / self._speed)
            except BaseException as exc:  # noqa: BLE001 - reported via RunSummary
                bundle.worker_errors.append(exc)
                bundle.stop.set()

        worker = Thread(target=drive, name="underwater-simulation", daemon=True)
        worker.start()
        return worker

    def _summary(self, bundle: _RunBundle) -> RunSummary:
        if bundle.worker_errors:
            status = "failed"
        elif bundle.stop.is_set():
            status = "stopped"
        elif bundle.worker is not None and not bundle.worker.is_alive():
            status = "completed"
        else:
            status = "running"
        publisher = getattr(bundle.loop, "_publisher", None)
        return RunSummary(
            run_id=bundle.run_dir.name,
            scenario_id=bundle.config.scenario.scenario_id,
            target_count=bundle.config.scenario.initial_target_count,
            seed=bundle.engine._seed,
            sim_time_s=bundle.engine._clock.sim_time_s,
            frame_count=publisher.frame_count if publisher is not None else 0,
            status=status,
            path=bundle.run_dir,
        )

    @staticmethod
    def _close_bundle(bundle: _RunBundle) -> None:
        bundle.stop.set()
        if bundle.worker is not None:
            bundle.worker.join(timeout=30.0)
        bundle.loop.write_manifest(bundle.run_dir)
        bundle.loop.close()
