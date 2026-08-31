"""One-command local UI showcase for rule-based future-event prediction.

The showcase uses the same strict IMM/B-spline inputs, rule predictor,
``OperationalFrame`` mapper, WebSocket contract, and React application as the
live system.  It replaces only the long-running mission/LLM orchestration with
one deterministic scenario so reviewers can inspect the feature immediately.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from math import atan2, cos, hypot, sin
import os
from pathlib import Path
import shutil
import subprocess
from typing import cast

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from underwater_tracking.api.frame_builder import build_operational_frame
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.models import (
    Contact,
    ContactClassification,
    DeploymentState,
    GroupQuality,
    GroupReport,
    SituationSnapshot,
    SurveillanceCapability,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.domain.ui_models import OperationalFrame
from underwater_tracking.world_model.demo import (
    SCENARIOS,
    ScenarioName,
    build_demo_input,
)
from underwater_tracking.world_model.models import RuleWorldModelInput
from underwater_tracking.world_model.rules import predict_future_events


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_UI_DIRECTORY = _REPOSITORY_ROOT / "src" / "underwater_tracking" / "ui"


def _target_state_at(
    inputs: RuleWorldModelInput, elapsed_s: float
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Advance the estimated target state without consulting simulator truth."""

    origin_x, origin_y = inputs.belief.position_xy
    velocity_x, velocity_y = inputs.belief.velocity_xy_mps
    speed = hypot(velocity_x, velocity_y)
    initial_heading = atan2(velocity_y, velocity_x) if speed > 0.0 else 0.0
    turn_rate = inputs.belief.turn_rate_rad_s
    heading = initial_heading + turn_rate * elapsed_s
    if speed <= 0.0:
        return (origin_x, origin_y), (0.0, 0.0), heading
    if abs(turn_rate) < 1.0e-9:
        position = (
            origin_x + velocity_x * elapsed_s,
            origin_y + velocity_y * elapsed_s,
        )
    else:
        radius = speed / turn_rate
        position = (
            origin_x + radius * (sin(heading) - sin(initial_heading)),
            origin_y - radius * (cos(heading) - cos(initial_heading)),
        )
    velocity = (speed * cos(heading), speed * sin(heading))
    return position, velocity, heading


def _rotated_offset(offset_xy: tuple[float, float], heading_rad: float) -> tuple[float, float]:
    offset_x, offset_y = offset_xy
    return (
        offset_x * cos(heading_rad) - offset_y * sin(heading_rad),
        offset_x * sin(heading_rad) + offset_y * cos(heading_rad),
    )


def _advance_demo_input(
    inputs: RuleWorldModelInput, *, elapsed_s: float, frame_id: int
) -> RuleWorldModelInput:
    """Move the estimate, formation and prediction forward for a live showcase."""

    elapsed_s = max(0.0, float(elapsed_s))
    frame_id = max(1, int(frame_id))
    offsets_s = tuple(time_s - inputs.as_of_s for time_s in inputs.trajectory.times_s)
    as_of_s = inputs.as_of_s + elapsed_s
    position, velocity, heading = _target_state_at(inputs, elapsed_s)
    future_states = tuple(_target_state_at(inputs, elapsed_s + offset_s) for offset_s in offsets_s)
    future_points = tuple(state[0] for state in future_states)
    future_times = tuple(as_of_s + offset_s for offset_s in offsets_s)
    origin_x, origin_y = inputs.belief.position_xy
    animated_uuvs = []
    for uuv in inputs.uuvs:
        base_offset = (
            uuv.position_xy[0] - origin_x,
            uuv.position_xy[1] - origin_y,
        )
        current_offset = _rotated_offset(base_offset, heading)
        current_position = (
            position[0] + current_offset[0],
            position[1] + current_offset[1],
        )
        planned_points = tuple(
            (
                future_position[0] + rotated[0],
                future_position[1] + rotated[1],
            )
            for (future_position, _, future_heading) in future_states
            for rotated in (_rotated_offset(base_offset, future_heading),)
        )
        animated_uuvs.append(
            uuv.model_copy(
                update={
                    "position_xy": current_position,
                    "velocity_xy_mps": velocity,
                    "planned_times_s": future_times,
                    "planned_points_xy": planned_points,
                }
            )
        )
    return inputs.model_copy(
        update={
            "as_of_s": as_of_s,
            "belief": inputs.belief.model_copy(
                update={"position_xy": position, "velocity_xy_mps": velocity}
            ),
            "trajectory": inputs.trajectory.model_copy(
                update={
                    "prediction_id": (f"{inputs.trajectory.prediction_id}-frame-{frame_id:06d}"),
                    "times_s": future_times,
                    "points_xy": future_points,
                }
            ),
            "uuvs": tuple(animated_uuvs),
            "source_observation_ids": (f"demo-observation-{frame_id:06d}",),
        }
    )


def build_showcase_frame(
    scenario: ScenarioName = "left_turn",
    *,
    elapsed_s: float = 0.0,
    frame_id: int = 1,
) -> OperationalFrame:
    """Build one frontend-ready frame without simulator truth or an LLM."""

    inputs = _advance_demo_input(build_demo_input(scenario), elapsed_s=elapsed_s, frame_id=frame_id)
    forecast = predict_future_events(inputs)
    covariance = (
        (400.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 400.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 4.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 4.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0e-4),
    )
    report = GroupReport(
        group_id="showcase_group_01",
        target_id=inputs.target_id,
        sim_time_s=int(inputs.as_of_s),
        member_ids=tuple(uuv.uuv_id for uuv in inputs.uuvs),
        belief=TargetBelief(
            target_id=inputs.target_id,
            sim_time_s=int(inputs.as_of_s),
            mean=(
                inputs.belief.position_xy[0],
                inputs.belief.position_xy[1],
                inputs.belief.velocity_xy_mps[0],
                inputs.belief.velocity_xy_mps[1],
                inputs.belief.turn_rate_rad_s,
            ),
            covariance=covariance,
            model_probabilities=inputs.belief.model_probabilities,
            source_observation_ids=inputs.source_observation_ids,
            fim_min_eigenvalue=0.01,
            fim_condition=2.0,
        ),
        quality=GroupQuality(
            instant=inputs.tracking.quality_ewma,
            window_mean=inputs.tracking.quality_ewma,
            ewma=inputs.tracking.quality_ewma,
            components={"showcase": inputs.tracking.quality_ewma},
        ),
        plan_revision=0,
    )
    uuv_states = tuple(
        UUVState(
            uuv_id=uuv.uuv_id,
            position_xy=uuv.position_xy,
            heading_rad=(
                atan2(uuv.velocity_xy_mps[1], uuv.velocity_xy_mps[0])
                if hypot(*uuv.velocity_xy_mps) > 0.0
                else 0.0
            ),
            speed_mps=hypot(*uuv.velocity_xy_mps),
            energy_fraction=uuv.energy_fraction,
            status=UUVStatus.TRACK,
            deployment_state=DeploymentState.DEPLOYED,
            group_id=report.group_id,
            capability=SurveillanceCapability(
                passive_range_m=uuv.passive_range_m,
                bearing_variance_rad2=uuv.bearing_variance_rad2,
            ),
        )
        for uuv in inputs.uuvs
    )
    snapshot = SituationSnapshot(
        scenario_id=inputs.scenario_id,
        snapshot_revision=frame_id,
        sim_time_s=int(inputs.as_of_s),
        uuvs=uuv_states,
        contacts=(
            Contact(
                contact_id=inputs.target_id,
                sim_time_s=int(inputs.as_of_s),
                classification=ContactClassification.SUBMARINE,
                classification_evidence=("showcase-estimator",),
                estimated_position_xy=inputs.belief.position_xy,
            ),
        ),
        group_reports=(report,),
        pending_events=(),
        map_bounds_xy=inputs.map_bounds_xy,
    )
    prediction = PredictedTrackRef(
        prediction_id=inputs.trajectory.prediction_id,
        target_id=inputs.target_id,
        sim_time_s=int(inputs.as_of_s),
        horizon_s=inputs.trajectory.times_s[-1] - inputs.as_of_s,
        sample_step_s=(
            inputs.trajectory.times_s[1] - inputs.trajectory.times_s[0]
            if len(inputs.trajectory.times_s) > 1
            else inputs.trajectory.times_s[0] - inputs.as_of_s
        ),
        times_s=inputs.trajectory.times_s,
        points_xy=inputs.trajectory.points_xy,
        corridor_radius_m=inputs.trajectory.corridor_radius_m,
        fallback_used=inputs.trajectory.fallback_used,
        fallback_reason=inputs.trajectory.fallback_reason,
        prediction_regime="bspline",
        imm_model_probabilities=inputs.belief.model_probabilities,
    )
    return build_operational_frame(
        snapshot,
        None,
        (),
        (),
        (),
        predictions={inputs.target_id: prediction},
        world_model_forecasts={inputs.target_id: forecast},
        frame_id=frame_id,
        uuv_only=True,
        run_phase="running",
        planning_snapshot_revision=frame_id,
        planning_sim_time_s=int(inputs.as_of_s),
        planning_data_age_s=0,
        planning_data_status="current",
        operational_stage_flags=("task_execution",),
        llm_thinking=("固定场景正在展示 IMM 与 B-spline 轨迹如何触发只读未来事件规则。"),
        llm_thinking_trigger="world_model_showcase",
        configured_roles=(),
    )


class _ShowcaseState:
    def __init__(
        self,
        scenario: ScenarioName,
        *,
        frame_interval_s: float,
        sim_step_s: float,
    ) -> None:
        self.scenario = scenario
        self.frame_interval_s = frame_interval_s
        self.sim_step_s = sim_step_s
        self.frame_id = 1
        self.elapsed_s = 0.0
        self.frame = build_showcase_frame(scenario)
        self.payload = cast(dict[str, object], self.frame.model_dump(mode="json"))
        self.history: deque[dict[str, object]] = deque((self.payload,), maxlen=600)
        self.subscribers: set[asyncio.Queue[dict[str, object]]] = set()

    def advance(self) -> dict[str, object]:
        self.frame_id += 1
        self.elapsed_s += self.sim_step_s
        self.frame = build_showcase_frame(
            self.scenario,
            elapsed_s=self.elapsed_s,
            frame_id=self.frame_id,
        )
        self.payload = cast(dict[str, object], self.frame.model_dump(mode="json"))
        self.history.append(self.payload)
        return self.payload


async def _run_animation(state: _ShowcaseState) -> None:
    while True:
        await asyncio.sleep(state.frame_interval_s)
        payload = state.advance()
        for queue in tuple(state.subscribers):
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with suppress(asyncio.QueueFull):
                queue.put_nowait(payload)


def create_showcase_app(
    scenario: ScenarioName = "left_turn",
    *,
    animate: bool = False,
    frame_interval_s: float = 0.25,
    sim_step_s: float = 10.0,
) -> FastAPI:
    """Create the minimal live/replay API consumed by the existing React UI."""

    if frame_interval_s <= 0.0:
        raise ValueError("frame_interval_s must be positive")
    if sim_step_s <= 0.0:
        raise ValueError("sim_step_s must be positive")
    state = _ShowcaseState(
        scenario,
        frame_interval_s=frame_interval_s,
        sim_step_s=sim_step_s,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        animation_task = asyncio.create_task(_run_animation(state)) if animate else None
        try:
            yield
        finally:
            if animation_task is not None:
                animation_task.cancel()
                with suppress(asyncio.CancelledError):
                    await animation_task

    app = FastAPI(title="Rule World Model Showcase", lifespan=lifespan)

    @app.get("/api/operational/snapshot")
    async def operational_snapshot() -> dict[str, object]:
        return state.payload

    @app.get("/api/replay")
    async def replay(
        start_s: int = 0,
        end_s: int | None = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> dict[str, object]:
        available = [
            payload
            for payload in state.history
            if float(cast(int | float, payload["sim_time_s"])) >= start_s
            and (end_s is None or float(cast(int | float, payload["sim_time_s"])) <= end_s)
        ]
        page = available[max(0, offset) : max(0, offset) + max(1, limit)]
        return {
            "frames": page,
            "count": len(page),
            "total_count": len(available),
            "offset": max(0, offset),
            "limit": max(1, limit),
        }

    @app.get("/api/runs")
    async def runs() -> dict[str, object]:
        frame = state.frame
        return {
            "runs": [
                {
                    "run_id": f"world-model-showcase-{scenario}",
                    "scenario_id": frame.scenario_id,
                    "target_count": len(frame.target_estimates),
                    "seed": 0,
                    "sim_time_s": frame.sim_time_s,
                    "frame_count": len(state.history),
                    "status": "running" if animate else "completed",
                    "path": "in-memory",
                    "effective_demo_speed": (sim_step_s / frame_interval_s if animate else None),
                    "phase": "running" if animate else "completed",
                }
            ]
        }

    @app.websocket("/ws/operational")
    async def operational_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=2)
        state.subscribers.add(queue)
        send_lock = asyncio.Lock()

        async def send_payload(payload: object) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        async def send_frames() -> None:
            while True:
                await send_payload(await queue.get())

        async def receive_commands() -> None:
            try:
                while True:
                    message = await websocket.receive_text()
                    if message == "ping":
                        async with send_lock:
                            await websocket.send_text("pong")
            except WebSocketDisconnect:
                return

        await send_payload(state.payload)
        tasks = {
            asyncio.create_task(send_frames()),
            asyncio.create_task(receive_commands()),
        }
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            return
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            state.subscribers.discard(queue)

    return app


def _start_ui(api_port: int, host: str, ui_port: int) -> subprocess.Popen[bytes]:
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is required to launch the showcase UI")
    if not (_UI_DIRECTORY / "node_modules").is_dir():
        raise RuntimeError(
            f"frontend dependencies are missing; run npm --prefix {_UI_DIRECTORY} install"
        )
    environment = os.environ.copy()
    environment["UNDERWATER_TRACKING_API_PORT"] = str(api_port)
    if os.name == "nt":
        command: str | list[str] = (
            f'"{npm}" --prefix "{_UI_DIRECTORY}" run dev -- '
            f'--host "{host}" --port {ui_port} --strictPort'
        )
        return subprocess.Popen(
            command,
            shell=True,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    command = [
        npm,
        "--prefix",
        str(_UI_DIRECTORY),
        "run",
        "dev",
        "--",
        "--host",
        host,
        "--port",
        str(ui_port),
        "--strictPort",
    ]
    return subprocess.Popen(command, env=environment, start_new_session=True)


def _stop_ui(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="left_turn")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=5173)
    parser.add_argument(
        "--static",
        action="store_true",
        help="serve the original single-frame showcase instead of animation",
    )
    parser.add_argument(
        "--frame-interval-s",
        type=float,
        default=0.25,
        help="real-time seconds between animated frames",
    )
    parser.add_argument(
        "--sim-step-s",
        type=float,
        default=10.0,
        help="simulation seconds advanced by each animated frame",
    )
    args = parser.parse_args(argv)
    scenario = cast(ScenarioName, args.scenario)
    ui_process = _start_ui(args.api_port, args.host, args.ui_port)
    try:
        print("\nRule world-model showcase:")
        print(f"  Scenario: {scenario}")
        print(f"  Web UI:   http://{args.host}:{args.ui_port}")
        print(f"  API:      http://{args.host}:{args.api_port}/docs")
        print(
            "  Motion:   "
            + (
                "static single frame"
                if args.static
                else (
                    f"animated ({args.sim_step_s:g} simulated seconds every "
                    f"{args.frame_interval_s:g} real seconds)"
                )
            )
        )
        if not args.static:
            print("  Map tip:  click 'fit current focus' to make motion easier to see")
        print("  Ctrl+C stops both services.\n", flush=True)
        uvicorn.run(
            create_showcase_app(
                scenario,
                animate=not args.static,
                frame_interval_s=args.frame_interval_s,
                sim_step_s=args.sim_step_s,
            ),
            host=args.host,
            port=args.api_port,
            log_level="info",
        )
    finally:
        _stop_ui(ui_process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
