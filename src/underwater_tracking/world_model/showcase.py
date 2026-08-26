"""One-command local UI showcase for rule-based future-event prediction.

The showcase uses the same strict IMM/B-spline inputs, rule predictor,
``OperationalFrame`` mapper, WebSocket contract, and React application as the
live system.  It replaces only the long-running mission/LLM orchestration with
one deterministic scenario so reviewers can inspect the feature immediately.
"""

from __future__ import annotations

import argparse
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
from underwater_tracking.world_model.rules import predict_future_events


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_UI_DIRECTORY = _REPOSITORY_ROOT / "src" / "underwater_tracking" / "ui"


def build_showcase_frame(scenario: ScenarioName = "left_turn") -> OperationalFrame:
    """Build one frontend-ready frame without simulator truth or an LLM."""

    inputs = build_demo_input(scenario)
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
            heading_rad=0.0,
            speed_mps=0.0,
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
        snapshot_revision=1,
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
        frame_id=1,
        uuv_only=True,
        run_phase="running",
        planning_snapshot_revision=1,
        planning_sim_time_s=int(inputs.as_of_s),
        planning_data_age_s=0,
        planning_data_status="current",
        operational_stage_flags=("task_execution",),
        llm_thinking=(
            "固定场景正在展示 IMM 与 B-spline 轨迹如何触发只读未来事件规则。"
        ),
        llm_thinking_trigger="world_model_showcase",
        configured_roles=(),
    )


def create_showcase_app(scenario: ScenarioName = "left_turn") -> FastAPI:
    """Create the minimal live/replay API consumed by the existing React UI."""

    app = FastAPI(title="Rule World Model Showcase")
    frame = build_showcase_frame(scenario)
    frame_payload = cast(dict[str, object], frame.model_dump(mode="json"))

    @app.get("/api/operational/snapshot")
    async def operational_snapshot() -> dict[str, object]:
        return frame_payload

    @app.get("/api/replay")
    async def replay(
        start_s: int = 0,
        end_s: int | None = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> dict[str, object]:
        in_range = frame.sim_time_s >= start_s and (
            end_s is None or frame.sim_time_s <= end_s
        )
        available = [frame_payload] if in_range else []
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
        return {
            "runs": [
                {
                    "run_id": f"world-model-showcase-{scenario}",
                    "scenario_id": frame.scenario_id,
                    "target_count": len(frame.target_estimates),
                    "seed": 0,
                    "sim_time_s": frame.sim_time_s,
                    "frame_count": 1,
                    "status": "completed",
                    "path": "in-memory",
                    "effective_demo_speed": None,
                    "phase": "completed",
                }
            ]
        }

    @app.websocket("/ws/operational")
    async def operational_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json(frame_payload)
        try:
            while True:
                message = await websocket.receive_text()
                if message == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            return

    return app


def _start_ui(api_port: int, host: str, ui_port: int) -> subprocess.Popen[bytes]:
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is required to launch the showcase UI")
    if not (_UI_DIRECTORY / "node_modules").is_dir():
        raise RuntimeError(
            "frontend dependencies are missing; run "
            f"npm --prefix {_UI_DIRECTORY} install"
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
    args = parser.parse_args(argv)
    scenario = cast(ScenarioName, args.scenario)
    ui_process = _start_ui(args.api_port, args.host, args.ui_port)
    try:
        print("\nRule world-model showcase:")
        print(f"  Scenario: {scenario}")
        print(f"  Web UI:   http://{args.host}:{args.ui_port}")
        print(f"  API:      http://{args.host}:{args.api_port}/docs")
        print("  Ctrl+C stops both services.\n", flush=True)
        uvicorn.run(
            create_showcase_app(scenario),
            host=args.host,
            port=args.api_port,
            log_level="info",
        )
    finally:
        _stop_ui(ui_process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
