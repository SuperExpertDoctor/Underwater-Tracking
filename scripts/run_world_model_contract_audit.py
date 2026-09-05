"""Bounded ordinary-physics diagnostic with captured, deterministic LLM replies.

This is NOT external-provider/joint mission acceptance. No target trajectory,
observation, owner transition, or controller threshold is injected or changed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from tests.integration.test_uuv_only_production_acceptance import FixedSeedUUVLLM
from underwater_tracking.cli import _AgentLoop, _mission_controller_for
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.simulation.engine import SimulationEngine


def run_once(output: Path, seconds: int, seed: int) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    config = load_app_config(ROOT / "configs/scenario/uuv_only_single_target.yaml")
    provider = FixedSeedUUVLLM()
    loop = _AgentLoop(
        config,
        database_path=output / "agent.db",
        llm={"master": provider},
        run_id=f"world-contract-{seed}",
        steps=seconds // config.timing.physics_step_s,
        seed=seed,
    )
    engine = SimulationEngine(
        config,
        seed=seed,
        output_dir=output / "simulation",
        carrier=loop.on_situation,
        mission_controller=_mission_controller_for(config),
    )
    rows = []
    failure = None
    try:
        loop.attach(engine)
        if loop.install_deterministic_baseline(engine.publication_situation()) is None:
            raise RuntimeError("baseline_missing")
        for step in range(seconds // config.timing.physics_step_s):
            engine.step()
            loop.publish_latest()
            frame = loop.hub.snapshot()
            if frame.sim_time_s % config.timing.observation_step_s:
                continue
            reports = engine.publication_situation().group_reports
            rows.append(
                {
                    "time_s": frame.sim_time_s,
                    "visible_uuvs": len(frame.uuvs),
                    "beliefs": [report.belief.model_dump(mode="json") for report in reports],
                    "world_models": [
                        target.world_model.model_dump(mode="json")
                        for target in frame.target_estimates
                        if target.world_model
                    ],
                    "prediction_ids": [
                        target.prediction.prediction_id
                        for target in frame.target_estimates
                        if target.prediction
                    ],
                    "owner": frame.execution.tracking_control.tracking_owner_group_id
                    if frame.execution
                    else None,
                    "execution_revision": frame.execution.execution_revision
                    if frame.execution
                    else None,
                }
            )
            if frame.sim_time_s % 600 == 0:
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "time_s": frame.sim_time_s,
                            "track_revisions": [r.belief.track_revision for r in reports],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    except Exception as exc:  # noqa: BLE001 - diagnostic must preserve failure evidence and close resources
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        errors = loop.carrier_error_details
        closed = loop.close(timeout_s=30.0)
        engine.logger.close()
    statuses = Counter(w["data_status"] for row in rows for w in row["world_models"])
    event_types = Counter(
        e["event_type"] for row in rows for w in row["world_models"] for e in w["events"]
    )
    observations = sorted(
        {
            oid
            for row in rows
            for belief in row["beliefs"]
            for oid in belief["source_observation_ids"]
        }
    )
    owners = [row["owner"] for row in rows if row["owner"]]
    outcome = {
        "mode": "ordinary_physics_with_deterministic_llm_stub",
        "live_external_provider": False,
        "seed": seed,
        "physics_step_s": config.timing.physics_step_s,
        "requested_seconds": seconds,
        "completed_seconds": rows[-1]["time_s"] if rows else 0,
        "failure": failure,
        "clean_shutdown": closed,
        "carrier_errors": errors,
        "observation_count": len(observations),
        "observation_ids": observations,
        "max_track_revision": max(
            (b["track_revision"] for r in rows for b in r["beliefs"]), default=0
        ),
        "forecast_status_counts": dict(statuses),
        "event_type_counts": dict(event_types),
        "first_owner_seen": bool(owners),
        "owner_change_seen": len(set(owners)) > 1,
        "joint_acceptance": "NOT_RUN_EXTERNAL_PROVIDER_AND_OTHER_OWNER_GATES",
        "trace_sha256": sha256(
            json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    (output / "public-trace.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = [run_once(args.output / f"run-{index}", args.seconds, args.seed) for index in (1, 2)]
    summary = {
        "runs": results,
        "repeat_trace_equal": results[0]["trace_sha256"] == results[1]["trace_sha256"],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if all(r["failure"] is None and r["clean_shutdown"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
