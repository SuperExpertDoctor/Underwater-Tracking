#!/usr/bin/env python3
"""Own and audit a real full-duration command-center process."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
from http.client import HTTPException
import json
from math import isfinite
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from threading import Event, Thread
import time
from typing import cast
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from underwater_tracking.config.loader import load_app_config  # noqa: E402
from underwater_tracking.verification.live_demo import (  # noqa: E402
    LiveDemoAcceptanceResult,
    collect_stage_ids,
    validate_transport_frame_consistency,
    validate_uuv_only_frame,
    verify_live_demo,
)
from underwater_tracking.verification.physics_invariants import (  # noqa: E402
    BattleEvidenceChain,
    BlueTrackingEvidenceChain,
    EntityMotionAudit,
    EntityMotionLimits,
    FullBattleAcceptance,
    PredictionIntentEvidenceChain,
    UUVTrackingEvidenceChain,
)

EXPECTED_ENTITIES = {
    *(f"uuv_{index:02d}" for index in range(12)),
    "target_00",
}

_UI_MAX_SIM_TIME_DRIFT_S = 60
_UI_MAX_PLAN_VERSION_DRIFT = 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the complete real live battle")
    parser.add_argument("--main", type=Path, default=ROOT / "main.py")
    parser.add_argument("--scenario", type=Path, default=ROOT / "configs/scenario/uuv_only_single_target.yaml")
    parser.add_argument("--wall-timeout-s", type=float, default=1200.0)
    parser.add_argument("--expected-duration-s", type=int, default=28_800)
    parser.add_argument("--require-real-provider", action="store_true")
    parser.add_argument("--output-report", type=Path, default=ROOT / "docs/verification/main-live-battle-acceptance.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args(argv)

    api_port = _free_port()
    output_root = args.output_root.resolve()
    run_dirs_before = _run_output_dirs(output_root)
    serve_dirs_before = _serve_output_dirs(output_root)
    main_argv = [
        sys.executable,
        str(args.main),
        "--config",
        str(args.scenario),
        "--port",
        str(api_port),
        "--output-root",
        str(output_root),
        "--verification-audit",
    ]
    if args.require_real_provider:
        main_argv.append("--require-real-provider")
    scenario_config = load_app_config(args.scenario)
    physics_step_s = scenario_config.timing.physics_step_s
    if args.expected_duration_s < scenario_config.scenario.duration_s:
        run_steps = (args.expected_duration_s + physics_step_s - 1) // physics_step_s
        main_argv.extend(
            ("--steps", str(run_steps), "--bootstrap-planning")
        )
    process = subprocess.Popen(
        main_argv,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(sys.platform != "win32"),
    )
    base_url = f"http://127.0.0.1:{api_port}"
    browser_stop = Event()
    browser_result: list[tuple[int, int, tuple[str, ...], tuple[str, ...]]] = []
    browser_thread = Thread(
        target=lambda: browser_result.append(
            _browser_audit(
                base_url,
                args.output_report.parent / "screenshots",
                stop_event=browser_stop,
            )
        ),
        name="live-battle-browser-audit",
        daemon=True,
    )
    # Real provider attestation happens during ``main.py`` construction. The
    # slowest role probe can therefore delay API startup without being a run
    # failure. Keep startup bounded, but let the configured wall budget cover
    # a cold provider connection instead of imposing a fixed 60 s cutoff.
    api_ready = _wait_for_api(
        base_url,
        timeout_s=min(max(args.wall_timeout_s, 60.0), 300.0),
    )
    try:
        if api_ready:
            run_output_dir = _current_run_output_dir(
                base_url, output_root=output_root
            )
            browser_thread.start()
            if run_output_dir is None:
                result = LiveDemoAcceptanceResult(
                    violations=("run_output_dir_unavailable",)
                )
            else:
                result = _wait_and_verify(
                    process,
                    base_url,
                    run_output_dir,
                    args.require_real_provider,
                    args.wall_timeout_s,
                    args.expected_duration_s,
                )
        else:
            result = LiveDemoAcceptanceResult(violations=("api_boot_timeout",))
    finally:
        browser_stop.set()
        if browser_thread.is_alive():
            browser_thread.join(timeout=30.0)
    browser_errors, failed_requests, screenshot_paths, browser_error_details = (
        browser_result[0] if browser_result else (1, 1, (), ("browser_audit_unavailable",))
    )
    physics, physics_request_failed = _safe_get_json(base_url, "/api/verification/physics")
    evidence, evidence_request_failed = _safe_get_json(base_url, "/api/verification/evidence")
    result = _assemble_result(
        result,
        physics,
        evidence,
        browser_errors,
        failed_requests,
        int(physics_request_failed) + int(evidence_request_failed),
        args.expected_duration_s,
        git_commit=_git_commit(),
        config_sha256=hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        require_real_provider=args.require_real_provider,
        screenshot_paths=screenshot_paths,
        browser_error_details=browser_error_details,
    )
    shutdown_started = time.monotonic()
    shutdown_timed_out = _stop_process(process)
    shutdown_s = round(time.monotonic() - shutdown_started, 3)
    shutdown_violations = []
    if shutdown_timed_out or shutdown_s > 10.0:
        shutdown_violations.append("shutdown_exceeded_10s")
    exit_violation = _process_exit_violation(process.returncode)
    if exit_violation is not None:
        shutdown_violations.append(exit_violation)
    closed_ports = _wait_for_closed_ports((api_port,), timeout_s=2.0)
    if api_port not in closed_ports:
        shutdown_violations.append("api_port_still_open")
    run_dirs_after = _run_output_dirs(output_root)
    created_run_dirs = run_dirs_after - run_dirs_before
    if len(created_run_dirs) != 1:
        shutdown_violations.append(
            f"run_directory_count_mismatch:{len(created_run_dirs)}"
        )
    if _serve_output_dirs(output_root) - serve_dirs_before:
        shutdown_violations.append("serve_directory_created")
    final_violations = tuple(
        dict.fromkeys((*result.violations, *shutdown_violations))
    )
    result = result.model_copy(
        update={
            "shutdown_s": shutdown_s,
            "wall_clock_end_utc": _utc_now(),
            "completed": result.completed and not final_violations,
            "violations": final_violations,
        }
    )
    _write_reports(result, args.output_report)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0 if result.completed and not result.violations else 1


def _wait_and_verify(
    process: subprocess.Popen[str],
    base_url: str,
    output_dir: Path,
    require_real_provider: bool,
    wall_timeout_s: float,
    expected_duration_s: int,
) -> LiveDemoAcceptanceResult:
    started = time.monotonic()
    while time.monotonic() - started < min(wall_timeout_s, 60.0):
        if process.poll() is not None:
            return LiveDemoAcceptanceResult(violations=(f"main_process_exit:{process.returncode}",))
        try:
            _get_json(base_url, "/api/health")
            break
        except (OSError, URLError, ValueError):
            time.sleep(0.25)
    else:
        return LiveDemoAcceptanceResult(violations=("api_boot_timeout",))
    return verify_live_demo(
        base_url=base_url,
        output_dir=output_dir,
        require_real_provider=require_real_provider,
        wall_timeout_s=wall_timeout_s,
        expected_duration_s=expected_duration_s,
    )


def _assemble_result(
    live: LiveDemoAcceptanceResult,
    physics: object,
    evidence: object,
    browser_errors: int,
    failed_requests: int,
    verification_request_failures: int,
    expected_duration_s: int,
    *,
    git_commit: str | None,
    config_sha256: str | None,
    require_real_provider: bool = False,
    screenshot_paths: tuple[str, ...] = (),
    browser_error_details: tuple[str, ...] = (),
) -> FullBattleAcceptance:
    violations = list(live.violations)
    audits: list[EntityMotionAudit] = []
    motion_limits: dict[str, EntityMotionLimits] = {}
    physics_coverage: dict[str, object] = {}
    observed_physics_frame_count = 0
    expected_physics_frame_count: int | None = None
    physics_step_s: int | None = None
    if physics is None:
        violations.append("physics_audit_unavailable")
    if isinstance(physics, Mapping):
        raw_audits = physics.get("audits", ())
        if isinstance(raw_audits, list):
            audits = [EntityMotionAudit.model_validate(item) for item in raw_audits]
        raw_limits = physics.get("limits", {})
        if isinstance(raw_limits, Mapping):
            motion_limits = {
                str(entity_id): EntityMotionLimits.model_validate(value)
                for entity_id, value in raw_limits.items()
                if isinstance(value, Mapping)
            }
        raw_coverage = physics.get("coverage", {})
        if isinstance(raw_coverage, Mapping):
            physics_coverage = dict(raw_coverage)
            raw_count = raw_coverage.get("observed_frame_count")
            if isinstance(raw_count, int) and not isinstance(raw_count, bool):
                observed_physics_frame_count = raw_count
        raw_step = physics.get("physics_step_s")
        if isinstance(raw_step, int) and not isinstance(raw_step, bool) and raw_step > 0:
            physics_step_s = raw_step
    by_id = {audit.entity_id: audit for audit in audits}
    if len(by_id) != len(audits):
        violations.append("duplicate_motion_entities")
    if not isinstance(physics, Mapping) or physics.get("entity_count") != len(EXPECTED_ENTITIES):
        violations.append("physics_entity_count_mismatch")
    missing_entities = sorted(EXPECTED_ENTITIES - set(by_id))
    extra_entities = sorted(set(by_id) - EXPECTED_ENTITIES)
    if missing_entities:
        violations.append("missing_motion_entities:" + ",".join(missing_entities))
    if extra_entities:
        violations.append("unexpected_motion_entities:" + ",".join(extra_entities))
    expected_audit_ids = set(EXPECTED_ENTITIES)
    if set(motion_limits) != expected_audit_ids:
        violations.append("motion_limits_entity_set_mismatch")
    for audit in audits:
        if audit.observed_steps <= 0:
            violations.append(f"no_observed_steps:{audit.entity_id}")
        if audit.teleport_count or audit.boundary_violation_count or audit.limit_violation_count:
            violations.append(f"motion_violation:{audit.entity_id}")
    if physics_coverage:
        if physics_coverage.get("expected_entity_count") != len(EXPECTED_ENTITIES):
            violations.append("physics_expected_entity_count_mismatch")
        observed_ids = physics_coverage.get("observed_entity_ids", ())
        if not isinstance(observed_ids, (list, tuple, set, frozenset)) or set(
            observed_ids
        ) != EXPECTED_ENTITIES:
            violations.append("physics_observed_entity_set_mismatch")
        for field in (
            "duplicate_frame_ids",
            "duplicate_entity_frame_ids",
            "frame_id_gaps",
            "nonmonotonic_frame_ids",
            "nonmonotonic_sim_time_frame_ids",
            "inconsistent_sample_frame_ids",
        ):
            value = physics_coverage.get(field, ())
            if isinstance(value, (list, tuple, set, frozenset)) and value:
                violations.append(f"physics_{field}")
        missing_frames = physics_coverage.get("missing_entity_frame_ids", {})
        if isinstance(missing_frames, Mapping) and any(missing_frames.values()):
            violations.append("physics_missing_entity_frames")
    else:
        violations.append("physics_frame_coverage_unavailable")
    if physics_step_s is not None:
        expected_physics_frame_count = expected_duration_s // physics_step_s + 1
        if expected_duration_s % physics_step_s:
            expected_physics_frame_count += 1
        if live.final_sim_time_s >= expected_duration_s:
            if observed_physics_frame_count != expected_physics_frame_count:
                violations.append(
                    "physics_frame_count_mismatch:"
                    f"{observed_physics_frame_count}!={expected_physics_frame_count}"
                )
            expected_steps = expected_physics_frame_count - 1
            for audit in audits:
                if audit.observed_steps != expected_steps:
                    violations.append(
                        f"physics_step_count_mismatch:{audit.entity_id}:"
                        f"{audit.observed_steps}!={expected_steps}"
                    )
    if isinstance(physics, Mapping):
        raw_physics_violations = physics.get("violations", ())
        if isinstance(raw_physics_violations, (list, tuple)) and raw_physics_violations:
            violations.append("physics_monitor_violations")
    chains = _evidence_chains(evidence)
    uuv_only_evidence = _evidence_is_uuv_only(evidence)
    uuv_tracking_chains, uuv_tracking_violations = _uuv_only_tracking_chains(evidence)
    violations.extend(uuv_tracking_violations)
    if uuv_only_evidence:
        blue_tracking_chains: list[BlueTrackingEvidenceChain] = []
    else:
        blue_tracking_chains, blue_tracking_violations = _blue_tracking_chains(evidence)
        violations.extend(blue_tracking_violations)
    prediction_intent_chains, prediction_intent_violations = (
        _prediction_intent_chains(evidence)
    )
    violations.extend(prediction_intent_violations)
    if require_real_provider:
        violations.extend(_real_provider_attestation_violations(evidence))
    if evidence is None:
        violations.append("battle_evidence_unavailable")
    elif isinstance(evidence, Mapping) and evidence.get("background_drain_completed") is False:
        violations.append("background_worker_drain_failed")
    if verification_request_failures:
        violations.append(f"verification_requests_failed:{verification_request_failures}")
    if not chains:
        violations.append("missing_counter_tracking_evidence_chain")
    if uuv_only_evidence and not uuv_tracking_chains:
        violations.append("missing_uuv_tracking_evidence_chain")
    if not uuv_only_evidence and not blue_tracking_chains:
        violations.append("missing_blue_tracking_evidence_chain")
    if live.adversary_llm_decision_count <= 0:
        violations.append("missing_adversary_llm_decision")
    if browser_errors:
        violations.append(f"browser_errors:{browser_errors}")
    if failed_requests:
        violations.append(f"failed_requests:{failed_requests}")
    if live.final_sim_time_s < expected_duration_s:
        violations.append("battle_not_completed")
    if live.final_sim_time_s >= expected_duration_s and live.final_run_phase != "completed":
        violations.append(f"battle_phase_not_completed:{live.final_run_phase}")
    return FullBattleAcceptance(
        completed=live.final_sim_time_s >= expected_duration_s and not violations,
        final_sim_time_s=live.final_sim_time_s,
        final_plan_version=live.final_plan_version,
        final_run_phase=live.final_run_phase,
        wall_clock_start_utc=live.wall_clock_start_utc,
        wall_clock_end_utc=live.wall_clock_end_utc,
        first_plan_wall_s=live.first_plan_wall_s,
        required_stage_ids=live.observed_stage_ids,
        stage_sim_times_s=dict(live.stage_sim_times_s),
        stage_plan_versions=dict(live.stage_plan_versions),
        battle_evidence_chains=tuple(chains),
        blue_tracking_chains=tuple(blue_tracking_chains),
        uuv_tracking_chains=tuple(uuv_tracking_chains),
        prediction_intent_chains=tuple(prediction_intent_chains),
        motion_audits=tuple(sorted(audits, key=lambda audit: audit.entity_id)),
        motion_limits=motion_limits,
        observed_physics_frame_count=observed_physics_frame_count,
        expected_physics_frame_count=expected_physics_frame_count,
        physics_frame_coverage=physics_coverage,
        browser_error_count=browser_errors,
        browser_error_details=browser_error_details,
        failed_request_count=(
            live.failed_request_count + failed_requests + verification_request_failures
        ),
        memory_event_count=live.memory_event_count,
        api_p95_ms=live.api_p95_ms,
        output_bytes=live.output_bytes,
        shutdown_s=0.0,
        git_commit=git_commit,
        config_sha256=config_sha256,
        screenshot_paths=screenshot_paths,
        violations=tuple(dict.fromkeys(violations)),
    )


def _evidence_chains(value: object) -> list[BattleEvidenceChain]:
    if not isinstance(value, Mapping):
        return []
    raw_events = value.get("events", ())
    raw_decisions = value.get("adversary_decisions", ())
    raw_llm_calls = value.get("llm_calls", ())
    raw_response_chains = value.get("blue_response_chains", ())
    root_epoch_id = value.get("blue_epoch_id")
    root_plan_version = value.get("blue_plan_version")
    if (
        not isinstance(raw_events, (list, tuple))
        or not isinstance(raw_decisions, (list, tuple))
        or not isinstance(raw_llm_calls, (list, tuple))
    ):
        return []
    if not isinstance(raw_response_chains, (list, tuple)) or not raw_response_chains:
        return []
    raw_public_observations = value.get("public_observations", ())
    if not isinstance(raw_public_observations, (list, tuple)) or not raw_public_observations:
        return []
    events = [item for item in raw_events if isinstance(item, Mapping)]
    decisions = [item for item in raw_decisions if isinstance(item, Mapping)]
    llm_calls = [item for item in raw_llm_calls if isinstance(item, Mapping)]
    if (
        len(events) != len(raw_events)
        or len(decisions) != len(raw_decisions)
        or len(llm_calls) != len(raw_llm_calls)
    ):
        return []
    event_by_id: dict[str, Mapping[str, object]] = {}
    for item in events:
        event_id = item.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in event_by_id:
            return []
        event_by_id[event_id] = item
    if not decisions or not event_by_id:
        return []
    observation_by_id: dict[str, Mapping[str, object]] = {}
    for item in raw_public_observations:
        if not isinstance(item, Mapping):
            return []
        observation_id = item.get("observation_id")
        target_id = item.get("target_id")
        sim_time_s = item.get("sim_time_s")
        if (
            not isinstance(observation_id, str)
            or not observation_id
            or not isinstance(target_id, str)
            or not target_id
            or not isinstance(sim_time_s, int)
            or isinstance(sim_time_s, bool)
            or sim_time_s < 0
            or observation_id in observation_by_id
        ):
            return []
        observation_by_id[observation_id] = item
    decisions_by_id: dict[str, Mapping[str, object]] = {}
    for item in decisions:
        decision_id = item.get("decision_id")
        if (
            not isinstance(decision_id, str)
            or not decision_id
            or decision_id in decisions_by_id
        ):
            return []
        decisions_by_id[decision_id] = item
    llm_calls_by_id: dict[str, Mapping[str, object]] = {}
    for item in llm_calls:
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id or call_id in llm_calls_by_id:
            return []
        llm_calls_by_id[call_id] = item
    if not isinstance(root_epoch_id, str) or not root_epoch_id:
        return []
    if not isinstance(root_plan_version, int) or isinstance(root_plan_version, bool):
        return []
    chains: list[BattleEvidenceChain] = []
    for raw_chain in raw_response_chains:
        if not isinstance(raw_chain, Mapping):
            continue
        target_id = raw_chain.get("target_id")
        decision_id = str(raw_chain.get("decision_id", ""))
        decision = decisions_by_id.get(decision_id)
        if not isinstance(target_id, str) or not target_id or decision is None:
            continue
        if decision.get("target_id") != target_id:
            continue
        decision_time = decision.get("sim_time_s")
        if not isinstance(decision_time, int) or isinstance(decision_time, bool):
            continue
        provider_call_id = decision.get("provider_call_id")
        provider_call = (
            llm_calls_by_id.get(provider_call_id)
            if isinstance(provider_call_id, str)
            else None
        )
        provider_model = provider_call.get("model") if provider_call is not None else None
        if (
            provider_call is None
            or provider_call.get("operation") != "adversary_mission_decision"
            or not isinstance(provider_model, str)
            or not provider_model
            or any(marker in provider_model.lower() for marker in _NON_REAL_MODEL_MARKERS)
            or provider_call.get("sim_time_s") != decision_time
            or not isinstance(provider_call.get("prompt_version"), str)
            or not provider_call.get("prompt_version")
            or not isinstance(provider_call.get("request_hash"), str)
            or not provider_call.get("request_hash")
            or not isinstance(provider_call.get("response_hash"), str)
            or not provider_call.get("response_hash")
            or provider_call.get("error_category") != ""
        ):
            continue
        decision_event_id = decision.get("decision_event_id")
        expected_decision_event_id = f"target_mission_decision:{target_id}:{decision_id}"
        if decision_event_id != expected_decision_event_id:
            continue
        decision_event = event_by_id.get(expected_decision_event_id)
        if (
            decision_event is None
            or decision_event.get("event_type") != "target_mission_decision"
            or decision_event.get("entity_id") != target_id
            or decision_event.get("sim_time_s") != decision_time
        ):
            continue
        maneuver_time = raw_chain.get("maneuver_time_s")
        if not isinstance(maneuver_time, int) or maneuver_time != decision_time:
            continue
        source_values = decision.get("trigger_event_ids", ())
        if not isinstance(source_values, (list, tuple)) or not source_values:
            continue
        if any(not isinstance(item, str) or not item for item in source_values):
            continue
        source_ids = tuple(source_values)
        if len(source_ids) != len(set(source_ids)):
            continue
        detection_ids = tuple(
            event_id
            for event_id in source_ids
            if (
                event_by_id.get(event_id, {}).get("event_type")
                in {"target_detection_acquired", "active_ping"}
                and event_by_id.get(event_id, {}).get("entity_id") == target_id
                and isinstance(event_by_id.get(event_id, {}).get("sim_time_s"), int)
                and int(event_by_id[event_id]["sim_time_s"]) <= decision_time
            )
        )
        if not detection_ids:
            continue
        if any(event_id not in event_by_id for event_id in source_ids):
            continue
        raw_estimate_ids = raw_chain.get("blue_estimate_ids", ())
        if not isinstance(raw_estimate_ids, (list, tuple)) or any(
            not isinstance(item, str) or not item for item in raw_estimate_ids
        ):
            continue
        estimate_ids = tuple(raw_estimate_ids)
        if len(estimate_ids) != len(set(estimate_ids)) or not estimate_ids:
            continue
        epoch_id = raw_chain.get("blue_epoch_id", root_epoch_id)
        plan_version = raw_chain.get("plan_version", root_plan_version)
        response_event_id = raw_chain.get("response_event_id")
        response_event = (
            event_by_id.get(response_event_id)
            if isinstance(response_event_id, str)
            else None
        )
        if (
            response_event is None
            or response_event.get("event_type") != "state_changed"
            or response_event.get("phase") != "blue_response"
            or response_event.get("entity_id") != target_id
            or not isinstance(response_event.get("sim_time_s"), int)
            or int(response_event["sim_time_s"]) <= maneuver_time
            or response_event.get("decision_id") != decision_id
        ):
            continue
        motion_effect_event_id = raw_chain.get("motion_effect_event_id")
        motion_effect_event = (
            event_by_id.get(motion_effect_event_id)
            if isinstance(motion_effect_event_id, str)
            else None
        )
        if (
            motion_effect_event is None
            or motion_effect_event.get("event_type") != "state_changed"
            or motion_effect_event.get("phase") != "adversary_motion_effect"
            or motion_effect_event.get("entity_id") != target_id
            or not isinstance(motion_effect_event.get("sim_time_s"), int)
            or int(motion_effect_event["sim_time_s"]) <= maneuver_time
            or motion_effect_event.get("decision_id") != decision_id
            or not all(
                isinstance(motion_effect_event.get(field), (int, float))
                and not isinstance(motion_effect_event.get(field), bool)
                and isfinite(float(motion_effect_event[field]))
                and float(motion_effect_event[field]) >= 0.0
                for field in (
                    "speed_delta_mps",
                    "heading_delta_rad",
                    "depth_delta_m",
                )
            )
            or not any(
                float(motion_effect_event[field]) > 1e-8
                for field in (
                    "speed_delta_mps",
                    "heading_delta_rad",
                    "depth_delta_m",
                )
            )
        ):
            continue
        if (
            not isinstance(plan_version, int)
            or isinstance(plan_version, bool)
            or plan_version < 1
            or response_event.get("plan_version") != plan_version
            or root_plan_version < plan_version
            or not isinstance(epoch_id, str)
            or not epoch_id
        ):
            continue
        observation_values = raw_chain.get("public_observation_ids", ())
        if not isinstance(observation_values, (list, tuple)):
            continue
        if any(not isinstance(item, str) or not item for item in observation_values):
            continue
        observation_ids = tuple(observation_values)
        if len(observation_ids) != len(set(observation_ids)) or not observation_ids:
            continue
        if any(
            observation_id not in observation_by_id
            or observation_by_id[observation_id].get("target_id") != target_id
            or not isinstance(
                observation_by_id[observation_id].get("sim_time_s"), int
            )
            or isinstance(observation_by_id[observation_id].get("sim_time_s"), bool)
            or observation_by_id[observation_id]["sim_time_s"] <= maneuver_time
            for observation_id in observation_ids
        ):
            continue
        valid_observation_ids = set(observation_ids)
        estimate_events: list[Mapping[str, object]] = []
        for estimate_id in estimate_ids:
            estimate_event = event_by_id.get(estimate_id)
            source_observation_ids = (
                estimate_event.get("source_observation_ids", ())
                if estimate_event is not None
                else ()
            )
            if (
                estimate_event is None
                or estimate_event.get("event_type")
                not in {
                    "target_maneuver_observed",
                    "target_speed_regime_changed",
                    "observability_feedback",
                }
                or estimate_event.get("entity_id") != target_id
                or not isinstance(estimate_event.get("sim_time_s"), int)
                or int(estimate_event["sim_time_s"]) <= maneuver_time
                or not isinstance(source_observation_ids, (list, tuple))
                or not source_observation_ids
                or any(
                    not isinstance(item, str) or not item
                    for item in source_observation_ids
                )
                or not set(source_observation_ids) <= valid_observation_ids
            ):
                break
            estimate_events.append(estimate_event)
        else:
            chains.append(
                BattleEvidenceChain(
                    target_detection_event_id=detection_ids[0],
                    adversary_decision_id=decision_id,
                    adversary_provider_call_id=str(provider_call_id),
                    adversary_provider_model=provider_model,
                    adversary_source_event_ids=source_ids,
                    resulting_public_observation_ids=observation_ids,
                    blue_estimate_ids=estimate_ids,
                    motion_effect_event_id=str(motion_effect_event_id),
                    blue_epoch_id=str(epoch_id),
                    blue_plan_version=plan_version,
                )
            )
            continue
        if not estimate_events:
            continue
    return chains


def _evidence_is_uuv_only(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("uuv_only") is True:
        return True
    raw_events = value.get("events", ())
    if not isinstance(raw_events, (list, tuple)):
        return False
    uuv_event_types = {
        "uuv_boundary_entry_started",
        "uuv_boundary_exit_started",
        "uuv_boundary_exited",
        "uuv_boundary_exit_completed",
        "uuv_boundary_replacement",
    }
    return any(
        isinstance(event, Mapping)
        and (
            event.get("event_type") in uuv_event_types
            or (
                event.get("event_type") == "handoff_completed"
                and isinstance(event.get("predecessor_uuv_ids"), (list, tuple))
            )
        )
        for event in raw_events
    )


def _uuv_only_tracking_chains(
    value: object,
) -> tuple[list[UUVTrackingEvidenceChain], list[str]]:
    """Resolve UUV-only evidence through entry, handoff, exit and replacement."""
    if not isinstance(value, Mapping):
        return [], ["missing_uuv_tracking_evidence_chain"]
    events = _unique_mappings(value.get("events"), "event_id")
    if not events:
        return [], ["missing_uuv_tracking_evidence_chain"]
    if any(
        event.get("event_type") in {
            "carrier_dispatch_completed",
            "uuv_deployed",
            "uuv_recovery_requested",
            "uuv_recovered",
            "carrier_returned_to_fleet",
        }
        for event in events.values()
    ):
        return [], ["legacy_carrier_lifecycle_event"]

    handoffs = tuple(
        sorted(
            (
                event
                for event in events.values()
                if event.get("event_type") == "handoff_completed"
            ),
            key=lambda event: (
                _strict_int(event.get("sim_time_s")) or 0,
                str(event.get("event_id", "")),
            ),
        )
    )
    if not handoffs:
        return [], ["missing_uuv_tracking_evidence_chain"]
    chains: list[UUVTrackingEvidenceChain] = []
    violations: list[str] = []
    for handoff in handoffs:
        handoff_id = handoff.get("event_id")
        target_id = handoff.get("target_id")
        region_id = handoff.get("predecessor_region_id")
        successor_region_id = handoff.get("successor_region_id")
        handoff_time = _strict_int(handoff.get("sim_time_s"))
        uuv_ids = _string_tuple(handoff.get("predecessor_uuv_ids"))
        successor_uuv_ids = _string_tuple(handoff.get("successor_uuv_ids"))
        plan_version = _strict_int(handoff.get("plan_revision"))
        if (
            not isinstance(handoff_id, str)
            or not handoff_id
            or not isinstance(target_id, str)
            or not target_id
            or not isinstance(region_id, str)
            or not region_id
            or not isinstance(successor_region_id, str)
            or not successor_region_id
            or handoff_time is None
            or len(uuv_ids) != 2
            or len(set(uuv_ids)) != 2
            or not successor_uuv_ids
            or plan_version is None
            or plan_version < 1
        ):
            violations.append("incomplete_uuv_tracking_chain")
            continue

        entries = tuple(
            event
            for event in events.values()
            if event.get("event_type") == "uuv_boundary_entry_started"
            and event.get("entity_id") in uuv_ids
            and _event_region_id(event) == region_id
            and (_strict_int(event.get("sim_time_s")) or -1) <= handoff_time
        )
        entry_ids = tuple(
            str(event["event_id"])
            for event in sorted(entries, key=_event_sort_key)
            if isinstance(event.get("event_id"), str) and event.get("event_id")
        )
        active_ping = _first_uuv_event(
            events.values(),
            event_types={"active_ping"},
            target_id=target_id,
            uuv_ids=uuv_ids,
            lower_time=None,
            upper_time=handoff_time,
        )
        detections = tuple(
            event
            for event in events.values()
            if event.get("event_type") == "target_detection_acquired"
            and event.get("entity_id") == target_id
            and _event_time_between(event, None, handoff_time)
            and _event_platform_ids(event) & set(uuv_ids)
        )
        estimates = tuple(
            event
            for event in sorted(events.values(), key=_event_sort_key)
            if event.get("event_type")
            in {
                "target_estimate_updated",
                "target_maneuver_observed",
                "target_speed_regime_changed",
                "observability_feedback",
            }
            and event.get("entity_id") == target_id
            and _event_time_between(
                event,
                _strict_int(active_ping.get("sim_time_s")) if active_ping else None,
                handoff_time,
            )
            and _string_tuple(event.get("source_observation_ids"))
        )
        exits = tuple(
            event
            for event in sorted(events.values(), key=_event_sort_key)
            if event.get("event_type")
            in {
                "uuv_boundary_exit_started",
                "uuv_boundary_exited",
                "uuv_boundary_exit_completed",
            }
            and event.get("entity_id") in uuv_ids
            and _event_region_id(event) == region_id
            and _event_time_between(event, handoff_time, None)
        )
        exit_time = max(
            (_strict_int(event.get("sim_time_s")) or handoff_time for event in exits),
            default=handoff_time,
        )
        replacement = next(
            (
                event
                for event in sorted(events.values(), key=_event_sort_key)
                if event.get("event_type") == "uuv_boundary_replacement"
                and _event_region_id(event) in {None, region_id}
                and (_strict_int(event.get("sim_time_s")) or -1) >= exit_time
                and event.get("outgoing_uuv_id") in uuv_ids
                and isinstance(event.get("replacement_uuv_id"), str)
                and event.get("replacement_uuv_id") in set(successor_uuv_ids)
            ),
            None,
        )
        blue_response = next(
            (
                event
                for event in sorted(events.values(), key=_event_sort_key)
                if event.get("event_type") == "state_changed"
                and event.get("phase") == "blue_response"
                and event.get("entity_id") == target_id
                and (_strict_int(event.get("sim_time_s")) or -1) >= handoff_time
            ),
            None,
        )
        if (
            len(entry_ids) == 0
            or active_ping is None
            or not detections
            or not estimates
            or not exits
            or replacement is None
        ):
            violations.append(
                f"incomplete_uuv_tracking_chain:{region_id}"
            )
            continue
        if blue_response is None:
            violations.append(f"missing_uuv_blue_response:{region_id}")
        chain = UUVTrackingEvidenceChain(
            target_id=target_id,
            region_id=region_id,
            uuv_ids=uuv_ids,
            boundary_entry_event_ids=entry_ids,
            active_ping_event_id=str(active_ping["event_id"]),
            detection_event_ids=tuple(
                str(event["event_id"])
                for event in detections
                if isinstance(event.get("event_id"), str) and event.get("event_id")
            ),
            estimate_event_ids=tuple(
                str(event["event_id"])
                for event in estimates
                if isinstance(event.get("event_id"), str) and event.get("event_id")
            ),
            handoff_event_id=handoff_id,
            boundary_exit_event_ids=tuple(
                str(event["event_id"])
                for event in exits
                if isinstance(event.get("event_id"), str) and event.get("event_id")
            ),
            boundary_replacement_event_id=str(replacement["event_id"]),
            replacement_uuv_id=str(replacement["replacement_uuv_id"]),
            blue_response_event_id=(
                str(blue_response["event_id"])
                if isinstance(blue_response, Mapping)
                and isinstance(blue_response.get("event_id"), str)
                else None
            ),
            plan_version=plan_version,
        )
        chains.append(chain)
    if not chains and not violations:
        violations.append("missing_uuv_tracking_evidence_chain")
    return chains, list(dict.fromkeys(violations))


def _event_sort_key(event: Mapping[str, object]) -> tuple[int, str]:
    return (
        _strict_int(event.get("sim_time_s")) or 0,
        str(event.get("event_id", "")),
    )


def _event_region_id(event: Mapping[str, object]) -> str | None:
    region_id = event.get("region_id")
    if isinstance(region_id, str) and region_id:
        return region_id
    payload = event.get("payload")
    if isinstance(payload, Mapping):
        region_id = payload.get("region_id")
        if isinstance(region_id, str) and region_id:
            return region_id
    reason = event.get("reason")
    if isinstance(reason, str) and reason.startswith("region:"):
        return reason.removeprefix("region:")
    return None


def _event_time_between(
    event: Mapping[str, object],
    lower_time: int | None,
    upper_time: int | None,
) -> bool:
    event_time = _strict_int(event.get("sim_time_s"))
    return (
        event_time is not None
        and (lower_time is None or event_time >= lower_time)
        and (upper_time is None or event_time <= upper_time)
    )


def _event_platform_ids(event: Mapping[str, object]) -> frozenset[str]:
    ids: set[str] = set()
    for field in ("platform_id", "uuv_id", "emitter_id", "source_uuv_id"):
        value = event.get(field)
        if isinstance(value, str) and value:
            ids.add(value)
    for field in ("platform_ids", "uuv_ids", "source_uuv_ids"):
        ids.update(_string_tuple(event.get(field)))
    payload = event.get("payload")
    if isinstance(payload, Mapping):
        for field in ("platform_id", "uuv_id", "emitter_id", "source_uuv_id"):
            value = payload.get(field)
            if isinstance(value, str) and value:
                ids.add(value)
        for field in ("platform_ids", "uuv_ids", "source_uuv_ids"):
            ids.update(_string_tuple(payload.get(field)))
    return frozenset(ids)


def _first_uuv_event(
    events: object,
    *,
    event_types: set[str],
    target_id: str,
    uuv_ids: tuple[str, ...],
    lower_time: int | None,
    upper_time: int | None,
) -> Mapping[str, object] | None:
    candidates = (
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("event_type") in event_types
        and event.get("entity_id") == target_id
        and _event_platform_ids(event) & set(uuv_ids)
        and _event_time_between(event, lower_time, upper_time)
        and isinstance(event.get("event_id"), str)
        and event.get("event_id")
    )
    return min(candidates, key=_event_sort_key, default=None)


def _blue_tracking_chains(
    value: object,
) -> tuple[list[BlueTrackingEvidenceChain], list[str]]:
    """Resolve one physical UUV-bound lifecycle from dispatch to return."""
    if not isinstance(value, Mapping):
        return [], ["missing_blue_tracking_evidence_chain"]
    events = _unique_mappings(value.get("events"), "event_id")
    if not events:
        return [], ["missing_blue_tracking_evidence_chain"]

    dispatches = tuple(
        event
        for event in events.values()
        if event.get("event_type") == "carrier_dispatch_completed"
    )
    chains: list[BlueTrackingEvidenceChain] = []
    incomplete_candidates: set[str] = set()
    for dispatch in sorted(
        dispatches,
        key=lambda event: (
            _strict_int(event.get("sim_time_s")) or 0,
            str(event.get("event_id", "")),
        ),
    ):
        dispatch_id = dispatch.get("event_id")
        carrier_id = dispatch.get("entity_id")
        candidate_id = dispatch.get("candidate_id")
        dispatch_time = _strict_int(dispatch.get("sim_time_s"))
        dispatched_uuv_ids = _string_tuple(dispatch.get("uuv_ids"))
        if (
            not isinstance(dispatch_id, str)
            or not dispatch_id
            or not isinstance(carrier_id, str)
            or not carrier_id
            or not isinstance(candidate_id, str)
            or not candidate_id
            or dispatch_time is None
            or not dispatched_uuv_ids
        ):
            continue
        deployment_events = tuple(
            event
            for event in events.values()
            if event.get("event_type") == "uuv_deployed"
            and event.get("entity_id") in dispatched_uuv_ids
            and _strict_int(event.get("sim_time_s")) is not None
            and (_strict_int(event.get("sim_time_s")) or -1) >= dispatch_time
            and (
                event.get("candidate_id") == candidate_id
                or event.get("reason") == f"deploy:{candidate_id}"
            )
        )
        deployed_uuv_ids = tuple(
            sorted({str(event["entity_id"]) for event in deployment_events})
        )
        if set(deployed_uuv_ids) != set(dispatched_uuv_ids):
            incomplete_candidates.add(candidate_id)
            continue
        deployment_end_time = max(
            _strict_int(event.get("sim_time_s")) or dispatch_time
            for event in deployment_events
        )
        ping_candidates = tuple(
            event
            for event in events.values()
            if event.get("event_type") == "active_ping"
            and isinstance(event.get("entity_id"), str)
            and _strict_int(event.get("sim_time_s")) is not None
            and (_strict_int(event.get("sim_time_s")) or -1) >= deployment_end_time
            and bool(
                (
                    set(_string_tuple(event.get("uuv_ids")))
                    | {
                        str(event[field])
                        for field in ("uuv_id", "emitter_id")
                        if isinstance(event.get(field), str) and event.get(field)
                    }
                )
                & set(dispatched_uuv_ids)
            )
        )
        for ping in sorted(
            ping_candidates,
            key=lambda event: (
                _strict_int(event.get("sim_time_s")) or 0,
                str(event.get("event_id", "")),
            ),
        ):
            target_id = ping.get("entity_id")
            ping_id = ping.get("event_id")
            ping_time = _strict_int(ping.get("sim_time_s"))
            if (
                not isinstance(target_id, str)
                or not target_id
                or not isinstance(ping_id, str)
                or not ping_id
                or ping_time is None
            ):
                continue
            estimate_events = tuple(
                event
                for event in events.values()
                if event.get("event_type")
                in {
                    "target_estimate_updated",
                    "target_maneuver_observed",
                    "target_speed_regime_changed",
                    "observability_feedback",
                }
                and event.get("entity_id") == target_id
                and (_strict_int(event.get("sim_time_s")) or -1) >= ping_time
                and bool(_string_tuple(event.get("source_observation_ids")))
            )
            if not estimate_events:
                continue
            estimate_events = tuple(
                sorted(
                    estimate_events,
                    key=lambda event: (
                        _strict_int(event.get("sim_time_s")) or 0,
                        str(event.get("event_id", "")),
                    ),
                )[:8]
            )
            estimate_end_time = max(
                _strict_int(event.get("sim_time_s")) or ping_time
                for event in estimate_events
            )
            handoff = next(
                (
                    event
                    for event in sorted(
                        events.values(),
                        key=lambda item: (
                            _strict_int(item.get("sim_time_s")) or 0,
                            str(item.get("event_id", "")),
                        ),
                    )
                    if event.get("event_type") == "handoff_completed"
                    and (_strict_int(event.get("sim_time_s")) or -1) >= estimate_end_time
                    and (
                        event.get("predecessor_region_id") == candidate_id
                        or event.get("entity_id") == candidate_id
                    )
                    and event.get("target_id") == target_id
                    and _string_tuple(event.get("predecessor_uuv_ids"))
                    and set(_string_tuple(event.get("predecessor_uuv_ids")))
                    & set(dispatched_uuv_ids)
                    and (_strict_int(event.get("plan_version")) or 0) >= 1
                ),
                None,
            )
            if handoff is None:
                continue
            handoff_time = _strict_int(handoff.get("sim_time_s")) or estimate_end_time
            predecessor_uuv_ids = set(
                _string_tuple(handoff.get("predecessor_uuv_ids"))
            )
            resource = next(
                (
                    event
                    for event in sorted(
                        events.values(),
                        key=lambda item: (
                            _strict_int(item.get("sim_time_s")) or 0,
                            str(item.get("event_id", "")),
                        ),
                    )
                    if event.get("event_type")
                    in {
                        "endurance_threshold_crossed",
                        "battery_rotation",
                        "uuv_range_exhausted",
                        "uuv_energy_depleted",
                    }
                    and (_strict_int(event.get("sim_time_s")) or -1) >= handoff_time
                    and str(event.get("entity_id", "")) in predecessor_uuv_ids
                ),
                None,
            )
            if resource is None:
                continue
            resource_time = _strict_int(resource.get("sim_time_s")) or handoff_time
            resource_uuv_id = str(resource.get("entity_id"))
            recovery = next(
                (
                    event
                    for event in sorted(
                        events.values(),
                        key=lambda item: (
                            _strict_int(item.get("sim_time_s")) or 0,
                            str(item.get("event_id", "")),
                        ),
                    )
                    if event.get("event_type") == "uuv_recovery_requested"
                    and event.get("entity_id") == resource_uuv_id
                    and (_strict_int(event.get("sim_time_s")) or -1) >= resource_time
                ),
                None,
            )
            if recovery is None:
                continue
            recovery_time = _strict_int(recovery.get("sim_time_s")) or resource_time
            recovered = next(
                (
                    event
                    for event in sorted(
                        events.values(),
                        key=lambda item: (
                            _strict_int(item.get("sim_time_s")) or 0,
                            str(item.get("event_id", "")),
                        ),
                    )
                    if event.get("event_type") == "uuv_recovered"
                    and event.get("entity_id") == resource_uuv_id
                    and (_strict_int(event.get("sim_time_s")) or -1) >= recovery_time
                ),
                None,
            )
            if recovered is None:
                continue
            recovered_time = _strict_int(recovered.get("sim_time_s")) or recovery_time
            carrier_return = next(
                (
                    event
                    for event in sorted(
                        events.values(),
                        key=lambda item: (
                            _strict_int(item.get("sim_time_s")) or 0,
                            str(item.get("event_id", "")),
                        ),
                    )
                    if event.get("event_type") == "carrier_returned_to_fleet"
                    and event.get("entity_id") == carrier_id
                    and (_strict_int(event.get("sim_time_s")) or -1) >= recovered_time
                    and resource_uuv_id in set(_string_tuple(event.get("sortie_uuv_ids")))
                ),
                None,
            )
            if carrier_return is None:
                continue
            chains.append(
                BlueTrackingEvidenceChain(
                    target_id=target_id,
                    carrier_id=carrier_id,
                    candidate_id=candidate_id,
                    uuv_ids=dispatched_uuv_ids,
                    dispatch_event_id=dispatch_id,
                    deployment_event_ids=tuple(
                        sorted(str(event["event_id"]) for event in deployment_events)
                    ),
                    active_ping_event_id=ping_id,
                    estimate_event_ids=tuple(
                        str(event["event_id"]) for event in estimate_events
                    ),
                    handoff_event_id=str(handoff["event_id"]),
                    resource_event_id=str(resource["event_id"]),
                    recovery_request_event_id=str(recovery["event_id"]),
                    recovered_event_id=str(recovered["event_id"]),
                    carrier_return_event_id=str(carrier_return["event_id"]),
                    plan_version=_strict_int(handoff.get("plan_version")) or 0,
                )
            )
            break
        if not any(chain.dispatch_event_id == dispatch_id for chain in chains):
            incomplete_candidates.add(candidate_id)

    violations: list[str] = []
    if not chains:
        violations.append("missing_blue_tracking_evidence_chain")
    if incomplete_candidates:
        violations.append(
            "incomplete_blue_tracking_candidates:" + ",".join(sorted(incomplete_candidates))
        )
    return chains, list(dict.fromkeys(violations))


_NON_REAL_MODEL_MARKERS = (
    "dummy",
    "fake",
    "fixed-seed",
    "heuristic",
    "mock",
    "scripted",
    "stub",
    "test",
    "test-model",
)


def _real_provider_attestation_violations(value: object) -> tuple[str, ...]:
    required_roles = {"master", "slave", "adversary"}
    if not isinstance(value, Mapping):
        return ("real_provider_attestation_unavailable",)
    raw_attestations = value.get("provider_attestations")
    if not isinstance(raw_attestations, (list, tuple)):
        return ("real_provider_attestation_unavailable",)
    attested_roles = {
        str(item.get("role"))
        for item in raw_attestations
        if isinstance(item, Mapping)
        and item.get("attested") is True
        and item.get("transport") == "httpx"
        and item.get("client_type")
        == "underwater_tracking.agent.llm_factory.RoleHTTPStructuredLLM"
        and isinstance(item.get("configured_endpoint"), str)
        and str(item.get("configured_endpoint")).startswith(("http://", "https://"))
    }
    missing = sorted(required_roles - attested_roles)
    violations = (
        ["real_provider_not_attested:" + ",".join(missing)] if missing else []
    )
    attestations_by_role = {
        str(item.get("role")): item
        for item in raw_attestations
        if isinstance(item, Mapping)
    }
    raw_calls = value.get("llm_calls", ())
    calls = (
        tuple(item for item in raw_calls if isinstance(item, Mapping))
        if isinstance(raw_calls, (list, tuple))
        else ()
    )
    operations_by_role = {
        "master": frozenset({"intent", "regional_strategy", "strategy"}),
        "slave": frozenset({"slave_sonar_decision"}),
        "adversary": frozenset({"adversary_mission_decision"}),
    }
    missing_success: list[str] = []
    for role in sorted(attested_roles):
        attestation = attestations_by_role.get(role)
        if attestation is not None and attestation.get("probe_successful") is True:
            continue
        model = attestation.get("model") if attestation is not None else None
        if not isinstance(model, str) or not model:
            missing_success.append(role)
            continue
        if not any(
            call.get("operation") in operations_by_role[role]
            and call.get("model") == model
            and isinstance(call.get("prompt_version"), str)
            and bool(call.get("prompt_version"))
            and isinstance(call.get("request_hash"), str)
            and bool(call.get("request_hash"))
            and isinstance(call.get("response_hash"), str)
            and bool(call.get("response_hash"))
            and call.get("error_category") == ""
            for call in calls
        ):
            missing_success.append(role)
    if missing_success:
        violations.append(
            "real_provider_no_successful_call:" + ",".join(missing_success)
        )
    return tuple(violations)


def _prediction_intent_chains(
    value: object,
) -> tuple[list[PredictionIntentEvidenceChain], list[str]]:
    """Resolve complete forecast-divergence chains through durable IDs."""
    if not isinstance(value, Mapping):
        return [], []
    events = _unique_mappings(value.get("events"), "event_id")
    diffs = _unique_mappings(value.get("prediction_diffs"), "diff_id")
    llm_calls = _unique_mappings(value.get("llm_calls"), "call_id")
    decisions = _unique_mappings(value.get("decisions"), "decision_id")
    plans = _unique_mappings(value.get("committed_plans"), "plan_id")
    suspicion_events = tuple(
        event
        for event in events.values()
        if event.get("event_type") == "target_intent_change_suspected"
    )
    sensor_maneuvers = tuple(
        event
        for event in events.values()
        if event.get("event_type")
        in {"target_maneuver_observed", "target_speed_regime_changed"}
    )
    if not suspicion_events:
        return (
            [],
            ["missing_prediction_diff"]
            if sensor_maneuvers
            else ["missing_prediction_intent_evidence_chain"],
        )

    chains: list[PredictionIntentEvidenceChain] = []
    violations: list[str] = []
    for suspicion in sorted(
        suspicion_events,
        key=lambda event: (
            _strict_int(event.get("sim_time_s")) or 0,
            str(event.get("event_id", "")),
        ),
    ):
        target_id = suspicion.get("entity_id")
        suspicion_id = suspicion.get("event_id")
        suspicion_time = _strict_int(suspicion.get("sim_time_s"))
        suspicion_payload = suspicion.get("payload")
        if (
            not isinstance(target_id, str)
            or not target_id
            or not isinstance(suspicion_id, str)
            or suspicion_time is None
            or not isinstance(suspicion_payload, Mapping)
        ):
            violations.append("missing_prediction_diff")
            continue
        diff_id = suspicion_payload.get("diff_id")
        diff = diffs.get(diff_id) if isinstance(diff_id, str) else None
        if diff is None or diff.get("target_id") != target_id:
            violations.append("missing_prediction_diff")
            continue
        diff_values = _validated_diff_values(diff)
        if diff_values is None:
            violations.append("missing_prediction_diff")
            continue
        if diff_values["exceeded"] is not True:
            violations.append("prediction_diff_below_threshold")
            continue

        confirmation = next(
            (
                event
                for event in events.values()
                if event.get("event_type") == "target_intent_changed"
                and event.get("entity_id") == target_id
                and isinstance(event.get("payload"), Mapping)
                and event["payload"].get("suspicion_event_id") == suspicion_id
                and event["payload"].get("diff_id") == diff_id
            ),
            None,
        )
        confirmation_time = (
            _strict_int(confirmation.get("sim_time_s"))
            if confirmation is not None
            else None
        )
        confirmation_payload = (
            confirmation.get("payload") if confirmation is not None else None
        )
        if (
            confirmation is None
            or confirmation_time is None
            or confirmation_time < suspicion_time
            or not isinstance(confirmation_payload, Mapping)
            or confirmation_payload.get("source") != "real_intent_llm"
        ):
            violations.append("missing_intent_confirmation")
            continue

        raw_call_ids = confirmation_payload.get("intent_llm_call_ids")
        call_ids = _string_tuple(raw_call_ids)
        qualifying_calls = tuple(llm_calls.get(call_id) for call_id in call_ids)
        raw_call_refs = confirmation_payload.get("intent_llm_calls")
        call_refs = (
            tuple(raw_call_refs)
            if isinstance(raw_call_refs, (list, tuple))
            else ()
        )
        if (
            len(call_ids) != 2
            or len(set(call_ids)) != 2
            or len(call_refs) != 2
            or any(not isinstance(call_ref, Mapping) for call_ref in call_refs)
            or any(call is None for call in qualifying_calls)
            or any(
                _intent_call_ref(call_ref) != _intent_call_ref(call)
                for call_ref, call in zip(call_refs, qualifying_calls, strict=True)
                if isinstance(call_ref, Mapping) and call is not None
            )
            or any(
                not _qualifying_real_intent_call(
                    call,
                    suspicion_time=suspicion_time,
                    confirmation_time=confirmation_time,
                )
                for call in qualifying_calls
                if call is not None
            )
        ):
            violations.append("missing_real_intent_provider")
            continue

        confirmed_event_id = confirmation.get("event_id")
        if not isinstance(confirmed_event_id, str) or not confirmed_event_id:
            violations.append("missing_intent_confirmation")
            continue
        decision = next(
            (
                item
                for item in decisions.values()
                if confirmed_event_id in _string_tuple(item.get("trigger_event_ids"))
                and (_strict_int(item.get("sim_time_s")) or -1) >= confirmation_time
            ),
            None,
        )
        if decision is None:
            violations.append("missing_regional_replan")
            continue
        plan_id = decision.get("final_plan_id")
        plan = plans.get(plan_id) if isinstance(plan_id, str) else None
        plan_revision = _strict_int(plan.get("revision")) if plan is not None else None
        if (
            plan is None
            or plan_revision is None
            or plan_revision < 1
            or plan.get("status")
            not in {"active", "degraded", "superseded", "completed"}
            or target_id not in _string_tuple(plan.get("target_ids"))
            or confirmed_event_id not in _string_tuple(plan.get("trigger_event_ids"))
        ):
            violations.append("missing_committed_plan")
            continue

        response_events = tuple(
            event
            for event in events.values()
            if event.get("event_type") == "state_changed"
            and event.get("entity_id") == target_id
            and (_strict_int(event.get("sim_time_s")) or -1) > confirmation_time
            and isinstance(event.get("payload"), Mapping)
            and event["payload"].get("phase") == "plan_applied"
            and _strict_int(event["payload"].get("plan_revision")) == plan_revision
            and event.get("event_id")
            == f"mission-plan:{plan_revision}:{target_id}:applied"
            and isinstance(event["payload"].get("region_id"), str)
            and bool(event["payload"].get("region_id"))
            and bool(_string_tuple(event["payload"].get("member_ids")))
        )
        if not response_events:
            violations.append("missing_blue_response")
            continue
        response_time = min(
            _strict_int(event.get("sim_time_s")) or confirmation_time
            for event in response_events
        )
        public_observations = _unique_mappings(
            value.get("public_observations"), "observation_id"
        )
        bound_estimates: list[Mapping[str, object]] = []
        for event in events.values():
            event_time = _strict_int(event.get("sim_time_s"))
            source_observation_ids = _string_tuple(
                event.get("source_observation_ids")
            )
            if (
                event.get("event_type")
                not in {
                    "target_estimate_updated",
                    "target_maneuver_observed",
                    "target_speed_regime_changed",
                    "observability_feedback",
                }
                or event.get("entity_id") != target_id
                or event_time is None
                or event_time < suspicion_time
                or event_time > response_time
                or not source_observation_ids
            ):
                continue
            if public_observations:
                source_observations = tuple(
                    public_observations.get(observation_id)
                    for observation_id in source_observation_ids
                )
                if any(
                    observation is None
                    or observation.get("target_id") != target_id
                    for observation in source_observations
                ):
                    continue
                response_members: set[str] = set()
                for response_event in response_events:
                    response_payload = response_event.get("payload")
                    if isinstance(response_payload, Mapping):
                        response_members.update(
                            _string_tuple(response_payload.get("member_ids"))
                        )
                observer_ids = {
                    str(observation.get("observer_id"))
                    for observation in source_observations
                    if isinstance(observation, Mapping)
                    and isinstance(observation.get("observer_id"), str)
                }
                if response_members and not response_members.intersection(observer_ids):
                    continue
            bound_estimates.append(event)
        if not bound_estimates:
            violations.append("missing_blue_response_estimate_binding")
            continue
        response_ids = tuple(
            sorted(str(event["event_id"]) for event in response_events)
        )
        first_response_time = response_time
        calls = tuple(call for call in qualifying_calls if call is not None)
        chains.append(
            PredictionIntentEvidenceChain(
                target_id=target_id,
                diff_id=diff_id,
                previous_prediction_id=diff_values["previous_prediction_id"],
                current_prediction_id=diff_values["current_prediction_id"],
                absolute_rms_m=diff_values["absolute_rms_m"],
                normalized_rms=diff_values["normalized_rms"],
                absolute_floor_m=diff_values["absolute_floor_m"],
                normalized_threshold=diff_values["normalized_threshold"],
                overlap_start_s=diff_values["overlap_start_s"],
                overlap_end_s=diff_values["overlap_end_s"],
                suspicion_event_id=suspicion_id,
                suspicion_sim_time_s=suspicion_time,
                intent_llm_call_ids=call_ids,
                intent_provider_models=tuple(
                    dict.fromkeys(str(call["model"]) for call in calls)
                ),
                confirmed_event_id=confirmed_event_id,
                confirmation_sim_time_s=confirmation_time,
                resulting_plan_id=str(plan_id),
                resulting_plan_revision=plan_revision,
                blue_response_event_ids=response_ids,
                response_latency_s=first_response_time - suspicion_time,
            )
        )
    return chains, list(dict.fromkeys(violations))


def _unique_mappings(value: object, id_field: str) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, (list, tuple)):
        return {}
    result: dict[str, Mapping[str, object]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        item_id = item.get(id_field)
        if isinstance(item_id, str) and item_id and item_id not in result:
            result[item_id] = item
    return result


def _strict_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    if any(not isinstance(item, str) or not item for item in value):
        return ()
    return tuple(value)


def _validated_diff_values(diff: Mapping[str, object]) -> dict[str, object] | None:
    strings = ("previous_prediction_id", "current_prediction_id")
    numbers = (
        "absolute_rms_m",
        "normalized_rms",
        "absolute_floor_m",
        "normalized_threshold",
        "overlap_start_s",
        "overlap_end_s",
    )
    if not isinstance(diff.get("exceeded"), bool):
        return None
    if any(not isinstance(diff.get(field), str) or not diff.get(field) for field in strings):
        return None
    if any(
        not isinstance(diff.get(field), (int, float))
        or isinstance(diff.get(field), bool)
        or not isfinite(float(diff[field]))
        for field in numbers
    ):
        return None
    values = {field: diff[field] for field in (*strings, *numbers)}
    values["exceeded"] = diff["exceeded"]
    if (
        float(values["absolute_rms_m"]) < 0
        or float(values["normalized_rms"]) < 0
        or float(values["absolute_floor_m"]) <= 0
        or float(values["normalized_threshold"]) <= 0
        or float(values["overlap_start_s"]) < 0
        or float(values["overlap_end_s"]) < float(values["overlap_start_s"])
    ):
        return None
    return values


def _intent_call_ref(value: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        value.get(field)
        for field in (
            "operation",
            "model",
            "prompt_version",
            "request_hash",
            "response_hash",
            "sim_time_s",
        )
    )


def _qualifying_real_intent_call(
    call: Mapping[str, object],
    *,
    suspicion_time: int,
    confirmation_time: int,
) -> bool:
    model = call.get("model")
    sim_time_s = _strict_int(call.get("sim_time_s"))
    return bool(
        call.get("operation") == "intent"
        and isinstance(model, str)
        and model
        and not any(marker in model.lower() for marker in _NON_REAL_MODEL_MARKERS)
        and isinstance(call.get("prompt_version"), str)
        and call.get("prompt_version")
        and isinstance(call.get("request_hash"), str)
        and call.get("request_hash")
        and isinstance(call.get("response_hash"), str)
        and call.get("response_hash")
        and call.get("error_category") == ""
        and sim_time_s is not None
        and suspicion_time <= sim_time_s <= confirmation_time
    )


def _record_websocket_frame(
    payload: object,
    frames: list[Mapping[str, object]],
    contract_errors: list[str],
) -> None:
    """Keep full operational WebSocket frames for cross-transport auditing."""
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError:
            return
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return
    if not isinstance(payload, Mapping) or "frame_id" not in payload:
        return
    frame = dict(payload)
    previous_frame_id = (
        frames[-1].get("frame_id")
        if frames and isinstance(frames[-1].get("frame_id"), int)
        else None
    )
    frames.append(frame)
    if frame.get("uuv_only") is True and (
        isinstance(frame.get("execution"), Mapping)
        or _numeric_sim_time(frame.get("sim_time_s")) > 0
    ):
        contract_errors.extend(
            validate_uuv_only_frame(frame, previous_frame_id=previous_frame_id)
        )


def _browser_audit(
    ui_url: str,
    screenshot_dir: Path,
    *,
    stop_event: Event | None = None,
) -> tuple[int, int, tuple[str, ...], tuple[str, ...]]:
    stop_event = stop_event or Event()
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return 1, 1, (), ("playwright_import_failed",)
    console_errors = 0
    failed_requests = 0
    websocket_seen = 0
    screenshot_paths: set[str] = set()
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except Exception:  # noqa: BLE001 - missing browser binary is a gate failure
            return 1, 1, (), ("browser_launch_failed",)
        try:
            page_states: list[dict[str, object]] = []
            for label, width, height in (("desktop", 1440, 900), ("mobile", 390, 844)):
                page = browser.new_page(viewport={"width": width, "height": height})
                console_messages: list[object] = []
                page.on(
                    "console",
                    lambda message, messages=console_messages: messages.append(message)
                    if getattr(message, "type", "") == "error"
                    else None,
                )
                page_errors: list[object] = []
                request_errors: list[object] = []
                page.on(
                    "pageerror",
                    lambda error, errors=page_errors: errors.append(error),
                )
                page.on(
                    "requestfailed",
                    lambda request, errors=request_errors: errors.append(request)
                    if not stop_event.is_set()
                    else None,
                )
                websocket_counts = {"ws_error": 0, "ws_close": 0, "ws_normal_close": 0}
                websocket_errors: list[str] = []
                websocket_frames: list[Mapping[str, object]] = []
                websocket_contract_errors: list[str] = []

                def increment_websocket_count(
                    key: str,
                    *,
                    counts: dict[str, int],
                    detail: object | None = None,
                ) -> None:
                    if key == "ws_error" and stop_event.is_set():
                        return
                    if key == "ws_error":
                        websocket_errors.append(str(detail))
                    counts[key] += 1

                def websocket_opened(
                    websocket: object,
                    *,
                    counts: dict[str, int] = websocket_counts,
                    frames: list[Mapping[str, object]] = websocket_frames,
                    contract_errors: list[str] = websocket_contract_errors,
                ) -> None:
                    nonlocal websocket_seen
                    websocket_seen += 1
                    on_error = getattr(websocket, "on", None)
                    if callable(on_error):
                        on_error(
                            "framereceived",
                            lambda payload, frames=frames, contract_errors=contract_errors: _record_websocket_frame(
                                payload, frames, contract_errors
                            ),
                        )
                        on_error(
                            "socketerror",
                            lambda error, counts=counts: increment_websocket_count(
                                "ws_error", counts=counts, detail=error
                            ),
                        )
                        on_error(
                            "close",
                            lambda *_args: increment_websocket_count(
                                "ws_close" if not stop_event.is_set() else "ws_normal_close",
                                counts=counts,
                            ),
                        )

                page.on("websocket", websocket_opened)
                state: dict[str, object] = {
                    "label": label,
                    "page": page,
                    "console_messages": console_messages,
                    "page_errors": page_errors,
                    "request_errors": request_errors,
                    "websocket_counts": websocket_counts,
                    "websocket_errors": websocket_errors,
                    "websocket_frames": websocket_frames,
                    "websocket_contract_errors": websocket_contract_errors,
                    "captured_stages": set(),
                    "last_memory_probe_monotonic": 0.0,
                }
                page_states.append(state)
                navigation_ready = False
                navigation_deadline = time.monotonic() + 10.0
                while time.monotonic() < navigation_deadline:
                    try:
                        page.goto(ui_url, wait_until="domcontentloaded", timeout=2_000)
                        navigation_ready = True
                        break
                    except Exception:
                        time.sleep(0.25)
                if navigation_ready:
                    # Connection-refused attempts during service boot are not
                    # application request failures once the UI is reachable.
                    console_messages.clear()
                    page_errors.clear()
                    request_errors.clear()
                    _exercise_ui(page, page_errors)
                    screenshot_name = f"{label}.png"
                    page.screenshot(path=str(screenshot_dir / screenshot_name), full_page=True)
                    screenshot_paths.add(f"screenshots/{screenshot_name}")
                else:
                    page_errors.append("navigation")
            while not stop_event.wait(0.5):
                for state in page_states:
                    label = str(state["label"])
                    page = state["page"]
                    assert hasattr(page, "screenshot")
                    try:
                        latest_name = f"{label}-latest.png"
                        page.screenshot(path=str(screenshot_dir / latest_name), full_page=True)
                        screenshot_paths.add(f"screenshots/{latest_name}")
                    except Exception:
                        cast(list[object], state["page_errors"]).append("screenshot")
                    now = time.monotonic()
                    probe_memory = (
                        now
                        - float(state["last_memory_probe_monotonic"])
                        >= 5.0
                    )
                    if probe_memory:
                        state["last_memory_probe_monotonic"] = now
                    observed_stages = _probe_ui_consistency(
                        page,
                        ui_url,
                        cast(list[object], state["page_errors"]),
                        cast(list[object], state["request_errors"]),
                        probe_memory=probe_memory,
                        websocket_frames=cast(
                            list[Mapping[str, object]], state["websocket_frames"]
                        ),
                    )
                    captured_stages = cast(set[str], state["captured_stages"])
                    for stage_id in sorted(observed_stages - captured_stages):
                        stage_name = f"{label}-stage-{stage_id}.png"
                        try:
                            page.screenshot(
                                path=str(screenshot_dir / stage_name),
                                full_page=True,
                            )
                            screenshot_paths.add(f"screenshots/{stage_name}")
                            captured_stages.add(stage_id)
                        except Exception:
                            cast(list[object], state["page_errors"]).append(
                                f"stage_screenshot:{stage_id}"
                            )
            for state in page_states:
                cast(object, state["page"]).close()
        finally:
            browser.close()
    browser_error_details: list[str] = []
    for state in page_states:
        label = str(state["label"])
        console_messages = cast(list[object], state["console_messages"])
        page_errors = cast(list[object], state["page_errors"])
        request_errors = cast(list[object], state["request_errors"])
        page_errors.extend(cast(list[str], state["websocket_contract_errors"]))
        console_errors += len(console_messages)
        console_errors += len(page_errors)
        failed_requests += len(request_errors)
        browser_error_details.extend(
            f"{label}:console:{_browser_error_detail(item)}"
            for item in console_messages
        )
        browser_error_details.extend(
            f"{label}:pageerror:{_browser_error_detail(item)}"
            for item in page_errors
        )
        browser_error_details.extend(
            f"{label}:requestfailed:{_browser_error_detail(item)}"
            for item in request_errors
        )
        # A close is normal during React dev-mode remounts and page cleanup;
        # only a socket error is a failed browser request.
        for detail in cast(list[str], state["websocket_errors"]):
            if (
                stop_event.is_set()
                and detail == "WebSocket is closed before the connection is established."
            ):
                continue
            failed_requests += 1
            browser_error_details.append(f"{label}:websocket:{detail}")
    if websocket_seen == 0:
        failed_requests += 1
        browser_error_details.append("websocket:no_connection_observed")
    return (
        console_errors,
        failed_requests,
        tuple(sorted(screenshot_paths)),
        tuple(browser_error_details[:100]),
    )


def _browser_error_detail(value: object) -> str:
    """Extract a bounded diagnostic without retaining Playwright objects."""
    if isinstance(value, str):
        return value[:500]
    text = getattr(value, "text", None)
    location = getattr(value, "location", None)
    if callable(location):
        try:
            location = location()
        except Exception:  # noqa: BLE001 - diagnostics must not affect the gate
            location = None
    location_url = location.get("url") if isinstance(location, Mapping) else None
    if isinstance(text, str) and text:
        if location_url:
            return f"{text} (url={location_url})"[:500]
        return text[:500]
    message = getattr(value, "message", None)
    if isinstance(message, str) and message:
        return message[:500]
    url = getattr(value, "url", None)
    failure = getattr(value, "failure", None)
    if callable(failure):
        try:
            failure = failure()
        except Exception:  # noqa: BLE001 - diagnostics must not affect the gate
            failure = None
    if url or failure:
        return f"url={url!s} failure={failure!s}"[:500]
    return str(value)[:500]


def _exercise_ui(page: object, page_errors: list[object]) -> None:
    """Open the operator surfaces so hidden tabs are part of the live audit."""
    try:
        toggle = page.locator('button[aria-label="切换任务详情"]')
        if not toggle.count():
            page_errors.append("missing_ui_surface:drawer_toggle")
        else:
            toggle.click(timeout=1_000)
        for label in ("时间线", "决策台账", "LLM 思考过程", "Memory Steam"):
            tab = page.get_by_role("tab", name=label)
            if not tab.count():
                page_errors.append(f"missing_ui_tab:{label}")
                continue
            tab.click(timeout=1_000)
            if not page.locator("#mission-panel").count():
                page_errors.append(f"missing_tab_panel:{label}")
    except Exception as exc:  # noqa: BLE001 - browser state is an acceptance gate
        page_errors.append(f"ui_surface_probe:{type(exc).__name__}")


def _read_ui_snapshot(
    page: object,
    ui_url: str,
    page_errors: list[object],
    request_errors: list[object],
) -> Mapping[str, object] | None:
    try:
        response = page.request.get(
            f"{ui_url.rstrip('/')}/api/operational/snapshot",
            timeout=5_000,
        )
    except Exception as exc:  # noqa: BLE001 - snapshot transport is an acceptance gate
        request_errors.append(f"snapshot_request:{type(exc).__name__}")
        return None
    if not response.ok:
        request_errors.append(f"snapshot_http_{response.status}")
        return None
    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - malformed backend JSON is an acceptance gate
        page_errors.append(f"snapshot_json:{type(exc).__name__}")
        return None
    if not isinstance(payload, Mapping):
        page_errors.append("snapshot_not_object")
        return None
    return payload


def _probe_ui_consistency(
    page: object,
    ui_url: str,
    page_errors: list[object],
    request_errors: list[object],
    *,
    probe_memory: bool = True,
    websocket_frames: list[Mapping[str, object]] | None = None,
) -> frozenset[str]:
    try:
        metrics = page.evaluate(
            "({width: window.innerWidth, documentWidth: document.documentElement.scrollWidth, "
            "bodyWidth: document.body ? document.body.scrollWidth : 0})"
        )
        if isinstance(metrics, Mapping):
            width = float(metrics.get("width", 0))
            if float(metrics.get("documentWidth", 0)) > width + 1 or float(
                metrics.get("bodyWidth", 0)
            ) > width + 1:
                page_errors.append("horizontal_layout_overflow")
        payload = _read_ui_snapshot(page, ui_url, page_errors, request_errors)
        if payload is None:
            return frozenset()
        if websocket_frames:
            http_frame_id = payload.get("frame_id")
            matching_websocket = next(
                (
                    frame
                    for frame in reversed(websocket_frames)
                    if frame.get("frame_id") == http_frame_id
                ),
                None,
            )
            if matching_websocket is not None:
                page_errors.extend(
                    "transport_" + item
                    for item in validate_transport_frame_consistency(
                        {"http": payload, "websocket": matching_websocket}
                    )
                )
        run_phase = payload.get("run_phase")
        sim_time_s = payload.get("sim_time_s")
        if run_phase in {"running", "completed"} or (
            isinstance(sim_time_s, (int, float)) and sim_time_s > 0
        ):
            surface_counts: dict[str, int] = {}
            missing_surfaces: tuple[str, ...] = ()
            for attempt in range(3):
                surface_counts = {
                    "drawer_toggle": page.locator(
                        'button[aria-label="切换任务详情"]'
                    ).count(),
                    "mission_panel": page.locator("#mission-panel").count(),
                    "plan_version": page.locator("[data-plan-version]").count(),
                    "sim_time": page.locator(".playback-readout.time").count(),
                    "memory_window": page.locator(".memory-window").count(),
                    "brain_section": page.locator(".brain-section").count(),
                }
                missing_surfaces = _required_ui_surface_violations(surface_counts)
                if not missing_surfaces or attempt == 2:
                    break
                wait_for_timeout = getattr(page, "wait_for_timeout", None)
                if callable(wait_for_timeout):
                    wait_for_timeout(100)
            for violation in missing_surfaces:
                if violation not in page_errors:
                    page_errors.append(violation)
            if (
                probe_memory
                and run_phase in {"running", "completed"}
                and int(payload.get("plan_version", 0)) > 0
            ):
                _probe_ui_content(
                    page,
                    ui_url,
                    payload,
                    page_errors,
                    request_errors,
                )
        plan_node = page.locator("[data-plan-version]").first
        if plan_node.count():
            consistency_violations: tuple[str, ...] = ()
            for attempt in range(8):
                if attempt:
                    refreshed = _read_ui_snapshot(
                        page,
                        ui_url,
                        page_errors,
                        request_errors,
                    )
                    if refreshed is not None:
                        payload = refreshed
                try:
                    api_plan_version = int(payload.get("plan_version", 0))
                    dom_plan_version = int(
                        plan_node.get_attribute("data-plan-version") or ""
                    )
                except (TypeError, ValueError):
                    consistency_violations = ("ui_plan_version_invalid",)
                    break
                consistency_violations = _ui_consistency_violations(
                    dom_plan_version=dom_plan_version,
                    api_plan_version=api_plan_version,
                    dom_sim_time_s=_dom_sim_time(page),
                    api_sim_time_s=_numeric_sim_time(payload.get("sim_time_s")),
                )
                if not any(
                    item in {"ui_plan_version_stale", "ui_sim_time_stale"}
                    for item in consistency_violations
                ) or attempt == 7:
                    break
                wait_for_timeout = getattr(page, "wait_for_timeout", None)
                if callable(wait_for_timeout):
                    wait_for_timeout(250)
            page_errors.extend(consistency_violations)
        time_node = page.locator(".playback-readout.time").first
        if time_node.count():
            dom_time = time_node.text_content()
            if dom_time and not dom_time.strip().endswith("s"):
                page_errors.append("ui_sim_time_invalid")
            elif _dom_sim_time(page) is None:
                page_errors.append("ui_sim_time_invalid")
        return collect_stage_ids(payload)
    except Exception as exc:  # noqa: BLE001 - browser/API consistency is a gate
        page_errors.append(f"ui_consistency_probe:{type(exc).__name__}")
        return frozenset()


def _probe_ui_content(
    page: object,
    ui_url: str,
    payload: Mapping[str, object],
    page_errors: list[object],
    request_errors: list[object],
) -> None:
    """Verify that populated backend views render populated tab content."""
    requirements: list[tuple[str, str, str]] = []
    raw_timeline = payload.get("plan_timeline", ())
    raw_events = payload.get("events", ())
    if isinstance(raw_timeline, list) and raw_timeline:
        requirements.append(("时间线", ".plan-timeline", "task_timeline_content"))
    elif isinstance(raw_events, list) and raw_events:
        requirements.append(("时间线", ".timeline-list", "task_timeline_content"))
    thinking = payload.get("llm_thinking")
    if isinstance(thinking, str) and thinking.strip():
        requirements.append(
            ("LLM 思考过程", ".llm-thinking-flow", "llm_thinking_content")
        )
    memory_has_events = False
    try:
        response = page.request.get(
            f"{ui_url.rstrip('/')}/api/assistant/memory/stream"
            "?user_id=operator&conversation_id=verification&limit=32",
            timeout=5_000,
        )
        if not response.ok:
            request_errors.append(f"memory_ui_http_{response.status}")
        else:
            memory_payload = response.json()
            memory_events = (
                memory_payload.get("events", ())
                if isinstance(memory_payload, Mapping)
                else ()
            )
            memory_has_events = isinstance(memory_events, list) and bool(memory_events)
    except Exception as exc:  # noqa: BLE001 - browser/API content is an acceptance gate
        request_errors.append(f"memory_ui_probe:{type(exc).__name__}")
    if memory_has_events:
        requirements.append(
            ("Memory Steam", '[data-testid="memory-steam-event"]', "memory_event_content")
        )
    for label, selector, error_id in requirements:
        try:
            tab = page.get_by_role("tab", name=label, exact=True)
            if not tab.count():
                page_errors.append(f"missing_ui_tab:{label}")
                continue
            tab.click(timeout=5_000)
            page.locator("#mission-panel").wait_for(state="visible", timeout=5_000)
            page.locator(selector).first.wait_for(state="visible", timeout=5_000)
        except Exception as exc:  # noqa: BLE001 - browser content is an acceptance gate
            page_errors.append(f"missing_ui_content:{error_id}:{type(exc).__name__}")


def _required_ui_surface_violations(
    surface_counts: Mapping[str, int],
) -> tuple[str, ...]:
    """Return stable names for required command-center surfaces that are absent."""
    return tuple(
        f"missing_ui_surface:{surface}"
        for surface in (
            "drawer_toggle",
            "mission_panel",
            "plan_version",
            "sim_time",
            "memory_window",
            "brain_section",
        )
        if surface_counts.get(surface, 0) <= 0
    )


def _numeric_sim_time(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _dom_sim_time(page: object) -> int | None:
    try:
        node = page.locator(".playback-readout.time").first
        if not node.count():
            return None
        text = str(node.text_content() or "").strip()
        return int(text.removesuffix("s"))
    except (AttributeError, TypeError, ValueError):
        return None


def _ui_consistency_violations(
    *,
    dom_plan_version: int,
    api_plan_version: int,
    dom_sim_time_s: int | None,
    api_sim_time_s: int,
) -> tuple[str, ...]:
    """Validate a live UI frame against a nearby backend snapshot.

    The UI receives frames over WebSocket while this probe reads HTTP. Exact
    equality would flag normal transport/render scheduling as an error. A
    bounded drift still catches a frozen or cross-plan UI.
    """
    violations: list[str] = []
    if dom_plan_version < 0 or api_plan_version < 0:
        violations.append("ui_plan_version_invalid")
    elif abs(dom_plan_version - api_plan_version) > _UI_MAX_PLAN_VERSION_DRIFT:
        violations.append("ui_plan_version_stale")
    if api_sim_time_s < 0:
        violations.append("ui_sim_time_invalid")
    elif dom_sim_time_s is not None and dom_sim_time_s < 0:
        violations.append("ui_sim_time_invalid")
    elif dom_sim_time_s is not None and abs(dom_sim_time_s - api_sim_time_s) > _UI_MAX_SIM_TIME_DRIFT_S:
        violations.append("ui_sim_time_stale")
    return tuple(violations)


def _get_json(base_url: str, path: str) -> object:
    with urlopen(Request(base_url.rstrip("/") + path, headers={"Accept": "application/json"}), timeout=5.0) as response:
        return json.loads(response.read())


def _wait_for_api(base_url: str, *, timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            _get_json(base_url, "/api/health")
            snapshot = _get_json(base_url, "/api/operational/snapshot")
            if isinstance(snapshot, Mapping):
                return True
        except (HTTPException, OSError, ValueError):
            time.sleep(0.25)
    return False


def _current_run_output_dir(
    base_url: str,
    *,
    output_root: Path = ROOT / "outputs",
    timeout_s: float = 10.0,
) -> Path | None:
    """Resolve the active run directory so output-size checks ignore old runs."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            payload = _get_json(base_url, "/api/replay")
        except (OSError, ValueError):
            time.sleep(0.25)
            continue
        if isinstance(payload, Mapping):
            run_id = payload.get("run_id")
            if isinstance(run_id, str) and run_id and Path(run_id).name == run_id:
                candidate = output_root / run_id
                if candidate.is_dir():
                    return candidate
        time.sleep(0.25)
    return None


def _run_output_dirs(output_root: Path) -> set[Path]:
    if not output_root.is_dir():
        return set()
    return {
        path
        for path in output_root.iterdir()
        if path.is_dir() and path.name.startswith("run-")
    }


def _serve_output_dirs(output_root: Path) -> set[Path]:
    if not output_root.is_dir():
        return set()
    return {
        path
        for path in output_root.iterdir()
        if path.is_dir() and path.name.startswith("serve-")
    }


def _safe_get_json(base_url: str, path: str) -> tuple[object | None, bool]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return _get_json(base_url, path), False
        except (OSError, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.1)
    assert last_error is not None
    return None, True


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _free_port(excluded: set[int] | None = None) -> int:
    excluded = excluded or set()
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        if port not in excluded:
            return port


def _stop_process(process: subprocess.Popen[str]) -> bool:
    if process.poll() is not None:
        return False
    descendant_groups = (
        _descendant_process_groups(process.pid) if os.name == "posix" else frozenset()
    )
    timed_out = False
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    else:
        process.send_signal(signal.CTRL_BREAK_EVENT)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=10.0)
    if os.name == "posix":
        for group_id in descendant_groups:
            if group_id == process.pid:
                continue
            try:
                os.killpg(group_id, signal.SIGTERM)
            except ProcessLookupError:
                continue
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and descendant_groups:
            live_groups = set()
            for group_id in descendant_groups:
                try:
                    os.killpg(group_id, 0)
                except ProcessLookupError:
                    continue
                else:
                    live_groups.add(group_id)
            if not live_groups:
                break
            time.sleep(0.05)
            descendant_groups = frozenset(live_groups)
        for group_id in descendant_groups:
            try:
                os.killpg(group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return timed_out


def _descendant_process_groups(root_pid: int) -> frozenset[int]:
    """Find child process groups that outlive the main process session."""
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,pgid="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    children: dict[int, list[tuple[int, int]]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, parent_pid, group_id = (int(field) for field in fields)
        except ValueError:
            continue
        children.setdefault(parent_pid, []).append((pid, group_id))
    pending = [root_pid]
    descendants: set[int] = set()
    groups: set[int] = set()
    while pending:
        parent_pid = pending.pop()
        for pid, group_id in children.get(parent_pid, ()):
            if pid in descendants:
                continue
            descendants.add(pid)
            groups.add(group_id)
            pending.append(pid)
    return frozenset(groups)


def _process_exit_violation(returncode: int | None) -> str | None:
    """Accept normal completion or the monitor's intentional SIGINT exit."""
    return None if returncode in {0, 130} else f"main_process_exit:{returncode}"


def _wait_for_closed_ports(
    ports: tuple[int, ...],
    *,
    timeout_s: float,
) -> frozenset[int]:
    deadline = time.monotonic() + timeout_s
    closed: set[int] = set()
    while time.monotonic() < deadline and len(closed) < len(ports):
        for port in ports:
            if port in closed:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.1)
                try:
                    probe.connect(("127.0.0.1", port))
                except OSError:
                    closed.add(port)
        if len(closed) < len(ports):
            time.sleep(0.05)
    return frozenset(closed)


def _write_reports(result: FullBattleAcceptance, output_report: Path) -> None:
    if result.completed and result.violations:
        result = result.model_copy(update={"completed": False})
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# Main Live Battle Acceptance",
        "",
        f"- Status: **{'PASS' if result.completed and not result.violations else 'BLOCKED/FAIL'}**",
        f"- Git commit: `{result.git_commit or 'unavailable'}`",
        f"- Config SHA-256: `{result.config_sha256 or 'unavailable'}`",
        f"- Wall-clock start (UTC): `{result.wall_clock_start_utc or 'unavailable'}`",
        f"- Wall-clock end (UTC): `{result.wall_clock_end_utc or 'unavailable'}`",
        f"- First plan latency: `{result.first_plan_wall_s if result.first_plan_wall_s is not None else 'unavailable'}` s",
        f"- Final run phase: `{result.final_run_phase}`",
        f"- Final simulation time: `{result.final_sim_time_s}` s",
        f"- Final plan version: `{result.final_plan_version}`",
        f"- Motion audits: `{len(result.motion_audits)}`",
        f"- Physics frames observed/expected: `"
        f"{result.observed_physics_frame_count}/"
        f"{result.expected_physics_frame_count if result.expected_physics_frame_count is not None else 'unavailable'}`",
        f"- Browser errors: `{result.browser_error_count}`",
        f"- Failed requests: `{result.failed_request_count}`",
        f"- Memory events: `{result.memory_event_count}`",
        f"- API p95: `{result.api_p95_ms}` ms",
        f"- Output bytes: `{result.output_bytes}`",
        f"- Shutdown: `{result.shutdown_s}` s",
        "",
        "## Stage Evidence",
        "",
        "| Stage | Simulation time (s) | Plan version |",
        "| --- | ---: | ---: |",
    ]
    for stage_id in sorted(result.stage_sim_times_s):
        markdown.append(
            f"| `{stage_id}` | `{result.stage_sim_times_s[stage_id]}` | "
            f"`{result.stage_plan_versions.get(stage_id, 0)}` |"
        )
    markdown.extend(["", "## Entity Motion Audits", ""])
    markdown.append(
        "| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Total / teleport / boundary / owner / route / formation / resource |"
    )
    markdown.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for audit in result.motion_audits:
        limits = result.motion_limits.get(audit.entity_id)
        depth_range = (
            f"{audit.min_depth_m}..{audit.max_depth_m}"
            if audit.min_depth_m is not None and audit.max_depth_m is not None
            else "n/a"
        )
        violation_breakdown = "/".join(
            str(value)
            for value in (
                audit.limit_violation_count,
                audit.teleport_count,
                audit.boundary_violation_count,
                audit.owner_colocation_violation_count,
                audit.route_violation_count,
                audit.formation_violation_count,
                audit.resource_violation_count,
            )
        )
        markdown.append(
            f"| `{audit.entity_id}` | `{audit.observed_steps}` | "
            f"`{audit.max_speed_mps}/{limits.max_speed_mps if limits else 'n/a'}` | "
            f"`{audit.max_acceleration_mps2}/{limits.max_acceleration_mps2 if limits else 'n/a'}` | "
            f"`{audit.max_deceleration_mps2}/{limits.max_deceleration_mps2 if limits else 'n/a'}` | "
            f"`{audit.max_turn_rate_rad_s}/{limits.max_turn_rate_rad_s if limits else 'n/a'}` | "
            f"`{depth_range}` | `{violation_breakdown}` |"
        )
    markdown.extend(["", "## Evidence Chains", ""])
    for chain in result.battle_evidence_chains:
        estimates = ", ".join(f"`{item}`" for item in chain.blue_estimate_ids)
        markdown.append(
            f"- detection `{chain.target_detection_event_id}` -> adversary decision "
            f"`{chain.adversary_decision_id}` via "
            f"`{chain.adversary_provider_model}` call "
            f"`{chain.adversary_provider_call_id}` -> blue epoch `{chain.blue_epoch_id}` "
            f"plan `{chain.blue_plan_version}`; estimates: {estimates}"
        )
    markdown.extend(["", "## UUV-only Tracking Chains", ""])
    markdown.extend(
        [
            "| Target | Region | UUVs | Boundary entry | Active ping | Detection | Estimates | Handoff | Boundary exit | Replacement | Blue response | Plan |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for chain in result.uuv_tracking_chains:
        markdown.append(
            f"| `{chain.target_id}` | `{chain.region_id}` | "
            f"{', '.join(f'`{item}`' for item in chain.uuv_ids)} | "
            f"{', '.join(f'`{item}`' for item in chain.boundary_entry_event_ids)} | "
            f"`{chain.active_ping_event_id}` | "
            f"{', '.join(f'`{item}`' for item in chain.detection_event_ids)} | "
            f"{', '.join(f'`{item}`' for item in chain.estimate_event_ids)} | "
            f"`{chain.handoff_event_id}` | "
            f"{', '.join(f'`{item}`' for item in chain.boundary_exit_event_ids)} | "
            f"`{chain.boundary_replacement_event_id}` -> "
            f"`{chain.replacement_uuv_id}` | "
            f"`{chain.blue_response_event_id or 'unavailable'}` | "
            f"`{chain.plan_version}` |"
        )
    markdown.extend(["", "## Blue Tracking Chains", ""])
    markdown.extend(
        [
            "| Target | Carrier / candidate | UUVs | Dispatch | Deploy | Active ping | Estimates | Handoff | Resource | Recovery | Recovered | Carrier return | Plan |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for chain in result.blue_tracking_chains:
        markdown.append(
            f"| `{chain.target_id}` | `{chain.carrier_id}` / `{chain.candidate_id}` | "
            f"{', '.join(f'`{item}`' for item in chain.uuv_ids)} | "
            f"`{chain.dispatch_event_id}` | "
            f"{', '.join(f'`{item}`' for item in chain.deployment_event_ids)} | "
            f"`{chain.active_ping_event_id}` | "
            f"{', '.join(f'`{item}`' for item in chain.estimate_event_ids)} | "
            f"`{chain.handoff_event_id}` | `{chain.resource_event_id}` | "
            f"`{chain.recovery_request_event_id}` | `{chain.recovered_event_id}` | "
            f"`{chain.carrier_return_event_id}` | `{chain.plan_version}` |"
        )
    markdown.extend(["", "## Prediction Intent Chains", ""])
    markdown.extend(
        [
            "| Target | Diff / thresholds | Window (s) | Suspicion | Intent provider / calls | Confirmation | Plan | Response latency | Blue response |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for chain in result.prediction_intent_chains:
        calls = ", ".join(f"`{call_id}`" for call_id in chain.intent_llm_call_ids)
        providers = ", ".join(chain.intent_provider_models)
        responses = ", ".join(
            f"`{event_id}`" for event_id in chain.blue_response_event_ids
        )
        markdown.append(
            f"| `{chain.target_id}` | `{chain.diff_id}`; "
            f"`{chain.absolute_rms_m}/{chain.absolute_floor_m}` m; "
            f"`{chain.normalized_rms}/{chain.normalized_threshold}` | "
            f"`{chain.overlap_start_s}..{chain.overlap_end_s}` | "
            f"`{chain.suspicion_event_id}` @ `{chain.suspicion_sim_time_s}` | "
            f"{providers}; {calls} | `{chain.confirmed_event_id}` @ "
            f"`{chain.confirmation_sim_time_s}` | "
            f"`{chain.resulting_plan_id}` / `{chain.resulting_plan_revision}` | "
            f"`{chain.response_latency_s}` s | {responses} |"
        )
    markdown.extend(["", "## Screenshots", ""])
    markdown.extend(f"- [{path}]({path})" for path in result.screenshot_paths)
    markdown.extend(["", "## Browser Diagnostics", ""])
    markdown.extend(
        f"- `{detail}`" for detail in result.browser_error_details
    )
    markdown.extend(["", "## Violations", ""])
    markdown.extend(f"- {item}" for item in result.violations or ("None",))
    output_report.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
