#!/usr/bin/env python3
"""Own and audit a real full-duration command-center process."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
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

from underwater_tracking.verification.live_demo import (  # noqa: E402
    LiveDemoAcceptanceResult,
    verify_live_demo,
)
from underwater_tracking.verification.physics_invariants import (  # noqa: E402
    BattleEvidenceChain,
    EntityMotionAudit,
    EntityMotionLimits,
    FullBattleAcceptance,
)

EXPECTED_ENTITIES = {
    "carrier_01",
    "carrier_02",
    "carrier_03",
    "carrier_04",
    *(f"uuv_{index:02d}" for index in range(12)),
    "target_00",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the complete real live battle")
    parser.add_argument("--main", type=Path, default=ROOT / "main.py")
    parser.add_argument("--scenario", type=Path, default=ROOT / "configs/scenario/uuv_only_single_target.yaml")
    parser.add_argument("--wall-timeout-s", type=float, default=1200.0)
    parser.add_argument("--expected-duration-s", type=int, default=28_800)
    parser.add_argument("--require-real-provider", action="store_true")
    parser.add_argument("--output-report", type=Path, default=ROOT / "docs/verification/main-live-battle-acceptance.json")
    args = parser.parse_args(argv)

    api_port = _free_port()
    ui_port = _free_port({api_port})
    process = subprocess.Popen(
        [
            sys.executable,
            str(args.main),
            "--config",
            str(args.scenario),
            "--port",
            str(api_port),
            "--ui-port",
            str(ui_port),
            "--verification-audit",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(sys.platform != "win32"),
    )
    base_url = f"http://127.0.0.1:{api_port}"
    browser_stop = Event()
    browser_result: list[tuple[int, int]] = []
    browser_thread = Thread(
        target=lambda: browser_result.append(
            _browser_audit(
                f"http://127.0.0.1:{ui_port}",
                args.output_report.parent / "screenshots",
                stop_event=browser_stop,
            )
        ),
        name="live-battle-browser-audit",
        daemon=True,
    )
    api_ready = _wait_for_api(base_url)
    try:
        if api_ready:
            run_output_dir = _current_run_output_dir(base_url)
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
    browser_errors, failed_requests = browser_result[0] if browser_result else (1, 1)
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
    )
    shutdown_started = time.monotonic()
    shutdown_timed_out = _stop_process(process)
    shutdown_s = round(time.monotonic() - shutdown_started, 3)
    shutdown_violations = []
    if shutdown_timed_out or shutdown_s > 10.0:
        shutdown_violations.append("shutdown_exceeded_10s")
    result = result.model_copy(
        update={
            "shutdown_s": shutdown_s,
            "wall_clock_end_utc": _utc_now(),
            "violations": tuple(
                dict.fromkeys((*result.violations, *shutdown_violations))
            ),
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
    if evidence is None:
        violations.append("battle_evidence_unavailable")
    if verification_request_failures:
        violations.append(f"verification_requests_failed:{verification_request_failures}")
    if not chains:
        violations.append("missing_counter_tracking_evidence_chain")
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
        motion_audits=tuple(sorted(audits, key=lambda audit: audit.entity_id)),
        motion_limits=motion_limits,
        observed_physics_frame_count=observed_physics_frame_count,
        expected_physics_frame_count=expected_physics_frame_count,
        physics_frame_coverage=physics_coverage,
        browser_error_count=browser_errors,
        failed_request_count=(
            live.failed_request_count + failed_requests + verification_request_failures
        ),
        memory_event_count=live.memory_event_count,
        api_p95_ms=live.api_p95_ms,
        output_bytes=live.output_bytes,
        shutdown_s=0.0,
        git_commit=git_commit,
        config_sha256=config_sha256,
        screenshot_paths=(
            "screenshots/desktop.png",
            "screenshots/mobile.png",
            "screenshots/desktop-latest.png",
            "screenshots/mobile-latest.png",
        ),
        violations=tuple(dict.fromkeys(violations)),
    )


def _evidence_chains(value: object) -> list[BattleEvidenceChain]:
    if not isinstance(value, Mapping):
        return []
    raw_events = value.get("events", ())
    raw_decisions = value.get("adversary_decisions", ())
    raw_response_chains = value.get("blue_response_chains", ())
    root_epoch_id = value.get("blue_epoch_id")
    root_plan_version = value.get("blue_plan_version")
    if not isinstance(raw_events, (list, tuple)) or not isinstance(raw_decisions, (list, tuple)):
        return []
    if not isinstance(raw_response_chains, (list, tuple)) or not raw_response_chains:
        return []
    raw_public_observations = value.get("public_observations", ())
    if not isinstance(raw_public_observations, (list, tuple)) or not raw_public_observations:
        return []
    events = [item for item in raw_events if isinstance(item, Mapping)]
    decisions = [item for item in raw_decisions if isinstance(item, Mapping)]
    if len(events) != len(raw_events) or len(decisions) != len(raw_decisions):
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
                    adversary_source_event_ids=source_ids,
                    resulting_public_observation_ids=observation_ids,
                    blue_estimate_ids=estimate_ids,
                    blue_epoch_id=str(epoch_id),
                    blue_plan_version=plan_version,
                )
            )
            continue
        if not estimate_events:
            continue
    return chains


def _browser_audit(
    ui_url: str,
    screenshot_dir: Path,
    *,
    stop_event: Event | None = None,
) -> tuple[int, int]:
    stop_event = stop_event or Event()
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return 1, 1
    console_errors = 0
    failed_requests = 0
    websocket_seen = 0
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except Exception:  # noqa: BLE001 - missing browser binary is a gate failure
            return 1, 1
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
                    websocket: object, *, counts: dict[str, int] = websocket_counts
                ) -> None:
                    nonlocal websocket_seen
                    websocket_seen += 1
                    on_error = getattr(websocket, "on", None)
                    if callable(on_error):
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
                    page.screenshot(path=str(screenshot_dir / f"{label}.png"), full_page=True)
                else:
                    page_errors.append("navigation")
            while not stop_event.wait(0.5):
                for state in page_states:
                    label = str(state["label"])
                    page = state["page"]
                    assert hasattr(page, "screenshot")
                    try:
                        page.screenshot(path=str(screenshot_dir / f"{label}-latest.png"), full_page=True)
                    except Exception:
                        cast(list[object], state["page_errors"]).append("screenshot")
                    _probe_ui_consistency(
                        page,
                        ui_url,
                        cast(list[object], state["page_errors"]),
                        cast(list[object], state["request_errors"]),
                    )
            for state in page_states:
                cast(object, state["page"]).close()
        finally:
            browser.close()
    for state in page_states:
        console_errors += len(cast(list[object], state["console_messages"]))
        console_errors += len(cast(list[object], state["page_errors"]))
        failed_requests += len(cast(list[object], state["request_errors"]))
        # A close is normal during React dev-mode remounts and page cleanup;
        # only a socket error is a failed browser request.
        for detail in cast(list[str], state["websocket_errors"]):
            if (
                stop_event.is_set()
                and detail == "WebSocket is closed before the connection is established."
            ):
                continue
            failed_requests += 1
    if websocket_seen == 0:
        failed_requests += 1
    return console_errors, failed_requests


def _exercise_ui(page: object, page_errors: list[object]) -> None:
    """Open the operator surfaces so hidden tabs are part of the live audit."""
    try:
        toggle = page.locator('button[aria-label="切换任务详情"]')
        if toggle.count():
            toggle.click(timeout=1_000)
            for label in ("时间线", "决策台账", "LLM 思考过程", "Memory Steam"):
                tab = page.get_by_role("tab", name=label)
                if tab.count():
                    tab.click(timeout=1_000)
                    if not page.locator("#mission-panel").count():
                        page_errors.append(f"missing_tab_panel:{label}")
    except Exception as exc:  # noqa: BLE001 - browser state is an acceptance gate
        page_errors.append(f"ui_surface_probe:{type(exc).__name__}")


def _probe_ui_consistency(
    page: object,
    ui_url: str,
    page_errors: list[object],
    request_errors: list[object],
) -> None:
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
        response = page.request.get(
            f"{ui_url.rstrip('/')}/api/operational/snapshot",
            timeout=2_000,
        )
        if not response.ok:
            request_errors.append(f"snapshot_http_{response.status}")
            return
        payload = response.json()
        if not isinstance(payload, Mapping):
            page_errors.append("snapshot_not_object")
            return
        run_phase = payload.get("run_phase")
        sim_time_s = payload.get("sim_time_s")
        if run_phase in {"running", "completed"} or (
            isinstance(sim_time_s, (int, float)) and sim_time_s > 0
        ):
            for selector in (".memory-window", ".brain-section"):
                try:
                    page.locator(selector).wait_for(state="attached", timeout=1_000)
                except Exception:
                    page_errors.append(f"missing_ui_surface:{selector}")
        plan_node = page.locator("[data-plan-version]").first
        if plan_node.count():
            dom_plan = plan_node.get_attribute("data-plan-version")
            if dom_plan != str(payload.get("plan_version", 0)):
                page_errors.append("ui_plan_version_mismatch")
        time_node = page.locator(".playback-readout.time").first
        if time_node.count():
            dom_time = time_node.text_content()
            if dom_time and dom_time.strip() != f"{payload.get('sim_time_s')}s":
                page_errors.append("ui_sim_time_mismatch")
    except Exception as exc:  # noqa: BLE001 - browser/API consistency is a gate
        page_errors.append(f"ui_consistency_probe:{type(exc).__name__}")


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
        except (OSError, ValueError):
            time.sleep(0.25)
    return False


def _current_run_output_dir(base_url: str, *, timeout_s: float = 10.0) -> Path | None:
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
                candidate = ROOT / "outputs" / run_id
                if candidate.is_dir():
                    return candidate
        time.sleep(0.25)
    return None


def _safe_get_json(base_url: str, path: str) -> tuple[object | None, bool]:
    try:
        return _get_json(base_url, path), False
    except (OSError, ValueError):
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
    process.send_signal(signal.SIGINT if sys.platform != "win32" else signal.CTRL_BREAK_EVENT)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)
        return True
    return False


def _write_reports(result: FullBattleAcceptance, output_report: Path) -> None:
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# Main Live Battle Acceptance",
        "",
        f"- Status: **{'PASS' if result.completed else 'BLOCKED/FAIL'}**",
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
        "| Entity | Steps | Max speed | Max accel | Max decel | Max turn | Depth range | Violations |"
    )
    markdown.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for audit in result.motion_audits:
        limits = result.motion_limits.get(audit.entity_id)
        depth_range = (
            f"{audit.min_depth_m}..{audit.max_depth_m}"
            if audit.min_depth_m is not None and audit.max_depth_m is not None
            else "n/a"
        )
        violation_count = (
            audit.limit_violation_count
            + audit.teleport_count
            + audit.boundary_violation_count
        )
        markdown.append(
            f"| `{audit.entity_id}` | `{audit.observed_steps}` | "
            f"`{audit.max_speed_mps}/{limits.max_speed_mps if limits else 'n/a'}` | "
            f"`{audit.max_acceleration_mps2}/{limits.max_acceleration_mps2 if limits else 'n/a'}` | "
            f"`{audit.max_deceleration_mps2}/{limits.max_deceleration_mps2 if limits else 'n/a'}` | "
            f"`{audit.max_turn_rate_rad_s}/{limits.max_turn_rate_rad_s if limits else 'n/a'}` | "
            f"`{depth_range}` | `{violation_count}` |"
        )
    markdown.extend(["", "## Evidence Chains", ""])
    for chain in result.battle_evidence_chains:
        estimates = ", ".join(f"`{item}`" for item in chain.blue_estimate_ids)
        markdown.append(
            f"- detection `{chain.target_detection_event_id}` -> adversary decision "
            f"`{chain.adversary_decision_id}` -> blue epoch `{chain.blue_epoch_id}` "
            f"plan `{chain.blue_plan_version}`; estimates: {estimates}"
        )
    markdown.extend(["", "## Screenshots", ""])
    markdown.extend(f"- [{path}]({path})" for path in result.screenshot_paths)
    markdown.extend(["", "## Violations", ""])
    markdown.extend(f"- {item}" for item in result.violations or ("None",))
    output_report.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
