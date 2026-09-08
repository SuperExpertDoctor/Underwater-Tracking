import type { OperationalFrame, TargetEstimateView } from "../types/frames";

type ContractFrameOptions = {
  simTimeS?: number;
  validUntilS?: number | null;
  freshnessStatus?: "live" | "expired" | "unknown";
  coverage?: number | null;
  pingCount?: number | null;
  scanCompleted?: boolean | null;
  entryCount?: number | null;
  handoffValidIds?: string[];
  lifecycle?: "active_scan" | "exiting" | "dedicated_release_pending";
  dedicated?: boolean;
};

const point = { x: 0, y: 0 };

export function createContractFrame(options: ContractFrameOptions = {}): OperationalFrame {
  const dedicated = options.dedicated ?? false;
  const lifecycle = options.lifecycle ?? "active_scan";
  const validUntilS = options.validUntilS === undefined ? 120 : options.validUntilS;
  const target = {
    target_id: "T1",
    mean: point,
    covariance_ellipse: null,
    estimate_health: {},
    intent: { label: "unknown", confidence: 0, alternatives: {} },
    prediction: { health: { status: "valid" } },
    estimate_freshness: {
      status: options.freshnessStatus ?? "live",
      estimate_time_s: 60,
      valid_until_s: validUntilS,
      data_age_s: 0,
      track_revision: 7,
      source_observation_ids: ["obs-1"],
    },
    quality: {
      quality_score: 1,
      estimated_rmse_m: 1,
      fim_min_eigenvalue: 1,
      fim_condition: 1,
    },
    classification: "submarine",
    last_ping_s: 60,
  } as unknown as TargetEstimateView;

  const region = {
    region_id: "T1:task:01",
    target_id: "T1",
    slot_index: 0,
    execution_revision: 7,
    prediction_id: "prediction-7",
    geometry: [point],
    start_s: 0,
    end_s: 120,
    geometry_revision: 2,
    predecessor_region_id: null,
    successor_region_id: "T1:task:02",
    handoff_start_s: 90,
    handoff_end_s: 120,
    status: lifecycle === "exiting" ? "handoff_pending" : "active",
    task_group_id: "T1:group:owner",
    evidence_ids: ["evidence-region-1"],
    scan_telemetry: {
      route_progress: 1,
      scan_round: 1,
      active_coverage_ratio: options.coverage ?? 0.25,
      source_backed_ping_count: options.pingCount ?? 3,
      scan_completed: options.scanCompleted ?? false,
      scan_completion_threshold: 0.8,
      evidence_ids: ["evidence-scan-1"],
    },
  };
  const secondRegion = {
    ...region,
    region_id: "T1:task:02",
    slot_index: 1,
    task_group_id: "T1:group:successor",
    scan_telemetry: {
      route_progress: 0,
      scan_round: 0,
      active_coverage_ratio: 0,
      source_backed_ping_count: 0,
      scan_completed: false,
      scan_completion_threshold: 0.8,
      evidence_ids: [],
    },
  };
  const ownerGroup = {
    group_instance_id: "T1:group:owner",
    target_id: "T1",
    region_id: "T1:task:01",
    deployment_revision: 3,
    member_uuv_ids: ["uuv-01", "uuv-02", "uuv-03"],
    deployed_member_uuv_ids: ["uuv-01", "uuv-02", "uuv-03"],
    passive_observation_uuv_ids: [],
    lifecycle: dedicated ? "dedicated_release_pending" : lifecycle,
    sensor_mode: "active",
    ownership_status: "owner",
    reason: "source-backed",
    evidence_ids: ["evidence-group-1"],
  };
  const successorGroup = {
    group_instance_id: "T1:group:successor",
    target_id: "T1",
    region_id: "T1:task:02",
    deployment_revision: 4,
    member_uuv_ids: ["uuv-04", "uuv-05", "uuv-06"],
    deployed_member_uuv_ids: ["uuv-04", "uuv-05", "uuv-06"],
    passive_observation_uuv_ids: ["uuv-04", "uuv-05", "uuv-06"],
    lifecycle: dedicated ? "active_scan" : "exiting",
    sensor_mode: "passive",
    ownership_status: "successor",
    reason: "handoff evidence",
    evidence_ids: ["evidence-group-2"],
  };
  const execution = {
    target_id: "T1",
    execution_revision: 7,
    source_snapshot_revision: 7,
    prediction_revision: 7,
    intent_revision: 7,
    data_age_s: 0,
    valid_from_s: 0,
    valid_until_s: 120,
    health_status: "current",
    health_reasons: [],
    region_generation_mode: "imm",
    plan_source: "deterministic",
    current_region_id: "T1:task:01",
    next_region_id: "T1:task:02",
    evidence_ids: ["evidence-execution-1"],
    regions: [region, secondRegion],
    task_groups: [ownerGroup, successorGroup],
    tracking_policy: { dedicated_release_remaining_mileage_m: 7000 },
    tracking_control: {
      mode: dedicated ? "dedicated" : "regional",
      tracking_owner_group_id: "T1:group:owner",
      pending_successor_group_id: "T1:group:successor",
      dedicated_release_triggered_at_m: dedicated ? 6850 : null,
      dedicated_release_reason: dedicated ? "remaining mileage lowest" : null,
      source_event_ids: ["event-1"],
    },
    replacements: [],
    entry_confirmation_required_cycles: 2,
    entry_confirmation_counts: { "T1:task:01": options.entryCount ?? 2 },
    region_entry_evidence: {
      "T1:task:01": {
        probability: 0.9,
        confirmation_count: options.entryCount ?? 2,
        required_cycles: 2,
        status: "confirmed",
        blocking_reasons: [],
        evidence_ids: ["evidence-entry-1"],
      },
    },
    handoff_evidence: {
      predecessor_group_id: "T1:group:owner",
      successor_group_id: "T1:group:successor",
      successor_region_id: "T1:task:02",
      observation_cycle_s: 90,
      required_uuv_ids: ["uuv-04", "uuv-05", "uuv-06"],
      valid_observation_uuv_ids: options.handoffValidIds ?? ["uuv-04", "uuv-05", "uuv-06"],
      status: (options.handoffValidIds?.length ?? 3) === 3 ? "completed" : "pending",
      blocking_reasons: [],
      evidence_ids: ["evidence-handoff-1"],
    },
    blocking_reasons: [],
    degraded: false,
    degradation_reasons: [],
    active_plan_preserved: true,
    refresh_status: "idle",
    refresh_attempt_id: null,
    refresh_due_at_s: null,
    refresh_last_attempt_s: null,
    refresh_last_result: "none",
    refresh_reason_codes: [],
    refresh_source_snapshot_revision: null,
  };

  const uuvs = ["uuv-01", "uuv-02", "uuv-03"].map((uuvId) => ({
    uuv_id: uuvId,
    status: "active",
    deployment_state: "deployed",
    physically_exposed: true,
    position: point,
    heading_rad: 0,
    speed_mps: 1,
    energy_fraction: 1,
    group_id: "T1:group:owner",
    group_instance_id: "T1:group:owner",
    deployment_revision: 3,
    group_lifecycle: lifecycle,
    current_waypoint: point,
    breadcrumb: [point],
    sensor_mode: "active",
    reserved: false,
    remaining_range_m: dedicated ? 6850 : 10000,
  }));

  return {
    schema_version: "1.0",
    scenario_id: "contract-test",
    frame_id: 60,
    sim_time_s: options.simTimeS ?? 60,
    physics_step_s: 1,
    plan_version: 7,
    run_phase: "running",
    map_bounds: { min_x: -100, min_y: -100, max_x: 100, max_y: 100 },
    uuvs,
    target_estimates: [target],
    bearing_rays: [],
    groups: [],
    events: [],
    plans: [],
    ledger: [],
    metrics: [],
    carrier: null,
    execution,
  } as unknown as OperationalFrame;
}
