import type {
  ExecutionView,
  RegionEntryEvidenceView,
  RegionTaskView,
  ScanTelemetryView,
  TargetEstimateView,
  TaskGroupInstanceView,
  OperationalFrame,
} from "../types/frames";

/**
 * Presentation status for an estimate.  This is deliberately smaller than
 * the transport health enum: the command centre needs to make one safe
 * visual decision without pretending that a legacy/missing field is live.
 */
export type EstimatePresentationStatus =
  | "live"
  | "stale"
  | "expired"
  | "unavailable"
  | "unknown";

export const ESTIMATE_STATUS_LABELS: Record<EstimatePresentationStatus, string> = {
  live: "LIVE / 当前",
  stale: "STALE / 陈旧",
  expired: "EXPIRED / 已过期",
  unavailable: "UNAVAILABLE / 不可用",
  unknown: "UNKNOWN / 未知",
};

export const ESTIMATE_STATUS_SHORT_LABELS: Record<EstimatePresentationStatus, string> = {
  live: "实时",
  stale: "陈旧",
  expired: "已过期",
  unavailable: "不可用",
  unknown: "未知",
};

function finite(value: number | null | undefined): value is number {
  return value != null && Number.isFinite(value);
}

function explicitFreshness(target: TargetEstimateView) {
  const freshness = target.estimate_freshness;
  if (freshness) return freshness;
  if (
    target.estimate_health_status != null
    || target.estimate_time_s != null
    || target.valid_until_s != null
    || target.data_age_s != null
    || target.track_revision != null
    || target.source_observation_ids != null
  ) {
    return {
      status: target.estimate_health_status ?? "unknown" as const,
      estimate_time_s: target.estimate_time_s,
      valid_until_s: target.valid_until_s,
      data_age_s: target.data_age_s,
      track_revision: target.track_revision,
      source_observation_ids: target.source_observation_ids,
    };
  }
  return null;
}

/**
 * Resolve freshness using only fields explicitly published for the estimate.
 * Prediction health is retained as a legacy fallback, but absent health is
 * always unknown.  `valid_until_s` wins over a contradictory live label so an
 * expired frame cannot keep looking current simply because of an old status.
 */
export function estimatePresentationStatus(
  frame: Pick<OperationalFrame, "sim_time_s" | "execution">,
  target: TargetEstimateView,
): EstimatePresentationStatus {
  const freshness = explicitFreshness(target);
  const validUntil = freshness?.valid_until_s;
  if (finite(validUntil) && frame.sim_time_s > validUntil) return "expired";

  if (freshness) {
    switch (freshness.status) {
      case "live":
      case "current":
        return "live";
      case "stale":
      case "degraded":
        return "stale";
      case "expired":
        return "expired";
      case "unavailable":
      case "failed":
        return "unavailable";
      case "unknown":
      default:
        return "unknown";
    }
  }

  // Runtime execution health is the only safe legacy signal for frames that
  // predate estimate_freshness. It is scoped to the same target; no trail or
  // frontend timer is allowed to manufacture freshness.
  if (frame.execution?.target_id === target.target_id) {
    if (frame.execution.health_status === "expired") return "expired";
    if (frame.execution.health_status === "degraded") return "stale";
    if (frame.execution.health_status === "failed") return "unavailable";
  }

  switch (target.prediction?.health?.status) {
    case "valid":
      return "live";
    case "degraded":
      return "stale";
    case "unavailable":
      return "unavailable";
    case "legacy_unknown":
    default:
      return "unknown";
  }
}

export function estimateTimestamp(target: TargetEstimateView): number | null {
  const freshness = explicitFreshness(target);
  return finite(freshness?.estimate_time_s) ? freshness.estimate_time_s : null;
}

export function estimateValidUntil(target: TargetEstimateView): number | null {
  const freshness = explicitFreshness(target);
  return finite(freshness?.valid_until_s) ? freshness.valid_until_s : null;
}

export function estimateDataAge(
  frame: Pick<OperationalFrame, "sim_time_s">,
  target: TargetEstimateView,
): number | null {
  const freshness = explicitFreshness(target);
  if (finite(freshness?.data_age_s)) return Math.max(0, freshness.data_age_s);
  const timestamp = estimateTimestamp(target);
  return timestamp == null ? null : Math.max(0, frame.sim_time_s - timestamp);
}

export function estimateTrackRevision(target: TargetEstimateView): number | null {
  const freshness = explicitFreshness(target);
  return finite(freshness?.track_revision) ? freshness.track_revision : null;
}

export function estimateObservationIds(target: TargetEstimateView): string[] {
  const freshness = explicitFreshness(target);
  return [...(freshness?.source_observation_ids ?? [])];
}

/** Map status to a small set of drawing cues shared by canvas and overlays. */
export function estimatePresentationStyle(status: EstimatePresentationStatus) {
  switch (status) {
    case "live":
      return {
        opacity: 1,
        stroke: "rgba(255, 120, 130, 0.96)",
        fill: "rgba(255, 120, 130, 0.08)",
        dash: [] as number[],
        marker: "solid" as const,
      };
    case "stale":
      return {
        opacity: 0.62,
        stroke: "rgba(247, 189, 69, 0.92)",
        fill: "rgba(247, 189, 69, 0.055)",
        dash: [7, 5],
        marker: "dim" as const,
      };
    case "expired":
      return {
        opacity: 0.38,
        stroke: "rgba(255, 183, 94, 0.86)",
        fill: "rgba(255, 183, 94, 0.035)",
        dash: [3, 7],
        marker: "frozen" as const,
      };
    case "unavailable":
      return {
        opacity: 0.28,
        stroke: "rgba(255, 120, 130, 0.72)",
        fill: "rgba(255, 120, 130, 0.02)",
        dash: [2, 6],
        marker: "missing" as const,
      };
    case "unknown":
    default:
      return {
        opacity: 0.46,
        stroke: "rgba(173, 190, 205, 0.82)",
        fill: "rgba(173, 190, 205, 0.025)",
        dash: [3, 5],
        marker: "unknown" as const,
      };
  }
}

/** Explicit telemetry only; effect.coverage_ratio is not active scan data. */
export function scanTelemetryForRegion(
  region: Pick<RegionTaskView, "scan_telemetry"> | null | undefined,
): ScanTelemetryView | null {
  return region?.scan_telemetry ?? null;
}

export function scanPingCount(telemetry: ScanTelemetryView | null): number | null {
  if (!telemetry) return null;
  const count = telemetry.source_backed_ping_count ?? telemetry.active_ping_count;
  return finite(count) ? Math.max(0, Math.round(count)) : null;
}

export function scanCoveragePercent(telemetry: ScanTelemetryView | null): number | null {
  const value = telemetry?.active_coverage_ratio;
  return finite(value) ? Math.max(0, Math.min(1, value)) * 100 : null;
}

export function scanRouteProgressPercent(telemetry: ScanTelemetryView | null): number | null {
  const value = telemetry?.route_progress;
  return finite(value) ? Math.max(0, Math.min(1, value)) * 100 : null;
}

export function entryEvidenceForRegion(
  execution: ExecutionView | null | undefined,
  regionId: string,
): RegionEntryEvidenceView | null {
  if (!execution) return null;
  const explicit = execution.region_entry_evidence?.[regionId];
  const probability = execution.region_entry_probabilities?.[regionId];
  const count = execution.entry_confirmation_counts?.[regionId];
  const blocking = execution.entry_blocking_reasons?.[regionId];
  if (explicit) return explicit;
  if (probability !== undefined || count !== undefined || blocking !== undefined) {
    return {
      probability: probability ?? null,
      confirmation_count: count ?? null,
      required_cycles: execution.entry_confirmation_required_cycles ?? null,
      blocking_reasons: blocking ?? [],
    };
  }
  return null;
}

export interface DeploymentProgress {
  deployed: number | null;
  total: number;
  source: "backend_deployed_ids" | "uuv_deployment_state" | "unavailable";
}

export function deploymentProgress(
  frame: OperationalFrame,
  group: TaskGroupInstanceView,
): DeploymentProgress {
  const total = group.member_uuv_ids.length;
  if (group.deployed_member_uuv_ids) {
    const members = new Set(group.member_uuv_ids);
    const deployed = group.deployed_member_uuv_ids.filter((id) => members.has(id));
    return { deployed: new Set(deployed).size, total, source: "backend_deployed_ids" };
  }
  const members = (frame.uuvs ?? []).filter(
    (uuv) => uuv.group_instance_id === group.group_instance_id,
  );
  if (!members.length) return { deployed: null, total, source: "unavailable" };
  return {
    deployed: members.filter((uuv) => uuv.deployment_state === "deployed").length,
    total,
    source: "uuv_deployment_state",
  };
}

export function groupPassiveObservationIds(
  group: TaskGroupInstanceView,
): string[] {
  return [...(group.passive_observation_uuv_ids ?? [])];
}

export const GROUP_LIFECYCLE_LABELS: Record<TaskGroupInstanceView["lifecycle"], string> = {
  entering: "进入中",
  active_scan: "主动扫描",
  passive_track: "被动跟踪",
  dedicated_track: "专用跟踪",
  dedicated_release_pending: "专用释放待定",
  exiting: "退出中",
  disappeared: "已消失",
};

export const GROUP_SENSOR_LABELS: Record<TaskGroupInstanceView["sensor_mode"], string> = {
  active: "主动",
  passive: "被动",
  off: "关闭",
};

export const HANDOFF_STATUS_LABELS: Record<NonNullable<ExecutionView["handoff_evidence"]>["status"], string> = {
  pending: "待确认",
  ready: "可交接",
  completed: "已完成",
  blocked: "受阻",
  unavailable: "不可用",
};

export function structuredBlockingReasons(
  execution: ExecutionView | null | undefined,
  regionId?: string | null,
): string[] {
  if (!execution) return [];
  const reasons: string[] = [];
  const seen = new Set<string>();
  const add = (reason: string | null | undefined) => {
    if (!reason || seen.has(reason)) return;
    seen.add(reason);
    reasons.push(reason);
  };
  execution.blocking_reasons?.forEach((reason) => add(reason.message ?? reason.code));
  if (regionId) {
    entryEvidenceForRegion(execution, regionId)?.blocking_reasons?.forEach(add);
  }
  execution.handoff_evidence?.blocking_reasons?.forEach(add);
  if (execution.health_status !== "current") (execution.health_reasons ?? []).forEach(add);
  (execution.degradation_reasons ?? []).forEach(add);
  return reasons;
}

export function formatSimSeconds(value: number | null | undefined): string {
  return finite(value) ? `${value.toFixed(value % 1 === 0 ? 0 : 1)}s` : "—";
}

export function formatPercent(value: number | null | undefined): string {
  return finite(value) ? `${Math.round(value)}%` : "不可用";
}

export function handoffProgress(execution: ExecutionView | null | undefined) {
  const handoff = execution?.handoff_evidence;
  if (!handoff) return null;
  const required = new Set(handoff.required_uuv_ids ?? []);
  const valid = new Set((handoff.valid_observation_uuv_ids ?? []).filter((id) => required.has(id)));
  return {
    valid: valid.size,
    required: required.size,
    missing: [...required].filter((id) => !valid.has(id)),
    status: handoff.status,
  };
}

export function targetGroups(
  frame: OperationalFrame,
  targetId: string,
): TaskGroupInstanceView[] {
  return (frame.execution?.task_groups ?? []).filter((group) => group.target_id === targetId);
}

export function targetHasRuntimeEstimate(frame: OperationalFrame, target: TargetEstimateView): boolean {
  return estimatePresentationStatus(frame, target) === "live";
}
