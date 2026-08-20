import type {
  AdversaryView,
  BearingRayView,
  BrainView,
  CarrierView,
  CommunicationLinkView,
  EventView,
  GroupView,
  MapBounds,
  MetricView,
  OperationalStage,
  OperationalFrame,
  PlanAdjustmentSuggestionView,
  PlanTimelineView,
  PlanView,
  Point2D,
  RegionalPlanView,
  RegionTimelineView,
  TargetEstimateView,
  TrackingEffectStatus,
  USVView,
  UUVView,
} from "../types/frames";

/** Deterministic front-end scenario: estimator → versioned prediction → regions → allocation. */
export const MOCK_PHYSICS_STEP_S = 5;
// 80 minutes: v1, maneuver/revision v2, recovery, then a distinct v3 coastal
// relay plan before the carrier resumes its original fixed patrol route.
export const MOCK_FRAME_COUNT = 961;
export const MOCK_MAP_BOUNDS: MapBounds = {
  min_x: -12000,
  min_y: -12000,
  max_x: 12000,
  max_y: 12000,
};
const CARRIER = "carrier_01",
  TARGET = "target",
  UUV = Array.from(
    { length: 12 },
    (_, i) => `uuv_${String(i).padStart(2, "0")}`,
  ),
  USV = Array.from(
    { length: 4 },
    (_, i) => `usv_${String(i).padStart(2, "0")}`,
  );
const MANEUVER = 840,
  REVISION = 870,
  PING = 855;
const REVISION_RENDER_BLEND_S = 60;
const REGION_FADE_S = 180;
const COMPLETED_PLAN_RETENTION_S = 180;
const p = (x: number, y: number): Point2D => ({
  x: Math.round(x * 10) / 10,
  y: Math.round(y * 10) / 10,
});
const clamp = (x: number, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const mix = (a: Point2D, b: Point2D, t: number) =>
  p(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
const d = (a: Point2D, b: Point2D) => Math.hypot(a.x - b.x, a.y - b.y);
type R = {
  id: string;
  label: string;
  start: number;
  handoff: number;
  end: number;
  members: string[];
  relay: string;
  center: number;
};
type Plan = { version: number; id: string; regions: R[] };
const r = (
  id: string,
  label: string,
  start: number,
  handoff: number,
  end: number,
  members: string[],
  relay: string,
  center: number,
): R => ({ id, label, start, handoff, end, members, relay, center });
const V1: Plan = {
  version: 1,
  id: "target:prediction:v1",
  regions: [
    r(
      "target:cell:-5:-5",
      "region_1",
      180,
      180,
      900,
      UUV.slice(0, 4),
      USV[0],
      540,
    ),
    r(
      "target:cell:-2:-5",
      "region_2",
      780,
      900,
      1500,
      UUV.slice(4, 8),
      USV[1],
      1140,
    ),
    r(
      "target:cell:1:-4",
      "region_3",
      1380,
      1500,
      2100,
      UUV.slice(8),
      USV[2],
      1740,
    ),
  ],
};
// v2 is intentionally not three fixed segments: the broad, turned corridor replaces C with C1/C2/D.
const V2: Plan = {
  version: 2,
  id: "target:prediction:v2",
  regions: [
    r(
      "target:cell:-5:-5",
      "region_1",
      180,
      180,
      900,
      UUV.slice(0, 4),
      USV[0],
      540,
    ),
    r(
      "target:cell:-2:-4",
      "region_2_revised",
      780,
      900,
      1500,
      UUV.slice(4, 8),
      USV[1],
      1140,
    ),
    r(
      "target:cell:0:-1",
      "region_3a",
      1380,
      1500,
      1800,
      UUV.slice(8),
      USV[2],
      1590,
    ),
    r(
      "target:cell:1:1",
      "region_3b",
      1680,
      1800,
      2100,
      UUV.slice(4, 8),
      USV[3],
      1890,
    ),
    r(
      "target:cell:2:3",
      "region_4",
      1980,
      2100,
      2400,
      UUV.slice(8),
      USV[2],
      2190,
    ),
  ],
};
const V3: Plan = {
  version: 3,
  id: "target:prediction:v3",
  regions: [
    r(
      "target:cell:1:5",
      "region_5a",
      2850,
      2850,
      3450,
      UUV.slice(0, 4),
      USV[0],
      3150,
    ),
    r(
      "target:cell:-1:7",
      "region_5b",
      3330,
      3450,
      4050,
      UUV.slice(4, 8),
      USV[1],
      3690,
    ),
    r(
      "target:cell:-3:8",
      "region_6",
      3930,
      4050,
      4500,
      UUV.slice(8),
      USV[2],
      4215,
    ),
  ],
};
const planAt = (t: number): Plan | null =>
  t < 300
    ? null
    : t < REVISION
      ? V1
      : t < 2400 + COMPLETED_PLAN_RETENTION_S
        ? V2
        : t >= 2700 && t < 4500 + COMPLETED_PLAN_RETENTION_S
          ? V3
          : null;
// Execution still needs the last committed plan while its assigned UUVs return.
const executionPlanAt = (t: number): Plan =>
  t < REVISION ? V1 : t < 2700 ? V2 : V3;

// Mirrors simulation/carrier.py: fixed outer square patrol, constant 5 m/s.
function carrierAt(t: number) {
  const c = [p(-3000, -3000), p(3000, -3000), p(3000, 3000), p(-3000, 3000)],
    leg = 1200,
    q = ((t % 4800) + 4800) % 4800,
    i = Math.floor(q / leg);
  return mix(c[i], c[(i + 1) % 4], (q % leg) / leg);
}
function headingAt(fn: (t: number) => Point2D, t: number) {
  const a = fn(t),
    b = fn(t + 5);
  return Math.atan2(b.y - a.y, b.x - a.x);
}
function truth(t: number): Point2D {
  if (t <= MANEUVER) return p(-4500 + 3.1 * t, -4300 + 210 * Math.sin(t / 220));
  const a = truth(MANEUVER),
    q = t - MANEUVER;
  if (t > 2700) {
    const coast = truth(2700),
      turn = t - 2700;
    return p(
      coast.x - 2.45 * turn + 180 * Math.sin(turn / 160),
      coast.y + 1.15 * turn,
    );
  }
  return p(
    a.x + 1.35 * q + 280 * Math.sin(q / 140),
    a.y + 3.25 * q + 160 * Math.sin(q / 170),
  );
}
function predicted(v: number, t: number): Point2D {
  if (v === 1) return p(-4500 + 3.1 * t, -4300 + 180 * Math.sin(t / 240));
  if (v === 3) {
    const a = truth(2700),
      q = t - 2700;
    return p(a.x - 2.4 * q, a.y + 1.2 * q + 90 * Math.sin(q / 190));
  }
  const a = truth(REVISION),
    q = t - REVISION;
  return p(a.x + 1.35 * q, a.y + 3.15 * q + 120 * Math.sin(q / 180));
}
function estimate(t: number) {
  const x = truth(t),
    err =
      t < REVISION
        ? 150 + 520 * clamp((t - MANEUVER) / (REVISION - MANEUVER))
        : 670 * Math.exp(-(t - REVISION) / 210);
  return p(x.x + err * Math.sin(t / 95), x.y - err * 0.55 * Math.cos(t / 110));
}
const taskAt = (id: string, t: number) =>
  executionPlanAt(t).regions.find(
    (z) => z.members.includes(id) && t >= z.start && t < z.end,
  ) ?? null;
const team = (z: R) => `G-${TARGET}-${z.label}`;
function formation(z: R, slot: number, t: number) {
  const c = predicted(
      z.label === "region_1"
        ? 1
        : z.label.startsWith("region_5") || z.label === "region_6"
          ? 3
          : 2,
      t,
    ),
    a = (slot * Math.PI) / 2 + t / 210,
    radius = 1250 + (slot % 2) * 180;
  return p(c.x + radius * Math.cos(a), c.y + radius * Math.sin(a));
}
function uuvsAt(t: number): UUVView[] {
  return UUV.map((id, n) => {
    const slot = n % 4,
      z = taskAt(id, t),
      first =
        t >= 2700
          ? n < 4
            ? V3.regions[0]
            : n < 8
              ? V3.regions[1]
              : V3.regions[2]
          : n < 4
            ? V1.regions[0]
            : n < 8
              ? V1.regions[1]
              : V2.regions[2],
      deploy = first.start - 150;
    let position = carrierAt(t),
      state: UUVView["deployment_state"] = "onboard",
      status: UUVView["status"] = "available",
      waypoint: Point2D | null = null,
      group: string | null = null,
      trail = [position];
    if (z) {
      position = formation(z, slot, t);
      if (z.label === "region_2_revised" && t >= REVISION && t < REVISION + 60)
        position = mix(
          formation(V1.regions[1], slot, t),
          position,
          (t - REVISION) / 60,
        );
      state = "deployed";
      status = "tracking";
      waypoint = formation(z, slot, t + 120);
      group = team(z);
      trail = Array.from({ length: 8 }, (_, i) =>
        formation(z, slot, Math.max(z.start, t - (7 - i) * 20)),
      );
    } else {
      const pl = executionPlanAt(t),
        prev = pl.regions
          .filter((q) => q.members.includes(id) && q.end <= t)
          .sort((a, b) => b.end - a.end)[0],
        next = pl.regions
          .filter((q) => q.members.includes(id) && q.start > t)
          .sort((a, b) => a.start - b.start)[0];
      if (next && t >= deploy) {
        const from = prev ? formation(prev, slot, prev.end) : carrierAt(deploy),
          to = formation(next, slot, next.start);
        position = mix(
          from,
          to,
          clamp(
            (t - (prev?.end ?? deploy)) / (next.start - (prev?.end ?? deploy)),
          ),
        );
        state = "deployed";
        waypoint = to;
        trail = [from, position];
      } else if (prev && t < prev.end + 300) {
        const from = formation(prev, slot, prev.end);
        position = mix(from, carrierAt(t), clamp((t - prev.end) / 300));
        state = "returning";
        status = "returning";
        waypoint = carrierAt(t);
        trail = [from, position];
      }
    }
    const next = waypoint ?? position,
      ping =
        z?.label === "region_2_revised" &&
        id === "uuv_04" &&
        t >= PING &&
        t < PING + 45;
    return {
      uuv_id: id,
      status,
      deployment_state: state,
      position,
      heading_rad: Math.atan2(next.y - position.y, next.x - position.x),
      speed_mps: state === "onboard" ? 0 : status === "tracking" ? 3.6 : 4,
      energy_fraction: clamp(
        0.98 - Math.max(0, t - 30) / 9500 - (status === "returning" ? 0.03 : 0),
        0.24,
        0.98,
      ),
      remaining_range_m: Math.round(clamp(0.98 - t / 9500, 0.24, 0.98) * 18000),
      group_id: group,
      current_waypoint: waypoint,
      breadcrumb: trail,
      sensor_mode: ping ? "active" : "passive",
      reserved: state === "deployed" && status !== "returning",
      passive_range_m: 4000,
      active_range_m: 3000,
      active_capable: true,
      is_group_leader: status === "tracking" && slot === 0,
      master_connected: state !== "returning",
      connected_peer_ids: z?.members.filter((x) => x !== id) ?? [],
      communication_status:
        state === "onboard"
          ? "carrier"
          : status === "returning"
            ? "relay"
            : "mesh",
      tracked_target_id: group ? TARGET : null,
    };
  });
}

// The operational allocator may revise a region while a relay is already at
// sea, but the relay itself still has to sail.  This schedule is deliberately
// independent of `planAt`: a completed v2 task can return continuously while
// the v3 plan is already visible to the commander.
const RELAY_SCHEDULE = [...V2.regions, ...V3.regions];
function relayFormation(z: R, time: number) {
  if (z.label !== "region_2_revised") return formation(z, 0, time);
  const before = formation(V1.regions[1], 0, time);
  if (time <= REVISION) return before;
  if (time >= REVISION + 60) return formation(z, 0, time);
  return mix(before, formation(z, 0, time), (time - REVISION) / 60);
}
function relayDestination(z: R, time: number) {
  const f = relayFormation(z, time);
  return p(f.x - 1000, f.y + 850);
}
function usvsAt(t: number): USVView[] {
  return USV.map((id, i) => {
    const tasks = RELAY_SCHEDULE.filter((region) => region.relay === id);
    const active = tasks.find((region) => t >= region.start && t < region.end);
    const next = tasks.find((region) => t < region.start);
    const previous = [...tasks].reverse().find((region) => region.end <= t);
    const transitStart = next
      ? Math.max(0, next.start - 300, previous?.end ?? next.start - 300)
      : null;
    const c = carrierAt(t);
    let pos: Point2D;
    let dest: Point2D;
    let speed = 4;
    let relayActive = false;
    if (active) {
      dest = relayDestination(active, t);
      pos = dest;
      relayActive = true;
    } else if (next && transitStart != null && t >= transitStart) {
      const from =
        previous && transitStart <= previous.end + 300
          ? relayDestination(previous, previous.end)
          : carrierAt(transitStart);
      dest = relayDestination(next, next.start);
      pos = mix(
        from,
        dest,
        clamp((t - transitStart) / (next.start - transitStart)),
      );
      speed = 8;
    } else if (previous && t < previous.end + 300) {
      const from = relayDestination(previous, previous.end);
      dest = c;
      pos = mix(from, c, clamp((t - previous.end) / 300));
      speed = 6;
    } else {
      // A relay that has completed recovery remains alongside the carrier;
      // keeping a synthetic offset here would create a visible final jump.
      dest = c;
      pos = dest;
    }
    const connected =
      d(pos, c) <= 9000 && !(id === "usv_03" && t >= 1680 && t < 1740);
    return {
      usv_id: id,
      position: pos,
      heading_rad: Math.atan2(dest.y - pos.y, dest.x - pos.x),
      speed_mps: speed,
      energy_fraction: clamp(0.99 - t / 30000 - i * 0.015, 0.5, 0.99),
      deployment_state: "deployed",
      sensor_mode: "passive",
      distance_to_carrier_m: d(pos, c),
      passive_range_m: 8500,
      active_range_m: 4000,
      active_capable: true,
      relay_active: relayActive,
      connected,
      connected_peer_ids: connected
        ? [CARRIER, ...(active?.members ?? [])]
        : [],
      communication_range_m: 9000,
    };
  });
}
function quality(t: number, us: UUVView[]) {
  const n = us.filter((u) => u.status === "tracking").length;
  return n === 0
    ? 0.38
    : t >= MANEUVER && t < PING
      ? 0.49
      : t >= PING && t < PING + 60
        ? 0.49 + 0.41 * ((t - PING) / 60)
        : n >= 8
          ? 0.9
          : clamp(0.78 + 0.06 * Math.sin(t / 120));
}
function estimateView(t: number, q: number): TargetEstimateView {
  const pl = planAt(t),
    // The estimator keeps its final v2 corridor during the post-plan coast;
    // it must never jump back to the obsolete v1 direction at completion.
    v = pl?.version ?? (t < REVISION ? 1 : t < 2700 ? 2 : 3),
    maneuverAge = Math.max(0, t - MANEUVER),
    maneuverUncertainty =
      2000 *
      (1 - Math.exp(-maneuverAge / 30)) *
      Math.exp(-Math.max(0, maneuverAge - 90) / 210),
    postPlan = t >= 2400 && t < 2700,
    baselineMajor = 500 + (1 - q) * 1100,
    baselineMinor = 240 + (1 - q) * 520,
    major = postPlan
      ? 1100 + Math.min(2400, (t - 2400) * 2.5)
      : baselineMajor + maneuverUncertainty,
    minor = postPlan
      ? 520 + Math.min(1300, (t - 2400) * 1.25)
      : baselineMinor + maneuverUncertainty * 0.24,
    samples = Array.from({ length: 7 }, (_, i) => predicted(v, t + i * 300));
  return {
    target_id: TARGET,
    mean: estimate(t),
    covariance_ellipse: {
      semimajor_m: major,
      semiminor_m: minor,
      rotation_rad: headingAt(truth, t),
    },
    intent: {
      label: t >= MANEUVER ? "evade" : "transit",
      confidence: clamp(0.55 + q * 0.35),
      alternatives:
        t >= MANEUVER
          ? { evade: 0.76, transit: 0.18 }
          : { transit: 0.78, evade: 0.12 },
    },
    prediction: {
      horizon_s: 1800,
      sample_step_s: 300,
      centerline_xy: samples,
      radius_m: samples.map((_, i) => (v === 1 ? 620 + i * 90 : 1250 - i * 70)),
    },
    quality: {
      quality_score: q,
      estimated_rmse_m: Math.round(Math.hypot(major, minor)),
      fim_min_eigenvalue: 0.001 + q * 0.004,
      fim_condition: 12 + (1 - q) * 56,
    },
    classification: t < 180 ? "unknown" : "submarine",
    last_ping_s: t >= PING && t < PING + 45 ? PING : null,
    detection_range_m: 5000,
    detected_platform_ids: t >= PING ? ["uuv_04"] : [],
  };
}
function groupsAt(us: UUVView[], q: number): GroupView[] {
  return [...new Set(us.filter((u) => u.group_id).map((u) => u.group_id!))].map(
    (id) => ({
      group_id: id,
      target_id: TARGET,
      member_ids: us.filter((u) => u.group_id === id).map((u) => u.uuv_id),
      quality: {
        instant: q,
        window_mean: clamp(q - 0.02),
        ewma: clamp(q - 0.01),
        components: {
          geometry: clamp(q - 0.04),
          connectivity: 0.9,
          energy: 0.82,
          bearing_diversity: clamp(q - 0.06),
        },
        hard_guard_reasons:
          q < 0.65 ? ["目标机动后方位交叉角不足，等待核验与中继校准"] : [],
      },
    }),
  );
}
function regionalAt(
  t: number,
  q: number,
  groups: GroupView[],
): RegionalPlanView | null {
  const pl = planAt(t);
  if (!pl) return null;
  // Current operational map only retains a completed region briefly. Regions
  // removed by a revision are absent entirely; their replacement is recorded
  // in plan_timeline/ledger rather than being mislabeled as handed_off.
  const visible = pl.regions.filter((region) => t < region.end + REGION_FADE_S);
  const rs = visible.map((z, i) => {
    const g = groups.find((x) => x.group_id === team(z)),
      past = t >= z.end,
      cal = t >= z.start && t < z.handoff,
      active = t >= z.handoff && t < z.end,
      relayOutage = z.relay === "usv_03" && t >= 1680 && t < 1740,
      status = past
        ? "handed_off"
        : relayOutage
          ? "degraded"
          : active && g
            ? "active"
            : cal && g && q >= 0.65
              ? "handoff_ready"
              : cal && g
                ? "degraded"
                : "planned",
      es: TrackingEffectStatus =
        status === "handed_off"
          ? "handoff_ready"
          : (status as TrackingEffectStatus),
      // A planner commits v2 atomically, but an operator must not see the
      // map's corridor cells teleport.  Over its first minute, the revised
      // cells fan out from their v1 predecessor; the data stays valid region
      // tasks throughout and then settles at the v2 geometry.
      revisionBlend =
        pl.version === 2 ? clamp((t - REVISION) / REVISION_RENDER_BLEND_S) : 1,
      predecessor =
        z.label === "region_2_revised"
          ? V1.regions[1]
          : z.label === "region_3a" ||
              z.label === "region_3b" ||
              z.label === "region_4"
            ? V1.regions[2]
            : null,
      finalCenter = predicted(
        pl.version === 1 || z.label === "region_1" ? 1 : pl.version,
        z.center,
      ),
      c =
        predecessor && revisionBlend < 1
          ? mix(predicted(1, predecessor.center), finalCenter, revisionBlend)
          : finalCenter,
      finalHalfWidth = pl.version === 2 && i > 1 ? 1200 : 1050,
      h =
        predecessor && revisionBlend < 1
          ? 1050 + (finalHalfWidth - 1050) * revisionBlend
          : finalHalfWidth,
      geometry = [
        p(c.x - h, c.y - h),
        p(c.x + h, c.y - h),
        p(c.x + h, c.y + h),
        p(c.x - h, c.y + h),
      ];
    return {
      region_id: z.id,
      display_name: z.label,
      target_id: TARGET,
      geometry,
      grid_x: i,
      grid_y: pl.version === 2 && i > 1 ? i - 2 : 0,
      start_time_s: z.start,
      end_time_s: z.end,
      visit_window_index: 0,
      visit_window: { start_s: z.start, end_s: z.end },
      predecessor_region_ids: i ? [visible[i - 1].id] : [],
      successor_region_ids: i < visible.length - 1 ? [visible[i + 1].id] : [],
      assigned_uuv_ids: [...z.members],
      assigned_usv_ids: [z.relay],
      tracking_mode: "uuv_primary_usv_relay" as const,
      uuv_roles: [
        "active_verifier",
        "passive_tracker",
        "passive_tracker",
        "handoff_reserve",
      ] as ("active_verifier" | "passive_tracker" | "handoff_reserve")[],
      usv_role: "surface_relay" as const,
      sonar_policy: {
        passive_required: true,
        active_allowed: true,
        active_mode: "probe" as const,
        active_cooldown_s: 60,
      },
      communication: {
        carrier_to_uuv: true,
        usv_relay_required: true,
        acoustic_link_required: true,
        relay_overlap_policy: "adjacent_connected" as const,
      },
      communication_links: [
        `${CARRIER}->${z.relay}`,
        ...z.members.map((id) => `${z.relay}->${id}`),
      ],
      relay_usv_ids: [z.relay],
      group_id: g?.group_id ?? null,
      status,
      revision: pl.version,
      effect: {
        status: es,
        coverage_ratio: active ? 1 : cal ? 0.75 : 0,
        quality_score: g?.quality.ewma ?? 0,
        handoff_progress: past
          ? 1
          : cal
            ? clamp((t - z.start) / (z.handoff - z.start))
            : 0,
        quality_source: "group_quality_proxy" as const,
        hard_guard_reasons:
          (relayOutage
            ? ["USV-03 进入波浪遮蔽区，水面中继暂时断开"]
            : g?.quality.hard_guard_reasons) ??
          (status === "planned" ? ["等待编队进入预测拦截区"] : []),
        expert_feedback_ids: [],
      },
    };
  });
  const current =
      rs.find((x) => x.status === "active") ??
      rs.find((x) => x.status === "handoff_ready" || x.status === "degraded"),
    next = rs.find((x) => x.start_time_s > t);
  return {
    target_id: TARGET,
    prediction_id: pl.id,
    revision: pl.version,
    cell_size_m: pl.version === 2 ? 500 : 750,
    grid_spec: {
      origin_xy: [0, 0],
      map_coordinate_convention: "global_xy_m",
      target_grid_cells: 64,
      min_cell_size_m: 125,
      max_cell_size_m: 2000,
      cell_size_rounding_m: 50,
      lateral_half_width_cells: 2,
      max_uncertainty_margin_cells: pl.version === 2 ? 2 : 1,
      require_uuv_per_region: false,
      require_usv_per_region: false,
      relay_overlap_policy: "adjacent_connected",
    },
    evidence_ids: [
      `${TARGET}:prediction:v${pl.version}`,
      `${TARGET}:bearing-window`,
    ],
    current_handoff_region_id: current?.region_id ?? null,
    next_handoff_region_id: next?.region_id ?? null,
    causal_event_ids:
      pl.version === 2 ? ["mock-maneuver", "mock-revision"] : ["mock-plan-v1"],
    llm_hashes: [
      `mock-regional-prompt-v${pl.version}`,
      `mock-regional-response-v${pl.version}`,
    ],
    regions: rs,
  };
}
function linksAt(
  c: Point2D,
  ss: USVView[],
  us: UUVView[],
): CommunicationLinkView[] {
  const out: CommunicationLinkView[] = [];
  ss.forEach((s) => {
    const x = d(c, s.position);
    out.push({
      source_id: CARRIER,
      target_id: s.usv_id,
      medium: "surface",
      distance_m: x,
      limit_m: 9000,
      status: x <= 9000 ? "connected" : "disconnected",
      relay: true,
    });
    us.filter(
      (u) => u.status === "tracking" && d(s.position, u.position) <= 5000,
    ).forEach((u) =>
      out.push({
        source_id: s.usv_id,
        target_id: u.uuv_id,
        medium: "acoustic",
        distance_m: d(s.position, u.position),
        limit_m: 5000,
        status: "connected",
        relay: true,
      }),
    );
  });
  return out;
}
function eventsAt(t: number): EventView[] {
  const e: Record<number, EventView[]> = {
    0: [
      {
        event_id: "mock-found",
        sim_time_s: 0,
        event_type: "target_found",
        level: "strategic",
        entity_id: TARGET,
        message: "USV 被动节点建立低置信度方位航迹。",
      },
    ],
    30: [
      {
        event_id: "mock-alpha",
        sim_time_s: 30,
        event_type: "deployment_started",
        level: "tactical",
        entity_id: CARRIER,
        message: "投放 A 编队前往首个预测拦截区。",
      },
    ],
    300: [
      {
        event_id: "mock-plan-v1",
        sim_time_s: 300,
        event_type: "plan_committed",
        level: "strategic",
        entity_id: TARGET,
        message: "提交 v1：三段预测走廊与中继预置。",
      },
    ],
    780: [
      {
        event_id: "mock-ab",
        sim_time_s: 780,
        event_type: "handoff_ready",
        level: "tactical",
        entity_id: TARGET,
        message: "A/B 进入并行方位校准窗口。",
      },
    ],
    840: [
      {
        event_id: "mock-maneuver",
        sim_time_s: 840,
        event_type: "target_maneuver_detected",
        level: "strategic",
        entity_id: TARGET,
        message: "目标偏离 v1 走廊，协方差沿新航向扩大。",
      },
    ],
    855: [
      {
        event_id: "mock-ping",
        sim_time_s: 855,
        event_type: "active_ping",
        level: "tactical",
        entity_id: "uuv_04",
        message: "B 编队执行一次受控主动核验。",
      },
    ],
    870: [
      {
        event_id: "mock-revision",
        sim_time_s: 870,
        event_type: "prediction_revision",
        level: "strategic",
        entity_id: TARGET,
        message: "v2 撤销旧远端区域，新增 C1/C2/D，并重分配 USV 中继。",
      },
    ],
    900: [
      {
        event_id: "mock-handoff-ab",
        sim_time_s: 900,
        event_type: "handoff",
        level: "strategic",
        entity_id: TARGET,
        message: "A→B 接力完成，A 编队返航。",
      },
    ],
    1680: [
      {
        event_id: "mock-relay-degraded",
        sim_time_s: 1680,
        event_type: "degradation",
        level: "tactical",
        entity_id: "usv_03",
        message:
          "USV-03 进入波浪遮蔽区，C2 区域中继短时断开，任务降级但保持 UUV 被动测向。",
      },
    ],
    1740: [
      {
        event_id: "mock-relay-restored",
        sim_time_s: 1740,
        event_type: "handoff_ready",
        level: "tactical",
        entity_id: "usv_03",
        message: "USV-03 恢复母舰链路，C2 区域重新满足接力条件。",
      },
    ],
    1800: [
      {
        event_id: "mock-handoff-c",
        sim_time_s: 1800,
        event_type: "handoff",
        level: "tactical",
        entity_id: TARGET,
        message: "C1→C2 接力完成。",
      },
    ],
    2700: [
      {
        event_id: "mock-plan-v3",
        sim_time_s: 2700,
        event_type: "plan_revision",
        level: "strategic",
        entity_id: TARGET,
        message: "目标转入沿岸走廊；提交 v3，重新投放已回收的 A 编队。",
      },
    ],
    3450: [
      {
        event_id: "mock-handoff-v3a",
        sim_time_s: 3450,
        event_type: "handoff",
        level: "tactical",
        entity_id: TARGET,
        message: "v3 中 A→B 接力完成，USV-01 接管中继。",
      },
    ],
    4050: [
      {
        event_id: "mock-handoff-v3b",
        sim_time_s: 4050,
        event_type: "handoff",
        level: "tactical",
        entity_id: TARGET,
        message: "v3 中 B→C 接力完成，进入最终沿岸区域。",
      },
    ],
    4800: [
      {
        event_id: "mock-recovery",
        sim_time_s: 4800,
        event_type: "recovery_complete",
        level: "informational",
        entity_id: CARRIER,
        message: "v3 完成，全部 UUV 回收，母舰继续固定巡逻。",
      },
    ],
  };
  return e[t] ?? [];
}
function llmThinkingAt(t: number): string | null {
  if (t < 300) return null;
  if (t < REVISION)
    return "已由 USV 的被动方位观测建立初始航迹。当前优先保持 A 组的测向几何，同时让 B、C 编队在预测走廊的后续时间窗前出；这样可在交接重叠期内持续缩小定位误差，并避免第一组因续航约束提前退出后出现跟踪空窗。";
  if (t < 2700)
    return "观测残差表明目标已偏离 v1 预测，协方差不确定性正在扩大。主动核验确认转向后，原有远端区域不再具有足够信息价值，因此将走廊拆分为 C1、C2 和 D；USV-03 前出维持中继，使新旧编队能够在接力窗口并行校准。";
  if (t < 4500)
    return "v2 的资源已完成回收，最新方位序列显示目标进入沿岸走廊。当前按新的未来时间窗重新投放 A、B、C 编队：A 组先建立近端观测，B 组提前进入中段拦截区，C 组在远端准备接替；USV-01 与 USV-02 保持相邻区域的通信连续性。";
  return "沿岸接力已完成，当前评估重点从区域接替转为返航回收与链路闭环确认。所有 UUV 将依次回收至母舰，母舰继续固定外圈巡逻，并保留对后续异常接触的再部署能力。";
}
function llmThinkingTriggerAt(t: number): string | null {
  if (t < 300) return null;
  if (t < REVISION)
    return "USV 被动声学接触达到建轨门限，且初始方位交叉角满足持续跟踪条件。";
  if (t < 2700)
    return "航迹创新量连续超阈值；UUV-04 的主动核验确认目标向北侧逃逸走廊转向。";
  if (t < 4500)
    return "v2 回收窗口完成，新的方位序列表明目标已进入沿岸预测走廊。";
  return "最终区域接力完成，所有跟踪编队已进入返航回收窗口。";
}
function operationalStageFlagsAt(t: number): OperationalStage[] {
  const flags: OperationalStage[] = ["task_execution"];
  if ((t >= MANEUVER && t < REVISION + 60) || (t >= 2700 && t < 2760))
    flags.push("event_trigger");
  if ((t >= REVISION && t < REVISION + 300) || (t >= 2700 && t < 3000))
    flags.push("dynamic_adjustment");
  // The v2 relay configuration receives an operator confirmation before its
  // first handoff window; it can coexist with execution and adjustment.
  if (t >= 1200 && t < 1500) flags.push("human_feedback");
  return flags;
}
export function createMockFrame(i: number): OperationalFrame {
  const frame = Math.max(0, Math.min(MOCK_FRAME_COUNT - 1, Math.round(i))),
    t = frame * 5,
    c = carrierAt(t),
    us = uuvsAt(t),
    ss = usvsAt(t),
    q = quality(t, us),
    est = estimateView(t, q),
    gs = groupsAt(us, q),
    rp = regionalAt(t, q, gs),
    pl = planAt(t),
    ls = linksAt(c, ss, us),
    carrier: CarrierView = {
      carrier_id: CARRIER,
      position: c,
      heading_rad: headingAt(carrierAt, t),
      speed_mps: 5,
      status: us.some((u) => u.deployment_state === "returning")
        ? "recovering"
        : us.some((u) => u.deployment_state === "onboard") &&
            us.some((u) => u.deployment_state === "deployed")
          ? "deploying"
          : "transit",
      onboard_uuv_ids: us
        .filter((u) => u.deployment_state === "onboard")
        .map((u) => u.uuv_id),
      deployed_uuv_ids: us
        .filter((u) => u.deployment_state === "deployed")
        .map((u) => u.uuv_id),
      returning_uuv_ids: us
        .filter((u) => u.deployment_state === "returning")
        .map((u) => u.uuv_id),
      support_radius_m: 16000,
    },
    pv: PlanView | null = pl
      ? {
          plan_id: `mock-plan-v${pl.version}`,
          version: pl.version,
          status:
            t >= (pl.version === 2 ? 2400 : 4500) ? "completed" : "active",
          concept:
            pl.version === 2
              ? "quality_first"
              : pl.version === 3
                ? "resource_saving"
                : "balanced",
          reason:
            pl.version === 2
              ? "目标机动触发预测修订、区域重构与 USV 再分配"
              : pl.version === 3
                ? "沿岸转向后的资源轮换与第二轮分段接力"
                : "初始纯方位分段接力",
          affected_targets: [TARGET],
          group_changes: gs.map(
            (g) => `${g.group_id}: ${g.member_ids.join(", ")}`,
          ),
          valid_from_s:
            pl.version === 1 ? 300 : pl.version === 2 ? REVISION : 2700,
          valid_until_s:
            pl.version === 1 ? REVISION : pl.version === 2 ? 2400 : 4500,
          segment_plan: pl.regions
            .filter((z) => z.end > t)
            .map((z) => `${z.label}:${z.start}-${z.end}`),
        }
      : null,
    adv: AdversaryView = {
      target_id: TARGET,
      sim_time_s: t,
      detection_range_m: 5000,
      detected_platform_ids: est.detected_platform_ids ?? [],
      trigger_event_ids: t >= MANEUVER ? ["mock-maneuver"] : ["mock-found"],
      decision_id: `mock-adversary-${Math.floor(t / 300)}`,
      maneuver: t >= MANEUVER ? "转向北侧逃逸走廊" : "保持东北航渡",
      intent: t >= MANEUVER ? "evade" : "transit",
      segment: pl?.id ?? "pre-track",
      speed_mps: t >= MANEUVER ? 6.2 : 3.2,
      heading_rad: headingAt(truth, t),
      decoy_count: 0,
      confidence: clamp(0.55 + q * 0.32),
      rationale: "仅依据可观测接触和主动声呐暴露风险机动。",
      communications_discipline: "silent",
      decision_status: q >= 0.65 ? "contact_maintained" : "uncertain",
    };
  const rays: BearingRayView[] = us
    .filter(
      (u) =>
        u.status === "tracking" &&
        d(u.position, truth(t)) >= 250 &&
        d(u.position, truth(t)) <= (u.passive_range_m ?? 0),
    )
    .map((u, n) => ({
      observation_id: `${TARGET}:bearing:${u.uuv_id}:${t}`,
      uuv_id: u.uuv_id,
      target_id: TARGET,
      origin: u.position,
      azimuth_rad:
        Math.atan2(est.mean.y - u.position.y, est.mean.x - u.position.x) +
        (n - 2) * 0.009,
      variance_rad2: 0.0007 + n * 0.00008,
      confidence: clamp(0.94 - n * 0.035),
    }));
  // The drawer expects a small rolling history for every metric.  Keep these
  // derived from the same state that drives the map rather than inventing a
  // second, unrelated set of dashboard values.
  const ellipseMajor = est.covariance_ellipse.semimajor_m;
  const connectedRelayFraction = ss.length
    ? ss.filter((usv) => usv.connected).length / ss.length
    : 0;
  const visibleRegions = rp?.regions ?? [];
  const coveredRegions = visibleRegions.filter(
    (region) =>
      region.status === "active" ||
      region.status === "handoff_ready" ||
      region.status === "handed_off",
  ).length;
  const regionalCoverage = visibleRegions.length
    ? coveredRegions / visibleRegions.length
    : 0;
  const metricSeries = (value: number, amplitude: number) =>
    Array.from({ length: 20 }, (_, index) =>
      Math.max(0, value + Math.sin((t / 75 + index) * 0.72) * amplitude),
    );
  const metrics: MetricView[] = [
    {
      metric_id: "tracking_quality",
      label: "跟踪质量 EWMA",
      value: q,
      unit: "score",
      threshold: 0.65,
      window_s: 120,
      series: metricSeries(q, 0.035),
      status: q >= 0.65 ? "OK" : "warning",
      mean_window: q,
      worst_window: q - 0.05,
      trend_per_sec: 0,
      valid_fraction: q >= 0.65 ? 1 : 0.6,
      reason: "方位交叉角、通信中继与协方差代理",
    },
    {
      metric_id: "regional_coverage",
      label: "分段区域覆盖率",
      value: regionalCoverage,
      unit: "ratio",
      threshold: 0.8,
      window_s: 180,
      series: metricSeries(regionalCoverage, 0.05),
      status:
        regionalCoverage >= 0.8
          ? "OK"
          : visibleRegions.length
            ? "warning"
            : "idle",
      mean_window: regionalCoverage,
      worst_window: Math.max(0, regionalCoverage - 0.08),
      trend_per_sec: 0,
      valid_fraction: visibleRegions.length ? 1 : 0,
      reason: visibleRegions.length
        ? "按当前可见区域的 active / handoff 状态计算。"
        : "当前处于回收、重整或下一轮投放准备窗口。",
    },
    {
      metric_id: "relay_link_health",
      label: "USV 中继可用率",
      value: connectedRelayFraction,
      unit: "ratio",
      threshold: 0.75,
      window_s: 120,
      series: metricSeries(connectedRelayFraction, 0.02),
      status: connectedRelayFraction >= 0.75 ? "OK" : "warning",
      mean_window: connectedRelayFraction,
      worst_window: Math.max(
        0,
        connectedRelayFraction - (t >= 1680 && t < 1740 ? 0.25 : 0.03),
      ),
      trend_per_sec: 0,
      valid_fraction: connectedRelayFraction,
      reason:
        t >= 1680 && t < 1740
          ? "USV-03 短时链路中断，C2 区域降级。"
          : "母舰—USV—UUV 中继链路正常。",
    },
    {
      metric_id: "prediction_uncertainty",
      label: "预测椭圆长半轴",
      value: ellipseMajor,
      unit: "m",
      threshold: 1600,
      window_s: 180,
      series: metricSeries(ellipseMajor, Math.min(180, ellipseMajor * 0.06)),
      status: ellipseMajor <= 1600 ? "OK" : "warning",
      mean_window: ellipseMajor,
      worst_window: ellipseMajor + 120,
      trend_per_sec: 0,
      valid_fraction: 1,
      reason:
        "由当前纯方位估计协方差生成，并随目标机动、主动核验和交接几何变化。",
    },
  ];
  const timeline: PlanTimelineView[] = [
    ...(t >= 300
      ? [
          {
            adjustment_id: "mock-v1",
            sim_time_s: 300,
            factors: [
              {
                kind: "event" as const,
                ref_id: "mock-found",
                label: "初始方位航迹",
                detail: "USV 被动观测",
              },
            ],
            plan: {
              plan_id: "mock-plan-v1",
              version: 1,
              status: "superseded" as const,
              summary: "三段接力",
              group_changes: ["A→B→C"],
            },
          },
        ]
      : []),
    ...(t >= REVISION
      ? [
          {
            adjustment_id: "mock-v2",
            sim_time_s: REVISION,
            factors: [
              {
                kind: "event" as const,
                ref_id: "mock-maneuver",
                label: "目标机动",
                detail: "走廊偏离和椭圆扩大",
              },
              {
                kind: "event" as const,
                ref_id: "mock-ping",
                label: "主动核验",
                detail: "恢复交接几何",
              },
            ],
            plan: {
              plan_id: "mock-plan-v2",
              version: 2,
              status: t >= 2700 ? ("completed" as const) : ("active" as const),
              summary: "区域拆分与 USV 重分配",
              group_changes: ["C→C1/C2/D", "USV-03 前出"],
            },
          },
        ]
      : []),
    ...(t >= 2700
      ? [
          {
            adjustment_id: "mock-v3",
            sim_time_s: 2700,
            factors: [
              {
                kind: "event" as const,
                ref_id: "mock-plan-v3",
                label: "沿岸转向",
                detail: "目标进入新走廊，已回收资源重新轮换",
              },
            ],
            plan: {
              plan_id: "mock-plan-v3",
              version: 3,
              status: t >= 4500 ? ("completed" as const) : ("active" as const),
              summary: "第二轮分段接力与资源轮换",
              group_changes: ["A/B/C 重新投放", "USV-01、USV-02 接续中继"],
            },
          },
        ]
      : []),
  ];
  const brains: BrainView[] = [
    {
      brain_id: "master-brain",
      role: "master",
      status: "online",
      last_update_s: t,
      message:
        t >= MANEUVER && t < REVISION
          ? "检测到偏差，重算预测、区域和资源。"
          : pl?.version === 3
            ? "执行 v3 沿岸走廊接力与第二轮投放。"
            : pl?.version === 2
              ? "执行 v2 动态区域和中继方案。"
              : "维护初始纯方位接力。",
      connected_platform_ids: [
        CARRIER,
        ...ss.filter((s) => s.connected).map((s) => s.usv_id),
      ],
    },
    {
      brain_id: "slave-brain-target",
      role: "slave",
      status: gs.length ? "online" : "paused",
      last_update_s: gs.length ? t : null,
      message:
        gs.length > 1
          ? "并行编队校准交叉角与中继。"
          : gs.length
            ? "维持当前区域测向几何。"
            : "等待 UUV 入区。",
      connected_platform_ids: gs.flatMap((g) => g.member_ids),
    },
    {
      brain_id: "adversary-brain",
      role: "adversary",
      status: "online",
      last_update_s: t,
      message: adv.maneuver ?? "保持航渡",
      connected_platform_ids: [],
    },
  ];
  const suggestions: PlanAdjustmentSuggestionView[] = [
    {
      suggestion_id: "mock-regional",
      category: "segmented_handoff",
      title: "维持滚动预测区域",
      rationale:
        pl?.version === 3
          ? "v3 已将回收完成的 A/B/C 编队重新投放至沿岸走廊。"
          : pl?.version === 2
            ? "v2 已拆分转向后的远端走廊。"
            : "等待更多方位观测。",
      proposed_feedback: "区域、UUV 航点和 USV 中继必须随版本共同更新。",
      target_ids: [TARGET],
      evidence_ids: rp?.evidence_ids ?? [],
      confidence: q,
    },
  ];
  const regionTimeline: RegionTimelineView[] = rp
    ? rp.regions.map((x) => ({
        region_id: x.region_id,
        target_id: x.target_id,
        center: p(
          (x.geometry[0].x + x.geometry[2].x) / 2,
          (x.geometry[0].y + x.geometry[2].y) / 2,
        ),
        bounds: {
          min_x: x.geometry[0].x,
          min_y: x.geometry[0].y,
          max_x: x.geometry[2].x,
          max_y: x.geometry[2].y,
        },
        start_offset_s: x.start_time_s - t,
        end_offset_s: x.end_time_s - t,
        status:
          x.status === "handoff_ready"
            ? "planned"
            : x.status === "handed_off"
              ? "handed_off"
              : (x.status as "planned" | "active" | "degraded" | "uncovered"),
        coverage_mode: x.grid_x === 0 ? "required" : "reserve",
        priority: x.grid_x === 0 ? 1 : 0.8,
        occupancy_likelihood: x.grid_y === 0 ? 0.82 : 0.66,
        uuv_assignments: x.assigned_uuv_ids.map((id, n) => ({
          platform_id: id,
          platform_kind: "uuv" as const,
          role: x.uuv_roles?.[n] ?? "passive_tracker",
          start_offset_s: x.start_time_s - t,
          end_offset_s: x.end_time_s - t,
          sonar_mode: n === 0 ? ("active" as const) : ("passive" as const),
        })),
        usv_assignments: x.assigned_usv_ids.map((id) => ({
          platform_id: id,
          platform_kind: "usv" as const,
          role: "surface_relay",
          start_offset_s: x.start_time_s - t,
          end_offset_s: x.end_time_s - t,
          sonar_mode: "passive" as const,
        })),
        communication_links: ls.filter((l) =>
          x.communication_links?.includes(`${l.source_id}->${l.target_id}`),
        ),
        handoff_from: x.predecessor_region_ids[0] ?? null,
        handoff_to: x.successor_region_ids[0] ?? null,
        evidence_ids: [],
        degraded_reasons: x.effect.hard_guard_reasons,
        plan_revision: x.revision ?? pl!.version,
      }))
    : [];
  return {
    schema_version: "1.0",
    frame_id: frame,
    sim_time_s: t,
    physics_step_s: 5,
    plan_version: pl?.version ?? 0,
    map_bounds: MOCK_MAP_BOUNDS,
    carrier,
    uuvs: us,
    usvs: ss,
    communication_links: ls,
    brains,
    adversaries: [adv],
    target_estimates: [est],
    bearing_rays: rays,
    groups: gs,
    regional_plans: rp ? { [TARGET]: rp } : {},
    events: eventsAt(t),
    plans: pv ? [pv] : [],
    ledger: [
      ...(t >= 300
        ? [
            {
              decision_id: "mock-decision-v1",
              sim_time_s: 300,
              outcome: "committed" as const,
              trigger_event_ids: ["mock-found"],
              evidence_ids: [`${TARGET}:bearing-window`],
              final_plan_id: "mock-plan-v1",
              final_plan_version: 1,
            },
          ]
        : []),
      ...(t >= REVISION
        ? [
            {
              decision_id: "mock-decision-v2",
              sim_time_s: REVISION,
              outcome: "committed" as const,
              trigger_event_ids: ["mock-maneuver", "mock-ping"],
              evidence_ids: [`${TARGET}:prediction:v2`, "mock-ping"],
              final_plan_id: "mock-plan-v2",
              final_plan_version: 2,
            },
          ]
        : []),
      ...(t >= 2700
        ? [
            {
              decision_id: "mock-decision-v3",
              sim_time_s: 2700,
              outcome: "committed" as const,
              trigger_event_ids: ["mock-plan-v3"],
              evidence_ids: [`${TARGET}:prediction:v3`, "mock-recovery"],
              final_plan_id: "mock-plan-v3",
              final_plan_version: 3,
            },
          ]
        : []),
    ],
    metrics,
    scheme: pv
      ? {
          scheme_id: `mock-scheme-v${pl!.version}`,
          version: pl!.version,
          valid_from_s: pv.valid_from_s,
          valid_until_s: pv.valid_until_s ?? 2400,
          target_priorities: { [TARGET]: 1 },
          minimum_quality: { [TARGET]: 0.65 },
          constraints: [
            "方位交叉角须达标",
            "USV 中继必须连通",
            "主动声呐仅质量异常时启用",
            "母舰固定外圈巡逻",
          ],
        }
      : null,
    intelligence: [
      {
        report_id: `mock-intel-${Math.floor(t / 300)}`,
        source: "sonar",
        target_id: TARGET,
        confidence: q,
        issued_at_s: Math.max(0, t - 30),
        valid_until_s: t + 300,
        content_summary:
          t >= MANEUVER
            ? "方位创新量支持目标转向；已触发滚动重规划。"
            : "被动声学特征与潜艇类别一致。",
      },
    ],
    plan_timeline: timeline,
    region_timeline: regionTimeline,
    plan_adjustment_suggestions: suggestions,
    operational_stage_flags: operationalStageFlagsAt(t),
    llm_thinking: llmThinkingAt(t),
    llm_thinking_trigger: llmThinkingTriggerAt(t),
  };
}
export const MOCK_FRAMES = Array.from({ length: MOCK_FRAME_COUNT }, (_, i) =>
  createMockFrame(i),
);
export function getMockFrames(startS = 0, endS?: number) {
  return MOCK_FRAMES.filter(
    (f) => f.sim_time_s >= startS && f.sim_time_s <= (endS ?? Infinity),
  );
}
