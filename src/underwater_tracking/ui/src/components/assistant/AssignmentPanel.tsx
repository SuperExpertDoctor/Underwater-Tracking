import { useState } from "react";
import type {
  TargetEstimateView,
  UUVView,
  RegionalPlanView,
} from "../../types/frames";
import RegionTaskGraph from "./RegionTaskGraph";
import { displayTargetName } from "../../utils/presentation";
import {
  scanCoveragePercent,
  scanPingCount,
  scanRouteProgressPercent,
  scanTelemetryForRegion,
} from "../../domain/operationalStatus";

export interface AssignmentPanelProps {
  targets: TargetEstimateView[];
  uuvs: UUVView[];
  onAssign?: (uuvIds: string[], targetId: string) => void;
  regionalPlans?: Record<string, RegionalPlanView>;
  selectedRegionId?: string | null;
  onSelectRegion?: (regionId: string | null) => void;
}

const MODE_LABELS: Record<
  RegionalPlanView["regions"][number]["tracking_mode"],
  string
> = {
  heuristic_uuv: "启发式 UUV 协同",
};

const STATUS_LABELS: Record<string, string> = {
  planned: "待执行",
  active: "跟踪中",
  handoff_ready: "接力就绪",
  degraded: "效果下降",
  uncovered: "未覆盖",
};

type AssignmentView = "graph" | "list";

export default function AssignmentPanel({
  targets,
  uuvs,
  regionalPlans,
  selectedRegionId,
  onSelectRegion,
}: AssignmentPanelProps) {
  const [view, setView] = useState<AssignmentView>("graph");
  const plans = Object.values(regionalPlans ?? {});
  const plan =
    plans.find((candidate) =>
      targets.some((target) => target.target_id === candidate.target_id),
    ) ?? plans[0];
  const targetId = plan?.target_id ?? targets[0]?.target_id ?? "";

  return (
    <section className="assignment-panel" aria-label="区域任务审查">
      <div className="section-heading">
        <span>区域任务审查</span>
        <small>LLM 自主编组 · 目标 {displayTargetName(targetId)}</small>
      </div>
      <p className="assignment-explanation">
        编组数量和平台角色由 LLM
        根据预测区域、通信条件与目标机动动态决定；此处只审查任务，不施加固定配额。
      </p>
      {plan ? (
        <>
          <div
            className="assignment-view-tabs"
            role="group"
            aria-label="区域任务视图"
          >
            {(["graph", "list"] as const).map((candidate) => (
              <button
                key={candidate}
                type="button"
                aria-pressed={view === candidate}
                onClick={() => setView(candidate)}
              >
                {candidate === "graph" ? "图谱" : "列表"}
              </button>
            ))}
          </div>
          {view === "graph" && (
            <>
              <RegionTaskGraph
                plan={plan}
                selectedRegionId={selectedRegionId}
                onSelectRegion={onSelectRegion}
              />
              <AssignmentEffects plan={plan} />
            </>
          )}
          {view === "list" && (
            <AssignmentEffects
              plan={plan}
              selectedRegionId={selectedRegionId}
              onSelectRegion={onSelectRegion}
              interactive
            />
          )}
        </>
      ) : (
        <div className="assignment-empty">
          当前帧尚未生成目标预测区域，等待 LLM 规划。
        </div>
      )}
      {uuvs.length === 0 && (
        <small className="adaptive-muted">
          当前帧未接入 UUV，区域任务仍由计划状态展示。
        </small>
      )}
    </section>
  );
}

function AssignmentEffects({
  plan,
  selectedRegionId,
  onSelectRegion,
  interactive = false,
}: {
  plan: RegionalPlanView;
  selectedRegionId?: string | null;
  onSelectRegion?: (regionId: string | null) => void;
  interactive?: boolean;
}) {
  return (
    <div
      className={`assignment-effect-list ${interactive ? "interactive" : ""}`}
      aria-label="区域跟踪效果"
    >
      {plan.regions.map((region) => {
        const entityCount = region.assigned_uuv_ids.length;
        const telemetry = scanTelemetryForRegion(region);
        const routeProgress = scanRouteProgressPercent(telemetry);
        const activeCoverage = scanCoveragePercent(telemetry);
        const pingCount = scanPingCount(telemetry);
        const quality = region.effect.quality_score == null
          ? "不可用"
          : `${Math.round(region.effect.quality_score * 100)}%`;
        const routeCompleted = routeProgress != null && routeProgress >= 100;
        const scanCompleted = telemetry?.scan_completed;
        const scanStatus = scanCompleted == null
          ? "扫描状态不可用"
          : scanCompleted
            ? "扫描已完成"
            : routeCompleted
              ? "路线完成 · 扫描未完成"
              : "扫描未完成";
        const handoff = region.handoff_evidence;
        const handoffProgress = handoff
          ? `${new Set(handoff.valid_observation_uuv_ids).size}/${new Set(handoff.required_uuv_ids).size}`
          : "不可用";
        const content = (
          <>
            <div className="assignment-effect-heading">
              <strong>{region.display_name}</strong>
              <span>
                {STATUS_LABELS[region.effect.status] ?? region.effect.status}
              </span>
            </div>
            <div className="assignment-effect-mode">
              {MODE_LABELS[region.tracking_mode]}
            </div>
            <div className="assignment-effect-facts">
              <span>
                路线 {routeProgress == null ? "不可用" : `${Math.round(routeProgress)}%`} · 主动覆盖 {activeCoverage == null ? "不可用" : `${Math.round(activeCoverage)}%`}
              </span>
              <span>
                source-backed ping {pingCount == null ? "不可用" : pingCount} · 质量 {quality}
              </span>
            </div>
            <div
              className={`assignment-scan-status ${scanCompleted === false && routeCompleted ? "scan-route-complete" : ""}`}
              data-scan-completed={scanCompleted == null ? undefined : String(scanCompleted)}
            >
              {scanStatus}
              {telemetry?.scan_round != null && <span> · 第 {telemetry.scan_round} 轮</span>}
            </div>
            <div className="assignment-effect-facts">
              <span>实体 {entityCount}</span>
              <span>接力证据 {handoffProgress}</span>
            </div>
            <div className="assignment-effect-members">
              {region.assigned_uuv_ids.map((id) => (
                <span key={id}>{formatEntity(id)}</span>
              ))}
            </div>
            {region.effect.hard_guard_reasons.length > 0 && (
              <small className="assignment-effect-warning">
                {region.effect.hard_guard_reasons.join("；")}
              </small>
            )}
          </>
        );
        return interactive ? (
          <button
            className={`assignment-effect status-${region.effect.status}`}
            type="button"
            key={region.region_id}
            aria-pressed={selectedRegionId === region.region_id}
            onClick={() =>
              onSelectRegion?.(
                selectedRegionId === region.region_id ? null : region.region_id,
              )
            }
          >
            {content}
          </button>
        ) : (
          <article
            className={`assignment-effect status-${region.effect.status}`}
            key={region.region_id}
          >
            {content}
          </article>
        );
      })}
    </div>
  );
}

function formatEntity(id: string): string {
  const match = id.match(/^uuv[_-]?0*(\d+)$/i);
  return match ? `UUV_${Number(match[1])}` : id;
}
