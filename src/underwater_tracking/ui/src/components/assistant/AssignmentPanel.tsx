import { useState } from "react";
import type { OperationalFrame, TargetEstimateView, UUVView, RegionalPlanView } from "../../types/frames";
import RegionTimelinePanel from "../RegionTimelinePanel";
import RegionTaskGraph from "./RegionTaskGraph";

export interface AssignmentPanelProps {
  targets: TargetEstimateView[];
  uuvs: UUVView[];
  onAssign?: (uuvIds: string[], targetId: string) => void;
  frame?: OperationalFrame | null;
  regionalPlans?: Record<string, RegionalPlanView>;
  selectedRegionId?: string | null;
  onSelectRegion?: (regionId: string | null) => void;
}

const MODE_LABELS: Record<RegionalPlanView["regions"][number]["tracking_mode"], string> = {
  uuv_primary_usv_relay: "UUV 主跟踪 + USV 中继",
  heuristic_uuv: "启发式 UUV 协同",
  heuristic_usv: "启发式 USV 协同",
};

const STATUS_LABELS: Record<string, string> = {
  planned: "待执行",
  active: "跟踪中",
  handoff_ready: "接力就绪",
  degraded: "效果下降",
  uncovered: "未覆盖",
};

type AssignmentView = "graph" | "timeline" | "list";

export default function AssignmentPanel({ targets, uuvs, frame = null, regionalPlans, selectedRegionId, onSelectRegion }: AssignmentPanelProps) {
  const [view, setView] = useState<AssignmentView>("graph");
  const plans = Object.values(regionalPlans ?? {});
  const plan = plans.find((candidate) => targets.some((target) => target.target_id === candidate.target_id)) ?? plans[0];
  const targetId = plan?.target_id ?? targets[0]?.target_id ?? "";

  return (
    <section className="assignment-panel" aria-label="区域任务审查">
      <div className="section-heading">
        <span>区域任务审查</span>
        <small>LLM 自主编组 · 目标 {targetId || "—"}</small>
      </div>
      <p className="assignment-explanation">编组数量和平台角色由 LLM 根据预测区域、通信条件与目标机动动态决定；此处只审查任务，不施加固定配额。</p>
      {plan ? (
        <>
          <div className="assignment-view-tabs" role="group" aria-label="区域任务视图">
            {(["graph", "timeline", "list"] as const).map((candidate) => (
              <button
                key={candidate}
                type="button"
                aria-pressed={view === candidate}
                onClick={() => setView(candidate)}
              >
                {candidate === "graph" ? "图谱" : candidate === "timeline" ? "时间线" : "列表"}
              </button>
            ))}
          </div>
          {view === "graph" && <>
            <RegionTaskGraph plan={plan} selectedRegionId={selectedRegionId} onSelectRegion={onSelectRegion} />
            <AssignmentEffects plan={plan} />
          </>}
          {view === "timeline" && <RegionTimelinePanel frame={frame} selectedRegionId={selectedRegionId} onSelectRegion={onSelectRegion} />}
          {view === "list" && <AssignmentEffects plan={plan} selectedRegionId={selectedRegionId} onSelectRegion={onSelectRegion} interactive />}
        </>
      ) : (
        <div className="assignment-empty">当前帧尚未生成目标预测区域，等待 LLM 规划。</div>
      )}
      {uuvs.length === 0 && <small className="adaptive-muted">当前帧未接入 UUV，区域任务仍由计划状态展示。</small>}
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
  return <div className={`assignment-effect-list ${interactive ? "interactive" : ""}`} aria-label="区域跟踪效果">
    {plan.regions.map((region) => {
      const entityCount = region.assigned_uuv_ids.length + region.assigned_usv_ids.length;
      const coverage = Math.round(region.effect.coverage_ratio * 100);
      const quality = Math.round(region.effect.quality_score * 100);
      const content = <>
        <div className="assignment-effect-heading">
          <strong>{region.display_name}</strong>
          <span>{STATUS_LABELS[region.effect.status] ?? region.effect.status}</span>
        </div>
        <div className="assignment-effect-mode">{MODE_LABELS[region.tracking_mode]}</div>
        <div className="assignment-effect-facts">
          <span>跟踪覆盖 {coverage}% · 质量 {quality}%</span>
          <span>实体 {entityCount} · 接力 {Math.round(region.effect.handoff_progress * 100)}%</span>
        </div>
        <div className="assignment-effect-members">
          {[...region.assigned_uuv_ids, ...region.assigned_usv_ids].map((id) => <span key={id}>{formatEntity(id)}</span>)}
          {region.effect.expert_feedback_ids.length > 0 && <small>专家反馈 {region.effect.expert_feedback_ids.length} 条</small>}
        </div>
        {region.effect.hard_guard_reasons.length > 0 && <small className="assignment-effect-warning">{region.effect.hard_guard_reasons.join("；")}</small>}
      </>;
      return interactive ? (
        <button
          className={`assignment-effect status-${region.effect.status}`}
          type="button"
          key={region.region_id}
          aria-pressed={selectedRegionId === region.region_id}
          onClick={() => onSelectRegion?.(selectedRegionId === region.region_id ? null : region.region_id)}
        >
          {content}
        </button>
      ) : <article className={`assignment-effect status-${region.effect.status}`} key={region.region_id}>{content}</article>;
    })}
  </div>;
}

function formatEntity(id: string): string {
  const match = id.match(/^(uuv|usv)[_-]?0*(\d+)$/i);
  return match ? `${match[1].toUpperCase()}_${Number(match[2])}` : id;
}
