import { GitCompareArrows } from "lucide-react";
import type {
  PredictionDiffView,
  TargetEstimateView,
} from "../types/frames";
import { displayTargetName } from "../utils/presentation";

const STATE_LABEL: Record<PredictionDiffView["state"], string> = {
  stable: "轨迹稳定",
  accumulating: "变化累积",
  suspected: "疑似行为变化",
  verifying: "意图核验中",
  confirmed: "意图已改变",
  reset: "变化已解除",
  unavailable: "证据不足",
};

const REASON_LABEL: Record<string, string> = {
  insufficient_overlap: "重叠走廊不足",
  missing_previous_prediction: "等待前序预测",
  invalid_covariance: "预测协方差无效",
  invalid_prediction: "预测数据无效",
};

interface PredictionDiffPanelProps {
  targets: TargetEstimateView[];
}

export default function PredictionDiffPanel({
  targets,
}: PredictionDiffPanelProps) {
  const evidence = targets.flatMap((target) => {
    const diff = target.prediction?.diff;
    return diff ? [{ target, diff }] : [];
  });

  if (evidence.length === 0) return null;

  return (
    <section className="prediction-diff-panel" aria-label="预测轨迹分歧">
      <div className="prediction-diff-heading">
        <span>
          <GitCompareArrows size={14} aria-hidden="true" />
          预测轨迹分歧
        </span>
        <small>{`${evidence.length} 个目标`}</small>
      </div>
      <div className="prediction-diff-list">
        {evidence.map(({ target, diff }) => (
          <article
            className={`prediction-diff-target state-${diff.state}`}
            key={diff.diff_id}
          >
            <div className="prediction-diff-status">
              <strong>{displayTargetName(target.target_id)}</strong>
              <span>{STATE_LABEL[diff.state]}</span>
            </div>
            <div className="prediction-diff-metrics">
              <Metric
                label="绝对偏移"
                value={formatMetres(diff.absolute_rms_m)}
                threshold={`下限 ${formatMetres(diff.absolute_floor_m)}`}
              />
              <Metric
                label="归一化距离"
                value={formatRatio(
                  diff.normalized_rms,
                  diff.normalized_threshold,
                )}
                threshold="观测 / 阈值"
              />
              <Metric
                label="连续周期"
                value={`${diff.consecutive_count} / ${diff.confirmation_cycles}`}
                threshold="当前 / 确认"
              />
              <Metric
                label="模型变化"
                value={diff.leading_model_changed ? "是" : "否"}
                threshold={
                  diff.js_distance == null
                    ? "JS —"
                    : `JS ${diff.js_distance.toFixed(2)}`
                }
              />
            </div>
            {diff.state === "unavailable" && diff.reason && (
              <p className="prediction-diff-reason">
                {REASON_LABEL[diff.reason] ?? diff.reason}
              </p>
            )}
            <div className="prediction-diff-provenance">
              <span title={diff.previous_prediction_id ?? "无前序预测"}>
                {shortId(diff.previous_prediction_id)}
              </span>
              <b aria-hidden="true">→</b>
              <span title={diff.current_prediction_id}>
                {shortId(diff.current_prediction_id)}
              </span>
              {diff.resulting_plan_revision != null && (
                <em>{`方案 v${diff.resulting_plan_revision}`}</em>
              )}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function Metric({
  label,
  value,
  threshold,
}: {
  label: string;
  value: string;
  threshold: string;
}) {
  return (
    <div className="prediction-diff-metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{threshold}</small>
    </div>
  );
}

function formatMetres(value: number | null) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${Number.isInteger(value) ? value : value.toFixed(1)} m`;
}

function formatRatio(value: number | null, threshold: number) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value.toFixed(2)} / ${threshold.toFixed(2)}`;
}

function shortId(value: string | null) {
  if (!value) return "—";
  return value.length > 18 ? `${value.slice(0, 8)}…${value.slice(-6)}` : value;
}
