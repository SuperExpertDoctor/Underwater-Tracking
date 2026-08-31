import type { PlanView } from "../../types/frames";

export interface SegmentOverlayProps {
  plans: PlanView[];
}

export default function SegmentOverlay({ plans }: SegmentOverlayProps) {
  const plan = plans.find((candidate) => candidate.status === "active") ?? plans[0];
  const segments = plan?.segment_plan ?? [];
  return (
    <div className="segment-overlay" aria-label="分段接力方案">
      <div className="section-heading">
        <span>分段接力</span>
        <small>{segments.length} 段</small>
      </div>
      {segments.length === 0 ? (
        <p>尚未生成分段接力方案</p>
      ) : (
        <ol className="segment-list">{segments.map((segment, index) => <li key={`${segment}-${index}`}>{segment}</li>)}</ol>
      )}
    </div>
  );
}
