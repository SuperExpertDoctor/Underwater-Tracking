import type { DirectiveStatus } from "../../services/assistantApi";

interface AssignmentReviewProps {
  job: DirectiveStatus;
  onConfirm: () => void;
  busy?: boolean;
  error?: string;
}

const STATUS_LABELS: Record<string, string> = {
  queued: "已排队",
  processing: "解析中",
  preview: "待确认",
  applying: "应用中",
  applied: "已应用，等待下一轮重规划",
  rejected: "已拒绝",
  needs_clarification: "需要澄清",
  error: "处理失败",
};

export default function AssignmentReview({ job, onConfirm, busy = false, error = "" }: AssignmentReviewProps) {
  const directive = job.directive;
  return (
    <section className="assistant-card assignment-review" aria-label="指派预览">
      <div className="assistant-card-heading">
        <div><span className="eyebrow">RESERVATION REVIEW</span><h2>指派预览</h2></div>
        <span className={`assistant-status status-${job.status}`}>{STATUS_LABELS[job.status] ?? job.status}</span>
      </div>
      {directive ? (
        <>
          <dl className="assignment-facts">
            <div><dt>目标</dt><dd>{(directive.assignment_target_id ?? directive.target_scope.join("、")) || "—"}</dd></div>
            <div><dt>资源</dt><dd>{directive.assignment_uuv_ids.join("、") || "—"}</dd></div>
            <div><dt>置信度</dt><dd>{Math.round(directive.confidence * 100)}%</dd></div>
          </dl>
          {directive.conflicts.length > 0 && <ul className="assignment-conflicts">{directive.conflicts.map((conflict) => <li key={conflict}>{conflict}</li>)}</ul>}
          <button className="secondary-btn" onClick={onConfirm} disabled={busy || job.status !== "preview"}>
            {busy ? "确认中…" : "确认应用指派"}
          </button>
        </>
      ) : (
        <p className="assistant-pending">正在生成指派预览，当前方案仍保持不变。</p>
      )}
      {error && <p className="assistant-error" role="alert">{error}</p>}
    </section>
  );
}
