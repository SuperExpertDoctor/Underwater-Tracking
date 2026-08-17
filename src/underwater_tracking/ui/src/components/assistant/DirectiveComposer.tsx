import { useEffect, useState } from "react";
import { Send } from "lucide-react";
import type { OperationalFrame, PlanAdjustmentSuggestionView } from "../../types/frames";
import {
  applyDirective,
  AssistantApiError,
  getDirectiveStatus,
  queueDirective,
  type DirectiveStatus,
} from "../../services/assistantApi";
import type { ExpertDirectiveView } from "../../types/assistant";

interface DirectiveComposerProps {
  frame: OperationalFrame | null;
  selectedTargetIds: string[];
  suggestions?: PlanAdjustmentSuggestionView[];
  onApplied?: () => void;
}

const STATUS_LABELS: Record<string, string> = {
  queued: "已排队",
  processing: "解析中",
  preview: "待确认",
  applying: "应用中",
  applied: "已应用，等待下一轮重规划",
  needs_clarification: "需要澄清",
  rejected: "已拒绝",
  error: "处理失败",
};

const SUGGESTION_CATEGORY_LABELS: Record<PlanAdjustmentSuggestionView["category"], string> = {
  tracking_quality: "跟踪质量",
  segmented_handoff: "分段接力",
  resource_rotation: "资源轮换",
  commander_preference: "指挥偏好",
};

export default function DirectiveComposer({ frame, selectedTargetIds, suggestions, onApplied }: DirectiveComposerProps) {
  const [text, setText] = useState("");
  const [author, setAuthor] = useState("operator");
  const [requestId, setRequestId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<DirectiveStatus | null>(null);
  const [directive, setDirective] = useState<ExpertDirectiveView | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const returnUuvIds = directive?.return_uuv_ids ?? [];
  const visibleSuggestions = suggestions ?? frame?.plan_adjustment_suggestions ?? [];

  useEffect(() => {
    if (!requestId || !jobStatus || !["queued", "processing", "applying"].includes(jobStatus.status)) return undefined;
    const timer = window.setInterval(() => {
      void getDirectiveStatus(requestId)
        .then((next) => {
          setJobStatus(next);
          if (next.directive) setDirective(next.directive);
        })
        .catch((reason: unknown) => setError(errorMessage(reason)));
    }, 500);
    return () => window.clearInterval(timer);
  }, [jobStatus, requestId]);

  const submit = async (
    feedbackText = text,
    targetIds = selectedTargetIds,
  ) => {
    const normalizedText = feedbackText.trim();
    if (!frame || !normalizedText || !author.trim()) return;
    setBusy(true);
    setError("");
    try {
      const response = await queueDirective({
        text: normalizedText,
        author: author.trim(),
        expected_plan_version: frame.plan_version,
        target_ids: [...targetIds].sort(),
      });
      setRequestId(response.request_id);
      setJobStatus({ request_id: response.request_id, status: response.status });
      setDirective(null);
    } catch (reason: unknown) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const sendSuggestion = (suggestion: PlanAdjustmentSuggestionView) => {
    setText(suggestion.proposed_feedback);
    void submit(
      suggestion.proposed_feedback,
      suggestion.target_ids.length > 0 ? suggestion.target_ids : selectedTargetIds,
    );
  };

  const confirm = async () => {
    if (!requestId) return;
    setBusy(true);
    setError("");
    try {
      const response = await applyDirective(requestId);
      setJobStatus((current) => current ? { ...current, status: response.status } : current);
      onApplied?.();
    } catch (reason: unknown) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const status = jobStatus?.status ?? "";
  return (
    <section className="assistant-card directive-composer" aria-label="专家指令">
      <div className="assistant-card-heading">
        <div><span className="eyebrow">HUMAN IN THE LOOP</span><h2>方案干预</h2></div>
        <span className="version-chip">方案 #{frame?.plan_version ?? "—"}</span>
      </div>
      <section className="suggestion-panel" aria-label="LLM方案调整建议">
        <div className="suggestion-panel-heading">
          <span>当前观测建议</span>
          <small>{visibleSuggestions.length}/4</small>
        </div>
        {visibleSuggestions.length > 0 ? visibleSuggestions.map((suggestion) => (
          <button
            className="suggestion-item"
            key={suggestion.suggestion_id}
            type="button"
            disabled={busy || !frame}
            onClick={() => sendSuggestion(suggestion)}
            aria-label={`发送建议：${suggestion.title}`}
          >
            <span className="suggestion-item-head">
              <span className="suggestion-category">{SUGGESTION_CATEGORY_LABELS[suggestion.category]}</span>
              <strong>{suggestion.title}</strong>
              <span className="suggestion-confidence">{Math.round(suggestion.confidence * 100)}%</span>
            </span>
            <span className="suggestion-rationale">{suggestion.rationale}</span>
            <span className="suggestion-feedback">反馈：{suggestion.proposed_feedback}</span>
            <span className="suggestion-send"><Send size={12} />发送反馈</span>
          </button>
        )) : (
          <p className="assistant-pending">等待主脑基于当前观测生成四条建议</p>
        )}
      </section>
      <label className="field">
        <span>专家指令</span>
        <textarea value={text} onChange={(event) => setText(event.target.value)} placeholder="例如：优先保证 T1 的观测质量，暂不释放已指派资源" rows={3} />
      </label>
      <label className="field inline-field">
        <span>操作者</span>
        <input value={author} onChange={(event) => setAuthor(event.target.value)} />
        {selectedTargetIds.length > 0 && <small>作用域：{selectedTargetIds.join("、")}</small>}
      </label>
      <div className="assistant-actions">
        <button className="primary-btn" onClick={() => void submit()} disabled={busy || !frame || !text.trim()}>
          {busy ? "处理中…" : "提交预览"}
        </button>
        {directive?.status === "preview" && (
          <button className="secondary-btn" onClick={() => void confirm()} disabled={busy}>确认应用</button>
        )}
      </div>
      {status && <div className={`assistant-status status-${status}`} role="status">{STATUS_LABELS[status] ?? status}</div>}
      {directive && (
        <div className="directive-preview">
          <div><span>置信度</span><b>{Math.round(directive.confidence * 100)}%</b></div>
          <div><span>类型</span><b>{directive.directive_type === "assignment" ? "资源指派" : "约束调整"}</b></div>
          {returnUuvIds.length > 0 && <div><span>返航资源</span><b>{returnUuvIds.join("、")}</b></div>}
          {directive.disabled_uuv_ids.length > 0 && <div><span>禁用资源</span><b>{directive.disabled_uuv_ids.join("、")}</b></div>}
          {directive.conflicts.length > 0 && <ul>{directive.conflicts.map((item) => <li key={item}>{item}</li>)}</ul>}
        </div>
      )}
      {error && <p className="assistant-error" role="alert">{error}</p>}
    </section>
  );
}

function errorMessage(reason: unknown): string {
  if (reason instanceof AssistantApiError) {
    const payload = reason.payload as { detail?: { message?: string } | string; message?: string } | null;
    if (typeof payload?.detail === "object" && payload.detail?.message) return payload.detail.message;
    if (typeof payload?.detail === "string") return payload.detail;
    if (payload?.message) return payload.message;
    if (reason.status === 409) return "方案版本已变化，请重新查看当前方案后提交。";
  }
  return reason instanceof Error ? reason.message : "请求失败，请检查连接后重试。";
}
