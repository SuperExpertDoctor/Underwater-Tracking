import { useState } from "react";
import { askQuestion, AssistantApiError, type QuestionAnswerView } from "../../services/assistantApi";

interface QuestionPanelProps {
  onSelectEvidence?: (evidenceId: string) => void;
  disabled?: boolean;
}

export default function QuestionPanel({ onSelectEvidence, disabled = false }: QuestionPanelProps) {
  const [text, setText] = useState("");
  const [counterfactual, setCounterfactual] = useState(false);
  const [answer, setAnswer] = useState<QuestionAnswerView | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (disabled || !text.trim()) return;
    setBusy(true); setError("");
    try {
      setAnswer(await askQuestion(text.trim(), counterfactual ? {} : undefined));
    } catch (reason: unknown) {
      if (reason instanceof AssistantApiError && reason.status === 422) {
        const payload = reason.payload as { message?: string; detail?: string } | null;
        setAnswer({ status: "insufficient_evidence", message: payload?.message ?? payload?.detail ?? "当前证据不足，无法可靠回答。" });
      } else {
        setError(reason instanceof Error ? reason.message : "质询请求失败");
      }
    } finally { setBusy(false); }
  };

  return <section className={`assistant-card question-panel ${disabled ? "disabled" : ""}`} aria-label="证据质询">
    <div className="assistant-card-heading"><div><span className="eyebrow">EVIDENCE EXPLANATION</span><h2>证据质询</h2></div><span className="read-only-chip">只读</span></div>
    <label className="field"><span>问题</span><textarea value={text} onChange={(event) => setText(event.target.value)} rows={2} placeholder={disabled ? "回放模式下不可质询在线运行" : "例如：为什么 T1 当前保持这组资源？"} disabled={disabled} /></label>
    <label className="check-row"><input type="checkbox" checked={counterfactual} onChange={(event) => setCounterfactual(event.target.checked)} disabled={disabled} /><span>附带反事实试算（不改变在线方案）</span></label>
    <button className="secondary-btn" onClick={() => void submit()} disabled={disabled || busy || !text.trim()}>{busy ? "检索证据…" : "提交问题"}</button>
    {answer?.status === "insufficient_evidence" ? <p className="assistant-warning" role="status">{answer.message}</p> : answer && <div className="answer-block" role="status">
      <p>{answer.answer}</p>
      {answer.counterfactual_summary && <div className="counterfactual-block"><span>反事实结果</span><p>{answer.counterfactual_summary}</p></div>}
      <div className="evidence-links">证据 {answer.evidence_ids?.length ? answer.evidence_ids.map((id) => <button key={id} onClick={() => onSelectEvidence?.(id)}>{id}</button>) : "—"}</div>
    </div>}
    {error && <p className="assistant-error" role="alert">{error}</p>}
  </section>;
}
