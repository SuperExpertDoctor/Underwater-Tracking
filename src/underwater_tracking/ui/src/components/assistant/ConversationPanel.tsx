import { useRef, useState } from "react";
import type { OperationalFrame } from "../../types/frames";
import {
  applyConversation,
  AssistantApiError,
  sendConversationMessage,
  type ConversationTurnView,
} from "../../services/assistantApi";

interface ConversationPanelProps {
  frame: OperationalFrame | null;
  selectedTargetIds: string[];
  disabled?: boolean;
  onSelectEvidence?: (evidenceId: string) => void;
}

const CLASSIFICATION_LABELS: Record<string, string> = {
  plan_revision: "方案修正",
  evidence_query: "证据质询",
  mixed: "方案 + 证据",
  clarification: "需要澄清",
};

export default function ConversationPanel({ frame, selectedTargetIds, disabled = false, onSelectEvidence }: ConversationPanelProps) {
  const [text, setText] = useState("");
  const [turn, setTurn] = useState<ConversationTurnView | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const conversationId = useRef(`conversation-${Math.random().toString(36).slice(2, 10)}`);

  const submit = async () => {
    if (disabled || !frame || !text.trim()) return;
    setBusy(true); setError("");
    try {
      const next = await sendConversationMessage({
        conversation_id: conversationId.current,
        text: text.trim(),
        expected_plan_version: frame.plan_version,
        target_ids: [...selectedTargetIds].sort(),
      });
      setTurn(next);
      setText("");
    } catch (reason: unknown) {
      setError(errorMessage(reason));
    } finally { setBusy(false); }
  };

  const apply = async () => {
    if (!turn?.proposal || !frame) return;
    setBusy(true); setError("");
    try {
      const turnId = turn.turn_id ?? `${turn.conversation_id}:turn:1`;
      setTurn(await applyConversation(turn.conversation_id, turnId, frame.plan_version));
    } catch (reason: unknown) {
      setError(errorMessage(reason));
    } finally { setBusy(false); }
  };

  const kind = typeof turn?.classification === "string" ? turn.classification : turn?.classification?.classification;
  const evidenceIds = turn?.evidence_ids ?? turn?.answer?.evidence_ids ?? [];
  const proposal = turn?.proposal;
  return <section className={`assistant-card conversation-panel ${disabled ? "disabled" : ""}`} aria-label="统一 LLM Client">
    <div className="assistant-card-heading">
      <div><span className="eyebrow">LLM CLIENT</span><h2>统一对话</h2></div>
      <span className="version-chip">方案 v{frame?.plan_version ?? "—"}</span>
    </div>
    <label className="field"><span>消息</span><textarea value={text} onChange={(event) => setText(event.target.value)} rows={3} disabled={disabled} placeholder={disabled ? "回放模式下不可修改在线方案" : "询问证据，或提出方案修正"} aria-label="统一对话" /></label>
    <button className="primary-btn" onClick={() => void submit()} disabled={disabled || busy || !frame || !text.trim()}>{busy ? "处理中…" : "发送"}</button>
    {turn && <div className="conversation-result" role="status">
      <div className="conversation-meta"><span className="conversation-kind">{CLASSIFICATION_LABELS[kind ?? ""] ?? kind}</span><span>证据 {evidenceIds.length} 条</span></div>
      {turn.messages.filter((message) => message.role === "assistant").map((message) => <article className="conversation-message" key={message.message_id}><p>{message.text}</p>{message.evidence_ids?.map((id) => <button className="evidence-chip" key={id} onClick={() => onSelectEvidence?.(id)} aria-label={`证据 ${id}`}>{id}</button>)}</article>)}
      {proposal && <div className="conversation-proposal"><strong>{proposal.summary || "方案修正预览"}</strong><span>状态：{proposal.status}</span>{proposal.status === "preview" && <button className="secondary-btn" onClick={() => void apply()} disabled={busy} aria-label="确认应用方案">确认应用方案</button>}</div>}
    </div>}
    {error && <p className="assistant-error" role="alert">{error}</p>}
  </section>;
}

function errorMessage(reason: unknown): string {
  if (reason instanceof AssistantApiError) {
    const payload = reason.payload as { detail?: { message?: string } | string; message?: string } | null;
    if (typeof payload?.detail === "object" && payload.detail?.message) return payload.detail.message;
    if (typeof payload?.detail === "string") return payload.detail;
    if (payload?.message) return payload.message;
  }
  return reason instanceof Error ? reason.message : "对话请求失败，请检查连接。";
}
