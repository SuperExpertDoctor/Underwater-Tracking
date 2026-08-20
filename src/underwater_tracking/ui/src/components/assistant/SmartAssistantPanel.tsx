import { useState } from "react";
import { Check, Send } from "lucide-react";
import {
  applyConversation,
  sendConversationMessage,
  type ConversationTurnView,
} from "../../services/assistantApi";
import type { MemoryEvidenceTraceView, MemoryRetrievalHitView } from "../../services/memoryApi";
import type { OperationalFrame } from "../../types/frames";

type AssistantMode = "plan_revision" | "evidence_query";

interface SmartAssistantPanelProps {
  frame: OperationalFrame | null;
  selectedTargetIds: string[];
  conversationId: string;
  userId: string;
  disabled?: boolean;
  onActivity?: () => void;
}

export default function SmartAssistantPanel({
  frame,
  selectedTargetIds,
  conversationId,
  userId,
  disabled = false,
  onActivity,
}: SmartAssistantPanelProps) {
  const [mode, setMode] = useState<AssistantMode>("plan_revision");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ConversationTurnView | null>(null);
  const [applyStatus, setApplyStatus] = useState("");

  const submit = async () => {
    if (disabled || !frame || !text.trim() || busy) return;
    setBusy(true);
    setError("");
    setApplyStatus("");
    try {
      const next = await sendConversationMessage({
        conversation_id: conversationId,
        user_id: userId,
        assistant_mode: mode,
        text: text.trim(),
        expected_plan_version: frame.plan_version,
        target_ids: [...selectedTargetIds].sort(),
      });
      setResult(next);
      setText("");
      onActivity?.();
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "智能助理请求失败");
    } finally {
      setBusy(false);
    }
  };

  const applyProposal = async () => {
    if (!result?.proposal || !result.turn_id || !isPlanClassification(result) || applying) return;
    setApplying(true);
    setError("");
    try {
      await applyConversation(
        conversationId,
        result.turn_id,
        result.proposal.expected_plan_version ?? result.expected_plan_version,
        userId,
      );
      setApplyStatus("方案已应用");
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "方案应用失败");
    } finally {
      setApplying(false);
    }
  };

  const resultClassification = result ? classificationName(result) : null;
  const inputIsEvidence = mode === "evidence_query";
  const showEvidence = resultClassification === "evidence_query" || resultClassification === "mixed";
  const canApplyProposal = Boolean(result?.proposal && result.turn_id && resultClassification && ["plan_revision", "mixed"].includes(resultClassification));
  const memoryContext = result?.memory_context;
  const answer = result?.answer;
  const traces = answer?.evidence_trace?.length
    ? answer.evidence_trace
    : memoryContext?.evidence_trace ?? [];

  return (
    <section className="smart-assistant" aria-label="智能助理">
      <div className="assistant-card-heading">
        <div>
          <span className="eyebrow">REAL API / MEMORY AWARE</span>
          <h2>智能助理</h2>
        </div>
        <span className="assistant-version">方案 v{frame?.plan_version ?? "—"}</span>
      </div>
      <div className="assistant-mode-tabs" aria-label="智能助理模式">
        <button
          type="button"
          aria-pressed={mode === "plan_revision"}
          onClick={() => setMode("plan_revision")}
        >
          方案调整
        </button>
        <button
          type="button"
          aria-pressed={mode === "evidence_query"}
          onClick={() => setMode("evidence_query")}
        >
          证据回溯
        </button>
      </div>
      {memoryContext && (
        <MemoryStatusNotice
          status={memoryContext.memory_status}
          reason={memoryContext.degraded_reason}
        />
      )}
      <form
        className="smart-assistant-form"
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
      >
        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          disabled={disabled || busy}
          rows={3}
          placeholder={disabled ? "回放模式下不可发送" : inputIsEvidence ? "输入需要回溯的问题" : "输入需要调整的方案"}
          aria-label="智能助理输入"
        />
        <button className="primary-btn" type="submit" disabled={disabled || busy || !frame || !text.trim()}>
          <Send size={14} />
          {busy ? "发送中" : "发送"}
        </button>
      </form>
      {error && <p className="assistant-error" role="alert">{error}</p>}
      {applyStatus && <p className="assistant-success" role="status"><Check size={14} />{applyStatus}</p>}
      {result && (
        <div className="assistant-result" aria-live="polite">
          {result.proposal && canApplyProposal && (
            <div className="assistant-proposal">
              <div className="assistant-result-heading">
                <strong>方案预览</strong>
                <span>待专家确认</span>
              </div>
              <p>{result.proposal.summary}</p>
              {result.proposal.diff && <pre>{JSON.stringify(result.proposal.diff, null, 2)}</pre>}
              <button className="secondary-btn" type="button" onClick={() => void applyProposal()} disabled={applying}>
                <Check size={14} />
                {applying ? "应用中" : "应用方案预览"}
              </button>
            </div>
          )}
          {showEvidence && answer && (
            <EvidenceAnswer answer={answer.answer ?? "后端未返回证据回答。"} hits={memoryContext?.long_term_material ?? []} traces={traces} />
          )}
          {!result.proposal && !answer && <p className="assistant-empty-result">后端暂未返回可展示结果。</p>}
        </div>
      )}
    </section>
  );
}

function classificationName(result: ConversationTurnView): string {
  if (typeof result.classification === "object" && result.classification !== null) {
    return result.classification.classification;
  }
  return result.classification || result.assistant_mode || "clarification";
}

function isPlanClassification(result: ConversationTurnView): boolean {
  return ["plan_revision", "mixed"].includes(classificationName(result));
}

function MemoryStatusNotice({ status, reason }: { status: string; reason?: string | null }) {
  return (
    <p className={`memory-status memory-status-${status}`}>
      记忆状态：{status}
      {reason ? ` · ${reason}` : ""}
    </p>
  );
}

function EvidenceAnswer({
  answer,
  hits,
  traces,
}: {
  answer: string;
  hits: MemoryRetrievalHitView[];
  traces: MemoryEvidenceTraceView[];
}) {
  const sourceIds = [...new Set(traces.flatMap((trace) => [
    ...(trace.source_message_ids ?? []),
    ...(trace.source_event_ids ?? []),
    ...(trace.source_decision_ids ?? []),
    ...(trace.source_knowledge_ids ?? []),
    ...(trace.source_plan_ids ?? []),
  ]))];
  const memoryIds = [...new Set(traces.flatMap((trace) => trace.memory_ids ?? []))];
  return (
    <div className="evidence-answer" aria-label="只读证据回答">
      <div className="assistant-result-heading">
        <strong>只读证据回答</strong>
        <span>{traces.some((trace) => trace.status !== "completed") ? "待验证" : "已验证"}</span>
      </div>
      <p>{answer}</p>
      <div className="evidence-memory-list">
        {memoryIds.length ? memoryIds.map((memoryId) => {
          const hit = hits.find((item) => item.memory.memory_id === memoryId);
          return <span key={memoryId}>{memoryId} · {hit ? `v${hit.memory.version}` : "version 未提供"}</span>;
        }) : <span>未命中记忆版本</span>}
      </div>
      <small>已验证来源：{sourceIds.length ? sourceIds.join("、") : "无"}</small>
    </div>
  );
}
