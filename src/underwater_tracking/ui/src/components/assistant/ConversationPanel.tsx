import { useRef, useState } from "react";
import type { OperationalFrame } from "../../types/frames";
import { sendConversationMessage } from "../../services/assistantApi";

interface ConversationPanelProps {
  frame: OperationalFrame | null;
  selectedTargetIds: string[];
  disabled?: boolean;
}

/** Compact command entry point. Responses are reflected by later operational frames. */
export default function ConversationPanel({
  frame,
  selectedTargetIds,
  disabled = false,
}: ConversationPanelProps) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const conversationId = useRef(
    `conversation-${Math.random().toString(36).slice(2, 10)}`,
  );

  const submit = async () => {
    if (disabled || !frame || !text.trim()) return;
    setBusy(true);
    try {
      await sendConversationMessage({
        conversation_id: conversationId.current,
        text: text.trim(),
        expected_plan_version: frame.plan_version,
        target_ids: [...selectedTargetIds].sort(),
      });
      setText("");
    } catch {
      // This compact control intentionally has no result or error area.
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="conversation-panel" aria-label="LLM Client 输入">
      <form
        className="llm-command-form"
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
      >
        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          disabled={disabled}
          rows={4}
          placeholder={disabled ? "回放模式下不可发送" : "输入问题或指令"}
          aria-label="LLM 输入"
        />
        <button
          className="primary-btn"
          type="submit"
          disabled={disabled || busy || !frame || !text.trim()}
        >
          {busy ? "发送中…" : "发送"}
        </button>
      </form>
    </section>
  );
}
