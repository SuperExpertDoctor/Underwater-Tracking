import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronRight, RefreshCw, Trash2 } from "lucide-react";
import {
  deleteMemory,
  getMemorySnapshot,
  getMemoryVersions,
  type MemoryFamily,
  type MemorySnapshotView,
  type MemoryVersionView,
} from "../../services/memoryApi";

interface MemoryWindowProps {
  userId: string;
  conversationId: string;
  scenarioId?: string;
  refreshKey?: number;
  snapshot?: MemorySnapshotView | null;
  onSnapshot?: (snapshot: MemorySnapshotView) => void;
  managed?: boolean;
  managedLoading?: boolean;
  managedError?: string;
  scopeUnavailable?: boolean;
}

const FAMILIES: Array<{ id: "short_term" | MemoryFamily; label: string }> = [
  { id: "short_term", label: "短期记忆" },
  { id: "episodic", label: "情景记忆" },
  { id: "semantic", label: "语义记忆" },
  { id: "procedural", label: "程序记忆" },
];

export default function MemoryWindow({
  userId,
  conversationId,
  scenarioId,
  refreshKey = 0,
  snapshot: externalSnapshot,
  onSnapshot,
  managed = false,
  managedLoading = false,
  managedError = "",
  scopeUnavailable = false,
}: MemoryWindowProps) {
  const unavailable = scopeUnavailable || !scenarioId;
  const [snapshot, setSnapshot] = useState<MemorySnapshotView | null>(externalSnapshot ?? null);
  const [activeFamily, setActiveFamily] = useState<typeof FAMILIES[number]["id"]>("short_term");
  const [loading, setLoading] = useState(!externalSnapshot);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [versions, setVersions] = useState<Record<string, MemoryVersionView[]>>({});
  const [versionLoading, setVersionLoading] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState("");

  const loadSnapshot = useCallback(async () => {
    if (!scenarioId) return;
    setLoading(true);
    setError("");
    try {
      const next = await getMemorySnapshot({ userId, conversationId, scenarioId });
      setSnapshot(next);
      onSnapshot?.(next);
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "无法读取记忆快照");
    } finally {
      setLoading(false);
    }
  }, [conversationId, onSnapshot, scenarioId, userId]);

  useEffect(() => {
    if (managed) {
      setSnapshot(externalSnapshot ?? null);
      setLoading(managedLoading);
      setError(managedError);
      return undefined;
    }
    if (!scenarioId) {
      setSnapshot(null);
      setLoading(false);
      setError("");
      return undefined;
    }
    if (externalSnapshot) {
      setSnapshot(externalSnapshot);
      setLoading(false);
      return;
    }
    void loadSnapshot();
    const timer = window.setInterval(() => void loadSnapshot(), 10_000);
    return () => window.clearInterval(timer);
  }, [externalSnapshot, loadSnapshot, managed, managedError, managedLoading, refreshKey, scenarioId]);

  const activeItems = useMemo(() => {
    if (!snapshot || activeFamily === "short_term") return [];
    return snapshot[activeFamily] ?? [];
  }, [activeFamily, snapshot]);

  const toggleVersions = async (memory: MemoryVersionView) => {
    if (expanded === memory.memory_id) {
      setExpanded(null);
      return;
    }
    setExpanded(memory.memory_id);
    if (versions[memory.memory_family_id]) return;
    setVersionLoading(memory.memory_family_id);
    setError("");
    try {
      if (!scenarioId) throw new Error("当前场景尚未确定，暂不可读取记忆版本");
      const result = await getMemoryVersions({ userId, memoryFamilyId: memory.memory_family_id, scenarioId });
      setVersions((current) => ({ ...current, [memory.memory_family_id]: result.versions }));
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "无法读取记忆版本");
    } finally {
      setVersionLoading(null);
    }
  };

  const removeMemory = async (memory: MemoryVersionView) => {
    setError("");
    try {
      if (!scenarioId) throw new Error("当前场景尚未确定，暂不可删除记忆");
      await deleteMemory({ userId, memoryId: memory.memory_id, scenarioId, conversationId });
      setActionMessage("该记忆已删除");
      setExpanded(null);
      await loadSnapshot();
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "无法删除记忆");
    }
  };

  return (
    <section className="memory-window" aria-label="记忆窗口">
      <div className="memory-window-heading">
        <div>
          <span className="eyebrow">PERSISTED / REAL SQLITE</span>
          <h3>记忆窗口</h3>
        </div>
        <button className="icon-btn" type="button" onClick={() => void loadSnapshot()} aria-label="刷新记忆快照" title="刷新记忆快照">
          <RefreshCw size={14} />
        </button>
      </div>
      {loading && <p className="memory-state">正在读取真实记忆快照…</p>}
      {unavailable && <p className="memory-state">等待当前场景确定，记忆暂不可用。</p>}
      {error && <p className="memory-error" role="alert">{error}</p>}
      {snapshot?.memory_status === "degraded" && (
        <p className="memory-degraded" role="status">
          <strong>记忆服务降级</strong>
          {snapshot.degraded_reason && <><span aria-hidden="true"> · </span><span>{snapshot.degraded_reason}</span></>}
        </p>
      )}
      {actionMessage && <p className="memory-success" role="status">{actionMessage}</p>}
      {!loading && snapshot && (
        <>
          <div className="memory-family-tabs" aria-label="记忆类型">
            {FAMILIES.map((family) => (
              <button
                type="button"
                key={family.id}
                aria-pressed={activeFamily === family.id}
                onClick={() => setActiveFamily(family.id)}
              >
                {family.label}
              </button>
            ))}
          </div>
          {activeFamily === "short_term" ? (
            <ShortTermMemory context={snapshot.short_term} />
          ) : (
            <LongTermMemoryList
              items={activeItems}
              expanded={expanded}
              versions={versions}
              versionLoading={versionLoading}
              onToggleVersions={(memory) => void toggleVersions(memory)}
              onDelete={(memory) => void removeMemory(memory)}
            />
          )}
        </>
      )}
      {!loading && !unavailable && !snapshot && !error && <p className="memory-state">暂无记忆快照。</p>}
    </section>
  );
}

function ShortTermMemory({ context }: { context: MemorySnapshotView["short_term"] }) {
  if (!context) return <p className="memory-empty">当前没有短期记忆</p>;
  return (
    <div className="short-term-memory">
      <p>{context.summary_text || "当前没有短期记忆"}</p>
      <div className="memory-meta">
        <span>version {context.summary_version}</span>
        <span>消息 {context.message_count}</span>
        <span>约 {context.estimated_tokens} tokens</span>
      </div>
    </div>
  );
}

function LongTermMemoryList({
  items,
  expanded,
  versions,
  versionLoading,
  onToggleVersions,
  onDelete,
}: {
  items: MemoryVersionView[];
  expanded: string | null;
  versions: Record<string, MemoryVersionView[]>;
  versionLoading: string | null;
  onToggleVersions: (memory: MemoryVersionView) => void;
  onDelete: (memory: MemoryVersionView) => void;
}) {
  if (!items.length) return <p className="memory-empty">当前没有该类型记忆</p>;
  return (
    <div className="memory-list">
      {items.map((memory) => {
        const familyVersions = versions[memory.memory_family_id] ?? [];
        const sourceCount = sourceIds(memory).length;
        return (
          <article className="memory-item" key={memory.memory_id}>
            <div className="memory-item-heading">
              <div>
                <strong>{memory.summary}</strong>
                <small>{memory.memory_id} · <span>v{memory.version}</span></small>
              </div>
              <button className="icon-btn" type="button" onClick={() => onDelete(memory)} aria-label={`删除记忆 ${memory.memory_id}`} title="删除记忆">
                <Trash2 size={14} />
              </button>
            </div>
            <div className="memory-meta">
              <span>重要性 {memory.importance_score.toFixed(2)}</span>
              <span>访问 {memory.access_count} 次</span>
              <span>来源 {sourceCount}</span>
            </div>
            <div className="memory-item-actions">
              <button className="memory-version-toggle" type="button" onClick={() => onToggleVersions(memory)} aria-label={`${expanded === memory.memory_id ? "收起" : "展开"}版本 ${memory.memory_id}`}>
                {expanded === memory.memory_id ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                {expanded === memory.memory_id ? "收起版本" : "展开版本"}
              </button>
              {versionLoading === memory.memory_family_id && <small>读取中…</small>}
            </div>
            {expanded === memory.memory_id && (
              <div className="memory-version-list">
                <strong>历史版本</strong>
                {familyVersions.length ? familyVersions.map((version) => <span key={version.memory_id}>v{version.version} · {version.summary}</span>) : <span>暂无其他版本</span>}
              </div>
            )}
          </article>
        );
      })}
    </div>
  );
}

function sourceIds(memory: MemoryVersionView): string[] {
  return [...new Set([
    ...(memory.source_message_ids ?? []),
    ...(memory.source_event_ids ?? []),
    ...(memory.source_decision_ids ?? []),
    ...(memory.source_knowledge_ids ?? []),
    ...(memory.source_plan_ids ?? []),
  ])];
}
