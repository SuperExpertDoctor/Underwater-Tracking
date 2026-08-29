import { useEffect, useMemo, useState } from "react";
import {
  Archive,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  Database,
  Eye,
  FileText,
  Filter,
  GitBranch,
  History,
  Link2,
  LoaderCircle,
  Search,
  Sparkles,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { MemoryStatus, MemoryStreamEventView } from "../services/memoryApi";

const MAX_VISIBLE_EVENTS = 300;
const MAX_VISIBLE_REFERENCES = 3;

type ExtendedMemoryPayload = NonNullable<MemoryStreamEventView["payload"]> & {
  question_id?: string | null;
  question_ids?: string[];
  decision_id?: string | null;
  decision_ids?: string[];
  plan_id?: string | null;
  plan_ids?: string[];
  source_message_ids?: string[];
  source_event_ids?: string[];
  source_decision_ids?: string[];
  source_plan_ids?: string[];
  plan_version?: number | null;
};

interface EventMeta {
  group: string;
  label: string;
  icon: LucideIcon;
}

interface CausalChain {
  questionIds: string[];
  memoryIds: string[];
  memoryFamilyId: string | null;
  version: number | null;
  planVersion: number | null;
  sourceIds: string[];
  sourceKinds: Map<string, string>;
}

export interface MemorySteamProps {
  events: MemoryStreamEventView[];
  status: MemoryStatus | "idle";
  loading: boolean;
  error: string;
  cursor: number;
  degradedReason?: string | null;
  executionRevision?: number | null;
  frameId?: number | null;
  onSelectEvidence?: (evidenceId: string) => void;
}

const EVENT_META: Record<string, EventMeta> = {
  context_loaded: { group: "上下文", label: "上下文已加载", icon: FileText },
  retrieval_started: { group: "检索", label: "检索开始", icon: Search },
  retrieval_completed: { group: "检索", label: "检索完成", icon: CheckCircle2 },
  memory_filtered: { group: "筛选", label: "记忆已过滤", icon: Filter },
  memory_extracted: { group: "提炼", label: "记忆已提炼", icon: Sparkles },
  short_term_compression_started: {
    group: "短期记忆",
    label: "短期记忆压缩开始",
    icon: LoaderCircle,
  },
  short_term_compressed: {
    group: "短期记忆",
    label: "短期记忆压缩完成",
    icon: CheckCircle2,
  },
  compression_degraded: {
    group: "短期记忆",
    label: "短期记忆压缩降级",
    icon: CircleAlert,
  },
  memory_version_created: {
    group: "版本",
    label: "记忆版本创建",
    icon: GitBranch,
  },
  memory_version_superseded: {
    group: "版本",
    label: "记忆版本替代",
    icon: History,
  },
  memory_accessed: { group: "访问", label: "记忆访问", icon: Eye },
  memory_archived: { group: "归档", label: "记忆归档", icon: Archive },
  memory_deleted: { group: "归档", label: "记忆删除", icon: Archive },
  evidence_trace_started: {
    group: "证据",
    label: "证据追溯开始",
    icon: Link2,
  },
  evidence_trace_completed: {
    group: "证据",
    label: "证据追溯完成",
    icon: Link2,
  },
  work_queued: { group: "状态", label: "记忆工作入队", icon: Database },
  work_processing: { group: "状态", label: "记忆工作处理", icon: LoaderCircle },
  work_completed: { group: "状态", label: "记忆工作完成", icon: CheckCircle2 },
  work_degraded: { group: "状态", label: "记忆处理降级", icon: CircleAlert },
  work_retry_scheduled: { group: "状态", label: "记忆工作重试", icon: History },
  source_read_degraded: { group: "状态", label: "来源读取降级", icon: CircleAlert },
  worker_recovered: { group: "状态", label: "记忆 Worker 已恢复", icon: CheckCircle2 },
};

const STATUS_LABELS: Record<MemoryStatus, string> = {
  pending: "记忆任务等待中",
  processing: "记忆处理中",
  completed: "Memory Steam 已同步",
  degraded: "记忆服务降级",
  failed: "Memory Steam 读取失败",
};

export default function MemorySteam({
  events,
  status,
  loading,
  error,
  cursor,
  degradedReason = null,
  executionRevision = null,
  frameId = null,
  onSelectEvidence,
}: MemorySteamProps) {
  const [expandedEvents, setExpandedEvents] = useState<Set<string>>(new Set());
  const [expandedReferences, setExpandedReferences] = useState<Set<string>>(
    new Set(),
  );
  const visibleEvents = useMemo(() => normalizeEvents(events), [events]);
  const showStatus = loading || status !== "idle" || Boolean(error) || Boolean(degradedReason);

  useEffect(() => {
    const visibleEventIds = new Set(visibleEvents.map((event) => event.event_id));
    setExpandedEvents((current) => {
      const next = new Set(
        [...current].filter((eventId) => visibleEventIds.has(eventId)),
      );
      return next.size === current.size ? current : next;
    });
    setExpandedReferences((current) => {
      const next = new Set(
        [...current].filter((eventId) => visibleEventIds.has(eventId)),
      );
      return next.size === current.size ? current : next;
    });
  }, [visibleEvents]);

  const toggleEvent = (eventId: string) => {
    setExpandedEvents((current) => {
      const next = new Set(current);
      if (next.has(eventId)) next.delete(eventId);
      else next.add(eventId);
      return next;
    });
  };

  const toggleReferences = (eventId: string) => {
    setExpandedReferences((current) => {
      const next = new Set(current);
      if (next.has(eventId)) next.delete(eventId);
      else next.add(eventId);
      return next;
    });
  };

  return (
    <section className="memory-steam" aria-label="Memory Steam">
      {showStatus && (
        <div
          className={`memory-steam-status status-${status}`}
          role="status"
          aria-live="polite"
        >
          <StatusIcon loading={loading} status={status} />
          <strong>{loading ? "正在读取 Memory Steam…" : status === "idle" ? "Memory Steam" : STATUS_LABELS[status]}</strong>
          {cursor > 0 && <span> · 已读取至游标 {cursor}</span>}
          {(executionRevision != null || frameId != null) && (
            <span> · 执行 v{executionRevision ?? "—"} · 帧 {frameId ?? "—"}</span>
          )}
          {error && <span> · {error}</span>}
          {degradedReason && <span> · {degradedReason}</span>}
        </div>
      )}
      <div className="memory-steam-events" aria-label="Memory Steam 事件流">
        {visibleEvents.length === 0 ? (
          <div className="memory-steam-empty">
            {loading ? "等待后台记忆事件…" : "暂无 Memory Steam 事件"}
          </div>
        ) : (
          visibleEvents.map((event) => {
            const isExpanded = expandedEvents.has(event.event_id);
            const meta = getEventMeta(event);
            const Icon = meta.icon;
            const chain = getCausalChain(event);
            const referencesExpanded = expandedReferences.has(event.event_id);
            const references = referencesExpanded
              ? chain.sourceIds
              : chain.sourceIds.slice(0, MAX_VISIBLE_REFERENCES);

            return (
              <article
                className={`memory-steam-event status-${event.status}`}
                data-testid="memory-steam-event"
                data-cursor={event.cursor}
                key={event.event_id}
              >
                <button
                  className="memory-steam-event-toggle"
                  type="button"
                  aria-expanded={isExpanded}
                  aria-label={`${isExpanded ? "收起" : "展开"}证据链 ${event.event_id}`}
                  onClick={() => toggleEvent(event.event_id)}
                >
                  <span className="memory-steam-event-heading">
                    <Icon size={14} aria-hidden="true" />
                    <span className="memory-steam-event-group">{meta.group}</span>
                    <strong>{meta.label}</strong>
                  </span>
                  <span className="memory-steam-event-trailing">
                    <span className="memory-steam-cursor">#{event.cursor}</span>
                    <span className={`memory-steam-event-status status-${event.status}`}>
                      {event.status}
                    </span>
                    {isExpanded ? (
                      <ChevronDown size={14} aria-hidden="true" />
                    ) : (
                      <ChevronRight size={14} aria-hidden="true" />
                    )}
                  </span>
                </button>
                {isExpanded && (
                  <div className="memory-steam-chain" aria-label={`证据链 ${event.event_id}`}>
                    <CausalStep label="问题 / 工作 ID">
                      <ReferenceValues values={chain.questionIds} empty="未提供" />
                    </CausalStep>
                    <CausalStep label="记忆 / 版本 / 家族">
                      <div className="memory-steam-reference-line">
                        <ReferenceValues values={chain.memoryIds} empty="记忆 ID 未提供" />
                        {chain.version !== null && <span>v{chain.version}</span>}
                        {chain.memoryFamilyId && <span>{chain.memoryFamilyId}</span>}
                        {chain.planVersion !== null && <span>方案 v{chain.planVersion}</span>}
                      </div>
                    </CausalStep>
                    <CausalStep label="来源 IDs">
                      {chain.sourceIds.length === 0 ? (
                        <span className="memory-steam-reference-empty">未提供</span>
                      ) : (
                        <div className="memory-steam-reference-grid">
                          {references.map((reference) => (
                            <button
                              className="memory-steam-reference"
                              key={reference}
                              type="button"
                              title={`选择证据 ${reference}`}
                              aria-label={`证据 ${reference}`}
                              onClick={() => onSelectEvidence?.(reference)}
                            >
                              <span>{chain.sourceKinds.get(reference) ?? "来源"}</span>
                              <b>{reference}</b>
                            </button>
                          ))}
                        </div>
                      )}
                      {chain.sourceIds.length > MAX_VISIBLE_REFERENCES && (
                        <button
                          className="memory-steam-expand-references"
                          type="button"
                          aria-expanded={referencesExpanded}
                          onClick={() => toggleReferences(event.event_id)}
                        >
                          {referencesExpanded ? "收起来源" : "展开全部来源"}
                          <span>
                            {referencesExpanded
                              ? ""
                              : `（还有 ${chain.sourceIds.length - MAX_VISIBLE_REFERENCES} 条）`}
                          </span>
                        </button>
                      )}
                    </CausalStep>
                  </div>
                )}
              </article>
            );
          })
        )}
      </div>
    </section>
  );
}

function normalizeEvents(events: MemoryStreamEventView[]): MemoryStreamEventView[] {
  const seenEventIds = new Set<string>();
  const seenCursors = new Set<number>();
  const unique: MemoryStreamEventView[] = [];
  events.forEach((event) => {
    if (seenEventIds.has(event.event_id) || seenCursors.has(event.cursor)) return;
    seenEventIds.add(event.event_id);
    seenCursors.add(event.cursor);
    unique.push(event);
  });
  return unique.sort((left, right) => left.cursor - right.cursor).slice(-MAX_VISIBLE_EVENTS);
}

function getEventMeta(event: MemoryStreamEventView): EventMeta {
  const known = EVENT_META[event.type];
  if (known) return known;
  if (event.status === "degraded" || event.status === "failed" || event.type.includes("degraded")) {
    return { group: "状态", label: "记忆服务降级", icon: CircleAlert };
  }
  return { group: "状态", label: "记忆处理事件", icon: BrainCircuit };
}

function getCausalChain(event: MemoryStreamEventView): CausalChain {
  const payload = (event.payload ?? {}) as ExtendedMemoryPayload;
  const questionIds = uniqueStrings([
    ...stringValues(payload.question_ids),
    ...stringValues(payload.question_id),
    ...stringValues(payload.work_id),
  ]);
  const memoryIds = uniqueStrings([
    ...stringValues(event.memory_id),
    ...stringValues(event.payload?.memory_ids),
  ]);
  const sourceKinds = new Map<string, string>();
  const decisionIds = uniqueStrings([
    ...stringValues(payload.decision_ids),
    ...stringValues(payload.decision_id),
    ...stringValues(payload.source_decision_ids),
  ]);
  const planIds = uniqueStrings([
    ...stringValues(payload.plan_ids),
    ...stringValues(payload.plan_id),
    ...stringValues(payload.source_plan_ids),
  ]);
  const sourceIds = uniqueStrings([
    ...stringValues(payload.source_ids),
    ...stringValues(payload.source_message_ids),
    ...stringValues(payload.source_event_ids),
    ...decisionIds,
    ...planIds,
  ]);
  stringValues(payload.source_message_ids).forEach((id) => sourceKinds.set(id, "消息"));
  stringValues(payload.source_event_ids).forEach((id) => sourceKinds.set(id, "事件"));
  decisionIds.forEach((id) => sourceKinds.set(id, "决策"));
  planIds.forEach((id) => sourceKinds.set(id, "方案"));
  sourceIds.forEach((id) => {
    if (sourceKinds.has(id)) return;
    if (/^decision(?:[-_:/]|$)/i.test(id)) sourceKinds.set(id, "决策");
    else if (/^plan(?:[-_:/]|$)/i.test(id)) sourceKinds.set(id, "方案");
    else sourceKinds.set(id, "来源");
  });

  return {
    questionIds,
    memoryIds,
    memoryFamilyId: event.memory_family_id ?? payload.memory_family_id ?? null,
    version: event.version ?? payload.version ?? null,
    planVersion: payload.plan_version ?? null,
    sourceIds,
    sourceKinds,
  };
}

function stringValues(value: unknown): string[] {
  if (typeof value === "string") return value ? [value] : [];
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string" && item.length > 0);
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values)];
}

function StatusIcon({
  loading,
  status,
}: {
  loading: boolean;
  status: MemoryStatus | "idle";
}) {
  if (loading || status === "pending" || status === "processing") {
    return <LoaderCircle className="memory-steam-status-icon spinning" size={14} aria-hidden="true" />;
  }
  if (status === "degraded" || status === "failed") {
    return <CircleAlert className="memory-steam-status-icon" size={14} aria-hidden="true" />;
  }
  return <CheckCircle2 className="memory-steam-status-icon" size={14} aria-hidden="true" />;
}

function CausalStep({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="memory-steam-causal-step">
      <span className="memory-steam-causal-label">{label}</span>
      <div className="memory-steam-causal-value">{children}</div>
    </div>
  );
}

function ReferenceValues({ values, empty }: { values: string[]; empty: string }) {
  if (!values.length) return <span className="memory-steam-reference-empty">{empty}</span>;
  return (
    <div className="memory-steam-reference-line">
      {values.map((value) => (
        <span className="memory-steam-reference-text" key={value}>
          {value}
        </span>
      ))}
    </div>
  );
}
