import type { ExpertDirectiveView } from "../types/assistant";

export interface DirectiveRequest {
  text: string;
  author: string;
  expected_plan_version: number;
  target_ids?: string[];
}

export interface AssignmentRequest {
  target_id: string;
  uuv_ids: string[];
  expected_plan_version: number;
}

export interface SensorModeRequest {
  uuv_id: string;
  mode: "passive" | "active";
  target_id?: string | null;
  expected_plan_version: number;
}

export interface DirectiveStatus {
  request_id: string;
  status: string;
  expected_plan_version?: number;
  directive?: ExpertDirectiveView;
  error?: string;
}

export interface QuestionAnswerView {
  answer?: string;
  evidence_ids?: string[];
  counterfactual_plan_id?: string | null;
  counterfactual_summary?: string | null;
  status?: string;
  message?: string;
}

export interface ConversationMessageRequest {
  conversation_id: string;
  text: string;
  expected_plan_version: number;
  target_ids?: string[];
  region_ids?: string[];
  evidence_ids?: string[];
}

export interface ConversationTurnView {
  conversation_id: string;
  turn_id?: string;
  classification: string | { classification: string };
  messages: Array<{
    message_id: string;
    role: string;
    text: string;
    classification?: string;
    evidence_ids?: string[];
    proposal?: ConversationProposalView | null;
  }>;
  proposal?: ConversationProposalView | null;
  answer?: { answer?: string; evidence_ids?: string[] } | null;
  evidence_ids?: string[];
  expected_plan_version: number;
  applied?: boolean;
}

export interface ConversationProposalView {
  proposal_id?: string;
  summary?: string;
  status: string;
  directive?: Record<string, unknown>;
  diff?: Record<string, unknown> | null;
}

const PENDING_DIRECTIVE_STATUSES = new Set(["queued", "processing", "applying"]);
const DEFAULT_DIRECTIVE_POLL_INTERVAL_MS = 500;
const DEFAULT_DIRECTIVE_TIMEOUT_MS = 30_000;
const API_REQUEST_TIMEOUT_MS = 15_000;
const MOCK_MODE = import.meta.env.VITE_MOCK_MODE === "true";
const mockDirectiveJobs = new Map<string, DirectiveStatus>();
let mockRequestSequence = 0;

export interface DirectivePollingOptions {
  intervalMs?: number;
  timeoutMs?: number;
}

export class AssistantApiError extends Error {
  readonly status: number;
  readonly payload: unknown;

  constructor(status: number, payload: unknown) {
    super(`assistant API request failed with HTTP ${status}`);
    this.name = "AssistantApiError";
    this.status = status;
    this.payload = payload;
  }
}

export async function queueDirective(request: DirectiveRequest): Promise<{ request_id: string; status: string }> {
  if (MOCK_MODE) {
    const requestId = `mock-directive-${++mockRequestSequence}`;
    const directive = createMockDirective(requestId, request);
    mockDirectiveJobs.set(requestId, { request_id: requestId, status: "queued", expected_plan_version: request.expected_plan_version, directive });
    return { request_id: requestId, status: "queued" };
  }
  return requestJson("/api/directives", { method: "POST", body: JSON.stringify(request) });
}

export async function getDirectiveStatus(requestId: string): Promise<DirectiveStatus> {
  if (MOCK_MODE) {
    const current = mockDirectiveJobs.get(requestId);
    if (!current) return { request_id: requestId, status: "error", error: "Mock 指令不存在" };
    if (current.status === "queued") {
      const next = { ...current, status: "preview" };
      mockDirectiveJobs.set(requestId, next);
      return next;
    }
    return current;
  }
  return requestJson(`/api/directives/${encodeURIComponent(requestId)}`);
}

export async function waitForDirectiveStatus(
  requestId: string,
  options: DirectivePollingOptions = {},
): Promise<DirectiveStatus> {
  const intervalMs = options.intervalMs ?? DEFAULT_DIRECTIVE_POLL_INTERVAL_MS;
  const timeoutMs = options.timeoutMs ?? DEFAULT_DIRECTIVE_TIMEOUT_MS;
  const startedAt = Date.now();
  let status = await getDirectiveStatus(requestId);

  while (PENDING_DIRECTIVE_STATUSES.has(status.status)) {
    const remainingMs = timeoutMs - (Date.now() - startedAt);
    if (remainingMs <= 0) {
      throw new Error("指令处理超时，请检查 LLM/本体服务后重试。");
    }
    await delay(Math.min(intervalMs, remainingMs));
    status = await getDirectiveStatus(requestId);
  }
  return status;
}

export async function applyDirective(requestId: string): Promise<{ request_id: string; status: string }> {
  if (MOCK_MODE) {
    const current = mockDirectiveJobs.get(requestId);
    if (current) mockDirectiveJobs.set(requestId, { ...current, status: "applied", directive: current.directive ? { ...current.directive, status: "applied" } : undefined });
    return { request_id: requestId, status: "applied" };
  }
  return requestJson(`/api/directives/${encodeURIComponent(requestId)}/apply`, { method: "POST" });
}

export async function assignTargets(request: AssignmentRequest): Promise<{ request_id: string; status: string }> {
  if (MOCK_MODE) {
    const requestId = `mock-assignment-${++mockRequestSequence}`;
    const directive: ExpertDirectiveView = {
      directive_id: requestId,
      raw_text: `将 ${request.uuv_ids.join("、")} 指派至 ${request.target_id}`,
      target_scope: [request.target_id],
      locked_members: { [request.target_id]: request.uuv_ids },
      target_priorities: { [request.target_id]: 1 },
      minimum_quality: { [request.target_id]: 0.7 },
      disabled_uuv_ids: [],
      return_uuv_ids: [],
      directive_type: "assignment",
      assignment_target_id: request.target_id,
      assignment_uuv_ids: request.uuv_ids,
      confidence: 0.98,
      conflicts: [],
      status: "preview",
    };
    mockDirectiveJobs.set(requestId, { request_id: requestId, status: "queued", expected_plan_version: request.expected_plan_version, directive });
    return { request_id: requestId, status: "queued" };
  }
  return requestJson("/api/assignments", { method: "POST", body: JSON.stringify(request) });
}

export async function setSensorMode(request: SensorModeRequest): Promise<{ status: string; passive_continuous: boolean }> {
  if (MOCK_MODE) return { status: "applied", passive_continuous: request.mode === "passive" };
  return requestJson("/api/sensor-modes", { method: "POST", body: JSON.stringify(request) });
}

export async function askQuestion(text: string, counterfactual?: Record<string, unknown>): Promise<QuestionAnswerView> {
  if (MOCK_MODE) {
    const suffix = counterfactual ? "（已基于反事实方案重新评估）" : "";
    return {
      answer: `当前 T-ALPHA 跟踪质量约为 0.87，6 艘 UUV 中有 5 艘保持稳定方位观测；建议继续保持编组并关注 UUV-06 的返航窗口。${suffix}`,
      evidence_ids: ["bearing-window-01", "energy-window-01"],
      counterfactual_plan_id: counterfactual ? "mock-counterfactual-01" : null,
      counterfactual_summary: counterfactual ? "反事实结果：质量略有下降，但仍高于最低门限。" : null,
      status: "answered",
    };
  }
  return requestJson("/api/questions", {
    method: "POST",
    body: JSON.stringify({ text, ...(counterfactual ? { counterfactual } : {}) }),
  });
}

export async function sendConversationMessage(request: ConversationMessageRequest): Promise<ConversationTurnView> {
  if (MOCK_MODE) return createMockConversationTurn(request);
  return requestJson("/api/conversation/messages", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function applyConversation(
  conversationId: string,
  turnId: string,
  expectedPlanVersion: number,
): Promise<ConversationTurnView> {
  if (MOCK_MODE) {
    return {
      conversation_id: conversationId,
      turn_id: turnId,
      classification: "plan_revision",
      messages: [{ message_id: `${turnId}:assistant`, role: "assistant", text: "Mock 模式已应用方案修正：保持 T-ALPHA 跟踪编组，并为下一窗口预留接替资源。", evidence_ids: ["region-forecast-01"] }],
      proposal: { proposal_id: `${conversationId}:proposal`, summary: "保持当前编组并预留接替资源", status: "applied" },
      evidence_ids: ["region-forecast-01"],
      expected_plan_version: expectedPlanVersion,
      applied: true,
    };
  }
  return requestJson(`/api/conversation/${encodeURIComponent(conversationId)}/apply`, {
    method: "POST",
    body: JSON.stringify({ turn_id: turnId, expected_plan_version: expectedPlanVersion }),
  });
}

async function requestJson<T>(url: string, init: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), API_REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, {
      ...init,
      signal: init.signal ?? controller.signal,
      headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
    });
    const payload: unknown = await response.json().catch(() => ({}));
    if (!response.ok) throw new AssistantApiError(response.status, payload);
    return payload as T;
  } catch (reason: unknown) {
    if (isAbortError(reason)) {
      throw new Error("请求超时，请检查后端服务后重试。");
    }
    throw reason;
  } finally {
    window.clearTimeout(timeout);
  }
}

function delay(durationMs: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, durationMs));
}

function isAbortError(reason: unknown): boolean {
  return typeof reason === "object" && reason !== null && "name" in reason
    && (reason as { name?: unknown }).name === "AbortError";
}

function createMockDirective(requestId: string, request: DirectiveRequest): ExpertDirectiveView {
  const text = request.text.toLowerCase();
  const returnUuvIds = text.includes("返航") || text.includes("轮换") ? ["uuv-06"] : [];
  const disabledUuvIds = text.includes("禁用") ? ["uuv-12"] : [];
  return {
    directive_id: requestId,
    raw_text: request.text,
    target_scope: request.target_ids?.length ? request.target_ids : ["T-ALPHA"],
    locked_members: {},
    target_priorities: { "T-ALPHA": 1 },
    minimum_quality: { "T-ALPHA": 0.7 },
    disabled_uuv_ids: disabledUuvIds,
    return_uuv_ids: returnUuvIds,
    directive_type: returnUuvIds.length || disabledUuvIds.length ? "constraint" : "assignment",
    assignment_target_id: request.target_ids?.[0] ?? "T-ALPHA",
    assignment_uuv_ids: [],
    confidence: 0.9,
    conflicts: [],
    status: "preview",
  };
}

function createMockConversationTurn(request: ConversationMessageRequest): ConversationTurnView {
  const planRevision = /方案|编组|接力|返航|指派/.test(request.text);
  const classification = planRevision ? "mixed" : "evidence_query";
  const evidenceIds = ["bearing-window-01", "region-forecast-01"];
  return {
    conversation_id: request.conversation_id,
    turn_id: `${request.conversation_id}:turn:${Date.now()}`,
    classification,
    messages: [
      { message_id: `${request.conversation_id}:user:${Date.now()}`, role: "user", text: request.text },
      { message_id: `${request.conversation_id}:assistant:${Date.now() + 1}`, role: "assistant", text: planRevision ? "Mock 分析：当前编组满足最低质量门限，建议保持 T-ALPHA 的主跟踪组，同时让备用资源准备区域接力。" : "Mock 分析：当前观测由多艘 UUV 提供独立方位，目标预测走廊暂无明显失稳。", evidence_ids: evidenceIds },
    ],
    proposal: planRevision ? { proposal_id: `${request.conversation_id}:proposal`, summary: "保持当前编组并预留区域接力资源", status: "preview", directive: { target_ids: request.target_ids ?? ["T-ALPHA"] }, diff: { plan_version: "3 → 4" } } : null,
    answer: { answer: "Mock 模式已完成证据检索。", evidence_ids: evidenceIds },
    evidence_ids: evidenceIds,
    expected_plan_version: request.expected_plan_version,
    applied: false,
  };
}
