import type { ExpertDirectiveView } from "../types/assistant";
import type { MemoryContextView, MemoryEvidenceTraceView } from "./memoryApi";

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
  user_id: string;
  assistant_mode: "auto" | "plan_revision" | "evidence_query";
  text: string;
  expected_plan_version: number;
  target_ids?: string[];
  region_ids?: string[];
  evidence_ids?: string[];
}

export interface ConversationTurnView {
  conversation_id: string;
  turn_id?: string;
  classification: ConversationClassificationView | string;
  messages: Array<{
    message_id: string;
    role: string;
    text: string;
    classification?: string;
    evidence_ids?: string[];
    proposal?: ConversationProposalView | null;
  }>;
  proposal?: ConversationProposalView | null;
  answer?: {
    answer?: string;
    evidence_ids?: string[];
    memory_ids?: string[];
    memory_status?: string | null;
    evidence_trace?: MemoryEvidenceTraceView[];
  } | null;
  evidence_ids?: string[];
  expected_plan_version: number;
  applied?: boolean;
  user_id?: string;
  assistant_mode?: "auto" | "plan_revision" | "evidence_query";
  memory_context?: MemoryContextView | null;
  memory_stream_cursor?: number | null;
  queued_memory_work_id?: string | null;
}

export interface ConversationClassificationView {
  classification: "plan_revision" | "evidence_query" | "mixed" | "clarification";
  confidence?: number;
  target_scope?: string[];
  region_scope?: string[];
  evidence_ids?: string[];
  memory_ids?: string[];
}

export interface ConversationProposalView {
  proposal_id?: string;
  expected_plan_version?: number;
  summary?: string;
  status: string;
  directive?: Record<string, unknown>;
  diff?: Record<string, unknown> | null;
}

const PENDING_DIRECTIVE_STATUSES = new Set(["queued", "processing", "applying"]);
const DEFAULT_DIRECTIVE_POLL_INTERVAL_MS = 500;
const DEFAULT_DIRECTIVE_TIMEOUT_MS = 30_000;
const API_REQUEST_TIMEOUT_MS = 15_000;

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
  return requestJson("/api/directives", { method: "POST", body: JSON.stringify(request) });
}

export async function getDirectiveStatus(requestId: string): Promise<DirectiveStatus> {
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
  return requestJson(`/api/directives/${encodeURIComponent(requestId)}/apply`, { method: "POST" });
}

export async function assignTargets(request: AssignmentRequest): Promise<{ request_id: string; status: string }> {
  return requestJson("/api/assignments", { method: "POST", body: JSON.stringify(request) });
}

export async function setSensorMode(request: SensorModeRequest): Promise<{ status: string; passive_continuous: boolean }> {
  return requestJson("/api/sensor-modes", { method: "POST", body: JSON.stringify(request) });
}

export async function askQuestion(text: string, counterfactual?: Record<string, unknown>): Promise<QuestionAnswerView> {
  return requestJson("/api/questions", {
    method: "POST",
    body: JSON.stringify({ text, ...(counterfactual ? { counterfactual } : {}) }),
  });
}

export async function sendConversationMessage(request: ConversationMessageRequest): Promise<ConversationTurnView> {
  return requestJson("/api/conversation/messages", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function applyConversation(
  conversationId: string,
  turnId: string,
  expectedPlanVersion: number,
  userId = "operator",
): Promise<ConversationTurnView> {
  return requestJson(`/api/conversation/${encodeURIComponent(conversationId)}/apply`, {
    method: "POST",
    body: JSON.stringify({
      user_id: userId,
      turn_id: turnId,
      expected_plan_version: expectedPlanVersion,
    }),
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
