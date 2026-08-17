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

async function requestJson<T>(url: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
  });
  const payload: unknown = await response.json().catch(() => ({}));
  if (!response.ok) throw new AssistantApiError(response.status, payload);
  return payload as T;
}
