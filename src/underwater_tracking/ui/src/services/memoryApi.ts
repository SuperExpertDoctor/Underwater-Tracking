export type MemoryFamily = "episodic" | "semantic" | "procedural";
export type MemoryStatus = "pending" | "processing" | "completed" | "degraded" | "failed";

export interface ShortTermMessageView {
  message_id: string;
  scenario_id?: string | null;
  turn_id?: string | null;
  role: "expert" | "user" | "assistant";
  text: string;
  source_evidence_ids?: string[];
  created_at?: string;
}

export interface ShortTermContextView {
  user_id: string;
  scenario_id?: string | null;
  conversation_id: string;
  summary_text: string;
  summary_version: number;
  recent_messages: ShortTermMessageView[];
  message_count: number;
  estimated_tokens: number;
  compression_count: number;
  last_compressed_at?: string | null;
  compression_status: MemoryStatus;
  updated_at?: string;
}

export interface MemoryVersionView {
  memory_id: string;
  memory_family_id: string;
  version: number;
  user_id: string;
  scenario_id?: string | null;
  memory_type: MemoryFamily;
  summary: string;
  importance_score: number;
  embedding?: number[];
  embedding_version?: string;
  status: string;
  supersedes_memory_id?: string | null;
  source_message_ids?: string[];
  source_event_ids?: string[];
  source_decision_ids?: string[];
  source_knowledge_ids?: string[];
  source_plan_ids?: string[];
  change_reason?: string;
  created_at?: string;
  last_accessed_at?: string | null;
  access_count: number;
  sim_time_s?: number | null;
}

export interface MemoryRetrievalHitView {
  memory: MemoryVersionView;
  similarity_score: number;
  rerank_score: number;
  retrieval_reason: string;
}

export interface MemoryEvidenceTraceView {
  trace_id: string;
  user_id: string;
  status: MemoryStatus;
  memory_ids: string[];
  source_message_ids: string[];
  source_event_ids: string[];
  source_decision_ids: string[];
  source_knowledge_ids: string[];
  source_plan_ids: string[];
  created_at?: string;
}

export interface MemoryContextView {
  user_id: string;
  scenario_id?: string | null;
  short_term_context?: ShortTermContextView | null;
  long_term_material: MemoryRetrievalHitView[];
  retrieved_memory_ids: string[];
  memory_status: MemoryStatus;
  degraded_reason?: string | null;
  evidence_trace: MemoryEvidenceTraceView[];
}

export interface MemorySnapshotView {
  user_id: string;
  scenario_id?: string | null;
  conversation_id?: string | null;
  short_term: ShortTermContextView | null;
  episodic: MemoryVersionView[];
  semantic: MemoryVersionView[];
  procedural: MemoryVersionView[];
  retrieved_hits: MemoryRetrievalHitView[];
  versions: MemoryVersionView[];
  memory_status: MemoryStatus;
  degraded_reason?: string | null;
}

export interface MemoryStreamEventView {
  cursor: number;
  event_id: string;
  user_id: string;
  scenario_id?: string | null;
  status: MemoryStatus;
  type: string;
  payload?: {
    reason_code?: string | null;
    hit_count?: number | null;
    memory_ids?: string[];
    memory_family_id?: string | null;
    work_id?: string | null;
    memory_type?: MemoryFamily | null;
    version?: number | null;
    summary_version?: number | null;
    source_ids?: string[];
    source_message_ids?: string[];
    source_event_ids?: string[];
    source_decision_ids?: string[];
    source_knowledge_ids?: string[];
    source_plan_ids?: string[];
    plan_version?: number | null;
    operation?: "create" | "update" | "ignore" | null;
  };
  conversation_id?: string | null;
  memory_id?: string | null;
  memory_family_id?: string | null;
  version?: number | null;
  created_at?: string;
  sim_time_s?: number | null;
}

export interface MemoryStreamView {
  user_id: string;
  conversation_id: string;
  scenario_id: string;
  events: MemoryStreamEventView[];
  after_cursor: number;
  next_cursor: number;
  include_scenario_events?: boolean;
  memory_status: MemoryStatus;
  degraded_reason?: string | null;
}

export interface MemorySnapshotRequest {
  userId: string;
  conversationId: string;
  scenarioId: string;
  query?: string;
  memoryType?: MemoryFamily;
  minImportanceScore?: number;
  limit?: number;
}

export interface MemoryVersionsRequest {
  userId: string;
  memoryFamilyId: string;
  scenarioId: string;
}

export interface MemoryDeleteRequest {
  userId: string;
  memoryId: string;
  scenarioId: string;
  conversationId: string;
}

export interface MemoryStreamRequest {
  userId: string;
  conversationId: string;
  scenarioId: string;
  afterCursor?: number;
  limit?: number;
  includeScenarioEvents?: boolean;
}

export interface MemoryVersionsView {
  user_id: string;
  memory_family_id: string;
  versions: MemoryVersionView[];
}

export class MemoryApiError extends Error {
  readonly status: number;
  readonly payload: unknown;

  constructor(status: number, payload: unknown) {
    super(`memory API request failed with HTTP ${status}`);
    this.name = "MemoryApiError";
    this.status = status;
    this.payload = payload;
  }
}

export class MemoryScopeError extends Error {
  constructor(message: string) {
    super(`memory API scope mismatch: ${message}`);
    this.name = "MemoryScopeError";
  }
}

const API_REQUEST_TIMEOUT_MS = 15_000;

export function getMemorySnapshot(request: MemorySnapshotRequest): Promise<MemorySnapshotView> {
  const params = new URLSearchParams({
    user_id: request.userId,
    conversation_id: request.conversationId,
  });
  addOptional(params, "scenario_id", request.scenarioId);
  addOptional(params, "query", request.query);
  addOptional(params, "memory_type", request.memoryType);
  addOptional(params, "min_importance_score", request.minImportanceScore);
  addOptional(params, "limit", request.limit);
  return requestJson<MemorySnapshotView>(`/api/assistant/memory?${params.toString()}`).then((payload) => {
    validateSnapshotScope(payload, request);
    return payload;
  });
}

export function getMemoryVersions(request: MemoryVersionsRequest): Promise<MemoryVersionsView> {
  const params = new URLSearchParams({ user_id: request.userId });
  addOptional(params, "scenario_id", request.scenarioId);
  return requestJson<MemoryVersionsView>(
    `/api/assistant/memory/${encodeURIComponent(request.memoryFamilyId)}/versions?${params.toString()}`,
  ).then((payload) => {
    validateVersionsScope(payload, request);
    return payload;
  });
}

export function deleteMemory(request: MemoryDeleteRequest): Promise<{ status: string; memory_id: string; user_id: string }> {
  const params = new URLSearchParams({ user_id: request.userId });
  addOptional(params, "scenario_id", request.scenarioId);
  addOptional(params, "conversation_id", request.conversationId);
  return requestJson<{ status: string; memory_id: string; user_id: string }>(`/api/assistant/memory/${encodeURIComponent(request.memoryId)}?${params.toString()}`, {
    method: "DELETE",
  }).then((payload) => {
    if (payload.user_id !== request.userId) {
      throw new MemoryScopeError("delete response user scope mismatch");
    }
    return payload;
  });
}

export function getMemoryStream(request: MemoryStreamRequest): Promise<MemoryStreamView> {
  const params = new URLSearchParams({
    user_id: request.userId,
    conversation_id: request.conversationId,
    after_cursor: String(request.afterCursor ?? 0),
    limit: String(request.limit ?? 100),
  });
  addOptional(params, "scenario_id", request.scenarioId);
  if (request.includeScenarioEvents !== undefined) {
    params.set("include_scenario_events", String(request.includeScenarioEvents));
  }
  return requestJson<MemoryStreamView>(`/api/assistant/memory/stream?${params.toString()}`).then((payload) => {
    validateStreamScope(payload, request);
    return payload;
  });
}

function addOptional(params: URLSearchParams, key: string, value: string | number | undefined): void {
  if (value !== undefined && value !== "") params.set(key, String(value));
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
    if (!response.ok) throw new MemoryApiError(response.status, payload);
    return payload as T;
  } catch (reason: unknown) {
    if (isAbortError(reason)) throw new Error("记忆请求超时，请检查后端服务后重试。");
    throw reason;
  } finally {
    window.clearTimeout(timeout);
  }
}

function validateSnapshotScope(payload: MemorySnapshotView, request: MemorySnapshotRequest): void {
  if (payload.user_id !== request.userId) throw new MemoryScopeError("snapshot user scope mismatch");
  if (payload.conversation_id !== request.conversationId) throw new MemoryScopeError("snapshot conversation scope mismatch");
  if (payload.scenario_id !== request.scenarioId) throw new MemoryScopeError("snapshot scenario scope mismatch");
  validateContextScope(payload.short_term, request, "snapshot short-term");
  [...payload.episodic, ...payload.semantic, ...payload.procedural, ...payload.versions].forEach((item) => {
    validateVersionScope(item, request, "snapshot memory");
  });
  payload.retrieved_hits.forEach((hit) => validateVersionScope(hit.memory, request, "snapshot retrieval"));
}

function validateVersionsScope(payload: MemoryVersionsView, request: MemoryVersionsRequest): void {
  if (payload.user_id !== request.userId) throw new MemoryScopeError("versions user scope mismatch");
  if (payload.memory_family_id !== request.memoryFamilyId) throw new MemoryScopeError("versions family scope mismatch");
  payload.versions.forEach((item) => validateVersionScope(item, request, "versions"));
}

function validateStreamScope(payload: MemoryStreamView, request: MemoryStreamRequest): void {
  if (payload.user_id !== request.userId) throw new MemoryScopeError("stream user scope mismatch");
  if (payload.conversation_id !== request.conversationId) throw new MemoryScopeError("stream conversation scope mismatch");
  if (payload.scenario_id !== request.scenarioId) throw new MemoryScopeError("stream scenario scope mismatch");
  const includeScenarioEvents = request.includeScenarioEvents ?? true;
  if (
    payload.include_scenario_events !== undefined
    && payload.include_scenario_events !== includeScenarioEvents
  ) {
    throw new MemoryScopeError("stream scenario event scope mismatch");
  }
  const requestedAfter = request.afterCursor ?? 0;
  if (!Number.isInteger(payload.after_cursor) || payload.after_cursor !== requestedAfter) {
    throw new MemoryScopeError("stream cursor scope mismatch");
  }
  if (!Number.isInteger(payload.next_cursor) || payload.next_cursor < requestedAfter) {
    throw new MemoryScopeError("stream cursor sequence mismatch");
  }
  let previousCursor = requestedAfter;
  let maximumCursor = requestedAfter;
  payload.events.forEach((event) => {
    if (event.user_id !== request.userId) throw new MemoryScopeError("stream event user scope mismatch");
    if (
      event.conversation_id !== request.conversationId
      && (!includeScenarioEvents || (event.conversation_id !== null && event.conversation_id !== undefined))
    ) {
      throw new MemoryScopeError("stream event conversation scope mismatch");
    }
    if (event.scenario_id !== request.scenarioId) throw new MemoryScopeError("stream event scenario scope mismatch");
    if (!Number.isInteger(event.cursor) || event.cursor <= requestedAfter || event.cursor <= previousCursor) {
      throw new MemoryScopeError("stream cursor sequence mismatch");
    }
    previousCursor = event.cursor;
    maximumCursor = Math.max(maximumCursor, event.cursor);
  });
  if (payload.next_cursor !== maximumCursor) throw new MemoryScopeError("stream cursor sequence mismatch");
}

function validateContextScope(
  context: ShortTermContextView | null,
  request: Pick<MemorySnapshotRequest, "userId" | "conversationId" | "scenarioId">,
  label: string,
): void {
  if (!context) return;
  if (context.user_id !== request.userId) throw new MemoryScopeError(`${label} user scope mismatch`);
  if (context.conversation_id !== request.conversationId) throw new MemoryScopeError(`${label} conversation scope mismatch`);
  if (context.scenario_id !== request.scenarioId) throw new MemoryScopeError(`${label} scenario scope mismatch`);
}

function validateVersionScope(
  item: MemoryVersionView,
  request: Pick<MemorySnapshotRequest, "userId" | "scenarioId"> | MemoryVersionsRequest,
  label: string,
): void {
  const familyId = "memoryFamilyId" in request ? request.memoryFamilyId : undefined;
  const expectedScenario = request.scenarioId;
  if (item.user_id !== request.userId) throw new MemoryScopeError(`${label} user scope mismatch`);
  if (item.scenario_id !== expectedScenario) throw new MemoryScopeError(`${label} scenario scope mismatch`);
  if (familyId !== undefined && item.memory_family_id !== familyId) {
    throw new MemoryScopeError(`${label} family scope mismatch`);
  }
}

function isAbortError(reason: unknown): boolean {
  return typeof reason === "object" && reason !== null && "name" in reason
    && (reason as { name?: unknown }).name === "AbortError";
}
