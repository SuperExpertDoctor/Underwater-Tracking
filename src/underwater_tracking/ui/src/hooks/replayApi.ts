import type { OperationalFrame } from "../types/frames";
import { MAX_REPLAY_FRAMES, mergeReplayFrames } from "../state/frameStore";

export const REPLAY_PAGE_SIZE = 1_000;

export interface ReplayPageResponse {
  frames: OperationalFrame[];
  count: number;
  total_count: number;
  offset: number;
  limit: number;
}

export interface ReplayRangeResult {
  frames: OperationalFrame[];
  totalCount: number;
}

export type ReplayFetcher = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>;

export async function loadReplayRange(
  fetcher: ReplayFetcher = fetch,
  startS = 0,
  endS?: number,
  pageSize = REPLAY_PAGE_SIZE,
  maxFrames = MAX_REPLAY_FRAMES,
): Promise<ReplayRangeResult> {
  const safePageSize = Math.max(1, Math.floor(pageSize));
  const safeMaxFrames = Math.max(1, Math.floor(maxFrames));
  const normalizedStart = Math.max(0, startS);
  const normalizedEnd = endS === undefined
    ? undefined
    : Math.max(normalizedStart, endS);
  let offset = 0;
  let totalCount: number | null = null;
  let frames: OperationalFrame[] = [];

  while (totalCount === null || offset < totalCount) {
    const params = new URLSearchParams({
      start_s: String(normalizedStart),
      offset: String(offset),
      limit: String(safePageSize),
    });
    if (normalizedEnd !== undefined) params.set("end_s", String(normalizedEnd));
    const response = await fetcher(`/api/replay?${params.toString()}`);
    if (!response.ok) throw new Error(`回放接口 HTTP ${response.status}`);
    const payload = (await response.json()) as Partial<ReplayPageResponse>;
    const declaredTotal = payload.total_count;
    if (
      !Array.isArray(payload.frames)
      || typeof declaredTotal !== "number"
      || !Number.isInteger(declaredTotal)
      || declaredTotal < 0
    ) {
      throw new Error("回放接口返回的数据格式无效");
    }
    if (totalCount === null) totalCount = declaredTotal;
    if (declaredTotal !== totalCount) {
      throw new Error("回放接口返回的总数在分页期间发生变化");
    }
    if (totalCount > safeMaxFrames) {
      throw new Error("回放帧数量超过客户端上限");
    }
    if (payload.offset !== undefined && payload.offset !== offset) {
      throw new Error("回放接口分页偏移不连续");
    }
    const pageFrames = payload.frames;
    if (pageFrames.length === 0) {
      if (offset < totalCount) throw new Error("回放接口提前返回空页");
      break;
    }
    if (offset + pageFrames.length > totalCount) {
      throw new Error("回放接口返回帧数超过声明总数");
    }
    frames = mergeReplayFrames(frames, pageFrames, safeMaxFrames);
    offset += pageFrames.length;
  }

  if (totalCount === null) return { frames: [], totalCount: 0 };
  if (frames.length !== totalCount) {
    throw new Error("回放数据不完整或包含重复帧");
  }
  return { frames, totalCount };
}

export function getReplayDelayMs(
  current: OperationalFrame,
  next: OperationalFrame,
  speed: number,
): number {
  const fallbackStepS = Math.max(0.001, current.physics_step_s ?? 5);
  const deltaS = next.sim_time_s - current.sim_time_s;
  const effectiveDeltaS = Number.isFinite(deltaS) && deltaS > 0 ? deltaS : fallbackStepS;
  const effectiveSpeed = Number.isFinite(speed) && speed > 0 ? speed : 1;
  return Math.max(1, (effectiveDeltaS * 1000) / effectiveSpeed);
}
