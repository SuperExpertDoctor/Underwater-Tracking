import { describe, expect, it, vi } from "vitest";
import type { OperationalFrame } from "../types/frames";
import { getReplayDelayMs, loadReplayRange } from "./replayApi";

function frame(frameId: number, simTimeS: number): OperationalFrame {
  return {
    schema_version: "1.0",
    frame_id: frameId,
    sim_time_s: simTimeS,
    plan_version: 1,
    map_bounds: { min_x: 0, min_y: 0, max_x: 100, max_y: 100 },
    uuvs: [],
    target_estimates: [],
    bearing_rays: [],
    groups: [],
    events: [],
    plans: [],
    ledger: [],
    metrics: [],
    carrier: null,
  };
}

describe("loadReplayRange", () => {
  it("loads every backend page and preserves the full chronological range", async () => {
    const pages = new Map<number, OperationalFrame[]>([
      [0, [frame(0, 0), frame(1, 5)]],
      [2, [frame(2, 10)]],
    ]);
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const offset = Number(new URL(String(input), "http://ui.test").searchParams.get("offset"));
      return {
        ok: true,
        json: async () => ({
          frames: pages.get(offset) ?? [],
          count: pages.get(offset)?.length ?? 0,
          total_count: 3,
          offset,
          limit: 2,
        }),
      } as Response;
    });

    const result = await loadReplayRange(fetcher, 0, 10, 2);

    expect(result.frames.map((item) => item.sim_time_s)).toEqual([0, 5, 10]);
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(String(fetcher.mock.calls[1][0])).toContain("offset=2");
  });

  it("calculates playback delay from adjacent simulation timestamps", () => {
    expect(getReplayDelayMs(frame(0, 0), frame(1, 5), 100)).toBe(50);
  });

  it("fails instead of silently truncating a range over the client ceiling", async () => {
    const fetcher = vi.fn(async () => ({
      ok: true,
      json: async () => ({ frames: [], count: 0, total_count: 10, offset: 0, limit: 2 }),
    }) as Response);

    await expect(loadReplayRange(fetcher, 0, undefined, 2, 5)).rejects.toThrow(
      "回放帧数量超过客户端上限",
    );
  });
});
