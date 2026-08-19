import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { OperationalFrame } from "../types/frames";
import useReplay from "./useReplay";

function frame(frameId: number, physicsStepS = 5): OperationalFrame {
  return {
    schema_version: "1.0", frame_id: frameId, sim_time_s: frameId * physicsStepS,
    physics_step_s: physicsStepS, plan_version: 1,
    map_bounds: { min_x: 0, min_y: 0, max_x: 100, max_y: 100 },
    uuvs: [], target_estimates: [], bearing_rays: [], groups: [], events: [], plans: [], ledger: [], metrics: [],
    carrier: null,
  };
}

describe("useReplay", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      json: async () => ({ frames: Array.from({ length: 605 }, (_, index) => frame(index)), count: 605 }),
    })));
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("loads a bounded time range through the replay API", async () => {
    const { result } = renderHook(() => useReplay(false));
    await act(async () => {
      await result.current.loadRange(0, 18_150);
    });
    expect(vi.mocked(fetch)).toHaveBeenCalledWith("/api/replay?start_s=0&end_s=18150");
    expect(result.current.frames).toHaveLength(600);
    expect(result.current.frames.at(-1)?.frame_id).toBe(604);
    expect(result.current.frame?.frame_id).toBe(5);
  });

  it("paces playback using the physical simulation step", async () => {
    vi.useFakeTimers();
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ frames: [frame(1, 5), frame(2, 5)], count: 2 }),
    } as Response);
    const { result } = renderHook(() => useReplay(true));
    await act(async () => {
      await vi.runOnlyPendingTimersAsync();
    });
    act(() => result.current.setIsPlaying(true));
    expect(result.current.index).toBe(0);
    act(() => vi.advanceTimersByTime(4_999));
    expect(result.current.index).toBe(0);
    act(() => vi.advanceTimersByTime(1));
    expect(result.current.index).toBe(1);
    vi.useRealTimers();
  });
});
