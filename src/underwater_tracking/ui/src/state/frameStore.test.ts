import { describe, expect, it } from "vitest";
import type { OperationalFrame } from "../types/frames";
import {
  acceptLiveFrame,
  createFrameStoreState,
  frameOrder,
  reduceOperationalFrame,
  mergeReplayFrames,
  isHeartbeat,
} from "./frameStore";

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

describe("operational frame store rules", () => {
  it("orders frames by simulation time before frame id", () => {
    expect(frameOrder(frame(2, 30), frame(1, 30))).toBeGreaterThan(0);
    expect(frameOrder(frame(1, 20), frame(2, 30))).toBeLessThan(0);
  });

  it("rejects stale or duplicate live frames", () => {
    const first = frame(4, 30);
    expect(acceptLiveFrame(null, first)).toEqual({ accepted: true, frame: first });
    expect(acceptLiveFrame(first, frame(3, 30)).accepted).toBe(false);
    expect(acceptLiveFrame(first, frame(4, 30)).accepted).toBe(false);
    expect(acceptLiveFrame(first, frame(5, 60)).accepted).toBe(true);
  });

  it("bounds replay memory and keeps replay separate from live state", () => {
    const replay = Array.from({ length: 5 }, (_, index) => frame(index, index));
    const merged = mergeReplayFrames(replay, [frame(5, 5), frame(6, 6)], 5);
    expect(merged).toHaveLength(5);
    expect(merged.map((item) => item.frame_id)).toEqual([2, 3, 4, 5, 6]);
  });

  it("keeps all 8-hour frames under the replay ceiling", () => {
    const replay = Array.from({ length: 5760 }, (_, index) => frame(index, index * 5));

    const merged = mergeReplayFrames([], replay);

    expect(merged).toHaveLength(5760);
    expect(merged[0].sim_time_s).toBe(0);
    expect(merged.at(-1)?.sim_time_s).toBe(28795);
  });

  it("recognizes heartbeat messages without treating them as frames", () => {
    expect(isHeartbeat({ type: "heartbeat", sim_time_s: 30 })).toBe(true);
    expect(isHeartbeat(frame(1, 30))).toBe(false);
  });

  it("uses frame id as the primary stream order and marks a jump for HTTP recovery", () => {
    const first = frame(10, 100);
    const laterButOlderSimulationTime = frame(11, 95);
    const jumped = frame(15, 125);
    let state = createFrameStoreState();

    state = reduceOperationalFrame(state, { type: "frame", frame: first }).state;
    state = reduceOperationalFrame(state, {
      type: "frame",
      frame: laterButOlderSimulationTime,
    }).state;
    expect(state.frame?.frame_id).toBe(11);

    const recovery = reduceOperationalFrame(state, { type: "frame", frame: jumped });
    expect(recovery.accepted).toBe(true);
    expect(recovery.requestSnapshot).toBe(true);
    expect(recovery.state.frame?.frame_id).toBe(15);
  });

  it("clears the recovery marker only after a newer HTTP snapshot is accepted", () => {
    const first = frame(10, 100);
    const jumped = frame(15, 125);
    let state = createFrameStoreState();
    state = reduceOperationalFrame(state, { type: "frame", frame: first }).state;
    state = reduceOperationalFrame(state, { type: "frame", frame: jumped }).state;
    expect(state.needsSnapshot).toBe(true);

    const recovered = reduceOperationalFrame(state, {
      type: "snapshot",
      frame: frame(16, 130),
    });
    expect(recovered.accepted).toBe(true);
    expect(recovered.state.needsSnapshot).toBe(false);
    expect(recovered.state.frame?.frame_id).toBe(16);
  });
});
