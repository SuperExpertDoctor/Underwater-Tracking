import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { OperationalFrame } from "../types/frames";
import useWebSocket from "./useWebSocket";

function frame(frameId: number, simTimeS: number): OperationalFrame {
  return {
    schema_version: "1.0",
    frame_id: frameId,
    sim_time_s: simTimeS,
    plan_version: 1,
    map_bounds: { min_x: 0, min_y: 0, max_x: 100, max_y: 100 },
    uuvs: [], target_estimates: [], bearing_rays: [], groups: [], events: [],
    plans: [], ledger: [], metrics: [],
    carrier: null,
  };
}

class MockWebSocket {
  static readonly OPEN = 1;
  static instances: MockWebSocket[] = [];
  readonly url: string;
  readyState = 0;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  send(_message: string) {}
  close() {
    this.readyState = 3;
    this.onclose?.();
  }
  open() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }
  emit(message: unknown) {
    this.onmessage?.({ data: JSON.stringify(message) });
  }
}

describe("useWebSocket", () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket);
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      return window.setTimeout(() => callback(performance.now()), 0);
    });
    vi.stubGlobal("cancelAnimationFrame", (handle: number) => window.clearTimeout(handle));
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => frame(1, 30) })));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("loads a fresh snapshot before accepting monotonic live deltas", async () => {
    const { result } = renderHook(() => useWebSocket(true));
    const socket = MockWebSocket.instances[0];
    expect(socket.url).toContain("/ws/operational");

    await act(async () => {
      socket.open();
      await Promise.resolve();
    });
    await waitFor(() => expect(result.current.frame?.frame_id).toBe(1));
    expect(vi.mocked(fetch)).toHaveBeenCalledWith("/api/operational/snapshot");

    await act(async () => {
      socket.emit(frame(0, 20));
      socket.emit(frame(2, 60));
      await new Promise((resolve) => window.setTimeout(resolve, 5));
    });
    expect(result.current.frame?.frame_id).toBe(2);
    expect(result.current.status).toBe("connected");
  });

  it("ignores heartbeat messages", async () => {
    const { result } = renderHook(() => useWebSocket(true));
    const socket = MockWebSocket.instances[0];
    await act(async () => {
      socket.open();
      await Promise.resolve();
    });
    await waitFor(() => expect(result.current.frame).not.toBeNull());
    await act(async () => {
      socket.emit({ type: "heartbeat", sim_time_s: 30 });
      await new Promise((resolve) => window.setTimeout(resolve, 5));
    });
    expect(result.current.frame?.frame_id).toBe(1);
  });
});
