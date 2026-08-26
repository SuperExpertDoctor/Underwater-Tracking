import { act, renderHook } from "@testing-library/react";
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

type FakeSocket = {
  open: () => void;
  emit: (value: OperationalFrame) => void;
  close: () => void;
  closed: boolean;
};

describe("useWebSocket frame lifecycle", () => {
  let sockets: FakeSocket[];
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    sockets = [];
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    class TestWebSocket {
      static OPEN = 1;
      readyState = 0;
      onopen: (() => void) | null = null;
      onmessage: ((event: { data: string }) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      closed = false;

      constructor() {
        const socket = this;
        sockets.push({
          open: () => {
            socket.readyState = TestWebSocket.OPEN;
            socket.onopen?.();
          },
          emit: (value) => socket.onmessage?.({ data: JSON.stringify(value) }),
          close: () => {
            socket.closed = true;
            socket.readyState = 3;
            socket.onclose?.();
          },
          get closed() {
            return socket.closed;
          },
        });
      }

      send() {}

      close() {
        this.closed = true;
        this.readyState = 3;
        this.onclose?.();
      }
    }
    vi.stubGlobal("WebSocket", TestWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("requests an HTTP snapshot after a stream frame jump and cleans timers on unmount", async () => {
    const initial = frame(10, 100);
    const jumped = frame(15, 125);
    const recovered = frame(16, 130);
    fetchMock
      .mockResolvedValueOnce({ ok: true, json: async () => initial })
      .mockResolvedValueOnce({ ok: true, json: async () => recovered });

    const { unmount } = renderHook(() => useWebSocket(true));
    expect(sockets).toHaveLength(1);

    await act(async () => {
      sockets[0].open();
      await Promise.resolve();
      await Promise.resolve();
      vi.runOnlyPendingTimers();
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      sockets[0].emit(jumped);
      await Promise.resolve();
      await Promise.resolve();
      vi.runOnlyPendingTimers();
    });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[1][0]).toBe("/api/operational/snapshot");

    unmount();
    expect(sockets[0].closed).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });
});
