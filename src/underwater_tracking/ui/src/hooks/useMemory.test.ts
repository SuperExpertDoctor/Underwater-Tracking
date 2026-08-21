import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MemorySnapshotView, MemoryStreamView } from "../services/memoryApi";
import useMemory from "./useMemory";

function response(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status });
}

const snapshot = (scenarioId: string): MemorySnapshotView => ({
  user_id: "operator",
  conversation_id: "conversation-1",
  scenario_id: scenarioId,
  short_term: null,
  episodic: [],
  semantic: [],
  procedural: [],
  retrieved_hits: [],
  versions: [],
  memory_status: "completed",
  degraded_reason: null,
});

const stream = (scenarioId: string, cursor: number): MemoryStreamView => ({
  user_id: "operator",
  conversation_id: "conversation-1",
  scenario_id: scenarioId,
  events: cursor
    ? [{
        cursor,
        event_id: `${scenarioId}-${cursor}`,
        user_id: "operator",
        conversation_id: "conversation-1",
        scenario_id: scenarioId,
        status: "completed",
        type: "memory_extracted",
      }]
    : [],
  after_cursor: 0,
  next_cursor: cursor,
  memory_status: "completed",
  degraded_reason: null,
});

describe("useMemory", () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("drops an older snapshot and stream response after the scope changes", async () => {
    let resolveSnapshotA!: (value: MemorySnapshotView) => void;
    let resolveStreamA!: (value: MemoryStreamView) => void;
    const snapshotA = new Promise<Response>((resolve) => { resolveSnapshotA = (value) => resolve(response(value)); });
    const streamA = new Promise<Response>((resolve) => { resolveStreamA = (value) => resolve(response(value)); });
    fetchMock
      .mockReturnValueOnce(snapshotA)
      .mockReturnValueOnce(streamA)
      .mockResolvedValueOnce(response(snapshot("scenario-b")))
      .mockResolvedValueOnce(response(stream("scenario-b", 3)));

    const { result, rerender } = renderHook(
      ({ scenarioId }: { scenarioId?: string }) => useMemory({
        userId: "operator",
        conversationId: "conversation-1",
        scenarioId,
        enabled: true,
      }),
      { initialProps: { scenarioId: "scenario-a" } },
    );

    rerender({ scenarioId: "scenario-b" });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    resolveSnapshotA(snapshot("scenario-a"));
    resolveStreamA(stream("scenario-a", 9));
    await act(async () => { await Promise.resolve(); });

    expect(result.current.snapshot?.scenario_id).toBe("scenario-b");
    expect(result.current.events.map((event) => event.event_id)).toEqual(["scenario-b-3"]);
  });

  it("keeps stream polling single-flight while a cursor request is pending", async () => {
    let resolveStream!: (value: MemoryStreamView) => void;
    fetchMock
      .mockResolvedValueOnce(response(snapshot("scenario-a")))
      .mockReturnValueOnce(new Promise<Response>((resolve) => {
        resolveStream = (value) => resolve(response(value));
      }));

    renderHook(() => useMemory({
      userId: "operator",
      conversationId: "conversation-1",
      scenarioId: "scenario-a",
      enabled: true,
    }));

    await act(async () => {
      vi.advanceTimersByTime(5_000);
      await Promise.resolve();
    });
    expect(fetchMock.mock.calls.filter(([url]) => String(url).includes("/api/assistant/memory/stream"))).toHaveLength(1);

    resolveStream(stream("scenario-a", 1));
  });

  it("reports the stream request loading state independently from the snapshot", async () => {
    let resolveStream!: (value: MemoryStreamView) => void;
    fetchMock
      .mockResolvedValueOnce(response(snapshot("scenario-a")))
      .mockReturnValueOnce(new Promise<Response>((resolve) => {
        resolveStream = (value) => resolve(response(value));
      }));

    const { result } = renderHook(() => useMemory({
      userId: "operator",
      conversationId: "conversation-1",
      scenarioId: "scenario-a",
      enabled: true,
    }));

    await act(async () => { await Promise.resolve(); });
    expect(result.current.snapshotLoading).toBe(false);
    expect(result.current.streamLoading).toBe(true);

    await act(async () => {
      resolveStream(stream("scenario-a", 1));
      await Promise.resolve();
    });
    expect(result.current.streamLoading).toBe(false);
  });

  it("does not query memory until an authoritative scenario is available", async () => {
    const { result } = renderHook(() => useMemory({
      userId: "operator",
      conversationId: "conversation-1",
      enabled: true,
    }));

    await act(async () => { await Promise.resolve(); });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(result.current.scopeUnavailable).toBe(true);
  });
});
