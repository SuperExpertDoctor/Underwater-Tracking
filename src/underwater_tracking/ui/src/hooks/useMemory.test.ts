import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getMemorySnapshot,
  getMemoryStream,
  type MemorySnapshotView,
  type MemoryStreamView,
} from "../services/memoryApi";
import useMemory from "./useMemory";

vi.mock("../services/memoryApi", () => ({
  getMemorySnapshot: vi.fn(),
  getMemoryStream: vi.fn(),
}));

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
  const snapshotMock = vi.mocked(getMemorySnapshot);
  const streamMock = vi.mocked(getMemoryStream);

  beforeEach(() => {
    vi.useFakeTimers();
    snapshotMock.mockResolvedValue(snapshot("scenario-a"));
    streamMock.mockResolvedValue(stream("scenario-a", 0));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("drops an older snapshot and stream response after the scope changes", async () => {
    let resolveSnapshotA!: (value: MemorySnapshotView) => void;
    let resolveStreamA!: (value: MemoryStreamView) => void;
    const snapshotA = new Promise<MemorySnapshotView>((resolve) => { resolveSnapshotA = resolve; });
    const streamA = new Promise<MemoryStreamView>((resolve) => { resolveStreamA = resolve; });
    snapshotMock.mockReturnValueOnce(snapshotA).mockResolvedValueOnce(snapshot("scenario-b"));
    streamMock.mockReturnValueOnce(streamA).mockResolvedValueOnce(stream("scenario-b", 3));

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
    streamMock.mockReturnValueOnce(new Promise<MemoryStreamView>((resolve) => { resolveStream = resolve; }));

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
    expect(streamMock).toHaveBeenCalledTimes(1);

    resolveStream(stream("scenario-a", 1));
  });

  it("does not query memory until an authoritative scenario is available", async () => {
    const { result } = renderHook(() => useMemory({
      userId: "operator",
      conversationId: "conversation-1",
      enabled: true,
    }));

    await act(async () => { await Promise.resolve(); });

    expect(snapshotMock).not.toHaveBeenCalled();
    expect(streamMock).not.toHaveBeenCalled();
    expect(result.current.scopeUnavailable).toBe(true);
  });
});
