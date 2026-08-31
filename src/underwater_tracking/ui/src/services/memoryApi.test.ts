import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  deleteMemory,
  getMemorySnapshot,
  getMemoryStream,
  getMemoryVersions,
} from "./memoryApi";

const snapshotPayload = {
  user_id: "operator",
  conversation_id: "conversation-1",
  scenario_id: "scenario-1",
  short_term: {
    user_id: "operator",
    scenario_id: "scenario-1",
    conversation_id: "conversation-1",
    summary_text: "已确认当前跟踪窗口。",
    summary_version: 2,
    recent_messages: [],
    message_count: 3,
    estimated_tokens: 42,
    compression_count: 1,
    compression_status: "completed",
    updated_at: "2026-08-21T08:00:00Z",
  },
  episodic: [],
  semantic: [
    {
      memory_id: "memory-1",
      memory_family_id: "family-1",
      version: 3,
      user_id: "operator",
      scenario_id: "scenario-1",
      memory_type: "semantic",
      summary: "区域接力需要保持重叠观测。",
      importance_score: 0.8,
      status: "active",
      source_event_ids: ["event-1"],
      access_count: 4,
      created_at: "2026-08-21T07:00:00Z",
      last_accessed_at: "2026-08-21T07:30:00Z",
    },
  ],
  procedural: [],
  retrieved_hits: [],
  versions: [],
  memory_status: "completed",
  degraded_reason: null,
};

describe("memoryApi", () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("parses a real snapshot and sends the scoped query to the backend", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify(snapshotPayload), { status: 200 }),
    );

    const snapshot = await getMemorySnapshot({
      userId: "operator",
      conversationId: "conversation-1",
      scenarioId: "scenario-1",
      query: "接力",
      limit: 20,
    });

    expect(snapshot.semantic[0]?.memory_id).toBe("memory-1");
    expect(snapshot.short_term?.summary_version).toBe(2);
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining(
        "/api/assistant/memory?user_id=operator&conversation_id=conversation-1&scenario_id=scenario-1&query=%E6%8E%A5%E5%8A%9B&limit=20",
      ),
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it("binds memory snapshot reads to the requested execution context", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({
        ...snapshotPayload,
        execution_revision: 9,
        frame_id: 42,
      }), { status: 200 }),
    );

    const snapshot = await getMemorySnapshot({
      userId: "operator",
      conversationId: "conversation-1",
      scenarioId: "scenario-1",
      executionRevision: 9,
      frameId: 42,
    });

    expect(snapshot.execution_revision).toBe(9);
    expect(snapshot.frame_id).toBe(42);
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("execution_revision=9&frame_id=42"),
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it("keeps degraded stream status and reason instead of inventing events", async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          user_id: "operator",
          conversation_id: "conversation-1",
          scenario_id: "scenario-1",
          events: [],
          after_cursor: 4,
          next_cursor: 4,
          memory_status: "degraded",
          degraded_reason: "Embedding credentials are unavailable",
        }),
        { status: 200 },
      ),
    );

    const stream = await getMemoryStream({
      userId: "operator",
      conversationId: "conversation-1",
      afterCursor: 4,
      scenarioId: "scenario-1",
    });

    expect(stream.events).toEqual([]);
    expect(stream.memory_status).toBe("degraded");
    expect(stream.degraded_reason).toContain("credentials");
  });

  it("uses the versions and DELETE endpoints and surfaces HTTP failures", async () => {
    fetchMock
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ user_id: "operator", memory_family_id: "family-1", versions: [] }),
          { status: 200 },
        ),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ status: "deleted", memory_id: "memory-1", user_id: "operator" }), {
          status: 200,
        }),
      )
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "unavailable" }), { status: 503 }));

    await expect(
      getMemoryVersions({ userId: "operator", memoryFamilyId: "family-1", scenarioId: "scenario-1" }),
    ).resolves.toEqual({ user_id: "operator", memory_family_id: "family-1", versions: [] });
    await expect(
      deleteMemory({ userId: "operator", memoryId: "memory-1", scenarioId: "scenario-1", conversationId: "conversation-1" }),
    ).resolves.toMatchObject({ status: "deleted", memory_id: "memory-1" });
    await expect(
      getMemorySnapshot({ userId: "operator", conversationId: "conversation-1", scenarioId: "scenario-1" }),
    ).rejects.toThrow("HTTP 503");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/assistant/memory/memory-1?user_id=operator&scenario_id=scenario-1&conversation_id=conversation-1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("rejects snapshot, stream, and version responses outside the requested scope", async () => {
    fetchMock
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ ...snapshotPayload, scenario_id: "scenario-2" }), { status: 200 }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({
          user_id: "operator",
          conversation_id: "conversation-1",
          scenario_id: "scenario-2",
          events: [],
          after_cursor: 0,
          next_cursor: 0,
          memory_status: "completed",
        }), { status: 200 }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({
          user_id: "operator",
          memory_family_id: "family-1",
          versions: [{ ...snapshotPayload.semantic[0], scenario_id: "scenario-2" }],
        }), { status: 200 }),
      );

    await expect(getMemorySnapshot({
      userId: "operator", conversationId: "conversation-1", scenarioId: "scenario-1",
    })).rejects.toThrow("scenario scope mismatch");
    await expect(getMemoryStream({
      userId: "operator", conversationId: "conversation-1", scenarioId: "scenario-1",
    })).rejects.toThrow("scenario scope mismatch");
    await expect(getMemoryVersions({
      userId: "operator", memoryFamilyId: "family-1", scenarioId: "scenario-1",
    })).rejects.toThrow("scenario scope mismatch");
  });

  it("rejects a stream response that advances beyond returned events", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({
        user_id: "operator",
        conversation_id: "conversation-1",
        scenario_id: "scenario-1",
        events: [],
        after_cursor: 0,
        next_cursor: 100,
        memory_status: "completed",
      }), { status: 200 }),
    );

    await expect(getMemoryStream({
      userId: "operator",
      conversationId: "conversation-1",
      scenarioId: "scenario-1",
    })).rejects.toThrow("cursor");
  });
});
