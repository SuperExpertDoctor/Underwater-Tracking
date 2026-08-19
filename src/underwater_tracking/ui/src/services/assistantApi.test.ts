import { afterEach, describe, expect, it, vi } from "vitest";
import { askQuestion, sendConversationMessage, waitForDirectiveStatus } from "./assistantApi";

describe("waitForDirectiveStatus", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("fails with a visible timeout when a directive never reaches a terminal state", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      json: async () => ({ request_id: "job-1", status: "processing" }),
    })));

    const pending = waitForDirectiveStatus("job-1", { intervalMs: 100, timeoutMs: 250 });
    const failure = expect(pending).rejects.toThrow("指令处理超时");
    await vi.advanceTimersByTimeAsync(300);
    await failure;
  });

  it("fails a question request when the backend never responds", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", vi.fn((_url: string, init?: RequestInit) => new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    })));

    const pending = askQuestion("为什么保持当前编组？");
    const failure = expect(pending).rejects.toThrow("请求超时");
    await vi.advanceTimersByTimeAsync(15_100);
    await failure;
  });

  it("posts a unified conversation message with the current plan scope", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ conversation_id: "conversation-1", classification: "evidence_query", messages: [] }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await sendConversationMessage({
      conversation_id: "conversation-1",
      text: "为什么保持 T1？",
      expected_plan_version: 4,
      target_ids: ["T1"],
      region_ids: ["T1:cell:0:0"],
    });

    expect(fetchMock).toHaveBeenCalledWith("/api/conversation/messages", expect.objectContaining({ method: "POST" }));
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      conversation_id: "conversation-1",
      text: "为什么保持 T1？",
      expected_plan_version: 4,
      target_ids: ["T1"],
      region_ids: ["T1:cell:0:0"],
    });
  });
});
